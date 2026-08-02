#!/usr/bin/env python3
"""Prepare and optionally upload the public supplement to Hugging Face.

The script intentionally reads credentials only from environment variables or
an existing Hugging Face login. It does not accept tokens as command-line
arguments, because command histories and process listings are too easy to leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = Path("/tmp/deep-research-hf-export")
DEFAULT_UPLOAD_DIR = Path("/tmp/deep-research-hf-upload")
DEFAULT_CARD_PATH = REPO_ROOT / "repro" / "HUGGINGFACE_DATASET_CARD.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise ValueError(f"directory is not empty: {path}; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_export_to_upload(export_dir: Path, upload_dir: Path, *, force: bool) -> None:
    _prepare_dir(upload_dir, force=force)
    for source in export_dir.rglob("*"):
        if not source.is_file():
            continue
        rel = source.relative_to(export_dir)
        destination = upload_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_hf_readme(upload_dir: Path, card_path: Path) -> None:
    card = card_path.read_text(encoding="utf-8").rstrip()
    readme_path = upload_dir / "README.md"
    original_readme = readme_path.read_text(encoding="utf-8").strip()
    readme_path.write_text(
        card
        + "\n\n---\n\n"
        + "## Full Repository README\n\n"
        + original_readme
        + "\n",
        encoding="utf-8",
    )


def _refresh_export_report(upload_dir: Path, *, repo_id: str) -> None:
    report_path = upload_dir / "PUBLIC_EXPORT_REPORT.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing PUBLIC_EXPORT_REPORT.json in {upload_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = report.get("files_copied", [])
    if not isinstance(files, list):
        raise TypeError("PUBLIC_EXPORT_REPORT.json field files_copied must be a list")
    report["generated_by"] = "scripts/publish_huggingface.py"
    report["created_utc"] = datetime.now(UTC).isoformat()
    report["huggingface"] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "readme": "README.md contains the Hugging Face dataset card followed by the GitHub README.",
    }
    report["file_sha256"] = {rel: _sha256_file(upload_dir / rel) for rel in files}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _audit_upload(upload_dir: Path) -> dict[str, Any]:
    from deep_research.release_audit import audit_release_tree

    result = audit_release_tree(upload_dir)
    return json.loads(result.to_json())


def prepare_upload_folder(
    *,
    repo_id: str,
    export_dir: Path,
    upload_dir: Path,
    card_path: Path,
    force: bool,
    reuse_export: bool,
) -> dict[str, Any]:
    from deep_research.public_export import export_public_tree

    export_dir = export_dir.resolve()
    upload_dir = upload_dir.resolve()
    card_path = card_path.resolve()

    if reuse_export:
        if not (export_dir / "PUBLIC_EXPORT_REPORT.json").exists():
            raise FileNotFoundError(
                f"--reuse-export requires an existing export with PUBLIC_EXPORT_REPORT.json: {export_dir}"
            )
    else:
        export_public_tree(REPO_ROOT, export_dir, force=force)

    _copy_export_to_upload(export_dir, upload_dir, force=force)
    _write_hf_readme(upload_dir, card_path)
    _refresh_export_report(upload_dir, repo_id=repo_id)
    audit = _audit_upload(upload_dir)
    if not audit.get("ok"):
        raise RuntimeError("Hugging Face upload folder failed release audit: " + json.dumps(audit))
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "export_dir": str(export_dir),
        "upload_dir": str(upload_dir),
        "audit_ok": True,
        "file_count": sum(1 for path in upload_dir.rglob("*") if path.is_file()),
        "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
    }


def _load_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face publishing requires the optional extra: "
            "python -m pip install -c constraints-public.txt -e '.[publish]'"
        ) from exc
    return HfApi()


def publish(upload_dir: Path, *, repo_id: str, private: bool, commit_message: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = _load_hf_api()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(upload_dir),
        commit_message=commit_message,
        token=token,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. user/name")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--dataset-card", type=Path, default=DEFAULT_CARD_PATH)
    parser.add_argument("--reuse-export", action="store_true", help="Use an existing export-dir")
    parser.add_argument("--force", action="store_true", help="Replace export/upload dirs")
    parser.add_argument("--private", action="store_true", help="Create/update a private HF repo")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and audit but do not upload")
    parser.add_argument(
        "--commit-message",
        default="Publish public paper supplement",
        help="HF commit message for uploads",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_upload_folder(
        repo_id=args.repo_id,
        export_dir=args.export_dir,
        upload_dir=args.upload_dir,
        card_path=args.dataset_card,
        force=args.force,
        reuse_export=args.reuse_export,
    )
    if args.dry_run:
        result["status"] = "dry-run"
        result["uploaded"] = False
        print(json.dumps(result, indent=2))
        return 0
    url = publish(
        args.upload_dir.resolve(),
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )
    result["status"] = "success"
    result["uploaded"] = True
    result["dataset_url"] = url
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
