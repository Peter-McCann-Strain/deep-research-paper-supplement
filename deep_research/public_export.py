"""Build a sanitized public release tree from PUBLIC_MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research.release_audit import (
    DEFAULT_MANIFEST_NAME,
    AuditFinding,
    audit_release_tree,
    is_manifest_allowed,
    iter_release_files,
    load_public_manifest,
    manifest_patterns,
)

EXPORT_REPORT_NAME = "PUBLIC_EXPORT_REPORT.json"


@dataclass(frozen=True)
class PublicExportResult:
    source_root: str
    output_root: str
    files_copied: int
    bytes_copied: int
    audit_ok: bool
    audit_findings: list[AuditFinding]
    report_path: str

    @property
    def ok(self) -> bool:
        return self.audit_ok

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_root": self.source_root,
                "output_root": self.output_root,
                "files_copied": self.files_copied,
                "bytes_copied": self.bytes_copied,
                "audit_ok": self.audit_ok,
                "audit_findings": [asdict(finding) for finding in self.audit_findings],
                "report_path": self.report_path,
            },
            indent=2,
        )


def _resolve_manifest_path(source_root: Path, manifest_path: Path | None) -> Path:
    if manifest_path is None:
        return source_root / DEFAULT_MANIFEST_NAME
    if manifest_path.is_absolute():
        return manifest_path
    return source_root / manifest_path


def _prepare_output(output_root: Path, *, force: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise ValueError(f"output directory is not empty: {output_root}")
        if not (output_root / EXPORT_REPORT_NAME).exists():
            raise ValueError(
                "refusing to overwrite a non-empty directory that was not produced by "
                f"`deep-research export-public` ({EXPORT_REPORT_NAME} missing)"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(source_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"commit": "unavailable", "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            metadata["commit"] = commit.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0:
            metadata["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        metadata["error"] = exc.__class__.__name__
    return metadata


def _manifest_files(source_root: Path, manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    includes, _, required = manifest_patterns(manifest)
    if not includes:
        raise ValueError("PUBLIC_MANIFEST.json must define at least one include pattern")

    missing = [rel for rel in required if not (source_root / rel).exists()]
    if missing:
        raise FileNotFoundError("required public files are missing: " + ", ".join(missing))

    files: list[tuple[str, Path]] = []
    for path in iter_release_files(source_root):
        rel = path.relative_to(source_root).as_posix()
        if rel == EXPORT_REPORT_NAME:
            continue
        if is_manifest_allowed(rel, manifest):
            files.append((rel, path))
    files.sort(key=lambda item: item[0])
    return files


def _write_public_report(
    output_root: Path,
    *,
    files: list[str],
    bytes_copied: int,
    audit_findings: list[AuditFinding] | None,
    source_root: Path,
    manifest_path: Path,
    git_metadata: dict[str, Any],
) -> Path:
    report_path = output_root / EXPORT_REPORT_NAME
    report = {
        "generated_by": "deep-research export-public",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest": DEFAULT_MANIFEST_NAME,
        "manifest_sha256": _sha256_file(manifest_path),
        "source_git": git_metadata,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "files_copied": files,
        "files_in_artifact": [*files, EXPORT_REPORT_NAME],
        "file_sha256": {rel: _sha256_file(output_root / rel) for rel in files},
        "file_count": len(files),
        "artifact_file_count": len(files) + 1,
        "bytes_copied": bytes_copied,
        "audit_ok": audit_findings == [],
        "audit_findings": [asdict(finding) for finding in (audit_findings or [])],
        "note": "All paths are repository-relative. Absolute local paths are intentionally omitted.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def export_public_tree(
    source_root: Path,
    output_root: Path,
    *,
    manifest_path: Path | None = None,
    force: bool = False,
    max_file_mb: int | None = None,
    allow_dirty: bool = False,
) -> PublicExportResult:
    """Copy the manifest allowlist into ``output_root`` and audit the result."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    manifest_path = _resolve_manifest_path(source_root, manifest_path)
    manifest = load_public_manifest(manifest_path)

    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output directory must be outside the source repository")

    git_metadata = _git_metadata(source_root)
    if git_metadata.get("dirty") is True and not allow_dirty:
        raise ValueError(
            "source git tree has uncommitted changes; commit them before exporting or pass "
            "--allow-dirty for an explicitly non-release export"
        )

    selected = _manifest_files(source_root, manifest)
    _prepare_output(output_root, force=force)

    copied_rels: list[str] = []
    bytes_copied = 0
    for rel, source_path in selected:
        destination = output_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_rels.append(rel)
        bytes_copied += destination.stat().st_size

    report_path = _write_public_report(
        output_root,
        files=copied_rels,
        bytes_copied=bytes_copied,
        audit_findings=None,
        source_root=source_root,
        manifest_path=manifest_path,
        git_metadata=git_metadata,
    )
    first_audit = audit_release_tree(
        output_root,
        max_file_mb=max_file_mb,
        manifest_path=output_root / DEFAULT_MANIFEST_NAME,
        enforce_manifest=True,
    )
    report_path = _write_public_report(
        output_root,
        files=copied_rels,
        bytes_copied=bytes_copied,
        audit_findings=first_audit.findings,
        source_root=source_root,
        manifest_path=manifest_path,
        git_metadata=git_metadata,
    )
    final_audit = audit_release_tree(
        output_root,
        max_file_mb=max_file_mb,
        manifest_path=output_root / DEFAULT_MANIFEST_NAME,
        enforce_manifest=True,
    )

    return PublicExportResult(
        source_root=str(source_root),
        output_root=str(output_root),
        files_copied=len(copied_rels),
        bytes_copied=bytes_copied,
        audit_ok=final_audit.ok,
        audit_findings=final_audit.findings,
        report_path=str(report_path),
    )
