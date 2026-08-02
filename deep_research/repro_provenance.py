"""Offline provenance checks for public reference artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research.repro_common import ReproductionReport
from deep_research.settings import PublicSettings

REFERENCE_MANIFEST_PATH = Path("repro/reference/REFERENCE_MANIFEST.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_provenance_check(settings: PublicSettings) -> ReproductionReport:
    manifest_path = settings.paths.project_root / REFERENCE_MANIFEST_PATH
    if not manifest_path.exists():
        return ReproductionReport(
            mode="provenance",
            status="error",
            message=f"missing reference manifest: {manifest_path}",
            created_utc=datetime.now(UTC).isoformat(),
            reference_path=str(manifest_path),
        )

    manifest = json.loads(manifest_path.read_text())
    entries: list[dict[str, Any]] = []
    ok = True
    for item in manifest.get("files", []):
        rel = str(item.get("path") or "")
        expected_sha = str(item.get("sha256") or "")
        path = settings.paths.project_root / rel
        exists = path.exists()
        actual_sha = _sha256_file(path) if exists else ""
        matches = exists and actual_sha == expected_sha
        ok = ok and matches
        entries.append(
            {
                "path": rel,
                "exists": exists,
                "sha256": actual_sha,
                "expected_sha256": expected_sha,
                "matches": matches,
            }
        )

    query_file = settings.paths.project_root / "data/eval_queries_v2.json"
    metrics_file = settings.paths.project_root / "repro/reference/paper_a_pattern_metrics.csv"
    actual_query_count = len(json.loads(query_file.read_text()).get("queries", []))
    actual_pattern_count = _count_csv_rows(metrics_file)
    count_checks = {
        "query_count": actual_query_count,
        "expected_query_count": manifest.get("query_count"),
        "pattern_count": actual_pattern_count,
        "expected_pattern_count": manifest.get("pattern_count"),
    }
    counts_match = (
        actual_query_count == manifest.get("query_count")
        and actual_pattern_count == manifest.get("pattern_count")
    )
    ok = ok and counts_match

    return ReproductionReport(
        mode="provenance",
        status="success" if ok else "error",
        message="public reference provenance verified" if ok else "public reference provenance mismatch",
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(manifest_path),
        details={
            "manifest_schema_version": manifest.get("schema_version"),
            "files": entries,
            "counts": count_checks,
            "counts_match": counts_match,
        },
    )
