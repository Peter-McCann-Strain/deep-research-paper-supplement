"""Public query and rubric helpers for reproduction runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from deep_research.repro_common import PUBLIC_QUERIES_PATH, REFERENCE_RESULTS_PATH

SAFE_FILE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def load_reference_results(project_root: Path) -> dict[str, Any]:
    path = project_root / REFERENCE_RESULTS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_public_queries(project_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((project_root / PUBLIC_QUERIES_PATH).read_text())
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        raise TypeError("data/eval_queries_v2.json must contain a `queries` list")
    return [query for query in queries if isinstance(query, dict) and query.get("query")]


def _select_queries(
    queries: list[dict[str, Any]],
    *,
    full: bool,
    limit: int,
) -> list[dict[str, Any]]:
    if full:
        return queries
    if limit < 1:
        raise ValueError("limit must be a positive integer unless full=True")
    return queries[:limit]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _safe_file_stem(value: str, *, max_length: int = 96) -> str:
    stem = SAFE_FILE_STEM_RE.sub("_", value).strip("._-")
    return stem[:max_length].strip("._-")


def _query_file_stem(query_record: dict[str, Any]) -> str:
    explicit_id = str(query_record.get("id") or "").strip()
    if explicit_id:
        safe_id = _safe_file_stem(explicit_id)
        if safe_id:
            return safe_id
    digest = hashlib.sha256(query_record["query"].encode("utf-8")).hexdigest()[:16]
    return f"query_{digest}"


def _criteria_from_query_record(query_record: dict[str, Any]) -> list[str]:
    rubric = query_record.get("rubric")
    if not isinstance(rubric, dict):
        return []
    criteria: list[str] = []
    for item in rubric.get("criteria", []):
        text = ""
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            value = item.get("text") or item.get("criterion") or item.get("description")
            if isinstance(value, str):
                text = value.strip()
        if text:
            criteria.append(text)
    return criteria
