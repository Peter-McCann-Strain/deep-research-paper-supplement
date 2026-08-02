"""Pattern-level comparison helpers for public reproduction."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research.repro_common import (
    COMPARABLE_PATTERN_METRICS,
    MAX_PATTERN_SCORE_DELTA,
    MEAN_PATTERN_SCORE_DELTA,
    PRIMARY_PATTERN_METRIC,
    REFERENCE_HEADLINE_PATH,
    ReproductionReport,
)
from deep_research.settings import PublicSettings


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _numeric_metric(item: dict[str, Any]) -> float | None:
    for key in (PRIMARY_PATTERN_METRIC, "score", "mean_score", "overall_score"):
        value = _coerce_float(item.get(key))
        if value is not None:
            return value
    return None


def _pattern_metric_row(pattern: str, item: dict[str, Any]) -> dict[str, Any] | None:
    score = _numeric_metric(item)
    if score is None:
        return None
    row: dict[str, Any] = {"pattern": str(pattern), PRIMARY_PATTERN_METRIC: score}
    for key in COMPARABLE_PATTERN_METRICS:
        value = _coerce_float(item.get(key))
        if value is not None:
            row[key] = value
    n_queries = _coerce_float(item.get("n_queries"))
    if n_queries is not None:
        row["n_queries"] = int(n_queries)
    ppi_ci95 = item.get("ppi_ci95")
    if isinstance(ppi_ci95, list) and len(ppi_ci95) == 2:
        low = _coerce_float(ppi_ci95[0])
        high = _coerce_float(ppi_ci95[1])
        if low is not None and high is not None:
            row["ppi_ci95_low"] = low
            row["ppi_ci95_high"] = high
    for key in ("ppi_ci95_low", "ppi_ci95_high"):
        value = _coerce_float(item.get(key))
        if value is not None:
            row[key] = value
    return row


def _extract_pattern_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [payload]
    for key in ("details", "reference_metrics", "metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        for key in ("primary_ordering", "pattern_metrics", "metrics_by_pattern"):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, list):
                metrics = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    pattern = item.get("pattern") or item.get("pattern_name") or item.get("name")
                    if not pattern:
                        continue
                    row = _pattern_metric_row(str(pattern), item)
                    if row:
                        metrics.append(row)
                if metrics:
                    return metrics
            if isinstance(value, dict):
                metrics = []
                for pattern, item in value.items():
                    if isinstance(item, (int, float)):
                        row = {"pattern": str(pattern), PRIMARY_PATTERN_METRIC: float(item)}
                    elif isinstance(item, dict):
                        row = _pattern_metric_row(str(pattern), item)
                    else:
                        row = None
                    if row:
                        metrics.append(row)
                if metrics:
                    return sorted(
                        metrics, key=lambda row: row[PRIMARY_PATTERN_METRIC], reverse=True
                    )
    return []


def _load_candidate_pattern_payload(run_summary_path: Path) -> dict[str, Any]:
    if run_summary_path.suffix.lower() == ".csv":
        rows: list[dict[str, Any]] = []
        with run_summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pattern = row.get("pattern") or row.get("pattern_name") or row.get("name")
                if not pattern:
                    continue
                metric_value = row.get("mean_3judge") or row.get("score") or row.get("mean_score")
                score = _coerce_float(metric_value)
                if score is not None:
                    metric_row: dict[str, Any] = {"pattern": pattern, "mean_3judge": score}
                    for key in COMPARABLE_PATTERN_METRICS:
                        value = _coerce_float(row.get(key))
                        if value is not None:
                            metric_row[key] = value
                    for key in ("n_queries", "ppi_ci95_low", "ppi_ci95_high"):
                        value = _coerce_float(row.get(key))
                        if value is not None:
                            metric_row[key] = int(value) if key == "n_queries" else value
                    rows.append(metric_row)
        return {"primary_ordering": rows}
    return json.loads(run_summary_path.read_text())


def compare_paper_a_run(settings: PublicSettings, run_summary_path: Path) -> ReproductionReport:
    """Compare a candidate pattern-metric run with the frozen public reference."""
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    headline = json.loads(headline_path.read_text())
    run_summary_path = run_summary_path.resolve()
    candidate = _load_candidate_pattern_payload(run_summary_path)

    reference_metrics = _extract_pattern_metrics(headline)
    candidate_metrics = _extract_pattern_metrics(candidate)
    if not candidate_metrics:
        details = candidate.get("details", {}) if isinstance(candidate.get("details"), dict) else {}
        return ReproductionReport(
            mode="compare",
            status="not-comparable",
            message=(
                "run summary has no pattern-level mean_3judge metrics; "
                "api-best-effort summaries are live demos, not the frozen 13-pattern paper matrix"
            ),
            created_utc=datetime.now(UTC).isoformat(),
            reference_path=str(headline_path),
            output_path=str(run_summary_path),
            details={
                "run_mode": candidate.get("mode"),
                "run_status": candidate.get("status"),
                "query_count": details.get("query_count"),
                "successful_generations": details.get("successful_generations"),
                "judge_requested": details.get("judge_requested"),
                "required_candidate_schema": {
                    "primary_ordering": [{"pattern": "base_p1", "mean_3judge": 0.0}]
                },
                "comparison_contract": headline.get("comparison_policy"),
            },
        )

    reference_by_pattern = {row["pattern"]: row for row in reference_metrics}
    candidate_by_pattern = {row["pattern"]: row for row in candidate_metrics}
    overlaps = [pattern for pattern in reference_by_pattern if pattern in candidate_by_pattern]
    if not overlaps:
        return ReproductionReport(
            mode="compare",
            status="not-comparable",
            message="candidate pattern names do not overlap the public reference",
            created_utc=datetime.now(UTC).isoformat(),
            reference_path=str(headline_path),
            output_path=str(run_summary_path),
            details={
                "reference_patterns": sorted(reference_by_pattern),
                "candidate_patterns": sorted(candidate_by_pattern),
            },
        )

    deltas = [
        {
            "pattern": pattern,
            "reference_mean_3judge": round(reference_by_pattern[pattern][PRIMARY_PATTERN_METRIC], 4),
            "candidate_mean_3judge": round(candidate_by_pattern[pattern][PRIMARY_PATTERN_METRIC], 4),
            "delta": round(
                candidate_by_pattern[pattern][PRIMARY_PATTERN_METRIC]
                - reference_by_pattern[pattern][PRIMARY_PATTERN_METRIC],
                4,
            ),
        }
        for pattern in overlaps
    ]
    reference_metric_names = [
        metric
        for metric in COMPARABLE_PATTERN_METRICS
        if any(metric in reference_by_pattern[p] for p in overlaps)
    ]
    missing_metric_cells = [
        {"pattern": pattern, "metric": metric}
        for pattern in overlaps
        for metric in reference_metric_names
        if metric in reference_by_pattern[pattern] and metric not in candidate_by_pattern[pattern]
    ]
    metric_names = [
        metric
        for metric in reference_metric_names
        if any(metric in reference_by_pattern[p] and metric in candidate_by_pattern[p] for p in overlaps)
    ]
    metric_deltas = []
    for pattern in overlaps:
        for metric in metric_names:
            if metric not in reference_by_pattern[pattern] or metric not in candidate_by_pattern[pattern]:
                continue
            metric_deltas.append(
                {
                    "pattern": pattern,
                    "metric": metric,
                    "reference": round(reference_by_pattern[pattern][metric], 4),
                    "candidate": round(candidate_by_pattern[pattern][metric], 4),
                    "delta": round(
                        candidate_by_pattern[pattern][metric] - reference_by_pattern[pattern][metric],
                        4,
                    ),
                }
            )
    reference_order = [row["pattern"] for row in reference_metrics if row["pattern"] in overlaps]
    candidate_order = [
        row["pattern"]
        for row in sorted(
            candidate_metrics, key=lambda item: item[PRIMARY_PATTERN_METRIC], reverse=True
        )
        if row["pattern"] in overlaps
    ]
    same_top = bool(
        reference_order and candidate_order and reference_order[0] == candidate_order[0]
    )
    ordering_matches_reference = candidate_order == reference_order
    reference_ranks = {pattern: index for index, pattern in enumerate(reference_order)}
    candidate_ranks = {pattern: index for index, pattern in enumerate(candidate_order)}
    rank_displacements = [
        abs(candidate_ranks[pattern] - reference_ranks[pattern]) for pattern in overlaps
    ]
    max_rank_displacement = max(rank_displacements) if rank_displacements else 0
    primary_abs_deltas = [abs(row["delta"]) for row in deltas]
    metric_abs_deltas = [abs(row["delta"]) for row in metric_deltas] or primary_abs_deltas
    max_abs_delta = max(metric_abs_deltas) if metric_abs_deltas else 0.0
    mean_abs_delta = sum(metric_abs_deltas) / len(metric_abs_deltas) if metric_abs_deltas else 0.0
    score_within_tolerance = (
        max_abs_delta <= MAX_PATTERN_SCORE_DELTA
        and mean_abs_delta <= MEAN_PATTERN_SCORE_DELTA
    )
    full_overlap = len(overlaps) == len(reference_by_pattern)
    full_metric_schema = not missing_metric_cells

    if not ordering_matches_reference or not score_within_tolerance:
        status = "diverged"
        message = (
            "candidate pattern metrics diverge from the public reference beyond the "
            "declared broad-range tolerance"
        )
    elif full_overlap and full_metric_schema:
        status = "success"
        message = f"compared {len(overlaps)}/{len(reference_by_pattern)} reference pattern metrics"
    elif full_overlap:
        status = "partial"
        message = (
            f"compared {len(overlaps)}/{len(reference_by_pattern)} reference patterns, "
            "but candidate is missing reference metric cells"
        )
    else:
        status = "partial"
        message = f"compared {len(overlaps)}/{len(reference_by_pattern)} reference pattern metrics"

    return ReproductionReport(
        mode="compare",
        status=status,
        message=message,
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(headline_path),
        output_path=str(run_summary_path),
        details={
            "metric": headline.get("primary_metric", PRIMARY_PATTERN_METRIC),
            "metrics_compared": metric_names,
            "reference_metrics_required": reference_metric_names,
            "full_metric_schema": full_metric_schema,
            "missing_metric_cells": missing_metric_cells,
            "overlap_count": len(overlaps),
            "reference_pattern_count": len(reference_by_pattern),
            "candidate_pattern_count": len(candidate_by_pattern),
            "top_pattern_matches_reference": same_top,
            "ordering_matches_reference": ordering_matches_reference,
            "max_rank_displacement": max_rank_displacement,
            "score_within_tolerance": score_within_tolerance,
            "max_abs_delta": round(max_abs_delta, 4),
            "mean_abs_delta": round(mean_abs_delta, 4),
            "tolerances": {
                "max_abs_delta": MAX_PATTERN_SCORE_DELTA,
                "mean_abs_delta": MEAN_PATTERN_SCORE_DELTA,
            },
            "reference_ordering_overlap": reference_order,
            "candidate_ordering_overlap": candidate_order,
            "deltas": deltas,
            "metric_deltas": metric_deltas,
            "comparison_contract": headline.get("comparison_policy"),
        },
    )
