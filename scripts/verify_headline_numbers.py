#!/usr/bin/env python3
"""Verify public headline Paper A numbers from released artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEADLINE = ROOT / "repro" / "reference" / "paper_a_headline_numbers.json"
DEFAULT_METRICS_CSV = ROOT / "repro" / "reference" / "paper_a_pattern_metrics.csv"
DEFAULT_CANONICAL = (
    ROOT / "paper_rebuild" / "paper_a_bounded_returns" / "analysis" / "canonical_numbers.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_metrics(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{path} has no metric rows")
    return rows


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    as_float = float(value)
    if math.isnan(as_float):
        return None
    return as_float


def _same_float(left: Any, right: Any, tolerance: float) -> bool:
    left_float = _as_float(left)
    right_float = _as_float(right)
    if left_float is None or right_float is None:
        return left_float is right_float
    return abs(left_float - right_float) <= tolerance


def _compare_rows(
    *,
    headline_row: dict[str, Any],
    csv_row: dict[str, str],
    canonical_row: dict[str, Any],
    tolerance: float,
) -> list[str]:
    pattern = headline_row["pattern"]
    failures: list[str] = []
    if int(headline_row["n_queries"]) != int(csv_row["n_queries"]):
        failures.append(f"{pattern}: headline n_queries differs from metrics CSV")
    if int(headline_row["n_queries"]) != int(canonical_row["n_queries"]):
        failures.append(f"{pattern}: headline n_queries differs from canonical numbers")

    checks = {
        "mean_3judge": "mean_3judge",
        "mean_gpt52": "mean_gpt52",
        "mean_opus": "mean_opus",
        "mean_sonnet_corrected": "mean_sonnet_corrected",
        "ppi_debiased_mean": "ppi_debiased",
    }
    for headline_key, canonical_key in checks.items():
        headline_value = headline_row.get(headline_key)
        csv_value = csv_row.get(headline_key)
        if canonical_key == "ppi_debiased":
            canonical_value = canonical_row.get("ppi_debiased", {}).get("ppi_mean")
        else:
            canonical_value = canonical_row.get(canonical_key)
        if not _same_float(headline_value, csv_value, tolerance):
            failures.append(f"{pattern}: headline {headline_key} differs from metrics CSV")
        if not _same_float(headline_value, canonical_value, tolerance):
            failures.append(f"{pattern}: headline {headline_key} differs from canonical numbers")

    ci = headline_row.get("ppi_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        failures.append(f"{pattern}: headline ppi_ci95 is not a two-value interval")
    else:
        canonical_ci = canonical_row.get("ppi_debiased", {}).get("ci95", [])
        csv_low = csv_row.get("ppi_ci95_low")
        csv_high = csv_row.get("ppi_ci95_high")
        for idx, (label, csv_value) in enumerate((("low", csv_low), ("high", csv_high))):
            if not _same_float(ci[idx], csv_value, tolerance):
                failures.append(f"{pattern}: headline ppi_ci95_{label} differs from metrics CSV")
            if not _same_float(ci[idx], canonical_ci[idx] if len(canonical_ci) > idx else None, tolerance):
                failures.append(f"{pattern}: headline ppi_ci95_{label} differs from canonical numbers")

    return failures


def verify_public_headlines(
    *,
    headline_path: Path,
    metrics_csv_path: Path,
    canonical_path: Path,
    tolerance: float,
) -> list[str]:
    headline = _load_json(headline_path)
    metrics_rows = _load_metrics(metrics_csv_path)
    canonical = _load_json(canonical_path)

    failures: list[str] = []
    ordering = headline.get("primary_ordering")
    if not isinstance(ordering, list) or not ordering:
        return ["headline primary_ordering must be a non-empty list"]
    if headline.get("paper") != "paper-a-bounded-returns":
        failures.append("headline paper id is not paper-a-bounded-returns")
    if headline.get("primary_metric") != "mean_3judge":
        failures.append("headline primary_metric is not mean_3judge")
    if int(headline.get("pattern_count", -1)) != len(ordering):
        failures.append("headline pattern_count differs from primary_ordering length")
    if int(headline.get("query_count", -1)) != 90:
        failures.append("headline query_count is not 90")
    if len(metrics_rows) != len(ordering):
        failures.append("metrics CSV row count differs from headline ordering length")

    csv_by_pattern = {row["pattern"]: row for row in metrics_rows}
    canonical_headline = canonical.get("headline", {})
    canonical_by_pattern = canonical_headline.get("per_pattern", {})
    if not isinstance(canonical_by_pattern, dict):
        failures.append("canonical headline.per_pattern is missing")
        canonical_by_pattern = {}

    ordering_patterns = [row["pattern"] for row in ordering]
    csv_patterns_by_rank = [row["pattern"] for row in sorted(metrics_rows, key=lambda row: int(row["rank"]))]
    if ordering_patterns != csv_patterns_by_rank:
        failures.append("headline ordering differs from metrics CSV rank order")

    sorted_by_score = [
        row["pattern"] for row in sorted(ordering, key=lambda row: float(row["mean_3judge"]), reverse=True)
    ]
    if ordering_patterns != sorted_by_score:
        failures.append("headline ordering is not sorted by descending mean_3judge")

    for headline_row in ordering:
        pattern = headline_row["pattern"]
        csv_row = csv_by_pattern.get(pattern)
        canonical_row = canonical_by_pattern.get(pattern)
        if csv_row is None:
            failures.append(f"{pattern}: missing from metrics CSV")
            continue
        if canonical_row is None:
            failures.append(f"{pattern}: missing from canonical headline.per_pattern")
            continue
        failures.extend(
            _compare_rows(
                headline_row=headline_row,
                csv_row=csv_row,
                canonical_row=canonical_row,
                tolerance=tolerance,
            )
        )

    ranges = headline.get("headline_ranges", {})
    first = ordering[0]
    last = ordering[-1]
    if ranges.get("best_pattern") != first["pattern"]:
        failures.append("headline best_pattern differs from first ranked pattern")
    if ranges.get("worst_pattern") != last["pattern"]:
        failures.append("headline worst_pattern differs from last ranked pattern")
    if not _same_float(ranges.get("best_mean_3judge"), first["mean_3judge"], tolerance):
        failures.append("headline best_mean_3judge differs from first ranked mean")
    if not _same_float(ranges.get("worst_mean_3judge"), last["mean_3judge"], tolerance):
        failures.append("headline worst_mean_3judge differs from last ranked mean")

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline", type=Path, default=DEFAULT_HEADLINE)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = verify_public_headlines(
        headline_path=args.headline,
        metrics_csv_path=args.metrics_csv,
        canonical_path=args.canonical,
        tolerance=args.tolerance,
    )
    payload = {
        "ok": not failures,
        "headline": str(args.headline),
        "metrics_csv": str(args.metrics_csv),
        "canonical": str(args.canonical),
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    elif failures:
        print("Headline number verification FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
    else:
        print("Headline number verification OK")
        print(f"Checked {args.headline}")
        print(f"Checked {args.metrics_csv}")
        print(f"Checked {args.canonical}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
