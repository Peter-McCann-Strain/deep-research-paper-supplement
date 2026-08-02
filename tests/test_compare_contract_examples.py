"""Offline examples for the pattern-metric comparison contract."""

from __future__ import annotations

import csv
from pathlib import Path

from deep_research.reproduce import compare_paper_a_run
from deep_research.settings import load_public_settings


def test_compare_contract_fixtures_cover_success_and_divergence(tmp_path):
    settings = load_public_settings(project_root=Path.cwd(), env={})

    good = compare_paper_a_run(settings, Path("repro/reference/examples/pattern_metrics_good.csv"))
    bad_order = compare_paper_a_run(
        settings, Path("repro/reference/examples/pattern_metrics_bad_order.csv")
    )
    bad_delta = compare_paper_a_run(
        settings, Path("repro/reference/examples/pattern_metrics_bad_delta.csv")
    )

    assert good.status == "success"
    assert good.details["ordering_matches_reference"] is True
    assert bad_order.status == "diverged"
    assert bad_order.details["ordering_matches_reference"] is False
    assert bad_delta.status == "diverged"
    assert bad_delta.details["score_within_tolerance"] is False
    assert "mean_opus" in bad_delta.details["metrics_compared"]


def test_compare_contract_partial_overlap_is_partial(tmp_path):
    settings = load_public_settings(project_root=Path.cwd(), env={})
    partial = tmp_path / "partial.csv"
    with Path("repro/reference/paper_a_pattern_metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))[:3]
        fieldnames = rows[0].keys()
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = compare_paper_a_run(settings, partial)

    assert report.status == "partial"
    assert report.details["overlap_count"] == 3
