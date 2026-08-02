"""Paper A reference and pattern-metric comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research.reproduce import (
    compare_paper_a_run,
    run_reference_summary,
    run_smoke_reproduction,
)
from deep_research.settings import load_public_settings


def _write_minimal_reference(root: Path) -> None:
    reference_dir = root / "repro" / "reference"
    reference_dir.mkdir(parents=True)
    (reference_dir / "paper_a_reference.json").write_text(
        json.dumps(
            {
                "paper": "paper-a",
                "reproduction_contract": "best effort",
                "reference_metrics": {},
            }
        )
    )
    (reference_dir / "paper_a_headline_numbers.json").write_text(
        json.dumps(
            {
                "query_count": 90,
                "pattern_count": 2,
                "primary_metric": "mean_3judge",
                "headline_ranges": {"best_pattern": "base_p1"},
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.67, "mean_opus": 0.8},
                    {"pattern": "base_p0", "mean_3judge": 0.49, "mean_opus": 0.5},
                ],
                "comparison_policy": "compare direction and broad score ranges",
            }
        )
    )


def test_smoke_reproduction_reads_reference(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_smoke_reproduction(settings)

    assert report.status == "success"
    assert report.mode == "smoke"


def test_reference_summary_reports_headline_ordering(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_reference_summary(settings)

    assert report.status == "success"
    assert report.mode == "reference"
    assert report.details["top_patterns"][0]["pattern"] == "base_p1"
    assert report.details["comparison_policy"]


def test_compare_rejects_api_demo_summary_without_pattern_metrics(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "mode": "api-best-effort",
                "status": "success",
                "details": {"query_count": 1, "successful_generations": 1},
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "not-comparable"
    assert "13-pattern" in report.message
    assert report.details["required_candidate_schema"]["primary_ordering"]


def test_compare_marks_missing_secondary_metrics_partial(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "pattern_summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.7},
                    {"pattern": "base_p0", "mean_3judge": 0.4},
                ]
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "partial"
    assert report.details["top_pattern_matches_reference"] is True
    assert report.details["ordering_matches_reference"] is True
    assert report.details["overlap_count"] == 2
    assert report.details["full_metric_schema"] is False
    assert report.details["missing_metric_cells"]


def test_compare_rejects_judge_metric_divergence(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "pattern_summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.67, "mean_opus": 0.2},
                    {"pattern": "base_p0", "mean_3judge": 0.49, "mean_opus": 0.9},
                ]
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "diverged"
    assert "mean_opus" in report.details["metrics_compared"]
    assert report.details["score_within_tolerance"] is False


def test_compare_rejects_large_score_divergence(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "pattern_summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.1},
                    {"pattern": "base_p0", "mean_3judge": 0.9},
                ]
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "diverged"
    assert report.details["score_within_tolerance"] is False
    assert report.details["top_pattern_matches_reference"] is False
    assert report.details["ordering_matches_reference"] is False
    assert report.details["max_rank_displacement"] > 0


def test_compare_accepts_public_pattern_metrics_csv():
    settings = load_public_settings(project_root=Path.cwd(), env={})

    report = compare_paper_a_run(settings, Path("repro/reference/paper_a_pattern_metrics.csv"))

    assert report.status == "success"
    assert report.details["overlap_count"] == 13
    assert report.details["top_pattern_matches_reference"] is True
