"""Offline judge parser and scoring fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research.api_judges import parse_json_response
from deep_research.repro_scoring import _score_judge_report

FIXTURES = Path("tests/fixtures/judge_responses")


def test_judge_validation_accepts_valid_fixture():
    payload = json.loads((FIXTURES / "valid_3criteria.json").read_text())

    assert parse_json_response(json.dumps(payload), expected_criteria_count=3) == payload


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("missing_criterion.json", "missing criterion indexes"),
        ("duplicate_criterion.json", "duplicate criterion_index"),
        ("unexpected_criterion.json", "unexpected criterion indexes"),
        ("bad_verdict.json", "invalid verdict"),
        ("empty_evidence.json", "must contain non-empty evidence"),
    ],
)
def test_judge_validation_rejects_adversarial_fixtures(name, message):
    payload = json.loads((FIXTURES / name).read_text())

    with pytest.raises((TypeError, ValueError), match=message):
        parse_json_response(json.dumps(payload), expected_criteria_count=3)


def test_judge_scoring_fixture_uses_dimension_weights():
    judge_payload = json.loads((FIXTURES / "valid_3provider_panel.json").read_text())
    query_record = {
        "id": "fixture",
        "query": "Fixture query",
        "rubric": {
            "dimension_weights": {"accuracy": 0.4, "coverage": 0.6},
            "criteria": [
                {"text": "cite sources", "dimension": "accuracy", "weight": 1},
                {"text": "state limitations", "dimension": "coverage", "weight": 1},
                {"text": "discuss tradeoffs", "dimension": "coverage", "weight": 1},
            ],
        },
    }

    score = _score_judge_report(judge_payload, query_record, [])

    assert score["successful_provider_scores"] == 3
    assert score["mean_3judge_current_api"] == 0.7
    assert score["provider_scores"][0]["scoring_method"] == "dimension_weighted"
    assert score["provider_scores"][0]["dimension_scores"] == {"accuracy": 1.0, "coverage": 0.5}
