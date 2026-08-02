"""Shared public-reproduction models and constants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REFERENCE_RESULTS_PATH = Path("repro/reference/paper_a_reference.json")
REFERENCE_HEADLINE_PATH = Path("repro/reference/paper_a_headline_numbers.json")
REFERENCE_PATTERN_METRICS_CSV_PATH = Path("repro/reference/paper_a_pattern_metrics.csv")
PUBLIC_QUERIES_PATH = Path("data/eval_queries_v2.json")
PUBLIC_CRITERIA_PATH = Path("data/public_judge_criteria.json")
COMPARABLE_PATTERN_METRICS = (
    "mean_3judge",
    "mean_gpt52",
    "mean_opus",
    "mean_sonnet_corrected",
    "ppi_debiased_mean",
)
PRIMARY_PATTERN_METRIC = "mean_3judge"
MAX_PATTERN_SCORE_DELTA = 0.15
MEAN_PATTERN_SCORE_DELTA = 0.10


@dataclass(frozen=True)
class ReproductionReport:
    mode: str
    status: str
    message: str
    created_utc: str
    reference_path: str
    output_path: str | None = None
    details: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
