"""Pattern dictionary coverage checks."""

from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: str):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_pattern_dictionary_covers_reference_csv_once():
    metrics = _rows("repro/reference/paper_a_pattern_metrics.csv")
    dictionary = _rows("repro/reference/PATTERN_DICTIONARY.csv")
    required_columns = {
        "pattern",
        "display_name",
        "paper_archetype",
        "variant_kind",
        "variant_notes",
        "description",
        "paper_mapping",
        "included_in_frozen_comparison",
        "rank",
        "n_queries",
        "mean_3judge",
    }

    metric_patterns = [row["pattern"] for row in metrics]
    dictionary_patterns = [row["pattern"] for row in dictionary]
    metric_by_pattern = {row["pattern"]: row for row in metrics}

    assert required_columns.issubset(dictionary[0].keys())
    assert sorted(dictionary_patterns) == sorted(metric_patterns)
    assert len(dictionary_patterns) == len(set(dictionary_patterns))
    assert all(row["included_in_frozen_comparison"] == "true" for row in dictionary)
    assert {row["rank"] for row in dictionary} == {row["rank"] for row in metrics}
    assert all(row["display_name"] and row["description"] for row in dictionary)
    assert all(row["paper_archetype"] and row["variant_kind"] for row in dictionary)
    assert all(len(row["variant_notes"]) >= 40 for row in dictionary)
    assert all(row["n_queries"] == metric_by_pattern[row["pattern"]]["n_queries"] for row in dictionary)
    assert all(row["mean_3judge"] == metric_by_pattern[row["pattern"]]["mean_3judge"] for row in dictionary)
