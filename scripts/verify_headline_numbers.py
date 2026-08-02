#!/usr/bin/env python3
"""Verify headline paper numbers against released artifacts.

This is intentionally narrow: it checks the numbers reviewers are likely to
audit first, and fails if the paper-facing contract drifts from canonical
artifacts. The expected values live in reports/paper_draft/headline_numbers.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPECTED = ROOT / "reports" / "paper_draft" / "headline_numbers.json"


def _cohen_kappa(y_true: list[bool], y_pred: list[bool]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    pe = 0.0
    for label in labels:
        p_true = sum(v == label for v in y_true) / n
        p_pred = sum(v == label for v in y_pred) / n
        pe += p_true * p_pred
    if math.isclose(1.0, pe):
        return 1.0 if math.isclose(1.0, po) else float("nan")
    return (po - pe) / (1.0 - pe)


def _accuracy(y_true: list[bool], y_pred: list[bool]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return float("nan")
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def _round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _parse_tost_counts() -> dict[str, Any]:
    path = ROOT / "reports" / "phase6a_corrections" / "03_tost_wilcoxon.md"
    text = path.read_text(encoding="utf-8")
    wilc = re.search(r"Wilcoxon TOST:\s*(\d+)/15 equivalent at ±0\.05,\s*(\d+)/15 at ±0\.02", text)
    paired_t = re.search(r"Paired-t TOST \(previous\):\s*(\d+)/15 at ±0\.05", text)
    return {
        "tost.wilcoxon_equiv_0p05": int(wilc.group(1)) if wilc else None,
        "tost.wilcoxon_equiv_0p02": int(wilc.group(2)) if wilc else None,
        "tost.previous_paired_t_equiv_0p05": int(paired_t.group(1)) if paired_t else None,
    }


def _bibliography_counts() -> dict[str, Any]:
    import bibtexparser

    bib_path = ROOT / "reports" / "paper_draft" / "bibliography.bib"
    paper_path = ROOT / "reports" / "paper_draft" / "paper_v9.md"
    bib = bibtexparser.loads(bib_path.read_text(encoding="utf-8"))
    keys = {entry["ID"] for entry in bib.entries}
    text = paper_path.read_text(encoding="utf-8")
    cited: set[str] = set()
    for inner in re.findall(r"\[([^\[\]]+)\]", text):
        if ":" in inner:
            continue
        for key in re.split(r"[,;\s]+", inner):
            if key in keys:
                cited.add(key)
    return {
        "bibliography.entries": len(keys),
        "bibliography.cited_entries": len(cited),
        "bibliography.uncited_entries": len(keys - cited),
    }


def compute_numbers() -> dict[str, Any]:
    numbers: dict[str, Any] = {}
    analysis = ROOT / "data" / "analysis"
    paper_v9 = (ROOT / "reports" / "paper_draft" / "paper_v9.md").read_text(encoding="utf-8")

    df_queries = pd.read_parquet(analysis / "df_queries.parquet")
    df_runs = pd.read_parquet(analysis / "df_runs.parquet")
    df_scores = pd.read_parquet(analysis / "df_scores.parquet")
    df_overall = pd.read_parquet(analysis / "df_overall_scores.parquet")
    df_verdicts = pd.read_parquet(analysis / "df_verdicts.parquet")

    numbers.update(
        {
            "df_queries.rows": int(len(df_queries)),
            "df_runs.rows": int(len(df_runs)),
            "df_scores.rows": int(len(df_scores)),
            "df_overall_scores.rows": int(len(df_overall)),
            "df_verdicts.rows": int(len(df_verdicts)),
            "df_overall_scores.base_p12.rows": int((df_overall["pattern"].astype(str) == "base_p12").sum()),
        }
    )

    base = df_overall[df_overall["pattern_family"].astype(str).eq("base")]
    score_col = "overall_score_recomputed"
    means = base.groupby("pattern", observed=True)[score_col].agg(["mean", "count"])
    for pattern in sorted(means.index.astype(str)):
        numbers[f"means.{pattern}.mean"] = float(means.loc[pattern, "mean"])
        numbers[f"means.{pattern}.n"] = int(means.loc[pattern, "count"])

    numbers.update(_parse_tost_counts())

    pa_path = ROOT / "reports" / "protocol_a" / "paired_bootstrap_summary.csv"
    pa = pd.read_csv(pa_path)
    numbers["protocol_a.patterns"] = int(len(pa))
    numbers["protocol_a.mean_delta_tavily_minus_bing"] = float(pa["delta_tav_minus_bing"].mean())

    c0 = pd.read_parquet(analysis / "df_c0_verdicts.parquet")
    c0_counts = c0["verdict"].value_counts(normalize=True).mul(100.0)
    for verdict in ["supports", "neutral", "contradicts", "no_source"]:
        numbers[f"c0.verdict_pct.{verdict}"] = float(c0_counts.get(verdict, 0.0))

    cits = pd.read_parquet(analysis / "df_citations.parquet")
    numbers["citations.rows"] = int(len(cits))
    numbers["citations.report_rows"] = int(cits[["pattern", "query_id"]].drop_duplicates().shape[0])
    pp = pd.read_csv(ROOT / "reports" / "phase7a_citation_verification" / "per_pattern_stats.csv")
    p4 = pp[pp["pattern"].eq("base_p4")].iloc[0]
    numbers["citations.base_p4.total_citations"] = int(p4["total_citations"])
    numbers["citations.base_p4.total_placeholder"] = int(p4["total_placeholder"])
    numbers["citations.base_p4.placeholder_pct"] = float(p4["mean_placeholder_rate"] * 100.0)

    dr = pd.read_parquet(ROOT / "reports" / "phase12_drjudge" / "eval_predictions_full.parquet")
    for label, sub in {
        "all": dr,
        "undisputed": dr[~dr["is_disputed"]],
        "disputed": dr[dr["is_disputed"]],
    }.items():
        target = sub["target"].astype(bool).tolist()
        pred = sub["predicted"].astype(bool).tolist()
        numbers[f"dr_judge.{label}.n"] = int(len(sub))
        numbers[f"dr_judge.{label}.kappa"] = float(_cohen_kappa(target, pred))
        numbers[f"dr_judge.{label}.accuracy"] = float(_accuracy(target, pred))

    p12_summary = ROOT / "reports" / "phase16_p12_eval" / "per_pattern_summary.md"
    if p12_summary.exists():
        text = p12_summary.read_text(encoding="utf-8")
        m = re.search(r"\|\s*\*\*mean overall_score\*\*\s*\|\s*\*\*([0-9.]+)\*\*", text)
        if m:
            numbers["p12.phase16.mean"] = float(m.group(1))

    numbers.update(_bibliography_counts())

    numbers["paper_v9.old_6_of_15_refs"] = len(re.findall(r"\b6 of 15\b|6-of-15", paper_v9))
    numbers["paper_v9.old_release_count_refs"] = paper_v9.count("168,793") + paper_v9.count("60,411") + paper_v9.count("36,566")
    numbers["paper_v9.old_citation_count_refs"] = paper_v9.count("22,522") + paper_v9.count("1,883")
    numbers["paper_v9.protocol_caption_overclaim_refs"] = paper_v9.count("survive judge change")
    top_level_cards = [
        ROOT / "models" / "DR-Judge-7B-LoRA" / "README.md",
        ROOT / "models" / "P12-RL-LoRA-v2" / "README.md",
    ]
    numbers["model_cards.top_level_placeholder_refs"] = sum(
        p.read_text(encoding="utf-8").count("[More Information Needed]")
        for p in top_level_cards
        if p.exists()
    )
    return {key: _round_float(value) for key, value in sorted(numbers.items())}


def _matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--dump-current", action="store_true", help="Print current computed numbers as JSON and exit.")
    args = parser.parse_args()

    actual = compute_numbers()
    if args.dump_current:
        print(json.dumps({"checks": {k: {"expected": v} for k, v in actual.items()}}, indent=2, sort_keys=True))
        return 0

    expected_payload = json.loads(args.expected.read_text(encoding="utf-8"))
    failures: list[str] = []
    print("Headline number verification")
    print("=" * 72)
    for key, spec in expected_payload["checks"].items():
        expected = spec["expected"]
        tolerance = float(spec.get("tolerance", 0))
        actual_value = actual.get(key)
        ok = _matches(actual_value, expected, tolerance)
        status = "OK" if ok else "FAIL"
        print(f"{status:4} {key:48} actual={actual_value!r} expected={expected!r} tol={tolerance:g}")
        if not ok:
            failures.append(key)

    extra = sorted(set(actual) - set(expected_payload["checks"]))
    if extra:
        print(f"\nNOTE: {len(extra)} computed checks are not in {args.expected}: {', '.join(extra[:20])}")

    if failures:
        print(f"\nFAILED: {len(failures)} headline checks drifted: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll headline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
