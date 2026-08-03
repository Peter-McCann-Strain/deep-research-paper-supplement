#!/usr/bin/env python3
"""Prepare human evaluation materials.

Generates anonymized evaluation forms from a stratified sample of reports,
following the protocol in docs/human_evaluation_protocol.md.

Usage:
    python scripts/prepare_human_eval.py --sample-size 30
    python scripts/prepare_human_eval.py --load-results /path/to/verdicts/
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from deep_research.evaluation.human_eval import (
    HumanVerdict,
    HumanEvalResult,
    compute_judge_human_agreement,
)
from deep_research.evaluation.query_registry import QueryRegistry, EvalQuery
from deep_research.evaluation.rubric_v2 import RubricV2
from deep_research.evaluation.multi_judge import fleiss_kappa


# ── Pattern names ────────────────────────────────────────────────────────────

PATTERNS = [
    "p0_baseline",
    "p1_iterative_rag",
    "p2_supervisor_parallel",
    "p3_meridian",
    "p4_perspective_storm",
    "p5_hierarchical_wd",
    "p6_reactive_interleaved",
]

DIFFICULTIES = ["simple", "moderate", "complex"]


# ── Sample selection ─────────────────────────────────────────────────────────


def select_sample(
    reports_dir: Path,
    manifest_path: Path,
    target_size: int = 30,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Select a stratified sample of (pattern, query_id) tuples.

    Stratifies by pattern (6) x difficulty (3) = 18 strata.
    Selects ceil(target_size / 18) per stratum.

    Args:
        reports_dir: Directory containing ``{pattern}/{query_id}.md`` files.
        manifest_path: Path to ``eval_queries_v2.json`` manifest.
        target_size: Target total number of reports to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of ``(pattern, query_id)`` tuples.
    """
    rng = random.Random(seed)

    # Load query metadata for difficulty lookup
    registry = QueryRegistry.from_manifest(manifest_path)
    query_map: dict[str, EvalQuery] = {q.id: q for q in registry.queries}

    # Discover all completed reports
    available: list[tuple[str, str, str]] = []  # (pattern, query_id, difficulty)
    for pattern in PATTERNS:
        pattern_dir = reports_dir / pattern
        if not pattern_dir.is_dir():
            continue
        for report_path in sorted(pattern_dir.glob("*.md")):
            query_id = report_path.stem
            eq = query_map.get(query_id)
            difficulty = eq.difficulty if eq else "moderate"
            available.append((pattern, query_id, difficulty))

    if not available:
        print("WARNING: No completed reports found in", reports_dir)
        return []

    # Build strata: pattern x difficulty
    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for pattern, query_id, difficulty in available:
        strata[(pattern, difficulty)].append((pattern, query_id))

    # Calculate per-stratum target
    n_strata = len(PATTERNS) * len(DIFFICULTIES)
    per_stratum = max(1, math.ceil(target_size / n_strata))

    selected: list[tuple[str, str]] = []
    leftover: list[tuple[str, str]] = []

    for key in sorted(strata.keys()):
        items = strata[key]
        rng.shuffle(items)
        take = min(per_stratum, len(items))
        selected.extend(items[:take])
        leftover.extend(items[take:])

    # If we overshot, trim; if we undershot, fill from leftover
    if len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]
    elif len(selected) < target_size:
        rng.shuffle(leftover)
        remaining = target_size - len(selected)
        selected.extend(leftover[:remaining])

    print(f"Selected {len(selected)} reports from {len(available)} available")
    print(f"  Strata populated: {len(strata)} / {n_strata}")
    return selected


# ── Form generation ──────────────────────────────────────────────────────────


def generate_forms(
    selected: list[tuple[str, str]],
    reports_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Generate anonymized evaluation forms for the selected sample.

    Creates:
      - ``forms/R-{id}.json`` for each report (with query, report text, criteria)
      - ``mapping.json`` (secret mapping from anonymous ID to pattern/query_id)
      - ``evaluation_template.json`` (empty template for evaluators)
      - ``README.md`` explaining the evaluation process

    Args:
        selected: List of ``(pattern, query_id)`` from :func:`select_sample`.
        reports_dir: Directory containing ``{pattern}/{query_id}.md`` files.
        manifest_path: Path to ``eval_queries_v2.json`` manifest.
        output_dir: Output directory for human evaluation materials.
        seed: Random seed for shuffling the order of reports.

    Returns:
        Path to the output directory.
    """
    rng = random.Random(seed)

    # Load query metadata
    registry = QueryRegistry.from_manifest(manifest_path)
    query_map: dict[str, EvalQuery] = {q.id: q for q in registry.queries}

    # Create output dirs
    forms_dir = output_dir / "forms"
    forms_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle report order so evaluators don't see patterns in sequence
    shuffled = list(selected)
    rng.shuffle(shuffled)

    # Build mapping
    mapping: list[dict] = []
    for idx, (pattern, query_id) in enumerate(shuffled, start=1):
        anon_id = f"R-{idx:04d}"
        mapping.append({
            "report_id": anon_id,
            "pattern": pattern,
            "query_id": query_id,
        })

    # Save mapping (secret)
    mapping_path = output_dir / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"Mapping saved to: {mapping_path}")

    # Generate form for each report
    for entry in mapping:
        anon_id = entry["report_id"]
        pattern = entry["pattern"]
        query_id = entry["query_id"]

        # Load report text
        report_path = reports_dir / pattern / f"{query_id}.md"
        if report_path.exists():
            report_text = report_path.read_text()
        else:
            report_text = f"[Report not found: {report_path}]"

        # Load query and rubric
        eq = query_map.get(query_id)
        query_text = eq.query if eq else f"[Query not found: {query_id}]"
        rubric = eq.rubric if eq else None

        # Build criteria list
        criteria_list: list[dict] = []
        if rubric:
            for i, crit in enumerate(rubric.criteria):
                criteria_list.append({
                    "index": i,
                    "dimension": crit.dimension,
                    "text": crit.text,
                    "weight": crit.weight,
                })

        # Create form
        form = {
            "report_id": anon_id,
            "query_text": query_text,
            "report_text": report_text,
            "criteria": criteria_list,
            "instructions": (
                "For each criterion, provide: "
                "verdict (SATISFIED/NOT_SATISFIED), "
                "confidence (0-1), brief comment."
            ),
        }

        form_path = forms_dir / f"{anon_id}.json"
        form_path.write_text(json.dumps(form, indent=2))

    print(f"Generated {len(mapping)} forms in: {forms_dir}")

    # Create evaluation template
    template = {
        "evaluator_id": "",
        "report_id": "R-0001",
        "verdicts": [
            {
                "criterion_index": 0,
                "verdict": "",
                "confidence": 0.0,
                "comment": "",
            }
        ],
        "overall_notes": "",
        "time_minutes": 0,
    }
    template_path = output_dir / "evaluation_template.json"
    template_path.write_text(json.dumps(template, indent=2))
    print(f"Template saved to: {template_path}")

    # Create README
    readme_text = _build_readme(len(mapping))
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_text)
    print(f"README saved to: {readme_path}")

    return output_dir


def _build_readme(n_reports: int) -> str:
    """Build the README.md content for the human evaluation directory."""
    return f"""\
# Human Evaluation Materials

## Overview

This directory contains anonymized evaluation forms for {n_reports} research
reports, selected via stratified sampling across patterns and difficulty levels.

## Directory Structure

- `forms/R-XXXX.json` -- Individual evaluation forms (one per report)
- `mapping.json` -- **SECRET**: Maps anonymous IDs to pattern/query_id (do NOT share with evaluators)
- `evaluation_template.json` -- Empty template showing the expected response format
- `completed/` -- Directory for completed evaluation JSONs (create before use)

## Evaluation Process

1. Each evaluator receives a set of `forms/R-XXXX.json` files.
2. For each form, the evaluator reads the query and report text.
3. For each criterion listed, the evaluator provides:
   - `verdict`: "SATISFIED" or "NOT_SATISFIED"
   - `confidence`: A float between 0 and 1 indicating self-assessed confidence
   - `comment`: A brief justification for the verdict
4. The evaluator also records:
   - `overall_notes`: Any general observations about the report
   - `time_minutes`: Approximate time spent on this evaluation
5. Completed evaluations should follow the format in `evaluation_template.json`.
6. Save completed evaluations as `completed/R-XXXX_{{evaluator_id}}.json`.

## Anonymization

Report IDs (R-0001, R-0002, ...) are randomly assigned and do NOT correspond
to pattern or query order. The mapping is stored in `mapping.json` and should
only be used during analysis, never shared with evaluators.

## Inter-Annotator Agreement

After all evaluations are collected, run:

    python scripts/prepare_human_eval.py --load-results reports/eval_v2/human_eval/completed/

This computes Fleiss' kappa for inter-annotator agreement and optionally
compares human evaluations against the LLM judge scores.
"""


# ── Load completed evaluations ───────────────────────────────────────────────


def load_completed_evaluations(
    completed_dir: Path,
    mapping_path: Path,
) -> list[HumanEvalResult]:
    """Load completed evaluation JSONs and compute agreement metrics.

    Args:
        completed_dir: Directory containing ``R-XXXX_{evaluator_id}.json`` files.
        mapping_path: Path to ``mapping.json`` for de-anonymization.

    Returns:
        List of :class:`HumanEvalResult` objects, one per report.
    """
    # Load mapping
    mapping_data = json.loads(mapping_path.read_text())
    id_to_meta: dict[str, dict] = {
        m["report_id"]: m for m in mapping_data
    }

    # Collect all completed evaluations grouped by report_id
    evals_by_report: dict[str, list[dict]] = defaultdict(list)
    for eval_path in sorted(completed_dir.glob("*.json")):
        data = json.loads(eval_path.read_text())
        report_id = data.get("report_id", "")
        if report_id:
            evals_by_report[report_id].append(data)

    if not evals_by_report:
        print("WARNING: No completed evaluations found in", completed_dir)
        return []

    # Build HumanEvalResult objects
    results: list[HumanEvalResult] = []
    for report_id, evaluations in sorted(evals_by_report.items()):
        meta = id_to_meta.get(report_id, {})
        pattern = meta.get("pattern", "unknown")
        query_id = meta.get("query_id", "unknown")
        evaluators = list({e.get("evaluator_id", "anon") for e in evaluations})

        verdicts: list[HumanVerdict] = []
        for eval_data in evaluations:
            evaluator_id = eval_data.get("evaluator_id", "anon")
            for v in eval_data.get("verdicts", []):
                verdicts.append(HumanVerdict(
                    evaluator_id=evaluator_id,
                    report_id=report_id,
                    criterion=str(v.get("criterion_index", "")),
                    dimension="",  # Will be enriched if needed
                    verdict=v.get("verdict", ""),
                    confidence=float(v.get("confidence", 0.0)),
                    comment=v.get("comment", ""),
                ))

        her = HumanEvalResult(
            report_id=report_id,
            pattern=pattern,
            query_id=query_id,
            evaluators=evaluators,
            verdicts=verdicts,
        )
        results.append(her)

    # Compute Fleiss' kappa across all reports and criteria
    _compute_and_print_agreement(results)

    return results


def _compute_and_print_agreement(results: list[HumanEvalResult]) -> None:
    """Compute and print Fleiss' kappa for inter-annotator agreement."""
    # Build a ratings matrix: rows = (report, criterion), columns = [NOT_SATISFIED, SATISFIED]
    # Each cell = count of raters choosing that category
    rows: list[list[int]] = []

    for her in results:
        # Group verdicts by criterion
        by_criterion: dict[str, list[str]] = defaultdict(list)
        for v in her.verdicts:
            by_criterion[v.criterion].append(v.verdict)

        for crit_id, verdict_list in sorted(by_criterion.items()):
            n_sat = sum(1 for v in verdict_list if v == "SATISFIED")
            n_not = sum(1 for v in verdict_list if v == "NOT_SATISFIED")
            if n_sat + n_not >= 2:  # Need at least 2 raters
                rows.append([n_not, n_sat])

    if not rows:
        print("Insufficient data to compute Fleiss' kappa (need >= 2 raters per item)")
        return

    matrix = np.array(rows, dtype=float)
    kappa = fleiss_kappa(matrix)

    print(f"\n{'='*60}")
    print(f"  Inter-Annotator Agreement")
    print(f"{'='*60}")
    print(f"  Items rated by >= 2 evaluators: {len(rows)}")
    print(f"  Fleiss' kappa: {kappa:.3f}")

    if kappa < 0.0:
        interpretation = "Less than chance agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    print(f"  Interpretation: {interpretation}")
    print(f"{'='*60}")


# ── Main CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare human evaluation materials or load completed results"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Target number of reports to sample (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eval_v2/human_eval"),
        help="Output directory for evaluation materials (default: reports/eval_v2/human_eval)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/eval_v2/reports"),
        help="Directory containing generated reports (default: reports/eval_v2/reports)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval_queries_v2.json"),
        help="Query manifest path (default: data/eval_queries_v2.json)",
    )
    parser.add_argument(
        "--load-results",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load completed evaluations from PATH and compute agreement",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    if args.load_results:
        # Load and analyze completed evaluations
        mapping_path = args.output_dir / "mapping.json"
        if not mapping_path.exists():
            print(f"ERROR: mapping.json not found at {mapping_path}")
            print("Run without --load-results first to generate evaluation materials.")
            sys.exit(1)

        results = load_completed_evaluations(args.load_results, mapping_path)
        print(f"\nLoaded {len(results)} report evaluations")

        # Summary
        total_verdicts = sum(len(r.verdicts) for r in results)
        all_evaluators = set()
        for r in results:
            all_evaluators.update(r.evaluators)
        print(f"Total verdicts: {total_verdicts}")
        print(f"Unique evaluators: {len(all_evaluators)}")

    else:
        # Generate evaluation materials
        print("=" * 60)
        print("  Human Evaluation Material Generator")
        print("=" * 60)

        if not args.manifest.exists():
            print(f"ERROR: Manifest not found at {args.manifest}")
            sys.exit(1)

        selected = select_sample(
            reports_dir=args.reports_dir,
            manifest_path=args.manifest,
            target_size=args.sample_size,
            seed=args.seed,
        )

        if not selected:
            print("No reports selected. Check that reports exist in", args.reports_dir)
            sys.exit(1)

        generate_forms(
            selected=selected,
            reports_dir=args.reports_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        print(f"\nDone. Materials written to: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
