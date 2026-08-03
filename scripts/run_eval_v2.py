#!/usr/bin/env python3
"""Master evaluation script v2.

Orchestrates the full V2 evaluation pipeline:
1. Generate phase: run all patterns against all evaluation queries
2. Judge phase: score all generated reports with multi-judge ensemble
3. Analyze phase: run statistical analysis on results

Usage:
    python scripts/run_eval_v2.py --phase all         # Run everything
    python scripts/run_eval_v2.py --phase generate     # Only generate reports
    python scripts/run_eval_v2.py --phase judge        # Only judge existing reports
    python scripts/run_eval_v2.py --phase analyze      # Only run statistical analysis
    python scripts/run_eval_v2.py --patterns p0,p4     # Specific patterns
    python scripts/run_eval_v2.py --max-queries 10     # Quick test run
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import (
    JUDGE_MODEL,
    JUDGE_OPENAI_API_KEY,
    JUDGE_OPENAI_ENDPOINT,
    MAX_COST_PER_RUN,
    EVAL_PIPELINE,
    JUDGE,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
)
from deep_research.evaluation.execution_pipeline import ExecutionPipeline, RunResult
from deep_research.evaluation.judge_pipeline import JudgePipeline
from deep_research.evaluation.multi_judge import JudgeConfig, MultiJudge, EnsembleResult
from deep_research.evaluation.query_registry import QueryRegistry, EvalQuery
from deep_research.evaluation.rubric_v2 import build_rubric_from_test_query
from deep_research.evaluation.test_queries import get_all_queries


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT_DIR = CHECKPOINTS_DIR / "eval_v2"
DEFAULT_RESULTS_DIR = REPORTS_DIR / "eval_v2"

ALL_PATTERNS = ExecutionPipeline.PATTERN_NAMES


# ── Query loading ─────────────────────────────────────────────────────────────


class _EvalQueryAdapter:
    """Adapts a TestQuery into the duck-typed interface expected by pipelines.

    Provides .id, .query, and .rubric attributes.
    """

    def __init__(self, test_query):
        self._tq = test_query
        self.id = test_query.id
        self.query = test_query.query
        self.rubric = build_rubric_from_test_query(test_query)


def load_eval_queries(max_queries: int = 0) -> list:
    """Load evaluation queries, optionally limiting count.

    Loads from V2 query manifest (data/eval_queries_v2.json) using
    QueryRegistry.from_manifest(). Falls back to the original test queries
    with RubricV2 conversion if manifest is missing.

    Args:
        max_queries: Maximum number of queries to load (0 = all).

    Returns:
        List of query objects with .id, .query, .rubric attributes.
    """
    manifest_path = Path("data/eval_queries_v2.json")
    if manifest_path.exists():
        print(f"  Loading V2 query manifest: {manifest_path}")
        registry = QueryRegistry.from_manifest(manifest_path)
        eval_queries = registry.queries
        print(f"  Loaded {len(eval_queries)} queries from manifest")

        if max_queries > 0:
            eval_queries = eval_queries[:max_queries]
        return eval_queries

    # Fall back to test queries with rubric conversion
    print("  V2 manifest not found, falling back to test queries")
    test_queries = get_all_queries()
    eval_queries = [_EvalQueryAdapter(tq) for tq in test_queries]

    if max_queries > 0:
        eval_queries = eval_queries[:max_queries]

    return eval_queries


# ── Phase: Generate ───────────────────────────────────────────────────────────


async def phase_generate(
    queries: list,
    patterns: list[str],
    budget: float,
    max_concurrent: int,
    resume: bool,
    checkpoint_dir: Path,
    results_dir: Path,
    n_repeats: int = 1,
    token_budget: int = 0,
    random_seed: int = 42,
) -> list:
    """Run all patterns against all queries to generate reports."""
    total_runs = len(patterns) * len(queries) * n_repeats
    print(f"\n{'='*70}")
    print(f"  PHASE: GENERATE")
    print(f"  {len(patterns)} patterns x {len(queries)} queries x {n_repeats} repeats = {total_runs} runs")
    print(f"  Budget: ${budget:.2f}/run  Concurrency: {max_concurrent}")
    if token_budget:
        print(f"  Token budget: {token_budget:,} per run")
    print(f"  Random seed: {random_seed}")
    print(f"  Resume: {resume}")
    print(f"{'='*70}\n")

    pipeline = ExecutionPipeline(
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        budget_per_run=budget,
        max_concurrent=max_concurrent,
        n_repeats=n_repeats,
        token_budget=token_budget,
        random_seed=random_seed,
    )

    results = await pipeline.run_all(
        queries=queries,
        patterns=patterns,
        resume=resume,
    )

    # Print summary
    succeeded = sum(1 for r in results if r.succeeded)
    failed = sum(1 for r in results if r.status in ("error", "content_filter", "budget_exceeded"))
    skipped = sum(1 for r in results if r.status == "skipped")

    print(f"\n  Generate complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    return results


# ── Phase: Judge ──────────────────────────────────────────────────────────────


def build_judge_configs() -> list[JudgeConfig]:
    """Build judge configurations from environment and config defaults.

    GPT-5.2 is THE judge for this work (user directive 2026-06-12): the whole corpus is GPT-5.2-
    judged, so all reports that feed results/comparisons are GPT-5.2-judged too. GPT-4o is NOT used
    as a primary scorer (weaker generation + incomparable scale).
    """
    # Primary judge
    configs = [
        JudgeConfig(
            label="gpt52_primary",
            model=JUDGE_MODEL,
            endpoint=JUDGE_OPENAI_ENDPOINT,
            api_key=JUDGE_OPENAI_API_KEY,
            temperature=JUDGE.temperature,
        ),
    ]

    # Second judge with higher temperature for diversity
    configs.append(
        JudgeConfig(
            label="gpt52_diverse",
            model=JUDGE_MODEL,
            endpoint=JUDGE_OPENAI_ENDPOINT,
            api_key=JUDGE_OPENAI_API_KEY,
            temperature=0.5,
        ),
    )

    return configs


async def phase_judge(
    queries: list,
    patterns: list[str],
    resume: bool,
    results_dir: Path,
    output_dir: Path,
    passes_per_judge: int = 3,
) -> list:
    """Score all generated reports with multi-judge ensemble."""
    print(f"\n{'='*70}")
    print(f"  PHASE: JUDGE")
    print(f"  Scoring reports with multi-judge ensemble")
    print(f"  Resume: {resume}")
    print(f"{'='*70}\n")

    judge_configs = build_judge_configs()
    print(f"  Judges: {[j.label for j in judge_configs]}")
    print(f"  Passes per judge: {passes_per_judge}")
    total_evals = len(judge_configs) * passes_per_judge
    print(f"  Total evaluations per report: {total_evals}")

    multi_judge = MultiJudge(
        judges=judge_configs,
        passes_per_judge=passes_per_judge,
        max_concurrent=JUDGE.max_concurrent,
    )

    judge_pipeline = JudgePipeline(
        multi_judge=multi_judge,
        reports_dir=results_dir,
        output_dir=output_dir,
    )

    results = await judge_pipeline.score_all(
        queries=queries,
        patterns=patterns,
        resume=resume,
    )

    print(f"\n  Judge complete: {len(results)} reports scored")
    if results:
        avg_score = sum(r.ensemble_overall for r in results) / len(results)
        avg_agreement = sum(r.inter_judge_agreement for r in results) / len(results)
        print(f"  Avg overall score: {avg_score:.3f}")
        print(f"  Avg inter-judge agreement (kappa): {avg_agreement:.3f}")

    return results


# ── Phase: Analyze ────────────────────────────────────────────────────────────


def compute_efficiency_analysis(
    results_dir: Path,
    checkpoint_dir: Path,
    patterns: list[str],
    judge_results: list,  # list of EnsembleResult
) -> dict:
    """Compute quality-per-compute efficiency metrics.

    Loads RunResult checkpoints to get token/call counts,
    matches with judge scores, and computes efficiency rankings.

    Args:
        results_dir: Directory for saving output files.
        checkpoint_dir: Directory containing RunResult checkpoint JSON files.
        patterns: List of pattern names to analyze.
        judge_results: List of EnsembleResult objects from the judge phase.

    Returns:
        Dict mapping pattern names to their efficiency metrics, or empty
        dict if insufficient data is available.
    """
    # Load all RunResult checkpoints
    run_results_by_key: dict[tuple[str, str], RunResult] = {}
    for pattern in patterns:
        pattern_dir = checkpoint_dir / pattern
        if not pattern_dir.exists():
            continue
        for cp_file in sorted(pattern_dir.glob("*.json")):
            try:
                data = json.loads(cp_file.read_text())
                rr = RunResult(**data)
                if rr.succeeded:
                    run_results_by_key[(rr.pattern, rr.query_id)] = rr
            except (json.JSONDecodeError, TypeError):
                continue

    if not run_results_by_key:
        print("  No RunResult checkpoints found -- skipping efficiency analysis.")
        return {}

    # Build judge score lookup: (pattern, query_id) -> ensemble_overall
    judge_lookup: dict[tuple[str, str], float] = {}
    for er in judge_results:
        judge_lookup[(er.pattern_name, er.query_id)] = er.ensemble_overall

    # Per-pattern aggregation
    efficiency: dict[str, dict] = {}
    for pattern in patterns:
        # Collect matched (run, score) pairs
        matched_tokens: list[int] = []
        matched_input_tokens: list[int] = []
        matched_output_tokens: list[int] = []
        matched_llm_calls: list[int] = []
        matched_quality: list[float] = []
        matched_search_queries: list[int] = []

        for (p, qid), rr in run_results_by_key.items():
            if p != pattern:
                continue
            score = judge_lookup.get((pattern, qid))
            if score is None:
                continue

            matched_tokens.append(rr.total_tokens)
            matched_input_tokens.append(rr.total_input_tokens)
            matched_output_tokens.append(rr.total_output_tokens)
            matched_llm_calls.append(rr.llm_call_count)
            matched_quality.append(score)

            # Search queries from metadata
            search_qs = rr.metadata.get("search_queries_sent", [])
            matched_search_queries.append(
                len(search_qs) if isinstance(search_qs, list) else 0
            )

        if not matched_quality:
            continue

        n = len(matched_quality)
        avg_tokens = sum(matched_tokens) / n
        avg_input_tokens = sum(matched_input_tokens) / n
        avg_output_tokens = sum(matched_output_tokens) / n
        avg_llm_calls = sum(matched_llm_calls) / n
        avg_quality = sum(matched_quality) / n
        avg_search_queries = sum(matched_search_queries) / n

        # Efficiency ratios (guard against division by zero)
        quality_per_1k_tokens = (
            avg_quality / (avg_tokens / 1000.0) if avg_tokens > 0 else 0.0
        )
        quality_per_llm_call = (
            avg_quality / avg_llm_calls if avg_llm_calls > 0 else 0.0
        )
        quality_per_search_query = (
            avg_quality / avg_search_queries if avg_search_queries > 0 else 0.0
        )

        efficiency[pattern] = {
            "n_matched": n,
            "avg_tokens": avg_tokens,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_llm_calls": avg_llm_calls,
            "avg_quality": avg_quality,
            "quality_per_1k_tokens": quality_per_1k_tokens,
            "quality_per_llm_call": quality_per_llm_call,
            "avg_search_queries": avg_search_queries,
            "quality_per_search_query": quality_per_search_query,
        }

    if not efficiency:
        print("  No matched (run, verdict) pairs found -- skipping efficiency analysis.")
        return {}

    # Save to disk
    out_path = results_dir / "efficiency_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(efficiency, indent=2))
    print(f"\n  Efficiency analysis saved to: {out_path}")

    return efficiency


def generate_results_manifest(
    results_dir: Path,
    checkpoint_dir: Path,
    patterns: list[str],
    queries: list,
) -> Path | None:
    """Generate a unified CSV + JSON manifest linking all artifacts.

    Scans checkpoint files, report files, and verdict files for every
    (pattern, query_id) combination, and writes a single manifest with
    per-row metadata suitable for downstream analysis.

    Args:
        results_dir: Root results directory (contains reports/, verdicts/).
        checkpoint_dir: Directory containing RunResult checkpoint JSONs.
        patterns: List of pattern names.
        queries: List of query objects with .id (and optionally .source,
            .difficulty, .domain attributes).

    Returns:
        Path to the generated CSV manifest, or None if no data was found.
    """
    rows: list[dict] = []

    # Build query metadata lookup
    query_meta: dict[str, dict] = {}
    for q in queries:
        meta: dict = {"query_id": q.id}
        for attr in ("source", "difficulty", "domain"):
            meta[attr] = getattr(q, attr, "")
        query_meta[q.id] = meta

    # Collect all known query IDs from all sources
    all_query_ids: set[str] = {q.id for q in queries}

    for pattern in patterns:
        # Also discover query IDs from checkpoint / verdict files
        cp_dir = checkpoint_dir / pattern
        if cp_dir.exists():
            for f in cp_dir.glob("*.json"):
                all_query_ids.add(f.stem)

        verdict_dir = results_dir / "verdicts" / pattern
        if verdict_dir.exists():
            for f in verdict_dir.glob("*.json"):
                all_query_ids.add(f.stem)

    for pattern in patterns:
        for qid in sorted(all_query_ids):
            row: dict = {
                "query_id": qid,
                "pattern": pattern,
                "source": query_meta.get(qid, {}).get("source", ""),
                "difficulty": query_meta.get(qid, {}).get("difficulty", ""),
                "domain": query_meta.get(qid, {}).get("domain", ""),
            }

            # -- Checkpoint (RunResult) --
            cp_path = checkpoint_dir / pattern / f"{qid}.json"
            if cp_path.exists():
                try:
                    cp_data = json.loads(cp_path.read_text())
                    row["run_status"] = cp_data.get("status", "")
                    row["total_tokens"] = cp_data.get("total_tokens", 0)
                    row["total_input_tokens"] = cp_data.get("total_input_tokens", 0)
                    row["total_output_tokens"] = cp_data.get("total_output_tokens", 0)
                    row["llm_call_count"] = cp_data.get("llm_call_count", 0)
                    row["elapsed_seconds"] = cp_data.get("elapsed_seconds", 0.0)
                    row["cost_usd"] = cp_data.get("cost_usd", 0.0)
                except (json.JSONDecodeError, TypeError):
                    row["run_status"] = "checkpoint_error"
            else:
                row["run_status"] = "missing"

            # -- Report --
            report_path = results_dir / "reports" / pattern / f"{qid}.md"
            row["report_exists"] = report_path.exists()
            row["report_path"] = str(report_path) if report_path.exists() else ""

            # -- Verdict (EnsembleResult) --
            verdict_path = results_dir / "verdicts" / pattern / f"{qid}.json"
            row["verdict_exists"] = verdict_path.exists()
            row["verdict_path"] = str(verdict_path) if verdict_path.exists() else ""

            if verdict_path.exists():
                try:
                    vdata = json.loads(verdict_path.read_text())
                    row["ensemble_overall"] = vdata.get("ensemble_overall", "")
                    # Per-dimension scores
                    dims = vdata.get("ensemble_dimensions", {})
                    for dim_name, dim_score in sorted(dims.items()):
                        row[f"dim_{dim_name}"] = dim_score
                except (json.JSONDecodeError, TypeError):
                    row["ensemble_overall"] = ""
            else:
                row["ensemble_overall"] = ""

            rows.append(row)

    if not rows:
        print("  No data found for manifest generation.")
        return None

    # Determine all column names (union of all row keys, preserving order)
    base_cols = [
        "query_id", "pattern", "source", "difficulty", "domain",
        "run_status", "report_exists", "report_path",
        "verdict_exists", "verdict_path", "ensemble_overall",
    ]
    metric_cols = [
        "total_tokens", "total_input_tokens", "total_output_tokens",
        "llm_call_count", "elapsed_seconds", "cost_usd",
    ]
    # Discover all dim_* columns
    dim_cols = sorted({
        k for row in rows for k in row if k.startswith("dim_")
    })
    all_cols = base_cols + metric_cols + dim_cols

    # -- Write CSV --
    csv_path = results_dir / "results_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # -- Write JSON --
    json_path = results_dir / "results_manifest.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str))

    # -- Print summary --
    n_with_runs = sum(1 for r in rows if r.get("run_status") == "success")
    n_with_reports = sum(1 for r in rows if r.get("report_exists"))
    n_with_verdicts = sum(1 for r in rows if r.get("verdict_exists"))

    print(f"\n  Results manifest generated:")
    print(f"    Total rows: {len(rows)}")
    print(f"    Successful runs: {n_with_runs}")
    print(f"    Reports on disk: {n_with_reports}")
    print(f"    Verdicts on disk: {n_with_verdicts}")
    print(f"    CSV: {csv_path}")
    print(f"    JSON: {json_path}")

    return csv_path


async def phase_analyze(
    output_dir: Path,
    patterns: list[str],
    checkpoint_dir: Path | None = None,
    queries: list | None = None,
) -> None:
    """Run statistical analysis on judge results."""
    print(f"\n{'='*70}")
    print(f"  PHASE: ANALYZE")
    print(f"{'='*70}\n")

    judge_pipeline = JudgePipeline(
        multi_judge=None,  # type: ignore[arg-type]
        reports_dir=output_dir,
        output_dir=output_dir,
    )

    all_results = judge_pipeline.load_all_results()
    if not all_results:
        print("  No judge results found. Run judge phase first.")
        return

    print(f"  Loaded {len(all_results)} judge verdicts")

    # Build score matrix: pattern -> list of scores
    pattern_scores: dict[str, list[float]] = {}
    for r in all_results:
        pattern_scores.setdefault(r.pattern_name, []).append(r.ensemble_overall)

    # Print per-pattern summary
    print(f"\n  {'Pattern':<30} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*30} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    import numpy as np

    for pattern in patterns:
        scores = pattern_scores.get(pattern, [])
        if not scores:
            continue
        arr = np.array(scores)
        print(
            f"  {pattern:<30} {len(scores):>4} "
            f"{arr.mean():>8.3f} {arr.std():>8.3f} "
            f"{arr.min():>8.3f} {arr.max():>8.3f}"
        )

    # Run full statistical analysis
    try:
        from deep_research.evaluation.statistical_analysis import (
            run_full_analysis,
            generate_summary_markdown,
        )

        # Only run if we have scores for at least 2 patterns
        valid_patterns = [p for p in patterns if len(pattern_scores.get(p, [])) > 0]
        if len(valid_patterns) >= 2:
            # Align scores: only include queries scored for all patterns
            query_ids_per_pattern: dict[str, set[str]] = {}
            for r in all_results:
                query_ids_per_pattern.setdefault(r.pattern_name, set()).add(r.query_id)

            common_queries = set.intersection(
                *[query_ids_per_pattern.get(p, set()) for p in valid_patterns]
            )

            if len(common_queries) >= 3:
                print(f"\n  Running full statistical analysis on {len(common_queries)} common queries...")

                # Build aligned score matrix (queries x patterns)
                scores_by_query: dict[str, dict[str, float]] = {}
                for r in all_results:
                    if r.query_id in common_queries and r.pattern_name in valid_patterns:
                        scores_by_query.setdefault(r.query_id, {})[r.pattern_name] = (
                            r.ensemble_overall
                        )

                score_matrix = np.array(
                    [
                        [scores_by_query[q][p] for p in valid_patterns]
                        for q in sorted(common_queries)
                    ]
                )

                # Run the complete Demsar pipeline
                analysis = run_full_analysis(score_matrix, valid_patterns)

                print(f"  Friedman statistic: {analysis.omnibus.statistic:.3f}")
                print(f"  Friedman p-value: {analysis.omnibus.p_value:.4f}")
                print(f"  Significant: {analysis.omnibus.is_significant}")

                if analysis.pairwise:
                    print(f"  Pairwise comparisons: {len(analysis.pairwise)}")
                    sig_pairs = sum(1 for p in analysis.pairwise if p.is_significant)
                    print(f"  Significant pairs (Holm-corrected): {sig_pairs}")

                if analysis.bootstrap_cis:
                    print(f"\n  Bootstrap 95% CIs (IQM):")
                    for ci in sorted(analysis.bootstrap_cis, key=lambda c: c.system):
                        print(f"    {ci.system:<30} [{ci.iqm_ci_lower:.3f}, {ci.iqm_ci_upper:.3f}] (IQM={ci.iqm:.3f})")

                if analysis.pairwise:
                    print(f"\n  Effect sizes (Cliff's Delta):")
                    for pw in sorted(analysis.pairwise, key=lambda p: abs(p.effect_size), reverse=True):
                        pair_label = f"{pw.system_a} vs {pw.system_b}"
                        print(f"    {pair_label:<50} {pw.effect_size:+.3f} ({pw.effect_size_label})")

                # Save full analysis as JSON
                analysis_path = output_dir / "statistical_analysis.json"
                analysis_data = {
                    "friedman_statistic": analysis.omnibus.statistic,
                    "friedman_p_value": analysis.omnibus.p_value,
                    "is_significant": analysis.omnibus.is_significant,
                    "n_queries": len(common_queries),
                    "n_patterns": len(valid_patterns),
                    "patterns": valid_patterns,
                    "avg_ranks": analysis.omnibus.avg_ranks,
                    "bootstrap_cis": [
                        {
                            "system": ci.system,
                            "mean": ci.mean,
                            "ci_lower": ci.ci_lower,
                            "ci_upper": ci.ci_upper,
                            "iqm": ci.iqm,
                            "iqm_ci_lower": ci.iqm_ci_lower,
                            "iqm_ci_upper": ci.iqm_ci_upper,
                        }
                        for ci in analysis.bootstrap_cis
                    ],
                    "pairwise": [
                        {
                            "system_a": pw.system_a,
                            "system_b": pw.system_b,
                            "test": pw.test_name,
                            "statistic": pw.statistic,
                            "p_value_raw": pw.p_value_raw,
                            "p_value_corrected": pw.p_value_corrected,
                            "is_significant": pw.is_significant,
                            "effect_size": pw.effect_size,
                            "effect_size_label": pw.effect_size_label,
                        }
                        for pw in analysis.pairwise
                    ],
                }
                analysis_path.write_text(json.dumps(analysis_data, indent=2))
                print(f"\n  Analysis saved to: {analysis_path}")

                # Generate and save the full markdown report
                md_report = generate_summary_markdown(analysis)
                md_path = output_dir / "statistical_analysis_report.md"
                md_path.write_text(md_report)
                print(f"  Markdown report: {md_path}")

                # ── Per-difficulty stratified analysis ──────────────────
                if queries:
                    print(f"\n  --- Per-Difficulty Stratified Analysis ---")
                    # Build query difficulty lookup
                    difficulty_lookup: dict[str, str] = {}
                    for q in queries:
                        diff = getattr(q, "difficulty", "")
                        if diff:
                            difficulty_lookup[q.id] = diff

                    if difficulty_lookup:
                        # Group results by difficulty
                        diff_scores: dict[str, dict[str, list[float]]] = {}
                        for r in all_results:
                            diff = difficulty_lookup.get(r.query_id, "unknown")
                            diff_scores.setdefault(diff, {}).setdefault(
                                r.pattern_name, []
                            ).append(r.ensemble_overall)

                        for diff_level in sorted(diff_scores.keys()):
                            level_data = diff_scores[diff_level]
                            n_queries_level = max(
                                len(scores) for scores in level_data.values()
                            ) if level_data else 0

                            print(f"\n  Difficulty: {diff_level} (n={n_queries_level})")

                            if n_queries_level < 5:
                                print(f"    WARNING: n={n_queries_level} < 5 — results are descriptive only, no statistical tests")

                            # Print per-pattern means for this difficulty level
                            print(f"    {'Pattern':<30} {'N':>4} {'Mean':>8} {'Std':>8}")
                            print(f"    {'-'*30} {'-'*4} {'-'*8} {'-'*8}")
                            for pattern in valid_patterns:
                                scores = level_data.get(pattern, [])
                                if scores:
                                    arr = np.array(scores)
                                    print(f"    {pattern:<30} {len(scores):>4} {arr.mean():>8.3f} {arr.std():>8.3f}")

                            # Only run Friedman if enough data
                            if n_queries_level >= 10 and len(valid_patterns) >= 2:
                                # Build aligned matrix for this difficulty
                                diff_query_ids: dict[str, set[str]] = {}
                                for r in all_results:
                                    if (
                                        difficulty_lookup.get(r.query_id) == diff_level
                                        and r.pattern_name in valid_patterns
                                    ):
                                        diff_query_ids.setdefault(r.pattern_name, set()).add(r.query_id)

                                diff_common = set.intersection(
                                    *[diff_query_ids.get(p, set()) for p in valid_patterns]
                                ) if all(p in diff_query_ids for p in valid_patterns) else set()

                                if len(diff_common) >= 5:
                                    diff_by_query: dict[str, dict[str, float]] = {}
                                    for r in all_results:
                                        if (
                                            r.query_id in diff_common
                                            and r.pattern_name in valid_patterns
                                        ):
                                            diff_by_query.setdefault(r.query_id, {})[
                                                r.pattern_name
                                            ] = r.ensemble_overall

                                    diff_matrix = np.array([
                                        [diff_by_query[q][p] for p in valid_patterns]
                                        for q in sorted(diff_common)
                                    ])

                                    try:
                                        diff_analysis = run_full_analysis(diff_matrix, valid_patterns)
                                        print(f"    Friedman p={diff_analysis.omnibus.p_value:.4f}, significant={diff_analysis.omnibus.is_significant}")
                                    except Exception as strat_err:
                                        print(f"    Friedman test failed: {strat_err}")

                        # Save stratified results
                        strat_path = output_dir / "stratified_analysis.json"
                        strat_data: dict = {}
                        for diff_level, level_data in diff_scores.items():
                            strat_data[diff_level] = {
                                pattern: {
                                    "n": len(scores),
                                    "mean": float(np.mean(scores)) if scores else 0.0,
                                    "std": float(np.std(scores)) if scores else 0.0,
                                }
                                for pattern, scores in level_data.items()
                            }
                        strat_path.write_text(json.dumps(strat_data, indent=2))
                        print(f"\n  Stratified analysis saved to: {strat_path}")

                # ── Generate visualizations ──────────────────────────────
                generated_figures: list[Path] = []
                try:
                    from deep_research.visualization.charts import (
                        generate_all_figures,
                        cost_quality_scatter,
                    )

                    figures_dir = output_dir / "figures"

                    # Build dimension_scores dict: {pattern: {dim: avg_score}}
                    dim_accum: dict[str, dict[str, list[float]]] = {}
                    for r in all_results:
                        if r.pattern_name in valid_patterns:
                            dim_accum.setdefault(r.pattern_name, {})
                            for dim, score in r.ensemble_dimensions.items():
                                dim_accum[r.pattern_name].setdefault(dim, []).append(score)
                    dimension_scores = {
                        p: {d: sum(vals) / len(vals) for d, vals in dims.items()}
                        for p, dims in dim_accum.items()
                        if dims
                    }

                    # Build CI data for bootstrap CI plot
                    ci_data_for_chart = None
                    if analysis.bootstrap_cis:
                        ci_data_for_chart = [
                            {
                                "system": ci.system,
                                "mean": ci.mean,
                                "ci_lower": ci.ci_lower,
                                "ci_upper": ci.ci_upper,
                            }
                            for ci in analysis.bootstrap_cis
                        ]

                    # Compute CD value for CD diagram (Nemenyi critical difference)
                    cd_value = 0.0
                    if analysis.omnibus.is_significant and len(valid_patterns) >= 2:
                        # CD = q_alpha * sqrt(k*(k+1)/(6*N))
                        # q_alpha values for alpha=0.05 (studentized range / sqrt(2))
                        q_alpha_table = {
                            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                        }
                        k = len(valid_patterns)
                        n_q = len(common_queries)
                        q_alpha = q_alpha_table.get(k, 2.850)
                        cd_value = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_q))

                    generated_figures = generate_all_figures(
                        results_dir=output_dir,
                        output_dir=figures_dir,
                        dimension_scores=dimension_scores if dimension_scores else None,
                        ci_data=ci_data_for_chart,
                        score_matrix=score_matrix,
                        system_names=valid_patterns,
                        avg_ranks=analysis.omnibus.avg_ranks if analysis.omnibus.avg_ranks else None,
                        n_tasks=len(common_queries),
                        cd=cd_value,
                    )

                    if generated_figures:
                        print(f"\n  Generated {len(generated_figures)} figures in {figures_dir}:")
                        for fig_path in generated_figures:
                            print(f"    {fig_path.name}")

                except ImportError as viz_err:
                    print(f"  Visualization not available: {viz_err}")
                except Exception as viz_err:
                    print(f"  Visualization generation failed: {viz_err}")

            else:
                print(f"  Not enough common queries ({len(common_queries)}) for statistical tests")
        else:
            print(f"  Not enough patterns with scores ({len(valid_patterns)}) for statistical tests")

    except ImportError as e:
        print(f"  Statistical analysis not available: {e}")

    # ── Efficiency analysis (Fix #14) ────────────────────────────────────
    if checkpoint_dir is not None:
        print(f"\n  --- Efficiency Analysis ---")
        efficiency = compute_efficiency_analysis(
            results_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            patterns=patterns,
            judge_results=all_results,
        )

        if efficiency:
            # Print efficiency summary table
            print(f"\n  {'Pattern':<30} {'Quality':>8} {'Tokens':>10} {'Q/1kT':>8} {'Calls':>7} {'Q/Call':>8} {'Searches':>9} {'Q/Srch':>8}")
            print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")
            for p in patterns:
                if p not in efficiency:
                    continue
                e = efficiency[p]
                print(
                    f"  {p:<30} {e['avg_quality']:>8.3f} {e['avg_tokens']:>10,.0f} "
                    f"{e['quality_per_1k_tokens']:>8.4f} {e['avg_llm_calls']:>7.1f} "
                    f"{e['quality_per_llm_call']:>8.4f} {e['avg_search_queries']:>9.1f} "
                    f"{e['quality_per_search_query']:>8.4f}"
                )

            # Generate cost-quality scatter if token data exists
            try:
                from deep_research.visualization.charts import cost_quality_scatter

                cost_data = [
                    {
                        "pattern": p,
                        "quality": e["avg_quality"],
                        "tokens": e["avg_tokens"],
                    }
                    for p, e in efficiency.items()
                ]
                if cost_data:
                    scatter_path = output_dir / "figures" / "cost_quality.png"
                    scatter_path.parent.mkdir(parents=True, exist_ok=True)
                    cost_quality_scatter(cost_data, scatter_path)
                    print(f"\n  Cost-quality scatter: {scatter_path}")
            except ImportError:
                pass
            except Exception as scatter_err:
                print(f"  Cost-quality scatter failed: {scatter_err}")

    # ── Results manifest (Fix #16) ───────────────────────────────────────
    if checkpoint_dir is not None and queries is not None:
        print(f"\n  --- Results Manifest ---")
        generate_results_manifest(
            results_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            patterns=patterns,
            queries=queries,
        )

    # ── Human evaluation check ────────────────────────────────────────
    human_eval_dir = output_dir / "human_eval"
    if human_eval_dir.exists() and any(human_eval_dir.glob("*.json")):
        print(f"\n  --- Human Evaluation Calibration ---")
        try:
            from deep_research.evaluation.human_eval import (
                HumanEvalResult,
                compute_judge_human_agreement,
                generate_human_eval_report,
            )
            # Load human eval data
            human_results: list = []
            for hf in sorted(human_eval_dir.glob("*.json")):
                data = json.loads(hf.read_text())
                human_results.append(HumanEvalResult(**data))

            if human_results and all_results:
                # Build judge score lookup
                judge_scores: dict[str, dict[str, float]] = {}
                for r in all_results:
                    report_id = f"{r.pattern_name}/{r.query_id}"
                    judge_scores[report_id] = r.ensemble_dimensions

                agreement = compute_judge_human_agreement(judge_scores, human_results)
                print(f"  Reports with human eval: {agreement.n_reports}")
                print(f"  Judge-Human Cohen's kappa: {agreement.overall_kappa:.3f}")
                print(f"  Judge-Human Pearson r: {agreement.overall_correlation:.3f}")
                print(f"  Agreement rate: {agreement.agreement_rate:.1%}")
                print(f"  Judge bias: {agreement.judge_bias:+.3f}")

                # Save report
                report = generate_human_eval_report(human_results, agreement)
                report_path = output_dir / "human_eval_report.md"
                report_path.write_text(report)
                print(f"  Report: {report_path}")
        except Exception as he_err:
            print(f"  Human eval processing failed: {he_err}")
    else:
        print(f"\n  Human evaluation: No data found at {human_eval_dir}")
        print(f"  To calibrate the LLM judge, have 3+ human evaluators score")
        print(f"  15-20 reports and save results as JSON in {human_eval_dir}/")

    # ── Summary of generated files ───────────────────────────────────────
    print(f"\n  --- Generated Files Summary ---")
    generated_files = []
    for suffix in (
        "statistical_analysis.json",
        "statistical_analysis_report.md",
        "efficiency_analysis.json",
        "results_manifest.csv",
        "results_manifest.json",
    ):
        candidate = output_dir / suffix
        if candidate.exists():
            generated_files.append(candidate)
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for fig in sorted(figures_dir.glob("*.png")):
            generated_files.append(fig)
    if generated_files:
        for gf in generated_files:
            print(f"    {gf}")
    else:
        print("    (none)")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="V2 evaluation pipeline: generate, judge, analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_eval_v2.py --phase all              # Run everything
    python scripts/run_eval_v2.py --phase generate         # Only generate reports
    python scripts/run_eval_v2.py --phase judge            # Only score reports
    python scripts/run_eval_v2.py --phase analyze          # Only analyze scores
    python scripts/run_eval_v2.py --patterns p0,p4         # Specific patterns
    python scripts/run_eval_v2.py --max-queries 2          # Quick test
    python scripts/run_eval_v2.py --no-resume              # Start fresh
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "generate", "judge", "analyze"],
        help="Pipeline phase to run (default: all)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="",
        help="Comma-separated pattern names (default: all 6)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max queries to use (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=MAX_COST_PER_RUN,
        help=f"Budget per run in USD (default: {MAX_COST_PER_RUN})",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=EVAL_PIPELINE.max_concurrent_runs,
        help=f"Max concurrent pattern runs (default: {EVAL_PIPELINE.max_concurrent_runs})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoints",
    )
    parser.add_argument(
        "--passes-per-judge",
        type=int,
        default=EVAL_PIPELINE.passes_per_judge,
        help=f"Number of passes per judge model (default: {EVAL_PIPELINE.passes_per_judge})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=EVAL_PIPELINE.default_n_repeats,
        help=f"Number of repeated runs per pattern x query (default: {EVAL_PIPELINE.default_n_repeats}, for variance estimation)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="Token budget per run (0 = unlimited). Use for equalized-compute comparison.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=EVAL_PIPELINE.default_random_seed,
        help=f"Random seed for run-order shuffling (default: {EVAL_PIPELINE.default_random_seed}). Use different seeds for sensitivity analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 pattern x 1 query through full pipeline to verify setup",
    )

    args = parser.parse_args()

    # Parse patterns
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(",")]
        # Validate
        for p in patterns:
            if p not in ALL_PATTERNS:
                print(f"ERROR: Unknown pattern '{p}'. Valid: {ALL_PATTERNS}")
                sys.exit(1)
    else:
        patterns = list(ALL_PATTERNS)

    resume = not args.no_resume
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)

    # Dry-run: limit to 1 pattern x 1 query
    if args.dry_run:
        args.max_queries = 1
        patterns = patterns[:1]
        args.phase = "all"
        print("  *** DRY RUN: 1 pattern x 1 query ***")

    # Load queries
    queries = load_eval_queries(max_queries=args.max_queries)

    print(f"V2 Evaluation Pipeline")
    print(f"  Phase: {args.phase}")
    print(f"  Patterns: {patterns}")
    print(f"  Queries: {len(queries)}")
    print(f"  Budget: ${args.budget:.2f}/run")
    print(f"  Repeats: {args.n_repeats}")
    if args.token_budget:
        print(f"  Token budget: {args.token_budget:,}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Resume: {resume}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Results dir: {results_dir}")

    start_time = time.time()

    # Run phases
    if args.phase in ("all", "generate"):
        await phase_generate(
            queries=queries,
            patterns=patterns,
            budget=args.budget,
            max_concurrent=args.max_concurrent,
            resume=resume,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            n_repeats=args.n_repeats,
            token_budget=args.token_budget,
            random_seed=args.random_seed,
        )

    if args.phase in ("all", "judge"):
        await phase_judge(
            queries=queries,
            patterns=patterns,
            resume=resume,
            results_dir=results_dir,
            output_dir=results_dir,
            passes_per_judge=args.passes_per_judge,
        )

    if args.phase in ("all", "analyze"):
        await phase_analyze(
            output_dir=results_dir,
            patterns=patterns,
            checkpoint_dir=checkpoint_dir,
            queries=queries,
        )

    # Generate results manifest after all phases (Fix #16)
    # Only if not already generated inside phase_analyze and any artifacts exist
    if args.phase != "analyze":
        has_any_artifacts = (
            (results_dir / "verdicts").exists()
            or (results_dir / "reports").exists()
            or checkpoint_dir.exists()
        )
        if has_any_artifacts:
            manifest_path = results_dir / "results_manifest.csv"
            if not manifest_path.exists():
                print(f"\n  --- Generating Results Manifest ---")
                generate_results_manifest(
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir,
                    patterns=patterns,
                    queries=queries,
                )

    total_elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  V2 EVALUATION COMPLETE")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Results: {results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
