#!/usr/bin/env python3
"""Comprehensive evaluation: all patterns × all queries × all benchmarks.

This script orchestrates:
1. All patterns (P0-P5) against all 5 test queries
2. All patterns against downloaded benchmark datasets (DRACO, ResearchQA, LitQA2, DeepSearchQA)

Progress is saved after every individual run, so partial results are preserved
if the script is interrupted.

Usage:
    # Run everything
    python scripts/run_full_evaluation.py

    # Run specific patterns only
    python scripts/run_full_evaluation.py --patterns p0,p1

    # Run only test queries (skip benchmarks)
    python scripts/run_full_evaluation.py --test-queries-only

    # Run only benchmarks (skip test queries)
    python scripts/run_full_evaluation.py --benchmarks-only

    # Limit benchmark queries per dataset
    python scripts/run_full_evaluation.py --max-bench-queries 3

    # Higher budget per query
    python scripts/run_full_evaluation.py --budget 5.0

    # Resume from saved progress
    python scripts/run_full_evaluation.py --resume
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Configuration ──────────────────────────────────────────────────────────

PATTERN_ORDER = [
    "p0_baseline",         # Fastest/cheapest — establishes floor
    "p1_iterative_rag",    # Next simplest
    "p2_supervisor_parallel",
    "p3_meridian",
    "p4_perspective_storm",
    "p5_hierarchical_wd",
    "p6_reactive_interleaved",  # Reactive agent loop — runs last
]

BENCHMARK_CONFIGS = [
    # (name, import_path, max_queries_default)
    ("draco", "deep_research.benchmarks.draco.DRACOBenchmark", 5),
    ("research_qa", "deep_research.benchmarks.research_qa.ResearchQABenchmark", 5),
    ("litqa2", "deep_research.benchmarks.litqa2.LitQA2Benchmark", 10),
    ("deepsearch_qa", "deep_research.benchmarks.deepsearch_qa.DeepSearchQABenchmark", 5),
]


def load_benchmark_class(import_path: str):
    """Dynamically import a benchmark class."""
    import importlib
    module_path, class_name = import_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


# ── Progress tracking ──────────────────────────────────────────────────────

class ProgressTracker:
    """Saves and loads evaluation progress to disk."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = save_dir / "progress.json"
        self.results_file = save_dir / "results.json"
        self.completed: set = set()
        self.results: list = []

    def load(self):
        """Load existing progress."""
        if self.progress_file.exists():
            data = json.loads(self.progress_file.read_text())
            self.completed = set(data.get("completed", []))
            print(f"  Resuming: {len(self.completed)} runs already completed")
        if self.results_file.exists():
            self.results = json.loads(self.results_file.read_text())

    def is_done(self, run_key: str) -> bool:
        return run_key in self.completed

    def mark_done(self, run_key: str, result: dict):
        self.completed.add(run_key)
        self.results.append(result)
        self._save()

    def _save(self):
        self.progress_file.write_text(json.dumps({
            "completed": sorted(self.completed),
            "last_updated": datetime.now().isoformat(),
        }, indent=2))
        self.results_file.write_text(json.dumps(self.results, indent=2))


# ── Test query evaluation ──────────────────────────────────────────────────

async def run_test_queries(
    patterns: list,
    budget_usd: float,
    tracker: ProgressTracker,
):
    """Run patterns against all 5 test queries."""
    from deep_research.evaluation.runner import run_single
    from deep_research.evaluation.test_queries import get_all_queries
    from deep_research.evaluation.metrics import evaluate_report

    queries = get_all_queries()
    total = len(patterns) * len(queries)
    done = 0

    print(f"\n{'='*60}")
    print(f"PHASE 1: Test Queries ({len(patterns)} patterns × {len(queries)} queries = {total} runs)")
    print(f"{'='*60}\n")

    for pattern in patterns:
        for tq in queries:
            run_key = f"testq:{pattern}:{tq.id}"
            if tracker.is_done(run_key):
                done += 1
                continue

            done += 1
            print(f"  [{done}/{total}] {pattern} × {tq.id} ...", end=" ", flush=True)

            try:
                start = time.time()
                report = await run_single(pattern, tq.query, budget_usd)
                elapsed = time.time() - start
                report.elapsed_seconds = elapsed

                eval_result = evaluate_report(report, tq)

                result = {
                    "type": "test_query",
                    "pattern": pattern,
                    "query_id": tq.id,
                    "query": tq.query[:100],
                    "overall_score": round(eval_result.overall_score, 4),
                    "coverage": round(eval_result.coverage_score, 4),
                    "citation_count": eval_result.citation_count,
                    "unique_sources": eval_result.unique_sources,
                    "cost_usd": round(report.total_cost_usd, 4),
                    "tokens": report.total_tokens,
                    "latency_s": round(elapsed, 1),
                    "sections": len(report.sections),
                    "word_count": len(report.full_text().split()),
                }
                tracker.mark_done(run_key, result)
                print(f"score={eval_result.overall_score:.3f} "
                      f"cost=${report.total_cost_usd:.3f} "
                      f"time={elapsed:.0f}s")

                # Save the report text separately
                report_dir = tracker.save_dir / "reports" / pattern
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / f"{tq.id}.md").write_text(report.full_text())

            except Exception as e:
                print(f"ERROR: {e}")
                tracker.mark_done(run_key, {
                    "type": "test_query",
                    "pattern": pattern,
                    "query_id": tq.id,
                    "error": str(e),
                    "overall_score": 0.0,
                })


# ── Benchmark evaluation ──────────────────────────────────────────────────

async def run_benchmarks(
    patterns: list,
    budget_usd: float,
    max_queries: int,
    tracker: ProgressTracker,
    benchmark_names: list | None = None,
):
    """Run patterns against benchmark datasets."""
    configs = BENCHMARK_CONFIGS
    if benchmark_names:
        configs = [c for c in configs if c[0] in benchmark_names]

    print(f"\n{'='*60}")
    print(f"PHASE 2: Benchmark Datasets ({len(patterns)} patterns × {len(configs)} benchmarks)")
    print(f"{'='*60}\n")

    for bench_name, bench_import, default_max in configs:
        bench_max = min(max_queries, default_max) if max_queries > 0 else default_max
        print(f"\n--- {bench_name.upper()} (loading up to {bench_max} queries) ---")

        try:
            benchmark = load_benchmark_class(bench_import)
            queries = await benchmark.load(max_queries=bench_max)
            print(f"  Loaded {len(queries)} queries")
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        total = len(patterns) * len(queries)
        done = 0

        for pattern in patterns:
            for query in queries:
                run_key = f"bench:{bench_name}:{pattern}:{query.id}"
                if tracker.is_done(run_key):
                    done += 1
                    continue

                done += 1
                print(f"  [{done}/{total}] {pattern} × {query.id} ...",
                      end=" ", flush=True)

                try:
                    import importlib
                    from deep_research.benchmarks.base import PATTERN_MODULES
                    mod = importlib.import_module(PATTERN_MODULES[pattern])

                    start = time.time()
                    report = await mod.run(query.query, budget_usd=budget_usd)
                    elapsed = time.time() - start
                    report.elapsed_seconds = elapsed
                    report.pattern_name = pattern

                    bench_result = await benchmark.score(query, report)

                    result = {
                        "type": "benchmark",
                        "benchmark": bench_name,
                        "pattern": pattern,
                        "query_id": query.id,
                        "query": query.query[:100],
                        "overall_score": round(bench_result.overall_score, 4),
                        "scores": {k: round(v, 4) for k, v in bench_result.scores.items()},
                        "cost_usd": round(report.total_cost_usd, 4),
                        "tokens": report.total_tokens,
                        "latency_s": round(elapsed, 1),
                        "sections": len(report.sections),
                        "word_count": len(report.full_text().split()),
                    }
                    tracker.mark_done(run_key, result)
                    print(f"score={bench_result.overall_score:.3f} "
                          f"cost=${report.total_cost_usd:.3f} "
                          f"time={elapsed:.0f}s")

                    # Save report
                    report_dir = tracker.save_dir / "reports" / f"bench_{bench_name}" / pattern
                    report_dir.mkdir(parents=True, exist_ok=True)
                    (report_dir / f"{query.id}.md").write_text(report.full_text())

                except Exception as e:
                    print(f"ERROR: {e}")
                    tracker.mark_done(run_key, {
                        "type": "benchmark",
                        "benchmark": bench_name,
                        "pattern": pattern,
                        "query_id": query.id,
                        "error": str(e),
                        "overall_score": 0.0,
                    })


# ── Report generation ──────────────────────────────────────────────────────

def generate_summary(tracker: ProgressTracker) -> str:
    """Generate a summary report from all results."""
    results = tracker.results
    if not results:
        return "No results to report."

    lines = [
        "# Full Evaluation Results",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total runs**: {len(results)}",
        "",
    ]

    # Split by type
    test_results = [r for r in results if r.get("type") == "test_query"]
    bench_results = [r for r in results if r.get("type") == "benchmark"]
    errors = [r for r in results if r.get("error")]

    # ── Test query results ──
    if test_results:
        lines.append("\n## Test Query Results\n")

        # Summary by pattern
        by_pattern = {}
        for r in test_results:
            if r.get("error"):
                continue
            p = r["pattern"]
            by_pattern.setdefault(p, []).append(r)

        lines.append("| Pattern | Avg Score | Avg Coverage | Avg Cost | Avg Time | Queries |")
        lines.append("|---------|-----------|-------------|----------|----------|---------|")

        for p in PATTERN_ORDER:
            if p not in by_pattern:
                continue
            rs = by_pattern[p]
            avg_score = sum(r["overall_score"] for r in rs) / len(rs)
            avg_cov = sum(r.get("coverage", 0) for r in rs) / len(rs)
            avg_cost = sum(r["cost_usd"] for r in rs) / len(rs)
            avg_time = sum(r["latency_s"] for r in rs) / len(rs)
            lines.append(
                f"| {p} | {avg_score:.3f} | {avg_cov:.3f} "
                f"| ${avg_cost:.3f} | {avg_time:.0f}s | {len(rs)} |"
            )

        # Per-query detail
        lines.append("\n### Per-Query Breakdown\n")
        lines.append("| Query | Pattern | Score | Coverage | Citations | Cost | Time |")
        lines.append("|-------|---------|-------|----------|-----------|------|------|")

        for r in sorted(test_results, key=lambda x: (x.get("query_id", ""), x.get("pattern", ""))):
            if r.get("error"):
                lines.append(f"| {r.get('query_id', '?')} | {r.get('pattern', '?')} | ERROR | | | | |")
            else:
                lines.append(
                    f"| {r['query_id']} | {r['pattern']} | {r['overall_score']:.3f} "
                    f"| {r.get('coverage', 0):.3f} | {r.get('citation_count', 0)} "
                    f"| ${r['cost_usd']:.3f} | {r['latency_s']:.0f}s |"
                )

    # ── Benchmark results ──
    if bench_results:
        lines.append("\n## Benchmark Results\n")

        by_bench = {}
        for r in bench_results:
            b = r.get("benchmark", "unknown")
            by_bench.setdefault(b, []).append(r)

        for bench_name, brs in sorted(by_bench.items()):
            lines.append(f"\n### {bench_name.upper()}\n")

            by_pattern = {}
            for r in brs:
                if r.get("error"):
                    continue
                p = r["pattern"]
                by_pattern.setdefault(p, []).append(r)

            if by_pattern:
                lines.append("| Pattern | Avg Score | Avg Cost | Avg Time | Queries |")
                lines.append("|---------|-----------|----------|----------|---------|")

                for p in PATTERN_ORDER:
                    if p not in by_pattern:
                        continue
                    rs = by_pattern[p]
                    avg_score = sum(r["overall_score"] for r in rs) / len(rs)
                    avg_cost = sum(r["cost_usd"] for r in rs) / len(rs)
                    avg_time = sum(r["latency_s"] for r in rs) / len(rs)
                    lines.append(
                        f"| {p} | {avg_score:.3f} | ${avg_cost:.3f} "
                        f"| {avg_time:.0f}s | {len(rs)} |"
                    )

                # Score dimensions
                all_dims = set()
                for r in brs:
                    all_dims.update(r.get("scores", {}).keys())
                if all_dims:
                    dim_list = sorted(all_dims)
                    lines.append(f"\n**Score Dimensions:**\n")
                    header = "| Pattern | " + " | ".join(dim_list) + " |"
                    sep = "|---------|" + "|".join("------" for _ in dim_list) + "|"
                    lines.append(header)
                    lines.append(sep)

                    for p in PATTERN_ORDER:
                        if p not in by_pattern:
                            continue
                        rs = by_pattern[p]
                        cells = []
                        for dim in dim_list:
                            vals = [r.get("scores", {}).get(dim, 0) for r in rs]
                            avg = sum(vals) / len(vals) if vals else 0
                            cells.append(f"{avg:.3f}")
                        lines.append(f"| {p} | " + " | ".join(cells) + " |")

    # ── Errors ──
    if errors:
        lines.append(f"\n## Errors ({len(errors)} total)\n")
        for e in errors:
            lines.append(f"- {e.get('pattern', '?')} × {e.get('query_id', '?')}: {e.get('error', '?')}")

    # ── Totals ──
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_time = sum(r.get("latency_s", 0) for r in results)
    lines.append(f"\n## Totals\n")
    lines.append(f"- **Total cost**: ${total_cost:.2f}")
    lines.append(f"- **Total tokens**: {total_tokens:,}")
    lines.append(f"- **Total time**: {total_time/60:.1f} minutes")
    lines.append(f"- **Successful runs**: {len(results) - len(errors)}/{len(results)}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Full evaluation: patterns × queries × benchmarks")
    parser.add_argument("--patterns", type=str, default="",
                        help="Comma-separated patterns (default: all). E.g., p0_baseline,p1_iterative_rag")
    parser.add_argument("--budget", type=float, default=2.0,
                        help="Budget per query in USD (default: 2.0)")
    parser.add_argument("--max-bench-queries", type=int, default=0,
                        help="Max queries per benchmark (0 = use defaults)")
    parser.add_argument("--test-queries-only", action="store_true",
                        help="Only run test queries, skip benchmarks")
    parser.add_argument("--benchmarks-only", action="store_true",
                        help="Only run benchmarks, skip test queries")
    parser.add_argument("--benchmarks", type=str, default="",
                        help="Comma-separated benchmark names (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved progress")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory (default: reports/full_eval_TIMESTAMP)")
    args = parser.parse_args()

    # Parse patterns
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(",")]
    else:
        patterns = PATTERN_ORDER

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / f"full_eval_{ts}"

    # Progress tracker
    tracker = ProgressTracker(out_dir)
    if args.resume:
        tracker.load()

    # Parse benchmark names
    bench_names = None
    if args.benchmarks:
        bench_names = [b.strip() for b in args.benchmarks.split(",")]

    print(f"Full Evaluation Run")
    print(f"  Patterns: {patterns}")
    print(f"  Budget: ${args.budget:.2f}/query")
    print(f"  Output: {out_dir}")
    print(f"  Resume: {args.resume}")

    start_time = time.time()

    # Phase 1: Test queries
    if not args.benchmarks_only:
        await run_test_queries(patterns, args.budget, tracker)

    # Phase 2: Benchmarks
    if not args.test_queries_only:
        await run_benchmarks(
            patterns, args.budget, args.max_bench_queries,
            tracker, bench_names,
        )

    total_elapsed = time.time() - start_time

    # Generate and save summary
    summary = generate_summary(tracker)
    summary_path = out_dir / "evaluation_summary.md"
    summary_path.write_text(summary)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"\n{summary}")


if __name__ == "__main__":
    asyncio.run(main())
