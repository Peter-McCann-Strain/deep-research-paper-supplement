#!/usr/bin/env python3
"""Run ALL patterns × ALL test sets SEQUENTIALLY. Saves after every run.

This runs ONE query at a time to avoid PTU rate limit contention.
Progress is saved to a single consolidated directory and can be resumed.

Usage:
    python scripts/run_all_sequential.py                    # Run everything
    python scripts/run_all_sequential.py --resume           # Resume from last checkpoint
    python scripts/run_all_sequential.py --patterns p0,p1   # Only specific patterns
    python scripts/run_all_sequential.py --skip-benchmarks  # Only test queries
    python scripts/run_all_sequential.py --skip-test-queries # Only benchmarks
"""

import argparse
import asyncio
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ─────────────────────────────────────────────────────────────────

PATTERNS = [
    "p0_baseline",
    "p1_iterative_rag",
    "p2_supervisor_parallel",
    "p3_meridian",
    "p4_perspective_storm",
    "p5_hierarchical_wd",
    "p6_reactive_interleaved",
]

PATTERN_MODULES = {
    "p0_baseline": "deep_research.patterns.p0_baseline.pipeline",
    "p1_iterative_rag": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2_supervisor_parallel": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3_meridian": "deep_research.patterns.p3_meridian.pipeline",
    "p4_perspective_storm": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5_hierarchical_wd": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6_reactive_interleaved": "deep_research.patterns.p6_reactive_interleaved.pipeline",
}

BENCHMARKS = {
    "draco": ("deep_research.benchmarks.draco", "DRACOBenchmark", 3),
    "research_qa": ("deep_research.benchmarks.research_qa", "ResearchQABenchmark", 3),
    "litqa2": ("deep_research.benchmarks.litqa2", "LitQA2Benchmark", 5),
    "deepsearch_qa": ("deep_research.benchmarks.deepsearch_qa", "DeepSearchQABenchmark", 3),
}

OUT_DIR = Path("reports/full_sequential_eval")


# ── Progress ───────────────────────────────────────────────────────────────

def load_progress() -> dict:
    f = OUT_DIR / "progress.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"completed": [], "results": []}


def save_progress(progress: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat()
    (OUT_DIR / "progress.json").write_text(json.dumps(progress, indent=2))
    (OUT_DIR / "results.json").write_text(json.dumps(progress["results"], indent=2))


def is_done(progress: dict, key: str) -> bool:
    return key in progress["completed"]


def mark_done(progress: dict, key: str, result: dict):
    progress["completed"].append(key)
    progress["results"].append(result)
    save_progress(progress)


# ── Run helpers ────────────────────────────────────────────────────────────

async def run_pattern(pattern: str, query: str, budget: float):
    """Run a single pattern on a single query. Returns ResearchReport."""
    # Reset the global AIMD limiter between runs to avoid stale state
    try:
        from deep_research.tools.llm_caller import reset_limiter
        reset_limiter()
    except Exception:
        pass

    mod = importlib.import_module(PATTERN_MODULES[pattern])
    start = time.time()
    report = await mod.run(query, budget_usd=budget)
    report.elapsed_seconds = time.time() - start
    report.pattern_name = pattern
    return report


# ── Phase 1: Test queries ─────────────────────────────────────────────────

async def run_test_queries(patterns: list, budget: float, progress: dict):
    from deep_research.evaluation.test_queries import get_all_queries
    from deep_research.evaluation.metrics import evaluate_report

    queries = get_all_queries()
    total = len(patterns) * len(queries)
    done = sum(1 for p in patterns for q in queries if is_done(progress, f"test:{p}:{q.id}"))

    print(f"\n{'='*70}")
    print(f"  PHASE 1: Our Test Queries  ({len(patterns)} patterns × {len(queries)} queries)")
    print(f"  Already done: {done}/{total}")
    print(f"{'='*70}")

    for pattern in patterns:
        for tq in queries:
            key = f"test:{pattern}:{tq.id}"
            if is_done(progress, key):
                continue

            done += 1
            print(f"\n  [{done}/{total}] {pattern} × {tq.id}")
            print(f"    Query: {tq.query[:80]}...")

            try:
                report = await run_pattern(pattern, tq.query, budget)
                ev = evaluate_report(report, tq)

                result = {
                    "type": "test_query",
                    "pattern": pattern,
                    "query_id": tq.id,
                    "query": tq.query[:120],
                    "difficulty": tq.difficulty,
                    "overall_score": round(ev.overall_score, 4),
                    "coverage": round(ev.coverage_score, 4),
                    "citation_count": ev.citation_count,
                    "unique_sources": ev.unique_sources,
                    "cost_usd": round(report.total_cost_usd, 4),
                    "tokens": report.total_tokens,
                    "latency_s": round(report.elapsed_seconds, 1),
                    "sections": len(report.sections),
                    "word_count": len(report.full_text().split()),
                }
                mark_done(progress, key, result)

                # Save report markdown
                rdir = OUT_DIR / "reports" / "test_queries" / pattern
                rdir.mkdir(parents=True, exist_ok=True)
                (rdir / f"{tq.id}.md").write_text(report.full_text())

                print(f"    ✓ score={ev.overall_score:.3f}  coverage={ev.coverage_score:.3f}  "
                      f"citations={ev.citation_count}  cost=${report.total_cost_usd:.3f}  "
                      f"time={report.elapsed_seconds:.0f}s")

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                # Log error but do NOT mark as completed — allows retry on resume
                progress.setdefault("errors", []).append({
                    "type": "test_query", "pattern": pattern,
                    "query_id": tq.id, "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                save_progress(progress)


# ── Phase 2: Benchmark datasets ───────────────────────────────────────────

async def run_benchmarks(patterns: list, budget: float, progress: dict,
                         bench_names: list | None = None):
    targets = bench_names or list(BENCHMARKS.keys())
    total_bench_runs = len(patterns) * sum(BENCHMARKS[b][2] for b in targets)
    done = 0

    print(f"\n{'='*70}")
    print(f"  PHASE 2: Benchmark Datasets  ({len(patterns)} patterns × {len(targets)} benchmarks)")
    print(f"{'='*70}")

    for bench_name in targets:
        mod_path, cls_name, max_q = BENCHMARKS[bench_name]
        print(f"\n  --- Loading {bench_name.upper()} ---")

        try:
            mod = importlib.import_module(mod_path)
            benchmark = getattr(mod, cls_name)()
            queries = await benchmark.load(max_queries=max_q)
            print(f"  Loaded {len(queries)} queries")
        except Exception as e:
            print(f"  ERROR loading {bench_name}: {e}")
            continue

        for pattern in patterns:
            for query in queries:
                key = f"bench:{bench_name}:{pattern}:{query.id}"
                if is_done(progress, key):
                    done += 1
                    continue

                done += 1
                print(f"\n  [{done}] {bench_name}/{pattern} × {query.id}")
                print(f"    Query: {query.query[:80]}...")

                try:
                    report = await run_pattern(pattern, query.query, budget)
                    bench_result = await benchmark.score(query, report)

                    result = {
                        "type": "benchmark",
                        "benchmark": bench_name,
                        "pattern": pattern,
                        "query_id": query.id,
                        "query": query.query[:120],
                        "domain": query.domain,
                        "overall_score": round(bench_result.overall_score, 4),
                        "scores": {k: round(v, 4) for k, v in bench_result.scores.items()},
                        "cost_usd": round(report.total_cost_usd, 4),
                        "tokens": report.total_tokens,
                        "latency_s": round(report.elapsed_seconds, 1),
                        "sections": len(report.sections),
                        "word_count": len(report.full_text().split()),
                    }
                    mark_done(progress, key, result)

                    # Save report
                    rdir = OUT_DIR / "reports" / "benchmarks" / bench_name / pattern
                    rdir.mkdir(parents=True, exist_ok=True)
                    (rdir / f"{query.id[:50]}.md").write_text(report.full_text())

                    print(f"    ✓ score={bench_result.overall_score:.3f}  "
                          f"cost=${report.total_cost_usd:.3f}  "
                          f"time={report.elapsed_seconds:.0f}s")

                except Exception as e:
                    print(f"    ✗ ERROR: {e}")
                    # Log error but do NOT mark as completed — allows retry on resume
                    progress.setdefault("errors", []).append({
                        "type": "benchmark", "benchmark": bench_name,
                        "pattern": pattern, "query_id": query.id,
                        "error": str(e), "timestamp": datetime.now().isoformat(),
                    })
                    save_progress(progress)


# ── Report generation ──────────────────────────────────────────────────────

def generate_final_report(progress: dict) -> str:
    results = progress["results"]
    ok = [r for r in results if not r.get("error")]  # All results are now successes
    err = progress.get("errors", [])

    lines = [
        "# Complete Evaluation Results",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Runs**: {len(ok)} successful, {len(err)} errors",
        "",
    ]

    # ── Test queries ──
    test_results = [r for r in ok if r.get("type") == "test_query"]
    if test_results:
        lines.append("\n## Our Test Queries (5 queries × 6 patterns)\n")

        by_p = defaultdict(list)
        for r in test_results:
            by_p[r["pattern"]].append(r)

        lines.append("| Pattern | Avg Score | Avg Coverage | Avg Citations | Avg Time | N |")
        lines.append("|---------|-----------|-------------|--------------|----------|---|")
        for p in PATTERNS:
            rs = by_p.get(p, [])
            if not rs:
                continue
            lines.append(
                f"| {p} "
                f"| {sum(r['overall_score'] for r in rs)/len(rs):.3f} "
                f"| {sum(r.get('coverage',0) for r in rs)/len(rs):.3f} "
                f"| {sum(r.get('citation_count',0) for r in rs)/len(rs):.1f} "
                f"| {sum(r.get('latency_s',0) for r in rs)/len(rs):.0f}s "
                f"| {len(rs)} |"
            )

        # Per-query detail
        lines.append("\n### Detailed Results\n")
        lines.append("| Query | Pattern | Score | Coverage | Citations | Words | Time |")
        lines.append("|-------|---------|-------|----------|-----------|-------|------|")
        for r in sorted(test_results, key=lambda x: (x["query_id"], PATTERNS.index(x["pattern"]) if x["pattern"] in PATTERNS else 99)):
            lines.append(
                f"| {r['query_id']} | {r['pattern']} "
                f"| {r['overall_score']:.3f} | {r.get('coverage',0):.3f} "
                f"| {r.get('citation_count',0)} | {r.get('word_count',0)} "
                f"| {r.get('latency_s',0):.0f}s |"
            )

    # ── Benchmarks ──
    bench_results = [r for r in ok if r.get("type") == "benchmark"]
    if bench_results:
        lines.append("\n## Benchmark Dataset Results\n")

        by_bench = defaultdict(list)
        for r in bench_results:
            by_bench[r["benchmark"]].append(r)

        for bname, brs in sorted(by_bench.items()):
            lines.append(f"\n### {bname.upper()}\n")

            by_p = defaultdict(list)
            for r in brs:
                by_p[r["pattern"]].append(r)

            lines.append("| Pattern | Avg Score | Avg Cost | Avg Time | N |")
            lines.append("|---------|-----------|----------|----------|---|")
            for p in PATTERNS:
                rs = by_p.get(p, [])
                if not rs:
                    continue
                lines.append(
                    f"| {p} "
                    f"| {sum(r['overall_score'] for r in rs)/len(rs):.3f} "
                    f"| ${sum(r.get('cost_usd',0) for r in rs)/len(rs):.3f} "
                    f"| {sum(r.get('latency_s',0) for r in rs)/len(rs):.0f}s "
                    f"| {len(rs)} |"
                )

            # Score dimensions
            all_dims = set()
            for r in brs:
                all_dims.update(r.get("scores", {}).keys())
            if all_dims:
                dim_list = sorted(all_dims)
                lines.append(f"\n**Dimensions:**\n")
                header = "| Pattern | " + " | ".join(d[:12] for d in dim_list) + " |"
                sep = "|---------|" + "|".join("------" for _ in dim_list) + "|"
                lines.append(header)
                lines.append(sep)
                for p in PATTERNS:
                    rs = by_p.get(p, [])
                    if not rs:
                        continue
                    cells = []
                    for d in dim_list:
                        vals = [r.get("scores", {}).get(d, 0) for r in rs]
                        cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
                    lines.append(f"| {p} | " + " | ".join(cells) + " |")

    # ── Errors ──
    if err:
        lines.append(f"\n## Errors ({len(err)})\n")
        for e in err:
            lines.append(f"- {e.get('pattern','?')} × {e.get('query_id','?')}: {e.get('error','?')[:80]}")

    # ── Totals ──
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_time = sum(r.get("latency_s", 0) for r in results)
    lines.append(f"\n## Resource Usage\n")
    lines.append(f"- **Total cost**: ${total_cost:.2f}")
    lines.append(f"- **Total tokens**: {total_tokens:,}")
    lines.append(f"- **Total wall time**: {total_time/3600:.1f} hours")

    return "\n".join(lines)


# ── Import existing results ───────────────────────────────────────────────

def import_existing_results(progress: dict):
    """Import results from previous partial runs."""
    import glob

    count = 0
    for d in sorted(glob.glob("reports/full_eval_*")):
        try:
            results = json.loads(open(f"{d}/results.json").read())
            for r in results:
                if r.get("error"):
                    continue
                if r.get("type") == "test_query":
                    key = f"test:{r['pattern']}:{r['query_id']}"
                elif r.get("type") == "benchmark":
                    key = f"bench:{r['benchmark']}:{r['pattern']}:{r['query_id']}"
                else:
                    continue
                if key not in progress["completed"]:
                    progress["completed"].append(key)
                    progress["results"].append(r)
                    count += 1
        except Exception:
            pass

    if count:
        save_progress(progress)
        print(f"  Imported {count} results from previous runs")


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Sequential full evaluation")
    parser.add_argument("--patterns", type=str, default="",
                        help="Comma-separated (default: all)")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--skip-test-queries", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", type=str, default="",
                        help="Comma-separated benchmark names")
    parser.add_argument("--resume", action="store_true",
                        help="Resume + import previous partial runs")
    args = parser.parse_args()

    patterns = [p.strip() for p in args.patterns.split(",")] if args.patterns else PATTERNS
    bench_names = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress() if args.resume else {"completed": [], "results": []}

    if args.resume:
        import_existing_results(progress)

    n_done = len([c for c in progress["completed"]])
    print(f"Sequential Full Evaluation")
    print(f"  Patterns: {patterns}")
    print(f"  Budget: ${args.budget:.2f}/query")
    print(f"  Output: {OUT_DIR}")
    print(f"  Already done: {n_done}")

    start = time.time()

    if not args.skip_test_queries:
        await run_test_queries(patterns, args.budget, progress)

    if not args.skip_benchmarks:
        await run_benchmarks(patterns, args.budget, progress, bench_names)

    # Final report
    report = generate_final_report(progress)
    (OUT_DIR / "evaluation_report.md").write_text(report)

    elapsed = time.time() - start
    total_runs = len(progress["results"])
    ok = len([r for r in progress["results"] if not r.get("error")])

    print(f"\n{'='*70}")
    print(f"  DONE — {ok}/{total_runs} successful in {elapsed/3600:.1f} hours")
    print(f"  Report: {OUT_DIR / 'evaluation_report.md'}")
    print(f"{'='*70}")
    print(f"\n{report}")


if __name__ == "__main__":
    asyncio.run(main())
