#!/usr/bin/env python3
"""Re-score all saved research reports using LLM-as-judge (GPT-5.2).

Reads the saved markdown reports from evaluation runs and scores them
using the new LLM-as-judge evaluator. No need to re-run the expensive
pattern pipelines — just re-evaluate the outputs.

Usage:
    python scripts/rescore_with_judge.py
    python scripts/rescore_with_judge.py --resume          # Skip already-scored reports
    python scripts/rescore_with_judge.py --patterns p0,p1  # Only specific patterns
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.evaluation.llm_judge import judge_report, DIMENSION_WEIGHTS
from deep_research.evaluation.test_queries import get_all_queries

# ── Config ─────────────────────────────────────────────────────────────────

PATTERNS = [
    "p0_baseline", "p1_iterative_rag", "p2_supervisor_parallel",
    "p3_meridian", "p4_perspective_storm", "p5_hierarchical_wd",
    "p6_reactive_interleaved",
]

# Where to find saved reports (search multiple directories)
REPORT_DIRS = [
    Path("reports/full_sequential_eval/reports/test_queries"),
    Path("reports/full_eval_20260309_155827/reports"),
    Path("reports/full_eval_20260309_p2p3/reports"),
    Path("reports/full_eval_20260309_p4p5/reports"),
    Path("reports/full_eval_benchmarks/reports"),
]

OUT_DIR = Path("reports/judge_evaluation")


# ── Find reports ───────────────────────────────────────────────────────────

def find_all_reports() -> dict[str, dict[str, Path]]:
    """Find all saved report files, organized by pattern -> query_id -> path.

    Returns dict like: {"p0_baseline": {"q1_bert_vs_gpt": Path(...)}}
    Later directories override earlier ones (sequential > partial runs).
    """
    reports: dict[str, dict[str, Path]] = defaultdict(dict)

    for base_dir in REPORT_DIRS:
        if not base_dir.exists():
            continue
        for pattern_dir in base_dir.iterdir():
            if not pattern_dir.is_dir():
                continue
            pattern = pattern_dir.name
            for md_file in pattern_dir.glob("*.md"):
                query_id = md_file.stem
                reports[pattern][query_id] = md_file

    return dict(reports)


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


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Re-score reports with LLM judge")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patterns", type=str, default="")
    args = parser.parse_args()

    patterns_filter = [p.strip() for p in args.patterns.split(",") if p.strip()] if args.patterns else None

    # Load test queries for rubrics
    test_queries = {q.id: q for q in get_all_queries()}

    # Find all saved reports
    all_reports = find_all_reports()

    # Build work list
    work = []
    for pattern in PATTERNS:
        if patterns_filter and pattern not in patterns_filter:
            continue
        if pattern not in all_reports:
            continue
        for query_id, report_path in sorted(all_reports[pattern].items()):
            if query_id in test_queries:  # Only score reports we have rubrics for
                work.append((pattern, query_id, report_path))

    progress = load_progress() if args.resume else {"completed": [], "results": []}
    done_keys = set(progress["completed"])

    pending = [(p, q, r) for p, q, r in work if f"{p}:{q}" not in done_keys]

    print(f"LLM-as-Judge Re-Scoring (GPT-5.2)")
    print(f"  Reports found: {sum(len(v) for v in all_reports.values())}")
    print(f"  With rubrics: {len(work)}")
    print(f"  Already scored: {len(done_keys)}")
    print(f"  To score: {len(pending)}")
    print(f"  Output: {OUT_DIR}")
    print()

    start = time.time()

    for i, (pattern, query_id, report_path) in enumerate(pending):
        key = f"{pattern}:{query_id}"
        tq = test_queries[query_id]

        print(f"  [{i+1}/{len(pending)}] {pattern} × {query_id}")

        report_text = report_path.read_text()
        print(f"    Report: {len(report_text.split())} words from {report_path}")

        try:
            result = await judge_report(
                query=tq.query,
                query_id=query_id,
                pattern_name=pattern,
                report_text=report_text,
                expected_elements=tq.expected_elements,
            )

            # Print dimension breakdown
            for dim_name, dim in sorted(result.dimensions.items()):
                print(f"    {dim_name}: {dim.score:.0%} ({dim.criteria_met}/{dim.criteria_total})")
            print(f"    OVERALL: {result.overall_score:.3f}  ({result.total_tokens} tokens, {result.latency_seconds:.1f}s)")

            # Save result
            progress["completed"].append(key)
            progress["results"].append(result.to_dict())
            save_progress(progress)

            # Save detailed verdict file
            verdict_dir = OUT_DIR / "verdicts" / pattern
            verdict_dir.mkdir(parents=True, exist_ok=True)
            (verdict_dir / f"{query_id}.json").write_text(
                json.dumps(result.to_dict(), indent=2)
            )

        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Generate summary report ──────────────────────────────────────────
    elapsed = time.time() - start
    results = progress["results"]

    if not results:
        print("\nNo results to report.")
        return

    report_lines = [
        "# LLM-as-Judge Evaluation Results",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Judge Model**: GPT-5.2",
        f"**Runs**: {len(results)}",
        f"**Time**: {elapsed/60:.1f} minutes",
        "",
    ]

    # Summary table
    by_p = defaultdict(list)
    for r in results:
        by_p[r["pattern"]].append(r)

    report_lines.append("## Pattern Comparison\n")
    dim_names = sorted(DIMENSION_WEIGHTS.keys())
    header = "| Pattern | Overall | " + " | ".join(d.replace("_", " ").title()[:12] for d in dim_names) + " | N |"
    sep = "|---|---|" + "|".join("---" for _ in dim_names) + "|---|"
    report_lines.append(header)
    report_lines.append(sep)

    for pat in PATTERNS:
        rs = by_p.get(pat, [])
        if not rs:
            continue
        avg_overall = sum(r["overall_score"] for r in rs) / len(rs)
        cells = [f"{pat}", f"{avg_overall:.3f}"]
        for dim in dim_names:
            vals = [r["dimensions"].get(dim, {}).get("score", 0) for r in rs]
            avg = sum(vals) / len(vals) if vals else 0
            cells.append(f"{avg:.2f}")
        cells.append(str(len(rs)))
        report_lines.append("| " + " | ".join(cells) + " |")

    # Per-query detail
    report_lines.append("\n## Per-Query Results\n")
    report_lines.append("| Query | Pattern | Overall | Coverage | Factual | Depth | Citations | Org |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["query_id"], PATTERNS.index(x["pattern"]) if x["pattern"] in PATTERNS else 99)):
        d = r["dimensions"]
        report_lines.append(
            f"| {r['query_id']} | {r['pattern']} "
            f"| {r['overall_score']:.3f} "
            f"| {d.get('coverage', {}).get('score', 0):.2f} "
            f"| {d.get('factual_accuracy', {}).get('score', 0):.2f} "
            f"| {d.get('analytical_depth', {}).get('score', 0):.2f} "
            f"| {d.get('citation_quality', {}).get('score', 0):.2f} "
            f"| {d.get('organization', {}).get('score', 0):.2f} |"
        )

    # Biggest wins/losses
    report_lines.append("\n## Notable Results\n")
    sorted_by_score = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    report_lines.append("**Top 5:**")
    for r in sorted_by_score[:5]:
        report_lines.append(f"- {r['pattern']} × {r['query_id']}: {r['overall_score']:.3f}")
    report_lines.append("\n**Bottom 5:**")
    for r in sorted_by_score[-5:]:
        report_lines.append(f"- {r['pattern']} × {r['query_id']}: {r['overall_score']:.3f}")

    report_text = "\n".join(report_lines)
    (OUT_DIR / "judge_report.md").write_text(report_text)

    print(f"\n{'='*70}")
    print(f"  DONE — {len(results)} reports scored in {elapsed/60:.1f} minutes")
    print(f"  Report: {OUT_DIR / 'judge_report.md'}")
    print(f"{'='*70}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    asyncio.run(main())
