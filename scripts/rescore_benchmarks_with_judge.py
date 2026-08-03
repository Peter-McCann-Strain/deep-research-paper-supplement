#!/usr/bin/env python3
"""Re-score all saved benchmark reports using LLM-as-judge (GPT-5.2).

Reads benchmark markdown reports and scores them using the judge evaluator,
converting each benchmark's native rubric format into judge criteria.

Usage:
    python scripts/rescore_benchmarks_with_judge.py
    python scripts/rescore_benchmarks_with_judge.py --resume
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.evaluation.llm_judge import (
    judge_benchmark_report, DIMENSION_WEIGHTS,
)
from deep_research.benchmarks.base import BenchmarkQuery

# ── Config ─────────────────────────────────────────────────────────────────

BENCHMARK_REPORT_DIRS = [
    Path("reports/full_eval_benchmarks/reports"),
    Path("reports/full_sequential_eval/reports/benchmarks"),
]

BENCHMARK_CACHE_DIR = Path("data/benchmarks")

OUT_DIR = Path("reports/judge_evaluation_benchmarks")


# ── Rubric conversion ─────────────────────────────────────────────────────

def draco_rubric_to_criteria(rubric: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Convert DRACO sections/criteria to judge criteria tuples."""
    criteria = []
    for section_name, section_data in rubric.items():
        if not isinstance(section_data, list):
            continue
        for criterion in section_data:
            desc = criterion.get("description", criterion.get("text", ""))
            weight = criterion.get("weight", 1)
            if not desc:
                continue
            if weight < 0:
                # Critical failure criterion
                criteria.append((
                    f"The report avoids: {desc}",
                    "factual_accuracy",
                ))
            else:
                # Positive criterion — map to dimension based on content
                dim = _infer_dimension(desc)
                criteria.append((f"The report addresses: {desc}", dim))
    return criteria


def research_qa_rubric_to_criteria(rubric: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Convert ResearchQA rubric items to judge criteria tuples."""
    criteria = []
    for item in rubric.get("criteria", []):
        question = item.get("question", "")
        if not question:
            continue
        item_types = item.get("type", [])
        dim = "coverage"  # Default
        if any(t in ["factual", "accuracy"] for t in item_types):
            dim = "factual_accuracy"
        elif any(t in ["synthesis", "analysis"] for t in item_types):
            dim = "analytical_depth"
        elif any(t in ["citation"] for t in item_types):
            dim = "citation_quality"
        criteria.append((question, dim))
    return criteria


def deepsearch_qa_rubric_to_criteria(
    rubric: Dict[str, Any], reference_answer: str,
) -> List[Tuple[str, str]]:
    """Convert DeepSearchQA rubric to judge criteria tuples."""
    criteria = []
    expected = rubric.get("expected_answer", reference_answer)
    answer_type = rubric.get("answer_type", "")

    if expected:
        criteria.append((
            f"The report contains or directly addresses the expected answer: {expected[:500]}",
            "factual_accuracy",
        ))
        criteria.append((
            "The report provides the correct factual answer to the research question",
            "factual_accuracy",
        ))
    criteria.append((
        "The report provides supporting evidence and reasoning for its answer",
        "analytical_depth",
    ))
    criteria.append((
        "The report synthesizes information from multiple sources",
        "analytical_depth",
    ))
    criteria.append((
        "The report cites specific, verifiable sources",
        "citation_quality",
    ))
    if answer_type == "List":
        criteria.append((
            "The report provides a comprehensive list addressing all parts of the question",
            "coverage",
        ))
    else:
        criteria.append((
            "The report directly addresses the specific question asked",
            "coverage",
        ))
    return criteria


def litqa2_rubric_to_criteria(
    rubric: Dict[str, Any], reference_answer: str,
) -> List[Tuple[str, str]]:
    """Convert LitQA2 rubric to judge criteria tuples."""
    ideal = rubric.get("ideal", reference_answer)
    distractors = rubric.get("distractors", [])

    criteria = [
        (f"The report identifies the correct answer: {ideal}", "factual_accuracy"),
        ("The report provides scientific reasoning supporting its answer", "analytical_depth"),
        ("The report cites relevant scientific literature", "citation_quality"),
        ("The report demonstrates understanding of the underlying science", "coverage"),
    ]
    if distractors:
        dist_text = ", ".join(distractors[:3])
        criteria.append((
            f"The report correctly distinguishes the answer from plausible alternatives ({dist_text})",
            "analytical_depth",
        ))
    return criteria


def _infer_dimension(description: str) -> str:
    """Infer the evaluation dimension from criterion text."""
    desc_lower = description.lower()
    if any(w in desc_lower for w in ["cite", "source", "reference", "bibliography"]):
        return "citation_quality"
    if any(w in desc_lower for w in ["analyze", "compar", "evaluat", "synthesiz", "depth"]):
        return "analytical_depth"
    if any(w in desc_lower for w in ["organiz", "structur", "section", "format"]):
        return "organization"
    if any(w in desc_lower for w in ["accurat", "correct", "error", "factual"]):
        return "factual_accuracy"
    return "coverage"


# ── Find reports and match queries ─────────────────────────────────────────

def load_benchmark_queries(benchmark_name: str) -> Dict[str, BenchmarkQuery]:
    """Load cached benchmark queries by ID."""
    cache_map = {
        "draco": "draco/draco_queries.json",
        "deepsearch_qa": "deepsearch_qa/deepsearch_qa_queries.json",
        "research_qa": "research_qa/research_qa_queries.json",
        "litqa2": "litqa2/litqa2_queries.json",
    }
    rel_path = cache_map.get(benchmark_name)
    if not rel_path:
        return {}
    cache_file = BENCHMARK_CACHE_DIR / rel_path
    if not cache_file.exists():
        return {}
    data = json.loads(cache_file.read_text())
    return {q["id"]: BenchmarkQuery(**q) for q in data}


def find_benchmark_reports() -> List[Dict[str, Any]]:
    """Find all benchmark report files with metadata."""
    reports = []
    seen = set()

    for base_dir in BENCHMARK_REPORT_DIRS:
        if not base_dir.exists():
            continue

        # Handle two directory layouts:
        # 1. full_eval_benchmarks: bench_draco/p0_baseline/query_id.md
        # 2. full_sequential_eval: draco/p1_iterative_rag/query_id.md
        for bench_dir in base_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            bench_name_raw = bench_dir.name.replace("bench_", "")
            for pattern_dir in bench_dir.iterdir():
                if not pattern_dir.is_dir():
                    continue
                pattern = pattern_dir.name
                for md_file in pattern_dir.glob("*.md"):
                    query_id = md_file.stem
                    key = f"{bench_name_raw}:{pattern}:{query_id}"
                    if key not in seen:
                        seen.add(key)
                        reports.append({
                            "benchmark": bench_name_raw,
                            "pattern": pattern,
                            "query_id": query_id,
                            "path": md_file,
                            "key": key,
                        })

    return sorted(reports, key=lambda r: (r["benchmark"], r["pattern"], r["query_id"]))


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
    parser = argparse.ArgumentParser(description="Re-score benchmark reports with LLM judge")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load all benchmark query caches
    benchmark_queries: Dict[str, Dict[str, BenchmarkQuery]] = {}
    for bench in ["draco", "deepsearch_qa", "research_qa", "litqa2"]:
        queries = load_benchmark_queries(bench)
        if queries:
            benchmark_queries[bench] = queries
            print(f"  Loaded {len(queries)} {bench} queries")

    # Find all benchmark reports
    all_reports = find_benchmark_reports()

    # Match reports to queries
    work = []
    unmatched = []
    for report_info in all_reports:
        bench = report_info["benchmark"]
        qid = report_info["query_id"]
        queries = benchmark_queries.get(bench, {})
        if qid in queries:
            work.append((report_info, queries[qid]))
        else:
            unmatched.append(report_info["key"])

    progress = load_progress() if args.resume else {"completed": [], "results": []}
    done_keys = set(progress["completed"])
    pending = [(r, q) for r, q in work if r["key"] not in done_keys]

    print(f"\nBenchmark LLM-as-Judge Scoring (GPT-5.2)")
    print(f"  Reports found: {len(all_reports)}")
    print(f"  Matched to queries: {len(work)}")
    print(f"  Unmatched: {len(unmatched)}")
    print(f"  Already scored: {len(done_keys)}")
    print(f"  To score: {len(pending)}")
    print(f"  Output: {OUT_DIR}")
    if unmatched:
        for u in unmatched:
            print(f"    SKIP (no query): {u}")
    print()

    start = time.time()

    for i, (report_info, query) in enumerate(pending):
        bench = report_info["benchmark"]
        pattern = report_info["pattern"]
        qid = report_info["query_id"]
        key = report_info["key"]

        print(f"  [{i+1}/{len(pending)}] {bench} / {pattern} × {qid}")

        report_text = report_info["path"].read_text()
        print(f"    Report: {len(report_text.split())} words")

        # Convert rubric to criteria based on benchmark type
        if bench == "draco":
            rubric_criteria = draco_rubric_to_criteria(query.rubric)
        elif bench == "research_qa":
            rubric_criteria = research_qa_rubric_to_criteria(query.rubric)
        elif bench == "deepsearch_qa":
            rubric_criteria = deepsearch_qa_rubric_to_criteria(
                query.rubric, query.reference_answer
            )
        elif bench == "litqa2":
            rubric_criteria = litqa2_rubric_to_criteria(
                query.rubric, query.reference_answer
            )
        else:
            print(f"    SKIP: unknown benchmark {bench}")
            continue

        print(f"    Criteria: {len(rubric_criteria)} benchmark + 10 general")

        try:
            result = await judge_benchmark_report(
                query=query.query,
                query_id=qid,
                pattern_name=pattern,
                report_text=report_text,
                rubric_criteria=rubric_criteria,
            )

            for dim_name, dim in sorted(result.dimensions.items()):
                print(f"    {dim_name}: {dim.score:.0%} ({dim.criteria_met}/{dim.criteria_total})")
            print(f"    OVERALL: {result.overall_score:.3f}  ({result.total_tokens} tokens, {result.latency_seconds:.1f}s)")

            # Save result with benchmark info
            result_dict = result.to_dict()
            result_dict["benchmark"] = bench

            progress["completed"].append(key)
            progress["results"].append(result_dict)
            save_progress(progress)

            # Save detailed verdict
            verdict_dir = OUT_DIR / "verdicts" / bench / pattern
            verdict_dir.mkdir(parents=True, exist_ok=True)
            (verdict_dir / f"{qid}.json").write_text(
                json.dumps(result_dict, indent=2)
            )

        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    results = progress["results"]

    if not results:
        print("\nNo results to report.")
        return

    report_lines = [
        "# Benchmark LLM-as-Judge Results",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Judge Model**: GPT-5.2",
        f"**Reports scored**: {len(results)}",
        f"**Time**: {elapsed/60:.1f} minutes",
        "",
    ]

    # By benchmark
    by_bench = defaultdict(list)
    for r in results:
        by_bench[r.get("benchmark", "unknown")].append(r)

    for bench_name, bench_results in sorted(by_bench.items()):
        report_lines.append(f"\n## {bench_name.upper()}\n")
        by_p = defaultdict(list)
        for r in bench_results:
            by_p[r["pattern"]].append(r)

        for pat, pat_results in sorted(by_p.items()):
            avg = sum(r["overall_score"] for r in pat_results) / len(pat_results)
            report_lines.append(f"- **{pat}**: {avg:.3f} (n={len(pat_results)})")
            for r in pat_results:
                report_lines.append(f"  - {r['query_id']}: {r['overall_score']:.3f}")

    report_text = "\n".join(report_lines)
    (OUT_DIR / "benchmark_judge_report.md").write_text(report_text)

    print(f"\n{'='*70}")
    print(f"  DONE — {len(results)} benchmark reports scored in {elapsed/60:.1f} minutes")
    print(f"  Report: {OUT_DIR / 'benchmark_judge_report.md'}")
    print(f"{'='*70}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    asyncio.run(main())
