#!/usr/bin/env python3
"""CLI: Run patterns against standard benchmarks.

Usage:
    # Run all benchmarks with default settings
    python scripts/run_benchmarks.py

    # Run specific benchmark with specific patterns
    python scripts/run_benchmarks.py --benchmark draco --patterns p0,p1 --max-queries 5

    # Run with higher budget
    python scripts/run_benchmarks.py --budget 10.0 --max-queries 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BENCHMARK_REGISTRY = {
    "draco": "deep_research.benchmarks.draco.DRACOBenchmark",
    "research_qa": "deep_research.benchmarks.research_qa.ResearchQABenchmark",
    "scholar_qa": "deep_research.benchmarks.scholar_qa.ScholarQABenchmark",
    "freshwiki": "deep_research.benchmarks.freshwiki.FreshWikiBenchmark",
    "deepsearch_qa": "deep_research.benchmarks.deepsearch_qa.DeepSearchQABenchmark",
    "litqa2": "deep_research.benchmarks.litqa2.LitQA2Benchmark",
}


def load_benchmark(name: str):
    """Dynamically load a benchmark class."""
    module_path, class_name = BENCHMARK_REGISTRY[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


async def main():
    parser = argparse.ArgumentParser(description="Run research patterns against benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        default="all",
        help="Benchmark to run (default: all)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="",
        help="Comma-separated pattern names (e.g., p0,p1,p2). Default: all.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Max queries per benchmark (default: 5)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="Budget per query in USD (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory (default: reports/benchmarks/)",
    )
    args = parser.parse_args()

    # Load benchmarks
    if args.benchmark == "all":
        benchmarks = [load_benchmark(name) for name in BENCHMARK_REGISTRY]
    else:
        benchmarks = [load_benchmark(args.benchmark)]

    # Parse patterns
    patterns = None
    if args.patterns:
        patterns = [p.strip() for p in args.patterns.split(",")]

    # Run suite
    from deep_research.benchmarks.base import BenchmarkSuite

    suite = BenchmarkSuite(
        benchmarks=benchmarks,
        patterns=patterns,
        budget_usd=args.budget,
        max_queries_per_benchmark=args.max_queries,
    )

    print(f"Running benchmarks: {[b.name for b in benchmarks]}")
    print(f"Patterns: {suite.patterns}")
    print(f"Max queries: {args.max_queries} | Budget: ${args.budget:.2f}\n")

    results = await suite.run_all()

    # Print report
    report = suite.generate_report()
    print(report)

    # Save
    out_dir = Path(args.output) if args.output else None
    saved = suite.save_results(out_dir)
    print(f"\nResults saved to {saved}")


if __name__ == "__main__":
    asyncio.run(main())
