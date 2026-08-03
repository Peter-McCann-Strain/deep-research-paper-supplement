#!/usr/bin/env python3
"""CLI: Run a single research pattern against a query."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PATTERNS = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3": "deep_research.patterns.p3_meridian.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6": "deep_research.patterns.p6_reactive_interleaved.pipeline",
    "p7": "deep_research.patterns.p7_graph_decomposition.pipeline",
    "p8": "deep_research.patterns.p8_beam_search.pipeline",
    "p9": "deep_research.patterns.p9_local_baseline.pipeline",
    "p10": "deep_research.patterns.p10_deep_researcher.pipeline",
    "p11": "deep_research.patterns.p11_react.pipeline",
    "p12": "deep_research.patterns.p12_rl_trained.pipeline",
}


async def main():
    parser = argparse.ArgumentParser(description="Run a deep research pattern")
    parser.add_argument("pattern", choices=PATTERNS.keys(), help="Pattern to run")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--budget", type=float, default=2.0, help="Max cost in USD")
    parser.add_argument("--output", type=str, default="", help="Output file path")
    args = parser.parse_args()

    import importlib
    mod = importlib.import_module(PATTERNS[args.pattern])

    print(f"Running {args.pattern} with budget ${args.budget:.2f}")
    print(f"Query: {args.query}\n")

    start = time.time()
    report = await mod.run(args.query, budget_usd=args.budget)
    elapsed = time.time() - start

    report.elapsed_seconds = elapsed

    print("\n" + "=" * 60)
    print(report.full_text())
    print("=" * 60)
    print(f"\nCost: ${report.total_cost_usd:.4f} | Tokens: {report.total_tokens:,}")
    print(f"Time: {elapsed:.1f}s")

    if args.output:
        Path(args.output).write_text(report.full_text())
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
