#!/usr/bin/env python3
"""Select evaluation queries and save a reproducible manifest.

Creates a QueryRegistry, loads queries from all benchmark sources with
stratified sampling, prints summary statistics, and saves the manifest
to ``data/eval_queries_v2.json``.

Usage:
    python scripts/select_eval_queries.py
    python scripts/select_eval_queries.py --output data/my_manifest.json
    python scripts/select_eval_queries.py --draco 20 --deepsearch 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.evaluation.query_registry import QueryRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select evaluation queries and save manifest"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_queries_v2.json"),
        help="Output manifest path (default: data/eval_queries_v2.json)",
    )
    parser.add_argument("--custom", type=int, default=5)
    parser.add_argument("--draco", type=int, default=40)
    parser.add_argument("--deepsearch", type=int, default=20)
    parser.add_argument("--research-qa", type=int, default=15)
    parser.add_argument("--litqa2", type=int, default=10)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing benchmark caches",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Evaluation Query Selection (V2)")
    print("=" * 70)

    registry = QueryRegistry(data_dir=args.data_dir)
    queries = registry.load_all(
        custom=args.custom,
        draco=args.draco,
        deepsearch=args.deepsearch,
        research_qa=args.research_qa,
        litqa2=args.litqa2,
    )

    summary = registry.summary
    print(f"\nTotal queries: {summary['total']}")

    print("\nBy source:")
    for source, count in sorted(summary["by_source"].items()):
        print(f"  {source:20s} {count:4d}")

    print(f"\nBy difficulty:")
    for diff, count in sorted(summary["by_difficulty"].items()):
        print(f"  {diff:20s} {count:4d}")

    print(f"\nBy domain ({len(summary['by_domain'])} unique domains):")
    for domain, count in sorted(
        summary["by_domain"].items(), key=lambda x: -x[1]
    )[:20]:
        print(f"  {domain:40s} {count:4d}")
    if len(summary["by_domain"]) > 20:
        print(f"  ... and {len(summary['by_domain']) - 20} more domains")

    # Show rubric statistics
    total_criteria = sum(q.rubric.total_criteria for q in queries)
    avg_criteria = total_criteria / len(queries) if queries else 0
    print(f"\nRubric statistics:")
    print(f"  Total criteria across all queries: {total_criteria}")
    print(f"  Average criteria per query: {avg_criteria:.1f}")

    # Save manifest
    registry.save_manifest(args.output)
    print(f"\nManifest saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
