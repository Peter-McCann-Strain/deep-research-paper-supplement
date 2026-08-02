#!/usr/bin/env python3
"""CLI: Run all patterns against all test queries."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    from deep_research.evaluation.runner import run_all_evaluations
    from deep_research.evaluation.comparator import generate_comparison

    results = await run_all_evaluations()
    comparison = generate_comparison(results)
    print(comparison)

    out = Path("reports/comparison.md")
    out.write_text(comparison)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
