#!/usr/bin/env python3
"""Validate API connectivity for all LLM models."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import MODELS
from deep_research.tools import LLMCaller, CostTracker


async def main():
    tracker = CostTracker(budget_usd=1.0)
    llm = LLMCaller(cost_tracker=tracker)

    print("=" * 60)
    print("API Connectivity Check")
    print("=" * 60)

    # Test each LLM model
    for name in MODELS:
        print(f"\n→ Testing {name}...", end=" ", flush=True)
        try:
            resp = await llm.complete(
                "Hello, respond with exactly one word.",
                model=name,
                max_tokens=10,
            )
            print(f"✓ Response: {resp.strip()}")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Cost summary:")
    print(tracker.summary_text())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
