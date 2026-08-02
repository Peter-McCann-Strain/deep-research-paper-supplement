#!/usr/bin/env python3
"""Validate API connectivity for configured API-backed LLM models."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="Model name from deep_research.config.MODELS. Repeat to test multiple models.",
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also attempt local-model entries. They are skipped by default for public API checks.",
    )
    return parser


async def run_checks(args: argparse.Namespace) -> int:
    from deep_research.config import MODELS
    from deep_research.tools import CostTracker, LLMCaller

    tracker = CostTracker(budget_usd=1.0)
    llm = LLMCaller(cost_tracker=tracker)
    model_names = args.model or list(MODELS)
    unknown = sorted(set(model_names) - set(MODELS))
    if unknown:
        print(f"Unknown configured model(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if not args.include_local:
        model_names = [
            name for name in model_names if str(MODELS[name].get("deployment", "")).lower() != "local"
        ]

    print("=" * 60)
    print("API Connectivity Check")
    print("=" * 60)

    if not model_names:
        print("No API-backed models selected.")

    for name in model_names:
        print(f"\nTesting {name}...", end=" ", flush=True)
        try:
            resp = await llm.complete(
                "Hello, respond with exactly one word.",
                model=name,
                max_tokens=args.max_tokens,
            )
            print(f"OK Response: {resp.strip()}")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("Cost summary:")
    print(tracker.summary_text())
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_checks(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
