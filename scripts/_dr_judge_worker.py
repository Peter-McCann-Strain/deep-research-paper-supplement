#!/usr/bin/env python3
"""Single-batch DR-Judge inference worker — runs in its own process so a
CUDA hang only loses one batch instead of the whole eval."""
import argparse, asyncio, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deep_research.tools.dr_judge_caller import DRJudgeCaller


def cohens_kappa(r1, r2):
    n = len(r1)
    if n == 0:
        return float("nan")
    a = sum(1 for x, y in zip(r1, r2) if x and y)
    b = sum(1 for x, y in zip(r1, r2) if x and not y)
    c = sum(1 for x, y in zip(r1, r2) if not x and y)
    d = sum(1 for x, y in zip(r1, r2) if not x and not y)
    p_o = (a + d) / n
    p1 = (a + b) / n
    p2 = (a + c) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else (1 if p_o == 1 else 0)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--per-call-timeout", type=float, default=30.0)
    args = parser.parse_args()

    test_path = Path("data/dr_judge_training/test.jsonl")
    examples = [json.loads(l) for l in test_path.open()]
    examples = examples[args.start:args.end]
    print(f"[worker] loading adapter; will judge examples {args.start}-{args.end} ({len(examples)} items)", flush=True)

    judge = DRJudgeCaller()
    judge._ensure_loaded()
    print(f"[worker] loaded; starting inference", flush=True)

    rows = []
    for i, ex in enumerate(examples):
        meta = ex.get("metadata", {})
        system = next((m["content"] for m in ex["messages"] if m["role"] == "system"), "")
        user = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        target = json.loads(next(m["content"] for m in ex["messages"] if m["role"] == "assistant"))
        target_sat = bool(target.get("satisfied"))
        t0 = time.time()
        pred_sat = None
        try:
            pred = await asyncio.wait_for(
                judge.complete_json(user, system=system, temperature=0.1, max_tokens=200),
                timeout=args.per_call_timeout,
            )
            if isinstance(pred, dict):
                pred_sat = bool(pred.get("satisfied", False))
        except (asyncio.TimeoutError, Exception):
            pred_sat = None
        rows.append({
            "global_idx": args.start + i,
            "pattern": meta.get("pattern", ""),
            "query_id": meta.get("query_id", ""),
            "criterion_id": meta.get("criterion_id", ""),
            "dimension": meta.get("dimension", ""),
            "is_disputed": meta.get("is_disputed", False),
            "n_judges": meta.get("n_judges", 3),
            "target": target_sat,
            "predicted": pred_sat,
            "elapsed_s": time.time() - t0,
        })
        if (i + 1) % 25 == 0:
            print(f"[worker] {i+1}/{len(examples)} done", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"[worker] wrote {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
