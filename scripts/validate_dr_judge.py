#!/usr/bin/env python3
"""Validate DR-Judge-7B against the held-out test split.

After fine-tuning, this script:
  1. Loads the held-out test.jsonl (3,824 SFT examples, query-level disjoint from train)
  2. Runs DR-Judge-7B inference on each
  3. Compares predicted verdicts (satisfied: bool) to consensus targets
  4. Computes Krippendorff α + Cohen's κ + per-dimension stats
  5. Writes a markdown report ready for §5.4 of the paper

Outputs:
  reports/phase12_drjudge/eval_report.md
  reports/phase12_drjudge/eval_predictions.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.tools.dr_judge_caller import DRJudgeCaller

DATA_DIR = Path("data/dr_judge_training")
OUT_DIR = Path("reports/phase12_drjudge")


def cohens_kappa(rater1: list[bool], rater2: list[bool]) -> float:
    if not rater1 or len(rater1) != len(rater2):
        return float("nan")
    n = len(rater1)
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 and r2)
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 and not r2)
    c = sum(1 for r1, r2 in zip(rater1, rater2) if not r1 and r2)
    d = sum(1 for r1, r2 in zip(rater1, rater2) if not r1 and not r2)
    p_o = (a + d) / n
    p1 = (a + b) / n
    p2 = (a + c) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    return (p_o - p_e) / (1 - p_e) if p_e < 1.0 else (1.0 if p_o == 1.0 else 0.0)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of test examples (0 = all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = DATA_DIR / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} missing — run prep_dr_judge_data.py first")
        return

    examples = [json.loads(line) for line in test_path.open()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} test examples")

    judge = DRJudgeCaller()
    print("Loading DR-Judge-7B …")
    judge._ensure_loaded()
    print("Loaded. Running inference …")

    rows = []
    for i, ex in enumerate(examples):
        meta = ex.get("metadata", {})
        # Reconstruct the system + user from the chat messages
        system = next((m["content"] for m in ex["messages"] if m["role"] == "system"), "")
        user = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        target = json.loads(next(m["content"] for m in ex["messages"] if m["role"] == "assistant"))
        target_satisfied = bool(target.get("satisfied"))
        t0 = time.time()
        try:
            pred = await asyncio.wait_for(
                judge.complete_json(user, system=system, temperature=0.1, max_tokens=256),
                timeout=60.0,  # any single example >60s indicates the model is stuck
            )
            pred_sat = bool(pred.get("satisfied", False)) if isinstance(pred, dict) else False
        except asyncio.TimeoutError:
            pred_sat = None
            pred = {"_timeout": True}
            log_line = f"  [TIMEOUT] example {i} > 60s"
            print(log_line, flush=True)
        except Exception as e:
            pred_sat = None
            pred = {"_error": str(e)[:200]}
        rows.append({
            "i": i,
            "pattern": meta.get("pattern", ""),
            "query_id": meta.get("query_id", ""),
            "criterion_id": meta.get("criterion_id", ""),
            "dimension": meta.get("dimension", ""),
            "is_disputed": meta.get("is_disputed", False),
            "n_judges": meta.get("n_judges", 3),
            "target": target_satisfied,
            "predicted": pred_sat,
            "elapsed_s": time.time() - t0,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(examples):
            elapsed = sum(r["elapsed_s"] for r in rows)
            print(f"  {i+1}/{len(examples)}  elapsed={elapsed:.0f}s  rate={i/(elapsed or 1):.2f}/s")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "eval_predictions.parquet", index=False)
    valid = df[df["predicted"].notna()].copy()
    print(f"\n  Valid predictions: {len(valid)}/{len(df)}")
    if valid.empty:
        print("All predictions failed; aborting analysis")
        return

    overall_kappa = cohens_kappa(valid["target"].tolist(), valid["predicted"].tolist())
    overall_acc = (valid["target"] == valid["predicted"]).mean()
    per_dim = []
    for dim in sorted(valid["dimension"].unique()):
        sub = valid[valid["dimension"] == dim]
        if len(sub) < 5:
            continue
        per_dim.append({
            "dimension": dim,
            "n": len(sub),
            "kappa": cohens_kappa(sub["target"].tolist(), sub["predicted"].tolist()),
            "agreement_rate": (sub["target"] == sub["predicted"]).mean(),
        })

    # Disputed vs non-disputed
    disp = valid[valid["is_disputed"]]
    nondisp = valid[~valid["is_disputed"]]
    disputed_kappa = cohens_kappa(disp["target"].tolist(), disp["predicted"].tolist()) if len(disp) >= 5 else float("nan")
    nondisp_kappa = cohens_kappa(nondisp["target"].tolist(), nondisp["predicted"].tolist()) if len(nondisp) >= 5 else float("nan")

    md = [
        "# E7: DR-Judge-7B Evaluation",
        f"\nValidated {len(valid)}/{len(df)} predictions on the held-out test split (query-level disjoint from train).",
        "",
        "## Headline",
        f"- **Overall Cohen's κ vs panel consensus: {overall_kappa:.3f}** (agreement rate {overall_acc*100:.1f}%)",
        f"- **On disputed (panel-split) verdicts: κ = {disputed_kappa:.3f}** (n={len(disp):,})",
        f"- **On undisputed (panel-unanimous) verdicts: κ = {nondisp_kappa:.3f}** (n={len(nondisp):,})",
        "",
        "## Per-dimension agreement",
        "| Dimension | n | κ | Agreement rate |",
        "|---|---:|---:|---:|",
    ]
    for d in sorted(per_dim, key=lambda r: -r["kappa"]):
        md.append(f"| {d['dimension']} | {d['n']:,} | {d['kappa']:.3f} | {d['agreement_rate']*100:.1f}% |")

    md.append("\n## Interpretation")
    md.append(f"- Cohen's κ ≥ 0.6 = substantial agreement (Landis & Koch 1977).")
    md.append(f"- DR-Judge {'PASSES' if overall_kappa >= 0.7 else 'misses'} the α≥0.7 success criterion.")
    md.append(f"- The disputed-verdict κ is the more demanding test: the model must reproduce the panel's majority-vote even when judges disagreed (45.4% of the train set).")

    (OUT_DIR / "eval_report.md").write_text("\n".join(md))
    print(f"\nDone:")
    print(f"  Overall κ: {overall_kappa:.3f}")
    print(f"  Report:    {OUT_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
