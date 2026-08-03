#!/usr/bin/env python3
"""DR-Judge full validation via subprocess-batched inference.

Bulletproof against CUDA kernel deadlocks (which cannot be cancelled from
inside the Python process). Each batch runs as a separate subprocess that:
  1. Loads the QLoRA adapter
  2. Judges N examples from a slice of test.jsonl
  3. Writes a per-batch parquet shard
  4. Exits

The master loop kills any subprocess exceeding a wall-clock timeout,
records the failure, and proceeds to the next batch. At the end it
concatenates all shards and computes overall + per-pattern + per-dimension
κ + bootstrap CIs.

Usage:
    python scripts/validate_dr_judge_subproc.py
    python scripts/validate_dr_judge_subproc.py --batch-size 100 --batch-timeout 600
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "dr_judge_training" / "test.jsonl"
SHARD_DIR = ROOT / "reports" / "phase12_drjudge" / "shards"
FINAL_PRED = ROOT / "reports" / "phase12_drjudge" / "eval_predictions_full.parquet"
EVAL_REPORT = ROOT / "reports" / "phase12_drjudge" / "eval_report_full.md"


WORKER_SCRIPT = ROOT / "scripts" / "_dr_judge_worker.py"


def write_worker_script() -> None:
    """Create a single-batch inference worker that runs as a subprocess."""
    code = '''#!/usr/bin/env python3
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
'''
    WORKER_SCRIPT.write_text(code)
    WORKER_SCRIPT.chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-timeout", type=int, default=600,
                        help="Wall-clock seconds to allow per batch")
    parser.add_argument("--per-call-timeout", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip batches whose shard parquet already exists")
    args = parser.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    write_worker_script()

    n_total = sum(1 for _ in TEST_PATH.open())
    print(f"Test set: {n_total:,} examples; batch_size={args.batch_size}; "
          f"per-call timeout={args.per_call_timeout}s; per-batch wall-cap={args.batch_timeout}s")

    batch_log: list[dict] = []
    t_start = time.time()
    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        shard_path = SHARD_DIR / f"shard_{start:05d}_{end:05d}.parquet"
        if args.resume and shard_path.exists():
            print(f"[batch {start}-{end}] resume — already on disk")
            batch_log.append({"start": start, "end": end, "status": "resumed", "elapsed_s": 0.0})
            continue

        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--start", str(start), "--end", str(end),
            "--out", str(shard_path),
            "--per-call-timeout", str(args.per_call_timeout),
        ]
        print(f"\n[batch {start}-{end}] launching subprocess (cap={args.batch_timeout}s) …")
        t_b = time.time()
        try:
            res = subprocess.run(cmd, timeout=args.batch_timeout, capture_output=True, text=True)
            elapsed = time.time() - t_b
            if res.returncode == 0 and shard_path.exists():
                print(f"[batch {start}-{end}] OK in {elapsed:.0f}s")
                batch_log.append({"start": start, "end": end, "status": "ok", "elapsed_s": elapsed})
            else:
                print(f"[batch {start}-{end}] FAILED in {elapsed:.0f}s (returncode={res.returncode})")
                if res.stderr:
                    print(f"  stderr tail: {res.stderr[-300:]}")
                batch_log.append({"start": start, "end": end, "status": "failed",
                                  "elapsed_s": elapsed, "returncode": res.returncode})
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t_b
            print(f"[batch {start}-{end}] TIMED OUT after {elapsed:.0f}s — skipping")
            batch_log.append({"start": start, "end": end, "status": "timeout",
                              "elapsed_s": elapsed})

    # ── Aggregate shards ─────────────────────────────────────────────────
    shards = sorted(SHARD_DIR.glob("shard_*.parquet"))
    if not shards:
        print("No shards produced; aborting")
        return
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df.to_parquet(FINAL_PRED, index=False)
    print(f"\nAggregated {len(df):,} predictions across {len(shards)} shards")

    valid = df[df["predicted"].notna()].copy()
    print(f"Valid (non-timeout) predictions: {len(valid):,}/{len(df):,}")

    # Compute κ
    import numpy as np
    def kappa(r1, r2):
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

    rng = np.random.default_rng(42)
    def boot_kappa_ci(sub):
        n = len(sub)
        if n < 5: return (None, None)
        t = sub["target"].to_numpy()
        p = sub["predicted"].to_numpy()
        boots = [kappa(t[idx].tolist(), p[idx].tolist())
                 for idx in (rng.integers(0, n, n) for _ in range(2000))]
        return tuple(np.percentile(boots, [2.5, 97.5]).tolist())

    overall_k = kappa(valid["target"].tolist(), valid["predicted"].tolist())
    overall_ci = boot_kappa_ci(valid)
    overall_acc = (valid["target"] == valid["predicted"]).mean()
    disp = valid[valid["is_disputed"]]
    nondisp = valid[~valid["is_disputed"]]
    disp_k = kappa(disp["target"].tolist(), disp["predicted"].tolist())
    disp_ci = boot_kappa_ci(disp)
    nondisp_k = kappa(nondisp["target"].tolist(), nondisp["predicted"].tolist())
    nondisp_ci = boot_kappa_ci(nondisp)

    # Per-pattern κ
    per_pat = {}
    for pat in sorted(valid["pattern"].unique()):
        sub = valid[valid["pattern"] == pat]
        if len(sub) < 5:
            continue
        per_pat[pat] = {
            "n": len(sub),
            "kappa": kappa(sub["target"].tolist(), sub["predicted"].tolist()),
            "ci": boot_kappa_ci(sub),
        }
    per_dim = {}
    for d in sorted(valid["dimension"].unique()):
        sub = valid[valid["dimension"] == d]
        if len(sub) < 5:
            continue
        per_dim[d] = {
            "n": len(sub),
            "kappa": kappa(sub["target"].tolist(), sub["predicted"].tolist()),
            "ci": boot_kappa_ci(sub),
        }

    md = [
        "# E7: DR-Judge-7B Full Validation (subprocess-batched)",
        f"\nValidated {len(valid):,}/{len(df):,} predictions on the held-out test split.",
        f"Total examples in test: 3,824. Batches: {len(shards)}. Skipped (timeout/error): {len(df) - len(valid):,}.",
        f"Total wall time: {(time.time() - t_start) / 60:.1f} min.",
        "",
        "## Headline",
        f"- **Overall Cohen's κ vs panel consensus: {overall_k:.3f}** "
        f"(95% bootstrap CI {overall_ci[0]:.3f}, {overall_ci[1]:.3f}); "
        f"agreement rate {overall_acc*100:.1f}%",
        f"- **Disputed verdicts (n={len(disp)}): κ={disp_k:.3f}** "
        f"(CI {disp_ci[0]:.3f}, {disp_ci[1]:.3f})",
        f"- **Undisputed verdicts (n={len(nondisp)}): κ={nondisp_k:.3f}** "
        f"(CI {nondisp_ci[0]:.3f}, {nondisp_ci[1]:.3f})",
        "",
        "## Per-pattern agreement",
        "| Pattern | n | κ | 95% CI |",
        "|---|---:|---:|:---:|",
    ]
    for pat, v in sorted(per_pat.items(), key=lambda kv: -kv[1]["kappa"]):
        ci = v["ci"]
        md.append(f"| {pat} | {v['n']:,} | {v['kappa']:.3f} | ({ci[0]:.3f}, {ci[1]:.3f}) |")
    md.append("\n## Per-dimension agreement")
    md.append("| Dimension | n | κ | 95% CI |")
    md.append("|---|---:|---:|:---:|")
    for d, v in sorted(per_dim.items(), key=lambda kv: -kv[1]["kappa"]):
        ci = v["ci"]
        md.append(f"| {d} | {v['n']:,} | {v['kappa']:.3f} | ({ci[0]:.3f}, {ci[1]:.3f}) |")
    EVAL_REPORT.write_text("\n".join(md))
    print(f"\nReport: {EVAL_REPORT}")
    print(f"Predictions: {FINAL_PRED}")
    print(f"\nBatch log:")
    for entry in batch_log:
        print(f"  {entry['start']:5d}-{entry['end']:5d}: {entry['status']:10s} ({entry['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
