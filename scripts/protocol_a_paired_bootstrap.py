#!/usr/bin/env python3
"""Reproducible paired-bootstrap CIs and p-values for the §6.2.1 Bing↔Tavily
intervention.

Pairs each `protocol_a_tavily_pK` cell with the corresponding `base_pK` cell
on the same query, computes the overall-score Δ = Tavily − Bing, then uses
a percentile bootstrap with N=10,000 resamples (seed=42) over the paired
differences to produce per-pattern 95% CIs and two-sided p-values.

This is the script the §6.2.1 numbers were computed with. The audit found
the analysis chain referenced these CIs but no committed script reproduced
them; this is that script.

Outputs:
  reports/protocol_a/paired_bootstrap_summary.md  — per-pattern table
  reports/protocol_a/paired_bootstrap_summary.csv — same data as csv

Usage:
  python scripts/protocol_a_paired_bootstrap.py
  python scripts/protocol_a_paired_bootstrap.py --judge gpt52 --seed 42 --n-boot 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DF_PATH = ROOT / "data" / "analysis" / "df_overall_scores.parquet"
OUT_DIR = ROOT / "reports" / "protocol_a"


def paired_bootstrap(
    deltas: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float, float]:
    """Return (mean_delta, ci_lo, ci_hi, p_two_sided)."""
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = deltas[idx].mean()
    mean_delta = float(deltas.mean())
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    # two-sided percentile p-value: fraction of bootstrap means on the wrong side of zero
    p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p_two = max(p_two, 1.0 / n_boot)  # floor by inverse N to avoid p=0
    return mean_delta, float(ci_lo), float(ci_hi), float(p_two)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="gpt52",
                        choices=["gpt52", "claude_opus", "claude_sonnet"],
                        help="Judge to evaluate against (Tavily was only run on gpt52 — "
                             "this is the documented single-judge basis for §6.2.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DF_PATH)
    df = df[df["judge"] == args.judge].copy()
    df = df[["pattern", "query_id", "overall_score"]].copy()

    rows = []
    for pat_idx in ["p0", "p1", "p3", "p4", "p5", "p8"]:
        bing = df[df["pattern"] == f"base_{pat_idx}"][["query_id", "overall_score"]]
        tav = df[df["pattern"] == f"protocol_a_tavily_{pat_idx}"][["query_id", "overall_score"]]
        if bing.empty or tav.empty:
            continue
        merged = bing.merge(tav, on="query_id", suffixes=("_bing", "_tav"))
        if merged.empty:
            continue
        deltas = (merged["overall_score_tav"] - merged["overall_score_bing"]).to_numpy()
        mean_d, ci_lo, ci_hi, p = paired_bootstrap(deltas, args.n_boot, args.seed)
        rows.append({
            "pattern_idx": pat_idx,
            "n": len(deltas),
            "mean_bing": float(merged["overall_score_bing"].mean()),
            "mean_tav": float(merged["overall_score_tav"].mean()),
            "delta_tav_minus_bing": mean_d,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_two_sided": p,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "paired_bootstrap_summary.csv", index=False)

    md = [
        f"# Protocol A — paired bootstrap summary",
        f"",
        f"Judge: `{args.judge}` (single-judge basis — Opus/Sonnet were not run on "
        f"`protocol_a_tavily_*`; see §6.2.1 caveat)",
        f"Bootstrap: N={args.n_boot:,} percentile resamples, seed={args.seed}",
        f"Pairing: per-query, `protocol_a_tavily_pK − base_pK` on `overall_score`",
        f"",
        f"| Pattern | n | mean(Bing) | mean(Tavily) | Δ (Tav−Bing) | 95% CI | p (two-sided) |",
        f"|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['pattern_idx']} | {r['n']} | {r['mean_bing']:.3f} | {r['mean_tav']:.3f} | "
            f"{r['delta_tav_minus_bing']:+.3f} | "
            f"({r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}) | "
            f"{r['p_two_sided']:.4f} |"
        )
    if rows:
        all_deltas = np.concatenate([
            (df[df["pattern"] == f"protocol_a_tavily_{r['pattern_idx']}"]
                .merge(df[df["pattern"] == f"base_{r['pattern_idx']}"], on="query_id",
                       suffixes=("_tav", "_bing")))
            .pipe(lambda d: (d["overall_score_tav"] - d["overall_score_bing"]).to_numpy())
            for r in rows
        ])
        mean_all, lo_all, hi_all, p_all = paired_bootstrap(all_deltas, args.n_boot, args.seed)
        md.append("")
        md.append(
            f"**Pooled across patterns (n={len(all_deltas)}):** mean Δ = {mean_all:+.3f}, "
            f"95% CI ({lo_all:+.3f}, {hi_all:+.3f}), p = {p_all:.4f}"
        )
    (out_dir / "paired_bootstrap_summary.md").write_text("\n".join(md) + "\n")
    print(out_dir / "paired_bootstrap_summary.md")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
