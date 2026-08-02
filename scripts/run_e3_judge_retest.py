#!/usr/bin/env python3
"""E3: Judge non-determinism re-test.

Re-judges a stratified 30-report subset twice with the same GPT-5.2 judge
configuration. Measures within-judge non-determinism via:
  - Per-criterion Cohen's κ between v1 and v2 verdicts
  - Per-dimension Pearson r between v1 and v2 scores
  - Per-pattern correlation aggregates

Outputs:
  reports/phase9_judge_retest/v1.parquet (verdicts from re-judge run 1)
  reports/phase9_judge_retest/v2.parquet (verdicts from re-judge run 2)
  reports/phase9_judge_retest/agreement.csv (per-criterion κ)
  reports/phase9_judge_retest/summary.md (publishable narrative)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the existing GPT-5.2 judge machinery
from scripts.run_gpt52_judge import (
    evaluate_one,
    load_queries,
    JUDGE_OUT,
    RESULTS_BASE,
)

OUT_DIR = Path("reports/phase9_judge_retest")
PATTERNS = ["base_p0", "base_p1", "base_p4", "base_p7", "base_p9", "base_p10"]
N_PER_PATTERN = 5  # 6 patterns × 5 = 30 reports
CONCURRENCY = 5


def stratified_sample(queries: dict[str, dict], rng_seed: int = 42) -> list[tuple[str, str]]:
    """Pick 5 query IDs per pattern that have completed reports.

    Returns list of (pattern, query_id).
    """
    import random
    rng = random.Random(rng_seed)
    cells = []
    for pattern in PATTERNS:
        exp_dir = RESULTS_BASE / pattern
        if not exp_dir.exists():
            print(f"  WARN: {exp_dir} missing, skipping {pattern}")
            continue
        files = sorted(exp_dir.glob("*.md"))
        # Filter to those whose query_id is in the manifest
        qids = [f.stem for f in files if f.stem in queries]
        if len(qids) < N_PER_PATTERN:
            print(f"  WARN: only {len(qids)} reports for {pattern}; using all")
            picks = qids
        else:
            picks = rng.sample(qids, N_PER_PATTERN)
        cells.extend([(pattern, q) for q in picks])
    return cells


async def re_judge_run(
    cells: list[tuple[str, str]],
    queries: dict[str, dict],
    run_label: str,
) -> list[dict]:
    """Re-evaluate every cell once. Returns flat list of verdict-level dicts."""
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"\n=== Re-judge run: {run_label} ({len(cells)} reports) ===", flush=True)
    results: list[dict] = []

    async def _one(pattern: str, query_id: str):
        try:
            report_path = RESULTS_BASE / pattern / f"{query_id}.md"
            report_text = report_path.read_text()
            query = queries[query_id]
            t0 = time.time()
            res = await evaluate_one(sem, pattern, query_id, query, report_text)
            elapsed = time.time() - t0
            print(f"  [{run_label}] {pattern}/{query_id[:30]} OK overall={res['overall_score']:.3f} "
                  f"({len(res['verdicts'])} verdicts, {elapsed:.0f}s)", flush=True)
            # Flatten verdicts
            for i, v in enumerate(res["verdicts"]):
                results.append({
                    "run_label": run_label,
                    "pattern": pattern,
                    "query_id": query_id,
                    "criterion_idx": i,
                    "criterion": v["criterion"],
                    "dimension": v["dimension"],
                    "satisfied": v["satisfied"],
                    "judge_overall": res["overall_score"],
                })
            # Also keep per-dimension scores for correlation analysis
            for dim, stats in res["dimensions"].items():
                results.append({
                    "run_label": run_label,
                    "pattern": pattern,
                    "query_id": query_id,
                    "criterion_idx": -1,  # marker for dim-level score
                    "criterion": f"__dim__{dim}",
                    "dimension": dim,
                    "satisfied": None,
                    "dim_score": stats["score"],
                    "judge_overall": res["overall_score"],
                })
        except Exception as e:
            print(f"  [{run_label}] {pattern}/{query_id[:30]} FAILED: {e}", flush=True)

    await asyncio.gather(*[_one(p, q) for (p, q) in cells])
    return results


def cohens_kappa(rater1: list[bool], rater2: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    if not rater1 or len(rater1) != len(rater2):
        return float("nan")
    n = len(rater1)
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 and r2)
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 and not r2)
    c = sum(1 for r1, r2 in zip(rater1, rater2) if not r1 and r2)
    d = sum(1 for r1, r2 in zip(rater1, rater2) if not r1 and not r2)
    p_o = (a + d) / n
    p1_y = (a + b) / n
    p2_y = (a + c) / n
    p_e = p1_y * p2_y + (1 - p1_y) * (1 - p2_y)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def analyse(v1_df: pd.DataFrame, v2_df: pd.DataFrame) -> dict:
    """Compute κ and correlation summaries between v1 and v2."""
    # Verdict-level: join on (pattern, query_id, criterion_idx) for criterion-level
    verdicts1 = v1_df[v1_df["criterion_idx"] >= 0].copy()
    verdicts2 = v2_df[v2_df["criterion_idx"] >= 0].copy()
    merged = verdicts1.merge(
        verdicts2, on=["pattern", "query_id", "criterion_idx"], suffixes=("_v1", "_v2")
    )
    print(f"\n  Joined {len(merged)} criterion-level pairs across {merged[['pattern','query_id']].drop_duplicates().shape[0]} reports")

    # Overall κ
    overall_k = cohens_kappa(merged["satisfied_v1"].tolist(), merged["satisfied_v2"].tolist())

    # Per-dimension κ
    per_dim = {}
    for dim in sorted(merged["dimension_v1"].unique()):
        sub = merged[merged["dimension_v1"] == dim]
        if len(sub) < 5:
            continue
        per_dim[dim] = {
            "n": len(sub),
            "kappa": cohens_kappa(sub["satisfied_v1"].tolist(), sub["satisfied_v2"].tolist()),
            "agreement_rate": (sub["satisfied_v1"] == sub["satisfied_v2"]).mean(),
        }

    # Per-pattern κ
    per_pattern = {}
    for pat in sorted(merged["pattern"].unique()):
        sub = merged[merged["pattern"] == pat]
        if len(sub) < 5:
            continue
        per_pattern[pat] = {
            "n": len(sub),
            "kappa": cohens_kappa(sub["satisfied_v1"].tolist(), sub["satisfied_v2"].tolist()),
        }

    # Dim-level Pearson r between v1 and v2 scores per (pattern, query_id, dim)
    dim1 = v1_df[v1_df["criterion_idx"] == -1].copy()
    dim2 = v2_df[v2_df["criterion_idx"] == -1].copy()
    dim_merged = dim1.merge(dim2, on=["pattern", "query_id", "dimension"], suffixes=("_v1", "_v2"))
    dim_corrs = {}
    for dim in sorted(dim_merged["dimension"].unique()):
        sub = dim_merged[dim_merged["dimension"] == dim]
        if len(sub) < 5:
            continue
        r = sub["dim_score_v1"].corr(sub["dim_score_v2"])
        dim_corrs[dim] = {"n": len(sub), "pearson_r": float(r) if pd.notna(r) else None,
                          "mean_abs_delta": float((sub["dim_score_v1"] - sub["dim_score_v2"]).abs().mean())}

    # Overall score Pearson r at the report level
    rep1 = v1_df.groupby(["pattern", "query_id"])["judge_overall"].first().rename("overall_v1")
    rep2 = v2_df.groupby(["pattern", "query_id"])["judge_overall"].first().rename("overall_v2")
    rep_merged = pd.concat([rep1, rep2], axis=1).dropna()
    overall_r = rep_merged["overall_v1"].corr(rep_merged["overall_v2"])
    overall_mean_abs_delta = (rep_merged["overall_v1"] - rep_merged["overall_v2"]).abs().mean()

    return {
        "n_reports": len(rep_merged),
        "n_criterion_pairs": len(merged),
        "overall_kappa": overall_k,
        "agreement_rate": (merged["satisfied_v1"] == merged["satisfied_v2"]).mean(),
        "per_dimension_kappa": per_dim,
        "per_pattern_kappa": per_pattern,
        "per_dimension_correlation": dim_corrs,
        "overall_score_pearson_r": float(overall_r) if pd.notna(overall_r) else None,
        "overall_score_mean_abs_delta": float(overall_mean_abs_delta),
    }


def write_summary(stats: dict, out_path: Path):
    lines = [
        "# E3: Judge Non-Determinism Re-Test",
        f"\nGenerated {datetime.now(timezone.utc).isoformat()}",
        "\n## Sampling",
        f"- {len(PATTERNS)} patterns × {N_PER_PATTERN} reports = {stats['n_reports']} reports",
        f"- Re-judged twice with the same GPT-5.2 configuration (T=0.1)",
        f"- Total criterion-pair observations: {stats['n_criterion_pairs']:,}",
        "\n## Headline",
        f"- **Cohen's κ (overall, all criteria pooled): {stats['overall_kappa']:.3f}** "
        f"(agreement rate {stats['agreement_rate']*100:.1f}%)",
        f"- **Report-level overall score Pearson r: {stats['overall_score_pearson_r']:.3f}**",
        f"- **Report-level mean |Δ| in overall score: {stats['overall_score_mean_abs_delta']:.4f}**",
        "\n## Per-dimension Cohen's κ",
        "| Dimension | n | κ | Agreement rate |",
        "|---|---:|---:|---:|",
    ]
    for dim, v in sorted(stats["per_dimension_kappa"].items(), key=lambda kv: -kv[1]["kappa"]):
        lines.append(f"| {dim} | {v['n']:,} | {v['kappa']:.3f} | {v['agreement_rate']*100:.1f}% |")
    lines.append("\n## Per-dimension score correlation (Pearson r between v1 and v2)")
    lines.append("| Dimension | n | r | mean |Δ| |")
    lines.append("|---|---:|---:|---:|")
    for dim, v in sorted(stats["per_dimension_correlation"].items(), key=lambda kv: -(kv[1]["pearson_r"] or -1)):
        lines.append(f"| {dim} | {v['n']:,} | {v['pearson_r']:.3f} | {v['mean_abs_delta']:.4f} |")
    lines.append("\n## Per-pattern Cohen's κ")
    lines.append("| Pattern | n | κ |")
    lines.append("|---|---:|---:|")
    for pat, v in stats["per_pattern_kappa"].items():
        lines.append(f"| {pat} | {v['n']:,} | {v['kappa']:.3f} |")
    lines.append("\n## Interpretation")
    lines.append(f"- Cohen's κ ≥ 0.8 = almost-perfect; 0.6–0.8 = substantial; 0.4–0.6 = moderate (Landis & Koch 1977).")
    lines.append("- A high κ here means the existing 3-judge panel results are reproducible across re-runs of GPT-5.2.")
    out_path.write_text("\n".join(lines))


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    cells = stratified_sample(queries)
    print(f"Stratified sample: {len(cells)} (pattern, query_id) cells")
    if not cells:
        print("ERROR: no eligible reports found")
        return

    # Run twice
    v1_results = await re_judge_run(cells, queries, run_label="v1")
    v2_results = await re_judge_run(cells, queries, run_label="v2")

    # Save raw verdicts
    v1_df = pd.DataFrame(v1_results)
    v2_df = pd.DataFrame(v2_results)
    v1_df.to_parquet(OUT_DIR / "v1.parquet", index=False)
    v2_df.to_parquet(OUT_DIR / "v2.parquet", index=False)
    print(f"\nSaved raw verdicts: v1={len(v1_df)}, v2={len(v2_df)}")

    # Analyse
    stats = analyse(v1_df, v2_df)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    write_summary(stats, OUT_DIR / "summary.md")
    print(f"\nE3 complete:")
    print(f"  Overall κ: {stats['overall_kappa']:.3f}")
    print(f"  Overall r: {stats['overall_score_pearson_r']:.3f}")
    print(f"  Mean |Δ|: {stats['overall_score_mean_abs_delta']:.4f}")
    print(f"  Summary: {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
