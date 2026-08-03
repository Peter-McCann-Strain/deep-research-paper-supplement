#!/usr/bin/env python3
"""E6: Placeholder-filtered cluster re-ranking — pure analysis.

Reuses `scripts.phase7a_citation_extraction.extract_citations_from_report`
(canonical citation classifier) to compute per-report placeholder rates
across the bing baseline (`base_p*`) and the Protocol A Tavily wave
(`protocol_a_tavily_p*`). Then joins with the judge `df_scores.parquet`
and re-derives the §5.3 cluster ordering on the filtered subset where
placeholder_rate < threshold.

Run AFTER:
  - Protocol A Tavily wave is complete (E1)
  - Tavily reports are judged with run_gpt52_judge.py --patterns-raw
    'protocol_a_tavily_p0,protocol_a_tavily_p1,protocol_a_tavily_p3,
     protocol_a_tavily_p4,protocol_a_tavily_p5,protocol_a_tavily_p8'
  - Parquets rebuilt with build_analysis_dataframes.py

Outputs:
  data/analysis/df_citations_protocol_a.parquet  (extended citations table)
  reports/phase12_filtered_rerank/per_backend_filtered_means.csv
  reports/phase12_filtered_rerank/cluster_shift.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the canonical citation extractor
from scripts.phase7a_citation_extraction import extract_citations_from_report

EXPERIMENTS_DIR = Path("results/experiments")
OUT_DIR = Path("reports/phase12_filtered_rerank")

PROTOCOL_A_PATTERNS = ["p0", "p1", "p3", "p4", "p5", "p8"]


def collect_citations(stratified_qids: list[str]) -> pd.DataFrame:
    """Walk both bing baseline and Tavily wave; classify citations."""
    records: list[dict] = []
    for pattern_short in PROTOCOL_A_PATTERNS:
        for backend, prefix in [("bing", f"base_{pattern_short}"),
                                 ("tavily", f"protocol_a_tavily_{pattern_short}")]:
            pdir = EXPERIMENTS_DIR / prefix
            if not pdir.exists():
                continue
            for rfile in pdir.glob("*.md"):
                qid = rfile.stem
                if stratified_qids and qid not in stratified_qids:
                    continue
                recs = extract_citations_from_report(rfile, prefix, qid)
                for r in recs:
                    r["backend"] = backend
                    r["pattern_short"] = pattern_short
                records.extend(recs)
    return pd.DataFrame(records)


def per_report_placeholder_rate(df_cit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate citation rows to per-report rates."""
    if df_cit.empty:
        return pd.DataFrame()
    grp = df_cit.groupby(["backend", "pattern_short", "pattern", "query_id"])
    out = []
    for (backend, pat_short, pat, qid), sub in grp:
        n = len(sub)
        n_ph = (sub["category"] == "placeholder").sum()
        n_url = (sub["category"] == "real_url").sum()
        n_acad = (sub["category"] == "academic").sum()
        out.append({
            "backend": backend, "pattern_short": pat_short, "pattern": pat,
            "query_id": qid, "n_citations": n,
            "placeholder_rate": n_ph / n if n else float("nan"),
            "url_rate": n_url / n if n else float("nan"),
            "academic_rate": n_acad / n if n else float("nan"),
        })
    return pd.DataFrame(out)


def join_with_scores(df_rates: pd.DataFrame) -> pd.DataFrame:
    df_scores = pd.read_parquet("data/analysis/df_scores.parquet")
    # Aggregate to per-report mean overall (across judges and dimensions)
    df_overall = (df_scores.groupby(["pattern", "query_id"], observed=True)
                  ["score"].mean().reset_index().rename(columns={"score": "overall_score"}))
    return df_rates.merge(df_overall, on=["pattern", "query_id"], how="left")


def filtered_rerank(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    full = df.copy()
    filtered = df[df["placeholder_rate"] < threshold]
    unf = (full.groupby(["backend", "pattern_short"])["overall_score"]
           .agg(["mean", "count"]).reset_index().rename(columns={"mean": "mean_unfiltered"}))
    fil = (filtered.groupby(["backend", "pattern_short"])["overall_score"]
           .agg(["mean", "count"]).reset_index().rename(columns={"mean": "mean_filtered", "count": "n_filtered"}))
    return unf.merge(fil, on=["backend", "pattern_short"], how="outer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--stratified-file", default="data/protocol_a_stratified_v2.json",
                        help="JSON with {'query_ids': [...]} restricting analysis to that subset")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stratified_qids: list[str] = []
    if args.stratified_file and Path(args.stratified_file).exists():
        stratified_qids = json.loads(Path(args.stratified_file).read_text())["query_ids"]
        print(f"  restricting to {len(stratified_qids)} stratified queries")

    df_cit = collect_citations(stratified_qids)
    if df_cit.empty:
        print("No citation data found — has the Tavily wave finished?")
        return
    print(f"  collected {len(df_cit):,} citation rows from "
          f"{df_cit['backend'].value_counts().to_dict()}")
    df_cit.to_parquet(Path("data/analysis/df_citations_protocol_a.parquet"), index=False)

    df_rates = per_report_placeholder_rate(df_cit)
    df_joined = join_with_scores(df_rates)
    if df_joined["overall_score"].isna().all():
        print("WARN: no judged scores — run run_gpt52_judge.py --patterns-raw on protocol_a_tavily_*")
    print(f"  per-report rates: {len(df_rates)} rows")
    print(df_rates.groupby(["backend", "pattern_short"])["placeholder_rate"]
          .agg(["mean", "median", "count"]).round(3))

    out = filtered_rerank(df_joined, args.threshold)
    out["delta_filtered_minus_unfiltered"] = out["mean_filtered"] - out["mean_unfiltered"]
    out_path = OUT_DIR / "per_backend_filtered_means.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    print(out.to_string(index=False))

    # Cluster ordering (filtered) per backend
    summaries = ["# E6: Placeholder-filtered re-ranking",
                 f"\nFilter: placeholder_rate < {args.threshold}",
                 f"Stratified subset: {len(stratified_qids)} queries\n"]
    for backend in sorted(out["backend"].dropna().unique()):
        sub = out[out["backend"] == backend].dropna(subset=["mean_filtered"])
        if sub.empty:
            continue
        unf_rank = sub.sort_values("mean_unfiltered", ascending=False)["pattern_short"].tolist()
        fil_rank = sub.sort_values("mean_filtered", ascending=False)["pattern_short"].tolist()
        summaries.append(f"### {backend}")
        summaries.append(f"- Unfiltered ranking: {' > '.join(unf_rank)}")
        summaries.append(f"- Filtered ranking: {' > '.join(fil_rank)}")
        if unf_rank == fil_rank:
            summaries.append("- **Cluster identity preserved** under placeholder filter.\n")
        else:
            summaries.append("- **Cluster identity SHIFTS** under placeholder filter.\n")
    md_path = OUT_DIR / "cluster_shift.md"
    md_path.write_text("\n".join(summaries))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
