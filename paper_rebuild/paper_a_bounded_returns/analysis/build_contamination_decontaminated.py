#!/usr/bin/env python
"""E6 STC-AUDIT — STEP 5: recompute the four headline effects on the DECONTAMINATED query set.

Part of E6 (prereg docs/publication/prereg/prereg_E6.md). Robustness appendix, NOT a headline. Wired
into rebuild_all.sh AFTER build_numbers, behind a guard: it runs only if a contaminated-query
list exists (results/contamination_e6/contaminated_queries.json, produced by
build_contamination.py STEP 4). With no list it prints a notice and exits 0 so the existing
rebuild chain is never broken.

Primary endpoint half #2 (the other half is the architecture coefficient in
build_contamination.py): DO the four headline effects SURVIVE dropping the contaminated
queries? We recompute, on the public-benchmark base panel with the contaminated query_ids
removed:

  H1  top-cluster flatness    : is the top cluster of architectures still statistically flat
                                (no judge-robust pairwise separation within the cluster)?
  H2  cluster vs P0 gap        : does the top-cluster-minus-P0 overall gap survive?
  H3  pairwise judge-robust ct : the count of judge-robust (Holm, all 3 judges, consistent
                                 sign) pairwise separations among base_p0..p10.
  H4  best-single overall      : the rank-1 architecture by 3-judge mean overall.

Each is reported FULL vs DECONTAMINATED with the delta, so the survive/not-survive call is
explicit and publishes whichever way it lands (per the prereg). Determinism: no randomness
beyond the Wilcoxon (exact/asymptotic, deterministic). Writes a side-car
results/contamination_e6/decontaminated_headline.json; appends canonical only via
build_contamination.py --finalize (this script never mutates canonical to keep the guard
single-sourced).

Usage:
    python paper_rebuild/paper_a_bounded_returns/analysis/build_contamination_decontaminated.py --help
    python paper_rebuild/paper_a_bounded_returns/analysis/build_contamination_decontaminated.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
E6 = ROOT / "results" / "contamination_e6"
CONTAM_LIST = E6 / "contaminated_queries.json"
JUDGES = ["gpt52", "claude_opus", "claude_sonnet"]
PATS = [f"base_p{i}" for i in range(11)]


def _holm(pv: np.ndarray) -> np.ndarray:
    idx = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(run, 1.0)
    return adj


def _judge_robust_count(base, pairs) -> Dict:
    """Joint-Holm judge-robust pairwise separations among PATS (mirrors build_joint_holm)."""
    from scipy.stats import wilcoxon
    pv, sgn, key = [], [], []
    for j in JUDGES:
        d = base[base.judge == j]
        wide = d.pivot_table(index="query_id", columns="pattern", values="ovc", observed=True)
        for a, b in pairs:
            if a not in wide or b not in wide:
                pv.append(1.0); sgn.append(0.0); key.append((j, a, b)); continue
            s = wide[[a, b]].dropna()
            try:
                p = wilcoxon(s[a], s[b]).pvalue if len(s) > 1 else 1.0
            except Exception:
                p = 1.0
            pv.append(p); sgn.append(float(np.sign((s[a] - s[b]).mean()))); key.append((j, a, b))
    adj = _holm(np.array(pv))
    sig = {key[i]: bool(adj[i] < 0.05) for i in range(len(key))}
    sg = {key[i]: sgn[i] for i in range(len(key))}
    robust_pairs = [
        (a, b) for a, b in pairs
        if all(sig[(j, a, b)] for j in JUDGES)
        and len({sg[(j, a, b)] for j in JUDGES}) == 1
    ]
    return {"count": len(robust_pairs), "pairs": [f"{a}>{b}" for a, b in robust_pairs]}


def _panel_summary(base, top_cluster: List[str]) -> Dict:
    """Mean 3-judge overall per pattern + the headline H1/H2/H4 derived numbers."""
    means = (base.groupby("pattern", observed=True)["ovc"].mean().sort_values(ascending=False))
    top1 = means.index[0] if len(means) else None
    cluster_present = [p for p in top_cluster if p in means.index]
    cluster_mean = float(means[cluster_present].mean()) if cluster_present else None
    p0_mean = float(means["base_p0"]) if "base_p0" in means.index else None
    return {
        "rank1_pattern": top1,
        "rank1_mean": round(float(means.iloc[0]), 4) if len(means) else None,
        "per_pattern_mean": {k: round(float(v), 4) for k, v in means.items()},
        "top_cluster_members_present": cluster_present,
        "top_cluster_mean": round(cluster_mean, 4) if cluster_mean is not None else None,
        "cluster_minus_p0": (round(cluster_mean - p0_mean, 4)
                             if (cluster_mean is not None and p0_mean is not None) else None),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contam-list", default=str(CONTAM_LIST),
                    help="JSON with {'contaminated_query_set': [...]} from build_contamination.py")
    ap.add_argument("--top-cluster", default="base_p1,base_p4,base_p6,base_p7,base_p8",
                    help="comma-separated top-cluster members for the flatness check")
    ap.add_argument("--dry-run", action="store_true", help="plan only; write nothing")
    args = ap.parse_args(argv)

    import pandas as pd

    contam_path = Path(args.contam_list)
    if not contam_path.is_absolute():
        contam_path = ROOT / contam_path
    if not contam_path.exists():
        # GUARD: keep rebuild_all.sh green when E6 has not run yet.
        print(f"[E6 decontam] no contaminated-query list at {contam_path}; "
              f"skipping decontamination recompute (run build_contamination.py first). "
              f"rebuild chain unaffected.")
        return 0

    contam = set(json.loads(contam_path.read_text()).get("contaminated_query_set", []))
    top_cluster = [c.strip() for c in args.top_cluster.split(",") if c.strip()]

    ov = pd.read_parquet(ROOT / "data" / "analysis" / "df_overall_scores.parquet")
    ov["ovc"] = ov["overall_score"].where(
        ~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
    base_all = ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(JUDGES)].copy()
    pairs = list(itertools.combinations(PATS, 2))

    cluster_pairs = list(itertools.combinations(
        [p for p in top_cluster], 2))

    def compute(df, tag):
        jr = _judge_robust_count(df[df.pattern.isin(PATS)], pairs)
        within = _judge_robust_count(df[df.pattern.isin(top_cluster)], cluster_pairs)
        summ = _panel_summary(df[df.pattern.isin(PATS)], top_cluster)
        return {
            "tag": tag,
            "n_queries": int(df.query_id.nunique()),
            "H1_top_cluster_flat": within["count"] == 0,
            "H1_within_cluster_robust_separations": within["count"],
            "H2_cluster_minus_p0": summ["cluster_minus_p0"],
            "H3_judge_robust_pairwise_count_of_55": jr["count"],
            "H4_rank1_pattern": summ["rank1_pattern"],
            "H4_rank1_mean": summ["rank1_mean"],
            "_panel": summ,
        }

    full = compute(base_all, "full")
    decon = compute(base_all[~base_all.query_id.isin(contam)], "decontaminated")

    survive = {
        "H1_top_cluster_still_flat": bool(decon["H1_top_cluster_flat"]),
        "H1_full_was_flat": bool(full["H1_top_cluster_flat"]),
        "H2_cluster_minus_p0_delta": (
            None if (full["H2_cluster_minus_p0"] is None or decon["H2_cluster_minus_p0"] is None)
            else round(decon["H2_cluster_minus_p0"] - full["H2_cluster_minus_p0"], 4)),
        "H3_judge_robust_count_full": full["H3_judge_robust_pairwise_count_of_55"],
        "H3_judge_robust_count_decon": decon["H3_judge_robust_pairwise_count_of_55"],
        "H4_rank1_unchanged": full["H4_rank1_pattern"] == decon["H4_rank1_pattern"],
    }

    out = {
        "_note": ("E6 decontamination recompute: four headline effects FULL vs query set with "
                  "contaminated queries dropped. Half #2 of the E6 primary endpoint. Ships "
                  "whichever way it lands (prereg)."),
        "prereg": "docs/publication/prereg/prereg_E6.md",
        "n_contaminated_queries_dropped": len(contam),
        "top_cluster": top_cluster,
        "full": full,
        "decontaminated": decon,
        "survive": survive,
    }

    print("=" * 70)
    print("E6 STC-AUDIT — STEP 5 decontamination recompute")
    print(f"  contaminated queries dropped : {len(contam)}")
    print(f"  H1 top-cluster flat   full={full['H1_top_cluster_flat']} "
          f"decon={decon['H1_top_cluster_flat']}")
    print(f"  H2 cluster-minus-p0   full={full['H2_cluster_minus_p0']} "
          f"decon={decon['H2_cluster_minus_p0']} (delta {survive['H2_cluster_minus_p0_delta']})")
    print(f"  H3 judge-robust /55   full={full['H3_judge_robust_pairwise_count_of_55']} "
          f"decon={decon['H3_judge_robust_pairwise_count_of_55']}")
    print(f"  H4 rank-1             full={full['H4_rank1_pattern']} "
          f"decon={decon['H4_rank1_pattern']} (unchanged={survive['H4_rank1_unchanged']})")
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] nothing written.")
        return 0

    E6.mkdir(parents=True, exist_ok=True)
    (E6 / "decontaminated_headline.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {E6/'decontaminated_headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
