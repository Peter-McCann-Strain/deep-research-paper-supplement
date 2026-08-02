#!/usr/bin/env python
"""Cluster-aware oracle CI (fixes the flat n=179 bootstrap; audit M2/RA1).

The headline oracle deltas (canonical['oracle']['cluster_dims'], gpt52) come from a FLAT
bootstrap that pools 6 patterns x 30 shared queries and resamples as if all ~179 paired
deltas were independent. They are not: deltas share a per-pattern oracle-injection effect and
a per-query difficulty effect, so the flat interval is anticonservative (effective n = 6
clusters, not 179). This recomputes citation_quality and factual_accuracy with a two-stage
block bootstrap (resample patterns, then queries within), plus a leave-P5-out sensitivity
(P5 is the +1113-word/+95-citation outlier arm). gpt52 only: this is the headline judge and
its df_scores per-dimension values are clean (the within-version Claude replication already
lives in oracle.panel_cross_check, built from raw opus48/sonnet48 baselines).

Appends canonical_numbers.json['oracle']['cluster_ci_robust'].
"""
import json
import numpy as np
import pandas as pd

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
S = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
CLUSTER = ["p1", "p4", "p5", "p6", "p7", "p8"]
DIMS = ["citation_quality", "factual_accuracy"]
JUDGE = "gpt52"
N_REPS = 5000
SEED = 20260610


def build_deltas(dim, patterns):
    rows = []
    for p in patterns:
        o = S[(S.pattern == f"oracle_t1_{p}") & (S.judge == JUDGE) & (S.dimension == dim) & (S.query_id.isin(VARQ))]
        b = S[(S.pattern == f"base_{p}") & (S.judge == JUDGE) & (S.dimension == dim) & (S.query_id.isin(VARQ))]
        om = o.set_index("query_id")["score"].to_dict()
        bm = b.set_index("query_id")["score"].to_dict()
        for q in om:
            if q in bm:
                rows.append({"pattern": p, "delta": float(om[q] - bm[q])})
    return pd.DataFrame(rows)


def flat_boot(df, rng, reps=N_REPS):
    a = df["delta"].to_numpy(); n = len(a)
    return np.array([a[rng.integers(0, n, n)].mean() for _ in range(reps)])


def block_boot(df, rng, reps=N_REPS):
    groups = {p: g["delta"].to_numpy() for p, g in df.groupby("pattern")}
    pats = list(groups); npat = len(pats)
    out = np.empty(reps)
    for i in range(reps):
        chosen = rng.integers(0, npat, npat)
        out[i] = np.concatenate([groups[pats[c]][rng.integers(0, len(groups[pats[c]]), len(groups[pats[c]]))] for c in chosen]).mean()
    return out


def ci(b):
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def w(c):
    return round(c[1] - c[0], 4)


res = {"_note": "Cluster-aware two-stage block bootstrap (resample 6 patterns, then queries within); "
                 "replaces the flat n=179 i.i.d. interval. gpt52 headline judge. seed=%d, reps=%d." % (SEED, N_REPS),
       "estimand": "pooled oracle-minus-base delta over 6 cluster patterns x variance-stratified queries",
       "dims": {}}
for dim in DIMS:
    df = build_deltas(dim, CLUSTER)
    point = round(float(df["delta"].mean()), 4)
    fci = ci(flat_boot(df, np.random.default_rng(SEED)))
    bci = ci(block_boot(df, np.random.default_rng(SEED)))
    dfn = build_deltas(dim, [p for p in CLUSTER if p != "p5"])
    bci_no5 = ci(block_boot(dfn, np.random.default_rng(SEED)))
    res["dims"][dim] = {
        "n_paired": int(len(df)),
        "n_clusters": df["pattern"].nunique(),
        "point_delta": point,
        "flat_ci": fci, "flat_width": w(fci),
        "block_ci": bci, "block_width": w(bci),
        "width_ratio_block_over_flat": round(w(bci) / w(fci), 2) if w(fci) else None,
        "block_ci_excludes_zero": bool(not (bci[0] <= 0 <= bci[1])),
        "leave_p5_out": {"point_delta": round(float(dfn["delta"].mean()), 4),
                          "block_ci": bci_no5,
                          "block_ci_excludes_zero": bool(not (bci_no5[0] <= 0 <= bci_no5[1]))},
    }

cn = json.load(open(f"{ANA}/canonical_numbers.json"))
cn["oracle"]["cluster_ci_robust"] = res
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
for dim, d in res["dims"].items():
    print(f"{dim:18s} point {d['point_delta']:+.4f} | flat {d['flat_ci']} w={d['flat_width']} "
          f"| block {d['block_ci']} w={d['block_width']} ({d['width_ratio_block_over_flat']}x) "
          f"| excl0={d['block_ci_excludes_zero']} | noP5 {d['leave_p5_out']['block_ci']}")
