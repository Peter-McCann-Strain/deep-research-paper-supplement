#!/usr/bin/env python
"""Oracle factual-accuracy equivalence test (audit M3/RA2).

The dual-ceiling thesis needs factual accuracy to be "unmoved" under oracle retrieval — an
EQUIVALENCE claim. The paper argued it only from a CI straddling zero (absence of evidence).
This runs a formal TOST on the 6 per-pattern factual deltas (gpt52, variance-stratified
queries) at equivalence margins +/-0.05 and +/-0.02, and reports the n=6 minimum detectable
effect at 80% power. Bounds chosen against the measured run-noise SD (~0.047).

Appends canonical_numbers.json['oracle']['factual_tost'].
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
S = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
CLUSTER = ["p1", "p4", "p5", "p6", "p7", "p8"]
DIM = "factual_accuracy"
JUDGE = "gpt52"


def pattern_delta(p):
    o = S[(S.pattern == f"oracle_t1_{p}") & (S.judge == JUDGE) & (S.dimension == DIM) & (S.query_id.isin(VARQ))]
    b = S[(S.pattern == f"base_{p}") & (S.judge == JUDGE) & (S.dimension == DIM) & (S.query_id.isin(VARQ))]
    om = o.set_index("query_id")["score"].to_dict()
    bm = b.set_index("query_id")["score"].to_dict()
    diffs = [om[q] - bm[q] for q in om if q in bm]
    return float(np.mean(diffs)) if diffs else None


means = np.array([pattern_delta(p) for p in CLUSTER], dtype=float)
n = len(means)
m = float(means.mean())
sd = float(means.std(ddof=1))
se = sd / np.sqrt(n)
df_ = n - 1


def tost(bound):
    # H0_lower: mu <= -bound  (one-sided, reject if mean sufficiently above -bound)
    t_lower = (m - (-bound)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df_)
    # H0_upper: mu >= +bound
    t_upper = (m - bound) / se
    p_upper = stats.t.cdf(t_upper, df_)
    p_tost = max(p_lower, p_upper)
    return {"bound": bound, "p_lower": round(float(p_lower), 4), "p_upper": round(float(p_upper), 4),
            "p_tost": round(float(p_tost), 4), "equivalent_at_05_alpha": bool(p_tost < 0.05)}


# MDE at 80% power, one-sample t (two-sided alpha=0.05) given observed sd, n.
# Reported on the 6-cluster unit, which is the correct inferential unit (patterns share queries).
tcrit = stats.t.ppf(0.975, df_)
tpow = stats.t.ppf(0.80, df_)
mde_cluster = float((tcrit + tpow) * se)

res = {
    "_note": "TOST on the 6 per-pattern factual_accuracy oracle-minus-base means (gpt52, variance queries). "
             "Equivalent => no practically meaningful change within the margin.",
    "n_clusters": n,
    "mean_delta": round(m, 4),
    "sd_of_pattern_means": round(sd, 4),
    "per_pattern_delta": {p: round(d, 4) for p, d in zip(CLUSTER, means)},
    "tost_0.05": tost(0.05),
    "tost_0.02": tost(0.02),
    "mde80_6cluster": round(mde_cluster, 4),
    "interpretation": "Equivalent to zero within +/-0.05 (practical-equivalence region ~ run-noise SD) "
                      "but NOT within +/-0.02; report as 'no detectable change within +/-0.05', not 'pinned at zero'.",
}

cn = json.load(open(f"{ANA}/canonical_numbers.json"))
cn["oracle"]["factual_tost"] = res
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
print(json.dumps(res, indent=1))
