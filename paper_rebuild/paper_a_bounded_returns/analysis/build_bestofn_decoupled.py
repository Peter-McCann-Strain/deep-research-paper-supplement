#!/usr/bin/env python
"""Noise-decoupled best-of-N: removes the winner's-curse circularity of build_bestofn.py.

The naive best-of-k curve selects and scores with the same judge score, so selecting the
max of k replicates is upward-biased by judge measurement noise (selection on noise).
Two corrections, both zero-API-cost reanalyses of the existing df_verdicts:

1. pure_noise: the Gaussian max-order-statistic prediction for selecting the best of k
   replicates of a FLAT process with the measured pooled within-query SD. If the observed
   best-of-k curve tracks this prediction, the curve is consistent with selection on noise
   alone.
2. decoupled (split-half): within each dimension, criteria are split by within-dimension
   parity into a selection half (A) and a disjoint scoring half (B). The selector picks the
   replicate with the best half-A score; the curve reports the half-B score of the selected
   replicate. Criterion-level judge noise is independent across halves, so half-B is an
   unbiased estimate of the selected report's quality on the held-out criteria (report-level
   and judging-session-level effects remain shared). The cluster reference is recomputed on
   the same half-B basis so scales match.

Appends canonical_numbers.json['best_of_n']['pure_noise' | 'decoupled'].
GPT-5.2 single-judge, 30 variance queries, same replicate ordering as build_bestofn.py.
"""
import pandas as pd, numpy as np, json, warnings, glob, os, re
warnings.filterwarnings("ignore")
ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

W = {"information_recall": 0.20, "factual_accuracy": 0.20, "coverage": 0.10,
     "analytical_depth": 0.15, "citation_quality": 0.10, "logical_coherence": 0.05,
     "organization": 0.05, "instruction_following": 0.10, "attribution_quality": 0.05}

VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
P0 = ["base_p0"] + sorted([os.path.basename(d) for d in glob.glob(f"{ROOT}/results/judge_gpt52/base_p0_v*")
                           if re.match(r"base_p0_v\d+$", os.path.basename(d))],
                          key=lambda s: int(s.split("_v")[1]))
CLUSTER = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]

ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
g = ov[ov.judge.eq("gpt52")]

def patq(p, q):
    d = g[(g.pattern == p) & (g.query_id == q)]
    return float(d.overall_score.iloc[0]) if len(d) and pd.notna(d.overall_score.iloc[0]) else None

# ---- replicate full scores (same construction as build_bestofn.py) ----
samples = {}
for q in VARQ:
    s = [patq(p, q) for p in P0]
    s = [x for x in s if x is not None]
    if len(s) >= 3:
        samples[q] = s
qs = sorted(samples)
N = min(len(samples[q]) for q in qs)

# ---- 1. pure-noise max-order-statistic prediction ----
# pooled within-query SD of the replicate scores
sigma = float(np.sqrt(np.mean([np.var(samples[q][:N], ddof=1) for q in qs])))
rng = np.random.default_rng(0)
sims = rng.standard_normal((200_000, N))
emax = [float(np.mean(np.max(sims[:, :k], axis=1))) for k in range(1, N + 1)]
k1_mean = float(np.mean([samples[q][0] for q in qs]))
flat_mean = float(np.mean([np.mean(samples[q][:N]) for q in qs]))
pure = {k: {"E_max_of_k_sd_units": round(emax[k - 1], 4),
            "predicted_best_of_k": round(flat_mean + sigma * emax[k - 1], 4)}
        for k in range(1, N + 1)}

# ---- 2. split-half decoupled selection ----
vd = pd.read_parquet(f"{A}/df_verdicts.parquet")
vd = vd[vd.judge.eq("gpt52") & vd.query_id.isin(qs) & vd.satisfied_is_known
        & vd.pattern.isin(P0 + CLUSTER)].copy()
vd = vd.sort_values("criterion_index")
vd["half"] = vd.groupby(["pattern", "query_id", "dimension"], observed=True).cumcount() % 2  # 0=A select, 1=B score

def half_score(rows):
    """Weighted overall score from one half's verdicts (renormalised over present dims)."""
    dim = rows.groupby("dimension", observed=True)["satisfied"].mean()
    wsum = sum(W[d] for d in dim.index)
    return float(sum(W[d] * dim[d] for d in dim.index) / wsum) if wsum else np.nan

hs = vd.groupby(["pattern", "query_id", "half"], observed=True).apply(half_score).rename("score").reset_index()
hsA = hs[hs.half == 0].set_index(["pattern", "query_id"]).score
hsB = hs[hs.half == 1].set_index(["pattern", "query_id"]).score

def hget(tbl, p, q):
    try:
        v = tbl.loc[(p, q)]
        return float(v) if pd.notna(v) else None
    except KeyError:
        return None

selA, scoreB = {}, {}
for q in qs:
    a = [hget(hsA, p, q) for p in P0]
    b = [hget(hsB, p, q) for p in P0]
    keep = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    selA[q] = [a[i] for i in keep][:N]
    scoreB[q] = [b[i] for i in keep][:N]
qs2 = [q for q in qs if len(selA[q]) >= 3]
N2 = min(len(selA[q]) for q in qs2)

cluster_B = float(np.mean([np.mean([x for x in (hget(hsB, p, q) for p in CLUSTER) if x is not None])
                           for q in qs2]))
cluster_B_q = {q: float(np.mean([x for x in (hget(hsB, p, q) for p in CLUSTER) if x is not None]))
               for q in qs2}
rng_gap = np.random.default_rng(3)
dec_curve, naive_curve = {}, {}
for k in range(1, N2 + 1):
    # decoupled: select on half A, score on half B
    sel_q = {q: scoreB[q][int(np.argmax(selA[q][:k]))] for q in qs2}
    dec = float(np.mean(list(sel_q.values())))
    # paired per-query gap (decoupled - cluster_B) with seeded query bootstrap CI
    gaps = np.array([sel_q[q] - cluster_B_q[q] for q in qs2])
    boot = [float(np.mean(gaps[rng_gap.integers(0, len(gaps), len(gaps))])) for _ in range(5000)]
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
    # naive same-half reference: select on half B, score on half B (winner's curse visible)
    nai = float(np.mean([max(scoreB[q][:k]) for q in qs2]))
    dec_curve[k] = {"best_of_k_decoupled": round(dec, 4), "gap_to_cluster_B": round(cluster_B - dec, 4),
                    "gap_ci95": ci}
    naive_curve[k] = round(nai, 4)

cn = json.load(open(f"{ANA}/canonical_numbers.json"))
bo = cn.setdefault("best_of_n", {})
bo["pure_noise"] = {
    "sigma_within_query": round(sigma, 4), "flat_mean": round(flat_mean, 4),
    "k1_mean": round(k1_mean, 4), "prediction": pure,
    "note": "predicted best-of-k if replicates were a flat process + iid noise at the measured "
            "within-query SD; observed curve tracking this prediction = selection on noise"}
bo["decoupled"] = {
    "judge": "gpt52", "n_queries": len(qs2), "n_samples": N2,
    "cluster_mean_half_B": round(cluster_B, 4),
    "curve": dec_curve, "naive_same_half_curve": naive_curve,
    "note": "select on criterion half A, score on disjoint half B (cluster reference on the same "
            "half-B basis); removes criterion-level selection noise, report/session effects remain shared"}
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
print(json.dumps({"pure_noise_pred_k4": pure.get(4), "pure_noise_pred_k5": pure.get(5),
                  "sigma": round(sigma, 4),
                  "decoupled_k_max": dec_curve[N2], "cluster_B": round(cluster_B, 4),
                  "decoupled_curve": {k: v["best_of_k_decoupled"] for k, v in dec_curve.items()},
                  "naive_curve": naive_curve}, indent=1))
