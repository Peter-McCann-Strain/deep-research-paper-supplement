#!/usr/bin/env python
"""Reversed-direction split-half decoupled best-of-N + shared report-level error estimate.

build_bestofn_decoupled.py selects on criterion half A and scores on the disjoint half B
(canonical best_of_n.decoupled.curve). This script runs the REVERSED direction (select on
half B, score on half A, cluster reference recomputed on the half-A basis) with the same
deterministic construction and seed conventions, reports the mean of the two directions,
and estimates the SHARED report-level error: the pooled within-query correlation between
half-A and half-B score deviations from each query's replicate mean. Decoupling removes
only half-specific criterion noise; replicate-level variation (true quality differences
between replicates + judging-session effects) is shared across halves, and this correlation
estimates its share of the within-query replicate variance.

The forward direction is recomputed and asserted equal to the stored canonical curve before
anything is written (drift guard). Appends NEW subkeys to canonical
best_of_n.decoupled: 'reversed_split', 'mean_of_directions', 'half_correlation'
(atomic tmp+os.replace; refuses to overwrite existing subkeys). GPT-5.2 single-judge,
30 variance queries, same replicate ordering as build_bestofn.py.
"""
import pandas as pd, numpy as np, json, warnings, glob, os, re
warnings.filterwarnings("ignore")
ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANONICAL = f"{ANA}/canonical_numbers.json"

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

# ---- replicate full scores (identical construction to build_bestofn_decoupled.py) ----
samples = {}
for q in VARQ:
    s = [patq(p, q) for p in P0]
    s = [x for x in s if x is not None]
    if len(s) >= 3:
        samples[q] = s
qs = sorted(samples)
N = min(len(samples[q]) for q in qs)

# ---- split-half scores (identical to build_bestofn_decoupled.py) ----
vd = pd.read_parquet(f"{A}/df_verdicts.parquet")
vd = vd[vd.judge.eq("gpt52") & vd.query_id.isin(qs) & vd.satisfied_is_known
        & vd.pattern.isin(P0 + CLUSTER)].copy()
vd = vd.sort_values("criterion_index")
vd["half"] = vd.groupby(["pattern", "query_id", "dimension"], observed=True).cumcount() % 2  # 0=A, 1=B

def half_score(rows):
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

halfA, halfB = {}, {}
for q in qs:
    a = [hget(hsA, p, q) for p in P0]
    b = [hget(hsB, p, q) for p in P0]
    keep = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    halfA[q] = [a[i] for i in keep][:N]
    halfB[q] = [b[i] for i in keep][:N]
qs2 = [q for q in qs if len(halfA[q]) >= 3]
N2 = min(len(halfA[q]) for q in qs2)

def cluster_ref(tbl):
    per_q = {q: float(np.mean([x for x in (hget(tbl, p, q) for p in CLUSTER) if x is not None]))
             for q in qs2}
    return float(np.mean(list(per_q.values()))), per_q

def run_direction(sel, score, cluster_tbl):
    """Select on `sel` half, score on disjoint `score` half; cluster ref on the scoring half.
    Same k-loop and rng seed (3) conventions as build_bestofn_decoupled.py."""
    cl_mean, cl_q = cluster_ref(cluster_tbl)
    rng_gap = np.random.default_rng(3)
    curve = {}
    for k in range(1, N2 + 1):
        sel_q = {q: score[q][int(np.argmax(sel[q][:k]))] for q in qs2}
        dec = float(np.mean(list(sel_q.values())))
        gaps = np.array([sel_q[q] - cl_q[q] for q in qs2])
        boot = [float(np.mean(gaps[rng_gap.integers(0, len(gaps), len(gaps))])) for _ in range(5000)]
        ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
        curve[k] = {"best_of_k_decoupled": round(dec, 4),
                    "gap_to_cluster": round(cl_mean - dec, 4), "gap_ci95": ci}
    return cl_mean, curve

# forward (select A, score B) -- drift guard against the stored canonical curve
cl_B, fwd = run_direction(halfA, halfB, hsB)
cn = json.load(open(CANONICAL))
dec_stored = cn["best_of_n"]["decoupled"]
assert round(cl_B, 4) == dec_stored["cluster_mean_half_B"], "forward cluster ref drifted"
for k in range(1, N2 + 1):
    assert fwd[k]["best_of_k_decoupled"] == dec_stored["curve"][str(k)]["best_of_k_decoupled"], \
        f"forward curve drifted at k={k}"

# reversed (select B, score A)
cl_A, rev = run_direction(halfB, halfA, hsA)

# mean of the two directions (per-k mean decoupled score and mean gap-to-cluster,
# with a seeded query bootstrap on the per-query mean gap)
_, cl_Bq = cluster_ref(hsB)
_, cl_Aq = cluster_ref(hsA)
rng_mean = np.random.default_rng(7)
mean_dir = {}
for k in range(1, N2 + 1):
    sAB = {q: halfB[q][int(np.argmax(halfA[q][:k]))] for q in qs2}
    sBA = {q: halfA[q][int(np.argmax(halfB[q][:k]))] for q in qs2}
    gaps = np.array([((cl_Bq[q] - sAB[q]) + (cl_Aq[q] - sBA[q])) / 2 for q in qs2])
    boot = [float(np.mean(gaps[rng_mean.integers(0, len(gaps), len(gaps))])) for _ in range(5000)]
    ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
    mean_dir[k] = {
        "best_of_k_decoupled": round(float(np.mean([(sAB[q] + sBA[q]) / 2 for q in qs2])), 4),
        "gap_to_cluster": round(float(gaps.mean()), 4), "gap_ci95": ci}

# ---- shared report-level error: within-query corr of half-A vs half-B deviations ----
devA, devB, per_q_n = [], [], []
for q in qs2:
    a = np.array(halfA[q]); b = np.array(halfB[q])
    devA.append(a - a.mean()); devB.append(b - b.mean()); per_q_n.append(len(a))
dA = np.concatenate(devA); dB = np.concatenate(devB)
r_obs = float(np.corrcoef(dA, dB)[0, 1])
cov = float(np.mean(dA * dB))
share_var = float(cov / np.mean([np.mean(dA**2), np.mean(dB**2)]))
rng_r = np.random.default_rng(11)
rboot = []
for _ in range(5000):
    idx = rng_r.integers(0, len(qs2), len(qs2))
    da = np.concatenate([devA[i] for i in idx]); db = np.concatenate([devB[i] for i in idx])
    if da.std() > 0 and db.std() > 0:
        rboot.append(float(np.corrcoef(da, db)[0, 1]))
r_ci = [round(float(np.percentile(rboot, 2.5)), 3), round(float(np.percentile(rboot, 97.5)), 3)]

half_corr = {
    "pearson_r": round(r_obs, 4), "r_ci95_query_bootstrap": r_ci,
    "shared_variance_share": round(share_var, 4),
    "n_replicate_pairs": int(len(dA)), "n_queries": len(qs2),
    "seed_bootstrap": 11,
    "note": ("corr of half-A vs half-B score deviations from each query's replicate mean, "
             "pooled over the P0 replicate corpus. This is the share of within-query "
             "replicate variance SHARED across criterion halves (true replicate quality "
             "differences + report/session-level judge error) that split-half decoupling "
             "cannot remove; 1-r is the half-specific criterion-noise share it does remove.")}

reversed_block = {
    "direction": "select on half B, score on half A (cluster reference on half-A basis)",
    "cluster_mean_half_A": round(cl_A, 4),
    "curve": {str(k): rev[k] for k in rev},
    "seed_conventions": "identical to forward (rng(3) gap bootstrap, 5000 draws, k-ordered)"}
mean_block = {
    "note": ("per-k mean of the two split directions; gap averaged per query against each "
             "direction's own-half cluster reference, seeded query bootstrap (rng(7))"),
    "curve": {str(k): mean_dir[k] for k in mean_dir}}

dec = cn["best_of_n"]["decoupled"]
for key in ["reversed_split", "mean_of_directions", "half_correlation"]:
    assert key not in dec, f"refusing to overwrite existing subkey {key}"
dec["reversed_split"] = reversed_block
dec["mean_of_directions"] = mean_block
dec["half_correlation"] = half_corr
tmp = CANONICAL + ".tmp"
with open(tmp, "w") as fh:
    fh.write(json.dumps(cn, indent=1))
os.replace(tmp, CANONICAL)

print(json.dumps({
    "forward_matches_stored": True,
    "cluster_half_B": round(cl_B, 4), "cluster_half_A": round(cl_A, 4),
    "forward_curve": {k: fwd[k]["best_of_k_decoupled"] for k in fwd},
    "reversed_curve": {k: rev[k]["best_of_k_decoupled"] for k in rev},
    "mean_curve": {k: mean_dir[k]["best_of_k_decoupled"] for k in mean_dir},
    "mean_gaps": {k: mean_dir[k]["gap_to_cluster"] for k in mean_dir},
    "half_correlation": half_corr}, indent=1))
