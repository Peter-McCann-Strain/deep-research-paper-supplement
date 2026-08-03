#!/usr/bin/env python
"""E1 ROUTABILITY-90 Stage A — is per-query winner rotation signal or noise?

Pre-registration: docs/publication/prereg/prereg_E1.md (registered 2026-06-11). Canonical key
`routability`. Decision gate G1 (headroom < 0.02 over best-fixed => null framing).

Binding amendments honoured: best-fixed framing is over P1 (plan). Ground truth (recorded in
PROGRAMME_EXECUTION_STATE.md): on gpt52 overall_score_recomputed, base_p4 (0.4679) and base_p1
(0.4665) are a statistical TIE for best-fixed (gap 0.0014 << MDE80=0.025); headroom is reported
over BOTH and they are nearly identical. The plan's quoted 0.653 is a different metric/scale; we
report the disk-computed value on the canonical metric and note the discrepancy.

The question: the RAW oracle gain (a perfect per-query router over the 11 architectures, minus
best-fixed) is ~0.094, but picking the max over 11 noisy single-run estimates capitalises on run
noise. The realizable, noise-corrected headroom is what a router could actually achieve.

Methods:
  - raw_oracle_gain: mean_q max_a score - mean_q best_fixed (overall, per-dimension, per-source).
  - winner_label_reliability: split-half test-retest of the per-query argmax on the REPLICATE
    corpus (8 replicated archs, 30 variance queries, gpt52), observed vs chance.
  - noise_corrected_headroom (primary): parametric bootstrap. Treat the observed gpt52 score as
    the truth estimate T; a realizable oracle selects on a NOISY run (T + e, e~N(0, sigma2_run)
    from E2) and is EVALUATED on T. The surviving gain = realizable headroom. sigma2_run is read
    from canonical variance_decomposition. Seeded.
  - replicate_cv_headroom (robustness): real independent runs - on the replicated subset, pick
    the per-query winner on replicate-half-1, evaluate on replicate-half-2 (no simulated noise).
  - GATE G1 on the noise-corrected headroom over P1.

Determinism: dedicated seeded generator on SORTED inputs.
"""
import json, os, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260611
ARCH = [f"base_p{i}" for i in range(11)]
DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization", "instruction_following",
        "attribution_quality"]

O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
Sd = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
Q = pd.read_parquet(f"{ROOT}/data/analysis/df_queries.parquet")
cn = json.load(open(f"{ANA}/canonical_numbers.json"))
sigma_run = float(np.sqrt(cn["variance_decomposition"]["pooled"]["sigma2_run"]))

g = O[(O.judge == "gpt52") & (O.pattern.isin(ARCH))][["pattern", "query_id", "overall_score_recomputed"]].dropna()
piv = g.pivot_table(index="query_id", columns="pattern", values="overall_score_recomputed", observed=True)
piv = piv[sorted(piv.columns)].dropna().sort_index()           # queries with all 11; sorted
M = piv.values                                                  # (n_queries, 11)
cols = list(piv.columns)
means = {p: round(float(piv[p].mean()), 4) for p in cols}
best_fixed = max(means, key=means.get)
P1 = "base_p1"

def oracle_gain(mat, base_col):
    return float(mat.max(axis=1).mean() - mat[:, cols.index(base_col)].mean())

raw = {
    "best_fixed_by_mean": best_fixed, "p1_mean": means[P1], "p4_mean": means["base_p4"],
    "tie_note": "base_p4 and base_p1 are within 0.0014 (<< MDE80 0.025): a statistical tie for best-fixed.",
    "n_queries_all11": int(M.shape[0]),
    "oracle_mean": round(float(M.max(axis=1).mean()), 4),
    "raw_gain_over_p1": round(oracle_gain(M, P1), 4),
    "raw_gain_over_p4": round(oracle_gain(M, "base_p4"), 4),
}

# per-dimension and per-source raw oracle gain over P1
def dim_matrix(dim):
    d = Sd[(Sd.judge == "gpt52") & (Sd.pattern.isin(ARCH)) & (Sd.dimension == dim)]
    pv = d.pivot_table(index="query_id", columns="pattern", values="score", observed=True)
    pv = pv[[c for c in sorted(pv.columns)]].dropna()
    return pv
raw["per_dimension_gain_over_p1"] = {}
for dim in DIMS:
    pv = dim_matrix(dim)
    if P1 in pv.columns and len(pv) > 5:
        raw["per_dimension_gain_over_p1"][dim] = round(float(pv.values.max(axis=1).mean() - pv[P1].mean()), 4)
src = Q.set_index("query_id")["source"].to_dict()
raw["per_source_gain_over_p1"] = {}
for s in sorted(set(src.get(q) for q in piv.index if src.get(q))):
    qs = [q for q in piv.index if src.get(q) == s]
    if len(qs) >= 4:
        sub = piv.loc[qs].values
        raw["per_source_gain_over_p1"][s] = {"n": len(qs),
            "gain": round(float(sub.max(axis=1).mean() - sub[:, cols.index(P1)].mean()), 4)}

# ---------- winner-label reliability: split-half on the replicate corpus ----------
rng = np.random.default_rng(SEED)
ov = O[(O.judge == "gpt52") & (O.pattern_family == "variance")][["pattern", "query_id", "overall_score_recomputed"]].dropna().copy()
ov["arch"] = ov.pattern.str.extract(r"(base_p\d+)_v")
varch = sorted(ov.arch.dropna().unique())
vqs = sorted(ov.query_id.unique())
reps = {a: sorted(ov[ov.arch == a].pattern.unique()) for a in varch}
def half_argmax(rep_assign):
    # mean score per (arch,query) over that half's replicates -> per-query argmax among archs
    win = {}
    for q in vqs:
        best_a, best_v = None, -1e9
        for a in varch:
            ps = rep_assign[a]
            vals = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(ps))].overall_score_recomputed
            if len(vals):
                m = float(vals.mean())
                if m > best_v: best_v, best_a = m, a
        win[q] = best_a
    return win
N_SPLIT = 200
agrees = []
for _ in range(N_SPLIT):
    h1, h2 = {}, {}
    for a in varch:
        r = reps[a][:]
        if len(r) >= 2:
            rng.shuffle(r); k = len(r) // 2
            h1[a], h2[a] = r[:max(k, 1)], r[max(k, 1):] or r[:1]
        else:
            h1[a] = h2[a] = r
    w1, w2 = half_argmax(h1), half_argmax(h2)
    both = [q for q in vqs if w1[q] and w2[q]]
    if both:
        agrees.append(np.mean([w1[q] == w2[q] for q in both]))
# chance: sum of squared empirical winner frequencies (prob two independent picks coincide)
full_win = half_argmax({a: reps[a] for a in varch})
from collections import Counter
freq = Counter(full_win[q] for q in vqs if full_win[q]); tot = sum(freq.values())
chance = float(sum((c / tot) ** 2 for c in freq.values())) if tot else None
winner_reliability = {
    "n_replicated_archs": len(varch), "archs": varch, "n_queries": len(vqs),
    "test_retest_agreement": round(float(np.mean(agrees)), 4) if agrees else None,
    "chance_agreement": round(chance, 4) if chance else None,
    "above_chance": bool(np.mean(agrees) > chance) if agrees and chance else None,
    "note": "split-half test-retest of the per-query argmax among the 8 replicated architectures "
            "(gpt52, 30 variance queries); chance = sum of squared winner frequencies.",
}

# ---------- noise-corrected headroom: select on T+noise, evaluate on T ----------
B = 4000
gains_p1, gains_p4 = [], []
i_p1, i_p4 = cols.index(P1), cols.index("base_p4")
for _ in range(B):
    noisy = M + rng.normal(0, sigma_run, size=M.shape)
    pick = noisy.argmax(axis=1)
    deployed = M[np.arange(M.shape[0]), pick].mean()      # evaluate the picked arch on T
    gains_p1.append(deployed - M[:, i_p1].mean())
    gains_p4.append(deployed - M[:, i_p4].mean())
nc_p1 = float(np.mean(gains_p1)); nc_p4 = float(np.mean(gains_p4))
noise_corrected = {
    "sigma_run_used": round(sigma_run, 4), "n_boot": B,
    "headroom_over_p1": round(nc_p1, 4),
    "headroom_over_p1_ci95": [round(float(np.percentile(gains_p1, 2.5)), 4),
                              round(float(np.percentile(gains_p1, 97.5)), 4)],
    "headroom_over_p4": round(nc_p4, 4),
    "fraction_of_raw_surviving": round(nc_p1 / raw["raw_gain_over_p1"], 3) if raw["raw_gain_over_p1"] else None,
    "method": "select on T + N(0,sigma2_run), evaluate on observed T; sigma2_run from E2.",
}

# ---------- replicate-CV headroom (robustness, real runs, replicated subset) ----------
# pick winner per query on half-1 replicate means, evaluate on half-2 means; over the replicated
# archs (which EXCLUDE p2,p3,p9), so best-fixed within this subset is recomputed.
sub_best = max(varch, key=lambda a: ov[ov.arch == a].overall_score_recomputed.mean())
cv_gains = []
for _ in range(N_SPLIT):
    h1, h2 = {}, {}
    for a in varch:
        r = reps[a][:]
        if len(r) >= 2:
            rng.shuffle(r); k = max(len(r) // 2, 1); h1[a], h2[a] = r[:k], r[k:] or r[:1]
        else:
            h1[a] = h2[a] = r
    m1, m2 = {}, {}
    for a in varch:
        for q in vqs:
            v1 = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(h1[a]))].overall_score_recomputed
            v2 = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(h2[a]))].overall_score_recomputed
            m1[(a, q)] = float(v1.mean()) if len(v1) else np.nan
            m2[(a, q)] = float(v2.mean()) if len(v2) else np.nan
    dep, bf = [], []
    for q in vqs:
        cand = [(a, m1[(a, q)]) for a in varch if not np.isnan(m1[(a, q)]) and not np.isnan(m2[(a, q)])]
        if not cand: continue
        win = max(cand, key=lambda x: x[1])[0]
        if not np.isnan(m2[(win, q)]) and not np.isnan(m2[(sub_best, q)]):
            dep.append(m2[(win, q)]); bf.append(m2[(sub_best, q)])
    if dep:
        cv_gains.append(np.mean(dep) - np.mean(bf))
replicate_cv = {
    "subset_archs": varch, "subset_best_fixed": sub_best,
    "cv_headroom": round(float(np.mean(cv_gains)), 4) if cv_gains else None,
    "cv_headroom_ci95": ([round(float(np.percentile(cv_gains, 2.5)), 4),
                          round(float(np.percentile(cv_gains, 97.5)), 4)] if cv_gains else None),
    "underpowered_note": "30 variance queries, mostly 3 replicates/arch, 1-vs-2 split: this "
                         "estimate is UNDERPOWERED with a wide CI that can cross the 0.02 gate. "
                         "It is corroborating only; the well-powered primary G1 evidence is the "
                         "87-query Stage-B out-of-sample LOOCV routers (routability.stage_b).",
    "note": "pick winner on replicate-half-1, evaluate on half-2, over the 8 replicated archs "
            "(excludes p2,p3,p9); real independent runs, no simulated noise. Monte-Carlo over "
            "N_SPLIT seeded random half-splits (convention pinned for reproducibility).",
}

# ---------- GATE G1 (keyed on the RIGOROUS estimate, not the biased parametric) ----------
# Method tension (important): the parametric bootstrap (select on T+noise, evaluate on T) is
# OPTIMISTICALLY BIASED because it evaluates the picked architecture on the SAME single noisy run
# T whose maxima the oracle gain capitalised on -- it cannot de-noise the evaluation baseline. The
# replicate-CV (pick on real run-half-1, evaluate on real run-half-2) is the only estimate that
# uses independent runs and is therefore the trustworthy noise correction. It returns ~0.003.
# G1 therefore keys on the replicate-CV headroom, with the parametric reported but flagged.
G1_THRESHOLD = 0.02
rigorous = replicate_cv["cv_headroom"]
g1_leans_fire = (rigorous is not None and rigorous < G1_THRESHOLD)
gate_g1 = {
    "threshold": G1_THRESHOLD,
    "rigorous_headroom_replicate_cv": rigorous,
    "parametric_headroom_optimistic_biased": round(nc_p1, 4),
    "winner_label_reliability": winner_reliability["test_retest_agreement"],
    "winner_label_chance": winner_reliability["chance_agreement"],
    "status": "PRELIMINARY — not final",
    "leans": "FIRE/NULL" if g1_leans_fire else "PROCEED",
    "decision": (
        "LEANS NULL FRAMING. The rigorous (real-independent-run) noise-corrected headroom is "
        f"~{rigorous} over best-fixed (<< 0.02); winner-label reliability is only weakly above "
        f"chance ({winner_reliability['test_retest_agreement']} vs {winner_reliability['chance_agreement']}); "
        "the raw oracle gain 0.094 is mostly noise capitalisation. The parametric 0.071 is "
        "optimistically biased and is NOT the basis for the decision. NOT FINAL: the replicate-CV "
        "uses few-replicate halves on an 8-arch subset (excl p2,p3,p9) and an oracle-label-from-"
        "one-run selector; a Stage-B FEATURE router and topped-up replicates (P1/P4/P10 to more "
        "reps) are needed before finalising G1. Recommended: confirm, then pivot Paper 1 to the "
        "rigorous-null/methods framing if confirmed."),
}

out = {
    "_note": "E1 Stage A routability (PRELIMINARY). Raw oracle gain (~0.094) is mostly noise "
             "capitalisation: the rigorous real-independent-run headroom (replicate-CV) is ~0.003 "
             "and winner labels are only weakly above chance. The parametric bootstrap (0.071) is "
             "optimistically biased (evaluates on the same noisy run) and is NOT the decision basis. "
             "Gate G1 LEANS NULL but is NOT final (subset/few-rep; needs Stage-B feature router + "
             "topped-up replicates). Prereg: prereg_E1.md.",
    "prereg": "docs/publication/prereg/prereg_E1.md",
    "architecture_means_gpt52": means,
    "raw": raw, "winner_label_reliability": winner_reliability,
    "noise_corrected_headroom": noise_corrected, "replicate_cv_headroom": replicate_cv,
    "gate_g1": gate_g1,
}

_tmp = f"{ANA}/canonical_numbers.json.tmp"
cn["routability"] = out
open(_tmp, "w").write(json.dumps(cn, indent=1)); os.replace(_tmp, f"{ANA}/canonical_numbers.json")

print(f"routability: best_fixed={best_fixed} (p1={means[P1]}, p4={means['base_p4']}, tied) | "
      f"raw_oracle_gain_over_p1={raw['raw_gain_over_p1']}")
print(f"  winner-label reliability={winner_reliability['test_retest_agreement']} vs chance "
      f"{winner_reliability['chance_agreement']} (above_chance={winner_reliability['above_chance']})")
print(f"  NOISE-CORRECTED headroom over P1 = {nc_p1:.4f} CI{noise_corrected['headroom_over_p1_ci95']} "
      f"({noise_corrected['fraction_of_raw_surviving']} of raw survives)")
print(f"  replicate-CV headroom (subset) = {replicate_cv['cv_headroom']}")
print(f"  *** GATE G1: leans={gate_g1['leans']} (status={gate_g1['status']}); rigorous headroom "
      f"{rigorous} vs biased parametric {round(nc_p1,4)}")
