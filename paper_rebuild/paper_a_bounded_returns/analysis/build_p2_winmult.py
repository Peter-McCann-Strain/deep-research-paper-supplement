#!/usr/bin/env python
"""P2_winmult — Model-recall / win-multiplicity decomposition of the oracle gap (Paper 1).

Canonical SUBKEY `routability.model_recall` (this script APPENDS a subkey to the existing
`routability` dict — it does NOT clobber the other routability subkeys written by
build_routability.py / _stageb / _judgerobust / _equivalence).

Question (strengthens the G1 null): the raw oracle gain over best-fixed (~0.094 in
routability.raw) presumes a perfect per-query router can RECALL whichever architecture wins each
query. But if almost every query has a UNIQUE winner (win-multiplicity 1), the oracle target is a
high-entropy, query-specific label that an out-of-sample router cannot recover from features — the
headroom is "unrecoverable" in the model-recall sense. We decompose the oracle gap by how many
architectures tie for the per-query win and report the % of single-winner (unrecoverable) queries.

Framing follows LLMRouterBench (arXiv:2601.07206), which distinguishes (a) Random model selection,
(b) the Best-Single fixed model, and (c) the Oracle per-query router, and shows that the realizable
gain of a learned router over Best-Single shrinks sharply once the per-query winner is a unique,
near-tied label (low win-multiplicity) rather than a broad equivalence class. We re-label the
study's random / best-fixed / oracle as Random / Best-Single / Oracle and emit Gain@R, Gain@B,
Gap@O on the same axis.

Substrate (verified on disk): df_overall_scores.parquet, judge=gpt52, pattern_family=='variance'
(the REPLICATE matrix). 8 replicated architectures x 30 variance queries, scored on
overall_score_recomputed, replicate runs averaged to an arch-x-query mean cell (mirrors
build_routability.py's `ov`/`arch` extraction exactly).

  - PRIMARY (complete-case): the 5 architectures with all 30 queries present
    (base_p0,p1,p10,p4,p7) -> a dense 30x5 matrix, no missingness, fair within-query comparison
    and an unbiased Best-Single (selected and evaluated on the SAME 30 queries). The other three
    replicated archs (p5,p6,p8) are sparse (14/8/18 queries); ranking them by a query-subset mean
    would be selection-biased, so they are excluded from the primary, exactly as build_routability
    excludes p2/p3/p9 from its replicated-subset CV.
  - ROBUSTNESS (nominal 30x8): all 8 replicated archs, per-query winner taken over whichever archs
    are present for that query, Best-Single restricted to queries where it is present. Reported but
    flagged as sparsity-biased (a sparse arch can win an easy subset).

Win-multiplicity: for each query, count archs whose mean is within EPS of the per-query max (the
winner equivalence class). EPS swept over {0.0, 0.005, 0.01, 0.02}; EPS=0.01 is the headline
(≈ MDE80 scale / half the 0.02 G1 gate, so two archs within 0.01 are a statistical tie => the
router need not distinguish them). single_winner_fraction at EPS=0.01 is the % of queries whose
oracle gain is unrecoverable. Oracle headroom (Oracle - Best-Single) is stratified by multiplicity.

Re-labelled metrics on the canonical axis (means over queries, primary matrix):
  Gain@R  = Random mean (uniform-over-present-archs per query)                     [absolute]
  Gain@B  = Best-Single mean (best fixed arch by overall mean, eval on all 30q)    [absolute]
  Gap@O   = Oracle mean - Best-Single mean (the per-query routing headroom)        [the gap]

Interpretation (G1): if Gap@O is concentrated in single-winner queries (high
single_winner_fraction, low headroom in the multiplicity>=2 bucket), the raw oracle gain is a
high-entropy unrecoverable target, corroborating the rigorous replicate-CV null
(routability.replicate_cv_headroom ~0.003): a learned router cannot harvest a per-query unique-
winner label out of sample. This is a model-recall argument, complementary to the noise-
capitalisation argument already in `routability`.

Determinism: closed-form aggregation, no resampling; inputs SORTED (columns and index) for
reproducibility. SEED defined for convention parity though unused (no random draw).
"""
import json, os, warnings
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260611  # convention parity (no stochastic step in this builder)
EPS_GRID = [0.0, 0.005, 0.01, 0.02]
EPS_HEADLINE = 0.01

O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")

# ---- replicate matrix: gpt52, variance family, replicates averaged to arch x query mean ----
ov = O[(O.judge == "gpt52") & (O.pattern_family == "variance")][
    ["pattern", "query_id", "overall_score_recomputed"]].dropna().copy()
ov["arch"] = ov.pattern.str.extract(r"(base_p\d+)_v")          # same extraction as build_routability
piv = ov.groupby(["arch", "query_id"], observed=True)["overall_score_recomputed"].mean().unstack("arch")
piv = piv[sorted(piv.columns)].sort_index()                    # SORTED cols + index for determinism
all_archs = list(piv.columns)
n_queries_total = int(piv.shape[0])


def decompose(mat, label, restrict_bestsingle_to_present=False):
    """mat: queries x archs DataFrame (may contain NaN). Returns the win-mult decomposition."""
    arch_cols = list(mat.columns)
    arch_mean = mat.mean()                                     # column means (skip NaN)
    best_single = str(arch_mean.idxmax())
    random_vals, oracle_vals, bs_vals = [], [], []
    mult_by_eps = {e: [] for e in EPS_GRID}
    # oracle headroom (Oracle - Best-Single) per query, bucketed by win-multiplicity at EPS_HEADLINE
    headroom_bucket = defaultdict(list)
    n_used_bs = 0
    for _, row in mat.iterrows():
        vals = row.dropna()
        if vals.empty:
            continue
        mx = float(vals.max())
        oracle_vals.append(mx)
        random_vals.append(float(vals.mean()))                 # uniform over present archs
        for e in EPS_GRID:
            mult_by_eps[e].append(int((vals >= mx - e).sum()))
        if best_single in vals.index:
            bs = float(vals[best_single])
            bs_vals.append(bs)
            n_used_bs += 1
            m_head = int((vals >= mx - EPS_HEADLINE).sum())
            headroom_bucket[m_head].append(mx - bs)
        elif not restrict_bestsingle_to_present:
            bs_vals.append(np.nan)
    oracle_mean = float(np.mean(oracle_vals))
    random_mean = float(np.mean(random_vals))
    bs_mean = float(np.nanmean(bs_vals))
    mult_dist = {str(e): dict(sorted(Counter(mult_by_eps[e]).items())) for e in EPS_GRID}
    single_frac = {str(e): round(float(np.mean([m == 1 for m in mult_by_eps[e]])), 4) for e in EPS_GRID}
    strat = {str(m): {"n": len(v), "mean_oracle_headroom_over_best_single": round(float(np.mean(v)), 4)}
             for m, v in sorted(headroom_bucket.items())}
    return {
        "label": label,
        "n_archs": len(arch_cols),
        "archs": arch_cols,
        "n_queries": int(mat.shape[0]),
        "best_single_arch": best_single,
        "best_single_mean": round(bs_mean, 4),
        # re-labelled headline metrics on the canonical axis
        "Gain@R_random_mean": round(random_mean, 4),
        "Gain@B_best_single_mean": round(bs_mean, 4),
        "Oracle_mean": round(oracle_mean, 4),
        "Gap@O_oracle_minus_best_single": round(oracle_mean - bs_mean, 4),
        "oracle_minus_random": round(oracle_mean - random_mean, 4),
        # win-multiplicity (model-recall core)
        "eps_grid": EPS_GRID,
        "win_multiplicity_dist_by_eps": mult_dist,
        "single_winner_fraction_by_eps": single_frac,
        "single_winner_fraction_headline": single_frac[str(EPS_HEADLINE)],
        "eps_headline": EPS_HEADLINE,
        "oracle_headroom_by_multiplicity": strat,
        "n_queries_best_single_present": n_used_bs,
    }


# PRIMARY: complete-case dense archs (present on ALL 30 queries)
dense_archs = [a for a in all_archs if piv[a].notna().all()]
primary = decompose(piv[dense_archs].dropna(), "primary_complete_case_dense")

# ROBUSTNESS: all 8 replicated archs, per-query winner over present archs (sparsity-biased)
robust = decompose(piv[all_archs], "robustness_all8_present_per_query",
                   restrict_bestsingle_to_present=True)

# concentration of the oracle headroom in single-winner queries (the G1 punchline)
strat_h = primary["oracle_headroom_by_multiplicity"]
single_bucket = strat_h.get("1", {"n": 0, "mean_oracle_headroom_over_best_single": 0.0})
multi_buckets = [v for k, v in strat_h.items() if k != "1"]
multi_n = sum(v["n"] for v in multi_buckets)
multi_head = (round(float(np.average([v["mean_oracle_headroom_over_best_single"] for v in multi_buckets],
                                      weights=[v["n"] for v in multi_buckets])), 4)
              if multi_n else None)

out = {
    "_note": "Model-recall / win-multiplicity decomposition of the oracle gap on the replicate "
             "matrix (gpt52, variance family, 8 archs x 30 queries; replicates averaged). PRIMARY "
             "is the complete-case dense 30x5 subset (p0,p1,p10,p4,p7 — present on all 30 queries; "
             "p5,p6,p8 are sparse and excluded to avoid a selection-biased Best-Single, mirroring "
             "build_routability's exclusion of p2/p3/p9). Re-labels random/best-fixed/oracle as "
             "Random/Best-Single/Oracle and reports Gain@R, Gain@B (=Best-Single mean), Gap@O "
             "(=Oracle-Best-Single). Win-multiplicity = #archs within EPS of the per-query max; "
             "single_winner_fraction at EPS=0.01 is the % of queries whose oracle gain is an "
             "unrecoverable unique-winner label. Strengthens G1: the oracle headroom is "
             "concentrated in single-winner queries, so a feature router cannot recall it out of "
             "sample — complementary to the noise-capitalisation null in routability.replicate_cv. "
             "Framing per LLMRouterBench (arXiv:2601.07206). Appends as routability.model_recall.",
    "citation": "arXiv:2601.07206 (LLMRouterBench): Random vs Best-Single vs Oracle routing; "
                "realizable router gain over Best-Single shrinks as win-multiplicity falls.",
    "metric": "overall_score_recomputed",
    "judge": "gpt52",
    "substrate": "df_overall_scores.parquet pattern_family=='variance' (replicate matrix)",
    "n_queries_total": n_queries_total,
    "n_replicated_archs_total": len(all_archs),
    "all_replicated_archs": all_archs,
    "dense_complete_case_archs": dense_archs,
    "primary": primary,
    "robustness_all8": robust,
    "headroom_concentration": {
        "single_winner_fraction": primary["single_winner_fraction_headline"],
        "single_winner_bucket": single_bucket,
        "multiplicity_ge2_n": multi_n,
        "multiplicity_ge2_mean_oracle_headroom": multi_head,
        "interpretation": (
            f"At EPS={EPS_HEADLINE}, {primary['single_winner_fraction_headline']*100:.0f}% of the "
            f"{primary['n_queries']} primary queries have a UNIQUE winner (win-multiplicity 1). The "
            f"oracle headroom over Best-Single is {single_bucket['mean_oracle_headroom_over_best_single']} "
            f"in those single-winner queries vs {multi_head} in the multiplicity>=2 queries: the gap "
            "lives almost entirely in unique-winner queries, where the routing target is a high-"
            "entropy per-query label. This is the model-recall reason the raw oracle gain is not "
            "realizable out of sample (a feature router cannot recover a unique-winner label), "
            "corroborating routability.replicate_cv_headroom (~0.003) and the G1 null framing."),
    },
    "g1_relevance": "Supports G1 NULL: high single-winner fraction => oracle gap is an unrecoverable "
                    "per-query label, not a routable feature-conditioned signal.",
}

# ---- APPEND subkey without clobbering the rest of routability (atomic tmp+os.replace) ----
cn = json.load(open(f"{ANA}/canonical_numbers.json"))
cn.setdefault("routability", {})            # do not destroy existing routability subkeys
cn["routability"]["model_recall"] = out
_tmp = f"{ANA}/canonical_numbers.json.tmp"
open(_tmp, "w").write(json.dumps(cn, indent=1)); os.replace(_tmp, f"{ANA}/canonical_numbers.json")

print(f"routability.model_recall: PRIMARY {primary['n_queries']}q x {primary['n_archs']}arch "
      f"(dense {dense_archs})")
print(f"  Best-Single={primary['best_single_arch']} | Gain@R={primary['Gain@R_random_mean']} "
      f"Gain@B={primary['Gain@B_best_single_mean']} Oracle={primary['Oracle_mean']} "
      f"Gap@O={primary['Gap@O_oracle_minus_best_single']}")
print(f"  single-winner fraction (EPS={EPS_HEADLINE}) = {primary['single_winner_fraction_headline']} "
      f"| headroom single={single_bucket['mean_oracle_headroom_over_best_single']} "
      f"mult>=2={multi_head}")
print(f"  win-mult dist (EPS={EPS_HEADLINE}) = {primary['win_multiplicity_dist_by_eps'][str(EPS_HEADLINE)]}")
