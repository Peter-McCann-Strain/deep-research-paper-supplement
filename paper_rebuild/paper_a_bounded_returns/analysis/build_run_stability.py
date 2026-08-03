#!/usr/bin/env python
"""Run-to-run stability of the rankings (answers the 'single-run artifact' objection).

For the four patterns with stochastic replicates (base_p{0,1,4,10}_v{1,2,3}, GPT-5.2),
report the within-query standard deviation across the three runs and the rank stability
(Spearman of per-pattern means across runs). The within-query SD is small relative to the
P0->cluster gaps, so the cluster is not a sampling artifact.

Uses `overall_score_recomputed` (the rubric-reweighted score), consistent with every
other variance builder in the suite (build_variance_decomposition,
build_p2_var_bootstrap, build_variance_3way_sonnet). The raw-vs-recomputed choice is
now consistent across the variance suite.
Appends canonical_numbers.json['run_stability'].
"""
import pandas as pd, numpy as np, json, warnings
from itertools import combinations
warnings.filterwarnings("ignore")
ROOT = "."; A = f"{ROOT}/data/analysis"
ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
g = ov[ov.judge.eq("gpt52")].copy()
PATS = ["p0", "p1", "p4", "p10"]
out = {}
for p in PATS:
    reps = {}
    for v in (1, 2, 3):
        d = g[g.pattern.eq(f"base_{p}_v{v}")]
        reps[v] = {q: float(s) for q, s in zip(d.query_id, d.overall_score_recomputed) if pd.notna(s)}
    common = set(reps[1]) & set(reps[2]) & set(reps[3])
    if not common:
        out[p] = {"_note": "no common replicate queries"}; continue
    # within-query SD across the 3 runs, averaged over queries
    sds = [np.std([reps[v][q] for v in (1, 2, 3)], ddof=1) for q in common]
    means = {v: np.mean([reps[v][q] for q in common]) for v in (1, 2, 3)}
    out[p] = {"n_common": len(common),
              "within_query_sd_mean": round(float(np.mean(sds)), 4),
              "within_query_sd_max": round(float(np.max(sds)), 4),
              "run_means": {f"v{v}": round(float(means[v]), 4) for v in (1, 2, 3)},
              "run_mean_spread": round(float(max(means.values()) - min(means.values())), 4)}

# rank stability across runs: build per-run pattern-mean vectors, Spearman between runs
vecs = {}
for v in (1, 2, 3):
    vecs[v] = {p: np.mean([float(s) for q, s in
                           zip(g[g.pattern.eq(f'base_{p}_v{v}')].query_id,
                               g[g.pattern.eq(f'base_{p}_v{v}')].overall_score_recomputed) if pd.notna(s)])
               for p in PATS}
from scipy.stats import spearmanr
sp = []
for a, b in combinations((1, 2, 3), 2):
    xs = [vecs[a][p] for p in PATS]; ys = [vecs[b][p] for p in PATS]
    sp.append(float(spearmanr(xs, ys).correlation))
out["rank_spearman_between_runs_mean"] = round(float(np.mean(sp)), 4)
out["pooled_within_query_sd"] = round(float(np.mean(
    [out[p]["within_query_sd_mean"] for p in PATS if "within_query_sd_mean" in out[p]])), 4)

p = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
p["run_stability"] = out
json.dump(p, open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
