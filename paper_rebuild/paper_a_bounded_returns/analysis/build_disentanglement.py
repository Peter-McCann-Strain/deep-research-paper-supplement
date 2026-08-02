#!/usr/bin/env python
"""Compute-vs-architecture disentanglement probe (audit M1/RA3).

"Bounded returns to orchestration" holds the tool INTERFACE fixed but not the compute
(token/tool-call) BUDGET, which co-varies ~17x with architecture. The released data contains
a matched-budget probe (pattern_family=='disentanglement': disentangle_matched_p1/p4,
gpt52-only) that was never reported. This pairs, by query:
  - UNMATCHED architecture gap:  base_pX  - base_p0   (full budget)
  - MATCHED-budget gap:          matched_pX - base_p0 (budget clamped ~12x->3.2x P0)
and decomposes the P1>P0 advantage into a compute share (gap erased by clamping) and an
architecture residual (gap surviving), with a per-dimension breakdown. The P4 arm is
under-covered (9/30 queries, no base gap) and reported but not used for the headline.

Appends canonical_numbers.json['disentanglement'].
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
V = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")
OV = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
SC = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
RUNS = pd.read_parquet(f"{ROOT}/data/analysis/df_runs.parquet")
for d in (V, OV, SC):
    d["pattern"] = d["pattern"].astype(str); d["judge"] = d["judge"].astype(str)
rng = np.random.default_rng(20260610)

dis = V[V.pattern_family == "disentanglement"]
Qp1 = sorted(dis[dis.pattern == "disentangle_matched_p1"].query_id.unique())
Qp4 = sorted(dis[dis.pattern == "disentangle_matched_p4"].query_id.unique())


def ov_by_q(pat, qs):
    d = OV[(OV.judge == "gpt52") & (OV.pattern == pat) & (OV.query_id.isin(qs))]
    return d.set_index("query_id")["overall_score"]


def paired(a_pat, b_pat, qs):
    a = ov_by_q(a_pat, qs); b = ov_by_q(b_pat, qs)
    common = sorted(set(a.index) & set(b.index))
    a = a.reindex(common); b = b.reindex(common)
    diff = (a - b).to_numpy(); n = len(diff)
    boots = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(10000)])
    try:
        wp = float(stats.wilcoxon(a.to_numpy(), b.to_numpy(), zero_method="wilcox").pvalue)
    except Exception:
        wp = float("nan")
    return {"n": n, "delta": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "wilcoxon_p": round(wp, 4), "significant": bool(wp < 0.05)}


def sc_q(pat, qs, dim):
    d = SC[(SC.judge == "gpt52") & (SC.pattern == pat) & (SC.query_id.isin(qs)) & (SC.dimension == dim)]
    return d.set_index("query_id")["score"]


DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth", "citation_quality",
        "logical_coherence", "organization", "instruction_following", "attribution_quality"]


def compute_pct_ratio_ci(unm_pat, mat_pat, base_pat, qs, reps=10000):
    """Bootstrap CI on the compute share = (gu-gm)/gu, resampling QUERIES (the cluster
    unit). gu and gm are recomputed on the SAME resampled query set each rep so the ratio
    propagates the joint sampling noise of the two correlated, paired gaps."""
    u = ov_by_q(unm_pat, qs); m = ov_by_q(mat_pat, qs); b0 = ov_by_q(base_pat, qs)
    common = sorted(set(u.index) & set(m.index) & set(b0.index))
    du = (u.reindex(common) - b0.reindex(common)).to_numpy()   # base_p1 - base_p0 per query
    dm = (m.reindex(common) - b0.reindex(common)).to_numpy()   # matched_p1 - base_p0 per query
    n = len(common)
    pcts = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        gu_b, gm_b = du[idx].mean(), dm[idx].mean()
        if gu_b != 0:
            pcts.append((gu_b - gm_b) / gu_b * 100.0)
    pcts = np.asarray(pcts)
    return {"n": n,
            "point": round((du.mean() - dm.mean()) / du.mean() * 100, 1) if du.mean() else None,
            "ci95": [round(float(np.percentile(pcts, 2.5)), 1), round(float(np.percentile(pcts, 97.5)), 1)],
            "ci_width": round(float(np.percentile(pcts, 97.5) - np.percentile(pcts, 2.5)), 1),
            "_note": "Ratio of two noisy paired gaps; CI is wide and may span 0-100%+ at small n, "
                     "so the point split is NOT interpretable as a precise 40/60 attribution."}


def signflip_p(a_pat, b_pat, qs, reps=20000):
    """Wild/sign-flip (permutation) robustness for the direct clamp effect on small n:
    randomly flip the sign of each paired difference and count how often |mean| >= observed.
    Distribution-free, valid under the exchangeable-sign null even at n~29."""
    a = ov_by_q(a_pat, qs); b = ov_by_q(b_pat, qs)
    common = sorted(set(a.index) & set(b.index))
    d = (a.reindex(common) - b.reindex(common)).to_numpy()
    obs = abs(d.mean()); n = len(d)
    cnt = 0
    for _ in range(reps):
        signs = rng.choice([-1.0, 1.0], n)
        if abs((d * signs).mean()) >= obs - 1e-12:
            cnt += 1
    return {"n": n, "observed_delta": round(float(d.mean()), 4),
            "signflip_p": round((cnt + 1) / (reps + 1), 4), "reps": reps}


def perdim(qs):
    out = {}
    for dim in DIMS:
        p0 = sc_q("base_p0", qs, dim); b1 = sc_q("base_p1", qs, dim); m1 = sc_q("disentangle_matched_p1", qs, dim)
        common = sorted(set(p0.index) & set(b1.index) & set(m1.index))
        if not common:
            continue
        p0m, b1m, m1m = p0.reindex(common).mean(), b1.reindex(common).mean(), m1.reindex(common).mean()
        out[dim] = {"unmatched_gap": round(float(b1m - p0m), 4), "matched_gap": round(float(m1m - p0m), 4)}
    return out


# budget spread from df_runs (cost proxy + tokens)
def runs_metric(pat, col):
    d = RUNS[RUNS.pattern.astype(str) == pat]
    return float(d[col].mean()) if len(d) and col in d.columns else None


costcol = next((c for c in ["cost_proxy_usd", "cost_usd", "cost_proxy", "blended_cost_usd"] if c in RUNS.columns), None)
tokcol = next((c for c in ["total_tokens", "tokens", "n_tokens"] if c in RUNS.columns), None)
budget = {}
if costcol:
    c0 = runs_metric("base_p0", costcol); c4 = runs_metric("base_p4", costcol); c1 = runs_metric("base_p1", costcol)
    budget["cost_col"] = costcol
    budget["base_p0"] = round(c0, 4) if c0 else None
    budget["base_p1"] = round(c1, 4) if c1 else None
    budget["base_p4"] = round(c4, 4) if c4 else None
    budget["p4_over_p0_ratio"] = round(c4 / c0, 1) if c0 and c4 else None
    budget["matched_p1"] = round(runs_metric("disentangle_matched_p1", costcol), 4) if runs_metric("disentangle_matched_p1", costcol) else None

p1 = {"unmatched": paired("base_p1", "base_p0", Qp1), "matched": paired("disentangle_matched_p1", "base_p0", Qp1),
      "clamp_effect": paired("disentangle_matched_p1", "base_p1", Qp1), "per_dimension": perdim(Qp1)}
gu, gm = p1["unmatched"]["delta"], p1["matched"]["delta"]
# Raw point split kept for reference but explicitly de-emphasised: it is a ratio of two noisy
# point estimates and the headline must NOT claim "compute explains ~40%" from it.
p1["compute_attributable_pct_point"] = round((gu - gm) / gu * 100) if gu else None
p1["architecture_residual_pct_point"] = round(gm / gu * 100) if gu else None
# (a) bootstrap CI on the compute-share RATIO (resample queries, recompute ratio)
p1["compute_attributable_pct_ci"] = compute_pct_ratio_ci(
    "base_p1", "disentangle_matched_p1", "base_p0", Qp1)
# (c) sign-flip / wild robustness on the DIRECT clamp effect (small n)
p1["clamp_effect_signflip"] = signflip_p("disentangle_matched_p1", "base_p1", Qp1)
# (b) the direct test is the primary, honest evidence: borderline, NOT significant at 0.05
ce = p1["clamp_effect"]
p1["primary_inference"] = (
    "The compute contribution is tested DIRECTLY (matched_p1 - base_p1): delta=%+.4f, "
    "Wilcoxon p=%.3f, sign-flip p=%.3f -> BORDERLINE and NOT significant at 0.05. "
    "The %d%%/%d%% point split is a ratio of two noisy paired gaps with a wide bootstrap CI "
    "%s and must not be read as a precise attribution."
    % (ce["delta"], ce["wilcoxon_p"], p1["clamp_effect_signflip"]["signflip_p"],
       p1["compute_attributable_pct_point"], p1["architecture_residual_pct_point"],
       p1["compute_attributable_pct_ci"]["ci95"]))

p4 = {"unmatched": paired("base_p4", "base_p0", Qp4), "matched": paired("disentangle_matched_p4", "base_p0", Qp4),
      "_note": "P4 arm under-covered (9/30 queries) and shows no base gap; reported for completeness, not used for the headline."}

res = {
    "_note": "Matched-budget disentanglement probe (gpt52). Tests whether the P1>P0 overall-score advantage "
             "survives clamping the token/tool-call budget. PRIMARY evidence is the DIRECT clamp effect "
             "(matched_p1 - base_p1), not the within-arm significance flip; the point compute/architecture "
             "split is a noisy ratio reported with a bootstrap CI and a sign-flip robustness check.",
    "budget_spread": budget,
    "p1_arm": p1,
    "p4_arm": p4,
    "headline": ("Full-budget P1>P0 by %+.4f (Wilcoxon p=%.3f); clamping the budget reduces the gap to %+.4f "
                 "(p=%.3f). The compute contribution is tested DIRECTLY (matched_p1 - base_p1 = %+.4f, p=%.3f, "
                 "sign-flip p=%.3f): BORDERLINE, not significant at 0.05. The point split is ~%d%% compute / "
                 "~%d%% architecture but its bootstrap CI is wide (%s), so we do NOT claim a precise "
                 "attribution; directionally, the surviving residual is analytical_depth and the attenuated "
                 "component is the retrieval-bound dimensions (citation, coverage).")
    % (gu, p1["unmatched"]["wilcoxon_p"], gm, p1["matched"]["wilcoxon_p"],
       p1["clamp_effect"]["delta"], p1["clamp_effect"]["wilcoxon_p"],
       p1["clamp_effect_signflip"]["signflip_p"],
       p1["compute_attributable_pct_point"], p1["architecture_residual_pct_point"],
       p1["compute_attributable_pct_ci"]["ci95"]),
}

import sys
if "--write" in sys.argv:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    cn["disentanglement"] = res
    json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
    print("[WROTE canonical_numbers.json['disentanglement']]")
else:
    print("[DRY-RUN: no write; pass --write to update canonical_numbers.json]")
print(json.dumps({k: res[k] for k in ["budget_spread", "headline"]}, indent=1))
print("compute_pct_ci:", p1["compute_attributable_pct_ci"])
print("clamp_effect:  ", p1["clamp_effect"])
print("clamp_signflip:", p1["clamp_effect_signflip"])
print("P1 unmatched:", p1["unmatched"]); print("P1 matched:  ", p1["matched"])
print("per-dim (unmatched->matched):")
for d, v in p1["per_dimension"].items():
    print(f"  {d:22s} {v['unmatched_gap']:+.3f} -> {v['matched_gap']:+.3f}")
