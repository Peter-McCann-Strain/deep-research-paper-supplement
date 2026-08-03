#!/usr/bin/env python
"""Per-dimension paired query-bootstrap CIs for the matched-budget disentanglement probe.

The store's disentanglement.p1_arm.per_dimension carries BARE point estimates of the
per-dimension P1-vs-P0 gap under full budget (unmatched_gap) and under the budget clamp
(matched_gap), n=29 queries, judge gpt52. The paper narrates these (e.g. citation_quality
+0.155 -> -0.103, analytical_depth +0.086 -> +0.198) without interval evidence. This
builder adds, for ALL NINE dimensions:
  * unmatched gap  (base_p1 - base_p0)        : paired query-bootstrap 95% CI
  * matched gap    (matched_p1 - base_p0)     : paired query-bootstrap 95% CI
  * change in gap  (matched_p1 - base_p1)     : paired query-bootstrap 95% CI
    (== matched_gap - unmatched_gap, the per-dimension clamp effect)
plus exact Wilcoxon signed-rank p-values for each contrast. Bootstrap resamples QUERIES
(the cluster unit); all three contrasts are recomputed on the SAME resampled query set
each rep, so the CIs are mutually consistent. Deterministic seed.

Data/method identical to build_disentanglement.py (df_scores.parquet, judge gpt52,
Qp1 = queries with a disentangle_matched_p1 report). APPEND-ONLY: lands the NEW subkey
canonical_numbers.json['disentanglement']['p1_arm']['per_dimension_ci']; refuses to
overwrite; atomic tempfile+replace write; never touches sibling keys.

Usage: python build_disentanglement_perdim_ci.py [--write] [--force]
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260702
N_BOOT = 10000

V = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")
SC = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
for d in (V, SC):
    d["pattern"] = d["pattern"].astype(str)
    d["judge"] = d["judge"].astype(str)

rng = np.random.default_rng(SEED)

dis = V[V.pattern_family == "disentanglement"]
Qp1 = sorted(dis[dis.pattern == "disentangle_matched_p1"].query_id.unique())

DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization",
        "instruction_following", "attribution_quality"]


def sc_q(pat, qs, dim):
    d = SC[(SC.judge == "gpt52") & (SC.pattern == pat)
           & (SC.query_id.isin(qs)) & (SC.dimension == dim)]
    return d.set_index("query_id")["score"]


def wilcox_p(x, y):
    try:
        return round(float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue), 4)
    except Exception:
        return None


def one_dim(dim):
    p0 = sc_q("base_p0", Qp1, dim)
    b1 = sc_q("base_p1", Qp1, dim)
    m1 = sc_q("disentangle_matched_p1", Qp1, dim)
    common = sorted(set(p0.index) & set(b1.index) & set(m1.index))
    p0v = p0.reindex(common).to_numpy()
    b1v = b1.reindex(common).to_numpy()
    m1v = m1.reindex(common).to_numpy()
    n = len(common)
    du = b1v - p0v          # unmatched gap per query
    dm = m1v - p0v          # matched (clamped) gap per query
    dc = m1v - b1v          # change in gap = clamp effect per query
    boots = np.empty((N_BOOT, 3))
    for r in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[r] = du[idx].mean(), dm[idx].mean(), dc[idx].mean()

    def block(point, col, p_w):
        lo, hi = np.percentile(boots[:, col], [2.5, 97.5])
        return {"gap": round(float(point), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "wilcoxon_p": p_w}

    return n, {
        "unmatched": block(du.mean(), 0, wilcox_p(b1v, p0v)),
        "matched": block(dm.mean(), 1, wilcox_p(m1v, p0v)),
        "clamp_change": block(dc.mean(), 2, wilcox_p(m1v, b1v)),
    }


out = {
    "_note": ("Paired query-bootstrap 95%% CIs (percentile, %d reps, seed %d, resampling "
              "queries; all contrasts recomputed on the same resample) for the per-dimension "
              "P1-vs-P0 gaps of the matched-budget probe, judge gpt52, plus exact Wilcoxon "
              "signed-rank p-values. 'unmatched' = base_p1 - base_p0 (full budget); 'matched' "
              "= disentangle_matched_p1 - base_p0 (budget clamped); 'clamp_change' = "
              "disentangle_matched_p1 - base_p1 (per-dimension clamp effect, = matched - "
              "unmatched). Point gaps reproduce disentanglement.p1_arm.per_dimension. Nine "
              "dimensions x three contrasts at n~29: intervals are wide and NOT multiplicity-"
              "corrected; read as per-dimension description, not certification."
              % (N_BOOT, SEED)),
    "seed": SEED,
    "n_boot": N_BOOT,
    "judge": "gpt52",
    "dimensions": {},
}

n_common = None
for dim in DIMS:
    n, blocks = one_dim(dim)
    n_common = n if n_common is None else n_common
    out["dimensions"][dim] = {"n": n, **blocks}

ad = out["dimensions"]["analytical_depth"]
out["headline_check"] = {
    "analytical_depth_matched_gap": ad["matched"]["gap"],
    "analytical_depth_matched_ci95": ad["matched"]["ci95"],
    "analytical_depth_matched_excludes_zero": ad["matched"]["excludes_zero"],
    "_note": "Direct answer to whether the +0.198 clamped analytical_depth gap excludes zero.",
}

# ---- consistency check vs the landed point estimates ----
cn = json.load(open(CANON))
landed = cn["disentanglement"]["p1_arm"]["per_dimension"]
mismatch = []
for dim in DIMS:
    if dim in landed:
        if abs(landed[dim]["unmatched_gap"] - out["dimensions"][dim]["unmatched"]["gap"]) > 1e-9 \
           or abs(landed[dim]["matched_gap"] - out["dimensions"][dim]["matched"]["gap"]) > 1e-9:
            mismatch.append(dim)
if mismatch:
    print(f"[WARN] point-gap mismatch vs landed per_dimension for: {mismatch}")
else:
    print("[OK] point gaps reproduce disentanglement.p1_arm.per_dimension exactly")

print(f"n common queries = {n_common}")
for dim in DIMS:
    b = out["dimensions"][dim]
    print(f"  {dim:22s} unm {b['unmatched']['gap']:+.4f} {b['unmatched']['ci95']} | "
          f"mat {b['matched']['gap']:+.4f} {b['matched']['ci95']} "
          f"{'*' if b['matched']['excludes_zero'] else ' '} | "
          f"chg {b['clamp_change']['gap']:+.4f} {b['clamp_change']['ci95']} "
          f"{'*' if b['clamp_change']['excludes_zero'] else ' '}")

if "--write" in sys.argv:
    key_parent = cn["disentanglement"]["p1_arm"]
    if "per_dimension_ci" in key_parent and "--force" not in sys.argv:
        print("[REFUSING] disentanglement.p1_arm.per_dimension_ci already exists (use --force)")
        sys.exit(1)
    key_parent["per_dimension_ci"] = out
    fd, tmp = tempfile.mkstemp(dir=ANA, prefix="canonical_numbers.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[WROTE canonical_numbers.json disentanglement.p1_arm.per_dimension_ci "
          f"(store {len(cn)} top-level keys preserved)]")
else:
    print("[DRY-RUN: no write; pass --write to land the key]")
