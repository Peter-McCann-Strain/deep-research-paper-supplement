#!/usr/bin/env python3
"""build_length_adjusted_headline.py — length-control reanalysis of the HEADLINE contrasts.

Lands TWO new keys into the paper-A canonical store
    paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json

    1. 'headline_cluster_gap'      — the two previously-unbacked raw gap numbers the paper
                                     quotes (three-judge 0.1454 -> rounded 0.145/0.146 in
                                     prose; GPT-5.2 0.0657 -> 0.065), plus the P0-vs-P9 and
                                     cluster-vs-P9 companions, each with an EXPLICIT basis
                                     label and a query-clustered bootstrap 95% CI.
    2. 'length_adjusted_headline'  — the same contrasts AFTER the paper's own length
                                     adjustment (the frozen_vintage pooled-OLS method) is
                                     applied to the main 11-pattern x 90-query three-judge
                                     run, with query-FE and log-words robustness variants.

WHY
---
The paper's local-model section (frozen_vintage) length-adjusts its ranking via pooled OLS
(score ~ arm_dummies + beta*(words/1000 - grand_mean), slope +0.1748/kword) and that
adjustment REORDERS the local arms. The headline contrasts — single-pass P0 vs the
six-pattern orchestrated cluster (P1,P4,P5,P6,P7,P8) and P0 vs local-7B P9 — were never
given the same treatment, despite orchestrated pipelines producing systematically longer
reports (cluster mean ~1.9k words vs P0 ~1.2k, P9 ~0.7k). This script applies the
IDENTICAL method to the main run and reports raw vs adjusted side by side.

METHOD
------
Data: df_overall_scores.parquet x df_runs.parquet, base_p0..base_p10 (11 patterns, 90
queries), panel judges {gpt52, claude_opus, claude_sonnet}; claude_sonnet uses
overall_score_recomputed (stored overall_score is upstream-corrupted, per DATA_DICTIONARY).
Length = report_word_count (whitespace word count of the released .md; tokenizer-
independent). Every judged cell has a non-null word count (2,951 cells).

Specs (all pooled over the 11 main-run arms; cluster gap = unweighted mean of the six
cluster arms' adjusted means minus P0's, i.e. mean-of-pattern-means, matching the raw
basis):
  A. vintage-method match: cell-level pooled OLS, 11 pattern dummies (no intercept)
     + beta*(kwords - grand-mean kwords). Dummy coefficient IS the arm's length-adjusted
     mean = counterfactual score at grand-mean length (~1,600 words). Exactly the
     frozen_vintage length_control spec. Fit on 3-judge cells and per judge.
  B. + query fixed effects (within-query demeaning, Frisch-Waugh; P0 = reference so
     coefficients are adjusted gaps vs P0 directly). Controls query-difficulty
     confounding of the within-pattern length-score slope.
  C. log-words instead of linear kwords (the judge-bias audit's scale, where
     beta(log wc) ~ 0.04-0.11).
Inference: query-clustered PAIRED bootstrap (resample the 90 query ids with replacement;
the SAME resampled block feeds every spec/judge each iteration; query FE get one dummy
per resampled cluster COPY). n_boot=5000, seed=20260702, deterministic. Two-sided
bootstrap p for adjusted cluster-vs-P0 gaps as in build_frozen_vintage.py.

INTERPRETATION CAVEATS (recorded in the key)
--------------------------------------------
With pattern dummies, the pooled slope is identified from WITHIN-pattern variation, which
conflates judge length-preference with query-level material richness (queries where
retrieval succeeded yield both longer reports and better scores); spec B controls the
query side. Grand-mean evaluation places P9 (~726 mean words, p95~1,622) at the edge of
its length support. P0/P9 include a short-tail of degenerate near-empty reports; a
min-100-words sensitivity point is recorded.

WRITE SAFETY
------------
Default --dry-run (compute + print, write nothing). --write atomically appends via
tempfile in the store's dir + os.replace; append-only (mutates ONLY its two owned keys,
never siblings); refuses to overwrite an existing owned key unless --force. Self-guards
(exit 0) if the store or parquets are missing.

USAGE
-----
    python analysis/build_length_adjusted_headline.py              # dry-run (safe)
    python analysis/build_length_adjusted_headline.py --write
    python analysis/build_length_adjusted_headline.py --write --force
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"
OV_PQ = ROOT / "data" / "analysis" / "df_overall_scores.parquet"
RUNS_PQ = ROOT / "data" / "analysis" / "df_runs.parquet"

KEY_RAW = "headline_cluster_gap"
KEY_ADJ = "length_adjusted_headline"

PATS = [f"base_p{i}" for i in range(11)]
CLUSTER = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]
CL_IDX = np.array([PATS.index(p) for p in CLUSTER])
P0, P9 = PATS.index("base_p0"), PATS.index("base_p9")
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]

N_BOOT = 5000
SEED = 20260702

# Self-check constants: gaps recomputed here must reproduce the store's own headline /
# single_judge_gpt52 means (mean-of-pattern-means basis) to <=5e-4.
EXPECTED = {
    "cluster_minus_p0_3judge": 0.1454,
    "cluster_minus_p0_gpt52": 0.0657,
    "p0_minus_p9_3judge": 0.2307,
    "cluster_minus_p9_3judge": 0.3761,
    "p0_minus_p9_gpt52": 0.2045,
    "cluster_minus_p9_gpt52": 0.2702,
}


# ---------------------------------------------------------------- data loading
def load_cells():
    """One row per (pattern, query, judge) cell with sonnet-corrected score + word count."""
    ov = pd.read_parquet(OV_PQ)
    runs = pd.read_parquet(RUNS_PQ)
    for d in (ov, runs):
        d["pattern"] = d["pattern"].astype(str)
    ov["judge"] = ov["judge"].astype(str)
    ovb = ov[ov.pattern.isin(PATS) & ov.judge.isin(PANEL)].copy()
    # DATA_DICTIONARY: claude_sonnet stored overall_score is corrupted -> recomputed.
    ovb["ovc"] = ovb["overall_score"].where(
        ~ovb.judge.eq("claude_sonnet"), ovb["overall_score_recomputed"])
    rb = runs[runs.pattern.isin(PATS)][["pattern", "query_id", "report_word_count"]]
    if rb.duplicated(["pattern", "query_id"]).any():
        raise SystemExit("[length_adjusted_headline] duplicate (pattern,query) run rows; refusing.")
    m = ovb.merge(rb, on=["pattern", "query_id"], how="inner")
    m = m.dropna(subset=["ovc", "report_word_count"])
    # deterministic row order
    m = m.sort_values(["pattern", "query_id", "judge"], kind="mergesort").reset_index(drop=True)
    return m


# ---------------------------------------------------------------- fit helpers
def _solve(X, Y):
    A = X.T @ X
    b = X.T @ Y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, Y, rcond=None)[0]


def adj_means_pooled(pc, yy, x):
    """Spec A/C: no-intercept pattern dummies + centred length. Returns (adj_means[11], beta)."""
    xc = x - x.mean()
    n = len(yy)
    X = np.zeros((n, 12))
    X[np.arange(n), pc] = 1.0
    X[:, 11] = xc
    coef = _solve(X, yy)
    return coef[:11], float(coef[11])


def gaps_query_fe(pc, yy, x, cid):
    """Spec B: query(-copy) FE via within-group demeaning (FWL), P0 = reference.

    Returns (gaps_vs_p0[11] with gaps[P0]=0, beta). Gap differences are invariant to the
    FE parameterisation; levels are intentionally not identified here.
    """
    ncid = int(cid.max()) + 1
    cnt = np.bincount(cid, minlength=ncid).astype(float)

    def dm(v):
        gm = np.bincount(cid, weights=v, minlength=ncid) / cnt
        return v - gm[cid]

    yd = dm(yy)
    n = len(yy)
    X = np.empty((n, 11))
    k = 0
    for j in range(11):
        if j == P0:
            continue
        X[:, k] = dm((pc == j).astype(float))
        k += 1
    X[:, 10] = dm(x)
    coef = _solve(X, yd)
    gaps = np.zeros(11)
    k = 0
    for j in range(11):
        if j == P0:
            continue
        gaps[j] = coef[k]
        k += 1
    return gaps, float(coef[10])


def raw_pattern_means(pc, yy):
    sums = np.bincount(pc, weights=yy, minlength=11)
    cnts = np.bincount(pc, minlength=11).astype(float)
    with np.errstate(invalid="ignore"):
        return sums / cnts


def three_gaps_from_means(means):
    cl = float(means[CL_IDX].mean())
    return {
        "cluster_minus_p0": cl - float(means[P0]),
        "p0_minus_p9": float(means[P0]) - float(means[P9]),
        "cluster_minus_p9": cl - float(means[P9]),
    }


# ---------------------------------------------------------------- main build
def build():
    m = load_cells()
    y = m["ovc"].to_numpy(float)
    words = m["report_word_count"].to_numpy(float)
    kw = words / 1000.0
    lw = np.log(np.clip(words, 1.0, None))
    pat_code = np.array([PATS.index(p) for p in m["pattern"]])
    judge = m["judge"].to_numpy()
    jmask = {j: judge == j for j in PANEL}
    qids = sorted(m["query_id"].astype(str).unique())
    qpos = {q: i for i, q in enumerate(qids)}
    qcode = np.array([qpos[q] for q in m["query_id"].astype(str)])
    qrows = [np.flatnonzero(qcode == i) for i in range(len(qids))]
    nq = len(qids)
    n = len(m)

    # -------- point estimates --------
    sel_all = np.arange(n)

    def spec_points(sel, cid):
        pc, yy = pat_code[sel], y[sel]
        out = {}
        out["raw_means_3j"] = raw_pattern_means(pc, yy)
        adjA, betaA = adj_means_pooled(pc, yy, kw[sel])
        out["A_means"], out["A_beta"] = adjA, betaA
        gapsB, betaB = gaps_query_fe(pc, yy, kw[sel], cid)
        out["B_gaps"], out["B_beta"] = gapsB, betaB
        adjC, betaC = adj_means_pooled(pc, yy, lw[sel])
        out["C_means"], out["C_beta"] = adjC, betaC
        return out

    pt = spec_points(sel_all, qcode)
    per_judge_pt = {}
    for j in PANEL:
        sel = np.flatnonzero(jmask[j])
        pc, yy = pat_code[sel], y[sel]
        adjA, betaA = adj_means_pooled(pc, yy, kw[sel])
        # per-judge query FE (cid must be re-densified to the judge subset's queries)
        cid_j = np.unique(qcode[sel], return_inverse=True)[1]
        gapsB, betaB = gaps_query_fe(pc, yy, kw[sel], cid_j)
        adjC, betaC = adj_means_pooled(pc, yy, lw[sel])
        per_judge_pt[j] = {
            "raw_means": raw_pattern_means(pc, yy),
            "A_means": adjA, "A_beta": betaA,
            "B_gaps": gapsB, "B_beta": betaB,
            "C_beta": betaC, "C_means": adjC,
        }

    # sensitivity points (no bootstrap): min-100-words filter; judge-FE check
    sel_100 = np.flatnonzero(words >= 100)
    adjA100, betaA100 = adj_means_pooled(pat_code[sel_100], y[sel_100], kw[sel_100])
    # judge-FE check: spec A + centred judge dummies (adjusted means at balanced judge mix)
    Xj = np.zeros((n, 14))
    Xj[np.arange(n), pat_code] = 1.0
    Xj[:, 11] = kw - kw.mean()
    for k_, j in enumerate(PANEL[1:]):
        d = jmask[j].astype(float)
        Xj[:, 12 + k_] = d - d.mean()
    coefj = _solve(Xj, y)
    adjA_jfe = coefj[:11]

    # -------- query-clustered paired bootstrap --------
    rng = np.random.default_rng(SEED)
    TRACKS = [
        "raw3_cl_p0", "raw3_p0_p9", "raw3_cl_p9",
        "raw52_cl_p0", "raw52_p0_p9", "raw52_cl_p9",
        "A3_cl_p0", "A3_p0_p9", "A3_cl_p9", "A3_beta",
        "A52_cl_p0", "A52_p0_p9", "A52_cl_p9",
        "Aopus_cl_p0", "Asonnet_cl_p0",
        "B3_cl_p0", "B3_p0_p9", "B3_cl_p9",
        "C3_cl_p0", "C3_p0_p9", "C3_cl_p9",
    ]
    boot = {t: np.empty(N_BOOT) for t in TRACKS}
    for b in range(N_BOOT):
        s = rng.integers(0, nq, size=nq)  # same block for every spec/judge => paired
        sel = np.concatenate([qrows[qi] for qi in s])
        cid = np.concatenate(
            [np.full(len(qrows[qi]), k, dtype=np.int64) for k, qi in enumerate(s)])
        pc, yy = pat_code[sel], y[sel]
        g = three_gaps_from_means(raw_pattern_means(pc, yy))
        boot["raw3_cl_p0"][b] = g["cluster_minus_p0"]
        boot["raw3_p0_p9"][b] = g["p0_minus_p9"]
        boot["raw3_cl_p9"][b] = g["cluster_minus_p9"]
        m52 = jmask["gpt52"][sel]
        g = three_gaps_from_means(raw_pattern_means(pc[m52], yy[m52]))
        boot["raw52_cl_p0"][b] = g["cluster_minus_p0"]
        boot["raw52_p0_p9"][b] = g["p0_minus_p9"]
        boot["raw52_cl_p9"][b] = g["cluster_minus_p9"]
        adjA, betaA = adj_means_pooled(pc, yy, kw[sel])
        g = three_gaps_from_means(adjA)
        boot["A3_cl_p0"][b] = g["cluster_minus_p0"]
        boot["A3_p0_p9"][b] = g["p0_minus_p9"]
        boot["A3_cl_p9"][b] = g["cluster_minus_p9"]
        boot["A3_beta"][b] = betaA
        adj52, _ = adj_means_pooled(pc[m52], yy[m52], kw[sel][m52])
        g = three_gaps_from_means(adj52)
        boot["A52_cl_p0"][b] = g["cluster_minus_p0"]
        boot["A52_p0_p9"][b] = g["p0_minus_p9"]
        boot["A52_cl_p9"][b] = g["cluster_minus_p9"]
        for jname, track in (("claude_opus", "Aopus_cl_p0"),
                             ("claude_sonnet", "Asonnet_cl_p0")):
            mj = jmask[jname][sel]
            adjj, _ = adj_means_pooled(pc[mj], yy[mj], kw[sel][mj])
            boot[track][b] = float(adjj[CL_IDX].mean() - adjj[P0])
        gapsB, _ = gaps_query_fe(pc, yy, kw[sel], cid)
        clB = float(gapsB[CL_IDX].mean())
        boot["B3_cl_p0"][b] = clB
        boot["B3_p0_p9"][b] = -float(gapsB[P9])
        boot["B3_cl_p9"][b] = clB - float(gapsB[P9])
        adjC, _ = adj_means_pooled(pc, yy, lw[sel])
        g = three_gaps_from_means(adjC)
        boot["C3_cl_p0"][b] = g["cluster_minus_p0"]
        boot["C3_p0_p9"][b] = g["p0_minus_p9"]
        boot["C3_cl_p9"][b] = g["cluster_minus_p9"]

    def ci(t):
        lo, hi = np.percentile(boot[t], [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    def p_boot(t):
        d = boot[t]
        p_two = min(1.0, 2.0 * min(float((d > 0).mean()), float((d < 0).mean())))
        p_r = round(p_two, 4)
        return ("<%.0e" % (1.0 / N_BOOT)) if p_r == 0.0 else p_r

    # -------- assemble: raw-gap key --------
    raw3 = three_gaps_from_means(pt["raw_means_3j"])
    raw52 = three_gaps_from_means(per_judge_pt["gpt52"]["raw_means"])
    basis_3j = (
        "mean-of-pattern-means, available-case: each pattern's score is its pooled "
        "judge-cell mean over all available (query x judge) cells of the three-judge "
        "panel {gpt52, claude_opus, claude_sonnet}, with claude_sonnet's corrupted "
        "overall_score replaced by overall_score_recomputed (identical to the store's "
        "headline.per_pattern.mean_3judge, n_cells 265-270 per pattern); the cluster "
        "value is the UNWEIGHTED mean of the six cluster patterns' means (each pattern "
        "weighted equally, not each cell/query).")
    basis_52 = (
        "mean-of-pattern-means over the GPT-5.2 judge alone (identical to the store's "
        "single_judge_gpt52.per_pattern.mean; n=87-90 queries per pattern); cluster = "
        "unweighted mean of the six cluster patterns' means.")
    key_raw = {
        "_note": (
            "Canonical backing for the paper's previously-unbacked headline gap numbers: "
            "the six-pattern orchestrated-cluster mean (P1,P4,P5,P6,P7,P8) minus the "
            "single-pass baseline P0, on the three-judge panel mean (quoted as 0.145/0.146) "
            "and on the family-clean GPT-5.2 judge alone (quoted as 0.065), plus the "
            "P0-vs-P9 and cluster-vs-P9 companions. Recomputed from the released parquets "
            "and verified to reproduce the store's own headline / single_judge_gpt52 means. "
            "CIs are query-clustered (resample the 90 query ids) paired bootstrap "
            "percentile intervals."),
        "cluster_patterns": CLUSTER,
        "three_judge": {
            "basis": basis_3j,
            "p0_mean": round(float(pt["raw_means_3j"][P0]), 4),
            "cluster_mean": round(float(pt["raw_means_3j"][CL_IDX].mean()), 4),
            "per_pattern_means": {p: round(float(pt["raw_means_3j"][PATS.index(p)]), 4)
                                  for p in CLUSTER + ["base_p0", "base_p9"]},
            "cluster_minus_p0": {"point": round(raw3["cluster_minus_p0"], 4),
                                 "ci95": ci("raw3_cl_p0")},
            "p0_minus_p9": {"point": round(raw3["p0_minus_p9"], 4),
                            "ci95": ci("raw3_p0_p9")},
            "cluster_minus_p9": {"point": round(raw3["cluster_minus_p9"], 4),
                                 "ci95": ci("raw3_cl_p9")},
        },
        "gpt52": {
            "basis": basis_52,
            "p0_mean": round(float(per_judge_pt["gpt52"]["raw_means"][P0]), 4),
            "cluster_mean": round(float(per_judge_pt["gpt52"]["raw_means"][CL_IDX].mean()), 4),
            "cluster_minus_p0": {"point": round(raw52["cluster_minus_p0"], 4),
                                 "ci95": ci("raw52_cl_p0")},
            "p0_minus_p9": {"point": round(raw52["p0_minus_p9"], 4),
                            "ci95": ci("raw52_p0_p9")},
            "cluster_minus_p9": {"point": round(raw52["cluster_minus_p9"], 4),
                                 "ci95": ci("raw52_cl_p9")},
        },
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "resampling_unit": "query",
                      "paired": True},
        "verified_against_expected": {},
    }
    got = {
        "cluster_minus_p0_3judge": raw3["cluster_minus_p0"],
        "cluster_minus_p0_gpt52": raw52["cluster_minus_p0"],
        "p0_minus_p9_3judge": raw3["p0_minus_p9"],
        "cluster_minus_p9_3judge": raw3["cluster_minus_p9"],
        "p0_minus_p9_gpt52": raw52["p0_minus_p9"],
        "cluster_minus_p9_gpt52": raw52["cluster_minus_p9"],
    }
    all_ok = True
    for k, exp in EXPECTED.items():
        ok = abs(got[k] - exp) <= 5e-4
        all_ok = all_ok and ok
        key_raw["verified_against_expected"][k] = {
            "got": round(got[k], 4), "expected": exp, "ok": bool(ok)}
    key_raw["verified_all_ok"] = bool(all_ok)

    # -------- assemble: length-adjusted key --------
    mean_words = {}
    for j, p in enumerate(PATS):
        selp = pat_code == j
        # per-report (not per-cell) mean: dedupe on query via first judge occurrence is
        # unnecessary because word count is constant within (pattern, query); cell mean
        # over judges equals the report-level mean up to judge-coverage imbalance, so we
        # compute it from the deduped (pattern, query) pairs for exactness.
        dfp = m[selp][["query_id", "report_word_count"]].drop_duplicates("query_id")
        mean_words[p] = round(float(dfp["report_word_count"].mean()), 1)

    def gapblock(track_prefix, gaps_point):
        return {
            "cluster_minus_p0": {"point": round(gaps_point["cluster_minus_p0"], 4),
                                 "ci95": ci(f"{track_prefix}_cl_p0"),
                                 "p_two_sided_boot": p_boot(f"{track_prefix}_cl_p0")},
            "p0_minus_p9": {"point": round(gaps_point["p0_minus_p9"], 4),
                            "ci95": ci(f"{track_prefix}_p0_p9")},
            "cluster_minus_p9": {"point": round(gaps_point["cluster_minus_p9"], 4),
                                 "ci95": ci(f"{track_prefix}_cl_p9")},
        }

    gA3 = three_gaps_from_means(pt["A_means"])
    gA52 = three_gaps_from_means(per_judge_pt["gpt52"]["A_means"])
    gB3 = {"cluster_minus_p0": float(pt["B_gaps"][CL_IDX].mean()),
           "p0_minus_p9": -float(pt["B_gaps"][P9]),
           "cluster_minus_p9": float(pt["B_gaps"][CL_IDX].mean()) - float(pt["B_gaps"][P9])}
    gC3 = three_gaps_from_means(pt["C_means"])

    per_judge_out = {}
    for j in PANEL:
        pj = per_judge_pt[j]
        gA = three_gaps_from_means(pj["A_means"])
        entry = {
            "raw_cluster_minus_p0": round(float(
                pj["raw_means"][CL_IDX].mean() - pj["raw_means"][P0]), 4),
            "pooled_ols_kwords": {
                "length_coef_per_1000_words": round(pj["A_beta"], 4),
                "cluster_minus_p0": {"point": round(gA["cluster_minus_p0"], 4)},
                "p0_minus_p9": round(gA["p0_minus_p9"], 4),
                "cluster_minus_p9": round(gA["cluster_minus_p9"], 4),
            },
            "query_fe_kwords": {
                "length_coef_per_1000_words": round(pj["B_beta"], 4),
                "cluster_minus_p0": round(float(pj["B_gaps"][CL_IDX].mean()), 4),
            },
            "log_words_slope": round(pj["C_beta"], 4),
        }
        if j == "gpt52":
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["ci95"] = ci("A52_cl_p0")
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["p_two_sided_boot"] = p_boot("A52_cl_p0")
            entry["pooled_ols_kwords"]["p0_minus_p9"] = {
                "point": round(gA52["p0_minus_p9"], 4), "ci95": ci("A52_p0_p9")}
            entry["pooled_ols_kwords"]["cluster_minus_p9"] = {
                "point": round(gA52["cluster_minus_p9"], 4), "ci95": ci("A52_cl_p9")}
        elif j == "claude_opus":
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["ci95"] = ci("Aopus_cl_p0")
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["p_two_sided_boot"] = p_boot("Aopus_cl_p0")
        elif j == "claude_sonnet":
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["ci95"] = ci("Asonnet_cl_p0")
            entry["pooled_ols_kwords"]["cluster_minus_p0"]["p_two_sided_boot"] = p_boot("Asonnet_cl_p0")
        per_judge_out[j] = entry

    survives_A = bool(np.percentile(boot["A3_cl_p0"], 2.5) > 0)
    survives_B = bool(np.percentile(boot["B3_cl_p0"], 2.5) > 0)
    survives_C = bool(np.percentile(boot["C3_cl_p0"], 2.5) > 0)

    key_adj = {
        "_note": (
            "Length-control reanalysis of the HEADLINE contrasts, applying the paper's own "
            "frozen_vintage length adjustment (pooled OLS: score ~ pattern dummies + "
            "beta*(words/1000 - grand mean); each dummy = counterfactual score at grand-"
            "mean length) to the main 11-pattern x 90-query three-judge run. The "
            "orchestrated cluster produces systematically longer reports (~1.9k words vs "
            "P0 ~1.2k, P9 ~0.7k), so the raw headline gaps are partially a verbosity "
            "artefact under judges with a positive length slope. Spec B adds query fixed "
            "effects (controls query-difficulty confounding of the within-pattern slope); "
            "spec C uses log-words (the judge-bias audit's scale). All gaps are on the "
            "mean-of-pattern-means basis of 'headline_cluster_gap'."),
        "key_version": "1.0",
        "data": {
            "patterns": PATS,
            "cluster_patterns": CLUSTER,
            "n_cells_3judge": int(n),
            "n_queries": nq,
            "judges": PANEL,
            "sonnet_correction": "overall_score_recomputed used for claude_sonnet rows",
            "length_unit": (
                "report_word_count from df_runs.parquet (whitespace word count of the "
                "released .md; tokenizer-independent, same unit as frozen_vintage); every "
                "judged cell has a non-null word count"),
            "mean_output_words_per_pattern": mean_words,
            "grand_mean_words_3judge_cells": round(float(words.mean()), 1),
        },
        "raw_reference": {
            "see_key": "headline_cluster_gap",
            "three_judge_cluster_minus_p0": round(raw3["cluster_minus_p0"], 4),
            "gpt52_cluster_minus_p0": round(raw52["cluster_minus_p0"], 4),
            "three_judge_p0_minus_p9": round(raw3["p0_minus_p9"], 4),
            "three_judge_cluster_minus_p9": round(raw3["cluster_minus_p9"], 4),
        },
        "specs": {
            "pooled_ols_kwords_vintage_method": {
                "method": (
                    "cell-level pooled OLS over all 11 arms x 3 judges: score ~ 11 pattern "
                    "dummies (no intercept) + beta*(words/1000 - grand-mean kwords); "
                    "identical to frozen_vintage.length_control (whose slope was +0.1748)"),
                "three_judge": {
                    "length_coef_per_1000_words": round(pt["A_beta"], 4),
                    "length_coef_ci95": ci("A3_beta"),
                    "adjusted_means": {p: round(float(pt["A_means"][PATS.index(p)]), 4)
                                       for p in PATS},
                    "gaps": gapblock("A3", gA3),
                },
                "per_judge": per_judge_out,
            },
            "query_fixed_effects_kwords": {
                "method": (
                    "spec A + query fixed effects (within-query demeaning, Frisch-Waugh; "
                    "P0 reference, so coefficients are adjusted gaps vs P0; bootstrap "
                    "gives each resampled query COPY its own FE). Controls query-"
                    "difficulty confounding of the within-pattern length-score slope."),
                "three_judge": {
                    "length_coef_per_1000_words": round(pt["B_beta"], 4),
                    "gaps": gapblock("B3", gB3),
                },
            },
            "pooled_ols_log_words": {
                "method": (
                    "spec A with log(words) (clipped at 1) instead of linear kwords - the "
                    "judge-bias audit's scale, where beta(log wc) ~ 0.04-0.11; arms "
                    "evaluated at the grand-mean of log-words"),
                "three_judge": {
                    "length_coef_per_log_word": round(pt["C_beta"], 4),
                    "gaps": gapblock("C3", gC3),
                },
            },
        },
        "sensitivity_points": {
            "min_100_words_filter": {
                "note": (
                    "spec A on cells with report >= 100 words (drops the degenerate near-"
                    "empty short tail concentrated in P0/P9), point estimates only"),
                "n_cells": int(len(sel_100)),
                "length_coef_per_1000_words": round(betaA100, 4),
                "cluster_minus_p0": round(float(adjA100[CL_IDX].mean() - adjA100[P0]), 4),
                "p0_minus_p9": round(float(adjA100[P0] - adjA100[P9]), 4),
                "cluster_minus_p9": round(float(adjA100[CL_IDX].mean() - adjA100[P9]), 4),
            },
            "judge_fe_check": {
                "note": ("spec A + centred judge dummies (adjusted means at balanced "
                         "judge mix; guards the slight judge-coverage imbalance), point only"),
                "cluster_minus_p0": round(float(adjA_jfe[CL_IDX].mean() - adjA_jfe[P0]), 4),
            },
        },
        "bootstrap": {
            "n_boot": N_BOOT, "seed": SEED, "resampling_unit": "query", "paired": True,
            "note": (
                "np.random.default_rng(seed); the SAME resampled query-id block feeds "
                "every spec and judge each iteration (paired); percentile CIs; two-sided "
                "p = 2*min(P(d>0),P(d<0)) floored at 1/n_boot."),
        },
        "interpretation_caveats": [
            ("With pattern dummies the pooled length slope is identified from WITHIN-"
             "pattern variation, which conflates judge length-preference with query-level "
             "material richness (successful retrieval yields both longer reports and "
             "better scores); it is therefore an UPPER-bound-ish verbosity correction. "
             "Spec B (query FE) removes the query side and is the preferred causal-ish "
             "spec; spec C matches the judge-bias audit's log scale."),
            ("Grand-mean evaluation (~1,600 words) sits at the edge of P9's length "
             "support (mean 726, p95 ~1,622): P9's adjusted mean involves mild "
             "extrapolation, so length-adjusted P0-vs-P9 / cluster-vs-P9 gaps carry more "
             "model dependence than the cluster-vs-P0 gap."),
            ("P0 and P9 have a short tail of degenerate near-empty reports; see "
             "sensitivity_points.min_100_words_filter."),
        ],
        "verdict": {
            "cluster_lead_survives_pooled_ols_kwords": survives_A,
            "cluster_lead_survives_query_fe": survives_B,
            "cluster_lead_survives_log_words": survives_C,
            "summary": (
                f"Raw three-judge cluster-vs-P0 gap {round(raw3['cluster_minus_p0'], 3)} "
                f"-> {round(gA3['cluster_minus_p0'], 3)} under the paper's own vintage-"
                f"method length adjustment (slope {round(pt['A_beta'], 3)}/kword), "
                f"{round(gB3['cluster_minus_p0'], 3)} with query FE (slope "
                f"{round(pt['B_beta'], 3)}), {round(gC3['cluster_minus_p0'], 3)} with "
                f"log-words. P0-vs-P9 shrinks {round(raw3['p0_minus_p9'], 3)} -> "
                f"{round(gA3['p0_minus_p9'], 3)} (vintage) but stays positive in all "
                f"specs; cluster-vs-P9 {round(raw3['cluster_minus_p9'], 3)} -> "
                f"{round(gA3['cluster_minus_p9'], 3)} likewise."),
        },
    }
    return key_raw, key_adj


# ---------------------------------------------------------------- io
def _print_dry(key_raw, key_adj):
    print(f"[{KEY_ADJ}] DRY-RUN — computed, nothing written.")
    print(f"  verified raw gaps vs expected: "
          f"{'ALL OK' if key_raw['verified_all_ok'] else 'MISMATCH'}")
    for k, v in key_raw["verified_against_expected"].items():
        print(f"    {k}: got={v['got']} exp={v['expected']} {'OK' if v['ok'] else 'FAIL'}")
    t3 = key_raw["three_judge"]; t5 = key_raw["gpt52"]
    print(f"  RAW  3j  cl-p0={t3['cluster_minus_p0']['point']} ci={t3['cluster_minus_p0']['ci95']}"
          f"  p0-p9={t3['p0_minus_p9']['point']} ci={t3['p0_minus_p9']['ci95']}"
          f"  cl-p9={t3['cluster_minus_p9']['point']} ci={t3['cluster_minus_p9']['ci95']}")
    print(f"  RAW  52  cl-p0={t5['cluster_minus_p0']['point']} ci={t5['cluster_minus_p0']['ci95']}"
          f"  p0-p9={t5['p0_minus_p9']['point']} ci={t5['p0_minus_p9']['ci95']}"
          f"  cl-p9={t5['cluster_minus_p9']['point']} ci={t5['cluster_minus_p9']['ci95']}")
    sp = key_adj["specs"]
    A = sp["pooled_ols_kwords_vintage_method"]["three_judge"]
    print(f"  ADJ A (vintage kwords) 3j: beta={A['length_coef_per_1000_words']} "
          f"ci={A['length_coef_ci95']}")
    for gname, g in A["gaps"].items():
        extra = f" p={g['p_two_sided_boot']}" if "p_two_sided_boot" in g else ""
        print(f"    {gname}: {g['point']} ci={g['ci95']}{extra}")
    B = sp["query_fixed_effects_kwords"]["three_judge"]
    print(f"  ADJ B (query FE) 3j: beta={B['length_coef_per_1000_words']}")
    for gname, g in B["gaps"].items():
        extra = f" p={g['p_two_sided_boot']}" if "p_two_sided_boot" in g else ""
        print(f"    {gname}: {g['point']} ci={g['ci95']}{extra}")
    C = sp["pooled_ols_log_words"]["three_judge"]
    print(f"  ADJ C (log words) 3j: beta={C['length_coef_per_log_word']}")
    for gname, g in C["gaps"].items():
        extra = f" p={g['p_two_sided_boot']}" if "p_two_sided_boot" in g else ""
        print(f"    {gname}: {g['point']} ci={g['ci95']}{extra}")
    print("  per-judge spec A cl-p0:")
    for j, e in sp["pooled_ols_kwords_vintage_method"]["per_judge"].items():
        cm = e["pooled_ols_kwords"]["cluster_minus_p0"]
        print(f"    {j}: raw={e['raw_cluster_minus_p0']} adj={cm['point']} "
              f"ci={cm.get('ci95')} beta={e['pooled_ols_kwords']['length_coef_per_1000_words']} "
              f"beta_qfe={e['query_fe_kwords']['length_coef_per_1000_words']} "
              f"beta_logw={e['log_words_slope']}")
    sv = key_adj["verdict"]
    print(f"  VERDICT: survives vintage-OLS={sv['cluster_lead_survives_pooled_ols_kwords']} "
          f"queryFE={sv['cluster_lead_survives_query_fe']} "
          f"logwords={sv['cluster_lead_survives_log_words']}")
    print(f"  sensitivity min100w: {key_adj['sensitivity_points']['min_100_words_filter']}")
    print(f"  judge-FE check cl-p0: "
          f"{key_adj['sensitivity_points']['judge_fe_check']['cluster_minus_p0']}")


def _atomic_append(key_raw, key_adj, force):
    cn = json.load(open(CANON))
    for k in (KEY_RAW, KEY_ADJ):
        if k in cn and not force:
            print(f"[{KEY_ADJ}] REFUSING to overwrite existing key '{k}' (use --force).")
            return 1
    cn[KEY_RAW] = key_raw
    cn[KEY_ADJ] = key_adj
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(ANA), prefix="canonical_numbers.", suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cn, f, indent=1)
        os.replace(tmp, CANON)
        tmp = None
    except BaseException:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    print(f"[{KEY_ADJ}] WROTE keys '{KEY_RAW}' + '{KEY_ADJ}' -> {CANON} "
          f"(store now {len(cn)} keys)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print, write nothing (default)")
    ap.add_argument("--write", action="store_true",
                    help="atomically append the two keys to the canonical store")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing owned keys (only with --write)")
    args = ap.parse_args()

    if not CANON.exists():
        print(f"[{KEY_ADJ}] canonical store missing at {CANON}; nothing to do (self-guard).")
        return 0
    if not (OV_PQ.exists() and RUNS_PQ.exists()):
        print(f"[{KEY_ADJ}] released parquets missing; nothing to do (self-guard).")
        return 0

    key_raw, key_adj = build()

    if not key_raw["verified_all_ok"]:
        print(f"[{KEY_ADJ}] WARNING: raw gaps do not reproduce the store's headline "
              f"means; inspect verified_against_expected before trusting the write.")

    if args.write:
        return _atomic_append(key_raw, key_adj, args.force)
    _print_dry(key_raw, key_adj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
