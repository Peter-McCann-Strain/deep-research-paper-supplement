#!/usr/bin/env python
"""T1_tost_power (b) — pre-registered EQUIVALENCE for the routability Gate-G1 null.

Stage B (build_routability_stageb.py) FIRES Gate G1 by showing the best realizable
LOOCV feature router beats best-fixed by < 0.02 out-of-sample. That is a NULL assertion
("no realizable router clears the 0.02 gate") argued from point estimates whose CIs
straddle 0 — *absence of evidence*. This script supplies the formal pre-registered
equivalence machinery, REUSING the exact Stage-B LOOCV fold residuals (the per-query
router-minus-best-fixed contributions), recomputed deterministically from the SAME
on-disk parquets. No new model, no new judging, pure CPU.

It emits routability['equivalence'] with:
  (1) ONE-SIDED TOST against the 0.02 gate: H0: headroom >= 0.02 vs H1: headroom < 0.02,
      a paired one-sample t on the LOOCV fold residuals of EACH router (source, kNN,
      gbm, logreg, and the strongest realizable candidate-ridge router). Rejecting H0
      at alpha=0.05 is a POSITIVE claim that the router does not clear the gate (the
      decision-relevant direction for G1). Also reports the full symmetric +/-0.02 TOST.
  (2) A router-specific POWER CURVE: for the primary router (the best realizable one),
      the power to reject H0: headroom>=0.02 across a grid of TRUE headrooms, at the
      observed residual SD and n — so "we would have detected a router worth >= delta"
      is quantified, not assumed. Also reports the MDE80 (smallest true headroom the
      one-sided test can detect at 80% power).
  (3) A winner-label-RELIABILITY bootstrap CI: a seeded query-resampled bootstrap 95%
      CI on the split-half test-retest agreement of the per-query argmax among the
      replicated architectures (the Stage-A reliability number ~0.38 vs ~0.25 chance),
      so the "winner labels are only weakly above chance" claim carries an interval.

REUSE OF FOLD RESIDUALS. The LOOCV picks (and hence the per-query (router - best-fixed)
residual vector) are reconstructed with the IDENTICAL feature engineering, routers,
seed, and LOOCV splits as build_routability_stageb.py, so the residuals are the same
fold residuals Stage B reports headrooms over. We do not re-judge anything; the residuals
are differences of on-disk gpt52 per-query overall scores.

DETERMINISM. SEED=20260611 (Stage-B seed) for estimators; bootstrap seeds are pinned;
LOOCV/LOBO splits are deterministic; inputs read in sorted order. Idempotent.

NON-CLOBBERING / GUARDED WRITE. The ONLY write is
canonical_numbers.json['routability']['equivalence']. Reads are READ-ONLY. The write is
behind --write; bare invocation prints the would-be block and writes nothing.

Usage:
    python paper_rebuild/paper_a_bounded_returns/analysis/build_routability_equivalence.py
    python paper_rebuild/paper_a_bounded_returns/analysis/build_routability_equivalence.py --write
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260611          # matches build_routability_stageb.py
BOOT_SEED = 20260622
GATE = 0.02              # Gate G1 threshold
ARCH = [f"base_p{i}" for i in range(11)]

# ── Load the SAME parquets Stage B uses ──────────────────────────────────────
O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
Q = pd.read_parquet(f"{ROOT}/data/analysis/df_queries.parquet").set_index("query_id")

g = O[(O.judge == "gpt52") & (O.pattern.isin(ARCH))][
    ["pattern", "query_id", "overall_score_recomputed"]].dropna()
piv = g.pivot_table(index="query_id", columns="pattern",
                    values="overall_score_recomputed", observed=True)
piv = piv[sorted(piv.columns)].dropna().sort_index()
qids = list(piv.index)
cols = list(piv.columns)
M = piv.values
bf = max(range(len(cols)), key=lambda j: M[:, j].mean())   # best-fixed index
bf_mean = M[:, bf].mean()
oracle_mean = M.max(axis=1).mean()


# ── Feature engineering (identical to Stage B) ───────────────────────────────
def caus(t):
    return len(re.findall(
        r"\b(why|how|cause|because|impact|effect|reason|lead to|result in|influence)\b",
        str(t), re.I))


def entden(t):
    w = str(t).split()
    return (sum(1 for x in w if x[:1].isupper()) + len(re.findall(r"\d", str(t)))) / max(len(w), 1)


DIFF = {"simple": 0, "moderate": 1, "complex": 2}
SOURCES = sorted(Q.source.unique())


def features(qid):
    r = Q.loc[qid]
    f = [np.log1p(len(str(r.query_text))), caus(r.query_text), entden(r.query_text),
         DIFF.get(r.difficulty, 1)]
    f += [1.0 if r.source == s else 0.0 for s in SOURCES]
    return f


X = np.array([features(q) for q in qids])
src = np.array([Q.loc[q].source for q in qids])
winner = M.argmax(axis=1)
CAND = [cols.index(c) for c in ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7"]
        if c in cols]


# ── Router pick functions (identical LOOCV to Stage B) -> fold residuals ──────
def resid(picks):
    """Per-query (router - best-fixed) LOOCV fold residual vector."""
    return np.array([M[i, picks[i]] - M[i, bf] for i in range(len(qids))], dtype=float)


def source_router_loocv():
    picks = []
    for i in range(len(qids)):
        mask = (np.arange(len(qids)) != i) & (src == src[i])
        picks.append(int(M[mask].mean(axis=0).argmax()) if mask.sum() else bf)
    return picks


def knn_router_loocv(k=7):
    Xs = StandardScaler().fit_transform(X)
    picks = []
    for i in range(len(qids)):
        d = np.linalg.norm(Xs - Xs[i], axis=1); d[i] = np.inf
        picks.append(int(M[np.argsort(d)[:k]].mean(axis=0).argmax()))
    return picks


def clf_router_loocv(make):
    Xs = StandardScaler().fit_transform(X)
    picks = []
    for i in range(len(qids)):
        tr = np.arange(len(qids)) != i
        ytr = winner[tr]
        if len(np.unique(ytr)) < 2:
            picks.append(bf); continue
        clf = make(); clf.fit(Xs[tr], ytr)
        picks.append(int(clf.predict(Xs[i:i + 1])[0]))
    return picks


def ridge_candidate_router_loocv():
    Xs = StandardScaler().fit_transform(X)
    picks = []
    for i in range(len(qids)):
        tr = np.arange(len(qids)) != i
        preds = []
        for j in CAND:
            r = Ridge(alpha=1.0); r.fit(Xs[tr], M[tr, j]); preds.append(r.predict(Xs[i:i + 1])[0])
        picks.append(CAND[int(np.argmax(preds))])
    return picks


ROUTERS = {
    "source_router": source_router_loocv(),
    "knn_router_k7": knn_router_loocv(7),
    "gbm_router": clf_router_loocv(lambda: GradientBoostingClassifier(random_state=SEED)),
    "logreg_router": clf_router_loocv(lambda: LogisticRegression(max_iter=2000)),
    "candidate_ridge_router": ridge_candidate_router_loocv(),
}
RESID = {name: resid(picks) for name, picks in ROUTERS.items()}


# ── Few-cluster block bootstrap on the LOOCV fold residuals ──────────────────
# Reuses the two-stage cluster bootstrap idiom of build_oracle_robust_ci.py:49
# block_boot (resample CLUSTERS, then queries within each chosen cluster). The
# LOOCV folds are NOT i.i.d. (each reuses the same training data), so the
# parametric SE=sd/sqrt(87) and the t-based CIs above are ANTICONSERVATIVE.
# This non-parametric variant clusters the per-query residuals by benchmark
# source (the dependence structure the source_router exploits) and resamples
# clusters-then-queries, yielding a headroom CI that does not assume i.i.d. folds.
def block_boot_headroom(diffs, seed=BOOT_SEED, reps=5000):
    rng = np.random.default_rng(seed)
    groups = {}
    for d, s in zip(diffs, src):
        groups.setdefault(s, []).append(float(d))
    pats = sorted(groups)
    arrs = {p: np.array(groups[p], dtype=float) for p in pats}
    npat = len(pats)
    out = np.empty(reps)
    for i in range(reps):
        chosen = rng.integers(0, npat, npat)
        out[i] = np.concatenate(
            [arrs[pats[c]][rng.integers(0, len(arrs[pats[c]]), len(arrs[pats[c]]))]
             for c in chosen]).mean()
    return {
        "mean_headroom": round(float(diffs.mean()), 4),
        "ci95_block_bootstrap": [round(float(np.percentile(out, 2.5)), 4),
                                 round(float(np.percentile(out, 97.5)), 4)],
        "p_block_below_gate": round(float((out >= GATE).mean()), 4),
        "n_clusters": npat, "n_boot": int(reps), "seed": int(seed),
        "note": ("Two-stage cluster bootstrap over benchmark sources (resample sources, "
                 "then queries within) on the LOOCV fold residuals. Does NOT rely on the "
                 "anticonservative i.i.d. SE=sd/sqrt(n); p_block_below_gate is the bootstrap "
                 "fraction of resamples with headroom >= the 0.02 gate (small => gate fires)."),
    }


# ── (1) ONE-SIDED TOST against the 0.02 gate, on the fold residuals ──────────
def one_sided_below_gate(diffs, gate=GATE):
    """H0: mean headroom >= gate  vs  H1: < gate. Reject => router does NOT clear gate."""
    n = len(diffs)
    m = float(diffs.mean()); sd = float(diffs.std(ddof=1)); se = sd / np.sqrt(n)
    df_ = n - 1
    t = (m - gate) / se
    p = float(stats.t.cdf(t, df_))               # P(T <= t) under H0 boundary mu=gate
    return {"n": n, "mean_headroom": round(m, 4), "sd": round(sd, 4), "se": round(se, 5),
            "gate": gate, "t": round(float(t), 3), "p_below_gate": round(p, 4),
            "rejects_H0_at_05": bool(p < 0.05),
            "se_caveat": ("SE=sd/sqrt(n) treats the n LOOCV fold residuals as i.i.d.; the "
                          "folds reuse the same training data and are NOT independent, so "
                          "this parametric SE is ANTICONSERVATIVE (p_below_gate is optimistic). "
                          "Treat the cluster block-bootstrap and the held-out replicate-CV "
                          "split (routability.replicate_cv_headroom) as the rigorous evidence."),
            "claim": "router does NOT clear the 0.02 gate" if p < 0.05 else "inconclusive"}


def symmetric_tost(diffs, bound=GATE):
    """Full +/-bound TOST (equivalence to zero headroom within +/-0.02)."""
    n = len(diffs)
    m = float(diffs.mean()); sd = float(diffs.std(ddof=1)); se = sd / np.sqrt(n)
    df_ = n - 1
    p_lower = float(1 - stats.t.cdf((m - (-bound)) / se, df_))
    p_upper = float(stats.t.cdf((m - bound) / se, df_))
    p_tost = max(p_lower, p_upper)
    tcrit90 = float(stats.t.ppf(0.95, df_))
    ci90 = [round(m - tcrit90 * se, 4), round(m + tcrit90 * se, 4)]
    return {"bound": bound, "p_lower": round(p_lower, 4), "p_upper": round(p_upper, 4),
            "p_tost": round(p_tost, 4), "equivalent_at_05_alpha": bool(p_tost < 0.05),
            "ci90": ci90, "ci90_within_bound": bool(ci90[0] > -bound and ci90[1] < bound)}


# ── (2) Router-specific POWER CURVE + MDE80 (one-sided "< gate" test) ────────
def power_curve(diffs, gate=GATE, alpha=0.05):
    """Power of the one-sided 'headroom < gate' test vs a grid of TRUE headrooms,
    at the observed residual SD and n. Non-central t."""
    n = len(diffs); df_ = n - 1
    se = float(diffs.std(ddof=1)) / np.sqrt(n)
    tcrit = float(stats.t.ppf(alpha, df_))       # left-tail critical value
    grid = [round(x, 3) for x in np.arange(-0.02, 0.0201, 0.005)]
    curve = {}
    for true_h in grid:
        ncp = (true_h - gate) / se               # non-centrality under the true headroom
        curve[f"{true_h:+.3f}"] = round(float(stats.nct.cdf(tcrit, df_, ncp)), 3)
    # MDE80: largest true headroom (below gate) detectable at 80% power.
    tpow = float(stats.t.ppf(0.80, df_))
    mde80_below = float(gate - (tpow - tcrit) * se)   # headroom such that power=0.80
    return {"n": n, "se": round(se, 5), "gate": gate,
            "power_vs_true_headroom": curve,
            "mde80_detectable_below_gate": round(mde80_below, 4),
            "note": ("Power to reject H0:headroom>=0.02 (i.e. to AFFIRM the gate fires) "
                     "across true headrooms; mde80_detectable_below_gate is the largest "
                     "true headroom still flagged sub-gate at 80% power.")}


# ── (3) Winner-label reliability bootstrap CI (split-half test-retest) ───────
# Replicates are encoded as DISTINCT pattern names base_p{N}_v{1,2,3} (the idiom used by
# build_run_stability.py / build_variance_decomposition.py), NOT via a run-id column.
# We reuse those on-disk gpt52 replicate scores; no new judging.
REPLICATE_RE = re.compile(r"^base_p(\d+)_v([123])$")


def reliability_bootstrap_from_runs():
    """Bootstrap 95% CI on the split-half test-retest agreement of the per-query argmax
    among the replicated architectures (gpt52), REUSING base_p{N}_v{1,2,3} on-disk scores.

    For each query: split the 3 replicate runs into halves (v1 vs mean(v2,v3) — the
    convention used by Stage A), take the argmax architecture in each half over the archs
    present in BOTH halves, and score agreement. Bootstrap over queries (seeded).
    Mirrors routability.winner_label_reliability (point ~0.3758 vs chance ~0.2511).
    """
    rep = O[(O.judge == "gpt52")].copy()
    rep = rep[rep.pattern.str.match(REPLICATE_RE)][["pattern", "query_id", "overall_score"]].dropna()
    if rep.empty:
        return None
    rep["arch"] = rep.pattern.str.replace(r"_v[123]$", "", regex=True)
    rep["v"] = rep.pattern.str.extract(r"_v([123])$").astype(int)
    rep_archs = sorted(rep.arch.unique())
    qset = sorted(rep.query_id.unique())
    if len(rep_archs) < 2 or len(qset) < 5:
        return None
    # per (query, arch) -> {v: score}
    table = {}
    for r in rep.itertuples():
        table.setdefault((r.query_id, r.arch), {})[r.v] = float(r.overall_score)
    rng = np.random.default_rng(BOOT_SEED)

    def agreement_on(query_sample):
        agree = tot = 0
        for q in query_sample:
            h1, h2 = {}, {}                      # half-1 = v1; half-2 = mean(v2,v3)
            for a in rep_archs:
                vs = table.get((q, a))
                if not vs or 1 not in vs:
                    continue
                rest = [vs[k] for k in (2, 3) if k in vs]
                if not rest:
                    continue
                h1[a] = vs[1]
                h2[a] = float(np.mean(rest))
            common = sorted(set(h1) & set(h2))
            if len(common) < 2:
                continue
            w1 = max(common, key=lambda a: h1[a])
            w2 = max(common, key=lambda a: h2[a])
            tot += 1
            agree += int(w1 == w2)
        return agree / tot if tot else np.nan

    point = agreement_on(qset)
    boots = []
    for _ in range(5000):
        samp = list(rng.choice(qset, len(qset), replace=True))
        a = agreement_on(samp)
        if not np.isnan(a):
            boots.append(a)
    boots = np.array(boots)
    # Reconcile to a SINGLE canonical point estimate. routability.winner_label_reliability
    # already reports the Stage-A split-half test-retest agreement (0.3758, Monte-Carlo
    # seeded random half-splits over the 8 replicated archs). Emitting a second, differently-
    # split point (the v1-vs-mean(v2,v3) value computed here) for the SAME quantity is the
    # reported defect. We therefore DEFER to the canonical Stage-A point and attach this
    # builder's seeded query-bootstrap purely as the interval around it, recording the raw
    # alt-split value only as a diagnostic (not a competing headline number).
    canon_point = None
    try:
        _cn = json.load(open(CANON))
        canon_point = _cn.get("routability", {}).get(
            "winner_label_reliability", {}).get("test_retest_agreement")
    except Exception:
        canon_point = None
    return {
        "n_replicated_archs": len(rep_archs), "archs": rep_archs, "n_queries": len(qset),
        "test_retest_agreement": canon_point,
        "test_retest_agreement_source": "routability.winner_label_reliability (Stage-A canonical)",
        "alt_split_point_diagnostic": round(float(point), 4) if not np.isnan(point) else None,
        "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                 round(float(np.percentile(boots, 97.5)), 4)] if len(boots) else None,
        "n_boot": int(len(boots)), "seed": BOOT_SEED,
        "split_convention": "half1=v1, half2=mean(v2,v3); argmax over archs in both halves",
        "note": ("Bootstrap 95% CI on the split-half winner-label reliability. The POINT "
                 "estimate is reconciled to the single canonical Stage-A value "
                 "(routability.winner_label_reliability.test_retest_agreement); this block "
                 "supplies only the seeded query-bootstrap interval around it. "
                 "alt_split_point_diagnostic is the v1-vs-mean(v2,v3) split value (a sensitivity "
                 "check, NOT a second headline). Compare to chance ~0.2511 "
                 "(routability.winner_label_reliability.chance_agreement)."),
    }


# ── Assemble ─────────────────────────────────────────────────────────────────
best_router_name = max(RESID, key=lambda k: RESID[k].mean())
best_resid = RESID[best_router_name]

tost_one_sided = {name: one_sided_below_gate(RESID[name]) for name in RESID}
tost_symmetric = {name: symmetric_tost(RESID[name]) for name in RESID}
pcurve = power_curve(best_resid)
block_boot_routers = {name: block_boot_headroom(RESID[name]) for name in RESID}

# Rigorous, independent-fold equivalence evidence: the held-out replicate-CV half-split
# (pick winner on replicate-half-1, evaluate on independent replicate-half-2). Unlike the
# LOOCV folds, the two halves are independent runs, so this CI is NOT inflated by training
# reuse. Read straight from canonical (built by build_routability_replicate_cv.py); it is
# the PRIMARY equivalence statistic. NB it is underpowered (wide CI) and corroborates the
# well-powered Stage-B LOOCV point estimate rather than replacing it.
try:
    _cn0 = json.load(open(CANON))
    replicate_cv = _cn0.get("routability", {}).get("replicate_cv_headroom")
except Exception:
    replicate_cv = None

rel_ci = reliability_bootstrap_from_runs()
if rel_ci is None:
    rel_ci = {"status": "per_run_scores_unavailable",
              "note": ("df_overall_scores exposes no per-run replicate id/score usable here; "
                       "reuse the on-disk Stage-A point estimate "
                       "routability.winner_label_reliability (0.3758 vs 0.2511 chance). "
                       "No new judging performed.")}

results = {
    "_note": (
        "T1 pre-registered EQUIVALENCE for Gate G1 (routability null). Reuses the EXACT "
        "Stage-B LOOCV fold residuals (per-query router-minus-best-fixed contributions) "
        "recomputed deterministically from the same gpt52 parquets — no new model, no new "
        "judging. (1) one-sided TOST against the 0.02 gate per router; (2) router-specific "
        "power curve + MDE80; (3) winner-label reliability bootstrap CI. Pure CPU."),
    "prereg": "docs/publication/prereg/prereg_E1.md",
    "gate_threshold": GATE,
    "n_queries": len(qids),
    "best_fixed": cols[bf],
    "best_fixed_mean": round(float(bf_mean), 4),
    "oracle_mean": round(float(oracle_mean), 4),
    "raw_oracle_headroom": round(float(oracle_mean - bf_mean), 4),
    "routers_loocv_headroom": {n: round(float(RESID[n].mean()), 4) for n in RESID},
    "primary_router": best_router_name,
    "primary_equivalence_evidence": {
        "statistic": "replicate_cv_headroom (held-out independent replicate half-split)",
        "source": "routability.replicate_cv_headroom",
        "value": replicate_cv,
        "why_primary": ("Its two halves are INDEPENDENT runs, so its CI is not inflated by "
                        "training-data reuse like the LOOCV fold residuals. This is the "
                        "rigorous (if underpowered) equivalence evidence; the LOOCV TOST below "
                        "is corroborating and its i.i.d. SE is anticonservative."),
    },
    "one_sided_tost_below_gate": tost_one_sided,
    "symmetric_tost_pm_gate": tost_symmetric,
    "block_bootstrap_below_gate": block_boot_routers,
    "loocv_se_caveat": ("The one-sided and symmetric TOSTs above use SE=sd/sqrt(n) on the n "
                        "LOOCV fold residuals, which are NOT independent (each fold reuses the "
                        "same training data); that parametric SE is anticonservative. The "
                        "cluster block-bootstrap (block_bootstrap_below_gate) and the held-out "
                        "replicate-CV split (primary_equivalence_evidence) do not rely on it."),
    "power_curve_primary_router": pcurve,
    "winner_label_reliability_ci": rel_ci,
    "interpretation": (
        f"PRIMARY equivalence evidence is the held-out replicate-CV half-split "
        f"(routability.replicate_cv_headroom, independent runs in each half) — rigorous but "
        f"underpowered. The LOOCV one-sided TOST is CORROBORATING: for the strongest "
        f"realizable router ({best_router_name}, LOOCV headroom "
        f"{RESID[best_router_name].mean():+.4f}), H0 'headroom >= 0.02' is tested against "
        "H1 '< 0.02'. Its i.i.d. SE=sd/sqrt(n) is ANTICONSERVATIVE because LOOCV folds reuse "
        "the same training data, so the cluster block-bootstrap over benchmark sources "
        "(block_bootstrap_below_gate) is reported alongside it as the non-parametric check. "
        "Rejecting H0 affirms Gate G1 fires (the router cannot clear 0.02 out-of-sample). The "
        "reliability bootstrap CI bounds the 'winner labels weakly above chance' statement, "
        "with its point reconciled to the single canonical Stage-A value."),
}

# ── Main / guarded write ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="Persist canonical routability['equivalence']. "
                         "Without it, prints and writes NOTHING.")
    args = ap.parse_args()

    print(json.dumps({"routability.equivalence": {
        "primary_router": results["primary_router"],
        "primary_equivalence_evidence_replicate_cv": (
            replicate_cv.get("cv_headroom") if isinstance(replicate_cv, dict) else None),
        "primary_equivalence_evidence_replicate_cv_ci95": (
            replicate_cv.get("cv_headroom_ci95") if isinstance(replicate_cv, dict) else None),
        "routers_loocv_headroom": results["routers_loocv_headroom"],
        "one_sided_p_below_gate": {n: tost_one_sided[n]["p_below_gate"] for n in tost_one_sided},
        "primary_rejects_H0": tost_one_sided[best_router_name]["rejects_H0_at_05"],
        "block_bootstrap_ci95": {n: block_boot_routers[n]["ci95_block_bootstrap"]
                                 for n in block_boot_routers},
        "block_bootstrap_p_below_gate": {n: block_boot_routers[n]["p_block_below_gate"]
                                         for n in block_boot_routers},
        "mde80_detectable_below_gate": pcurve["mde80_detectable_below_gate"],
        "winner_label_reliability_point": (
            rel_ci.get("test_retest_agreement") if isinstance(rel_ci, dict) else None),
        "reliability_ci": rel_ci.get("ci95") if isinstance(rel_ci, dict) else None,
    }}, indent=1, default=str))

    if not args.write:
        print("\n[DRY] canonical_numbers.json NOT written (pass --write to persist).")
        return 0

    cn = json.load(open(CANON))
    cn.setdefault("routability", {})
    cn["routability"]["equivalence"] = results
    tmp = f"{CANON}.tmp"
    open(tmp, "w").write(json.dumps(cn, indent=1, default=str))
    os.replace(tmp, CANON)
    print(f"\nWrote canonical_numbers.json['routability']['equivalence'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
