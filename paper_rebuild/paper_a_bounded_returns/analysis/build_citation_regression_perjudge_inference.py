#!/usr/bin/env python
"""Admissible inference for the PER-JUDGE and QUERY-FE citation regressions.

The pooled fit (canonical_numbers.json['citation_regression']) already reports
query-clustered (G=90) asymptotic p-values plus a wild-cluster restricted (WCR)
bootstrap at the pattern grouping (G=12), because asymptotic cluster-robust SEs
are anticonservative with only 12 clusters. But the PER-JUDGE refits
(['density_per_judge']) and the QUERY-FE refit (['citation_regression']['query_fe'])
were quoted with G=12 ASYMPTOTIC pattern-clustered p-values -- exactly the
inference the paper itself rules inadmissible.

This script adds, for each per-judge fit and the query-FE fit:
  (a) query-clustered (G~=90) asymptotic p-values (admissible grouping), and
  (b) WCR bootstrap p-values at the pattern grouping, Rademacher weights
      ENUMERATED over all 2^G sign patterns (G=12 -> 4096; claude_opus G=11 ->
      2048, Opus never scored base_p11). Exact randomisation test, fully
      deterministic, no MC seed needed; includes the identity pattern so
      p >= 2^-G. t-statistics use the same CRVE1 cluster covariance as
      statsmodels cov_type='cluster' (verified by assertion against statsmodels
      on the observed data).

NEW keys only -- never overwrites or deletes existing keys:
  density_per_judge.<judge>.p_density_cluster_query
  density_per_judge.<judge>.p_provenance_cluster_query
  density_per_judge.<judge>.p_density_wild_cluster_pattern
  density_per_judge.<judge>.p_provenance_wild_cluster_pattern
  density_per_judge.<judge>.g_query_clusters / .g_pattern_clusters
  density_per_judge.inference_note
  citation_regression.query_fe.density.p_wild_cluster_pattern
  citation_regression.query_fe.provenance.p_wild_cluster_pattern
  citation_regression.query_fe.inference_note
"""
import pandas as pd, numpy as np, json, warnings, sys
import statsmodels.formula.api as smf
import patsy
warnings.filterwarnings("ignore")
WRITE = "--write" in sys.argv
ROOT = "."; A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

# ---- rebuild the EXACT dataframes of build_citation_regression.py ----
# (same reuse convention as build_review_robustness.py: exec the source prefix
#  up to the first model fit, which yields `feat` and the pooled `df`)
src = open(f"{ANA}/build_citation_regression.py").read()
_source_namespace = globals()
exec(src.split("m=smf.ols")[0], _source_namespace)
feat = _source_namespace["feat"]
df = _source_namespace["df"]

# per-judge factual_accuracy frames -- replicated VERBATIM from
# build_citation_regression.py (its per-judge block sits after the exec cut)
sc_all = pd.read_parquet(f"{A}/df_scores.parquet")
sc_all = sc_all[sc_all.pattern.str.match(r"^base_p([0-9]|10)$") & sc_all.dimension.eq("factual_accuracy")]
def _judge_factual(j):
    d = sc_all[sc_all.judge == j]
    return d.groupby(["pattern", "query_id"], observed=True)["score"].mean().rename("fa")

# ---- exact WCR bootstrap: enumerate all 2^G Rademacher sign patterns ----
def wcr_enum_p(formula, data, term, group_col="pattern"):
    """Wild-cluster restricted bootstrap-t p for `term`, clustering by `group_col`.

    Enumerates all 2^G per-cluster Rademacher sign vectors applied to the
    restricted (null-imposed) model residuals; t-stats use CRVE1 with the
    statsmodels small-sample correction c = G/(G-1) * (n-1)/(n-k). Exact and
    deterministic (sorted cluster order, full enumeration -- no RNG).
    Returns (t_obs, p, G, R)."""
    y_dm, X_dm = patsy.dmatrices(formula, data, return_type="dataframe")
    cols = list(X_dm.columns); jix = cols.index(term)
    X = X_dm.to_numpy(); yv = y_dm.to_numpy().ravel()
    n, k = X.shape
    groups = data[group_col].astype(str).to_numpy()
    clusters = sorted(set(groups)); G = len(clusters)
    gidx = [np.flatnonzero(groups == c) for c in clusters]
    XtXi = np.linalg.inv(X.T @ X)
    H = XtXi @ X.T
    corr = G / (G - 1) * (n - 1) / (n - k)
    aj = XtXi[jix]
    zg = [X[ix] @ aj for ix in gidx]  # per-cluster projections -> V_jj directly

    def tstat(Y):  # Y: n x R matrix of outcomes
        B = H @ Y
        U = Y - X @ B
        Vjj = np.zeros(Y.shape[1])
        for z, ix in zip(zg, gidx):
            Vjj += (z @ U[ix]) ** 2
        return B[jix] / np.sqrt(corr * Vjj)

    t_obs = float(tstat(yv[:, None])[0])
    # verify our CRVE1 t matches statsmodels cov_type='cluster' exactly
    smt = float(smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[group_col].astype(str).values}).tvalues[term])
    assert np.isclose(t_obs, smt, rtol=1e-6), (term, t_obs, smt)

    # restricted model: drop the tested term (impose beta_term = 0)
    rhs = [x.strip() for x in formula.split("~", 1)[1].split("+")]
    restr = formula.split("~", 1)[0] + "~" + " + ".join(x for x in rhs if x.strip() != term)
    rfit = smf.ols(restr, data=data).fit()
    fitted = rfit.fittedvalues.to_numpy(); resid = rfit.resid.to_numpy()

    R = 2 ** G  # full enumeration (G=12 -> 4096)
    W = (((np.arange(R)[:, None] >> np.arange(G)) & 1) * 2 - 1).astype(np.float64)
    Y = np.empty((n, R))
    for g, ix in enumerate(gidx):
        Y[ix] = fitted[ix, None] + resid[ix, None] * W[:, g][None, :]
    tb = tstat(Y)
    p = float(np.mean(np.abs(tb) >= abs(t_obs) - 1e-12))
    return t_obs, p, G, R

# ---- 1. per-judge fits: query-clustered asymptotic + WCR(pattern, enumerated) ----
FORM_J = "fa ~ provenance_rate + log_cit + log_words + C(pattern)"
perjudge_new = {}
for j in ["gpt52", "claude_opus", "claude_sonnet"]:
    dj = feat.merge(_judge_factual(j), on=["pattern", "query_id"])
    dj = dj[dj.n_cit > 0].copy()
    dj["log_cit"] = np.log(dj.n_cit); dj["log_words"] = np.log(dj.words.clip(lower=1))
    dj["pattern"] = dj.pattern.astype(str)
    mjq = smf.ols(FORM_J, data=dj).fit(
        cov_type="cluster", cov_kwds={"groups": dj["query_id"].astype(str).values})
    t_d, p_wd, Gp, R = wcr_enum_p(FORM_J, dj, "log_cit", "pattern")
    t_p, p_wp, _, _ = wcr_enum_p(FORM_J, dj, "provenance_rate", "pattern")
    perjudge_new[j] = {
        "p_density_cluster_query": float(mjq.pvalues["log_cit"]),
        "p_provenance_cluster_query": float(mjq.pvalues["provenance_rate"]),
        "p_density_wild_cluster_pattern": round(p_wd, 6),
        "p_provenance_wild_cluster_pattern": round(p_wp, 6),
        "g_query_clusters": int(dj["query_id"].nunique()),
        "g_pattern_clusters": int(Gp),
    }
    print(f"[{j}] n={len(dj)} G_q={perjudge_new[j]['g_query_clusters']} G_p={Gp} "
          f"density: p_q={mjq.pvalues['log_cit']:.5g} p_wild={p_wd:.5g} | "
          f"provenance: p_q={mjq.pvalues['provenance_rate']:.5g} p_wild={p_wp:.5g}")

# ---- 2. query-FE fit: WCR(pattern, enumerated) + verify stored asymptotics ----
FORM_Q = "factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern) + C(query_id)"
m_pat = smf.ols(FORM_Q, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["pattern"].astype(str).values})
m_qry = smf.ols(FORM_Q, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["query_id"].astype(str).values})
qfe_new = {}
for name, t in [("density", "log_cit"), ("provenance", "provenance_rate")]:
    _, p_w, Gp, R = wcr_enum_p(FORM_Q, df, t, "pattern")
    qfe_new[name] = {"p_wild_cluster_pattern": round(p_w, 6)}
    print(f"[query_fe/{name}] beta={m_pat.params[t]:.4f} "
          f"p_pat_asym={m_pat.pvalues[t]:.5g} p_qry_asym={m_qry.pvalues[t]:.5g} "
          f"p_wild_pattern={p_w:.5g} (G={Gp}, R={R})")

# ---- 3. cross-check: pooled model, enumerated WCR vs stored 1999-rep values ----
FORM_P = "factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern)"
for t in ["provenance_rate", "log_cit"]:
    _, p_w, _, _ = wcr_enum_p(FORM_P, df, t, "pattern")
    print(f"[pooled xcheck/{t}] p_wild_enum={p_w:.5g} "
          f"(stored reps=1999 value in citation_regression.coefs for comparison)")

INF_NOTE = ("Per-judge/query-FE inference upgrade (2026-07-02): the *_cluster_pattern "
            "p-values above are few-cluster (G<=12) ASYMPTOTIC and anticonservative "
            "(see citation_regression.se_note) -- do not quote them. Quote "
            "p_*_cluster_query (G=90, admissible asymptotic) and/or "
            "p_*_wild_cluster_pattern (WCR bootstrap-t, Rademacher signs ENUMERATED "
            "over all 2^G pattern sign patterns -- G=12 i.e. 4096 draws, except "
            "claude_opus where G=11 (Opus never scored base_p11) i.e. 2048; exact, "
            "deterministic, p >= 2^-G).")

# ---- land NEW keys only (update-not-overwrite, verified) ----
import copy
store = json.load(open(f"{ANA}/canonical_numbers.json"))
before = copy.deepcopy(store)
for j, upd in perjudge_new.items():
    for k_, v_ in upd.items():
        store["density_per_judge"][j].setdefault(k_, v_)
store["density_per_judge"].setdefault("inference_note", INF_NOTE)
for name, upd in qfe_new.items():
    for k_, v_ in upd.items():
        store["citation_regression"]["query_fe"][name].setdefault(k_, v_)
store["citation_regression"]["query_fe"].setdefault(
    "inference_note",
    INF_NOTE + " For query_fe, p_cluster_query is the admissible asymptotic value; "
    "p_cluster_pattern (the paper's previously quoted 0.043/0.019) is inadmissible.")

def _assert_superset(old, new, path=""):
    """Every pre-existing key/value must survive unchanged (NaN-aware)."""
    if isinstance(old, dict):
        for k_ in old:
            assert k_ in new, f"deleted key {path}.{k_}"
            _assert_superset(old[k_], new[k_], f"{path}.{k_}")
    elif isinstance(old, list):
        assert isinstance(new, list) and len(old) == len(new), f"changed list at {path}"
        for i_, (o_, n_) in enumerate(zip(old, new)):
            _assert_superset(o_, n_, f"{path}[{i_}]")
    else:
        same = (old == new) or (isinstance(old, float) and isinstance(new, float)
                                and np.isnan(old) and np.isnan(new))
        assert same, f"changed value at {path}: {old!r} -> {new!r}"
_assert_superset(before, store)

if WRITE:
    json.dump(store, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
    print("[WROTE] canonical_numbers.json (new keys only; all pre-existing keys verified unchanged)")
else:
    print("[DRY-RUN] no write (pass --write to persist)")
print("perjudge_new:", json.dumps(perjudge_new, indent=1))
print("qfe_new:", json.dumps(qfe_new, indent=1))
