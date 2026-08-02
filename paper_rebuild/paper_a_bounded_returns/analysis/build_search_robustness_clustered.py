#!/usr/bin/env python
"""Arm-clustered inference for the search-robustness per-query regression (Moulton fix).

The landed search_robustness.cluster_gap_conditioned_on_search.per_query_robustness
regresses per-query GPT-5.2 score on is_cluster + centred retrieval yield over n=399
query rows, but the treatment (is_cluster) varies ONLY across the 11 arms, so the plain
OLS SE (0.0181, CI [0.155, 0.226]) ignores within-arm correlation and is anticonservative
(Moulton 1990). This builder recomputes the SAME regression (rows rebuilt with the
machinery imported from scripts/build_search_robustness.py; identity checked against the
landed coef/n) with cluster-correct inference at G=11 arms:
  * CR1 cluster-robust SE (Liang-Zeger with the usual small-G correction), t(G-1) CI;
  * CR2 (Bell-McCaffrey) SE with the BM/Satterthwaite degrees of freedom, t(df_BM) CI;
  * wild cluster bootstrap over arms, null-imposed, Rademacher weights FULLY ENUMERATED
    (2^11 = 2048 sign vectors -> exact given weights), CR1-studentised: p-value for
    H0: is_cluster coef = 0; plus Webb 6-point weights as a small-G sensitivity;
  * unrestricted percentile-t wild bootstrap CI (Rademacher, enumerated).
Also restates the ARM-level CI with t(8) instead of 1.96 (the arm-level regression has
11 obs and already treats the arm as the unit; its landed CI used a normal quantile).

APPEND-ONLY: lands the NEW subkey canonical_numbers.json['search_robustness']
['per_query_clustered_inference']; refuses to overwrite; atomic write; deterministic
(enumerated Rademacher; Webb seeded).

Usage: python build_search_robustness_clustered.py [--write] [--force]
"""
import importlib.util
import itertools
import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260702
WEBB_B = 4999

spec = importlib.util.spec_from_file_location(
    "bsr", f"{ROOT}/scripts/build_search_robustness.py")
bsr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsr)

from scipy.stats import t as tdist  # noqa: E402  (after module import, like siblings)

cn0 = json.load(open(CANON))
sr = cn0["search_robustness"]
cluster_members = set(sr["cluster_definition"]["cluster_members"])
landed_pq = sr["cluster_gap_conditioned_on_search"]["per_query_robustness"]
landed_arm = sr["cluster_gap_conditioned_on_search"]

# ---------------------------------------------------------------- rebuild the 399 rows
shared = bsr._shared_canon_qids()
rows_score, rows_cluster, rows_yield, rows_arm = [], [], [], []
for dir_name, short in bsr.PAT_DIR.items():
    arm_key = f"base_{short}"
    canon_set = bsr._canon_qids(short) or shared
    tr = bsr.aggregate_traces(short, dir_name, canon_set)
    if not tr or not tr["_per_q_yield"]:
        continue
    scores_q = bsr.load_per_query_gpt52(short)
    if not scores_q:
        continue
    is_cl = 1.0 if arm_key in cluster_members else 0.0
    for qid in sorted(tr["_per_q_yield"]):
        y = tr["_per_q_yield"][qid]
        s = scores_q.get(qid)
        if s is None or y is None or (isinstance(y, float) and math.isnan(y)):
            continue
        rows_score.append(s); rows_cluster.append(is_cl)
        rows_yield.append(y); rows_arm.append(arm_key)

y = np.array(rows_score, float)
cl = np.array(rows_cluster, float)
yl = np.array(rows_yield, float)
yl_c = yl - yl.mean()
X = np.column_stack([np.ones(len(y)), cl, yl_c])
arms = np.array(rows_arm)
uniq_arms = sorted(set(rows_arm))
G = len(uniq_arms)
n, k = X.shape

M = np.linalg.pinv(X.T @ X)
beta = M @ X.T @ y
resid = y - X @ beta
coef = float(beta[1])

# identity check vs the landed per-query regression
ok_n = (n == landed_pq["n_query_rows"])
ok_coef = abs(coef - landed_pq["coef"]) < 5e-4
print(f"[{'OK' if ok_n and ok_coef else 'MISMATCH'}] rows n={n} (landed {landed_pq['n_query_rows']}), "
      f"coef={coef:.4f} (landed {landed_pq['coef']})")
if not (ok_n and ok_coef):
    print("ABORT: row reconstruction does not reproduce the landed regression"); sys.exit(1)

cluster_idx = [np.where(arms == g)[0] for g in uniq_arms]


def cr1_se(Xm, res, Mm):
    meat = np.zeros((k, k))
    for idx in cluster_idx:
        s = Xm[idx].T @ res[idx]
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * Mm @ meat @ Mm
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def cr2_se_and_bmdf(Xm, res, Mm, coef_col=1):
    Hfull = Xm @ Mm @ Xm.T
    A_blocks = {}
    meat = np.zeros((k, k))
    for gi, idx in enumerate(cluster_idx):
        Hgg = Hfull[np.ix_(idx, idx)]
        w, Q = np.linalg.eigh(np.eye(len(idx)) - Hgg)
        w = np.maximum(w, 1e-12)
        Ag = Q @ np.diag(1.0 / np.sqrt(w)) @ Q.T
        A_blocks[gi] = Ag
        s = Xm[idx].T @ (Ag @ res[idx])
        meat += np.outer(s, s)
    V = Mm @ meat @ Mm
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    # Bell-McCaffrey / Satterthwaite df for the coef of interest (Imbens-Kolesar 2016)
    c_vec = np.zeros(k); c_vec[coef_col] = 1.0
    ImH = np.eye(n) - Hfull
    Bg = []
    for gi, idx in enumerate(cluster_idx):
        v = A_blocks[gi] @ (Xm[idx] @ (Mm @ c_vec))     # n_g vector
        b = ImH[:, idx] @ v                              # n vector ((I-H) is symmetric)
        Bg.append(b)
    W = np.array([[float(bi @ bj) for bj in Bg] for bi in Bg])
    lam = np.linalg.eigvalsh(W)
    df_bm = float(lam.sum() ** 2 / (lam ** 2).sum())
    return se, df_bm


se_cr1 = cr1_se(X, resid, M)
t10 = float(tdist.ppf(0.975, G - 1))
ci_cr1 = [round(coef - t10 * se_cr1[1], 4), round(coef + t10 * se_cr1[1], 4)]

se_cr2, df_bm = cr2_se_and_bmdf(X, resid, M)
t_bm = float(tdist.ppf(0.975, df_bm))
ci_cr2 = [round(coef - t_bm * se_cr2[1], 4), round(coef + t_bm * se_cr2[1], 4)]

t_obs = coef / se_cr1[1]

# ------------------------------------------------- wild cluster bootstrap (Rademacher)
# Null-imposed (restricted) CGM bootstrap, CR1-studentised, all 2^11 sign vectors.
Xr = X[:, [0, 2]]                       # restricted model: drop is_cluster
Mr = np.linalg.pinv(Xr.T @ Xr)
beta_r = Mr @ Xr.T @ y
fit_r = Xr @ beta_r
res_r = y - fit_r

signs_iter = itertools.product([-1.0, 1.0], repeat=G)
t_null = np.empty(2 ** G)
fit_u = X @ beta
for bi, sg in enumerate(signs_iter):
    w = np.empty(n)
    for gi, idx in enumerate(cluster_idx):
        w[idx] = sg[gi]
    y_star = fit_r + res_r * w
    b_star = M @ X.T @ y_star
    r_star = y_star - X @ b_star
    se_star = cr1_se(X, r_star, M)
    t_null[bi] = b_star[1] / se_star[1] if se_star[1] > 0 else 0.0
p_wild_rademacher = float(np.mean(np.abs(t_null) >= abs(t_obs) - 1e-12))

# Webb 6-point weights, seeded (small-G sensitivity per Webb 2014 / MacKinnon-Webb)
rng = np.random.default_rng(SEED)
webb = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5),
                 math.sqrt(0.5), 1.0, math.sqrt(1.5)])
cnt = 0
for _ in range(WEBB_B):
    sg = rng.choice(webb, G)
    w = np.empty(n)
    for gi, idx in enumerate(cluster_idx):
        w[idx] = sg[gi]
    y_star = fit_r + res_r * w
    b_star = M @ X.T @ y_star
    r_star = y_star - X @ b_star
    se_star = cr1_se(X, r_star, M)
    tt = b_star[1] / se_star[1] if se_star[1] > 0 else 0.0
    if abs(tt) >= abs(t_obs) - 1e-12:
        cnt += 1
p_wild_webb = float((cnt + 1) / (WEBB_B + 1))

# Wild-bootstrap CI by TEST INVERSION (the MacKinnon-Webb recommended construction):
# CI = { beta0 : null-imposed wild test of H0 beta1=beta0 does not reject at 5% }.
# Enumerated Rademacher draws -> deterministic; endpoints found by bisection.
ALL_SIGNS = list(itertools.product([-1.0, 1.0], repeat=G))
W_MAT = np.empty((2 ** G, n))
for bi, sg in enumerate(ALL_SIGNS):
    for gi, idx in enumerate(cluster_idx):
        W_MAT[bi, idx] = sg[gi]


def wild_p_at(beta0):
    y_t = y - beta0 * cl
    br = Mr @ Xr.T @ y_t
    fit0 = Xr @ br + beta0 * cl
    res0 = y_t - Xr @ br
    tobs = (coef - beta0) / se_cr1[1]
    hits = 0
    for bi in range(2 ** G):
        y_star = fit0 + res0 * W_MAT[bi]
        b_star = M @ X.T @ y_star
        r_star = y_star - X @ b_star
        se_star = cr1_se(X, r_star, M)
        ts = (b_star[1] - beta0) / se_star[1] if se_star[1] > 0 else 0.0
        if abs(ts) >= abs(tobs) - 1e-12:
            hits += 1
    return hits / 2 ** G


def invert_endpoint(lo, hi, iters=22):
    """bisection for the beta0 where p(beta0) crosses 0.05 (p >= .05 inside the CI)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if wild_p_at(mid) >= 0.05:
            lo = mid   # mid inside CI -> move toward outside
        else:
            hi = mid
        if abs(hi - lo) < 1e-4:
            break
    return 0.5 * (lo + hi)


span = 5.0 * se_cr1[1]
# lower endpoint: search between coef (inside) and coef-span (outside)
ci_wild_inv = [round(invert_endpoint(coef, coef - span), 4),
               round(invert_endpoint(coef, coef + span), 4)]
ci_wild_inv = sorted(ci_wild_inv)

# Unrestricted percentile-t wild bootstrap CI (Rademacher, enumerated)
t_unres = np.empty(2 ** G)
for bi, sg in enumerate(itertools.product([-1.0, 1.0], repeat=G)):
    w = np.empty(n)
    for gi, idx in enumerate(cluster_idx):
        w[idx] = sg[gi]
    y_star = fit_u + resid * w
    b_star = M @ X.T @ y_star
    r_star = y_star - X @ b_star
    se_star = cr1_se(X, r_star, M)
    t_unres[bi] = (b_star[1] - coef) / se_star[1] if se_star[1] > 0 else 0.0
q_lo, q_hi = np.percentile(t_unres, [2.5, 97.5])
ci_wild = [round(coef - q_hi * se_cr1[1], 4), round(coef - q_lo * se_cr1[1], 4)]

# arm-level CI restated with t(8) (landed one used 1.96 with dof=8)
arm_coef = landed_arm["coef"]; arm_se = landed_arm["coef_se"]; arm_dof = landed_arm["dof"]
t8 = float(tdist.ppf(0.975, arm_dof))
arm_t_ci = [round(arm_coef - t8 * arm_se, 4), round(arm_coef + t8 * arm_se, 4)]

out = {
    "_note": ("Cluster-correct inference for per_query_robustness (Moulton problem: "
              "is_cluster varies only across the 11 arms, so the landed plain-OLS SE "
              "0.0181 / CI [0.1547, 0.2256] over n=399 rows is anticonservative). Rows "
              "rebuilt with the machinery of scripts/build_search_robustness.py and "
              "identity-checked against the landed coef/n. G=11 clusters (arms). The "
              "plain-OLS per-query CI should be read as DESCRIPTIVE; cluster-level "
              "inference here is the citable interval. Deterministic: Rademacher wild "
              "bootstrap fully enumerated (2^11=2048); Webb seeded."),
    "seed": SEED,
    "model": "OLS gpt52_overall_score ~ 1 + is_cluster + centred(docs_per_attempt_q)",
    "n_query_rows": int(n),
    "n_clusters_G": int(G),
    "coef": round(coef, 4),
    "plain_ols": {"se": landed_pq["coef_se"], "ci": landed_pq["ci"],
                  "status": "DESCRIPTIVE (ignores within-arm correlation)"},
    "cr1": {"se": round(float(se_cr1[1]), 4), "t_crit": round(t10, 3),
            "df": G - 1, "ci95": ci_cr1,
            "excludes_zero": bool(ci_cr1[0] > 0 or ci_cr1[1] < 0)},
    "cr2_bell_mccaffrey": {"se": round(float(se_cr2[1]), 4),
                           "df_bm_satterthwaite": round(df_bm, 2),
                           "t_crit": round(t_bm, 3), "ci95": ci_cr2,
                           "excludes_zero": bool(ci_cr2[0] > 0 or ci_cr2[1] < 0)},
    "wild_cluster_bootstrap": {
        "type": "null-imposed CGM, CR1-studentised",
        "rademacher_enumerated_p": round(p_wild_rademacher, 4),
        "n_rademacher": 2 ** G,
        "webb_p": round(p_wild_webb, 4),
        "n_webb": WEBB_B,
        "test_inversion_ci95": ci_wild_inv,
        "test_inversion_ci_excludes_zero": bool(ci_wild_inv[0] > 0 or ci_wild_inv[1] < 0),
        "percentile_t_ci95_unrestricted": ci_wild,
        "percentile_t_ci_excludes_zero": bool(ci_wild[0] > 0 or ci_wild[1] < 0),
        "_ci_note": ("test_inversion_ci95 is the recommended wild-cluster CI "
                     "(MacKinnon-Webb): the set of beta0 not rejected by the null-imposed "
                     "studentised test at 5%. The unrestricted percentile-t interval is a "
                     "cruder construction, reported as sensitivity."),
    },
    "arm_level_t_ci_restated": {
        "_note": "landed arm-level CI [0.1647, 0.3605] used 1.96 with dof=8; t(8) CI here",
        "coef": arm_coef, "se": arm_se, "df": arm_dof,
        "ci95_t": arm_t_ci,
        "excludes_zero": bool(arm_t_ci[0] > 0 or arm_t_ci[1] < 0),
    },
}
primary_ok = (ci_cr1[0] > 0 and ci_cr2[0] > 0 and p_wild_rademacher < 0.05
              and p_wild_webb < 0.05 and ci_wild_inv[0] > 0)
out["verdict"] = (
    f"is_cluster coef {coef:+.4f}: CR1 CI {ci_cr1} (t(10)), CR2/Bell-McCaffrey CI {ci_cr2} "
    f"(df_BM={df_bm:.1f}), wild-cluster (null-imposed, Rademacher enumerated) p = "
    f"{p_wild_rademacher:.4f}, Webb p = {p_wild_webb:.4f}, test-inversion wild CI "
    f"{ci_wild_inv}. "
    + (("SURVIVES honest G=11 inference at the 5% level under every cluster-correct "
        "construction, but only MARGINALLY (CR2 lower bound "
        f"{ci_cr2[0]:+.4f}; the cruder unrestricted percentile-t interval {ci_wild} "
        "just crosses zero). The plain-OLS per-query CI [0.1547, 0.2256] is DEMOTED TO "
        "DESCRIPTIVE and must not be quoted as inference; quote the CR1 CI "
        f"{ci_cr1} (or the arm-level t(8) CI {arm_t_ci}) instead, and read the per-query "
        "regression as a confirmation of the arm-level result, not as independent "
        "precision.")
       if primary_ok
       else ("Cluster-correct constructions do NOT all reject zero: DEMOTE the per-query "
             "result to descriptive and lean on the arm-level regression "
             f"(t(8) CI {arm_t_ci}) only.")))

print(json.dumps({kk: out[kk] for kk in
                  ["coef", "cr1", "cr2_bell_mccaffrey", "wild_cluster_bootstrap",
                   "arm_level_t_ci_restated", "verdict"]}, indent=1))

if "--write" in sys.argv:
    cn = json.load(open(CANON))  # fresh read: keep read-modify-write window short
    if "per_query_clustered_inference" in cn["search_robustness"] and "--force" not in sys.argv:
        print("[REFUSING] search_robustness.per_query_clustered_inference exists (use --force)")
        sys.exit(1)
    cn["search_robustness"]["per_query_clustered_inference"] = out
    fd, tmp = tempfile.mkstemp(dir=ANA, prefix="canonical_numbers.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[WROTE search_robustness.per_query_clustered_inference "
          f"(store {len(cn)} top-level keys preserved)]")
else:
    print("[DRY-RUN: no write; pass --write to land the key]")
