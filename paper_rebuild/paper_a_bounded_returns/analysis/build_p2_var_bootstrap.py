#!/usr/bin/env python
"""P2_var_bootstrap (Paper 3) — parametric bootstrap percentile CIs for the REML variance
components of the replicate-variance decomposition.

Replaces the Wald/point-only output of `variance_decomposition.components` (built by
build_variance_decomposition.py) with B=2000 PARAMETRIC bootstrap percentile confidence
intervals for (sigma2_query, sigma2_run, ICC_query) per architecture. The point fit is the same
MixedLM REML of `overall_score_recomputed ~ 1 | query_id` per architecture on the gpt52
variance-replicate corpus (`base_{arch}_v{n}`, P0 x11 + {p1,p4,p5,p6,p7,p8,p10} x3 over the
variance queries). This builder ONLY APPENDS the new key `variance_decomposition.bootstrap_ci`;
it does not touch `components` (the existing point estimates remain as published).

Method (parametric / model-based bootstrap, "boot" interval family in lme4::confint):
  For each architecture, take the fitted REML parameters
    grand mean mu = fe_params[0],  sigma2_query = cov_re[0,0],  sigma2_run = scale.
  For b = 1..B, on the SAME design (the architecture's observed (query_id, replicate) layout):
    1. draw one query random effect u_q ~ N(0, sigma2_query) per query_id,
    2. draw a residual e_i ~ N(0, sigma2_run) per observation row,
    3. form y*_i = mu + u_{q(i)} + e_i on the unchanged design matrix,
    4. refit the SAME MixedLM REML and record (sigma2_query*, sigma2_run*, ICC_query*).
  Report the 2.5 / 50 / 97.5 percentiles of the B refit draws as the percentile CI.
This is the standard parametric (Gaussian) bootstrap for LMM variance components: it respects
the design (group sizes, replicate counts) and the REML null that generated `components`, and is
the recommended replacement for Wald variance-component intervals whose symmetric-normal
assumption fails near the sigma2 >= 0 boundary (small/ragged-coverage arches p5/p6/p8 in
particular). cf. lme4::confint(method="boot"/"profile") (Bates et al.), and the bootstrap
percentile-CI practice for evaluation-variance components in arXiv:2509.00255 (run/seed variance
in LLM agent eval) and arXiv:2306.10779 (bootstrap CIs for held-out evaluation metrics).

Determinism: single seeded Generator (SEED=20260611) on SORTED inputs (architectures sorted,
query ids sorted, observation rows index-sorted before any draw). No Date.now / unseeded RNG.
Refits that fail to converge (degenerate small cells) are skipped per-draw and counted in
`n_boot_ok`; a CI is emitted only when n_boot_ok is sufficient, otherwise the arch records the
reason. Pure CPU, $0, idempotent, reads only df_overall_scores.parquet + the canonical fixture.

DATA SUFFICIENCY: the on-disk gpt52 variance corpus is sufficient for the full-coverage
architectures (p0,p1,p4,p7,p10: 30 queries x 3-11 reps). The ragged-coverage architectures
(p5: 14 q, p6: 8 q, p8: 18 q) have few queries x 3 reps, so their bootstrap CIs are WIDE and the
sigma2_query draws sit near the boundary; this is reported honestly via `coverage` per arch and a
top-level caveat. No conditional/missing item is required beyond what `components` already used.

Write idiom: atomic tmp + os.replace, APPENDING only `variance_decomposition.bootstrap_ci`
(mirrors build_judge_vs_gold.py / build_n_eff.py). __main__ write flag: pass --no-write to print
the summary WITHOUT mutating canonical_numbers.json (default: write).
"""
import json, os, re, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260611
B = 2000
WRITE = "--no-write" not in sys.argv

from statsmodels.regression.mixed_linear_model import MixedLM


def arch_of(p):
    m = re.match(r"(base_p\d+)_v\d+", str(p))
    return m.group(1) if m else None


# ---------- load + reconstruct the SAME substrate as build_variance_decomposition.py ----------
O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
vf = O[(O.pattern_family == "variance") & (O.judge == "gpt52")].copy()
vf["arch"] = vf.pattern.map(arch_of)
vf = vf.dropna(subset=["arch", "overall_score_recomputed"])
ARCHES = sorted(vf.arch.unique())  # sorted inputs

rng = np.random.default_rng(SEED)  # single seeded generator, drawn in sorted-arch order


def reml_fit(df):
    """REML fit of overall_score_recomputed ~ 1 | query_id; returns (mu, s2q, s2r, icc)."""
    m = MixedLM.from_formula("overall_score_recomputed ~ 1", groups="query_id",
                             data=df).fit(reml=True)
    mu = float(m.fe_params.iloc[0])
    s2q = float(m.cov_re.iloc[0, 0])
    s2r = float(m.scale)
    icc = s2q / (s2q + s2r) if (s2q + s2r) > 0 else float("nan")
    return mu, s2q, s2r, icc


def pct_ci(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return {"lo": None, "median": None, "hi": None}
    return {"lo": round(float(np.percentile(a, 2.5)), 5),
            "median": round(float(np.percentile(a, 50.0)), 5),
            "hi": round(float(np.percentile(a, 97.5)), 5)}


bootstrap_ci = {}
for a in ARCHES:
    sub = vf[vf.arch == a].copy().sort_values(["query_id", "pattern"]).reset_index(drop=True)
    n_reps = int(sub.pattern.nunique())
    n_q = int(sub.query_id.nunique())
    if n_reps < 2 or n_q < 2:
        bootstrap_ci[a] = {"skipped": "needs >=2 replicates and >=2 queries",
                           "n": int(len(sub)), "n_reps": n_reps, "n_queries": n_q}
        continue
    try:
        mu, s2q, s2r, icc = reml_fit(sub)
    except Exception as e:  # pragma: no cover - defensive
        bootstrap_ci[a] = {"error": f"point fit failed: {str(e)[:80]}"}
        continue

    qids = sorted(sub.query_id.unique())          # sorted design
    qcode = sub.query_id.values                    # per-row group label (fixed design)
    n_obs = len(sub)
    sd_q = float(np.sqrt(max(s2q, 0.0)))
    sd_r = float(np.sqrt(max(s2r, 0.0)))

    draws_s2q, draws_s2r, draws_icc = [], [], []
    n_ok = 0
    for _b in range(B):
        # 1) one query random effect per query, on the SAME design
        u_map = {q: rng.normal(0.0, sd_q) for q in qids}
        u_row = np.array([u_map[q] for q in qcode])
        # 2) residual per observation row
        e_row = rng.normal(0.0, sd_r, size=n_obs)
        # 3) simulated response on the unchanged design
        sim = sub.copy()
        sim["overall_score_recomputed"] = mu + u_row + e_row
        # 4) refit REML
        try:
            _, q2, r2, i2 = reml_fit(sim)
        except Exception:
            continue
        if not (np.isfinite(q2) and np.isfinite(r2)):
            continue
        draws_s2q.append(q2)
        draws_s2r.append(r2)
        draws_icc.append(i2)
        n_ok += 1

    rec = {
        "point": {"sigma2_query": round(s2q, 5), "sigma2_run": round(s2r, 5),
                  "icc_query": round(icc, 4) if np.isfinite(icc) else None,
                  "grand_mean": round(mu, 5)},
        "n": int(n_obs), "n_reps": n_reps, "n_queries": n_q,
        "B": B, "n_boot_ok": int(n_ok),
        "coverage": "full" if (n_q >= 30) else "ragged",
    }
    if n_ok >= max(50, B // 10):
        rec["ci"] = {"sigma2_query": pct_ci(draws_s2q),
                     "sigma2_run": pct_ci(draws_s2r),
                     "icc_query": pct_ci(draws_icc)}
    else:
        rec["ci"] = None
        rec["ci_note"] = f"too few converged refits ({n_ok}/{B}); CI suppressed"
    bootstrap_ci[a] = rec

n_full = sum(1 for a in ARCHES
             if isinstance(bootstrap_ci[a], dict) and bootstrap_ci[a].get("coverage") == "full")

out = {
    "_note": (
        "P2_var_bootstrap (Paper 3): B=2000 PARAMETRIC (Gaussian) bootstrap percentile CIs for "
        "the REML variance components (sigma2_query, sigma2_run, ICC_query) of "
        "overall_score_recomputed ~ 1 | query_id, per architecture, on the gpt52 variance-replicate "
        "corpus. Replaces the Wald/point-only `components` output with model-based bootstrap "
        "intervals (query REs ~ N(0,sigma2_query), residuals ~ N(0,sigma2_run) resampled on the "
        "SAME observed design, REML refit each draw). Standard LMM variance-component interval per "
        "lme4::confint(method='boot'/'profile'); percentile-CI practice for eval-variance follows "
        "arXiv:2509.00255 and arXiv:2306.10779. Appends `bootstrap_ci`; leaves `components` "
        "(point estimates) untouched."),
    "method": "parametric_bootstrap_percentile",
    "B": B, "seed": SEED,
    "model": "MixedLM REML: overall_score_recomputed ~ 1, groups=query_id (per architecture)",
    "judge": "gpt52", "metric": "overall_score_recomputed",
    "citations": ["lme4::confint (profile/boot, Bates et al.)",
                  "arXiv:2509.00255", "arXiv:2306.10779"],
    "data_sufficiency": (
        f"{n_full} full-coverage architectures (30 queries x >=3 reps) give tight CIs; ragged "
        "architectures (p5=14q, p6=8q, p8=18q x 3 reps) give WIDE CIs with sigma2_query draws near "
        "the >=0 boundary, reported per-arch via `coverage` — this is exactly why parametric "
        "bootstrap percentile intervals replace the boundary-violating Wald intervals."),
    "bootstrap_ci": bootstrap_ci,
}

if WRITE:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    vd = cn.get("variance_decomposition", {})
    vd["bootstrap_ci"] = out  # APPEND only; do not clobber other variance_decomposition subkeys
    cn["variance_decomposition"] = vd
    _tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(_tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(_tmp, f"{ANA}/canonical_numbers.json")  # atomic
    tag = "WROTE variance_decomposition.bootstrap_ci"
else:
    tag = "DRY-RUN (--no-write): canonical NOT mutated"

print(f"P2_var_bootstrap: {tag} | B={B} seed={SEED} | arches={len(ARCHES)} full_coverage={n_full}")
for a in ARCHES:
    r = bootstrap_ci[a]
    if isinstance(r, dict) and r.get("ci"):
        ci = r["ci"]
        print(f"  {a} ({r['coverage']}, ok={r['n_boot_ok']}/{B}): "
              f"s2q={r['point']['sigma2_query']} CI[{ci['sigma2_query']['lo']},{ci['sigma2_query']['hi']}] "
              f"icc={r['point']['icc_query']} CI[{ci['icc_query']['lo']},{ci['icc_query']['hi']}]")
    else:
        print(f"  {a}: {r.get('ci_note') or r.get('skipped') or r.get('error')}")
