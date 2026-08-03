#!/usr/bin/env python
"""
Canonical numbers for the world-class rewrite.
Single source of truth: recomputes EVERY headline statistic from the current
released parquets so the paper can never drift from the data again.

Run:  ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/build_numbers.py
Out:  paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json
      paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.md

Key data quirk handled everywhere: claude_sonnet's stored `overall_score` is
corrupted -> we use `overall_score_recomputed` for sonnet rows (per DATA_DICTIONARY).
"""
import json, warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
A = f"{ROOT}/data/analysis"
OUT = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
DIMS = ["information_recall","factual_accuracy","coverage","analytical_depth",
        "citation_quality","logical_coherence","organization",
        "instruction_following","attribution_quality"]
R = {}   # results
def section(name, fn):
    try:
        R[name] = fn(); print(f"[ok] {name}")
    except Exception as e:
        R[name] = {"_error": f"{type(e).__name__}: {e}"}; print(f"[ERR] {name}: {e}")

def corrected_overall(df):
    c = df["overall_score"].copy()
    m = df["judge"].eq("claude_sonnet")
    if "overall_score_recomputed" in df.columns:
        c = c.where(~m, df["overall_score_recomputed"])
    return c

# ---------- load ----------
ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
ov["ovc"] = corrected_overall(ov)
sc = pd.read_parquet(f"{A}/df_scores.parquet")

def is_base_headline(p):  # base_p0..base_p10 (3-judge); exclude variance reruns _v1/_v2/_v3
    return p.startswith("base_p") and not p.rstrip("0123456789").endswith("_v")

# ---------- PPI++ / prediction-powered debiased mean (method upgrade) ----------
# arXiv:2601.05420 (PPI++/EIF). The headline estimand is the 3-judge panel mean. The cheap,
# ABUNDANT single judge (gpt52) scores every base report and serves as the *prediction*; the
# smaller intersection where the full panel labelled the report is the *gold* subset, with the
# gold label = the panel mean itself. We report the PPI++ rectifier-form debiased mean
#     mu_ppi = mean_{all preds}(gpt52) + mean_{labelled}(gold - gpt52)
# ALONGSIDE (never replacing) the raw means, with a query-level i.i.d. bootstrap CI (the
# all-predictions and labelled-correction draws are independent => resampled independently).
# ASSUMPTION (documented): when gpt52 coverage == the labelled panel (n_pred == n_lab on the
# same queries, true for the full-panel patterns p0..p10) the rectifier is algebraically the
# raw panel mean, so mu_ppi == mean_3judge there; debiasing only moves patterns with unlabelled
# gpt52 predictions (p1/p3/p5/p6 marginally; p11/p12, which lack the opus rater, use the
# available {gpt52,sonnet} panel as gold and are flagged panel_size=2). Deterministic: fixed
# seed, sorted query index, fixed reps.
PPI_SEED = 20260622
PPI_REPS = 2000
def _ppi_debiased(g):
    avail = [j for j in PANEL if g.judge.eq(j).any()]          # panel judges present for this pattern
    w = g.pivot_table(index="query_id", columns="judge", values="ovc", observed=True)
    w = w.sort_index()                                         # sorted inputs -> deterministic
    pred_all = w["gpt52"].dropna()                             # gpt52 prediction on EVERY report
    complete = w[avail].dropna()                               # gold-labelled subset (full panel)
    if len(complete) == 0 or len(pred_all) == 0:
        return None
    gold = complete.mean(axis=1)                               # gold label = panel mean
    pred_lab = complete["gpt52"]
    rectifier = float((gold - pred_lab).mean())
    point = float(pred_all.mean()) + rectifier
    rng = np.random.default_rng(PPI_SEED)
    pa = pred_all.to_numpy(); gl = gold.to_numpy(); pl = pred_lab.to_numpy()
    nA, nL = len(pa), len(gl)
    boot = np.empty(PPI_REPS)
    for i in range(PPI_REPS):
        bp = pa[rng.integers(0, nA, nA)].mean()                # bootstrap the abundant predictions
        idx = rng.integers(0, nL, nL)                          # independently bootstrap the labelled correction
        boot[i] = bp + (gl[idx] - pl[idx]).mean()
    return {
        "ppi_mean": round(point, 4),
        "rectifier": round(rectifier, 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                 round(float(np.percentile(boot, 97.5)), 4)],
        "panel_size": len(avail),
        "n_pred": int(nA),
        "n_labelled": int(nL),
        "n_unlabelled": int(nA - nL),
    }

# ---------- headline per-pattern means ----------
def headline():
    base = ov[ov.pattern.str.match(r"^base_p\d+$")].copy()
    panel = base[base.judge.isin(PANEL)]
    rows = {}
    ppi = {}
    for pat, g in panel.groupby("pattern", observed=True):
        pj = g.groupby("judge", observed=True)["ovc"].mean()
        pj_raw = g.groupby("judge", observed=True)["overall_score"].mean()
        rows[pat] = {
            "n_cells": int(len(g)),
            "n_queries": int(g.query_id.nunique()),
            "mean_3judge": round(float(g["ovc"].mean()), 4),
            "std_3judge": round(float(g["ovc"].std()), 4),
            "mean_gpt52": round(float(pj.get("gpt52", np.nan)), 4),
            "mean_opus": round(float(pj.get("claude_opus", np.nan)), 4),
            "mean_sonnet_corrected": round(float(pj.get("claude_sonnet", np.nan)), 4),
            "mean_sonnet_raw": round(float(pj_raw.get("claude_sonnet", np.nan)), 4),
        }
        rec = _ppi_debiased(g)
        if rec is not None:
            rows[pat]["ppi_debiased"] = rec          # companion field, alongside the raw means
            ppi[pat] = rec["ppi_mean"]
    # rank by 3-judge mean
    order = sorted(rows, key=lambda k: -rows[k]["mean_3judge"])
    # rank by PPI++ debiased mean (to report whether debiasing reorders any pattern)
    order_ppi = sorted(ppi, key=lambda k: -ppi[k])
    raw_order_on_ppi_set = [p for p in order if p in ppi]
    rank_changed = (raw_order_on_ppi_set != order_ppi)
    # A "reduced-panel" pattern is one whose panel is genuinely smaller than the full
    # 3-judge panel (p11/p12: 2-judge). Patterns with only a stray 1-report gap
    # (e.g. p1/p3, n_unlabelled==1 on an otherwise full 3-judge panel) are treated as
    # effectively full-panel for this caveat, since the rectifier there is still ~the
    # raw panel mean up to one report.
    reduced = sorted(p for p in ppi
                     if rows[p].get("ppi_debiased", {}).get("panel_size", 3) < 3)
    n_reduced = len(reduced)
    return {"per_pattern": rows, "rank_desc": order,
            "rank_desc_ppi": order_ppi,
            "ppi_rank_changed_vs_raw": bool(rank_changed),
            "_ppi_note": "PPI++/EIF rectifier (arXiv:2601.05420): gpt52=prediction (covers all "
                         "reports), full panel=gold label; mu_ppi=mean(pred)+mean(gold-pred|labelled). "
                         "Companion to mean_3judge, not a replacement.",
            "_ppi_caveat": (
                "IMPORTANT: the PPI 'gold' anchor is the PANEL MEAN, not external truth/gold. "
                "So this is PANEL-anchored debiasing (gpt52 -> panel mean), NOT truth/gold-anchored "
                "debiasing. For the full-panel patterns (every report scored by all panel judges, "
                "n_unlabelled=0) the rectifier is ALGEBRAICALLY the raw panel mean_3judge "
                "(mean(pred)+mean(panel-pred)=mean(panel)), so PPI does NO real work there; it "
                f"corrects gpt52->panel and does real work ONLY on the {n_reduced} REDUCED-PANEL "
                f"patterns ({', '.join(reduced) if reduced else 'none'}; 2-judge panels with many "
                "unlabelled reports), where the abundant gpt52 predictions extend the panel-mean "
                "estimate onto reports the reduced panel never scored. (p1/p3 carry only a stray "
                "1-report gap on an otherwise full 3-judge panel and are effectively full-panel.) "
                "This is the lambda=1 rectifier (rectifier applied in full), NOT lambda-tuned "
                "PPI++.")}

# ---------- single-judge (gpt52) means incl p11/p12 ----------
def single_judge_gpt52():
    base = ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.eq("gpt52")].copy()
    rows = {}
    for pat, g in base.groupby("pattern", observed=True):
        rows[pat] = {"n": int(len(g)),
                     "mean": round(float(g.overall_score.mean()), 4),
                     "std": round(float(g.overall_score.std()), 4)}
    order = sorted(rows, key=lambda k: -rows[k]["mean"])
    return {"per_pattern": rows, "rank_desc": order}

# ---------- per-dimension means (3-judge, simple mean of per-dim scores) ----------
def per_dimension():
    base = sc[sc.pattern.str.match(r"^base_p\d+$") & sc.judge.isin(PANEL)].copy()
    tab = (base.groupby(["pattern","dimension"], observed=True)["score"].mean()
              .unstack().reindex(columns=DIMS))
    return {p: {d: round(float(tab.loc[p, d]), 4) for d in DIMS}
            for p in tab.index if p.startswith("base_p")}

# ---------- IRR ----------
def irr():
    import krippendorff
    base = ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(PANEL)].copy()
    w = base.pivot_table(index=["pattern","query_id"], columns="judge",
                         values="ovc", observed=True).dropna()
    X = w[PANEL].values
    n, k = X.shape
    gm = X.mean()
    MSR = k*((X.mean(1)-gm)**2).sum()/(n-1)
    MSC = n*((X.mean(0)-gm)**2).sum()/(k-1)
    MSE = ((X - X.mean(1,keepdims=True) - X.mean(0,keepdims=True) + gm)**2).sum()/((n-1)*(k-1))
    icc_ak = (MSR-MSE)/(MSR+(MSC-MSE)/n)
    icc_a1 = (MSR-MSE)/(MSR+(k-1)*MSE+(k/n)*(MSC-MSE))
    alpha_overall = krippendorff.alpha(reliability_data=w[PANEL].T.values,
                                       level_of_measurement="interval")
    # per-dimension alpha
    perdim = {}
    for d in DIMS:
        sub = sc[sc.pattern.str.match(r"^base_p\d+$") & sc.judge.isin(PANEL) & sc.dimension.eq(d)]
        wd = sub.pivot_table(index=["pattern","query_id"], columns="judge",
                             values="score", observed=True).dropna()
        if len(wd) > 5:
            try:
                perdim[d] = round(float(krippendorff.alpha(
                    reliability_data=wd[PANEL].T.values, level_of_measurement="interval")), 4)
            except Exception:
                perdim[d] = None
    pearson = w[PANEL].corr(method="pearson").round(4).to_dict()
    # ranking concordance on per-pattern means
    pm = base.groupby(["pattern","judge"], observed=True)["ovc"].mean().unstack()
    spear = pm.corr(method="spearman").round(4).to_dict()
    return {"n_complete_cells": int(n), "krippendorff_alpha_overall": round(float(alpha_overall),4),
            "icc_a1": round(float(icc_a1),4), "icc_ak3": round(float(icc_ak),4),
            "per_dimension_alpha": perdim, "judge_pearson_cells": pearson,
            "judge_spearman_pattern_means": spear}

# ---------- variance components (crossed RE) ----------
def variance_components():
    import statsmodels.formula.api as smf
    # eleven canonical base patterns only; excludes base_p11/base_p12, the post-hoc
    # single-judge-by-design probes (caught by adversarial review 2026-07-28, round 12:
    # this fit was silently pooling in 311 of 3262 rows, 9.5%, from those two patterns).
    base = ov[ov.pattern.str.match(r"^base_p([0-9]|10)$") & ov.judge.isin(PANEL)].copy()
    base = base.rename(columns={"query_id":"query"})
    # Cast grouping vars to plain strings so the design matrix only includes
    # levels actually present (the parquet's `pattern`/`judge` categoricals carry
    # unused levels from oracle/variance/protocol_a patterns -> otherwise singular).
    for c in ("pattern", "judge", "query"):
        base[c] = base[c].astype(str)
    base["grp"] = 1
    md = smf.mixedlm("ovc ~ C(pattern)", base, groups=base["grp"],
                     vc_formula={"query":"0+C(query)","judge":"0+C(judge)"})
    f = md.fit(reml=True, method="lbfgs")
    vq = float(f.vcomp[0]); vj = float(f.vcomp[1]); ve = float(f.scale)
    tot = vq+vj+ve
    return {"sigma2_query":round(vq,5),"sigma2_judge":round(vj,5),"sigma2_resid":round(ve,5),
            "icc_query":round(vq/tot,4),"icc_judge":round(vj/tot,4),"n":int(len(base))}

# ---------- verdict reconciliation (TRUE counts) ----------
def verdicts():
    v = pd.read_parquet(f"{A}/df_verdicts.parquet")
    g = v.groupby(["pattern","query_id","criterion_id"], observed=True)["judge"].nunique()
    base = v[v.pattern_family=="base"]
    # The "base" family pools the eleven canonical patterns' verdicts with several
    # post-hoc, single-judge probes (P11 ReAct, P12 our GRPO agent, a P11 16-turn
    # variant, 7B-backbone replicates of P1/P4) that this paper explicitly excludes
    # from "the eleven base patterns" (adversarial review 2026-07-28, round 33:
    # a table row previously labelled this whole pooled count "Base patterns
    # (P0--P12)", contradicting that exclusion elsewhere in the paper).
    main11_mask = base.pattern.str.match(r"^base_p([0-9]|10)$")
    return {"total_rows": int(len(v)),
            "by_family": {k:int(x) for k,x in v.pattern_family.value_counts().items()},
            "by_judge": {k:int(x) for k,x in v.judge.value_counts().items()},
            "base_rows": int((v.pattern_family=="base").sum()),
            "base_rows_main11": int(main11_mask.sum()),
            "base_rows_posthoc_probes": int((~main11_mask).sum()),
            "ablation_rows": int((v.pattern_family=="ablation").sum()),
            "protocol_a_rows": int((v.pattern_family=="protocol_a").sum()),
            "variance_rows": int((v.pattern_family=="variance").sum()),
            "triples_total": int(len(g)),
            "triples_ge2_judges": int((g>=2).sum()),
            "triples_eq3_judges": int((g==3).sum())}

# ---------- citation provenance ----------
def citations():
    c = pd.read_parquet(f"{A}/df_citations.parquet")
    # eleven canonical base patterns only; excludes base_p11/base_p12, the post-hoc
    # single-judge-by-design probes (adversarial review 2026-07-28, round 13: the pooled
    # total_base_citations here was silently including base_p11's 344 citations, inflating
    # the quoted "22,903" to a 12-pattern sum even though the per-pattern breakdown below
    # was already correctly scoped).
    base = c[c.pattern.str.match(r"^base_p([0-9]|10)$")].copy()
    out = {}
    for pat, g in base.groupby("pattern", observed=True):
        vc = g.category.value_counts(normalize=True)
        nrep = g.query_id.nunique()
        out[pat] = {"n_citations": int(len(g)), "n_reports": int(nrep),
                    "cites_per_report": round(len(g)/max(nrep,1),2),
                    # 6dp not 4dp: make_tables prints 100*rate to 1dp, and a 4dp pre-round can
                    # double-round across a .5 boundary (e.g. 0.434451 -> 0.4345 -> 43.5 vs true 43.4).
                    "placeholder_rate": round(float(vc.get("placeholder",0)),6),
                    "academic_rate": round(float(vc.get("academic",0)),6),
                    "real_url_rate": round(float(vc.get("real_url",0)),6),
                    "suspicious_rate": round(float(vc.get("suspicious",0)),6)}
    totals = {"total_citations": int(len(c)),
              "total_base_citations": int(len(base)),
              "by_category": {k:int(x) for k,x in c.category.value_counts().items()}}
    return {"per_pattern": out, "totals": totals}

# ---------- DR-Judge ----------
def drjudge():
    from sklearn.metrics import cohen_kappa_score
    e = pd.read_parquet(f"{ROOT}/reports/phase12_drjudge/eval_predictions_full.parquet")
    def kap(df): return float(cohen_kappa_score(df.target.astype(str), df.predicted.astype(str)))
    und, dis = e[~e.is_disputed], e[e.is_disputed]
    perpat = {}
    for p,g in e.groupby("pattern", observed=True):
        if len(g)>20: perpat[p] = {"n":int(len(g)), "kappa":round(kap(g),4)}
    perdim = {}
    for d,g in e.groupby("dimension", observed=True):
        if len(g)>20: perdim[d] = {"n":int(len(g)), "kappa":round(kap(g),4)}
    return {"n_test":int(len(e)), "kappa_overall":round(kap(e),4),
            "agreement_overall":round(float((e.target.astype(str)==e.predicted.astype(str)).mean()),4),
            "kappa_undisputed":round(kap(und),4),"n_undisputed":int(len(und)),
            "kappa_disputed":round(kap(dis),4),"n_disputed":int(len(dis)),
            "per_pattern":perpat,"per_dimension":perdim}

# ---------- ablations (2-judge gpt52+sonnet corrected) ----------
def ablations():
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(42)
    abl = ov[ov.pattern.str.startswith("ablation_") & ov.judge.isin(["gpt52","claude_sonnet"])].copy()
    base = ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(["gpt52","claude_sonnet"])].copy()
    def cellmean(df):
        return df.groupby(["pattern","query_id"], observed=True)["ovc"].mean()
    # Holm step-down across a family of raw p-values (largest-index scaling).
    def _holm(pv):
        idx = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
        for r, i in enumerate(idx):
            run = max(run, (m - r) * pv[i]); adj[i] = min(run, 1.0)
        return adj
    out = {}
    for ablpat in sorted(abl.pattern.unique()):
        if "no_citation_verify" in ablpat:  # excluded (2/90)
            out[ablpat] = {"excluded": True}; continue
        base_pat = "base_p" + ablpat.split("_p")[1].split("_")[0]
        a = cellmean(abl[abl.pattern.eq(ablpat)])
        b = cellmean(base[base.pattern.eq(base_pat)])
        common = a.index.get_level_values("query_id").intersection(b.index.get_level_values("query_id"))
        av = a.groupby("query_id").mean().loc[common]
        bv = b.groupby("query_id").mean().loc[common]
        d = (av - bv).dropna()
        if len(d) < 10:
            out[ablpat] = {"base": base_pat, "n": int(len(d)), "_note":"n<10"}; continue
        boot = [rng.choice(d.values, len(d), replace=True).mean() for _ in range(2000)]
        try: wp = float(wilcoxon(d.values).pvalue)
        except Exception: wp = None
        out[ablpat] = {"base": base_pat, "n": int(len(d)),
                       "delta": round(float(d.mean()),4),
                       "ci95": [round(float(np.percentile(boot,2.5)),4), round(float(np.percentile(boot,97.5)),4)],
                       "wilcoxon_p": wp}
    # --- Holm multiplicity correction across the family of testable contrasts ---
    # Family = every contrast carrying a wilcoxon_p (excludes the 'excluded' cell
    # and any n<10 cell). Adds p_holm + holm_sig per contrast so reviewers see the
    # multiplicity-corrected significance. Holm flips ablation_p5_fixed_width from
    # raw-significant (p~0.023) to ns (p_holm~0.069), matching the paper prose
    # which already calls that contrast 'indeterminate'.
    fam = [k for k, v in out.items()
           if isinstance(v, dict) and v.get("wilcoxon_p") is not None]
    if fam:
        pv = np.array([out[k]["wilcoxon_p"] for k in fam], dtype=float)
        adj = _holm(pv)
        for k, pa in zip(fam, adj):
            out[k]["p_holm"] = round(float(pa), 6)
            out[k]["holm_sig"] = bool(pa < 0.05)
        n_sig = int(sum(out[k]["holm_sig"] for k in fam))
        out["holm_family_note"] = (
            f"Holm-Bonferroni step-down across the {len(fam)} testable ablation "
            f"contrasts (family excludes the excluded no_citation_verify cell). "
            f"p_holm is the multiplicity-corrected Wilcoxon p; holm_sig = p_holm<0.05. "
            f"{n_sig}/{len(fam)} survive at alpha=0.05: ablation_p5_fixed_width flips "
            f"raw-significant (p={out.get('ablation_p5_fixed_width',{}).get('wilcoxon_p')}) "
            f"-> ns post-Holm (p_holm="
            f"{out.get('ablation_p5_fixed_width',{}).get('p_holm')}), consistent with the "
            f"paper prose that already calls it 'indeterminate'; the four large P4/P5 "
            f"effects survive at p_holm<=0.0001.")
    return out

# ---------- runs / cost / wordcount ----------
def runs():
    r = pd.read_parquet(f"{A}/df_runs.parquet")
    base = r[r.pattern.str.match(r"^base_p\d+$")].copy()
    out = {}
    for p,g in base.groupby("pattern", observed=True):
        out[p] = {"n_reports": int(g.report_exists.sum()),
                  "mean_word_count": round(float(g.report_word_count.mean()),0) if g.report_word_count.notna().any() else None,
                  "mean_cost_proxy_usd": round(float(g.cost_proxy_usd.mean()),4) if "cost_proxy_usd" in g and g.cost_proxy_usd.notna().any() else None}
    return out

# ---------- C0 verification ----------
def c0():
    try:
        per = pd.read_parquet(f"{A}/df_c0_per_report.parquet")
    except Exception:
        per = None
    v = pd.read_parquet(f"{A}/df_c0_verdicts.parquet")
    res = {"n_claims": int(len(v))}
    for col in ["verdict","citation_idx"]:
        if col in v.columns:
            res[f"{col}_dist"] = {str(k):int(x) for k,x in v[col].value_counts(dropna=False).head(10).items()}
    if "citation_idx" in v.columns:
        res["pct_citation_idx_none"] = round(float(v.citation_idx.isna().mean()),4)
    return res

# ---------- oracle-retrieval arm (gpt52, paired on the 30 variance queries) ----------
def oracle():
    rng = np.random.default_rng(7)
    VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
    CLUSTER = ("p1", "p4", "p5", "p6", "p7", "p8")  # ORDERED tuple, not a set: iterating a set of
    # strings varies with PYTHONHASHSEED per process, so np.mean over set-order float sums drifts at
    # the 4th decimal (cluster_factual_delta flipped -0.0072/-0.0071 across rebuilds). Rule-5 fix.
    g = ov[ov.judge.eq("gpt52")]
    scg = sc[sc.judge.eq("gpt52")]
    def overall_map(pat):
        d = g[g.pattern.eq(pat)]
        return {q: float(v) for q, v in zip(d.query_id, d.overall_score)
                if q in VARQ and pd.notna(v)}
    def dim_map(pat, dim):
        d = scg[scg.pattern.eq(pat) & scg.dimension.eq(dim)]
        return {q: float(v) for q, v in zip(d.query_id, d.score)
                if q in VARQ and pd.notna(v)}
    def paired_delta(omap, bmap):
        common = sorted(q for q in omap if q in bmap)  # sorted: set/hash order varies per process
        if not common:
            return None
        d = np.array([omap[q] - bmap[q] for q in common])
        rng_pd = np.random.default_rng(7)
        boot = [rng_pd.choice(d, len(d), replace=True).mean() for _ in range(2000)]
        return {"n": int(len(d)), "delta": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                         round(float(np.percentile(boot, 97.5)), 4)],
                "oracle_mean": round(float(np.mean([omap[q] for q in common])), 4),
                "base_mean": round(float(np.mean([bmap[q] for q in common])), 4)}
    per_pattern = {}
    for i in range(9):
        p = f"p{i}"; opat = f"oracle_t1_{p}"; bpat = f"base_{p}"
        ov_o, ov_b = overall_map(opat), overall_map(bpat)
        rec = {"is_cluster": p in CLUSTER, "overall": paired_delta(ov_o, ov_b),
               "dims": {d: paired_delta(dim_map(opat, d), dim_map(bpat, d)) for d in DIMS}}
        per_pattern[p] = rec
    def cl_dim(dim):
        vals = [per_pattern[p]["dims"][dim]["delta"] for p in CLUSTER
                if per_pattern[p]["dims"].get(dim)]
        return round(float(np.mean(vals)), 4) if vals else None
    # Cluster-level per-dimension delta with a TWO-STAGE BLOCK bootstrap (audit fix):
    # the six cluster patterns are clusters (deltas share a per-pattern oracle-injection
    # effect), so a flat n=179 i.i.d. bootstrap is anticonservative. Resample the 6 patterns
    # WITH replacement, then queries WITHIN each chosen pattern (block_boot, mirrors
    # build_oracle_robust_ci.py:49). Returns the empirical block-bootstrap dist so a
    # two-sided p-value can be derived and Holm-corrected across the 9 dimensions.
    def cl_dim_blocks(dim):
        # ordered (pattern -> sorted per-query deltas); CLUSTER is an ordered tuple, q sorted
        groups = []
        for p in CLUSTER:
            om, bm = dim_map(f"oracle_t1_{p}", dim), dim_map(f"base_{p}", dim)
            arr = np.array([om[q] - bm[q] for q in sorted(om) if q in bm], dtype=float)
            if len(arr):
                groups.append(arr)
        if not groups:
            return None
        npat = len(groups)
        n = int(sum(len(g) for g in groups))
        point = float(np.concatenate(groups).mean())
        rng_local = np.random.default_rng(11)  # own stream: CI must not drift with upstream draws
        boot = np.empty(2000)
        for i in range(2000):
            chosen = rng_local.integers(0, npat, npat)            # resample patterns (clusters)
            boot[i] = np.concatenate(
                [groups[c][rng_local.integers(0, len(groups[c]), len(groups[c]))]  # queries within
                 for c in chosen]).mean()
        return {"n": n, "n_clusters": npat, "delta": round(point, 4),
                "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                         round(float(np.percentile(boot, 97.5)), 4)],
                "_boot": boot, "_point": point}
    # Holm correction across the 9 per-dimension tests (reuses build_pairwise.py:12 logic).
    def holm(pv):
        idx = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
        for r, i in enumerate(idx):
            run = max(run, (m - r) * pv[i]); adj[i] = min(run, 1.0)
        return adj
    _raw = {d: cl_dim_blocks(d) for d in DIMS}
    _dims_present = [d for d in DIMS if _raw[d] is not None]
    # Two-sided bootstrap p-value: 2 * min(P(boot<=0), P(boot>=0)), reflected about the point.
    def _boot_p(rec):
        b = rec["_boot"]
        p = 2.0 * min(float(np.mean(b <= 0.0)), float(np.mean(b >= 0.0)))
        return min(p, 1.0)
    _pv = np.array([_boot_p(_raw[d]) for d in _dims_present])
    _adj = holm(_pv) if len(_pv) else _pv
    cluster_dims = {}
    for k, d in enumerate(_dims_present):
        rec = _raw[d]; ci = rec["ci95"]
        cluster_dims[d] = {
            "n": rec["n"], "n_clusters": rec["n_clusters"], "delta": rec["delta"],
            "ci95": ci, "ci_excludes_zero": bool(not (ci[0] <= 0 <= ci[1])),
            "p_block_boot": round(float(_pv[k]), 4),
            "p_holm": round(float(_adj[k]), 4),
            "holm_sig": bool(_adj[k] < 0.05)}
    for d in DIMS:
        if _raw[d] is None:
            cluster_dims[d] = None
    cl_overall = [per_pattern[p]["overall"]["delta"] for p in CLUSTER if per_pattern[p]["overall"]]
    base_cluster = np.mean([per_pattern[p]["overall"]["base_mean"] for p in CLUSTER])
    orac_cluster = np.mean([per_pattern[p]["overall"]["oracle_mean"] for p in CLUSTER])
    p0_base = per_pattern["p0"]["overall"]["base_mean"]
    p0_orac = per_pattern["p0"]["overall"]["oracle_mean"]
    gains = {p: per_pattern[p]["overall"]["delta"] for p in per_pattern if per_pattern[p]["overall"]}
    top = max(gains, key=gains.get)
    return {"judge": "gpt52", "n_variance_queries": len(VARQ), "per_pattern": per_pattern,
            "cluster_dims": cluster_dims,
            "cluster_citation_delta": cl_dim("citation_quality"),
            "cluster_factual_delta": cl_dim("factual_accuracy"),
            "cluster_overall_delta": round(float(np.mean(cl_overall)), 4),
            "gap_p0_to_cluster_base": round(float(base_cluster - p0_base), 4),
            "gap_p0_to_cluster_oracle": round(float(orac_cluster - p0_orac), 4),
            "max_overall_gainer": top, "max_overall_gain": round(float(gains[top]), 4),
            "p0_overall_gain": round(float(gains["p0"]), 4)}

for nm, fn in [("headline",headline),("single_judge_gpt52",single_judge_gpt52),
               ("per_dimension",per_dimension),("irr",irr),
               ("variance_components",variance_components),("verdicts",verdicts),
               ("citations",citations),("drjudge",drjudge),("ablations",ablations),
               ("runs",runs),("c0",c0),("oracle",oracle)]:
    section(nm, fn)

os.makedirs(OUT, exist_ok=True)
# MERGE-PRESERVING write: build_numbers owns only the 12 core keys below; it must
# NOT clobber the ~34 sibling keys written by the other builders. Load the existing
# canonical (if any) and update in place, then atomically replace. This makes
# `build_numbers.py` (and therefore `rebuild_all.sh`) non-destructive and idempotent.
_canon_path = f"{OUT}/canonical_numbers.json"
try:
    _existing = json.load(open(_canon_path))
    if not isinstance(_existing, dict):
        _existing = {}
except (FileNotFoundError, json.JSONDecodeError):
    _existing = {}
_existing.update(R)
_tmp = f"{_canon_path}.tmp"
with open(_tmp, "w") as f:
    json.dump(_existing, f, indent=1)
os.replace(_tmp, _canon_path)
print(f"\nWROTE {_canon_path} (merge-preserving; {len(R)} keys updated, "
      f"{len(_existing)} total)")

# markdown digest
def md():
    L = ["# Canonical numbers (regenerated from current parquets)\n",
         "Single source of truth for the rewrite. sonnet uses overall_score_recomputed.\n"]
    h = R["headline"]["per_pattern"]
    L.append("## Headline per-pattern (3-judge, sonnet-corrected)\n")
    L.append("| pattern | n_cells | n_q | mean | std | gpt52 | opus | sonnet_corr |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for p in R["headline"]["rank_desc"]:
        r=h[p]; L.append(f"| {p} | {r['n_cells']} | {r['n_queries']} | {r['mean_3judge']} | {r['std_3judge']} | {r['mean_gpt52']} | {r['mean_opus']} | {r['mean_sonnet_corrected']} |")
    irr=R["irr"]
    if "_error" not in irr:
        L.append(f"\n## IRR\nKrippendorff α={irr['krippendorff_alpha_overall']}, ICC(A,1)={irr['icc_a1']}, ICC(A,k=3)={irr['icc_ak3']} (n={irr['n_complete_cells']})\n")
        L.append("per-dim α: " + ", ".join(f"{k}={v}" for k,v in irr['per_dimension_alpha'].items()))
    vc=R["variance_components"]
    if "_error" not in vc:
        L.append(f"\n## Variance components\nICC(query)={vc['icc_query']}, ICC(judge)={vc['icc_judge']}, σ²resid={vc['sigma2_resid']}\n")
    vd=R["verdicts"]
    L.append(f"\n## Verdicts (TRUE counts)\ntotal={vd['total_rows']}; base={vd['base_rows']}; ablation={vd['ablation_rows']}; protocol_a={vd['protocol_a_rows']}; variance={vd['variance_rows']}; triples≥2={vd['triples_ge2_judges']}; triples=3={vd['triples_eq3_judges']}\n")
    dj=R["drjudge"]
    L.append(f"\n## DR-Judge\nκ overall={dj['kappa_overall']} (n={dj['n_test']}); undisputed={dj['kappa_undisputed']}; disputed={dj['kappa_disputed']}; agree={dj['agreement_overall']}\n")
    ct=R["citations"]["totals"]
    L.append(f"\n## Citations\ntotal={ct['total_citations']}; by_category={ct['by_category']}\n")
    return "\n".join(L)
with open(f"{OUT}/canonical_numbers.md","w") as f:
    f.write(md())
print(f"WROTE {OUT}/canonical_numbers.md")
