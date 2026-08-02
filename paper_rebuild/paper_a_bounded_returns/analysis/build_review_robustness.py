#!/usr/bin/env python
"""External-review robustness pair (June 2026):
1. Pattern ranking with the citation_quality dimension removed (re-weighted, renormalised):
   does the headline tier structure depend on the judge-artefact dimension?
2. Citation-density regression with query fixed effects added (density identified only
   within-query): does the density bonus survive the omitted-query-difficulty control?
Appends canonical_numbers.json['ranking_no_citation'] and ['citation_regression']['query_fe'].
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

W = {"information_recall": 0.20, "factual_accuracy": 0.20, "coverage": 0.10,
     "analytical_depth": 0.15, "citation_quality": 0.10, "logical_coherence": 0.05,
     "organization": 0.05, "instruction_following": 0.10, "attribution_quality": 0.05}

# ---- 1. ranking without citation_quality ----
v = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")
v = v[v.pattern.str.match(r"^base_p\d+$") & v.judge.isin(["gpt52", "claude_opus", "claude_sonnet"])
      & v.satisfied_is_known & v.pattern.isin([f"base_p{i}" for i in range(11)])]
dim = v.groupby(["pattern", "query_id", "judge", "dimension"], observed=True)["satisfied"].mean().reset_index()

def ranking(drop=None):
    w = {k: x for k, x in W.items() if k != drop}
    d2 = dim[dim.dimension.isin(w)].copy()
    d2["w"] = d2.dimension.map(w)
    g = d2.groupby(["pattern", "query_id", "judge"], observed=True)\
          .apply(lambda x: (x.satisfied * x.w).sum() / x.w.sum(), include_groups=False)\
          .rename("score").reset_index()
    return g.groupby("pattern", observed=True).score.mean().sort_values(ascending=False)

full = ranking()
nocit = ranking(drop="citation_quality")
rho = float(spearmanr(full[full.index].values, nocit[full.index].values).statistic)
CLUSTER = {"base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"}
top6_preserved = set(nocit.index[:6]) == CLUSTER
no_cit_out = {
    "spearman_vs_full": round(rho, 4),
    "top6_cluster_preserved": bool(top6_preserved),
    "ranking_no_citation": {p: round(float(s), 4) for p, s in nocit.items()},
    "note": "overall re-weighted without citation_quality (weights renormalised); "
            "tier structure (cluster / mid / P0 / 7B) must be read against the full ranking"}

# ---- 2. query fixed effects in the density regression ----
src = open(f"{ANA}/build_citation_regression.py").read()
_source_namespace = globals()
exec(src.split("m=smf.ols")[0], _source_namespace)  # reuses the exact dataframe construction
df = _source_namespace["df"]
m_pat = smf.ols("factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern) + C(query_id)",
                data=df).fit(cov_type="cluster", cov_kwds={"groups": df["pattern"].astype(str).values})
m_qry = smf.ols("factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern) + C(query_id)",
                data=df).fit(cov_type="cluster", cov_kwds={"groups": df["query_id"].astype(str).values})
qfe = {t_name: {"beta": round(float(m_pat.params[t]), 4),
                "p_cluster_pattern": float(m_pat.pvalues[t]),
                "p_cluster_query": float(m_qry.pvalues[t])}
       for t_name, t in [("density", "log_cit"), ("provenance", "provenance_rate")]}
qfe["n"] = int(len(df))
qfe["note"] = "query fixed effects added to the pattern-FE model; density identified within-query only"

cn = json.load(open(f"{ANA}/canonical_numbers.json"))
cn["ranking_no_citation"] = no_cit_out
cn.setdefault("citation_regression", {})["query_fe"] = qfe
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
print(json.dumps({"no_cit_rho": rho, "top6_preserved": top6_preserved, "query_fe": qfe}, indent=1))
