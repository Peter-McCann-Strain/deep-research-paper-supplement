#!/usr/bin/env python
"""Re-verify the citation-density-vs-grounding regression (paper's central C2 claim).
factual_accuracy ~ provenance_rate + log_citation_count + log_words + placeholder_rate
                   + pattern fixed effects.  -> canonical_numbers.json['citation_regression']
Also: correlations of citation_quality with density vs provenance."""
import pandas as pd, numpy as np, json, warnings, sys
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
WRITE="--write" in sys.argv
ROOT="."; A=f"{ROOT}/data/analysis"
ANA=f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

cit=pd.read_parquet(f"{A}/df_citations.parquet")
cit=cit[cit.pattern.str.match(r"^base_p([0-9]|10)$")]  # eleven canonical base patterns only;
# excludes base_p11/base_p12, the post-hoc single-judge-by-design probes the paper's own
# "eleven base patterns" convention (main.tex Sec. Experimental Design) keeps out of every
# panel-averaged family (caught by adversarial review 2026-07-28, round 12: this regression
# was silently pooling in base_p11's 79 rows, inflating n_reports to 995 against a 990-row max
# for eleven patterns x 90 queries).
# per-report citation features
g=cit.groupby(["pattern","query_id"],observed=True)
feat=pd.DataFrame({
 "n_cit": g.size(),
 "placeholder_rate": g.apply(lambda d:(d.category=="placeholder").mean()),
 "provenance_rate": g.apply(lambda d:d.category.isin(["academic","real_url"]).mean()),
 "words": g["report_word_count"].first(),
}).reset_index()

# per-report 3-judge factual_accuracy and citation_quality
sc=pd.read_parquet(f"{A}/df_scores.parquet")
sc=sc[sc.pattern.str.match(r"^base_p([0-9]|10)$") & sc.judge.isin(["gpt52","claude_opus","claude_sonnet"])]
def dim_report(dim):
    s=sc[sc.dimension==dim].groupby(["pattern","query_id"],observed=True)["score"].mean()
    return s.rename(dim)
fa=dim_report("factual_accuracy"); cq=dim_report("citation_quality")
df=feat.merge(fa,on=["pattern","query_id"]).merge(cq,on=["pattern","query_id"])
df=df[df.n_cit>0].copy()
df["log_cit"]=np.log(df.n_cit); df["log_words"]=np.log(df.words.clip(lower=1))
df["pattern"]=df.pattern.astype(str)

# provenance_rate and placeholder_rate are complementary fractions (near-collinear);
# include only provenance_rate so coefficients are interpretable.
m=smf.ols("factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern)",
          data=df).fit()
# Cluster-robust refits (audit M-headline-iii/RA4): observations are clustered by query (all
# patterns answer the same queries) and by pattern. iid SEs are anticonservative; report
# cluster-robust p-values grouped by query_id (G=90) and by pattern (G=11, most demanding).
mq=smf.ols("factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern)",
           data=df).fit(cov_type="cluster",cov_kwds={"groups":df["query_id"].astype(str).values})
mp=smf.ols("factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern)",
           data=df).fit(cov_type="cluster",cov_kwds={"groups":df["pattern"].astype(str).values})
terms=["provenance_rate","log_cit","log_words"]

# ---- Wild-cluster sign-flip bootstrap (audit: G=11 patterns -> asymptotic cluster-robust SEs
# are anticonservative with so few clusters). Rademacher (+/-1) weights applied per CLUSTER to
# the restricted-model residuals; bootstrap-t with restricted (null-imposed) residuals (WCR).
# Reps>=999, fixed seed, sorted cluster order -> deterministic. Reports a p alongside asymptotic.
def wild_cluster_p(formula, data, term, group_col, reps=1999, seed=20260629):
    rng=np.random.default_rng(seed)
    full=smf.ols(formula,data=data).fit(
        cov_type="cluster",cov_kwds={"groups":data[group_col].astype(str).values})
    t_obs=float(full.tvalues[term])
    # restricted model: drop the tested term (impose beta_term=0), keep its column for resampling
    rhs=[x.strip() for x in formula.split("~",1)[1].split("+")]
    restr_formula=formula.split("~",1)[0]+"~"+" + ".join(x for x in rhs if x.strip()!=term)
    rfit=smf.ols(restr_formula,data=data).fit()
    resid=rfit.resid.to_numpy(); fitted=rfit.fittedvalues.to_numpy()
    lhs=formula.split("~",1)[0].strip()
    clusters=sorted(data[group_col].astype(str).unique())
    cidx={c:(data[group_col].astype(str).values==c) for c in clusters}
    cnt=0
    for _ in range(reps):
        w_=rng.integers(0,2,len(clusters))*2-1  # Rademacher +/-1 per cluster, deterministic order
        ystar=fitted.copy()
        for k,c in enumerate(clusters):
            ystar[cidx[c]]=fitted[cidx[c]]+w_[k]*resid[cidx[c]]
        bd=data.copy(); bd[lhs]=ystar
        bf=smf.ols(formula,data=bd).fit(
            cov_type="cluster",cov_kwds={"groups":bd[group_col].astype(str).values})
        if abs(float(bf.tvalues[term]))>=abs(t_obs): cnt+=1
    return (cnt+1)/(reps+1)

WC_REPS=1999
wcb={t:round(float(wild_cluster_p(
        "factual_accuracy ~ provenance_rate + log_cit + log_words + C(pattern)",
        df,t,"pattern",reps=WC_REPS)),5) for t in terms}

coefs={t:{"beta":round(float(m.params[t]),4),"se":round(float(m.bse[t]),4),
          "t":round(float(m.tvalues[t]),3),"p":float(m.pvalues[t]),
          "p_cluster_query":float(mq.pvalues[t]),"se_cluster_query":round(float(mq.bse[t]),4),
          "p_cluster_pattern":float(mp.pvalues[t]),"se_cluster_pattern":round(float(mp.bse[t]),4),
          "p_wild_cluster_pattern":wcb[t],
          "ci":[round(float(m.conf_int().loc[t,0]),4),round(float(m.conf_int().loc[t,1]),4)]} for t in terms}
out={"n_reports":int(len(df)),"r2":round(float(m.rsquared),4),"r2_adj":round(float(m.rsquared_adj),4),
     "dependent_variable":"factual_accuracy",
     "dv_note":"DESPITE the key name 'citation_regression', the DEPENDENT variable is factual_accuracy "
               "(3-judge mean), NOT a citation outcome. Specification: factual_accuracy ~ provenance_rate "
               "+ log_cit + log_words + C(pattern). The regressors include citation features (provenance_rate, "
               "log_cit); the outcome is grounding/factual accuracy.",
     "se_note":"p is iid-OLS (anticonservative); p_cluster_query (G=90) and p_cluster_pattern (G=11) are "
               "cluster-robust asymptotic. With only G=11 pattern clusters, asymptotic cluster SEs are "
               "anticonservative; p_wild_cluster_pattern is a wild-cluster-restricted bootstrap (Rademacher "
               "sign-flips per pattern, reps=%d, seed=20260629) and is the value to report for the pattern "
               "clustering. Report the clustered/wild values, not iid p."%WC_REPS,
     "coefs":coefs,
     "corr_cq_density_pearson":round(float(df[["citation_quality","log_cit"]].corr().iloc[0,1]),4),
     "corr_cq_provenance_pearson":round(float(df[["citation_quality","provenance_rate"]].corr().iloc[0,1]),4),
     "corr_fa_provenance_pearson":round(float(df[["factual_accuracy","provenance_rate"]].corr().iloc[0,1]),4)}
p=json.load(open(f"{ANA}/canonical_numbers.json")); p["citation_regression"]=out

# ---- per-judge density coefficient (paper §6: is the density->score effect judge-specific?) ----
# Refit the SAME regression as the pooled model -- factual_accuracy ~ provenance + log(count) +
# log(words) + pattern FE -- but SEPARATELY per judge, using that judge's own factual_accuracy.
# This is the exact decomposition §6 cites; reproduce it from data rather than trusting the draft.
sc_all=pd.read_parquet(f"{A}/df_scores.parquet")
sc_all=sc_all[sc_all.pattern.str.match(r"^base_p([0-9]|10)$") & sc_all.dimension.eq("factual_accuracy")]
def _judge_factual(j):
    d=sc_all[sc_all.judge==j]
    return d.groupby(["pattern","query_id"],observed=True)["score"].mean().rename("fa")
density_per_judge={}
for j in ["gpt52","claude_opus","claude_sonnet"]:
    dj=feat.merge(_judge_factual(j),on=["pattern","query_id"])
    dj=dj[dj.n_cit>0].copy()
    dj["log_cit"]=np.log(dj.n_cit); dj["log_words"]=np.log(dj.words.clip(lower=1)); dj["pattern"]=dj.pattern.astype(str)
    mj=smf.ols("fa ~ provenance_rate + log_cit + log_words + C(pattern)",data=dj).fit()
    mjp=smf.ols("fa ~ provenance_rate + log_cit + log_words + C(pattern)",
                data=dj).fit(cov_type="cluster",cov_kwds={"groups":dj["pattern"].astype(str).values})
    density_per_judge[j]={"beta_density":round(float(mj.params["log_cit"]),4),
                          "p_density":round(float(mj.pvalues["log_cit"]),6),
                          "p_density_cluster_pattern":float(mjp.pvalues["log_cit"]),
                          "beta_provenance":round(float(mj.params["provenance_rate"]),4),
                          "p_provenance":round(float(mj.pvalues["provenance_rate"]),6),
                          "p_provenance_cluster_pattern":float(mjp.pvalues["provenance_rate"]),
                          "n":int(len(dj))}
p["density_per_judge"]=density_per_judge

if WRITE:
    json.dump(p,open(f"{ANA}/canonical_numbers.json","w"),indent=1)
    print("[WROTE] canonical_numbers.json")
else:
    print("[DRY-RUN] no write (pass --write to persist)")
print(json.dumps(out,indent=1))
print("density_per_judge:",json.dumps(density_per_judge,indent=1))
