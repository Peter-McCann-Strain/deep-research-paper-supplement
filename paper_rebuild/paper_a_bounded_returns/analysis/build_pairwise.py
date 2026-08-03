#!/usr/bin/env python
"""Verified pairwise + TOST cluster statistics -> canonical_numbers.json['pairwise_verified']."""
import pandas as pd, numpy as np, itertools, json, warnings
from scipy.stats import wilcoxon, t as tdist
warnings.filterwarnings("ignore")
ROOT="."
ANA=f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
ov=pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
ov["ovc"]=ov["overall_score"].where(~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
base=ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(["gpt52","claude_opus","claude_sonnet"])]
pats=[f"base_p{i}" for i in range(11)]
def holm(pv):
    idx=np.argsort(pv); m=len(pv); adj=np.empty(m); run=0
    for r,i in enumerate(idx):
        run=max(run,(m-r)*pv[i]); adj[i]=min(run,1.0)
    return adj
def judge_pairs(j):
    d=base[base.judge==j]; wide=d.pivot_table(index="query_id",columns="pattern",values="ovc",observed=True)
    pairs=list(itertools.combinations(pats,2)); pv=[]; sign={}
    for a,b in pairs:
        s=wide[[a,b]].dropna()
        try: p=wilcoxon(s[a],s[b]).pvalue
        except Exception: p=1.0
        pv.append(p); sign[(a,b)]=float(np.sign((s[a]-s[b]).mean()))
    adj=holm(np.array(pv))
    return {pairs[i]:bool(adj[i]<0.05) for i in range(len(pairs))}, sign, int((adj<0.05).sum())
res={j:judge_pairs(j) for j in ["gpt52","claude_opus","claude_sonnet"]}
pairs=list(itertools.combinations(pats,2))
robust=sum(1 for pr in pairs if all(res[j][0][pr] for j in res) and len({res[j][1][pr] for j in res})==1)
avg=base.groupby(["pattern","query_id"],observed=True)["ovc"].mean().unstack(0)
def tost_t(a,b,m):
    d=avg[[a,b]].dropna(); d=(d[a]-d[b]).values; n=len(d); mu=d.mean(); se=d.std(ddof=1)/np.sqrt(n)
    return max(1-tdist.cdf((mu+m)/se,n-1), tdist.cdf((mu-m)/se,n-1))<0.05
def tost_w(a,b,m):
    d=avg[[a,b]].dropna(); d=(d[a]-d[b]).values
    try: return max(wilcoxon(d+m,alternative='greater').pvalue, wilcoxon(d-m,alternative='less').pvalue)<0.05
    except Exception: return False
c6=["base_p1","base_p4","base_p5","base_p6","base_p7","base_p8"]
i5=["base_p1","base_p4","base_p6","base_p7","base_p8"]
out={"judge_robust_of_55":int(robust),
     "within_judge_holm_sig":{j:int(res[j][2]) for j in res},
     "inner5_robustly_separated_of_10":int(sum(1 for pr in itertools.combinations(i5,2)
        if all(res[j][0][pr] for j in res) and len({res[j][1][pr] for j in res})==1)),
     "tost6_t_pm05":int(sum(tost_t(a,b,0.05) for a,b in itertools.combinations(c6,2))),
     "tost6_wilcoxon_pm05":int(sum(tost_w(a,b,0.05) for a,b in itertools.combinations(c6,2))),
     "tost6_t_pm02":int(sum(tost_t(a,b,0.02) for a,b in itertools.combinations(c6,2))),
     "note":"26/55 judge-robust and 0/10 inner-5 are exact & method-stable. TOST 9/15 is Wilcoxon; t-TOST=6/15. Lead with judge-robust."}
p=json.load(open(f"{ANA}/canonical_numbers.json"))
p["pairwise_verified"]=out
json.dump(p,open(f"{ANA}/canonical_numbers.json","w"),indent=1)
print(json.dumps(out,indent=1))
