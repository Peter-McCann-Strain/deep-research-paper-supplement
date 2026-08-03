#!/usr/bin/env python
"""Compute-matched control: does orchestration beat best-of-N over a single pass?

Uses the independent P0 replicates (base_p0 + base_p0_v{1,2,3}, GPT-5.2) on the 30
variance queries as N samples of the single-pass pipeline. For each k it reports the
oracle best-of-k score (an upper bound: select the best sample by the evaluation judge)
and the mean-of-k, against the orchestrated-cluster GPT-5.2 mean on the same queries.
Appends canonical_numbers.json['best_of_n']. GPT-5.2 single-judge; oracle selection is an
upper bound on any realizable selector.
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
ROOT="."; A=f"{ROOT}/data/analysis"
import glob, os, re
ov=pd.read_parquet(f"{A}/df_overall_scores.parquet")
g=ov[ov.judge.eq("gpt52")]
VARQ=set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
# auto-discover every judged P0 replicate (base_p0 + base_p0_v{k}) so N grows as more land
P0=["base_p0"]+sorted([os.path.basename(d) for d in glob.glob(f"{ROOT}/results/judge_gpt52/base_p0_v*")
                       if re.match(r"base_p0_v\d+$", os.path.basename(d))],
                      key=lambda s:int(s.split("_v")[1]))
CLUSTER=["base_p1","base_p4","base_p5","base_p6","base_p7","base_p8"]
runs=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["runs"]

def patq(p,q):
    d=g[(g.pattern==p)&(g.query_id==q)]
    return float(d.overall_score.iloc[0]) if len(d) and pd.notna(d.overall_score.iloc[0]) else None

samples={}
for q in VARQ:
    s=[patq(p,q) for p in P0]; s=[x for x in s if x is not None]
    if len(s)>=3: samples[q]=s
qs=list(samples); N=min(len(samples[q]) for q in qs)
cluster_q={q:np.mean([patq(p,q) for p in CLUSTER if patq(p,q) is not None]) for q in qs}
cl=float(np.mean([cluster_q[q] for q in qs]))

curve={}
for k in range(1,N+1):
    best=float(np.mean([max(samples[q][:k]) for q in qs]))
    mean=float(np.mean([np.mean(samples[q][:k]) for q in qs]))
    curve[k]={"best_of_k":round(best,4),"mean_of_k":round(mean,4),"gap_to_cluster":round(cl-best,4)}

p0_cost=runs["base_p0"]["mean_cost_proxy_usd"]
cluster_cost=float(np.mean([runs[p]["mean_cost_proxy_usd"] for p in CLUSTER]))
out={"judge":"gpt52","n_queries":len(qs),"n_samples":N,
     "cluster_mean":round(cl,4),"curve":curve,
     "best_of_N":curve[N]["best_of_k"],"gap_best_of_N_to_cluster":curve[N]["gap_to_cluster"],
     "p0_cost_usd":round(p0_cost,3),"best_of_N_cost_usd":round(N*p0_cost,3),
     "cluster_cost_usd":round(cluster_cost,3),
     "note":"oracle (judge-max) selection = upper bound on realizable best-of-N; GPT-5.2 single-judge"}
cn=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
cn["best_of_n"]=out
json.dump(cn,open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json","w"),indent=1)
print(json.dumps(out,indent=1))
