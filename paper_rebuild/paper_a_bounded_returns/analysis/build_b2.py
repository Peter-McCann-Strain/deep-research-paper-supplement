#!/usr/bin/env python
"""B2 external-validity: the orchestration premium at 7B vs at GPT-4o.

P1/P4 run on the local Qwen2.5-7B backbone (run-tag 7b) over the B2 subset; P9 is the
single-pass (P0-architecture) baseline on the same 7B. The premium P1_7B - P9 (and P4_7B - P9)
isolates whether orchestration helps a 7B backbone, for comparison with P1 - P0 / P4 - P0 at
GPT-4o scale. GPT-5.2 single-judge (the 7B reports are newly judged). Appends
canonical_numbers.json['b2_7b_premium'].
"""
import json, glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT="."
SUB=set(json.load(open(f"{ROOT}/data/b2_subset.json"))["query_ids"])

def mean_overall(pat):
    vals=[]
    for f in glob.glob(f"{ROOT}/results/judge_gpt52/{pat}/*.json"):
        q=os.path.basename(f)[:-5]
        if q in SUB:
            s=json.load(open(f)).get("overall_score")
            if s is not None: vals.append(float(s))
    return (round(float(np.mean(vals)),4), len(vals)) if vals else (None,0)

out={"judge":"gpt52","n_subset":len(SUB),"note":"GPT-5.2 single-judge; P9 is P0-arch on the same 7B backbone"}
# GPT-4o scale premium (P1/P4 vs P0) on the same subset, for comparison
for tag,patt in [("p1_7b","base_p1_7b"),("p4_7b","base_p4_7b"),("p9","base_p9"),
                 ("p1_gpt4o","base_p1"),("p4_gpt4o","base_p4"),("p0_gpt4o","base_p0")]:
    m,n=mean_overall(patt); out[tag]={"mean":m,"n":n}
def prem(a,b):
    return round(out[a]["mean"]-out[b]["mean"],4) if out[a]["mean"] is not None and out[b]["mean"] is not None else None
out["premium_7b_p1"]=prem("p1_7b","p9")
out["premium_7b_p4"]=prem("p4_7b","p9")
out["premium_gpt4o_p1"]=prem("p1_gpt4o","p0_gpt4o")
out["premium_gpt4o_p4"]=prem("p4_gpt4o","p0_gpt4o")

# paired uncertainty for the premiums (query-paired Wilcoxon + seeded bootstrap CI)
from scipy.stats import wilcoxon
def paired(a_pat,b_pat):
    qa,qb={},{}
    for f in glob.glob(f"{ROOT}/results/judge_gpt52/{a_pat}/*.json"):
        q=os.path.basename(f)[:-5]
        if q in SUB:
            s_=json.load(open(f)).get("overall_score")
            if s_ is not None: qa[q]=float(s_)
    for f in glob.glob(f"{ROOT}/results/judge_gpt52/{b_pat}/*.json"):
        q=os.path.basename(f)[:-5]
        if q in SUB:
            s_=json.load(open(f)).get("overall_score")
            if s_ is not None: qb[q]=float(s_)
    qs=sorted(set(qa)&set(qb)); d=np.array([qa[q]-qb[q] for q in qs])
    try: pv=float(wilcoxon(d).pvalue)
    except Exception: pv=1.0
    rng=np.random.default_rng(0)
    boots=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(10000)]
    return {"n":len(d),"delta":round(float(d.mean()),4),"wilcoxon_p":round(pv,3),
            "ci95":[round(float(np.percentile(boots,2.5)),3),round(float(np.percentile(boots,97.5)),3)],
            "significant":bool(pv<0.05)}
out["premium_7b_p1_test"]=paired("base_p1_7b","base_p9")
out["premium_7b_p4_test"]=paired("base_p4_7b","base_p9")
out["premium_gpt4o_p1_test"]=paired("base_p1","base_p0")
out["premium_gpt4o_p4_test"]=paired("base_p4","base_p0")
cn=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
cn["b2_7b_premium"]=out
json.dump(cn,open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json","w"),indent=1)
print(json.dumps({k:out[k] for k in out if "premium" in k or k in ("p1_7b","p4_7b","p9")}, indent=1))
