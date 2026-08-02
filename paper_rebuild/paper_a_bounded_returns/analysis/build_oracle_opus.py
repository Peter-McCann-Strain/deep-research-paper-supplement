#!/usr/bin/env python
"""A5: cross-family Opus re-scoring of the oracle arm (within-version, confound-free).

Oracle cluster reports and their baselines were BOTH re-scored by the same Opus-4.8 so the
oracle-minus-baseline delta carries no judge-version artifact. Confirms whether the dual
mechanism (citation rises, factual flat) replicates on a second, cross-family judge.
Appends canonical_numbers.json['oracle']['opus_cross_check'].
"""
import json, glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT="."
VARQ=set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
CLUSTER=["p1","p4","p5","p6","p7","p8"]
DIMS=["citation_quality","factual_accuracy","information_recall","coverage","analytical_depth",
      "logical_coherence","organization","instruction_following","attribution_quality"]
rng=np.random.default_rng(11)

def load(d, dim):
    out={}
    for f in glob.glob(f"{ROOT}/{d}/*.json"):
        q=os.path.basename(f)[:-5]
        if q in VARQ:
            dd=json.load(open(f)).get("dimensions",{})
            if dim in dd and dd[dim].get("score") is not None: out[q]=float(dd[dim]["score"])
    return out

def paired(oracle_tmpl, base_tmpl, dim):
    deltas=[]
    for p in CLUSTER:
        o=load(oracle_tmpl.format(p=p), dim); b=load(base_tmpl.format(p=p), dim)
        deltas += [o[q]-b[q] for q in o if q in b]
    if not deltas: return None
    a=np.array(deltas)
    boot=[rng.choice(a,len(a),replace=True).mean() for _ in range(2000)]
    return {"n":int(len(a)),"delta":round(float(a.mean()),4),
            "ci95":[round(float(np.percentile(boot,2.5)),4),round(float(np.percentile(boot,97.5)),4)]}

opus={d:paired("results/judge_claude_opus/oracle_t1_{p}","results/judge_claude_opus48/base_{p}",d) for d in DIMS}
sonnet={d:paired("results/judge_claude_sonnet/oracle_t1_{p}","results/judge_claude_sonnet48/base_{p}",d) for d in DIMS}
cn=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
gpt=cn["oracle"]["cluster_dims"]
def trio(dim): return {"gpt52":gpt[dim]["delta"],"opus":opus[dim]["delta"] if opus[dim] else None,
                       "sonnet":sonnet[dim]["delta"] if sonnet[dim] else None}
out={"note":"within-version Claude re-scoring (oracle and base both judged by the same Claude version); full three-judge cross-check of the GPT-5.2 dual mechanism",
     "opus_cluster_dims":opus,"sonnet_cluster_dims":sonnet,
     "panel_citation":trio("citation_quality"),"panel_factual":trio("factual_accuracy"),
     "panel_info_recall":trio("information_recall")}
cn["oracle"]["panel_cross_check"]=out
cn["oracle"]["opus_cross_check"]={"cluster_dims":opus}  # back-compat
json.dump(cn,open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json","w"),indent=1)
for dim in ["citation_quality","factual_accuracy","information_recall"]:
    t=trio(dim); print(f"{dim:20s} gpt52 {t['gpt52']:+.3f} | opus {t['opus']:+.3f} | sonnet {t['sonnet']:+.3f}")
