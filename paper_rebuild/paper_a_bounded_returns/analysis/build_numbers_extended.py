#!/usr/bin/env python
"""Extended canonical numbers: stratification, per-source leaders, Bing-vs-Tavily
intervention, C0 verification. Uses the TRUE released manifest (df_queries).
Appends to canonical_numbers.json under key 'extended' and writes more .tex tables.
"""
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT="."; A=f"{ROOT}/data/analysis"
ANA=f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"; TAB=f"{ROOT}/paper_rebuild/paper_a_bounded_returns/tables"
PANEL=["gpt52","claude_opus","claude_sonnet"]
PRETTY={"base_p0":"P0","base_p1":"P1","base_p2":"P2","base_p3":"P3","base_p4":"P4","base_p5":"P5",
 "base_p6":"P6","base_p7":"P7","base_p8":"P8","base_p9":"P9","base_p10":"P10","base_p11":"P11","base_p12":"P12"}
R=json.load(open(f"{ANA}/canonical_numbers.json"))
EXT={}

ov=pd.read_parquet(f"{A}/df_overall_scores.parquet")
ov["ovc"]=ov["overall_score"].where(~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
q=pd.read_parquet(f"{A}/df_queries.parquet")[["query_id","source","difficulty"]]
ovq=ov.merge(q,on="query_id",how="left")
base=ovq[ovq.pattern.str.match(r"^base_p([0-9]|10)$") & ovq.judge.isin(PANEL)].copy()

# ---- manifest ----
EXT["manifest"]={"n_queries":int(q.query_id.nunique()),
 "by_source":{k:int(v) for k,v in q.source.value_counts().items()},
 "by_difficulty":{k:int(v) for k,v in q.difficulty.value_counts().items()}}

# ---- per-source per-pattern mean (3-judge cell mean) ----
def cellmean(df,keys): return df.groupby(keys,observed=True)["ovc"].mean()
src_tab=(base.groupby(["source","pattern"],observed=True)["ovc"].mean().unstack())
EXT["per_source"]={}
for s in src_tab.index:
    row=src_tab.loc[s].dropna().sort_values(ascending=False)
    top=[(PRETTY.get(p,p),round(float(v),3)) for p,v in row.items() if p in PRETTY][:4]
    EXT["per_source"][s]={"n":int(q[q.source==s].query_id.nunique()),"top4":top}
# ---- per-difficulty ----
dif_tab=(base.groupby(["difficulty","pattern"],observed=True)["ovc"].mean().unstack())
EXT["per_difficulty"]={}
for s in dif_tab.index:
    row=dif_tab.loc[s].dropna().sort_values(ascending=False)
    EXT["per_difficulty"][s]={"n":int(q[q.difficulty==s].query_id.nunique()),
        "by_pattern":{PRETTY.get(p,p):round(float(v),3) for p,v in row.items() if p in PRETTY}}

# ---- Bing vs Tavily (gpt52) ----
pa=ov[ov.pattern.str.startswith("protocol_a_tavily_") & ov.judge.eq("gpt52")]
pa_qs=pa.query_id.unique()
rng=np.random.default_rng(42); BVT={}
for tp in sorted(pa.pattern.unique()):
    pid=tp.replace("protocol_a_tavily_","")
    base_pat=f"base_{pid}"
    tav=ov[(ov.pattern==tp)&(ov.judge=="gpt52")].set_index("query_id")["overall_score"]
    bing=ov[(ov.pattern==base_pat)&(ov.judge=="gpt52")&(ov.query_id.isin(pa_qs))].set_index("query_id")["overall_score"]
    common=tav.index.intersection(bing.index)
    d=(tav.loc[common]-bing.loc[common]).dropna()
    if len(d)<5: continue
    boot=[rng.choice(d.values,len(d),replace=True).mean() for _ in range(10000)]
    BVT[pid.upper()]={"n":int(len(d)),"bing":round(float(bing.loc[common].mean()),3),
        "tavily":round(float(tav.loc[common].mean()),3),"delta":round(float(d.mean()),3),
        "ci":[round(float(np.percentile(boot,2.5)),3),round(float(np.percentile(boot,97.5)),3)],
        "p_lt0":round(float((np.array(boot)>=0).mean()),4)}
EXT["bing_vs_tavily"]=BVT

# ---- C0 per-pattern ----
try:
    c0=pd.read_parquet(f"{A}/df_c0_per_report.parquet")
    col=next((c for c in c0.columns if "factual" in c.lower() or "verified" in c.lower() or "support" in c.lower()),None)
    if col:
        EXT["c0_per_pattern"]={PRETTY.get(p,p):round(float(g[col].mean()),4)
            for p,g in c0.groupby("pattern",observed=True) if p in PRETTY}
        EXT["c0_metric_col"]=col
except Exception as e:
    EXT["c0_per_pattern"]={"_error":str(e)}

R["extended"]=EXT
json.dump(R,open(f"{ANA}/canonical_numbers.json","w"),indent=1)

# ---- tables ----
# per-source leader table
rows=[]
SRCN={"custom":"Custom","draco":"DRACO","deepsearch_qa":"DeepSearchQA","litqa2":"LitQA2","research_qa":"ResearchQA"}
for s,info in EXT["per_source"].items():
    cells=" & ".join(f"{nm} ({v:.3f})" for nm,v in info["top4"][:3])
    rows.append(f"{SRCN.get(s,s)} ($n{{=}}{info['n']}$) & {cells} \\\\")
t=r"""\begin{tabular}{lccc}
\toprule
Source & Best & 2nd & 3rd\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}"""
open(f"{TAB}/tab_per_source.tex","w").write(t)

# Bing vs Tavily table
rows=[]
for p,r in EXT["bing_vs_tavily"].items():
    rows.append(f"{p} & {r['bing']:.3f} & {r['tavily']:.3f} & {r['delta']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] \\\\")
t=r"""\begin{tabular}{lrrrc}
\toprule
Pattern & Bing & Tavily & $\Delta$ & 95\% CI\\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}"""
open(f"{TAB}/tab_bing_tavily.tex","w").write(t)

print("EXT manifest:",EXT["manifest"])
print("per_source leaders:",{s:i["top4"][0] for s,i in EXT["per_source"].items()})
print("bing_vs_tavily:",{k:(v["delta"],v["p_lt0"]) for k,v in EXT["bing_vs_tavily"].items()})
print("c0:",EXT.get("c0_per_pattern"))
print("wrote tab_per_source.tex, tab_bing_tavily.tex")
