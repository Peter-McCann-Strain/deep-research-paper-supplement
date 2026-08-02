#!/usr/bin/env python
"""Cost-quality figure: overall score vs per-query compute cost (log-x), from verified
canonical numbers. Makes the 'flat cluster -> pick the cheapest member' point clean."""
import json, warnings, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, C_CLUSTER, C_OTHER, C_LOCAL
# print scale s=0.789 (0.85 linewidth fraction) -> declared 14pt prints at ~11pt,
# comfortably clears the 9pt floor including the legend.
apply_style(base_size=14, legend_fontsize=12)
ROOT="."
d=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
h=d["headline"]["per_pattern"]; runs=d["runs"]
NAME={f"base_p{i}":n for i,n in zip(range(11),
  ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10"])}
CLUSTER={"base_p1","base_p4","base_p5","base_p6","base_p7","base_p8"}
LOCAL={"base_p9","base_p10"}
def cat(p): return "cluster" if p in CLUSTER else ("local" if p in LOCAL else "other")
def color(p): return {"cluster":C_CLUSTER,"other":C_OTHER,"local":C_LOCAL}[cat(p)]
def marker(p): return {"cluster":"o","other":"s","local":"^"}[cat(p)]
pts=[]
for p in NAME:
    if p in h and p in runs and runs[p].get("mean_cost_proxy_usd"):
        pts.append((p, runs[p]["mean_cost_proxy_usd"], h[p]["mean_3judge"]))
fig,ax=plt.subplots(figsize=(7.2,4.6))
# cluster band
cy=[y for _,_,y in pts if _ in CLUSTER] if False else None
ymin=min(h[p]["mean_3judge"] for p in CLUSTER); ymax=max(h[p]["mean_3judge"] for p in CLUSTER)
ax.axhspan(ymin-0.005,ymax+0.005,color=C_CLUSTER,alpha=0.08,zorder=0)
# Pareto-efficient frontier over the GPT-4o patterns ONLY (caption/prose contract:
# local-7B GPU-amortised cost is not commensurable with API spend, so P9/P10 are
# plotted but excluded from the frontier). A pattern is efficient if no cheaper
# GPT-4o pattern scores at least as high; staircase runs P0->P2->P7->P6->P1.
eff=[]
for p,c,s in sorted(pts,key=lambda r:r[1]):
    if p in LOCAL:
        continue
    if all(s>so for _,co,so in eff) or not eff:
        eff.append((p,c,s))
ex=[c for _,c,_ in eff]; ey=[s for _,_,s in eff]
ax.step(ex+[max(c for _,c,_ in pts)],ey+[ey[-1]],where="post",color="#117733",lw=1.8,
        ls="--",alpha=0.85,zorder=1,label="Pareto frontier")
# label only the decision-relevant points; the callout carries the rest
# place labels clear of the Pareto staircase: frontier points above, others below
LAB={"base_p0":(0.82,0.0,"right","center"),
     "base_p1":(1.12,0.012,"left","bottom"),
     "base_p2":(0.86,0.030,"center","bottom"),
     "base_p3":(1.0,-0.014,"center","top"),
     "base_p4":(1.08,-0.004,"left","center"),
     "base_p5":(1.0,-0.015,"center","top"),
     "base_p6":(0.93,0.016,"right","bottom"),
     "base_p7":(0.88,-0.012,"right","top"),
     "base_p8":(1.07,0.0,"left","center"),
     "base_p9":(1.12,0.028,"left","bottom"),
     "base_p10":(1.0,0.030,"center","bottom")}
SHORTLBL={"base_p8"}  # crowded corner next to the P4 marker/label: pattern id only
for p,c,s in pts:
    ax.scatter(c,s,s=110,marker=marker(p),color=color(p),zorder=3,edgecolor="white",linewidth=0.9)
    if p in LAB:
        mx,dy,ha,va=LAB[p]
        txt=NAME[p] if p in SHORTLBL else f"{NAME[p]} (\\${c:.2f})"
        ax.annotate(txt,(c,s),xytext=(c*mx,s+dy),
                    fontsize=11,ha=ha,va=va,color=color(p))
ax.set_xscale("log")
ax.set_xlabel("Mean compute cost per query (USD, log scale)")
ax.set_ylabel("Overall score (three-judge mean)")
ax.set_ylim(0.20,0.72)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="both",ls=":",alpha=0.35,zorder=0)
# annotate the message in the empty lower-centre, pointing to the dominated P4
ax.annotate("flat top cluster: 2.3$\\times$ cost range,\nno quality gain (P4 \\$5.66 $\\approx$ P7 \\$2.42)",
    xy=(5.66,0.635),xytext=(0.42,0.42),fontsize=11,color=C_CLUSTER,ha="left",
    arrowprops=dict(arrowstyle="->",color=C_CLUSTER,lw=1.0,
                    connectionstyle="arc3,rad=-0.18"))
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="o",color="w",mfc=C_CLUSTER,mec="white",mew=0.8,ms=11,label="GPT-4o top cluster"),
     Line2D([0],[0],marker="s",color="w",mfc=C_OTHER,mec="white",mew=0.8,ms=11,label="GPT-4o (other)"),
     Line2D([0],[0],marker="^",color="w",mfc=C_LOCAL,mec="white",mew=0.8,ms=11,label="Local 7B"),
     Line2D([0],[0],color="#117733",lw=1.8,ls="--",label="Pareto frontier (GPT-4o)")]
ax.legend(handles=leg,loc="lower right",frameon=True,framealpha=0.9,edgecolor="#cccccc",fontsize=11)
plt.tight_layout()
for ext in ("pdf","png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_cost.{ext}",dpi=300,bbox_inches="tight")
# persist quality-per-dollar + frontier membership to canonical
spd={NAME[p]:round(h[p]['mean_3judge']/runs[p]['mean_cost_proxy_usd'],2)
     for p in NAME if p in runs and runs[p].get('mean_cost_proxy_usd')}
cn=json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
cn["cost_per_quality"]={"score_per_usd":spd,
                        "pareto_frontier":[NAME[p] for p,_,_ in eff],
                        "cheapest_on_frontier":NAME[eff[0][0]]}
json.dump(cn,open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json","w"),indent=1)
print("wrote fig_cost; Pareto frontier:",[NAME[p] for p,_,_ in eff]," score/$:",spd)
