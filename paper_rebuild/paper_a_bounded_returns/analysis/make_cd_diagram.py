#!/usr/bin/env python
"""Clean Friedman--Nemenyi critical-difference diagram (no baked-in title) from
verified per-query 3-judge mean ranks over the 11 base patterns."""
import pandas as pd, numpy as np, warnings, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style
# print scale s~0.81 (0.95 linewidth fraction) -> declared 12pt prints at ~9.7pt,
# already clears the floor; kept as-is besides the shared font-family fix.
apply_style(base_size=12)
ROOT="."
ov=pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
ov["ovc"]=ov["overall_score"].where(~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
base=ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(["gpt52","claude_opus","claude_sonnet"])]
perq=base.groupby(["pattern","query_id"],observed=True)["ovc"].mean().unstack(0)
pats=[f"base_p{i}" for i in range(11)]
perq=perq[pats].dropna()
N=len(perq); k=len(pats)
# rank per query: 1 = best (highest score)
ranks=perq.rank(axis=1,ascending=False)
avg=ranks.mean().sort_values()
NAME={f"base_p{i}":n for i,n in zip(range(11),
  ["P0 Single-pass","P1 Iterative RAG","P2 Supervisor","P3 MERIDIAN","P4 STORM",
   "P5 Hier. W&D","P6 Reactive","P7 Graph","P8 Beam","P9 Qwen2.5-7B","P10 DeepResearcher"])}
# Nemenyi CD at alpha=0.05
q05={10:3.164,11:3.219,12:3.268}[k]
CD=q05*np.sqrt(k*(k+1)/(6.0*N))
print(f"N={N} k={k} CD={CD:.3f}")
methods=list(avg.index); rvals=avg.values
lo,hi=1,k
# Okabe-Ito colourblind-safe palette (blue / bluish-green / vermillion) with a
# redundant marker shape per category (circle / square / triangle) placed on each elbow.
C_CLUSTER="#0072B2"; C_OTHER="#009E73"; C_LOCAL="#D55E00"
CLU={"base_p1","base_p4","base_p5","base_p6","base_p7","base_p8"}
LOC={"base_p9","base_p10"}
def cat(p): return "cluster" if p in CLU else ("local" if p in LOC else "other")
def color(p): return {"cluster":C_CLUSTER,"other":C_OTHER,"local":C_LOCAL}[cat(p)]
def marker(p): return {"cluster":"o","other":"s","local":"^"}[cat(p)]
fig,ax=plt.subplots(figsize=(7.8,4.2)); ax.set_xlim(lo-0.4,hi+0.4); ax.set_ylim(0,1)
ax.axis("off")
# top axis
yaxis=0.78
ax.plot([lo,hi],[yaxis,yaxis],"k-",lw=1.3)
for r in range(lo,hi+1):
    ax.plot([r,r],[yaxis,yaxis+0.02],"k-",lw=1.1); ax.text(r,yaxis+0.05,str(r),ha="center",va="bottom",fontsize=11)
ax.text((lo+hi)/2,yaxis+0.14,"average rank (1 = best)",ha="center",fontsize=11.5)
# CD bar
ax.plot([lo,lo+CD],[yaxis+0.20,yaxis+0.20],"k-",lw=2.4)
ax.plot([lo,lo],[yaxis+0.18,yaxis+0.22],"k-",lw=1.1); ax.plot([lo+CD,lo+CD],[yaxis+0.18,yaxis+0.22],"k-",lw=1.1)
ax.text(lo+CD/2,yaxis+0.23,f"CD = {CD:.2f}",ha="center",fontsize=11)
# place labels: left half / right half
half=int(np.ceil(k/2))
left=methods[:half]; right=methods[half:][::-1]
def elbow(rank,name,side,yi,m,col,mk):
    ax.plot([rank,rank],[yaxis,yi],color=col,lw=1.6)
    xend = lo-0.35 if side=="L" else hi+0.35
    ax.plot([rank,xend],[yi,yi],color=col,lw=1.6)
    # redundant shape marker on the rank tick, so the category reads in greyscale
    ax.plot([rank],[yaxis],marker=mk,color=col,ms=8,mec="white",mew=0.8,zorder=6,clip_on=False)
    ax.text(xend+(-0.05 if side=="L" else 0.05),yi,f"{name}",
            ha="right" if side=="L" else "left",va="center",fontsize=11,color=col)
y0=0.62; dy=0.115
for i,m in enumerate(left): elbow(rvals[methods.index(m)],NAME[m],"L",y0-i*dy,m,color(m),marker(m))
for i,m in enumerate(right): elbow(rvals[methods.index(m)],NAME[m],"R",y0-i*dy,m,color(m),marker(m))
# clique bars (consecutive groups within CD)
ybar=yaxis-0.05; used=[False]*k
groups=[]
i=0
while i<k:
    j=i
    while j+1<k and (rvals[j+1]-rvals[i])<CD: j+=1
    if j>i: groups.append((rvals[i],rvals[j]))
    i=j+1 if j>i else i+1
# draw maximal cliques (simple sweep)
cl=[]
for a in range(k):
    b=a
    while b+1<k and rvals[b+1]-rvals[a]<CD: b+=1
    if b>a: cl.append((a,b))
# keep maximal
maxcl=[c for c in cl if not any(c!=d and d[0]<=c[0] and c[1]<=d[1] for d in cl)]
off=0
for (a,b) in maxcl:
    ax.plot([rvals[a]-0.03,rvals[b]+0.03],[ybar-off,ybar-off],color="#333",lw=3.4,solid_capstyle="round")
    off+=0.035
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="o",color=C_CLUSTER,mfc=C_CLUSTER,mec="white",mew=0.8,lw=3,ms=9,label="GPT-4o top cluster"),
     Line2D([0],[0],marker="s",color=C_OTHER,mfc=C_OTHER,mec="white",mew=0.8,lw=3,ms=9,label="GPT-4o (other)"),
     Line2D([0],[0],marker="^",color=C_LOCAL,mfc=C_LOCAL,mec="white",mew=0.8,lw=3,ms=9,label="Local 7B")]
ax.legend(handles=leg,loc="upper center",ncol=3,frameon=False,fontsize=10.5,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout()
for ext in ("pdf","png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_cd_clean.{ext}",dpi=300,bbox_inches="tight")
print("wrote fig_cd_clean; avg ranks:",{NAME[m]:round(float(avg[m]),2) for m in methods})
