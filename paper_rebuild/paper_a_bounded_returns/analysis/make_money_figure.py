#!/usr/bin/env python
"""Figure 1 ("money figure"): per-pattern overall score with the judge-robust top
cluster bracketed and the model-scale gap annotated. Thesis-independent; built from
verified per-query 3-judge means."""
import pandas as pd, numpy as np, warnings, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, C_CLUSTER, C_OTHER, C_LOCAL
# Larger base typography: this figure is scaled to ~0.92 linewidth (~6in) in the
# single-column article, so in-figure fonts must be generous to clear the ~9pt
# print floor at this print scale (measured s=0.792, so declared 14pt -> ~11pt
# printed, comfortably above the floor including the legend). Okabe-Ito
# colourblind-safe palette + distinct marker shapes give a redundant
# (colour + shape) 3-way category encoding that survives greyscale.
apply_style(base_size=14, legend_fontsize=12)
ROOT="."
ov=pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
ov["ovc"]=ov["overall_score"].where(~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
base=ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(["gpt52","claude_opus","claude_sonnet"])]
perq=base.groupby(["pattern","query_id"],observed=True)["ovc"].mean()
NAME={f"base_p{i}":n for i,n in zip(range(11),
  ["P0 Single-pass","P1 Iterative RAG","P2 Supervisor","P3 MERIDIAN","P4 STORM",
   "P5 Hier. W&D","P6 Reactive","P7 Graph","P8 Beam","P9 Qwen2.5-7B","P10 DeepResearcher"])}
CLUSTER={"base_p1","base_p4","base_p5","base_p6","base_p7","base_p8"}
LOCAL={"base_p9","base_p10"}
def cat(p): return "cluster" if p in CLUSTER else ("local" if p in LOCAL else "other")
def color(p): return {"cluster":C_CLUSTER,"other":C_OTHER,"local":C_LOCAL}[cat(p)]
def marker(p): return {"cluster":"o","other":"s","local":"^"}[cat(p)]
rng=np.random.default_rng(7)
rows=[]
for p in NAME:
    v=perq.xs(p,level=0).values
    boot=[rng.choice(v,len(v),replace=True).mean() for _ in range(5000)]
    rows.append((p, v.mean(), np.percentile(boot,2.5), np.percentile(boot,97.5)))
df=pd.DataFrame(rows,columns=["pat","mean","lo","hi"]).sort_values("mean")
y=np.arange(len(df))
cols=[color(p) for p in df.pat]

fig,ax=plt.subplots(figsize=(7.6,5.4))
# cluster band
cl_y=[i for i,p in enumerate(df.pat) if p in CLUSTER]
ax.axhspan(min(cl_y)-0.45,max(cl_y)+0.45,color=C_CLUSTER,alpha=0.08,zorder=0)
# error bars (shape-neutral grey whiskers; coloured shaped markers drawn on top)
ax.errorbar(df["mean"],y,xerr=[df["mean"]-df["lo"],df["hi"]-df["mean"]],
            fmt="none",capsize=4,lw=1.6,ecolor="#555",zorder=3)
for yi,(_,r) in zip(y,df.iterrows()):
    ax.plot(r["mean"],yi,marker=marker(r["pat"]),ms=9,color=color(r["pat"]),
            mec="white",mew=0.9,zorder=4)
ax.set_yticks(y); ax.set_yticklabels([NAME[p] for p in df.pat])
ax.set_xlabel("Overall score (three-judge mean, 0–1 rubric)")
ax.set_xlim(0,0.85); ax.set_ylim(-0.7,len(df)-0.3)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x",ls=":",alpha=0.4,zorder=0)
# cluster bracket label (parked in the clear lower-right whitespace above the legend)
ymid=np.mean(cl_y)
ax.annotate("Top cluster (6 patterns;\n0/10 pairs among the inner five\njudge-robustly separated;\nP5 joins under uncorrected\nequivalence)",
    xy=(0.70,max(cl_y)+0.1),xytext=(0.80,ymid),fontsize=10.5,color=C_CLUSTER,va="center",ha="left",
    bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=C_CLUSTER,lw=0.9))
# model-scale gap arrow (P4 top cluster vs P9)
p4=df[df.pat=="base_p4"]["mean"].values[0]; y4=[i for i,p in enumerate(df.pat) if p=="base_p4"][0]
p9=df[df.pat=="base_p9"]["mean"].values[0]; y9=[i for i,p in enumerate(df.pat) if p=="base_p9"][0]
ax.add_patch(FancyArrowPatch((p9,y9+0.0),(p4,y4-0.0),arrowstyle="<->",mutation_scale=14,
            lw=1.6,color=C_LOCAL,ls="-",zorder=5,shrinkA=8,shrinkB=8))
ax.text((p4+p9)/2-0.02,(y4+y9)/2,f"frontier$-$7B gap\n$\\approx${p4-p9:.2f}  (P4 $\\leftrightarrow$ P9)",
        fontsize=10.5,color=C_LOCAL,ha="right",va="center",style="italic")
# family legend (colour + shape redundant), parked in clear lower-right whitespace
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="o",color="w",mfc=C_CLUSTER,mec="white",mew=0.8,ms=10,label="GPT-4o top cluster"),
     Line2D([0],[0],marker="s",color="w",mfc=C_OTHER,mec="white",mew=0.8,ms=10,label="GPT-4o (other)"),
     Line2D([0],[0],marker="^",color="w",mfc=C_LOCAL,mec="white",mew=0.8,ms=10,label="Local 7B")]
# park in the genuinely empty lower-right quadrant (the two lowest patterns P9/P10
# sit at x<=0.35, so the right ~60% of those rows is clear whitespace)
ax.legend(handles=leg,loc="lower left",bbox_to_anchor=(0.52,0.0),frameon=True,
          framealpha=0.92,edgecolor="#cccccc",fontsize=11)
plt.tight_layout()
for ext in ("pdf","png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig1_money.{ext}",dpi=300,bbox_inches="tight")
print("wrote fig1_money.pdf/png ; cluster band rows:",cl_y,"scale gap:",round(p4-p9,3))
