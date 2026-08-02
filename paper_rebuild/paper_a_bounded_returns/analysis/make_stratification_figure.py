#!/usr/bin/env python
"""Complexity-gradient figure: the orchestration premium (cluster mean minus P0) grows
with task difficulty. Recomputed from df_overall_scores (3-judge, sonnet-corrected) so it
cannot drift from 04_stratification.md. Shows the flat cluster is a competence-conditioned
main effect, not a task-averaging artifact -> motivates the scoped thesis.
"""
import pandas as pd, numpy as np, json, warnings, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s~0.76 (0.84 linewidth fraction) -> declared 13pt prints at ~9.9pt,
# already clears the floor; kept as-is besides the shared font-family fix.
apply_style(base_size=13, legend_fontsize=12)
# The two series already carry redundant marker shapes (P0 = circle, cluster =
# square) so the colour distinction survives greyscale too.
C_P0 = OKABE_ITO["bluish_green"]; C_CLUSTER = OKABE_ITO["blue"]
ROOT = "."
A = f"{ROOT}/data/analysis"
ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
ov["ovc"] = ov["overall_score"].where(~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
q = pd.read_parquet(f"{A}/df_queries.parquet")[["query_id", "source"]]
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
CLUSTER = [f"base_p{i}" for i in (1, 4, 5, 6, 7, 8)]
base = ov[ov.pattern.str.match(r"^base_p\d+$") & ov.judge.isin(PANEL)].merge(q, on="query_id")
# per (pattern, source) 3-judge mean
m = base.groupby(["pattern", "source"], observed=True)["ovc"].mean().unstack()
# order sources by P0 competence (proxy for difficulty), hard -> easy
src_order = m.loc["base_p0"].sort_values().index.tolist()
SLAB = {"deepsearch_qa": "DeepSearch-QA", "draco": "DRACO", "litqa2": "LitQA2",
        "custom": "Custom", "research_qa": "Research-QA"}
p0 = m.loc["base_p0", src_order].values
clu = m.loc[CLUSTER, src_order].mean().values
gap = clu - p0

fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(len(src_order))
ax.plot(x, p0, "o-", color=C_P0, lw=2.0, ms=10, mec="white", mew=0.9,
        label="P0 single-pass", zorder=3)
ax.plot(x, clu, "s-", color=C_CLUSTER, lw=2.0, ms=10, mec="white", mew=0.9,
        label="orchestrated cluster (mean)", zorder=3)
ax.fill_between(x, p0, clu, color=C_CLUSTER, alpha=0.08, zorder=1)
for xi, g in zip(x, gap):
    ax.annotate(f"+{g:.2f}", (xi, (p0[xi] + clu[xi]) / 2), fontsize=11,
                ha="center", va="center", color=C_CLUSTER, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88))
ax.set_xticks(x); ax.set_xticklabels([SLAB.get(s, s) for s in src_order])
ax.set_xlabel("Benchmark source (ordered hardest $\\rightarrow$ easiest for the single-pass baseline)")
ax.set_ylabel("Overall score (three-judge mean)")
ax.set_ylim(0.25, 0.80)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.legend(loc="lower right", frameon=False, fontsize=12)
ax.set_title("Orchestration pays only when the task is hard for a single pass", fontsize=13, loc="left")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_stratification.{ext}", dpi=300, bbox_inches="tight")
# persist the gradient to canonical
out = {SLAB.get(s, s): {"p0": round(float(p0[i]), 4), "cluster": round(float(clu[i]), 4),
                        "gap": round(float(gap[i]), 4)} for i, s in enumerate(src_order)}
p = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
p["complexity_gradient"] = {"source_order_hard_to_easy": [SLAB.get(s, s) for s in src_order],
                            "per_source": out,
                            "gap_hardest": round(float(gap[0]), 4),
                            "gap_easiest": round(float(gap[-1]), 4)}
json.dump(p, open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json", "w"), indent=1)
print("wrote fig_stratification; gap hard->easy:", [round(float(g), 3) for g in gap])
