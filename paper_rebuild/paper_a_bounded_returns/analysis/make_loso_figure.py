#!/usr/bin/env python
"""LOSO source-jackknife figure (the composition-artefact defence).

Reads canonical_numbers.json['loso_robustness']. Drops one benchmark source at a time
(custom / deepsearch_qa / draco / litqa2 / research_qa), refits the Gate-1 crossed-RE
mixed model and re-derives the Gate-3 judge-robust pairwise separations and 3-judge rank
table. Shows the headline structure is not a composition artefact of any single source:

  (a) ICC_query (between-query variance share) under each source drop vs the full panel,
      with the full-panel band; all drops sit close to the full-panel value.
  (b) Top-1 pattern and maximum rank displacement vs the full rank table under each drop;
      the winner never changes and rank churn is small.

June-2026 framing: leave-one-source-out is the composition-artefact defence for multi-source
LLM-judge evaluations. CPU-only; no API.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
L = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["loso_robustness"]

base = L["full_panel_baseline"]
full_icc = base["gate1_mixed"]["icc_query"]
full_g3 = base["gate3_judge_robust_of_55"]
full_top1 = base["rank_table"]["rank_desc"][0]
loso = L["leave_one_source_out"]
SLAB = {"deepsearch_qa": "DeepSearch-QA", "draco": "DRACO", "litqa2": "LitQA2",
        "custom": "Custom", "research_qa": "Research-QA"}

srcs = list(loso.keys())
iccs = [loso[s]["gate1_mixed"]["icc_query"] for s in srcs]
disp = [loso[s]["rank_max_displacement_vs_full"] for s in srcs]
g3 = [loso[s]["gate3_judge_robust_of_55"] for s in srcs]
top1ok = [loso[s]["top1_matches_full"] for s in srcs]
ndrop = [loso[s]["n_dropped_queries"] for s in srcs]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1.05, 1]})

# ---- Panel A: ICC_query stability ----
x = np.arange(len(srcs))
axA.axhline(full_icc, color="#1f4e79", ls="--", lw=1.4, zorder=2,
            label=f"full panel ({full_icc:.3f})")
axA.bar(x, iccs, color="#7f9bb5", width=0.6, zorder=3, edgecolor="white")
for xi, v, nd in zip(x, iccs, ndrop):
    axA.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.0, color="#333")
axA.set_xticks(x)
axA.set_xticklabels([f"$-${SLAB[s]}\n($n_q$$-${nd})" for s, nd in zip(srcs, ndrop)], fontsize=8.0)
axA.set_ylabel("ICC$_{\\mathrm{query}}$ (between-query variance share)")
axA.set_ylim(0, max(max(iccs), full_icc) * 1.25)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axA.legend(loc="upper right", frameon=False, fontsize=8.6)
axA.set_title("(a) Query-variance share is source-robust", fontsize=10.2, loc="left")

# ---- Panel B: rank displacement + top-1 stability + Gate-3 separations ----
axB2 = axB.twinx()
axB.bar(x, disp, color="#c0a060", width=0.55, zorder=3, label="max rank displacement")
for xi, d_, ok in zip(x, disp, top1ok):
    mark = "$\\checkmark$ top-1 held" if ok else "$\\times$ top-1 flip"
    axB.text(xi, d_ + 0.06, mark, ha="center", va="bottom", fontsize=7.6,
             color=("#1f7a3d" if ok else "#c0504d"), rotation=0)
axB2.plot(x, g3, "o-", color="#1f4e79", lw=1.6, ms=6, zorder=4, label="Gate-3 separations")
axB2.axhline(full_g3, color="#1f4e79", ls=":", lw=1.0, zorder=2)
axB2.text(len(srcs) - 0.5, full_g3, f" full = {full_g3}/55", fontsize=7.8, color="#1f4e79", va="bottom", ha="right")
axB.set_xticks(x); axB.set_xticklabels([f"$-${SLAB[s]}" for s in srcs], fontsize=8.2)
axB.set_ylabel("max rank displacement vs full")
axB.set_ylim(0, max(max(disp), 1) + 1.4)
axB2.set_ylabel("Gate-3 judge-robust separations (of 55)", color="#1f4e79")
axB2.tick_params(axis="y", labelcolor="#1f4e79")
axB2.set_ylim(0, full_g3 + 8)
axB.spines[["top"]].set_visible(False); axB2.spines[["top"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.3, zorder=0)
axB.set_title("(b) Winner and rank order survive every drop", fontsize=10.2, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_loso.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_loso; full ICC_q={full_icc:.4f}, drops {[round(v,4) for v in iccs]}; "
      f"max disp={max(disp)}, top1 held all={all(top1ok)}; "
      f"Gate-3 full={full_g3}, drops={g3}")
