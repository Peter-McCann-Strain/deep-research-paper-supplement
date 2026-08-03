#!/usr/bin/env python
"""Oracle-retrieval figure (the counterfactual decomposition).

Two panels, both from canonical_numbers.json['oracle'] (single source of truth):
  (A) Dual mechanism: pooled cluster per-dimension oracle-minus-baseline delta with
      95% bootstrap CIs. Citation quality rises far (CI excludes 0); factual accuracy
      does not move (CI spans 0). Retrieval-bound vs synthesis-bound, shown not argued.
  (B) Gap compression: P0 and the orchestrated cluster, baseline vs oracle. Ideal
      sources lift the simple pipeline more, so the architecture gap narrows.
"""
import json, warnings, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s=0.644 (full linewidth on a wide 2-panel canvas) -> declared 14pt
# prints at ~9.0pt, right at the floor; bumped from 13 to compensate for the
# harder shrink this wide figure takes versus the single-panel ones.
apply_style(base_size=14, legend_fontsize=12)
C_POS = OKABE_ITO["blue"]         # positive, CI excludes 0
C_NULL = "#999999"                # CI spans 0 (neutral grey, not a category colour)
C_NEG = OKABE_ITO["vermillion"]   # negative pole / synthesis-bound
ROOT = "."
o = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["oracle"]

DLAB = {"citation_quality": "Citation quality", "information_recall": "Information recall",
        "instruction_following": "Instruction following", "coverage": "Coverage",
        "analytical_depth": "Analytical depth", "organization": "Organization",
        "attribution_quality": "Attribution quality", "logical_coherence": "Logical coherence",
        "factual_accuracy": "Factual accuracy"}
cd = o["cluster_dims"]
rows = sorted([(k, v["delta"], v["ci95"]) for k, v in cd.items() if v], key=lambda r: r[1])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- Panel A: dual mechanism ----
y = np.arange(len(rows))
for i, (k, dlt, ci) in enumerate(rows):
    excl0 = ci[0] > 0 or ci[1] < 0
    col = C_POS if (dlt > 0 and excl0) else (C_NULL if not excl0 else C_NEG)
    axA.barh(i, dlt, color=col, height=0.66, zorder=3)
    axA.plot(ci, [i, i], color="#333", lw=1.3, zorder=4)
    axA.plot([ci[0], ci[0]], [i - 0.12, i + 0.12], color="#333", lw=1.3)
    axA.plot([ci[1], ci[1]], [i - 0.12, i + 0.12], color="#333", lw=1.3)
axA.axvline(0, color="#444", lw=0.8)
axA.set_yticks(y); axA.set_yticklabels([DLAB[k] for k, _, _ in rows])
axA.set_xlabel("Oracle $-$ baseline score (paired, GPT-5.2)")
axA.set_xlim(-0.06, 0.29)  # citation-quality CI upper whisker reaches 0.264; keep the cap inside the axes
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axA.set_title("(a) Better sources fix citations, not facts", fontsize=13, loc="left")
# call out the two poles
ci_cit = cd["citation_quality"]; ci_fac = cd["factual_accuracy"]
icit = [i for i, (k, _, _) in enumerate(rows) if k == "citation_quality"][0]
ifac = [i for i, (k, _, _) in enumerate(rows) if k == "factual_accuracy"][0]
axA.annotate("retrieval-bound", xy=(ci_cit["delta"], icit), xytext=(0.115, icit - 1.4),
             fontsize=11, color=C_POS,
             arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.0))
axA.annotate("synthesis-bound\n(no detectable change, ±0.05 TOST)", xy=(ci_fac["delta"], ifac),
             xytext=(0.045, ifac + 0.5), fontsize=11, color=C_NEG,
             arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.0))

# ---- Panel B: gap compression ----
pp = o["per_pattern"]; CLUSTER = ["p1", "p4", "p5", "p6", "p7", "p8"]
p0_b = pp["p0"]["overall"]["base_mean"]; p0_o = pp["p0"]["overall"]["oracle_mean"]
cl_b = float(np.mean([pp[p]["overall"]["base_mean"] for p in CLUSTER]))
cl_o = float(np.mean([pp[p]["overall"]["oracle_mean"] for p in CLUSTER]))
C_P0 = "#009E73"; C_CL = "#0072B2"   # Okabe-Ito: P0 bluish-green (circle), cluster blue (square)
# categorical x positions: spread apart so the (now larger) tick labels don't collide
xb, xo = 0, 1.6
axB.plot([xb, xo], [p0_b, p0_o], "-", color=C_P0, lw=2.0, zorder=2)
axB.plot([xb, xo], [cl_b, cl_o], "-", color=C_CL, lw=2.0, zorder=2)
axB.scatter([xb, xo], [p0_b, p0_o], s=120, marker="o", color=C_P0, zorder=3, edgecolor="white", lw=1.1)
axB.scatter([xb, xo], [cl_b, cl_o], s=120, marker="s", color=C_CL, zorder=3, edgecolor="white", lw=1.1)
axB.text(xb - 0.12, p0_b, "P0 single-pass", ha="right", va="center", fontsize=11, color=C_P0,
         bbox=dict(fc="white", ec="none", pad=0.5))
axB.text(xb - 0.12, cl_b, "orchestrated cluster", ha="right", va="center", fontsize=11, color=C_CL,
         bbox=dict(fc="white", ec="none", pad=0.5))
# gap brackets
for x, gb, ptop, pbot, lab in [(xb, o["gap_p0_to_cluster_base"], cl_b, p0_b, "baseline"),
                               (xo, o["gap_p0_to_cluster_oracle"], cl_o, p0_o, "oracle")]:
    axB.annotate("", xy=(x, ptop), xytext=(x, pbot),
                 arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    axB.text(x + 0.10, (ptop + pbot) / 2,
             f"gap\n{gb:.3f}", fontsize=10.5, va="center",
             ha="left", color="#333", bbox=dict(fc="white", ec="none", pad=0.3))
axB.set_xticks([xb, xo]); axB.set_xticklabels(["live retrieval", "oracle sources"])
axB.set_xlim(-1.35, 2.15)
axB.set_ylabel("Overall score (GPT-5.2, variance set)")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axB.set_title("(b) The orchestration gap narrows", fontsize=13, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_oracle.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote fig_oracle; citation {cd['citation_quality']['delta']:+.3f} CI{cd['citation_quality']['ci95']}, "
      f"factual {cd['factual_accuracy']['delta']:+.3f} CI{cd['factual_accuracy']['ci95']}; "
      f"gap {o['gap_p0_to_cluster_base']:.3f} -> {o['gap_p0_to_cluster_oracle']:.3f}")
