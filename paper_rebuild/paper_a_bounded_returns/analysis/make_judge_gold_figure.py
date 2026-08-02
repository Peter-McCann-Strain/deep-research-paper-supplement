#!/usr/bin/env python
"""Paper 2 (judge science) -- cross-family judge-vs-human-gold asymmetry.

From canonical_numbers.json['judge_vs_gold'] (E13' source-4, LitQA2+DeepSearchQA
verifiable-answer slice; single source of truth). The load-bearing finding (per the
canonical _note, after the 2026-06-11 adversarial review) is the CROSS-FAMILY ASYMMETRY:
GPT-5.2's rubric verdicts discriminate answer-correct from answer-incorrect reports
(dimension AUCs > 0.6, bootstrap delta CIs exclude 0) while BOTH Claude judges are
near-chance (AUC ~ 0.50-0.54, CIs span 0). Two panels:
  (A) Per-(judge x dimension) AUC for separating gold-correct from gold-incorrect reports,
      with the 0.5 chance line. Only GPT-5.2 clears it.
  (B) Bootstrap mean-score gaps (correct - incorrect) with 95% CIs; GPT-5.2's exclude 0,
      Claude's straddle 0.
"""
import json, warnings, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s=0.591 (0.9 linewidth fraction) -> the prior declared 10pt base
# printed at ~5.9pt, the worst-offending figure in the paper (well under the 9pt
# floor). base_size=18 (rather than the naive 9/0.591=15.2) because enlarging
# the font also grows the tight-bbox canvas, which pushes the effective scale
# down again -- iterated once against the rendered PDF's measured width to
# land at >=9pt after that feedback, not just after the first-order estimate.
# Migrated off the ad hoc purple-triad palette onto the shared Okabe-Ito palette.
apply_style(base_size=18, axes_linewidth=0.8)
ROOT = "."
C = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
jvg = C["judge_vs_gold"]
pj = jvg["per_judge"]

JLAB = {"gpt52": "GPT-5.2\n(OpenAI)", "claude_opus": "Opus\n(Anthropic)",
        "claude_sonnet": "Sonnet\n(Anthropic)"}
JCOL = {"gpt52": OKABE_ITO["blue"], "claude_opus": OKABE_ITO["vermillion"],
        "claude_sonnet": OKABE_ITO["orange"]}
DIMS = ["factual_accuracy", "citation_quality"]
DLAB = {"factual_accuracy": "Factual", "citation_quality": "Citation"}
judges = ["gpt52", "claude_opus", "claude_sonnet"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.3), gridspec_kw={"width_ratios": [1, 1.05]})

# ---- Panel A: AUC bars grouped by judge ----
ngrp = len(judges); ndim = len(DIMS)
width = 0.36
x = np.arange(ngrp)
for di, dim in enumerate(DIMS):
    aucs = [pj[j][dim]["auc"] for j in judges]
    bars = axA.bar(x + (di - 0.5) * width, aucs, width,
                   color=[JCOL[j] for j in judges],
                   alpha=0.95 if di == 0 else 0.55,
                   edgecolor="white", lw=0.6, zorder=3,
                   label=DLAB[dim])
    for xi, j in zip(x + (di - 0.5) * width, judges):
        axA.text(xi, pj[j][dim]["auc"] + 0.006, f"{pj[j][dim]['auc']:.2f}",
                 ha="center", va="bottom", fontsize=11.6, color="#222")
axA.axhline(0.5, color=OKABE_ITO["vermillion"], lw=1.0, ls="--", zorder=2)
# axes-fraction anchor (not a data coordinate near the Sonnet bars, whose own
# data labels sit right at y~0.5-0.53) so the annotation can never collide with
# a bar-top label regardless of bar heights; found by direct visual inspection,
# adversarial review 2026-07-28, round 27.
axA.text(0.97, 0.56, "chance (0.5)", transform=axA.transAxes,
         color=OKABE_ITO["vermillion"], fontsize=11.9, ha="right", va="center")
axA.set_xticks(x); axA.set_xticklabels([JLAB[j] for j in judges], fontsize=13.1)
axA.set_ylabel("AUC (gold-correct vs incorrect)")
# y-axis starts above 0 (not at 0) to make the 0.5-chance line and the AUC
# spread legible; the chance line itself is drawn so the truncation cannot
# misrepresent the effect as larger than it is. Disclosed in the caption too.
axA.set_ylim(0.45, 0.72)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.3, zorder=0)
# legend: solid=factual, faded=citation -- use proxy
from matplotlib.patches import Patch
axA.legend(handles=[Patch(fc="#555", alpha=0.95, label="Factual accuracy"),
                    Patch(fc="#555", alpha=0.55, label="Citation quality")],
           fontsize=11.9, loc="upper right", frameon=False)
# (y-axis-truncation note lives in the caption, not the title -- a prior version
# duplicated it here as a second title line, which at this figure's width ran long
# enough to visually collide with panel (b)'s title; found by direct visual
# inspection, adversarial review 2026-07-28, round 27.)
axA.set_title("(a) Only the OpenAI judge tracks ground-truth answers",
              fontsize=14.9, loc="left")

# ---- Panel B: bootstrap mean-score gap with CIs ----
labels, deltas, lo, hi, cols, excl = [], [], [], [], [], []
for dim in DIMS:
    for j in judges:
        bd = pj[j][dim]["boot_diff"]
        labels.append(f"{DLAB[dim]} | {j.replace('claude_','').replace('gpt52','GPT-5.2')}")
        deltas.append(bd["delta"]); lo.append(bd["ci95"][0]); hi.append(bd["ci95"][1])
        cols.append(JCOL[j]); excl.append(bd["excludes_0"])
yy = np.arange(len(labels))[::-1]
for i, yv in enumerate(yy):
    axB.plot([lo[i], hi[i]], [yv, yv], color=cols[i], lw=1.4, zorder=3,
             alpha=1.0 if excl[i] else 0.5)
    axB.scatter([deltas[i]], [yv], s=46,
                color=cols[i], zorder=4, edgecolor="white", lw=0.8,
                marker="o" if excl[i] else "s")
axB.axvline(0, color="#444", lw=0.8)
axB.set_yticks(yy[::-1] if False else np.arange(len(labels))[::-1])
axB.set_yticks(yy); axB.set_yticklabels(labels, fontsize=11.9)
# NB: plain matplotlib text (usetex off) -- use '%', not '\\%'; smaller font so the
# centred label stays inside the figure box instead of clipping at the right edge.
axB.set_xlabel("Mean rubric score: gold-correct $-$ gold-incorrect (95% boot CI)",
               fontsize=13.7)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="x", ls=":", alpha=0.3, zorder=0)
axB.set_title("(b) GPT-5.2 gaps exclude 0; Claude gaps straddle 0", fontsize=14.9, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_judge_gold.{ext}",
                dpi=200, bbox_inches="tight")
g = pj["gpt52"]
print(f"wrote fig_judge_gold; n_slice_queries={jvg['slice']['n_queries']}, "
      f"effective_signal_clusters={jvg['effective_signal_clusters']}; "
      f"GPT-5.2 factual AUC={g['factual_accuracy']['auc']:.3f} (delta {g['factual_accuracy']['boot_diff']['delta']:+.3f} "
      f"excl0={g['factual_accuracy']['boot_diff']['excludes_0']}); "
      f"Opus factual AUC={pj['claude_opus']['factual_accuracy']['auc']:.3f} "
      f"excl0={pj['claude_opus']['factual_accuracy']['boot_diff']['excludes_0']}")
