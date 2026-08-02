#!/usr/bin/env python
"""Matched-budget disentanglement figure (the compute-vs-architecture probe).

Reads canonical_numbers.json['disentanglement'] (single source of truth) and shows,
for the headline P1 arm, that the P1>P0 overall advantage is only significant at FULL
budget and collapses to non-significant once compute is clamped (~12x -> ~3.2x P0):

  (a) Unmatched vs matched paired delta with 95% CIs and Wilcoxon p; the matched CI
      spans 0. ~40% of the gap is compute, ~60% is architecture residual.
  (b) Per-dimension unmatched vs matched gap: the surviving residual is analytical_depth
      (synthesis-bound); the erased component is the retrieval-bound dimensions
      (citation_quality, coverage) -- exactly the oracle-probe poles.

June-2026 framing: matched-budget controls for compute confound (compute-vs-architecture).
CPU-only; no API.
"""
import json, warnings, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s=0.587 (0.92 linewidth fraction) -> the prior declared 10pt base
# printed at ~5.9pt, well under the 9pt floor. base_size=18 (rather than the
# naive 9/0.587=15.3) because enlarging the font also grows the tight-bbox
# canvas, pushing the effective scale down again -- iterated against the
# rendered PDF's measured width to land at >=9pt after that feedback. Every
# hardcoded inline fontsize below is scaled off the original 1.53x estimate;
# Migrated onto the shared Okabe-Ito palette (was an ad hoc navy/tan scheme).
apply_style(base_size=18, axes_linewidth=0.8)
C_FULL = OKABE_ITO["blue"]
C_MATCHED = OKABE_ITO["vermillion"]
ROOT = "."
D = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["disentanglement"]

arm = D["p1_arm"]
um, mt = arm["unmatched"], arm["matched"]
# Schema updated by the stats-fix builder: the point split is now '*_pct_point' and
# carries a wide bootstrap CI, so it must be shown as indicative, not a precise split.
cmp_pct, arch_pct = arm["compute_attributable_pct_point"], arm["architecture_residual_pct_point"]
cmp_ci = arm.get("compute_attributable_pct_ci", {})
DLAB = {"citation_quality": "Citation quality", "information_recall": "Information recall",
        "instruction_following": "Instruction following", "coverage": "Coverage",
        "analytical_depth": "Analytical depth", "organization": "Organization",
        "attribution_quality": "Attribution quality", "logical_coherence": "Logical coherence",
        "factual_accuracy": "Factual accuracy"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})

# ---- Panel A: overall delta, unmatched vs matched ----
labels = ["full budget\n(~12$\\times$ P0)", "matched budget\n(~3.2$\\times$ P0)"]
deltas = [um["delta"], mt["delta"]]
cis = [um["ci95"], mt["ci95"]]
ps = [um["wilcoxon_p"], mt["wilcoxon_p"]]
sig = [um["significant"], mt["significant"]]
x = np.arange(2)
cols = [C_FULL, C_MATCHED]
for i in range(2):
    axA.bar(i, deltas[i], color=cols[i], width=0.56, zorder=3)
    lo, hi = cis[i]
    axA.plot([i, i], [lo, hi], color="#333", lw=1.2, zorder=4)
    axA.plot([i - 0.08, i + 0.08], [lo, lo], color="#333", lw=1.2)
    axA.plot([i - 0.08, i + 0.08], [hi, hi], color="#333", lw=1.2)
    star = "significant" if sig[i] else "n.s."
    axA.text(i, hi + 0.006, f"$p$={ps[i]:.3f}\n({star})", ha="center", va="bottom", fontsize=12.5,
             color=(cols[i] if sig[i] else "#777"))
axA.axhline(0, color="#444", lw=0.8)
axA.set_xticks(x); axA.set_xticklabels(labels)
axA.set_ylabel("P1 $-$ P0 overall (paired, GPT-5.2)")
axA.set_ylim(-0.03, 0.16)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
# The compute/architecture share is a ratio of two noisy paired gaps: the canonical CI is
# uninformatively wide, so the title reports point estimate + CI, not a certified split.
def _rnd(v):  # round half away from zero, matching the paper's [-29%, 149%]
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(-v + 0.5))
ci_lo, ci_hi = cmp_ci.get("ci95", (float("nan"), float("nan")))
axA.set_title("(a) Clamping compute leaves the gap borderline\n"
              f"compute share: point ~{cmp_pct}%, 95% CI [{_rnd(ci_lo)}%, {_rnd(ci_hi)}%] (wide)",
              fontsize=14.4, loc="left")

# ---- Panel B: per-dimension unmatched vs matched gap ----
pd_ = arm["per_dimension"]
rows = sorted(pd_.items(), key=lambda kv: kv[1]["unmatched_gap"])
y = np.arange(len(rows))
h = 0.38
um_vals = [v["unmatched_gap"] for _, v in rows]
mt_vals = [v["matched_gap"] for _, v in rows]
axB.barh(y + h / 2, um_vals, height=h, color=C_FULL, label="full budget", zorder=3)
axB.barh(y - h / 2, mt_vals, height=h, color=C_MATCHED, label="matched budget", zorder=3)
axB.axvline(0, color="#444", lw=0.8)
axB.set_yticks(y); axB.set_yticklabels([DLAB[k] for k, _ in rows])
axB.set_xlabel("P1 $-$ P0 gap (paired)")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axB.legend(loc="lower right", frameon=False, fontsize=13.2)
axB.set_title("(b) Residual = analytical depth; erased = retrieval-bound dims",
              fontsize=15.3, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_disentanglement.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_disentanglement; unmatched {um['delta']:+.4f} (p={um['wilcoxon_p']:.3f}, sig={um['significant']}); "
      f"matched {mt['delta']:+.4f} (p={mt['wilcoxon_p']:.3f}, sig={mt['significant']}); "
      f"~{cmp_pct}% compute / ~{arch_pct}% architecture")
