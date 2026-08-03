#!/usr/bin/env python
"""E5 gold-fraction dose-response figure (the synthesis-bound confirmation).

Reads canonical_numbers.json['e5_dose_response']. Pooled over P0/P1/P4, factual accuracy
is FLAT in the gold fraction injected into the context (prereg-flat slope, one-sided 95%
upper bound below the equivalence margin), while citation quality drifts up modestly.
Injecting ground-truth facts does not raise the judged factual score: the factual ceiling
is synthesis-bound, not retrieval-bound -- the same mechanism the oracle probe shows.

CPU-only; no API.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
E = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["e5_dose_response"]

pm = E["per_fraction_means"]["pooled"]
gcells = ["g000", "g025", "g050", "g075", "g100"]
gf = [pm[c]["gold_fraction"] for c in gcells]
fac = [pm[c]["factual_accuracy_mean"] for c in gcells]
cit = [pm[c]["citation_quality_mean"] for c in gcells]
n_per = pm["g000"]["n"]
# binomial-ish SE per cell mean for visual error bars (mean of {0,1} proportions, n queries)
fac_se = [np.sqrt(max(p * (1 - p), 1e-6) / n_per) for p in fac]
cit_se = [np.sqrt(max(p * (1 - p), 1e-6) / n_per) for p in cit]

fslope = E["factual_accuracy_slope"]
cslope = E["citation_quality_slope"]
il = E["per_fraction_means"]["pooled"]["interleaved"]

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.errorbar(gf, fac, yerr=fac_se, fmt="o-", color="#c0504d", lw=1.8, ms=7, capsize=3,
            label="factual accuracy", zorder=4)
ax.errorbar(gf, cit, yerr=cit_se, fmt="s-", color="#1f4e79", lw=1.8, ms=6, capsize=3,
            label="citation quality", zorder=3)
# interleaved (context-overload) point off-axis at x=1.15
ax.errorbar([1.16], [il["factual_accuracy_mean"]], yerr=[np.sqrt(il["factual_accuracy_mean"]*(1-il["factual_accuracy_mean"])/il["n"])],
            fmt="D", color="#c0504d", ms=6, mfc="white", capsize=3, zorder=4)
ax.errorbar([1.16], [il["citation_quality_mean"]], yerr=[np.sqrt(il["citation_quality_mean"]*(1-il["citation_quality_mean"])/il["n"])],
            fmt="D", color="#1f4e79", ms=6, mfc="white", capsize=3, zorder=3)
ax.text(1.16, il["factual_accuracy_mean"] - 0.022, "interleaved", ha="center", va="top", fontsize=7.6, color="#555")
ax.axvline(1.08, color="#ccc", lw=0.8, ls=":")
ax.set_xlabel("Gold fraction injected into context")
ax.set_ylabel("Judged score (GPT-5.2, pooled P0/P1/P4)")
ax.set_xticks(gf + [1.16]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0", "intl."])
ax.set_ylim(0.18, 0.37)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.legend(loc="upper left", frameon=False, fontsize=9)
# annotate the prereg slope verdicts
ax.text(0.02, 0.02,
        f"factual slope = {fslope['slope']:+.4f} (prereg-flat; one-sided 95\\% UB "
        f"{fslope['one_sided_ci95']:+.3f} $<$ {fslope['margin']:.2f} margin)\n"
        f"citation slope = {cslope['slope']:+.4f} (prereg monotone; one-sided 95\\% LB "
        f"{cslope['one_sided_ci95']:+.3f})",
        transform=ax.transAxes, fontsize=7.8, color="#444", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))
ax.set_title("Injecting gold facts does not raise factual accuracy",
             fontsize=10.5, loc="left")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_e5_dose.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_e5_dose; factual slope {fslope['slope']:+.4f} (1-sided UB {fslope['one_sided_ci95']:+.4f}, "
      f"p2={fslope['p_value_two_sided']:.3f}); citation slope {cslope['slope']:+.4f}; n_queries={E['n_queries']}")
