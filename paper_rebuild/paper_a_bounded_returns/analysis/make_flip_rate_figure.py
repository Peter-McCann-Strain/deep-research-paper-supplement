#!/usr/bin/env python
"""Paper 3 (Randomness / variance) -- per-dimension replicate flip rate.

From canonical_numbers.json['variance_decomposition']['flip_rates']: the mean
criterion-level binary disagreement between independent re-runs of the same
(architecture, query), per rubric dimension. This is the binary process that
GENERATES run noise. Subjective dimensions (analytical depth, logical coherence,
citation quality) flip ~10% of the time across re-runs; structural ones
(organization, attribution) are nearly run-stable. The macro-average over
dimensions is the headline 0.073 flip rate.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
FR = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["variance_decomposition"]["flip_rates"]

DLAB = {"citation_quality": "Citation quality", "information_recall": "Information recall",
        "instruction_following": "Instruction following", "coverage": "Coverage",
        "analytical_depth": "Analytical depth", "organization": "Organization",
        "attribution_quality": "Attribution quality", "logical_coherence": "Logical coherence",
        "factual_accuracy": "Factual accuracy"}
pd_ = FR["per_dimension"]
rows = sorted([(k, v["mean_disagreement"], v["n_cells"]) for k, v in pd_.items()],
              key=lambda r: r[1])
macro = FR["overall_macro_over_dims"]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
y = np.arange(len(rows))
for i, (k, rate, n) in enumerate(rows):
    col = "#c0504d" if rate >= 0.09 else ("#1f4e79" if rate >= 0.06 else "#9aa7b5")
    ax.barh(i, rate, color=col, height=0.66, zorder=3)
    ax.text(rate + 0.0015, i, f"{rate:.3f}", va="center", ha="left", fontsize=8.0, color="#333")
ax.axvline(macro, color="#444", lw=1.2, ls="--", zorder=4)
ax.text(macro + 0.0015, len(rows) - 0.55, f"macro-avg {macro:.3f}",
        color="#444", fontsize=8.4, va="top")
ax.set_yticks(y)
ax.set_yticklabels([f"{DLAB[k]}  ($n{{=}}{n}$)" for k, _, n in rows], fontsize=8.6)
ax.set_xlabel("Mean criterion-level disagreement between re-runs (GPT-5.2)")
ax.set_xlim(0, max(r for _, r, _ in rows) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", ls=":", alpha=0.35, zorder=0)
ax.set_title("Subjective dimensions flip across re-runs; structural ones do not",
             fontsize=10.3, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_flip_rate.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_flip_rate; macro={macro:.4f}; "
      f"max={rows[-1][0]} {rows[-1][1]:.4f}; min={rows[0][0]} {rows[0][1]:.4f}")
