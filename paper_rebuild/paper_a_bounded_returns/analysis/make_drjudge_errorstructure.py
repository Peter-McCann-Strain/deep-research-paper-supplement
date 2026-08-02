#!/usr/bin/env python
"""DR-Judge-7B error-structure figure (Paper 4: judge-reward RL fails).

Single source of truth: canonical_numbers.json['drjudge_error_structure'].

Two panels:
  (a) Per-dimension false-negative vs false-positive rate for the RL-distilled
      DR-Judge-7B against the GPT-5.2-anchored adjudicated gold. The asymmetry
      FNR >> FPR is the headline: the distilled judge mostly WITHHOLDS the
      reward (calls a satisfied criterion unsatisfied), so a policy trained to
      maximise its reward is pushed toward the few dimensions it over-credits,
      not toward genuine quality.
  (b) The same asymmetry at the panel level: overall FPR=0.128 vs FNR=0.369.

No paid API; CPU-only; reads frozen confusion fixture only.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
es = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["drjudge_error_structure"]

DLAB = {"information_recall": "Information recall", "factual_accuracy": "Factual accuracy",
        "coverage": "Coverage", "analytical_depth": "Analytical depth",
        "citation_quality": "Citation quality", "logical_coherence": "Logical coherence",
        "organization": "Organization", "instruction_following": "Instruction following",
        "attribution_quality": "Attribution quality"}

pd = es["per_dimension"]
# order by FNR descending so the withhold-reward dimensions sit at top
rows = sorted(pd.items(), key=lambda kv: kv[1]["fnr"])
labs = [DLAB[k] for k, _ in rows]
fnr = [v["fnr"] for _, v in rows]
fpr = [v["fpr"] for _, v in rows]
n = [v["n"] for _, v in rows]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

# ---- Panel A: per-dimension FNR vs FPR ----
y = np.arange(len(rows))
h = 0.38
axA.barh(y + h/2, fnr, height=h, color="#c0504d", label="False-negative rate (withholds reward)", zorder=3)
axA.barh(y - h/2, fpr, height=h, color="#7f9bb5", label="False-positive rate (over-credits)", zorder=3)
axA.set_yticks(y); axA.set_yticklabels(labs)
axA.set_xlabel("Error rate vs GPT-5.2-anchored gold")
axA.set_xlim(0, 1.0)
axA.axvline(0, color="#444", lw=0.8)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axA.legend(loc="lower right", fontsize=8.2, frameon=False)
axA.set_title("(a) DR-Judge-7B mostly withholds reward (FNR $\\gg$ FPR)",
              fontsize=10.3, loc="left")
# annotate the cleanest illustration: attribution_quality FNR~0.89
ki = [i for i, (k, _) in enumerate(rows) if k == "attribution_quality"]
if ki:
    i = ki[0]
    axA.annotate("withholds reward on\n89% of truly-good\nattribution criteria",
                 xy=(fnr[i], i + h/2), xytext=(0.50, i - 0.1),
                 fontsize=7.8, color="#c0504d",
                 arrowprops=dict(arrowstyle="->", color="#c0504d", lw=0.8))

# ---- Panel B: overall confusion asymmetry ----
c = es["confusion"]
bars = [("FPR\n(over-credit)", c["fpr"], "#7f9bb5"),
        ("FNR\n(withhold)", c["fnr"], "#c0504d"),
        ("Error rate", c["error_rate"], "#9aa7b5")]
xb = np.arange(len(bars))
axB.bar(xb, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62, zorder=3)
for x, (_, val, _) in zip(xb, bars):
    axB.text(x, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8.6)
axB.set_xticks(xb); axB.set_xticklabels([b[0] for b in bars], fontsize=8.6)
axB.set_ylim(0, max(b[1] for b in bars) * 1.28)
axB.set_ylabel("Rate (overall, $n=%s$)" % f"{c['n']:,}")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axB.set_title("(b) Pooled error is asymmetric", fontsize=10.3, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_drjudge_errorstructure.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_drjudge_errorstructure; overall FPR={c['fpr']:.4f} FNR={c['fnr']:.4f} "
      f"(FNR/FPR={c['fnr']/c['fpr']:.2f}x), n={c['n']}")
