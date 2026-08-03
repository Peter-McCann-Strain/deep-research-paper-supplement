#!/usr/bin/env python
"""DR-Judge-7B Rate-or-Fate Youden-J phase map (Paper 4: judge-reward RL fails).

Single source of truth: canonical_numbers.json['drjudge_youden_j'].

Signed Youden's J (J = TPR - FPR, prevalence-free informedness) for the
RL-distilled DR-Judge-7B and every panel judge, against the SAME adjudicated
gold, placed on the arXiv:2601.04411 "Rate or Fate" phase map:
    J > +eps  = usable reward gradient (Rate)
    |J| <= eps = no gradient
    J < -eps  = anti-informative (Fate; RL would degrade the policy).

Panel (a): heatmap of signed J, judge x dimension, with the phase epsilon band.
Panel (b): overall signed J per judge with the bootstrap gap DR-Judge minus the
best panel judge (Opus), which excludes 0 -- the distilled judge is a usable but
strictly weaker reward model than the panel it was meant to replace.

CPU-only; no API.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
yj = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["drjudge_youden_j"]
eps = yj["j_zero_epsilon"]

DLAB = {"information_recall": "Info recall", "factual_accuracy": "Factual acc.",
        "coverage": "Coverage", "analytical_depth": "Anal. depth",
        "citation_quality": "Citation qual.", "logical_coherence": "Log. coherence",
        "organization": "Organization", "instruction_following": "Instr. follow",
        "attribution_quality": "Attribution"}
JLAB = {"DR-Judge-7B": "DR-Judge-7B\n(RL distilled)", "claude_opus": "Opus\n(panel)",
        "claude_sonnet": "Sonnet\n(panel)", "gpt52": "GPT-5.2\n(panel)"}
DIM_ORDER = ["factual_accuracy", "citation_quality", "coverage", "analytical_depth",
             "information_recall", "instruction_following", "logical_coherence",
             "organization", "attribution_quality"]
JUDGE_ORDER = ["DR-Judge-7B", "claude_opus", "claude_sonnet", "gpt52"]

# matrix J[judge, dim]
M = np.full((len(JUDGE_ORDER), len(DIM_ORDER)), np.nan)
for ji, jn in enumerate(JUDGE_ORDER):
    perd = yj["judges"][jn]["per_dimension"]
    for di, dn in enumerate(DIM_ORDER):
        M[ji, di] = perd[dn]["youden_j_signed"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [2.05, 1]})

# ---- Panel A: signed-J heatmap ----
norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=1.0)
im = axA.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
axA.set_xticks(range(len(DIM_ORDER)))
axA.set_xticklabels([DLAB[d] for d in DIM_ORDER], rotation=40, ha="right", fontsize=8.4)
axA.set_yticks(range(len(JUDGE_ORDER)))
axA.set_yticklabels([JLAB[j] for j in JUDGE_ORDER], fontsize=8.6)
for ji in range(len(JUDGE_ORDER)):
    for di in range(len(DIM_ORDER)):
        v = M[ji, di]
        # phase glyph
        phase = "Rate" if v > eps else ("Fate" if v < -eps else "0")
        txt = f"{v:.2f}"
        axA.text(di, ji, txt, ha="center", va="center", fontsize=7.3,
                 color="white" if v > 0.55 or v < -0.05 else "#222")
axA.set_title("(a) Signed Youden's $J$ = TPR $-$ FPR: every cell is in the \"Rate\" phase ($J>%.2f$)"
              % eps, fontsize=9.8, loc="left")
cb = fig.colorbar(im, ax=axA, fraction=0.030, pad=0.02)
cb.set_label("signed $J$", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)

# ---- Panel B: overall signed-J per judge + gap ----
ov = {jn: yj["judges"][jn]["overall"]["youden_j_signed"] for jn in JUDGE_ORDER}
order = sorted(JUDGE_ORDER, key=lambda j: ov[j])
cols = ["#c0504d" if j == "DR-Judge-7B" else "#7f9bb5" for j in order]
yb = np.arange(len(order))
axB.barh(yb, [ov[j] for j in order], color=cols, height=0.6, zorder=3)
for y, j in zip(yb, order):
    axB.text(ov[j] + 0.012, y, f"{ov[j]:.3f}", va="center", fontsize=8.4)
axB.axvline(eps, color="#888", ls="--", lw=0.9)
axB.text(eps + 0.005, -0.7, "Rate $\\to$", fontsize=7.6, color="#666")
axB.set_yticks(yb); axB.set_yticklabels([JLAB[j].replace("\n", " ") for j in order], fontsize=7.8)
axB.set_xlabel("Overall signed $J$")
axB.set_xlim(0, 1.0)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="x", ls=":", alpha=0.35, zorder=0)
gap = yj["gap_bootstrap_drjudge_minus_best_panel"]
axB.set_title("(b) Usable but weaker:\nDR-Judge $-$ Opus $J=%.3f$\n95%% CI [%.3f, %.3f] (excl. 0)"
              % (gap["obs_gap_overall_J"], gap["ci95"][0], gap["ci95"][1]),
              fontsize=8.6, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_drjudge_youden_phasemap.{ext}",
                dpi=200, bbox_inches="tight")
counts = yj["rate_or_fate_phase_map"]["counts"]
print(f"wrote fig_drjudge_youden_phasemap; phase counts rate={counts['rate']} "
      f"fate_boundary={counts['fate_boundary']} fate={counts['fate']}; "
      f"DR-Judge overall J={ov['DR-Judge-7B']:.4f}, gap vs Opus={gap['obs_gap_overall_J']:.4f}")
