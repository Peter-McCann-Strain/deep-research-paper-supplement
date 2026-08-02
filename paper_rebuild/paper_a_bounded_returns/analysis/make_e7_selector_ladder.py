#!/usr/bin/env python
"""E7 best-of-N selector ladder + Gate G2 (Paper 4: judge-reward RL fails).

Single source of truth: canonical_numbers.json['e7_selector_kappa'].

Panel (a): the realizable selector ladder. Realised GAIN over the single-run
mean for ORACLE (argmax of true GPT-5.2 score, upper bound), a GPT-5.2-quality
second-pass selector, a GPT-4o-quality selector, and RANDOM (lower bound), with
95% bootstrap CIs. Even the oracle selector only buys +0.052; a realistic
selector buys far less -- so wrapping a judge-as-selector around the policy
cannot substitute for orchestration.

Panel (b): Gate G2. At matched selector reliability (kappa 0.20/0.35/0.50),
STRUCTURED (criteria-correlated) judge error and RANDOM (i.i.d.) judge error
produce indistinguishable selection gains; max |structured - random| = 0.0036 <
0.005 threshold. The "structured vs random" framing the RL plan presupposed is
empirically void before any GPU is spent.

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
e7 = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["e7_selector_kappa"]

lad = e7["selector_ladder"]
single = lad["single_run_mean"]
RUNGS = [("random_lower_bound", "Random pick\n(lower bound)", "#9aa7b5"),
         ("gpt4o_noise", "GPT-4o-quality\nselector", "#7f9bb5"),
         ("gpt52_noise", "GPT-5.2-quality\nselector", "#4a6fa5"),
         ("oracle_upper_bound", "Oracle selector\n(upper bound)", "#1f4e79")]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.2, 1.15]})

# ---- Panel A: realizable selector ladder ----
y = np.arange(len(RUNGS))
gains = [lad[k]["gain"] for k, _, _ in RUNGS]
los = [lad[k]["gain"] - lad[k]["gain_ci95"][0] for k, _, _ in RUNGS]
his = [lad[k]["gain_ci95"][1] - lad[k]["gain"] for k, _, _ in RUNGS]
cols = [c for _, _, c in RUNGS]
axA.barh(y, gains, color=cols, height=0.6, zorder=3)
axA.errorbar(gains, y, xerr=[los, his], fmt="none", ecolor="#333", lw=1.1, capsize=3, zorder=4)
for yi, (k, _, _) in zip(y, RUNGS):
    g = lad[k]["gain"]
    axA.text(g + (0.0015 if g >= 0 else -0.0015), yi, f"{g:+.4f}",
             va="center", ha="left" if g >= 0 else "right", fontsize=8.2)
axA.axvline(0, color="#444", lw=0.8)
axA.set_yticks(y); axA.set_yticklabels([lab for _, lab, _ in RUNGS], fontsize=8.4)
axA.set_xlabel("Realised gain over single-run mean (GPT-5.2)")
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axA.set_title("(a) Best-of-$N$ selection ceiling is small\n(single-run mean $=%.3f$)" % single,
              fontsize=9.8, loc="left")

# ---- Panel B: Gate G2 structured vs random at matched kappa ----
mk = e7["matched_kappa"]
ks = ["kappa0.20", "kappa0.35", "kappa0.50"]
kx = np.arange(len(ks))
w = 0.34
s_gain = [mk[k]["structured_gain"] for k in ks]
r_gain = [mk[k]["random_gain"] for k in ks]
s_lo = [mk[k]["structured_gain"] - mk[k]["structured_gain_ci95"][0] for k in ks]
s_hi = [mk[k]["structured_gain_ci95"][1] - mk[k]["structured_gain"] for k in ks]
r_lo = [mk[k]["random_gain"] - mk[k]["random_gain_ci95"][0] for k in ks]
r_hi = [mk[k]["random_gain_ci95"][1] - mk[k]["random_gain"] for k in ks]
axB.bar(kx - w/2, s_gain, w, yerr=[s_lo, s_hi], capsize=3, color="#c0504d",
        label="Structured (criteria-correlated)", zorder=3, error_kw=dict(lw=1.0))
axB.bar(kx + w/2, r_gain, w, yerr=[r_lo, r_hi], capsize=3, color="#7f9bb5",
        label="Random (i.i.d.)", zorder=3, error_kw=dict(lw=1.0))
axB.set_xticks(kx); axB.set_xticklabels(["$\\kappa{=}0.20$", "$\\kappa{=}0.35$", "$\\kappa{=}0.50$"], fontsize=8.6)
axB.set_ylabel("Selection gain")
axB.legend(loc="upper left", fontsize=7.8, frameon=False)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.35, zorder=0)
g2 = e7["gate_g2"]
axB.set_title("(b) Gate G2: structured $\\approx$ random\nmax$|\\Delta|=%.4f<%.3f$ (equivalent)"
              % (g2["max_abs_structured_minus_random"], g2["equivalence_threshold"]),
              fontsize=9.0, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_e7_selector_ladder.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_e7_selector_ladder; oracle gain={lad['oracle_upper_bound']['gain']:+.4f}, "
      f"gpt52 selector={lad['gpt52_noise']['gain']:+.4f}, random={lad['random_lower_bound']['gain']:+.4f}; "
      f"Gate G2 max|s-r|={g2['max_abs_structured_minus_random']:.4f} fires={g2['gate_fires_b_approx_e']}")
