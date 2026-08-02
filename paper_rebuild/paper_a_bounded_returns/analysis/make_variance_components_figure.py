#!/usr/bin/env python
"""Paper 3 (Randomness / variance) -- variance-component figure.

Two panels, both from canonical_numbers.json['variance_decomposition']:
  (a) 3-way crossed REML variance fractions (run / query / judge / residual) on the
      gpt52+Sonnet balanced intersection. Run variance is the LARGEST single component
      (0.42), so stochastic re-running of the pipeline moves a report's score more than
      either the difficulty of the query or the stringency of the judge.
  (b) Per-architecture run noise: sigma_run with parametric-bootstrap 95% CIs, against
      the MDE80(n=90,r=1)=0.025 run-noise floor. Every architecture's run SD sits at or
      above the smallest gap a 90-query single-replicate design can resolve.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
V = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["variance_decomposition"]

PRETTY = {"base_p0": "P0", "base_p1": "P1", "base_p4": "P4", "base_p5": "P5",
          "base_p6": "P6", "base_p7": "P7", "base_p8": "P8", "base_p10": "P10"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})

# ---- Panel A: 3-way variance fractions (stacked single bar) ----
vf = V["three_way"]["reml_3way"]["var_fraction"]
order = ["run", "query", "judge", "resid"]
labs = {"run": "Run (re-execution)", "query": "Query difficulty",
        "judge": "Judge stringency", "resid": "Residual"}
cols = {"run": "#c0504d", "query": "#1f4e79", "judge": "#e8a33d", "resid": "#9aa7b5"}
bottom = 0.0
for k in order:
    frac = vf[k]
    axA.bar(0, frac, bottom=bottom, color=cols[k], width=0.55, zorder=3,
            edgecolor="white", lw=0.8)
    axA.text(0.36, bottom + frac / 2, f"{labs[k]}\n{frac*100:.1f}\\%",
             va="center", ha="left", fontsize=8.6, color=cols[k])
    bottom += frac
axA.set_xlim(-0.5, 1.7)
axA.set_ylim(0, 1.0)
axA.set_xticks([])
axA.set_ylabel("Share of variance in overall score")
axA.spines[["top", "right", "bottom"]].set_visible(False)
axA.set_title("(a) Re-running dominates the variance budget", fontsize=10.3, loc="left")
axA.annotate("crossed run$\\times$query$\\times$judge REML\n(GPT-5.2 + Sonnet, $n{=}1328$)",
             xy=(0.0, 1.0), xytext=(-0.45, 1.04), fontsize=7.8, color="#444",
             annotation_clip=False)

# ---- Panel B: per-architecture run SD with bootstrap CI ----
bc = V["bootstrap_ci"]["bootstrap_ci"]
arches = [a for a in ["base_p0", "base_p1", "base_p4", "base_p7", "base_p10",
                      "base_p5", "base_p6", "base_p8"] if a in bc]
y = np.arange(len(arches))
mde = V["mde"]["main_study_n90_r1"]   # 0.0247
for i, a in enumerate(arches):
    d = bc[a]
    sd_pt = np.sqrt(d["point"]["sigma2_run"])
    lo = np.sqrt(d["ci"]["sigma2_run"]["lo"])
    hi = np.sqrt(d["ci"]["sigma2_run"]["hi"])
    ragged = d["coverage"] == "ragged"
    col = "#9aa7b5" if ragged else "#1f4e79"
    axB.plot([lo, hi], [i, i], color="#333", lw=1.1, zorder=3)
    axB.plot([lo, lo], [i - 0.13, i + 0.13], color="#333", lw=1.1)
    axB.plot([hi, hi], [i - 0.13, i + 0.13], color="#333", lw=1.1)
    axB.scatter([sd_pt], [i], s=46, color=col, zorder=4, edgecolor="white", lw=0.8)
axB.axvline(mde, color="#c0504d", lw=1.1, ls="--", zorder=2)
axB.text(mde + 0.001, len(arches) - 0.5, f"MDE$_{{80}}$(90,1)\n$={mde:.3f}$",
         color="#c0504d", fontsize=8.0, va="top")
axB.set_yticks(y)
axB.set_yticklabels([PRETTY[a] + (" *" if bc[a]["coverage"] == "ragged" else "") for a in arches])
axB.invert_yaxis()
axB.set_xlabel("Run-to-run SD of overall score ($\\sigma_{\\mathrm{run}}$, GPT-5.2)")
axB.set_xlim(0, max(0.10, max(np.sqrt(bc[a]["ci"]["sigma2_run"]["hi"]) for a in arches) * 1.05))
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axB.set_title("(b) Per-run noise meets the detectable-gap floor", fontsize=10.3, loc="left")
axB.text(0.99, 0.02, "* ragged coverage (wide CI)", transform=axB.transAxes,
         ha="right", va="bottom", fontsize=7.2, color="#777")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_variance_components.{ext}",
                dpi=200, bbox_inches="tight")
print("wrote fig_variance_components; var_fraction run/query/judge/resid = "
      f"{vf['run']:.3f}/{vf['query']:.3f}/{vf['judge']:.3f}/{vf['resid']:.3f}; "
      f"MDE80(90,1)={mde:.4f}; pooled sigma_run={V['run_noise']['pooled_sd']}")
