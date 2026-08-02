#!/usr/bin/env python
"""Paper-5 E5 dose-response figure (the dose ceiling).

All numbers from canonical_numbers.json['e5_dose_response'] and ['e5_equivalence']
(single source of truth; nothing hardcoded).

Two panels:
  (a) Outcome: pooled (P0/P1/P4) per-query factual-accuracy and citation-quality
      means versus injected gold fraction g000->g100, GPT-5.2. Factual accuracy is
      FLAT; citation quality rises modestly. The interleaved cell is shown off-axis
      as the context-overload rescue contrast.
  (b) Equivalence: the factual-flat TOST. Per-query g100-g000 mean difference with
      its 90% CI inside the +/-0.05 margin (anchored to ~2x the E2 MDE80=0.0247),
      i.e. a positive equivalence result, not merely a CI straddling zero.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
C = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
E5 = C["e5_dose_response"]
EQ = C["e5_equivalence"]

pf = E5["per_fraction_means"]["pooled"]
order = ["g000", "g025", "g050", "g075", "g100"]
gf = [pf[c]["gold_fraction"] for c in order]
fac = [pf[c]["factual_accuracy_mean"] for c in order]
cit = [pf[c]["citation_quality_mean"] for c in order]
il = pf["interleaved"]
fslope = E5["factual_accuracy_slope"]
cslope = E5["citation_quality_slope"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})

# ---- Panel A: dose-response curves ----
axA.plot(gf, fac, "-o", color="#c0504d", lw=1.7, ms=6, zorder=3,
         label="Factual accuracy")
axA.plot(gf, cit, "-s", color="#1f4e79", lw=1.7, ms=6, zorder=3,
         label="Citation quality")
# interleaved cell drawn off to the right of g100 as a contrast marker
xi = 1.18
axA.plot([xi], [il["factual_accuracy_mean"]], "o", color="#c0504d", ms=6,
         mfc="white", mew=1.4, zorder=3)
axA.plot([xi], [il["citation_quality_mean"]], "s", color="#1f4e79", ms=6,
         mfc="white", mew=1.4, zorder=3)
axA.axvspan(1.06, 1.30, color="#eeeeee", zorder=0)
axA.text(xi, 0.205, "inter-\nleaved", ha="center", va="top", fontsize=7.6, color="#555")
axA.set_xticks(gf + [xi])
axA.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00", "prog."])
axA.set_xlabel("Injected gold fraction (oracle dose)")
axA.set_ylabel("Score (per-query mean, GPT-5.2)")
axA.set_ylim(0.20, 0.34)
axA.set_xlim(-0.06, 1.32)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axA.legend(loc="upper left", fontsize=8.4, frameon=False)
axA.set_title("(a) Facts flat, citations creep with the dose", fontsize=10.5, loc="left")
# slope annotations
axA.text(0.02, 0.225,
         f"factual slope {fslope['slope']:+.3f}\n(1-sided 95\\% UB {fslope['one_sided_ci95']:+.3f})",
         fontsize=7.8, color="#c0504d")
axA.text(0.40, 0.305,
         f"citation slope {cslope['slope']:+.3f}\n(1-sided 95\\% LB {cslope['one_sided_ci95']:+.3f})",
         fontsize=7.8, color="#1f4e79")

# ---- Panel B: factual-flat TOST ----
ff = EQ["factual_flat"]
tost = ff["tost"]
bound = tost["bound"]
md = tost["mean_diff"]
ci90 = tost["ci90_inside_bound"]
bs = ff["paired_bootstrap"]["ci95"]
anchor = EQ["e2_mde80_anchor"]
# margin band
axB.axvspan(-bound, bound, color="#dfe8f0", zorder=0, label=f"$\\pm${bound:.2f} margin")
axB.axvline(0, color="#888", lw=0.8, zorder=1)
axB.axvline(-anchor, color="#9aa7b5", lw=0.9, ls="--", zorder=1)
axB.axvline(anchor, color="#9aa7b5", lw=0.9, ls="--", zorder=1)
axB.text(anchor, 1.62, f"E2 MDE80\n$\\pm${anchor:.3f}", fontsize=7.2, color="#5a7596",
         ha="center", va="bottom")
# 95% bootstrap CI (lighter, wider) then 90% TOST CI (darker)
y0 = 1.0
axB.plot(bs, [y0 + 0.16, y0 + 0.16], color="#7f9bb5", lw=2.2, zorder=2)
axB.plot(ci90, [y0, y0], color="#1f4e79", lw=3.2, zorder=3,
         solid_capstyle="round")
axB.plot([md], [y0], "o", color="#1f4e79", ms=8, zorder=4, mec="white", mew=1.2)
axB.text(md, y0 - 0.22, f"$\\Delta={md:+.3f}$", ha="center", fontsize=8.4, color="#1f4e79")
axB.text(bs[1] + 0.002, y0 + 0.16, "95\\% boot", fontsize=7.4, va="center", color="#5a7596")
axB.text(ci90[1] + 0.002, y0, "90\\% TOST", fontsize=7.4, va="center", color="#1f4e79")
axB.set_xlim(-bound * 1.35, bound * 1.35)
axB.set_ylim(0.4, 1.95)
axB.set_yticks([])
axB.set_xlabel("Factual accuracy: g100 $-$ g000 (paired)")
axB.spines[["top", "right", "left"]].set_visible(False)
verdict = "equivalent" if tost["equivalent_at_05_alpha"] else "not equivalent"
axB.set_title(f"(b) Full-dose factual gain is null ($p_{{TOST}}={tost['p_tost']:.3f}$, {verdict})",
              fontsize=9.6, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_dose.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_dose; factual slope {fslope['slope']:+.4f} (1s95UB {fslope['one_sided_ci95']:+.4f}); "
      f"citation slope {cslope['slope']:+.4f}; TOST dmean {md:+.4f} CI90 {ci90} p_tost {tost['p_tost']}; "
      f"margin {bound} vs E2 MDE80 {anchor} (x{ff['margin_vs_e2_mde80']})")
