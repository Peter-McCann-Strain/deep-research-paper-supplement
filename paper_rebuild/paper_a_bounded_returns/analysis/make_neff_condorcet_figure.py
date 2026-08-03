#!/usr/bin/env python
"""N_eff / Condorcet-null figure (why three correlated judges are ~1.65 effective votes).

Reads canonical_numbers.json['n_eff']['diagnostics']. The observed 3-judge panel reaches
unanimity far more often than three INDEPENDENT judges with the same marginal SATISFIED
rates would, on the SAME crossed cells. That correlation-driven excess unanimity is exactly
why N_eff (~1.65) sits far below k=3 and N_eff/k hugs the saturation line: a 4th correlated
LLM judge would add far less than one independent vote.

  (a) Unanimity and panel self-agreement: observed vs Condorcet-independent null.
  (b) N_eff vs k=3, with the N_eff/k saturation line (arXiv:2605.29800).

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
DG = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["n_eff"]["diagnostics"]
C = DG["condorcet_null"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})

# ---- Panel A: observed vs independent null ----
cats = ["unanimity", "panel self-\nagreement"]
obs = [C["unanimity_observed"], C["panel_self_agreement_observed"]]
nul = [C["unanimity_condorcet_null"], C["panel_self_agreement_condorcet_null"]]
x = np.arange(len(cats)); w = 0.36
axA.bar(x - w / 2, obs, width=w, color="#1f4e79", label="observed (correlated panel)", zorder=3)
axA.bar(x + w / 2, nul, width=w, color="#c0a060", label="Condorcet-independent null", zorder=3)
for xi, o, n in zip(x, obs, nul):
    axA.text(xi - w / 2, o + 0.01, f"{o:.3f}", ha="center", va="bottom", fontsize=8.0)
    axA.text(xi + w / 2, n + 0.01, f"{n:.3f}", ha="center", va="bottom", fontsize=8.0)
    axA.annotate("", xy=(xi, o), xytext=(xi, n),
                 arrowprops=dict(arrowstyle="<->", color="#333", lw=0.9))
axA.text(0, (obs[0] + nul[0]) / 2, f"  +{C['excess_unanimity']:.3f}\n  excess", fontsize=8.2,
         va="center", ha="left", color="#c0504d")
axA.set_xticks(x); axA.set_xticklabels(cats)
axA.set_ylabel(f"Agreement rate (same {C['n_cells']:,} crossed cells)")
axA.set_ylim(0, 1.0)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axA.legend(loc="upper right", frameon=False, fontsize=8.4)
axA.set_title("(a) Correlated judges over-agree vs independence", fontsize=10.0, loc="left")

# ---- Panel B: N_eff vs k ----
k = DG["k"]; neff = DG["n_eff_overall"]; ratio = DG["n_eff_over_k"]; thr = DG["neff_k_caution_threshold"]
axB.bar([0], [k], width=0.5, color="#7f9bb5", zorder=3)
axB.bar([1], [neff], width=0.5, color="#1f4e79", zorder=3)
axB.text(0, k + 0.05, f"$k$ = {k}", ha="center", va="bottom", fontsize=9)
axB.text(1, neff + 0.05, f"$N_{{\\mathrm{{eff}}}}$ = {neff:.2f}", ha="center", va="bottom", fontsize=9)
# saturation line as N_eff value = thr*k
axB.axhline(thr * k, color="#c0504d", ls="--", lw=1.2, zorder=2)
axB.text(1.35, thr * k, f"saturation line\n$N_{{\\mathrm{{eff}}}}/k={thr}$", fontsize=7.8,
         color="#c0504d", va="center", ha="left")
axB.set_xticks([0, 1]); axB.set_xticklabels(["nominal\njudges", "effective\nvotes"])
axB.set_xlim(-0.6, 2.2)
axB.set_ylim(0, k + 0.6)
axB.set_ylabel("number of votes")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axB.set_title(f"(b) $N_{{\\mathrm{{eff}}}}/k$ = {ratio:.3f}", fontsize=10.0, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_neff_condorcet.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_neff_condorcet; unanimity obs {C['unanimity_observed']:.4f} vs null "
      f"{C['unanimity_condorcet_null']:.4f} (excess {C['excess_unanimity']:.4f}); "
      f"N_eff={neff:.4f}/k={k} = {ratio:.4f}")
