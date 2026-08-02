#!/usr/bin/env python
"""E5 gold-dose response figure (Paper 4: judge-reward RL fails -- the reward is flat).

Single source of truth: canonical_numbers.json['e5_dose_response'],
['e5_equivalence'], ['e5_gold_consumption'].

Mechanism behind why an RL policy cannot climb the judge reward on facts:
the judged factual-accuracy score is FLAT as injected gold rises g000->g100,
even though the reports demonstrably ingest essentially ALL the gold
(gold_cited tracks gold_available ~1:1). Citation is not the bottleneck;
synthesis is, and the judge's factual signal carries no usable gradient.

Panel (a): pooled per-fraction judged means for factual_accuracy and
citation_quality vs gold fraction, with the fitted factual slope and its
+/-0.05 TOST equivalence band (E5 asserts FLAT).
Panel (b): gold consumed (gold_cited / gold_available, pooled P0/P1/P4) vs
gold fraction -- ~1:1, so the flat reward is not a retrieval failure.

CPU-only; no API.
"""
import json, warnings, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s=0.658 (full linewidth) -> the prior declared 10pt base printed at
# ~6.6pt, under the 9pt floor; bumped to 13.7 (9/0.658) and every hardcoded
# inline fontsize below scaled by the same 1.37x ratio. Migrated off the
# MS-Office-2007-theme ad hoc colours onto the shared Okabe-Ito palette.
apply_style(base_size=13.7, axes_linewidth=0.8)
C_FACTUAL = OKABE_ITO["vermillion"]
C_CITATION = OKABE_ITO["blue"]
C_CONSUMED = OKABE_ITO["bluish_green"]
ROOT = "."
R = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
dr = R["e5_dose_response"]
eq = R["e5_equivalence"]
gc = R["e5_gold_consumption"]

# --- pooled per-fraction means ---
pf = dr["per_fraction_means"]["pooled"]
cells = ["g000", "g025", "g050", "g075", "g100"]
gx = [dr["gold_fractions"][c] for c in cells]
fac = [pf[c]["factual_accuracy_mean"] for c in cells]
cit = [pf[c]["citation_quality_mean"] for c in cells]

fac_slope = dr["factual_accuracy_slope"]["slope"]
cit_slope = dr["citation_quality_slope"]["slope"]
fac_tost = eq["factual_flat"]["tost"]
margin = eq["factual_flat"]["margin"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.3),
                                gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32})

# ---- Panel A: dose-response ----
axA.plot(gx, fac, "o-", color=C_FACTUAL, lw=1.6, ms=6, label="Factual accuracy (judged)", zorder=4)
axA.plot(gx, cit, "s-", color=C_CITATION, lw=1.6, ms=6, label="Citation quality (judged)", zorder=4)
# fitted factual slope line through the g000 intercept + equivalence band
base = fac[0]
xs = np.array([0, 1.0])
axA.plot(xs, base + fac_slope * xs, "--", color=C_FACTUAL, lw=1.0, alpha=0.8, zorder=3)
axA.fill_between(xs, base - margin / 2, base + margin / 2, color=C_FACTUAL, alpha=0.08, zorder=1,
                 label="$\\pm0.05$ TOST equivalence band")
axA.set_xlabel("Injected gold fraction")
axA.set_ylabel("Judged score (GPT-5.2, pooled P0/P1/P4)")
axA.set_xticks(gx)
axA.set_ylim(0.18, 0.36)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(ls=":", alpha=0.35, zorder=0)
axA.legend(loc="upper left", fontsize=10.4, frameon=False)
slope_ci = dr["factual_accuracy_slope"]["ci95_two_sided"]
slope_p = dr["factual_accuracy_slope"]["p_value_two_sided"]
axA.set_title("(a) Factual reward is flat across the gold dose\n"
              "slope $%+.3f$, 95%% CI [$%.3f$, $%.3f$], $p=%.2f$"
              % (fac_slope, slope_ci[0], slope_ci[1], slope_p), fontsize=12.7, loc="left")

# ---- Panel B: gold consumption ----
# pooled gold_cited / gold_available across P0/P1/P4 per dose cell
pc = gc["per_cell"]
arch = gc["architectures"]
frac_consumed = []
for c in cells:
    avail = sum(pc[a][c]["gold_available"] for a in arch)
    cited = sum(pc[a][c]["gold_cited"] for a in arch)
    frac_consumed.append(cited / avail if avail else np.nan)
gxc = [gc["gold_fractions"][c] for c in cells]
# y is the ratio gold_cited/gold_available, so 1:1 tracking (cited = available) is the
# horizontal ceiling at 1.0, NOT the diagonal y=x (which is meaningless on these axes).
axB.axhline(1.0, ls=":", color="#999", lw=1.4, zorder=2,
            label="1:1 ceiling (all available gold cited)")
axB.plot(gxc, frac_consumed, "o-", color=C_CONSUMED, lw=1.7, ms=7, zorder=4,
         label="gold cited / available")
for x, y in zip(gxc, frac_consumed):
    if not np.isnan(y):
        axB.text(x, y - 0.06, f"{y:.2f}", ha="center", fontsize=10.4, color=C_CONSUMED)
axB.set_xlabel("Injected gold fraction")
axB.set_ylabel("Fraction of available gold cited")
axB.set_xticks(gxc)
axB.set_ylim(-0.02, 1.08)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(ls=":", alpha=0.35, zorder=0)
axB.legend(loc="lower right", fontsize=10.7, frameon=False)
axB.set_title("(b) Reports ingest nearly all the gold\n(retrieval is not the bottleneck)",
              fontsize=12.7, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_e5_dose_response.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_e5_dose_response; factual slope={fac_slope:+.4f} (TOST p={fac_tost['p_tost']:.3f}, "
      f"equiv={fac_tost['equivalent_at_05_alpha']}); citation slope={cit_slope:+.4f}; "
      f"gold consumed g100={frac_consumed[-1]:.2f}")
