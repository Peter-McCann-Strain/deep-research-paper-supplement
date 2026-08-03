#!/usr/bin/env python
"""Model-recall / win-multiplicity decomposition of the oracle gap (Paper 1 -- Routability).

All numbers from canonical_numbers.json['routability']['model_recall'] (read-only).
Frames the routability null in the LLMRouterBench (arXiv:2601.07206) idiom:
Random vs Best-Single vs Oracle routing, plus how concentrated the oracle gap is.

Two panels:
  (a) Routing ladder. Random-routing (Gain@R), best-single (Gain@B), and oracle
      (Gain@O) mean scores on the 30-query replicate corpus. The Oracle-minus-best-single
      gap (Gap@O) is the only headroom a router can chase; it is a thin sliver above
      best-single, and most of the random->oracle distance is already captured by simply
      picking the best single architecture.
  (b) Win concentration. Fraction of queries with a UNIQUE within-eps winner as the
      tolerance eps grows. At the headline eps the winner is unique on ~90% of queries,
      so the per-query best architecture rarely rotates -- the oracle gap is unrecoverable
      and concentrated in a handful of queries.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
import sys
_R = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["routability"]
MR = _R.get("model_recall")
if MR is None:
    # routability.model_recall transiently absent (concurrent canonical rebuild); skip
    # rather than fabricate. Never hardcode the numbers.
    print("SKIP fig_routability_modelrecall: routability.model_recall absent from canonical")
    sys.exit(0)
pri = MR["primary"]
rob = MR["robustness_all8"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1, 1.05]})

# ---- Panel A: routing ladder (primary 5-arch complete-case) ----
names = ["Random\n(Gain@R)", "Best-single\n(Gain@B)", "Oracle\n(Gain@O)"]
vals = [pri["Gain@R_random_mean"], pri["Gain@B_best_single_mean"], pri["Oracle_mean"]]
cols = ["#9aa7b5", "#3f6f9f", "#1f4e79"]
xb = np.arange(3)
axA.bar(xb, vals, 0.62, color=cols, zorder=3)
for i, v in enumerate(vals):
    axA.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.6, color="#222")
# bracket: Gap@O = oracle - best-single (the recoverable headroom)
gap = pri["Gap@O_oracle_minus_best_single"]
ytop = vals[2]; ybot = vals[1]
axA.annotate("", xy=(2.42, ytop), xytext=(2.42, ybot),
             arrowprops=dict(arrowstyle="<->", color="#c0504d", lw=1.2))
axA.text(2.5, (ytop + ybot) / 2, f"Gap@O\n{gap:+.3f}", color="#c0504d", fontsize=8.3,
         va="center", ha="left")
# bracket: random -> best-single (captured for free)
free = pri["Gain@B_best_single_mean"] - pri["Gain@R_random_mean"]
axA.annotate("", xy=(0.5, vals[1]), xytext=(0.5, vals[0]),
             arrowprops=dict(arrowstyle="<->", color="#5a6470", lw=1.0))
axA.text(0.58, (vals[0] + vals[1]) / 2, f"+{free:.3f}\nrandom$\\to$best-fixed", color="#5a6470",
         fontsize=7.8, va="center", ha="left")
axA.set_xticks(xb); axA.set_xticklabels(names, fontsize=8.8)
axA.set_ylabel("Mean overall score (GPT-5.2, replicate corpus)")
axA.set_ylim(0.40, 0.58)
axA.set_xlim(-0.6, 3.2)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axA.set_title(f"(a) Routing ladder ($n{{=}}{pri['n_queries']}$, {pri['n_archs']} archs)",
              fontsize=10.3, loc="left")

# ---- Panel B: single-winner fraction vs eps (both configurations) ----
def curve(d):
    xs = sorted(float(k) for k in d["single_winner_fraction_by_eps"])
    ys = [d["single_winner_fraction_by_eps"][f"{x}".rstrip("0").rstrip(".") if False else _key(d, x)] for x in xs]
    return xs, ys
def _key(d, x):
    # match the json key formatting (e.g. 0.0, 0.005, 0.01, 0.02)
    for k in d["single_winner_fraction_by_eps"]:
        if abs(float(k) - x) < 1e-9:
            return k
    raise KeyError(x)

for d, lab, col, mk in [(pri, f"{pri['n_archs']}-arch (primary)", "#1f4e79", "o"),
                        (rob, f"{rob['n_archs']}-arch (robustness)", "#c0504d", "s")]:
    xs = sorted(float(k) for k in d["single_winner_fraction_by_eps"])
    ys = [d["single_winner_fraction_by_eps"][_key(d, x)] for x in xs]
    axB.plot(xs, ys, "-", color=col, lw=1.6, marker=mk, ms=6, label=lab, zorder=3)
eps_h = pri["eps_headline"]; swf_h = pri["single_winner_fraction_headline"]
axB.axvline(eps_h, color="#888", lw=0.9, ls=":", zorder=1)
axB.scatter([eps_h], [swf_h], s=120, facecolors="none", edgecolors="#1f4e79", lw=1.6, zorder=4)
axB.annotate(f"headline $\\epsilon{{=}}{eps_h}$:\n{swf_h:.0%} unique winner",
             xy=(eps_h, swf_h), xytext=(eps_h - 0.013, swf_h - 0.18), fontsize=8.2,
             color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8))
axB.set_xlabel(r"Within-$\epsilon$ tie tolerance")
axB.set_ylabel("Fraction of queries with a unique winner")
axB.set_ylim(0.45, 1.03)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(ls=":", alpha=0.35, zorder=0)
axB.legend(loc="lower left", fontsize=8.2, frameon=False)
axB.set_title("(b) Per-query winner rarely rotates", fontsize=10.3, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_routability_modelrecall.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_routability_modelrecall; Gap@O={gap:+.4f}, single-winner@eps{eps_h}={swf_h:.3f}, "
      f"best-single={pri['best_single_arch']}")
