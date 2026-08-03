#!/usr/bin/env python
"""Paper-5 R x U|R conditional-utilisation figure (DeepResearch-Slice decomposition).

All numbers from canonical_numbers.json['oracle']['rxu_conditional'] (single source
of truth). Re-expresses the factual support rate as the product of a retrieval-bound
share mean(R) and a utilisation-bound share mean(U|R), instead of the flat
cited/available ratio it replaces (component_eval.py:286).

Single scatter: each covered pattern placed at (mean_R, mean_U|R). Iso-support
contours (R*U = const) show that patterns at very different (R, U|R) coordinates
land on essentially the same low support rate (~0.27-0.41): the factual ceiling is
a UTILISATION ceiling (U|R ~ 0.32-0.42), not a retrieval one (R ~ 0.83-1.0).
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
RX = C["oracle"]["rxu_conditional"]
pp = RX["per_pattern"]

PRETTY = {"base_p0": "P0", "base_p1": "P1", "base_p4": "P4", "base_p5": "P5",
          "base_p7": "P7", "base_p8": "P8", "base_p9": "P9", "base_p10": "P10",
          "base_p11": "P11"}

fig, ax = plt.subplots(figsize=(6.4, 5.2))

# iso-support contours: R*U|R = const
rr = np.linspace(0.55, 1.005, 200)
for s in (0.25, 0.30, 0.35, 0.40):
    uu = s / rr
    m = uu <= 0.62
    ax.plot(rr[m], uu[m], color="#bbbbbb", lw=0.8, ls=":", zorder=1)
    # label near the top of each contour
    xlab = s / 0.58
    if xlab <= 1.0:
        ax.text(xlab, 0.60, f"{s:.2f}", fontsize=7.0, color="#999", ha="center")
ax.text(0.585, 0.605, "support rate", fontsize=7.4, color="#999", rotation=0)

for key, v in pp.items():
    R = v["mean_R"]; U = v["mean_U_given_R"]; ci = v["u_given_r_ci95"]
    lab = PRETTY.get(key, key.replace("base_", "").upper())
    # colour: retrieval-bound (lower R) vs utilisation-bound (high R, low U|R)
    col = "#c0504d" if R < 0.9 else "#1f4e79"
    ax.errorbar([R], [U], yerr=[[U - ci[0]], [ci[1] - U]], fmt="o", ms=8,
                color=col, ecolor="#999", elinewidth=1.0, capsize=2.5, zorder=3,
                mec="white", mew=1.0)
    dx, dy = 0.006, 0.006
    ax.annotate(lab, (R, U), (R + dx, U + dy), fontsize=8.2, color=col, zorder=4)

# cluster mean support reference
cms = RX["cluster_mean_support_rate"]; cci = RX["cluster_mean_support_rate_ci95"]
ax.set_xlabel(r"$\overline{R}$ — retrieval-bound share (evidence present)")
ax.set_ylabel(r"$\overline{U\mid R}$ — utilisation-bound share (entailed $\mid$ retrieved)")
ax.set_xlim(0.78, 1.02)
ax.set_ylim(0.27, 0.62)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(ls=":", alpha=0.3, zorder=0)
ax.set_title("Factual support is utilisation-bound, not retrieval-bound", fontsize=10.5, loc="left")
ax.text(0.795, 0.585,
        "high $\\overline{R}$, low $\\overline{U|R}$:\nevidence retrieved\nbut not entailed",
        fontsize=7.8, color="#1f4e79", va="top")
ax.text(0.795, 0.44,
        f"cluster support\n{cms:.3f}  [{cci[0]:.3f}, {cci[1]:.3f}]",
        fontsize=7.8, color="#555", va="top",
        bbox=dict(fc="white", ec="#ccc", lw=0.6, pad=2))

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_rxu.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_rxu; {len(pp)} patterns; R range "
      f"[{min(v['mean_R'] for v in pp.values()):.3f},{max(v['mean_R'] for v in pp.values()):.3f}] "
      f"U|R range [{min(v['mean_U_given_R'] for v in pp.values()):.3f},"
      f"{max(v['mean_U_given_R'] for v in pp.values()):.3f}]; cluster support {cms}")
