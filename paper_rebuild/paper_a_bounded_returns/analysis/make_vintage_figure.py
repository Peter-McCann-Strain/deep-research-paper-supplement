#!/usr/bin/env python
"""Paper-A frozen-vintage figure — de-confounded vintage + capacity curve.

All numbers from canonical_numbers.json['frozen_vintage'] (single source of truth).
This is the DE-CONFOUNDED 4-arm result (2026-06-30): every arm decodes under one
llama.cpp GGUF greedy backend over the same hash-pinned 89-query frozen source set,
judged by GPT-5.2 (judge-independent of the Qwen-family arms). Two contrast axes:

    vintage_axis  (release DATE, capacity ~constant 7-8B):
        p9  Qwen2.5-7B-Instruct         (2024-09)
        p14 DeepSeek-R1-Distill-Qwen-7B (2025-01)
        p13 Qwen3-8B                     (2025-04)
    capacity_axis (parameters, vintage held at 2024-09):
        p9  Qwen2.5-7B-Instruct  (2024-09)
        p17 Qwen2.5-14B-Instruct (2024-09)

P17 (14B) shares P9's 2024-09 vintage so it is drawn as a SEPARATE capacity point,
not a third date point (two points at x=0 years would be an axis error).

Panel (a): the vintage-date curve — length-adjusted overall score vs release date,
           with raw-mean markers shown for reference.
Panel (b): the capacity contrast at fixed 2024-09 vintage (7B -> 14B), annotated with
           the canonical RAW paired-bootstrap p17-p9 gap (+0.025, CI, p) and the
           length-adjusted point gap (+0.034, point only).

Panels are drawn SIDE-BY-SIDE on a wide canvas so the figure prints short at full
\\linewidth (a tall vertical stack floated ~20 pages past its text reference). Fonts are
bumped throughout so the year/model x-tick labels stay well above the legibility floor
despite the narrower per-panel width.
"""
import json, warnings, sys, os
from datetime import date
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, OKABE_ITO
# print scale s=0.537 (widest 2-panel canvas of any figure in the paper, at full
# linewidth) -> declared 13pt was printing at ~7.0pt, well under the 9pt floor;
# bumped to 17pt (9/0.537 ~= 16.8) to clear it, keeping the wide layout the
# module docstring above explains is deliberate (avoids a ~20-page float delay).
apply_style(base_size=17, legend_fontsize=15)
C_ADJ = OKABE_ITO["blue"]         # length-adjusted
C_RAW = OKABE_ITO["vermillion"]   # raw
ROOT = "."
C = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))
FV = C["frozen_vintage"]
arms = FV["arms"]
n = FV["n_queries"]

# index arms by pattern id for axis lookups
by_pat = {a["pattern"]: a for a in arms.values()}

# short model names for on-panel annotation (full ids stay in the caption)
SHORT = {"p9": "Qwen2.5-7B", "p14": "DS-R1-Distill-7B",
         "p13": "Qwen3-8B", "p17": "Qwen2.5-14B"}


def _to_x(release_date):
    """release_date 'YYYY-MM' -> years since 2024-09 (the P9 anchor date)."""
    y, m = (int(t) for t in release_date.split("-"))
    d = date(y, m, 1)
    base = date(2024, 9, 1)
    return (d - base).days / 365.25


# Side-by-side panels on a wide canvas: prints short at full \linewidth so the figure
# places inline near its text reference instead of floating to the appendix.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---- Panel A: vintage-date curve (length-adjusted, raw markers for reference) ----
vpats = FV["vintage_axis"]["ordered_patterns"]          # ['p9','p14','p13']
xs = [_to_x(by_pat[p]["release_date"]) for p in vpats]
adj = [by_pat[p]["length_adjusted_overall_mean"] for p in vpats]
raw = [by_pat[p]["raw_overall_mean"] for p in vpats]
ci = [by_pat[p]["overall_ci95_paired_bootstrap"] for p in vpats]

# length-adjusted curve (the de-confound-clean series)
axA.plot(xs, adj, "-o", color=C_ADJ, lw=2.0, ms=9, mec="white", mew=0.9, zorder=4,
         label="Length-adjusted overall")
# raw means with paired-bootstrap 95% CI whiskers, offset slightly for legibility
xoff = [x + 0.02 for x in xs]
yerr = np.array([[a - lo for a, (lo, hi) in zip(raw, ci)],
                 [hi - a for a, (lo, hi) in zip(raw, ci)]])
axA.errorbar(xoff, raw, yerr=yerr, fmt="s", color=C_RAW, ms=8, lw=1.4,
             capsize=3.5, mec="white", mew=0.8, zorder=3, label="Raw overall (95% CI)")

# Compact P-label + short model name at each point (release dates sit on the x ticks;
# full model ids are in the caption). Avoids overlap while keeping the tick labels
# well above the legibility floor.
for p, x, ya in zip(vpats, xs, adj):
    a = by_pat[p]
    axA.annotate(f"{a['label']}\n{SHORT[p]}",
                 (x, ya), textcoords="offset points", xytext=(0, 12),
                 fontsize=10, ha="center", color="#333")

axA.set_xlabel("Model release date")
axA.set_ylabel(f"Overall score (GPT-5.2, $n={n}$)")
axA.set_xticks(xs)
axA.set_xticklabels([by_pat[p]["release_date"] for p in vpats])
axA.set_xlim(min(xs) - 0.08, max(xs) + 0.12)
axA.set_ylim(0.18, 0.42)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axA.legend(loc="upper left", fontsize=11, frameon=False)
axA.set_title("(a) Vintage-date axis, frozen GGUF scaffold ($\\sim$7--8B)",
              fontsize=13, loc="left")

# ---- Panel B: capacity contrast at fixed 2024-09 vintage (7B -> 14B) ----
cpats = FV["capacity_axis"]["ordered_patterns"]          # ['p9','p17']
cd = FV["capacity_axis"]["paired_diffs"]["p17_minus_p9"]
raw_gap = cd["point"]
raw_lo, raw_hi = cd["ci95"]
raw_p = cd["p_two_sided_boot"]
adj_gap = round(by_pat["p17"]["length_adjusted_overall_mean"]
                - by_pat["p9"]["length_adjusted_overall_mean"], 3)

xb = np.arange(2)
raw_b = [by_pat[p]["raw_overall_mean"] for p in cpats]
adj_b = [by_pat[p]["length_adjusted_overall_mean"] for p in cpats]
w = 0.34
axB.bar(xb - w / 2, raw_b, width=w, color=C_RAW, label="Raw overall", zorder=3)
axB.bar(xb + w / 2, adj_b, width=w, color=C_ADJ, label="Length-adjusted", zorder=3)
for i, (r, a) in enumerate(zip(raw_b, adj_b)):
    axB.text(xb[i] - w / 2, r + 0.006, f"{r:.3f}", ha="center", fontsize=10.5, color="#7a2b28")
    axB.text(xb[i] + w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=10.5, color="#163a5c")

axB.set_xticks(xb)
axB.set_xticklabels([f"{by_pat[p]['label']}\n{by_pat[p]['model'].split('/')[-1]}"
                     for p in cpats], fontsize=11)
axB.set_xlim(-0.6, 1.6)
axB.set_ylim(0, 0.42)
axB.set_ylabel(f"Overall score (GPT-5.2, $n={n}$)")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", ls=":", alpha=0.35, zorder=0)
axB.legend(loc="upper left", fontsize=11, frameon=False)
axB.set_title("(b) Capacity axis at fixed 2024-09 vintage (7B $\\to$ 14B)",
              fontsize=13, loc="left")
axB.text(0.5, 0.08,
         f"P17$-$P9 raw {raw_gap:+.3f}\nCI [{raw_lo:.3f}, {raw_hi:.3f}], $p={raw_p}$\n"
         f"len-adj {adj_gap:+.3f} (point only)",
         transform=axB.transAxes, ha="center", va="bottom", fontsize=11, color="#333",
         bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#ccc", lw=0.6))

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_vintage.{ext}",
                dpi=300, bbox_inches="tight")
print(f"wrote fig_vintage (frozen_vintage, de-confounded); "
      f"vintage len-adj {[by_pat[p]['length_adjusted_overall_mean'] for p in vpats]}; "
      f"capacity raw gap p17-p9 {raw_gap:+.3f} CI [{raw_lo}, {raw_hi}] p={raw_p}; "
      f"len-adj gap {adj_gap:+.3f}")
