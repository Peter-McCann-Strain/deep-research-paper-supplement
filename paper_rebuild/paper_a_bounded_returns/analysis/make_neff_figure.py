#!/usr/bin/env python
"""Paper 2 (judge science) -- effective-judge-count figure.

Two panels, both from canonical_numbers.json['n_eff'] (single source of truth):
  (A) Per-dimension within-family (Opus-Sonnet) minus cross-family (GPT-5.2-Claude)
      verdict-agreement phi. The artefact signature: dimensions where the two same-lab
      Claude judges agree FAR more than they agree with the cross-family judge are where
      a two-Claude majority would manufacture spurious reliability. Sorted descending.
  (B) Saturation: N_eff vs k for the observed 3-judge panel and the within-OpenAI /
      within-Anthropic 2-judge control cells, against the arXiv:2605.29800 N_eff/k = 0.5
      caution line. The panel sits just above the line: a 4th correlated LLM judge would
      add far less than one independent vote.
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
ne = C["n_eff"]

DLAB = {"citation_quality": "Citation quality", "information_recall": "Information recall",
        "instruction_following": "Instruction following", "coverage": "Coverage",
        "analytical_depth": "Analytical depth", "organization": "Organization",
        "attribution_quality": "Attribution quality", "logical_coherence": "Logical coherence",
        "factual_accuracy": "Factual accuracy"}

# ---- Panel A: per-dimension within-minus-cross phi ----
pd_ = ne["per_dimension"]
rows = sorted([(k, v["within_minus_cross"],
                v["within_family_phi_opus_sonnet"], v["cross_family_phi_gpt52_claude"])
               for k, v in pd_.items()], key=lambda r: r[1])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

y = np.arange(len(rows))
for i, (k, gap, win, cross) in enumerate(rows):
    # flag the rubric dimensions of substantive interest (citation/attribution) in accent
    col = "#c0504d" if k in ("citation_quality", "attribution_quality") else "#1f4e79"
    axA.barh(i, gap, color=col, height=0.66, zorder=3)
axA.axvline(0, color="#444", lw=0.8)
axA.set_yticks(y); axA.set_yticklabels([DLAB[k] for k, *_ in rows])
axA.set_xlabel(r"Within-family $-$ cross-family agreement  ($\phi_{\mathrm{Opus,Son}} - \phi_{\mathrm{GPT,Claude}}$)")
axA.set_xlim(0, 0.30)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="x", ls=":", alpha=0.35, zorder=0)
axA.set_title("(a) Same-lab redundancy is largest on subjective dimensions",
              fontsize=10.0, loc="left")
# annotate the leading two and citation_quality
top_k = rows[-1][0]
icit = [i for i, (k, *_) in enumerate(rows) if k == "citation_quality"][0]
axA.annotate("rubric artefact carrier", xy=(rows[icit][1], icit),
             xytext=(rows[icit][1] + 0.005, icit - 1.6), fontsize=8.2, color="#c0504d",
             arrowprops=dict(arrowstyle="->", color="#c0504d", lw=0.8))

# ---- Panel B: N_eff/k saturation ----
diag = ne.get("diagnostics")
wo = ne.get("within_openai")
if diag is None or wo is None:
    raise SystemExit("make_neff_figure: n_eff.diagnostics / n_eff.within_openai not yet in "
                     "canonical (mid-rebuild). Re-run after rebuild_all.sh completes.")
# points: (label, k, N_eff)
pts = [
    ("3-judge panel\n(GPT-5.2+Opus+Sonnet)", 3, ne["overall"]["n_eff"], "#1f4e79"),
    ("within-OpenAI\n(GPT-5.2+GPT-4.1)", 2, wo["within_openai"]["n_eff"], "#2e7d32"),
    ("within-Anthropic\n(Opus+Sonnet)", 2, wo["within_anthropic"]["n_eff"], "#8e44ad"),
    ("4-judge grid\n(2 OpenAI+2 Anthropic)", 4, wo["full_grid"]["n_eff"], "#b8860b"),
]
# caution region N_eff/k < 0.5
kk = np.linspace(1.6, 4.4, 50)
axB.fill_between(kk, 0, 0.5 * kk, color="#f2dede", alpha=0.7, zorder=0,
                 label="caution: $N_{\\mathrm{eff}}/k<0.5$ (arXiv:2605.29800)")
axB.plot(kk, kk, ls="--", color="#888", lw=1.0, zorder=1, label="ideal: $N_{\\mathrm{eff}}=k$")
axB.plot(kk, 0.5 * kk, ls=":", color="#c0504d", lw=1.1, zorder=1)
for lab, k, neff, col in pts:
    axB.scatter([k], [neff], s=95, color=col, zorder=4, edgecolor="white", lw=1.0)
    dy = 0.10 if k != 3 else 0.12
    ha = "left" if k < 4 else "right"
    xoff = 0.07 if k < 4 else -0.07
    axB.annotate(lab, xy=(k, neff), xytext=(k + xoff, neff + dy),
                 fontsize=7.6, color=col, ha=ha, va="bottom")
axB.set_xlabel("Number of judges $k$")
axB.set_ylabel("Effective judge count $N_{\\mathrm{eff}}$")
axB.set_xlim(1.6, 4.4)
axB.set_ylim(1.0, 4.3)
axB.set_xticks([2, 3, 4])
axB.spines[["top", "right"]].set_visible(False)
axB.grid(ls=":", alpha=0.3, zorder=0)
axB.legend(fontsize=7.2, loc="upper left", frameon=False)
axB.set_title(f"(b) $N_{{\\mathrm{{eff}}}}/k={diag['n_eff_over_k']:.2f}$: a 4th correlated judge buys almost nothing",
              fontsize=9.4, loc="left")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_neff.{ext}",
                dpi=200, bbox_inches="tight")
print(f"wrote fig_neff; overall N_eff={ne['overall']['n_eff']:.3f} (k=3, N_eff/k={diag['n_eff_over_k']:.3f}); "
      f"top within-minus-cross dim={rows[-1][0]} ({rows[-1][1]:+.3f}); "
      f"citation within-minus-cross={pd_['citation_quality']['within_minus_cross']:+.3f}")
