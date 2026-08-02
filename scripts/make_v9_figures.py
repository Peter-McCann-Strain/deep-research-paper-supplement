"""Generate the eight new V9 figures for paper_v9.md.

Outputs go to reports/phase3_figures/v9/.

Figures produced:
  fig4_ablations_v9.{pdf,png}        - regenerated ablation cascade
  fig9_protocol_a.{pdf,png}          - Bing vs Tavily + Sonnet probe
  fig10_sigma_run.{pdf,png}          - sigma(run) per pattern + 3-rerun spread
  fig11_dr_judge.{pdf,png}           - DR-Judge per-pattern + per-dim kappa
  fig12_c0.{pdf,png}                 - C0 verdict distribution + per-pattern band
  fig13_p12_training.{pdf,png}       - P12 GRPO training curve
  fig14_trajectory.{pdf,png}         - process trajectory pattern-level scatter
  fig15_placeholder_filter.{pdf,png} - placeholder-filtered re-rank survival
"""
from __future__ import annotations
import json
import re
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "phase3_figures" / "v9"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PATTERN_COLORS = {
    "P0": "#7f7f7f", "P1": "#1f77b4", "P2": "#2ca02c", "P3": "#9467bd",
    "P4": "#d62728", "P5": "#8c564b", "P6": "#17becf", "P7": "#e377c2",
    "P8": "#bcbd22", "P9": "#ff7f0e", "P10": "#ff9896", "P11": "#aec7e8",
    "P12": "#c49c94",
}

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{pdf,png}}")


# ============================================================
# Fig 4 (regen): Ablation cascade — V9 corrected numbers
# ============================================================
def fig4_ablations_v9():
    """Regenerated to match the V9 §7 corrected ablation table."""
    rows = [
        # (name, base, delta, ci_lo, ci_hi, cliff_d, p_holm, sig)
        ("ablation_p3_no_quality_eval",   "P3", -0.007, -0.021, +0.008, -0.053, 0.465, "ns"),
        ("ablation_p3_no_topic_mining",   "P3", -0.002, -0.018, +0.015, -0.027, 0.465, "ns"),
        ("ablation_p4_fixed_perspectives","P4", -0.030, -0.044, -0.016, -0.235, 1.45e-4, "*"),
        ("ablation_p4_no_conversations",  "P4", -0.037, -0.053, -0.020, -0.236, 2.5e-5, "*"),
        ("ablation_p4_no_triangulation",  "P4", -0.060, -0.080, -0.042, -0.383, 2.15e-9, "*"),
        ("ablation_p5_fixed_width",       "P5", -0.024, -0.044, -0.004, -0.174, 0.0215, "*"),
        ("ablation_p5_no_meta_eval",      "P5", -0.047, -0.064, -0.029, -0.285, 5.94e-6, "*"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.1, 1.05]})
    titles = {"P3": "P3 ablations", "P4": "P4 ablations", "P5": "P5 ablations"}
    for ax, base in zip(axes, ["P3", "P4", "P5"]):
        sub = [r for r in rows if r[1] == base]
        ys = list(range(len(sub)))
        deltas = [r[2] for r in sub]
        los = [r[3] for r in sub]
        his = [r[4] for r in sub]
        cliffs = [r[5] for r in sub]
        sigs = [r[7] for r in sub]
        labels = [
            r[0]
            .replace(f"ablation_{base.lower()}_", "")
            .replace("quality_eval", "quality\neval")
            .replace("topic_mining", "topic\nmining")
            .replace("fixed_perspectives", "fixed\nperspectives")
            .replace("no_conversations", "no\nconversations")
            .replace("no_triangulation", "no\ntriangulation")
            .replace("fixed_width", "fixed\nwidth")
            .replace("no_meta_eval", "no\nmeta eval")
            for r in sub
        ]
        colors = ["#d62728" if s == "*" else "#999999" for s in sigs]
        # ROPE band ±0.02
        ax.axvspan(-0.02, 0.02, color="#cccccc", alpha=0.35, zorder=0,
                   label="±0.02 ROPE" if base == "P3" else None)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        for y, d, lo, hi, cd, s, c in zip(ys, deltas, los, his, cliffs, sigs, colors):
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]],
                        fmt="o", color=c, capsize=3, lw=1.4, markersize=6, zorder=3)
            star = "  *" if s == "*" else ""
            ax.text(min(hi + 0.005, 0.018), y,
                    f"δ={cd:+.2f}{star}", va="center", fontsize=8, color=c)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(titles[base])
        ax.set_xlim(-0.085, 0.030)
        ax.set_xlabel("Δ overall vs full pattern")
        ax.invert_yaxis()
        if base == "P3":
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 4 (V9). Component ablation cascade — Δ overall (paired Wilcoxon, Holm-corrected; "
                 "* = p_holm<0.05; bars = 95% bootstrap CI; grey = ±0.02 ROPE)",
                 fontsize=10.5)
    save(fig, "fig4_ablations_v9")


# ============================================================
# Fig 9: Bing-vs-Tavily Protocol A
# ============================================================
def fig9_protocol_a():
    pa = pd.read_csv(ROOT / "reports/protocol_a/paired_bootstrap_summary.csv")
    sn = pd.read_csv(ROOT / "reports/phase15_tavily_sonnet/sonnet_summary.csv")
    sn["pattern"] = sn["tavily_pattern"].str.replace("protocol_a_tavily_", "").str.upper()
    pa["pat"] = pa["pattern_idx"].str.upper()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    # Panel A: GPT-5.2 paired-bootstrap Δ across 6 patterns
    ax = axes[0]
    pa_sorted = pa.sort_values("delta_tav_minus_bing")
    ys = np.arange(len(pa_sorted))
    deltas = pa_sorted["delta_tav_minus_bing"].values
    los = pa_sorted["ci95_lo"].values
    his = pa_sorted["ci95_hi"].values
    pats = pa_sorted["pat"].values
    for y, d, lo, hi, p in zip(ys, deltas, los, his, pats):
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]],
                    fmt="o", color=PATTERN_COLORS.get(p, "#444"),
                    capsize=3, lw=1.4, markersize=7, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(pats, fontsize=10)
    ax.set_xlabel("Δ overall (Tavily − Bing), GPT-5.2 axis")
    ax.set_title("(A) Protocol A — paired bootstrap (n=28-29)")
    # add p<0.001 / p<0.05 markers
    for y, p in zip(ys, pa_sorted["p_two_sided"].values):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        ax.text(pa_sorted.iloc[y]["ci95_hi"] + 0.005, y, sig, va="center", fontsize=8)

    # Panel B: Sonnet vs GPT-5.2 cross-judge — Δ(Sonnet-Tav − GPT-Bing) and Δ(Sonnet-Tav − GPT-Tav)
    ax = axes[1]
    sn_sorted = sn.sort_values("delta_sonnetTav_minus_gptBing")
    ys = np.arange(len(sn_sorted))
    width = 0.38
    for y, row in zip(ys, sn_sorted.itertuples()):
        # parse CIs which are stored as "(lo, hi)" strings
        for col, off, color, label in [
            ("ci_jb", -width/2, "#d62728", "Δ vs GPT-Bing"),
            ("ci_jo", +width/2, "#1f77b4", "Δ vs GPT-Tavily"),
        ]:
            ci_str = getattr(row, col)
            m = re.match(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", ci_str)
            if not m:
                continue
            lo, hi = float(m.group(1)), float(m.group(2))
            d = row.delta_sonnetTav_minus_gptBing if col == "ci_jb" else row.delta_sonnetTav_minus_gptTav
            ax.errorbar(d, y + off, xerr=[[d - lo], [hi - d]],
                        fmt="o", color=color, capsize=3, lw=1.4, markersize=6, zorder=3,
                        label=label if y == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(sn_sorted["pattern"].values, fontsize=10)
    ax.set_xlabel("Sonnet-Tav Δ (95% CI; n=4 per pattern)")
    ax.set_title("(B) Cross-judge probe — Sonnet on same Tavily reports")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Figure 9. Tool-layer intervention (§6.2.1): Tavily depresses scores universally under "
                 "GPT-5.2 (A), but the depression is partly judge-specific under Sonnet (B).",
                 fontsize=10.5, y=1.04)
    save(fig, "fig9_protocol_a")


# ============================================================
# Fig 10: σ(run) variance + 3-rerun spread
# ============================================================
def fig10_sigma_run():
    sr = pd.read_csv(ROOT / "reports/phase11_variance/sigma_run_per_pattern.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})

    # Panel A: σ(run) per pattern with ROPE reference
    ax = axes[0]
    sr["pat"] = sr["pattern"].str.replace("base_", "").str.upper()
    pats = sr["pat"].values
    sigmas = sr["sigma_run_mean"].values
    medians = sr["sigma_run_median"].values
    ys = np.arange(len(sr))
    for i, (y, p, s, m) in enumerate(zip(ys, pats, sigmas, medians)):
        c = PATTERN_COLORS.get(p, "#444")
        ax.barh(y, s, color=c, alpha=0.85)
        ax.scatter(m, y, marker="|", color="black", s=80, zorder=3,
                   label="median" if i == 0 else None)
    ax.axvline(0.05, color="#d62728", lw=1.5, ls="--", label="±0.05 ROPE width")
    ax.axvline(0.02, color="#9467bd", lw=1, ls=":", label="±0.02 ROPE width")
    ax.set_yticks(ys)
    ax.set_yticklabels(pats, fontsize=10)
    ax.set_xlabel("σ(run) — within-(pattern,query) std across 3 reruns")
    ax.set_title("(A) Per-pattern run-noise vs ROPE")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: max within-query range (per pattern) — shows worst-case spread
    ax = axes[1]
    ranges = sr["max_within_query_range"].values
    for y, p, r in zip(ys, pats, ranges):
        c = PATTERN_COLORS.get(p, "#444")
        ax.barh(y, r, color=c, alpha=0.85)
    ax.axvline(0.10, color="black", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(pats, fontsize=10)
    ax.set_xlabel("Max within-(pattern,query) 3-rerun range")
    ax.set_title("(B) Worst-case 3-rerun spread per pattern")

    fig.suptitle("Figure 10. σ²(run) variance experiment (§5.2): pooled σ(run) = 0.050 sits *at* the "
                 "±0.05 ROPE width. P1/P4/P7 exceed it; P0/P10 are below.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig10_sigma_run")


# ============================================================
# Fig 11: DR-Judge per-pattern + per-dim κ
# ============================================================
def fig11_dr_judge():
    pp = [
        ("P9",  338, 0.621, (0.524, 0.714)),
        ("P10", 338, 0.553, (0.455, 0.641)),
        ("P0",  444, 0.495, (0.413, 0.573)),
        ("P4",  338, 0.450, (0.360, 0.537)),
        ("P8",  338, 0.421, (0.326, 0.515)),
        ("P6",  338, 0.415, (0.332, 0.499)),
        ("P3",  338, 0.401, (0.313, 0.486)),
        ("P2",  338, 0.374, (0.282, 0.467)),
        ("P7",  338, 0.334, (0.245, 0.419)),
        ("P5",  338, 0.332, (0.247, 0.417)),
        ("P1",  338, 0.210, (0.129, 0.296)),
    ]
    pd_ = [
        ("organization",          404, 0.778, (0.634, 0.889)),
        ("factual_accuracy",      809, 0.452, (0.394, 0.506)),
        ("analytical_depth",      408, 0.401, (0.337, 0.469)),
        ("citation_quality",      406, 0.398, (0.307, 0.484)),
        ("coverage",              472, 0.392, (0.320, 0.458)),
        ("logical_coherence",     309, 0.355, (0.232, 0.462)),
        ("information_recall",    409, 0.264, (0.203, 0.331)),
        ("instruction_following", 405, 0.249, (0.171, 0.323)),
        ("attribution_quality",   202, 0.147, (0.016, 0.310)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, data, title, ylabel in [
        (axes[0], pp, "(A) DR-Judge κ vs panel — per pattern", "Pattern"),
        (axes[1], pd_, "(B) DR-Judge κ vs panel — per dimension", "Dimension"),
    ]:
        ys = np.arange(len(data))
        kappas = [r[2] for r in data]
        los = [r[3][0] for r in data]
        his = [r[3][1] for r in data]
        labels = [r[0] for r in data]
        # Color by κ band
        colors = ["#1a9641" if k >= 0.6 else ("#fdae61" if k >= 0.4 else "#d7191c") for k in kappas]
        for y, k, lo, hi, c in zip(ys, kappas, los, his, colors):
            ax.errorbar(k, y, xerr=[[k - lo], [hi - k]],
                        fmt="o", color=c, capsize=3, lw=1.5, markersize=7, zorder=3)
        ax.axvline(0.6, color="#1a9641", lw=1, ls=":", alpha=0.7,
                   label="substantial (κ≥0.6)")
        ax.axvline(0.4, color="#fdae61", lw=1, ls=":", alpha=0.7,
                   label="moderate (κ≥0.4)")
        ax.axvline(0.7, color="black", lw=1.5, ls="--",
                   label="pre-reg target κ≥0.7")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Cohen's κ vs panel consensus")
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.0)
        ax.invert_yaxis()
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Figure 11. DR-Judge-7B held-out evaluation (§5.6, n=3,824) — pre-reg miss; "
                 "agreement is high on patterns/dimensions where the panel itself agrees and falls "
                 "where the panel doesn't.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig11_dr_judge")


# ============================================================
# Fig 12: C0 verdict distribution + per-pattern band
# ============================================================
def fig12_c0():
    df_per = pd.read_parquet(ROOT / "data/analysis/df_c0_per_report.parquet")
    df_v   = pd.read_parquet(ROOT / "data/analysis/df_c0_verdicts.parquet")
    cit    = pd.read_parquet(ROOT / "data/analysis/df_citations.parquet") \
                if (ROOT / "data/analysis/df_citations.parquet").exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [0.85, 1.15]})

    # Panel A: overall verdict distribution stacked bar (single column) +
    # comparison vs prompt-sensitivity strict/soft
    ax = axes[0]
    counts = df_v["verdict"].value_counts()
    total = counts.sum()
    bars = [
        ("All claims (strict v3)", total,
         counts.get("supports", 0), counts.get("neutral", 0),
         counts.get("contradicts", 0), counts.get("no_source", 0)),
        ("Top-cluster 50-claim subset (strict)",  50, 8,  41, 1, 0),
        ("Top-cluster 50-claim subset (soft)",    50, 20, 29, 1, 0),
    ]
    cats = ["supports", "neutral", "contradicts", "no_source"]
    cat_colors = {"supports": "#1a9641", "neutral": "#fdae61",
                  "contradicts": "#d7191c", "no_source": "#999999"}
    ys = np.arange(len(bars))
    bottoms = np.zeros(len(bars))
    for i, cat in enumerate(cats):
        vals = np.array([b[i + 2] / b[1] * 100 for b in bars])
        ax.barh(ys, vals, left=bottoms, color=cat_colors[cat], label=cat,
                edgecolor="white", lw=0.5)
        # text labels
        for y, v, b in zip(ys, vals, bottoms):
            if v >= 4:
                ax.text(b + v / 2, y, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, color="white",
                        weight="bold" if cat in ("supports", "neutral") else "normal")
        bottoms = bottoms + vals
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of atomic claims")
    ax.set_title("(A) C0 verdict distribution — strict vs soft prompt")
    ax.legend(loc="lower right", fontsize=8, ncol=4, bbox_to_anchor=(1.02, -0.2))

    # Panel B: per-pattern verified_factual_accuracy means colored by placeholder rate
    ax = axes[1]
    # Pull placeholder rate from §6.2 table values for the 9 patterns + P11
    ph_rate = {"P0": 0.0, "P1": 0.002, "P3": 0.614, "P4": 0.629, "P5": 0.553,
               "P7": 0.002, "P8": 0.604, "P9": 0.0, "P10": 0.279, "P11": 0.165}
    perpat = (df_per.groupby("pattern")["verified_factual_accuracy"]
                       .agg(["mean", "median", "count"]).reset_index())
    perpat["pat"] = perpat["pattern"].str.replace("base_", "").str.upper()
    perpat["ph"] = perpat["pat"].map(ph_rate).fillna(0.0)
    perpat = perpat.sort_values("mean", ascending=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=0.65)
    cmap = mpl.colormaps["RdYlGn_r"]
    colors = [cmap(norm(p)) for p in perpat["ph"].values]
    ys = np.arange(len(perpat))
    ax.barh(ys, perpat["mean"].values, color=colors, edgecolor="black", lw=0.4)
    for y, m, med, n in zip(ys, perpat["mean"].values,
                            perpat["median"].values, perpat["count"].values):
        ax.text(m + 0.005, y, f"  {m:.3f} (med {med:.2f}, n={int(n)})",
                va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(perpat["pat"].values, fontsize=10)
    ax.set_xlabel("Verified-factual-accuracy (strict-direct-support entailment)")
    ax.set_title("(B) Per-pattern C0 verified-factual-accuracy")
    ax.set_xlim(0, 0.55)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("placeholder rate (§6.2)", fontsize=8)

    fig.suptitle("Figure 12. C0 citation-grounded verification (§6.8) — strict-direct-support "
                 "entailment routes 69.5% of claims to `neutral`; partial-support rubric lifts supports 16%→40%.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig12_c0")


# ============================================================
# Fig 13: P12 GRPO training curve
# ============================================================
def fig13_p12_training():
    df = pd.read_csv(ROOT / "reports/phase16_p12_eval/training_log.csv")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df["reward_std"] = pd.to_numeric(df["reward_std"], errors="coerce")
    df = df.dropna(subset=["reward"]).sort_values("step")
    df["rolling"] = df["reward"].rolling(50, min_periods=10).mean()
    quintiles = [(1, 400), (401, 800), (801, 1200), (1201, 1600), (1601, 2000)]
    qmeans = []
    for lo, hi in quintiles:
        sub = df[(df["step"] >= lo) & (df["step"] <= hi)]
        qmeans.append((lo, hi, sub["reward"].mean(), sub["reward"].std()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})

    # Panel A: per-step reward + rolling mean + quintile means
    ax = axes[0]
    ax.scatter(df["step"], df["reward"], s=4, alpha=0.18, color="#7f7f7f",
               label="per-step reward")
    ax.plot(df["step"], df["rolling"], color="#1f77b4", lw=1.6,
            label="50-step rolling mean")
    for (lo, hi, m, s) in qmeans:
        ax.hlines(m, lo, hi, color="#d62728", lw=2.5, zorder=4)
        ax.text((lo + hi) / 2, m + 0.04, f"{m:.4f}", ha="center", fontsize=8, color="#d62728")
    # Annotate quintile range
    ax.set_xlabel("GRPO step (of 2,000)")
    ax.set_ylabel("DR-Judge reward (mean per step)")
    ax.set_ylim(0.20, 1.00)
    ax.set_title("(A) Training-curve — flat across 2,000 steps")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.96,
            f"5-quintile range: {min(q[2] for q in qmeans):.4f}–"
            f"{max(q[2] for q in qmeans):.4f}\n"
            f"frac_reward_zero_std=1: {(df['frac_reward_zero_std']==1).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            va="top")

    # Panel B: panel-side outcome (P12 = 0.182) vs comparators
    ax = axes[1]
    panel = [("P0 GPT-4o", 0.385, "#1f77b4"),
             ("P10 RL 7B", 0.213, "#ff7f0e"),
             ("P12 own RL", 0.182, "#d62728"),
             ("P9 base 7B", 0.180, "#bcbd22")]
    ys = np.arange(len(panel))
    for y, (n, m, c) in zip(ys, panel):
        ax.barh(y, m, color=c, alpha=0.85)
        ax.text(m + 0.005, y, f"  {m:.3f}", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _ in panel], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("GPT-5.2 single-judge mean (n=90)")
    ax.set_title("(B) Panel outcome — P12 ≈ P9 base")

    fig.suptitle("Figure 13. P12 GRPO training (§6.5) — DR-Judge reward signal flowed through (77% non-zero "
                 "advantage) but the policy did not improve. Panel score is statistically tied with the "
                 "untrained Qwen2.5-7B base.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig13_p12_training")


# ============================================================
# Fig 14: Process trajectory pattern-level scatter
# ============================================================
def fig14_trajectory():
    rows = [
        # pattern, retrieval_div, tool_eff, reasoning_coh, iterative_ref, outcome
        ("P0",  0.830, 0.571, 0.677, 0.247, 0.488),
        ("P1",  0.571, 0.590, 0.707, 0.676, 0.673),
        ("P4",  0.996, 0.999, 0.787, 0.512, 0.640),
        ("P7",  0.866, 0.871, 0.811, 0.811, 0.630),
        ("P10", 0.735, 0.547, 0.453, 0.237, 0.336),
    ]
    dims = [("retrieval_diversity", -0.10, 0.87, 1),
            ("tool_efficiency",     +0.60, 0.29, 2),
            ("reasoning_coherence", +0.70, 0.19, 3),
            ("iterative_refinement",+0.90, 0.037, 4)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (dim, rho, p, idx) in zip(axes, dims):
        xs = [r[idx] for r in rows]
        ys = [r[5] for r in rows]
        for r, x, y in zip(rows, xs, ys):
            c = PATTERN_COLORS.get(r[0], "#444")
            ax.scatter(x, y, s=140, color=c, edgecolor="black", lw=0.6, zorder=3)
            ax.annotate(r[0], (x, y), xytext=(7, 5),
                        textcoords="offset points", fontsize=10, weight="bold")
        # Regression line
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs) * 0.95, max(xs) * 1.02, 50)
        ax.plot(x_line, m * x_line + b, color="black", lw=0.9, ls="--", alpha=0.6)
        ax.set_xlabel(dim.replace("_", " "))
        ax.set_title(f"ρ = {rho:+.2f} (p = {p:.3f})",
                     fontsize=10, color=("#1a9641" if p < 0.05 else "#444"))
    axes[0].set_ylabel("3-judge outcome (§5.3)")
    fig.suptitle("Figure 14. Process trajectory ↔ outcome (§6.9; n = 5 instrumented patterns) — "
                 "iterative_refinement is the only dim that predicts outcome at α=0.05.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig14_trajectory")


# ============================================================
# Fig 15: Placeholder-filtered re-rank survival
# ============================================================
def fig15_placeholder_filter():
    rows = [
        # (pattern, n_total, n_filtered, unfiltered_mean, filtered_mean)
        ("P1",  90, 90, 0.653, 0.653),
        ("P7",  90, 90, 0.618, 0.618),
        ("P6",  87, 86, 0.616, 0.618),
        ("P0",  81, 81, 0.525, 0.525),
        ("P9",  52, 52, 0.424, 0.424),
        ("P10", 72, 51, 0.342, 0.372),
        ("P2",  90,  6, 0.540, np.nan),
        ("P3",  86,  1, 0.543, np.nan),
        ("P4",  90,  1, 0.623, np.nan),
        ("P5",  88,  1, 0.589, np.nan),
        ("P8",  90,  0, 0.601, np.nan),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={"width_ratios": [0.95, 1.05]})

    # Panel A: retention % per pattern
    ax = axes[0]
    pats = [r[0] for r in rows]
    retention = [r[2] / r[1] * 100 if r[1] > 0 else 0 for r in rows]
    colors = ["#1a9641" if rt >= 70 else ("#fdae61" if rt >= 7 else "#d7191c") for rt in retention]
    ys = np.arange(len(rows))
    ax.barh(ys, retention, color=colors, edgecolor="black", lw=0.4)
    ax.axvline(7, color="black", lw=0.8, ls=":", label="<7% = unscoreable (n<10)")
    for y, rt, n in zip(ys, retention, [r[2] for r in rows]):
        ax.text(rt + 1, y, f"  n={n} ({rt:.0f}%)", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(pats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of reports retained under <10% placeholder filter")
    ax.set_title("(A) Retention under placeholder filter")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: unfiltered vs filtered mean (only survivors)
    ax = axes[1]
    survivors = [r for r in rows if not np.isnan(r[4])]
    pats = [r[0] for r in survivors]
    unfilt = [r[3] for r in survivors]
    filt = [r[4] for r in survivors]
    width = 0.38
    ys = np.arange(len(survivors))
    ax.barh(ys - width/2, unfilt, width, color="#777", label="Unfiltered")
    ax.barh(ys + width/2, filt, width, color="#1f77b4", label="Filtered (<10% placeholder)")
    for y, uf, ft in zip(ys, unfilt, filt):
        delta = ft - uf
        ax.text(max(uf, ft) + 0.005, y, f"Δ = {delta:+.3f}", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(pats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean overall score")
    ax.set_title("(B) Surviving patterns — filtered ranking")
    ax.set_xlim(0, 0.85)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Figure 15. Placeholder-filtered re-rank (§6.2) — under <10% placeholder filter, "
                 "5 of 11 patterns drop below 7% retention; surviving order P1 > P6 ≈ P7 > P0 > P9 > P10.",
                 fontsize=10.5, y=1.04)
    save(fig, "fig15_placeholder_filter")


if __name__ == "__main__":
    print("Generating V9 figures →", OUT)
    fig4_ablations_v9()
    fig9_protocol_a()
    fig10_sigma_run()
    fig11_dr_judge()
    fig12_c0()
    fig13_p12_training()
    fig14_trajectory()
    fig15_placeholder_filter()
    print("Done.")
