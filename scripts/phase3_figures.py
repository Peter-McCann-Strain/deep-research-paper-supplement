"""Phase 3: publication-grade figures.

Outputs to reports/phase3_figures/ as .pdf and .png at 300dpi.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from deep_research.evaluation.rubric_v2 import DIMENSION_WEIGHTS_V2
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.insert(0, str(Path(".")))
    from deep_research.evaluation.rubric_v2 import DIMENSION_WEIGHTS_V2

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "data" / "analysis"
OUT = ROOT / "reports" / "phase3_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "figure.autolayout": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe
OI = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermilion": "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}

# 3 paradigm families
FAMILY_COLORS = {
    "GPT-4o pipeline": OI["blue"],
    "Local 7B": OI["orange"],
    "RL 7B": OI["vermilion"],
}

BASE_PATTERNS = [f"base_p{i}" for i in range(11)]
PATTERN_LABEL = {f"base_p{i}": f"P{i}" for i in range(11)}

def family_of(pattern: str) -> str:
    if pattern == "base_p9":
        return "Local 7B"
    if pattern == "base_p10":
        return "RL 7B"
    return "GPT-4o pipeline"

# V2 rubric weights (from memory)
DIM_WEIGHTS = {
    "information_recall": 0.20,
    "factual_accuracy":   0.20,
    "coverage":           0.10,
    "analytical_depth":   0.15,
    "citation_quality":   0.10,
    "logical_coherence":  0.05,
    "organization":       0.05,
    "instruction_following": 0.10,
    "attribution_quality": 0.05,
}
DIM_ORDER = list(DIM_WEIGHTS.keys())
HEADLINE_DIMS = [d for d in DIM_ORDER if d != "attribution_quality"]

# ---------------------------------------------------------------------------
# Data load + canonical overall
# ---------------------------------------------------------------------------
def load_data():
    runs = pd.read_parquet(ANALYSIS / "df_runs.parquet")
    overall = pd.read_parquet(ANALYSIS / "df_overall_scores.parquet")
    scores = pd.read_parquet(ANALYSIS / "df_scores.parquet")
    queries = pd.read_parquet(ANALYSIS / "df_queries.parquet")

    for df in (runs, overall, scores, queries):
        for c in df.select_dtypes(include="category"):
            df[c] = df[c].astype(str)

    # Trustworthy overall
    overall["overall"] = np.where(
        overall["overall_score_trustworthy"],
        overall["overall_score"],
        overall["overall_score_recomputed"],
    )
    return runs, overall, scores, queries


def judge_mean_overall(overall: pd.DataFrame) -> pd.DataFrame:
    """Return per (pattern, query) 3-judge mean overall score."""
    g = overall.groupby(["pattern", "query_id"], as_index=False)["overall"].mean()
    return g


def judge_mean_by_dim(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (pattern, query, dimension) 3-judge mean score."""
    g = scores.groupby(["pattern", "query_id", "dimension"], as_index=False)["score"].mean()
    return g


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------
def boot_ci(values: np.ndarray, n=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    mu = values.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mu, lo, hi


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save(fig, name: str):
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf + .png")


# ---------------------------------------------------------------------------
# Figure 1 — Cost-Quality Pareto
# ---------------------------------------------------------------------------
def fig1_pareto(runs, overall_mean, queries):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rows = []
    rng = np.random.default_rng(42)
    for p in BASE_PATTERNS:
        sub_runs = runs[(runs.pattern == p) & (~runs.excluded_from_analysis)]
        if sub_runs.empty:
            continue
        mean_cost = sub_runs["cost_proxy_usd"].mean()
        med_elapsed = sub_runs["elapsed_seconds"].median()
        overall_sub = overall_mean[overall_mean.pattern == p]["overall"].to_numpy()
        mu, lo, hi = boot_ci(overall_sub, n=1000, rng=rng)
        rows.append({
            "pattern": p,
            "label": PATTERN_LABEL[p],
            "cost": mean_cost,
            "elapsed": med_elapsed,
            "mean": mu,
            "lo": lo,
            "hi": hi,
            "family": family_of(p),
        })
    df = pd.DataFrame(rows)

    # error bars
    for _, r in df.iterrows():
        ax.errorbar(
            r["cost"], r["mean"],
            yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
            fmt="none", ecolor="grey", alpha=0.6, linewidth=0.8, capsize=2,
        )

    # Legend proxies for family colours (actual points drawn later with per-point alpha)
    for fam, col in FAMILY_COLORS.items():
        ax.scatter([], [], c=[col], alpha=0.85, edgecolors="black", linewidths=0.6, label=fam)

    # Pareto frontier: non-dominated (minimize cost, maximize score)
    df_sorted = df.sort_values("cost")
    frontier = []
    max_y = -np.inf
    for _, r in df_sorted.iterrows():
        if r["mean"] > max_y:
            frontier.append(r)
            max_y = r["mean"]
    fdf = pd.DataFrame(frontier)
    frontier_labels = set(fdf["label"]) if len(fdf) else set()

    # Mark dominated points with lower alpha
    df["on_frontier"] = df["label"].isin(frontier_labels)
    for _, r in df.iterrows():
        alpha_val = 1.0 if r["on_frontier"] else 0.4
        fam_col = FAMILY_COLORS[r["family"]]
        ax.scatter(r["cost"], r["mean"],
                   s=30 + (r["elapsed"] / df["elapsed"].max() * 250),
                   c=[fam_col], alpha=alpha_val,
                   edgecolors="black", linewidths=0.6, zorder=3)

    if len(fdf) >= 2:
        # stepped line
        ax.step(fdf["cost"], fdf["mean"], where="post",
                color=OI["green"], linewidth=1.5, alpha=0.7, zorder=2,
                label="Pareto frontier")

    # annotate — use adjustText to prevent label collisions
    from adjustText import adjust_text
    texts = []
    for _, r in df.iterrows():
        t = ax.text(r["cost"], r["mean"], r["label"], fontsize=8, fontweight="bold")
        texts.append(t)
    adjust_text(
        texts,
        x=df["cost"].values,
        y=df["mean"].values,
        ax=ax,
        expand=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per query (USD, log scale)")
    ax.set_ylabel("Mean overall score (3-judge avg)")
    ax.set_title("Figure 1. Cost–quality Pareto frontier across 11 deep-research patterns")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig1_pareto")
    return df


# ---------------------------------------------------------------------------
# Figure 2 — System × Dimension Heatmap
# ---------------------------------------------------------------------------
def fig2_heatmap(scores, dim_mean, overall_mean):
    # dim_mean: per (pattern, query, dimension) 3-judge avg
    mat = (dim_mean[dim_mean.pattern.isin(BASE_PATTERNS)]
           .groupby(["pattern", "dimension"])["score"].mean()
           .unstack("dimension"))
    mat = mat.reindex(index=BASE_PATTERNS, columns=DIM_ORDER)
    labels = [PATTERN_LABEL[p] for p in mat.index]

    # Weighted overall (headline: exclude attribution_quality; re-normalize weights)
    w_head = np.array([DIM_WEIGHTS[d] for d in HEADLINE_DIMS])
    w_head = w_head / w_head.sum()
    overall_weighted = (mat[HEADLINE_DIMS].values * w_head).sum(axis=1)

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[0.25, 10, 1.8],
                          height_ratios=[10, 0.9],
                          wspace=0.08, hspace=0.08)
    ax_cb = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_hm)
    ax_w = fig.add_subplot(gs[1, 1], sharex=ax_hm)

    im = ax_hm.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_hm.set_yticks(range(len(labels)))
    ax_hm.set_yticklabels(labels)
    ax_hm.set_xticks(range(len(DIM_ORDER)))
    ax_hm.set_xticklabels([d.replace("_", "\n") for d in DIM_ORDER], rotation=0, fontsize=7)
    ax_hm.tick_params(axis="x", labelbottom=True, bottom=True)
    ax_hm.set_title("Figure 2. Mean dimension scores by pattern (3-judge avg)")

    # Annotate cells; bold rank-1 per column
    top_per_col = mat.idxmax(axis=0)
    for i, p in enumerate(mat.index):
        for j, d in enumerate(DIM_ORDER):
            v = mat.loc[p, d]
            is_top = top_per_col[d] == p
            color = "white" if v < 0.5 else "black"
            ax_hm.text(j, i, f"{v:.2f}",
                       ha="center", va="center",
                       color=color,
                       fontsize=7,
                       fontweight="bold" if is_top else "normal")
    ax_hm.grid(False)

    # right: weighted overall bar
    y = np.arange(len(labels))
    ax_bar.barh(y, overall_weighted, color=OI["blue"], alpha=0.8, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(overall_weighted):
        ax_bar.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    ax_bar.set_xlim(0, max(overall_weighted) * 1.2)
    ax_bar.set_xlabel("Weighted\noverall")
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(False)
    # imshow default puts row 0 at top; no inversion needed

    # bottom: V2 weights — columns must align 1:1 with heatmap columns (same DIM_ORDER)
    ax_w.bar(range(len(DIM_ORDER)), [DIMENSION_WEIGHTS_V2[d] for d in DIM_ORDER],
             color=OI["grey"], alpha=0.7, edgecolor="black", linewidth=0.4)
    ax_w.set_ylabel("V2 wt", fontsize=7)
    ax_w.set_ylim(0, 0.25)
    # xlim locked to heatmap via sharex; set explicitly to match imshow extent
    ax_w.set_xlim(-0.5, len(DIM_ORDER) - 0.5)
    ax_w.set_xticks(range(len(DIM_ORDER)))
    ax_w.set_xticklabels([d.replace("_", "\n") for d in DIM_ORDER], rotation=0, fontsize=6)
    ax_w.tick_params(axis="y", labelsize=6)
    ax_w.grid(False)

    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Score")
    ax_cb.yaxis.set_ticks_position("left")
    ax_cb.yaxis.set_label_position("left")
    save(fig, "fig2_heatmap")


# ---------------------------------------------------------------------------
# Figure 3 — Critical Difference Diagram (Wilcoxon-Holm)
# ---------------------------------------------------------------------------
def fig3_cd(overall_mean):
    # Build wide matrix: rows=queries, cols=patterns, values=3-judge mean overall
    wide = (overall_mean[overall_mean.pattern.isin(BASE_PATTERNS)]
            .pivot_table(index="query_id", columns="pattern", values="overall"))
    wide = wide.reindex(columns=BASE_PATTERNS)
    # drop queries missing any pattern
    wide = wide.dropna(how="any")

    # Higher is better -> rank with descending (rank 1 = best)
    ranks = wide.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()

    # Wilcoxon-Holm pairwise
    from scipy.stats import wilcoxon
    from itertools import combinations
    patterns = list(wide.columns)
    k = len(patterns)
    pmat = pd.DataFrame(np.ones((k, k)), index=patterns, columns=patterns)
    pairs = []
    for a, b in combinations(patterns, 2):
        try:
            _, p = wilcoxon(wide[a], wide[b], zero_method="wilcox")
        except ValueError:
            p = 1.0
        pairs.append((a, b, p))

    # Holm correction
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    m = len(pairs_sorted)
    holm = {}
    for i, (a, b, p) in enumerate(pairs_sorted):
        adj = min(1.0, p * (m - i))
        holm[(a, b)] = adj
        holm[(b, a)] = adj
    for (a, b), p in holm.items():
        pmat.loc[a, b] = p

    # Try scikit-posthocs CD diagram
    try:
        import scikit_posthocs as sp
        fig, ax = plt.subplots(figsize=(9, 3.2))
        # scikit_posthocs.critical_difference_diagram expects Series of mean ranks
        # + significance matrix (p-values)
        sp.critical_difference_diagram(
            ranks=mean_ranks,
            sig_matrix=pmat,
            ax=ax,
            label_fmt_left="{label} ({rank:.2f})",
            label_fmt_right="{label} ({rank:.2f})",
            text_h_margin=0.1,
        )
        # relabel: replace base_pN with PN
        for txt in ax.texts:
            s = txt.get_text()
            for orig, new in PATTERN_LABEL.items():
                s = s.replace(orig, new)
            txt.set_text(s)
        ax.set_title("Figure 3. Critical difference diagram (Wilcoxon-Holm, α=0.05)")

        # Caption line clarifying crossbar meaning
        fig.text(0.5, -0.04,
                 "Crossbars connect patterns that are NOT significantly different "
                 "(Wilcoxon-Holm, α=0.05). Lower mean rank = better.",
                 ha="center", va="top", fontsize=7, style="italic")

        # Paradigm family colour legend
        from matplotlib.lines import Line2D
        family_color = {
            "GPT-4o pipeline": OI["blue"],
            "Local 7B": OI["orange"],
            "RL 7B": OI["vermilion"],
        }
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=col, markersize=7, label=lbl)
            for lbl, col in family_color.items()
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False)

        save(fig, "fig3_cd_diagram")
        return
    except Exception as e:
        print(f"  CD via scikit_posthocs failed: {e}; falling back to bar rank plot")

    # Fallback: rank plot with significance crossbars
    fig, ax = plt.subplots(figsize=(9, 3.5))
    mr = mean_ranks.sort_values()
    xs = mr.values
    labels = [PATTERN_LABEL[p] for p in mr.index]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        ax.plot([x], [0], "o", color=OI["blue"], markersize=8)
        ax.annotate(lab, (x, 0), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_yticks([])
    ax.set_title("Figure 3. Mean ranks (CD diagram fallback)")
    fig.text(0.5, -0.06,
             "Crossbars connect patterns that are NOT significantly different "
             "(Wilcoxon-Holm, α=0.05). Lower mean rank = better.",
             ha="center", va="top", fontsize=7, style="italic")
    save(fig, "fig3_cd_diagram")


# ---------------------------------------------------------------------------
# Figure 4 — Ablation Cascade
# ---------------------------------------------------------------------------
def fig4_ablations():
    csv_path = ROOT / "reports" / "phase2_statistics" / "fixes" / "03_ablations_2judge.csv"
    if not csv_path.exists():
        print(f"  skip fig4: missing {csv_path}")
        return
    abl = pd.read_csv(csv_path)

    # Group by base pattern
    base_groups = {"base_p3": [], "base_p4": [], "base_p5": []}
    for _, r in abl.iterrows():
        base_groups[r["base"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
    for ax, base in zip(axes, ["base_p3", "base_p4", "base_p5"]):
        rows = base_groups[base]
        if not rows:
            ax.set_title(f"{PATTERN_LABEL[base]} (no ablations)")
            continue
        names = [r["ablation"].replace(f"ablation_{base.replace('base_', '')}_", "") for r in rows]
        deltas = [r["mean_diff"] for r in rows]
        los = [r["ci_lo"] for r in rows]
        his = [r["ci_hi"] for r in rows]
        cliffs = [r["cliffs_delta"] for r in rows]
        sig = [r["sig_holm"] for r in rows]

        # Sort by delta ascending (most harmful first)
        order = np.argsort(deltas)
        names = [names[i] for i in order]
        deltas = [deltas[i] for i in order]
        los = [los[i] for i in order]
        his = [his[i] for i in order]
        cliffs = [cliffs[i] for i in order]
        sig = [sig[i] for i in order]

        colors = [OI["vermilion"] if d < 0 else OI["green"] for d in deltas]
        y = np.arange(len(names))
        ax.barh(y, deltas, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        # CI
        for i, (d, lo, hi) in enumerate(zip(deltas, los, his)):
            ax.plot([lo, hi], [i, i], color="black", linewidth=1, alpha=0.7)
            ax.plot([lo, lo], [i - 0.1, i + 0.1], color="black", linewidth=1)
            ax.plot([hi, hi], [i - 0.1, i + 0.1], color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{PATTERN_LABEL[base]} ablations")
        ax.set_xlabel("Δ overall vs full pattern")
        # Annotations
        for i, (d, cd, s) in enumerate(zip(deltas, cliffs, sig)):
            txt = f"δ={cd:.2f}" + (" *" if s else "")
            x_txt = d + (0.003 if d >= 0 else -0.003)
            ha = "left" if d >= 0 else "right"
            ax.text(x_txt, i, txt, va="center", ha=ha, fontsize=7)

    fig.suptitle("Figure 4. Ablation cascade (2-judge avg, Wilcoxon-Holm *significance, 95% CI)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig4_ablations")


# ---------------------------------------------------------------------------
# Figure 5 — Raincloud
# ---------------------------------------------------------------------------
def fig5_raincloud(overall_mean):
    df = overall_mean[overall_mean.pattern.isin(BASE_PATTERNS)].copy()
    df["label"] = df.pattern.map(PATTERN_LABEL)
    # Sort by median desc
    med = df.groupby("label")["overall"].median().sort_values(ascending=False)
    order = med.index.tolist()

    # Restricted to the top non-sig clique from Fig 3 CD diagram (P5 excluded)
    top_cluster = {"P1", "P4", "P6", "P7", "P8"}

    try:
        import ptitprince as pt
        # ptitprince expects categorical on x-axis and numeric on y-axis regardless of orient
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        pt.RainCloud(
            x="label", y="overall", data=df,
            order=order, orient="h",
            palette=[OI["blue"] if lab in top_cluster else OI["grey"] for lab in order],
            bw=.2, width_viol=.6, ax=ax, move=.2, alpha=0.7, point_size=2,
        )
    except Exception as e:
        print(f"  ptitprince failed ({e}); falling back to seaborn violin+strip")
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        palette = [OI["blue"] if lab in top_cluster else OI["grey"] for lab in order]
        sns.violinplot(data=df, x="overall", y="label", order=order,
                       orient="h", inner=None, palette=palette, cut=0, linewidth=0.6, ax=ax)
        sns.boxplot(data=df, x="overall", y="label", order=order,
                    orient="h", width=0.18, showcaps=True,
                    boxprops={"facecolor": "white", "edgecolor": "black"},
                    medianprops={"color": "red"}, fliersize=0, ax=ax)
        sns.stripplot(data=df, x="overall", y="label", order=order,
                      orient="h", size=1.8, color="black", alpha=0.4, jitter=0.18, ax=ax)

    # shade top cluster
    for i, lab in enumerate(order):
        if lab in top_cluster:
            ax.axhspan(i - 0.5, i + 0.5, color=OI["blue"], alpha=0.05, zorder=0)

    ax.set_xlabel("Overall score (3-judge avg)")
    ax.set_ylabel("Pattern (median-sorted)")
    ax.set_xlim(0, 1)
    ax.set_title("Figure 5. Per-system distribution of overall scores across 90 queries")
    save(fig, "fig5_raincloud")


# ---------------------------------------------------------------------------
# Figure 6 — Stratified small multiples
# ---------------------------------------------------------------------------
def fig6_stratified(overall_mean, queries):
    df = overall_mean.merge(queries[["query_id", "source", "difficulty"]], on="query_id")
    df = df[df.pattern.isin(BASE_PATTERNS)].copy()
    df["label"] = df.pattern.map(PATTERN_LABEL)

    sources = ["custom", "draco", "deepsearch_qa", "research_qa", "litqa2"]
    difficulties = ["simple", "moderate", "complex"]

    fig, axes = plt.subplots(2, 5, figsize=(13, 6), sharey=True)
    rng = np.random.default_rng(123)

    def _plot_panel(ax, sub, title):
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            ax.set_ylim(0, 1)
            return
        rows = []
        for p in BASE_PATTERNS:
            vals = sub[sub.pattern == p]["overall"].to_numpy()
            mu, lo, hi = boot_ci(vals, n=500, rng=rng)
            rows.append({"label": PATTERN_LABEL[p], "mu": mu, "lo": lo, "hi": hi,
                         "family": family_of(p)})
        r = pd.DataFrame(rows)
        x = np.arange(len(r))
        for fam, col in FAMILY_COLORS.items():
            mask = r["family"] == fam
            ax.errorbar(x[mask], r["mu"][mask],
                        yerr=[r["mu"][mask] - r["lo"][mask],
                              r["hi"][mask] - r["mu"][mask]],
                        fmt="o", color=col, capsize=2, markersize=4, linewidth=0.8,
                        label=fam)
        ax.set_xticks(x)
        ax.set_xticklabels(r["label"], rotation=45, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)

    LOW_POWER_THRESHOLD = 10

    for j, src in enumerate(sources):
        sub = df[df.source == src]
        n_queries = sub.query_id.nunique()
        if n_queries < LOW_POWER_THRESHOLD:
            title = f"source: {src}\n(n={n_queries}, low power)"
        else:
            title = f"source: {src}\n(n={n_queries})"
        _plot_panel(axes[0, j], sub, title)

    for j, diff in enumerate(difficulties):
        sub = df[df.difficulty == diff]
        n_queries = sub.query_id.nunique()
        if n_queries < LOW_POWER_THRESHOLD:
            title = f"difficulty: {diff}\n(n={n_queries}, low power)"
        else:
            title = f"difficulty: {diff}\n(n={n_queries})"
        _plot_panel(axes[1, j], sub, title)

    # Fill empty cells in bottom row (3 of 5)
    for j in range(3, 5):
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Overall (3-judge)")
    axes[1, 0].set_ylabel("Overall (3-judge)")

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower right",
               bbox_to_anchor=(0.98, 0.02), ncol=3, frameon=False)
    fig.suptitle("Figure 6. Stratified performance by query source (top) and difficulty (bottom)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, "fig6_stratified")


# ---------------------------------------------------------------------------
# Figure 7 — Process-Quality Scatter
# ---------------------------------------------------------------------------
def fig7_process(runs, overall_mean):
    # Merge per (pattern, query_id)
    m = (runs[~runs.excluded_from_analysis]
         .groupby(["pattern", "query_id"], as_index=False)
         .agg(total_tokens=("total_tokens", "mean"),
              citations=("citations", "mean"),
              elapsed=("elapsed_seconds", "mean")))
    m = m.merge(overall_mean, on=["pattern", "query_id"])
    m = m[m.pattern.isin(BASE_PATTERNS)]
    m["family"] = m.pattern.map(family_of)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def _scatter(ax, xcol, xlabel, logx=False):
        for fam, col in FAMILY_COLORS.items():
            sub = m[m.family == fam]
            ax.scatter(sub[xcol], sub["overall"], c=col, alpha=0.35, s=10,
                       edgecolors="none", label=fam)
        # LOESS overall
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            x_all = m[xcol].values
            y_all = m["overall"].values
            mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0 if logx else True)
            xs, ys = x_all[mask], y_all[mask]
            sm = lowess(ys, xs, frac=0.3, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=1.5, label="LOESS")
        except Exception as e:
            print(f"  LOESS failed on {xcol}: {e}")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall (3-judge)")
        ax.set_ylim(0, 1)

    _scatter(axes[0], "total_tokens", "Total tokens", logx=True)
    axes[0].set_title("(a) Tokens vs quality")
    _scatter(axes[1], "citations", "Citations per report")
    axes[1].set_title("(b) Citations vs quality")
    _scatter(axes[2], "elapsed", "Elapsed seconds (log)", logx=True)
    axes[2].set_title("(c) Elapsed vs quality")

    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    fig.suptitle("Figure 7. Process metrics vs quality (all patterns, base runs)")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "fig7_process")


# ---------------------------------------------------------------------------
# Figure 8 — Judge Agreement Diagnostics
# ---------------------------------------------------------------------------
def fig8_judge_agreement(overall, scores):
    base = overall[overall.pattern.isin(BASE_PATTERNS)]
    pivot = base.pivot_table(index=["pattern", "query_id"], columns="judge",
                             values="overall").dropna(how="any")

    from scipy.stats import pearsonr, spearmanr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) gpt52 vs sonnet scatter
    ax = axes[0]
    x = pivot["gpt52"].values
    y = pivot["claude_sonnet"].values
    ax.scatter(x, y, s=10, alpha=0.4, color=OI["blue"], edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
    r, _ = pearsonr(x, y)
    # ICC(2,1) via simple formula
    try:
        # two-way random, single rater, absolute agreement
        import pingouin as pg
        long = pivot[["gpt52", "claude_sonnet"]].reset_index().melt(
            id_vars=["pattern", "query_id"], var_name="judge", value_name="s"
        )
        long["target"] = long["pattern"] + "|" + long["query_id"].astype(str)
        icc = pg.intraclass_corr(data=long, targets="target", raters="judge", ratings="s")
        icc21 = icc[icc["Type"] == "ICC2"]["ICC"].values[0]
    except Exception:
        # fallback manual ICC(2,1): treating each (pattern,query_id) cell
        from scipy.stats import f as fdist  # noqa
        ratings = pivot[["gpt52", "claude_sonnet"]].values
        n, k = ratings.shape
        mean_rows = ratings.mean(axis=1)
        mean_cols = ratings.mean(axis=0)
        grand = ratings.mean()
        msr = ((mean_rows - grand) ** 2).sum() * k / (n - 1)
        msc = ((mean_cols - grand) ** 2).sum() * n / (k - 1)
        msw = (((ratings - mean_rows[:, None]) ** 2).sum()) / (n * (k - 1))
        mse = (((ratings - mean_rows[:, None] - mean_cols[None, :] + grand) ** 2).sum()
               / ((n - 1) * (k - 1)))
        icc21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

    ax.set_xlabel("GPT-5.2 overall")
    ax.set_ylabel("Claude Sonnet overall")
    ax.set_title(f"(a) gpt52 vs sonnet: r={r:.3f}, ICC(2,1)={icc21:.3f}, n={len(x)}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Regression line to visualise systematic bias vs the y=x reference
    from numpy.polynomial.polynomial import polyfit as nppolyfit
    mask_finite = np.isfinite(x) & np.isfinite(y)
    if mask_finite.sum() >= 2:
        coeffs = np.polyfit(x[mask_finite], y[mask_finite], 1)
        xs_reg = np.linspace(0, 1, 100)
        ys_reg = np.polyval(coeffs, xs_reg)
        ax.plot(xs_reg, ys_reg, color=OI["vermilion"], linewidth=1.2, alpha=0.8,
                label=f"OLS fit (slope={coeffs[0]:.2f})")
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    # ICC annotation with interpretation
    icc_interp = "poor agreement; Sonnet systematically more lenient"
    ax.annotate(
        f"ICC(2,1) = {icc21:.3f} ({icc_interp})",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=7, va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )

    # (b) Spearman rho heatmap: 3 judges × 3 judges for overall (single heatmap — simpler and sharper)
    ax = axes[1]
    judges = ["gpt52", "claude_opus", "claude_sonnet"]
    # per-pattern means across queries for each judge
    means = (overall[overall.pattern.isin(BASE_PATTERNS)]
             .groupby(["pattern", "judge"])["overall"].mean()
             .unstack("judge"))
    means = means.reindex(columns=judges)
    rho = means.corr(method="spearman")
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(judges, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(judges)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    ax.set_title("(b) Spearman ρ of per-pattern means (overall)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 8. Judge agreement diagnostics")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig8_judge_agreement")


# ---------------------------------------------------------------------------
# S1 — full clustermap
# ---------------------------------------------------------------------------
def figS1_clustermap(overall_mean):
    df = overall_mean[overall_mean.pattern.isin(BASE_PATTERNS)]
    wide = df.pivot_table(index="pattern", columns="query_id", values="overall")
    wide = wide.reindex(index=BASE_PATTERNS).dropna(axis=1, how="any")
    wide.index = [PATTERN_LABEL[p] for p in wide.index]

    try:
        g = sns.clustermap(
            wide, cmap="viridis", vmin=0, vmax=1,
            figsize=(16, 5.5),
            row_cluster=True, col_cluster=True,
            dendrogram_ratio=(0.08, 0.12),
            cbar_kws={"label": "Overall (3-judge)"},
            xticklabels=False,
        )
        g.fig.suptitle("Figure S1. Pattern × query clustermap (3-judge avg overall)", y=1.02)
        for ext in ("pdf", "png"):
            g.savefig(OUT / f"figS1_clustermap.{ext}", bbox_inches="tight", dpi=300)
        plt.close(g.fig)
        print("  saved figS1_clustermap.pdf + .png")
    except Exception as e:
        print(f"  S1 clustermap failed: {e}")


# ---------------------------------------------------------------------------
# S2 — per-judge ranking bars
# ---------------------------------------------------------------------------
def figS2_per_judge(overall):
    df = overall[overall.pattern.isin(BASE_PATTERNS)].copy()
    judges = ["gpt52", "claude_opus", "claude_sonnet"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    rng = np.random.default_rng(321)
    for ax, j in zip(axes, judges):
        sub = df[df.judge == j]
        rows = []
        for p in BASE_PATTERNS:
            vals = sub[sub.pattern == p]["overall"].to_numpy()
            mu, lo, hi = boot_ci(vals, n=1000, rng=rng)
            rows.append({"label": PATTERN_LABEL[p], "mu": mu, "lo": lo, "hi": hi,
                         "family": family_of(p)})
        r = pd.DataFrame(rows).sort_values("mu", ascending=False).reset_index(drop=True)
        x = np.arange(len(r))
        colors = [FAMILY_COLORS[f] for f in r["family"]]
        ax.bar(x, r["mu"], yerr=[r["mu"] - r["lo"], r["hi"] - r["mu"]],
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.4, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(r["label"], rotation=45, fontsize=8)
        ax.set_title(f"Judge: {j}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Overall")
    # Shared legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Figure S2. Per-judge pattern ranking (sorted desc, 95% CI)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figS2_per_judge")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    runs, overall, scores, queries = load_data()
    overall_mean = judge_mean_overall(overall)
    dim_mean = judge_mean_by_dim(scores)

    registry: list[tuple[str, Callable]] = [
        ("fig1_pareto", lambda: fig1_pareto(runs, overall_mean, queries)),
        ("fig2_heatmap", lambda: fig2_heatmap(scores, dim_mean, overall_mean)),
        ("fig3_cd", lambda: fig3_cd(overall_mean)),
        ("fig4_ablations", lambda: fig4_ablations()),
        ("fig5_raincloud", lambda: fig5_raincloud(overall_mean)),
        ("fig6_stratified", lambda: fig6_stratified(overall_mean, queries)),
        ("fig7_process", lambda: fig7_process(runs, overall_mean)),
        ("fig8_judge_agreement", lambda: fig8_judge_agreement(overall, scores)),
        ("figS1_clustermap", lambda: figS1_clustermap(overall_mean)),
        ("figS2_per_judge", lambda: figS2_per_judge(overall)),
    ]

    for name, fn in registry:
        print(f"\n-> {name}")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAILED {name}: {e}")
            traceback.print_exc()

    print("\nAll figures attempted. Output:", OUT)


if __name__ == "__main__":
    main()
