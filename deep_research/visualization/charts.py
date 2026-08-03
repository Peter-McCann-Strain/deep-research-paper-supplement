"""Publication-quality visualization for research evaluation results.

Generates figures suitable for NeurIPS/ICML/ACL papers:
- Dimension heatmaps (patterns x dimensions)
- Critical difference diagrams (Demsar-style)
- Bootstrap CI forest plots
- Cost-quality scatter plots
- Radar/spider charts per pattern
- Performance profiles (Agarwal et al., 2021)
- Concordance heatmaps
- Ablation bar charts
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional


# Publication style settings
PAPER_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Pattern display names and colors
PATTERN_NAMES = {
    'p0_baseline': 'P0: Baseline',
    'p1_iterative_rag': 'P1: Iterative RAG',
    'p2_supervisor_parallel': 'P2: Supervisor',
    'p3_meridian': 'P3: MERIDIAN',
    'p4_perspective_storm': 'P4: STORM',
    'p5_hierarchical_wd': 'P5: Width-Depth',
    'p6_reactive_interleaved': 'P6: Reactive',
}

PATTERN_COLORS = {
    'p0_baseline': '#4C72B0',
    'p1_iterative_rag': '#DD8452',
    'p2_supervisor_parallel': '#55A868',
    'p3_meridian': '#C44E52',
    'p4_perspective_storm': '#8172B3',
    'p5_hierarchical_wd': '#937860',
    'p6_reactive_interleaved': '#CCB974',
}

DIMENSION_DISPLAY = {
    'factual_accuracy': 'Factual Acc.',
    'coverage': 'Coverage',
    'analytical_depth': 'Depth',
    'citation_quality': 'Citation',
    'organization': 'Organization',
    'instruction_following': 'Instr. Follow.',
    'attribution_quality': 'Attribution',
}


def _apply_style() -> None:
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PAPER_STYLE)
    sns.set_style("whitegrid")


def dimension_heatmap(
    dimension_scores: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Pattern Performance by Evaluation Dimension",
    figsize: tuple[float, float] = (10, 5),
    annotate: bool = True,
    cmap: str = "RdYlGn",
) -> None:
    """Create a patterns x dimensions heatmap with color scale.

    Args:
        dimension_scores: ``{pattern: {dimension: score}}``
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches ``(width, height)``.
        annotate: Show score values in cells.
        cmap: Colormap name.
    """
    _apply_style()

    patterns = sorted(dimension_scores.keys())
    dimensions = sorted(next(iter(dimension_scores.values())).keys())

    data = np.array([
        [dimension_scores[p].get(d, 0.0) for d in dimensions]
        for p in patterns
    ])

    fig, ax = plt.subplots(figsize=figsize)

    display_patterns = [PATTERN_NAMES.get(p, p) for p in patterns]
    display_dims = [DIMENSION_DISPLAY.get(d, d) for d in dimensions]

    sns.heatmap(
        data,
        xticklabels=display_dims,
        yticklabels=display_patterns,
        annot=annotate,
        fmt='.2f',
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def bootstrap_ci_plot(
    ci_data: list[dict],
    output_path: Path,
    title: str = "Overall Score with 95% Bootstrap Confidence Intervals",
    figsize: tuple[float, float] = (8, 5),
    metric_name: str = "Overall Score",
) -> None:
    """Forest plot of bootstrap confidence intervals per system.

    Shows mean with horizontal CI bars, sorted by mean score.

    Args:
        ci_data: List of dicts with keys ``system``, ``mean``,
            ``ci_lower``, ``ci_upper``.
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
        metric_name: X-axis label.
    """
    _apply_style()

    sorted_data = sorted(ci_data, key=lambda x: x['mean'], reverse=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = range(len(sorted_data))
    means = [d['mean'] for d in sorted_data]
    ci_lows = [d['mean'] - d['ci_lower'] for d in sorted_data]
    ci_highs = [d['ci_upper'] - d['mean'] for d in sorted_data]
    labels = [PATTERN_NAMES.get(d['system'], d['system']) for d in sorted_data]
    colors = [PATTERN_COLORS.get(d['system'], '#666666') for d in sorted_data]

    ax.barh(y_positions, means, xerr=[ci_lows, ci_highs],
            color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=0.5)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.set_xlim(0, 1)

    # Add value annotations
    for i, mean in enumerate(means):
        ax.text(mean + 0.02, i, f'{mean:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def radar_chart(
    dimension_scores: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Pattern Comparison Across Dimensions",
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """Radar/spider chart showing each pattern's dimension profile.

    Args:
        dimension_scores: ``{pattern: {dimension: score}}``
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
    """
    _apply_style()

    patterns = sorted(dimension_scores.keys())
    dimensions = sorted(next(iter(dimension_scores.values())).keys())
    n_dims = len(dimensions)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    for pattern in patterns:
        values = [dimension_scores[pattern].get(d, 0.0) for d in dimensions]
        values += values[:1]

        color = PATTERN_COLORS.get(pattern, '#666666')
        label = PATTERN_NAMES.get(pattern, pattern)
        ax.plot(angles, values, 'o-', linewidth=1.5, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [DIMENSION_DISPLAY.get(d, d) for d in dimensions],
    )
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def cost_quality_scatter(
    data: list[dict],
    output_path: Path,
    title: str = "Quality vs. Efficiency",
    figsize: tuple[float, float] = (8, 6),
    x_metric: str = "tokens",
) -> None:
    """Scatter plot of quality vs cost/latency with pattern labels.

    Args:
        data: List of dicts with keys ``pattern``, ``quality``, ``tokens``,
            ``latency_s``.
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
        x_metric: Which metric for x-axis: ``"tokens"`` or ``"latency_s"``.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    for d in data:
        pattern = d['pattern']
        x = d[x_metric]
        y = d['quality']
        color = PATTERN_COLORS.get(pattern, '#666666')
        label = PATTERN_NAMES.get(pattern, pattern)

        ax.scatter(x, y, c=color, s=150, edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(10, 5), fontsize=8)

    x_label = "Total Tokens" if x_metric == "tokens" else "Latency (seconds)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Overall Quality Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def critical_difference_diagram(
    avg_ranks: dict[str, float],
    n_tasks: int,
    cd: float,
    output_path: Path,
    title: str = "Critical Difference Diagram",
    figsize: tuple[float, float] = (10, 3),
) -> None:
    """Demsar-style critical difference diagram.

    Shows average ranks on a number line with bars connecting
    systems that are NOT significantly different.

    Args:
        avg_ranks: ``{pattern: avg_rank}`` (lower = better).
        n_tasks: Number of tasks used to compute ranks.
        cd: Critical difference value (from Nemenyi test).
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
    """
    _apply_style()

    sorted_systems = sorted(avg_ranks.items(), key=lambda x: x[1])
    n_systems = len(sorted_systems)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw the rank axis
    min_rank = 1
    max_rank = n_systems
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 1)

    # Draw tick marks
    for i in range(1, n_systems + 1):
        ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.3)

    # Place systems on top and bottom halves
    top_half = sorted_systems[:n_systems // 2]
    bottom_half = sorted_systems[n_systems // 2:]

    for _i, (name, rank) in enumerate(top_half):
        label = PATTERN_NAMES.get(name, name)
        ax.plot(rank, 0.7, 'ko', markersize=8)
        ax.annotate(label, (rank, 0.75), ha='center', fontsize=8, rotation=30)

    for _i, (name, rank) in enumerate(bottom_half):
        label = PATTERN_NAMES.get(name, name)
        ax.plot(rank, 0.3, 'ko', markersize=8)
        ax.annotate(label, (rank, 0.2), ha='center', fontsize=8, rotation=30)

    # Draw CD bar
    ax.plot([1, 1 + cd], [0.95, 0.95], 'k-', linewidth=2)
    ax.text(1 + cd / 2, 0.97, f'CD = {cd:.2f}', ha='center', fontsize=8)

    # Draw cliques (groups not significantly different)
    for i, (name_i, rank_i) in enumerate(sorted_systems):
        for j, (name_j, rank_j) in enumerate(sorted_systems[i + 1:], i + 1):
            if abs(rank_i - rank_j) < cd:
                y_pos = 0.5 - 0.05 * (i % 3)
                ax.plot([rank_i, rank_j], [y_pos, y_pos],
                        'k-', linewidth=3, alpha=0.3)

    ax.set_xlabel("Average Rank")
    ax.set_title(title)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def ablation_bar_chart(
    comparisons: list[dict],
    output_path: Path,
    title: str = "Ablation Study: Component Contributions",
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Grouped bar chart showing base vs ablated performance.

    Args:
        comparisons: List of dicts with keys ``component``, ``pattern``,
            ``base_mean``, ``ablated_mean``, ``significant`` (bool).
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    n = len(comparisons)
    x = np.arange(n)
    width = 0.35

    base_means = [c['base_mean'] for c in comparisons]
    ablated_means = [c['ablated_mean'] for c in comparisons]
    labels = [
        f"{c['component']}\n({PATTERN_NAMES.get(c['pattern'], c['pattern'])})"
        for c in comparisons
    ]

    ax.bar(x - width / 2, base_means, width, label='Full Pattern',
           color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, ablated_means, width, label='Ablated',
           color='#DD8452', edgecolor='black', linewidth=0.5)

    # Mark significant differences
    for i, c in enumerate(comparisons):
        if c.get('significant', False):
            max_y = max(base_means[i], ablated_means[i])
            ax.text(i, max_y + 0.02, '*', ha='center', fontsize=14, fontweight='bold')

    ax.set_ylabel('Overall Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def concordance_heatmap(
    tau_matrix: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Evaluation Method Concordance (Kendall's \u03c4)",
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """Method x Method concordance heatmap.

    Args:
        tau_matrix: ``{method_a: {method_b: tau_value}}``.
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
    """
    _apply_style()

    methods = sorted(tau_matrix.keys())
    n = len(methods)

    data = np.ones((n, n))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if m1 in tau_matrix and m2 in tau_matrix[m1]:
                data[i][j] = tau_matrix[m1][m2]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=methods,
        yticklabels=methods,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def performance_profile(
    score_matrix: np.ndarray,
    system_names: list[str],
    output_path: Path,
    title: str = "Performance Profiles",
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Agarwal et al. (2021) performance profiles.

    For each threshold tau, shows the fraction of tasks where each system
    scores above tau.  Better systems have curves that are higher and to
    the right.

    Args:
        score_matrix: ``(n_tasks, n_systems)`` array of scores.
        system_names: Name for each column.
        output_path: Where to save the figure.
        title: Figure title.
        figsize: Figure size in inches.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    thresholds = np.linspace(0, 1, 100)

    for j, name in enumerate(system_names):
        scores = score_matrix[:, j]
        fractions = [float(np.mean(scores >= t)) for t in thresholds]
        color = PATTERN_COLORS.get(name, f'C{j}')
        label = PATTERN_NAMES.get(name, name)
        ax.plot(thresholds, fractions, label=label, color=color, linewidth=2)

    ax.set_xlabel("Score Threshold (\u03c4)")
    ax.set_ylabel("Fraction of Tasks with Score \u2265 \u03c4")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_figures(
    results_dir: Path,
    output_dir: Path,
    dimension_scores: dict[str, dict[str, float]] | None = None,
    ci_data: list[dict] | None = None,
    cost_data: list[dict] | None = None,
    ablation_data: list[dict] | None = None,
    concordance_data: dict[str, dict[str, float]] | None = None,
    score_matrix: np.ndarray | None = None,
    system_names: list[str] | None = None,
    avg_ranks: dict[str, float] | None = None,
    n_tasks: int = 0,
    cd: float = 0.0,
) -> list[Path]:
    """Generate all available figures from the provided data.

    Only generates a figure when the necessary data is supplied.

    Args:
        results_dir: Directory containing raw results (for future use).
        output_dir: Directory to save generated figures.
        dimension_scores: For heatmap and radar chart.
        ci_data: For bootstrap CI forest plot.
        cost_data: For cost-quality scatter.
        ablation_data: For ablation bar chart.
        concordance_data: For concordance heatmap.
        score_matrix: For performance profiles.
        system_names: System names for performance profiles.
        avg_ranks: For critical difference diagram.
        n_tasks: Number of tasks (for CD diagram).
        cd: Critical difference value.

    Returns:
        List of paths to generated figure files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if dimension_scores:
        path = output_dir / "dimension_heatmap.png"
        dimension_heatmap(dimension_scores, path)
        generated.append(path)

        path = output_dir / "radar_chart.png"
        radar_chart(dimension_scores, path)
        generated.append(path)

    if ci_data:
        path = output_dir / "bootstrap_ci.png"
        bootstrap_ci_plot(ci_data, path)
        generated.append(path)

    if cost_data:
        path = output_dir / "cost_quality.png"
        cost_quality_scatter(cost_data, path)
        generated.append(path)

    if ablation_data:
        path = output_dir / "ablation_results.png"
        ablation_bar_chart(ablation_data, path)
        generated.append(path)

    if concordance_data:
        path = output_dir / "concordance_heatmap.png"
        concordance_heatmap(concordance_data, path)
        generated.append(path)

    if score_matrix is not None and system_names:
        path = output_dir / "performance_profiles.png"
        performance_profile(score_matrix, system_names, path)
        generated.append(path)

    if avg_ranks and n_tasks > 0 and cd > 0:
        path = output_dir / "critical_difference.png"
        critical_difference_diagram(avg_ranks, n_tasks, cd, path)
        generated.append(path)

    return generated
