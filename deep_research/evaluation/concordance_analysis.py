"""Evaluation method concordance analysis.

Formally analyzes agreement between different evaluation methods
(manual rubric, keyword matching, LLM judge) to quantify sensitivity
of pattern rankings to evaluation methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

from deep_research.evaluation.statistical_analysis import kendalls_w, kendalls_tau


@dataclass
class MethodRanking:
    """Ranking from a single evaluation method."""
    method_name: str
    pattern_names: list[str]    # in rank order (best first)
    scores: dict[str, float]    # pattern -> score


@dataclass
class ConcordanceReport:
    """Full concordance analysis results."""
    methods: list[MethodRanking]
    kendalls_w: float
    kendalls_w_p: float
    pairwise_tau: dict[str, float]  # "method_a vs method_b" -> tau
    pairwise_tau_p: dict[str, float]
    rank_changes: dict[str, dict[str, int]]  # pattern -> method -> rank
    most_stable_pattern: str    # pattern with smallest rank variance
    most_volatile_pattern: str  # pattern with largest rank variance
    summary: str


def analyze_concordance(
    rankings: list[MethodRanking],
) -> ConcordanceReport:
    """Analyze ranking concordance across evaluation methods.

    1. Compute Kendall's W (overall concordance)
    2. Compute pairwise Kendall's tau
    3. Identify most stable and volatile patterns
    4. Generate summary

    Args:
        rankings: list of MethodRanking objects, one per evaluation method.
            Each must rank the same set of patterns.

    Returns:
        ConcordanceReport with full analysis.

    Raises:
        ValueError: if fewer than 2 methods or patterns are inconsistent.
    """
    if len(rankings) < 2:
        raise ValueError("Need at least 2 evaluation methods for concordance analysis")

    # Collect all pattern names from the first method
    all_patterns = list(rankings[0].scores.keys())
    n_patterns = len(all_patterns)
    n_methods = len(rankings)

    if n_patterns < 2:
        raise ValueError("Need at least 2 patterns for concordance analysis")

    # Build ranking matrix: (n_methods, n_patterns) of ranks (1 = best)
    rank_matrix = np.zeros((n_methods, n_patterns), dtype=float)

    for i, method in enumerate(rankings):
        scores = np.array([method.scores[p] for p in all_patterns], dtype=float)
        # Rank descending: highest score -> rank 1
        rank_matrix[i] = stats.rankdata(-scores, method="average")

    # 1. Kendall's W
    w, w_p = kendalls_w(rank_matrix)

    # 2. Pairwise Kendall's tau
    pairwise_tau: dict[str, float] = {}
    pairwise_tau_p: dict[str, float] = {}

    for i, j in combinations(range(n_methods), 2):
        tau, tau_p = kendalls_tau(rank_matrix[i], rank_matrix[j])
        key = f"{rankings[i].method_name} vs {rankings[j].method_name}"
        pairwise_tau[key] = tau
        pairwise_tau_p[key] = tau_p

    # 3. Build rank_changes: pattern -> method -> rank
    rank_changes: dict[str, dict[str, int]] = {}
    for p_idx, pattern in enumerate(all_patterns):
        rank_changes[pattern] = {}
        for m_idx, method in enumerate(rankings):
            rank_changes[pattern][method.method_name] = int(rank_matrix[m_idx, p_idx])

    # 4. Find most stable and most volatile pattern
    rank_variances: dict[str, float] = {}
    for pattern in all_patterns:
        ranks_for_pattern = [rank_changes[pattern][m.method_name] for m in rankings]
        rank_variances[pattern] = float(np.var(ranks_for_pattern))

    most_stable = min(rank_variances, key=rank_variances.get)
    most_volatile = max(rank_variances, key=rank_variances.get)

    # 5. Generate summary
    summary_lines = [
        f"Concordance analysis across {n_methods} evaluation methods and {n_patterns} patterns.",
        f"Kendall's W = {w:.4f} (p = {w_p:.4f}).",
    ]

    if w > 0.7:
        summary_lines.append("Strong agreement among evaluation methods.")
    elif w > 0.4:
        summary_lines.append("Moderate agreement among evaluation methods.")
    else:
        summary_lines.append("Weak agreement among evaluation methods -- rankings are sensitive to methodology.")

    summary_lines.append(f"Most stable pattern: {most_stable} (rank variance = {rank_variances[most_stable]:.2f}).")
    summary_lines.append(f"Most volatile pattern: {most_volatile} (rank variance = {rank_variances[most_volatile]:.2f}).")

    summary = " ".join(summary_lines)

    return ConcordanceReport(
        methods=rankings,
        kendalls_w=w,
        kendalls_w_p=w_p,
        pairwise_tau=pairwise_tau,
        pairwise_tau_p=pairwise_tau_p,
        rank_changes=rank_changes,
        most_stable_pattern=most_stable,
        most_volatile_pattern=most_volatile,
        summary=summary,
    )


def identify_dimension_drivers(
    method_dimension_scores: dict[str, dict[str, dict[str, float]]],
    # method -> pattern -> dimension -> score
) -> dict[str, str]:
    """Identify which dimensions cause ranking changes between methods.

    For each method pair where rankings differ, find the dimension(s)
    with the largest score differential.

    Args:
        method_dimension_scores: nested dict of
            method_name -> pattern_name -> dimension_name -> score.

    Returns:
        dict mapping "method_a vs method_b" to the dimension name that
        drives the largest ranking disagreement between those methods.
    """
    methods = list(method_dimension_scores.keys())
    if len(methods) < 2:
        return {}

    # Get all patterns and dimensions from the first method
    first_method = methods[0]
    patterns = list(method_dimension_scores[first_method].keys())
    if not patterns:
        return {}
    dimensions = list(method_dimension_scores[first_method][patterns[0]].keys())

    drivers: dict[str, str] = {}

    for m_a, m_b in combinations(methods, 2):
        key = f"{m_a} vs {m_b}"

        max_disagreement = -1.0
        driver_dim = dimensions[0] if dimensions else ""

        for dim in dimensions:
            # Build per-pattern score differences for this dimension
            diffs = []
            for pattern in patterns:
                score_a = method_dimension_scores[m_a].get(pattern, {}).get(dim, 0.0)
                score_b = method_dimension_scores[m_b].get(pattern, {}).get(dim, 0.0)
                diffs.append(score_a - score_b)

            # The "disagreement" is the variance of the differences across patterns.
            # A dimension where all patterns have a similar offset between methods
            # doesn't change rankings; a dimension where the offset varies does.
            if len(diffs) >= 2:
                disagreement = float(np.var(diffs))
            else:
                disagreement = 0.0

            if disagreement > max_disagreement:
                max_disagreement = disagreement
                driver_dim = dim

        drivers[key] = driver_dim

    return drivers


def generate_concordance_report(result: ConcordanceReport) -> str:
    """Generate markdown concordance analysis.

    Args:
        result: ConcordanceReport from analyze_concordance().

    Returns:
        Markdown-formatted string with tables and analysis.
    """
    lines = ["# Evaluation Method Concordance Analysis\n"]

    # Overall concordance
    lines.append("## Overall Concordance\n")
    lines.append(f"- **Kendall's W**: {result.kendalls_w:.4f} (p = {result.kendalls_w_p:.4f})")

    w = result.kendalls_w
    if w > 0.7:
        interpretation = "Strong agreement"
    elif w > 0.4:
        interpretation = "Moderate agreement"
    else:
        interpretation = "Weak agreement"
    lines.append(f"- **Interpretation**: {interpretation}")
    lines.append("")

    # Pairwise correlations
    lines.append("## Pairwise Method Correlations\n")
    lines.append("| Method Pair | Kendall's tau | p-value |")
    lines.append("|---|---|---|")
    for pair_key in sorted(result.pairwise_tau.keys()):
        tau = result.pairwise_tau[pair_key]
        p = result.pairwise_tau_p[pair_key]
        lines.append(f"| {pair_key} | {tau:.4f} | {p:.4f} |")
    lines.append("")

    # Rank table
    lines.append("## Pattern Rankings by Method\n")

    # Collect method names
    method_names = [m.method_name for m in result.methods]
    header = "| Pattern | " + " | ".join(method_names) + " | Rank Variance |"
    separator = "|---" + "|---" * len(method_names) + "|---|"
    lines.append(header)
    lines.append(separator)

    all_patterns = list(result.rank_changes.keys())
    for pattern in sorted(all_patterns):
        ranks = result.rank_changes[pattern]
        rank_values = [ranks[m] for m in method_names]
        variance = float(np.var(rank_values))
        rank_strs = [str(ranks[m]) for m in method_names]
        lines.append(f"| {pattern} | " + " | ".join(rank_strs) + f" | {variance:.2f} |")
    lines.append("")

    # Stability
    lines.append("## Stability Analysis\n")
    lines.append(f"- **Most stable pattern**: {result.most_stable_pattern}")
    lines.append(f"- **Most volatile pattern**: {result.most_volatile_pattern}")
    lines.append("")

    # Summary
    lines.append("## Summary\n")
    lines.append(result.summary)
    lines.append("")

    return "\n".join(lines)
