"""Statistical analysis for multi-system comparison.

Implements:
- Friedman test + Iman-Davenport F correction (Iman & Davenport, 1980)
- Nemenyi post-hoc with critical difference (Demsar, JMLR 2006)
- Wilcoxon signed-rank pairwise with Holm-Bonferroni correction
- Bootstrap confidence intervals (BCa via scipy, percentile fallback)
- Cliff's Delta effect sizes (non-parametric)
- Holm-Bonferroni correction for multiple comparisons
- Kendall's W and tau for ranking concordance
- Interquartile mean (robust aggregate)
- Power analysis for Friedman test
- Bootstrap rank distributions with confidence intervals
- Stratified analysis by query strata
- Critical difference diagram generation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OmnibusResult:
    """Result of Friedman omnibus test."""

    statistic: float
    p_value: float
    is_significant: bool
    df: int
    n_systems: int
    n_tasks: int
    avg_ranks: dict[str, float]  # system name -> average rank
    iman_davenport_f: float = 0.0
    iman_davenport_p: float = 1.0


@dataclass
class PairwiseResult:
    """Result of a single pairwise comparison."""

    system_a: str
    system_b: str
    test_name: str
    statistic: float
    p_value_raw: float
    p_value_corrected: float
    is_significant: bool
    effect_size: float
    effect_size_label: str  # "negligible", "small", "medium", "large"
    ci_lower: float
    ci_upper: float
    mean_diff: float


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single system-metric pair."""

    system: str
    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    iqm: float
    iqm_ci_lower: float
    iqm_ci_upper: float
    std: float
    n_samples: int
    n_bootstrap: int


@dataclass
class ConcordanceResult:
    """Concordance analysis across multiple evaluation methods."""

    kendalls_w: float
    kendalls_w_p_value: float
    pairwise_tau: dict[tuple[str, str], tuple[float, float]]  # pair -> (tau, p)
    rankings_per_method: dict[str, list[str]]  # method -> ranked system names


@dataclass
class FullAnalysisResult:
    """Complete statistical analysis results."""

    omnibus: OmnibusResult
    pairwise: list[PairwiseResult]
    bootstrap_cis: list[BootstrapCI]
    per_dimension_omnibus: dict[str, OmnibusResult]
    per_dimension_pairwise: dict[str, list[PairwiseResult]]
    summary_markdown: str
    nemenyi_pairwise: list[PairwiseResult] = field(default_factory=list)
    per_dimension_nemenyi_pairwise: dict[str, list[PairwiseResult]] = field(default_factory=dict)
    critical_difference: float = 0.0
    rank_distributions: list[dict] = field(default_factory=list)
    stratified_results: dict = field(default_factory=dict)  # {strata_key: {stratum: FullAnalysisResult}}


# ---------------------------------------------------------------------------
# Nemenyi critical difference (Fix #12)
# ---------------------------------------------------------------------------

# q_alpha values for the Studentized Range distribution at alpha=0.05
# for k groups and df=infinity (Nemenyi test uses this).
# Sources: Table values from Demsar (2006) and standard statistical tables.
_QALPHA_005 = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Compute the Nemenyi critical difference.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))

    where q_alpha is the critical value of the Studentized Range distribution
    for k groups at the given alpha level.

    Args:
        k: number of systems (groups).
        n: number of tasks (observations per system).
        alpha: significance level (default 0.05).

    Returns:
        The critical difference value. Pairs of systems whose average rank
        difference exceeds CD are significantly different.
    """
    if k < 2:
        raise ValueError(f"Need at least 2 systems, got {k}")
    if n < 1:
        raise ValueError(f"Need at least 1 task, got {n}")

    # Try scipy's studentized_range if available
    q_alpha = None
    if alpha == 0.05 and k in _QALPHA_005:
        q_alpha = _QALPHA_005[k]
    else:
        try:
            q_alpha = float(stats.studentized_range.ppf(1 - alpha, k, np.inf))
        except (AttributeError, Exception):
            if k in _QALPHA_005 and alpha == 0.05:
                q_alpha = _QALPHA_005[k]
            else:
                raise ValueError(
                    f"Cannot compute q_alpha for k={k}, alpha={alpha}. "
                    f"Lookup table only available for alpha=0.05 and k in {sorted(_QALPHA_005.keys())}."
                )

    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    return cd


# ---------------------------------------------------------------------------
# Friedman omnibus test
# ---------------------------------------------------------------------------


def friedman_test(
    score_matrix: np.ndarray,
    system_names: list[str],
    alpha: float = 0.05,
) -> OmnibusResult:
    """Friedman test for comparing k related samples.

    The Friedman test is a non-parametric alternative to repeated-measures
    ANOVA.  It ranks the systems within each task (row) and tests whether
    the average ranks differ significantly.

    Also computes the Iman-Davenport F-statistic correction (Iman &
    Davenport, 1980) which has better statistical properties than the
    chi-squared approximation.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.  Each row is one task,
            each column is one system's score on that task.
        system_names: names for each column (length ``k``).
        alpha: significance level (default 0.05).

    Returns:
        :class:`OmnibusResult` with test statistic, p-value, significance flag,
        average ranks per system, and Iman-Davenport correction.

    Raises:
        ValueError: if inputs are invalid (wrong shape, mismatched names, etc.).
    """
    score_matrix = np.asarray(score_matrix, dtype=float)

    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D, got {score_matrix.ndim}-D"
        )

    n_tasks, k_systems = score_matrix.shape

    if k_systems < 2:
        raise ValueError(
            f"Need at least 2 systems to compare, got {k_systems}"
        )
    if n_tasks < 2:
        raise ValueError(
            f"Need at least 2 tasks (rows) for Friedman test, got {n_tasks}"
        )
    if len(system_names) != k_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != "
            f"number of columns ({k_systems})"
        )

    # Compute average ranks (rank *within* each task, higher score = lower rank number
    # by convention in Demsar, but scipy ranks ascending; we rank *descending*
    # so that the best system gets rank 1).
    ranks = np.zeros_like(score_matrix, dtype=float)
    for i in range(n_tasks):
        # rankdata ranks ascending; we want descending, so negate.
        ranks[i] = stats.rankdata(-score_matrix[i], method="average")

    avg_ranks = {
        name: float(ranks[:, j].mean()) for j, name in enumerate(system_names)
    }

    iman_f = 0.0
    iman_p = 1.0

    if k_systems >= 3:
        # scipy.stats.friedmanchisquare requires >= 3 groups
        columns = [score_matrix[:, i] for i in range(k_systems)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            statistic, p_value = stats.friedmanchisquare(*columns)
        # When all scores are identical within each row, scipy returns NaN.
        # Treat this as no significant difference.
        if np.isnan(statistic):
            statistic = 0.0
        if np.isnan(p_value):
            p_value = 1.0

        # Iman-Davenport F-statistic (Iman & Davenport, 1980)
        chi2_f = float(statistic)
        denom = n_tasks * (k_systems - 1) - chi2_f
        if denom > 0:
            iman_f = ((n_tasks - 1) * chi2_f) / denom
            df1 = k_systems - 1
            df2 = (k_systems - 1) * (n_tasks - 1)
            iman_p = float(1.0 - stats.f.cdf(iman_f, df1, df2))
        else:
            iman_f = float('inf')
            iman_p = 0.0
    else:
        # For k == 2, fall back to Wilcoxon signed-rank test which is
        # equivalent to the Friedman test for two treatments.
        diff = score_matrix[:, 0] - score_matrix[:, 1]
        if np.all(diff == 0):
            statistic = 0.0
            p_value = 1.0
        else:
            stat_result = stats.wilcoxon(
                score_matrix[:, 0], score_matrix[:, 1],
                zero_method="pratt", alternative="two-sided",
            )
            statistic = stat_result.statistic
            p_value = stat_result.pvalue

    return OmnibusResult(
        statistic=float(statistic),
        p_value=float(p_value),
        is_significant=bool(p_value < alpha),
        df=k_systems - 1,
        n_systems=k_systems,
        n_tasks=n_tasks,
        avg_ranks=avg_ranks,
        iman_davenport_f=float(iman_f),
        iman_davenport_p=float(iman_p),
    )


# ---------------------------------------------------------------------------
# Post-hoc tests
# ---------------------------------------------------------------------------


def nemenyi_posthoc(
    score_matrix: np.ndarray,
    system_names: list[str],
    alpha: float = 0.05,
) -> list[PairwiseResult]:
    """Nemenyi post-hoc test after significant Friedman.

    Uses ``scikit-posthocs`` for the test, computes Cliff's Delta for effect
    sizes.  Nemenyi already controls the family-wise error rate (FWER), so
    no additional Holm-Bonferroni correction is applied (that would be a
    statistically incorrect double correction).

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        alpha: family-wise significance level.

    Returns:
        List of :class:`PairwiseResult` for every unique pair of systems.
    """
    import scikit_posthocs as sp

    score_matrix = np.asarray(score_matrix, dtype=float)
    n_tasks, k_systems = score_matrix.shape

    if len(system_names) != k_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != "
            f"columns ({k_systems})"
        )

    # scikit-posthocs wants (tasks x systems) matrix with rows=blocks, cols=groups
    p_matrix = sp.posthoc_nemenyi_friedman(score_matrix)

    # p_matrix is a DataFrame indexed 0..k-1 by default
    results: list[PairwiseResult] = []
    for i, j in combinations(range(k_systems), 2):
        raw_p = float(p_matrix.iloc[i, j])

        scores_a = score_matrix[:, i]
        scores_b = score_matrix[:, j]

        delta, delta_label = cliffs_delta(scores_a, scores_b)
        diff = scores_a - scores_b

        # Bootstrap CI on the mean difference
        mean_diff = float(diff.mean())
        if len(diff) >= 2:
            _, ci_lo, ci_hi = bootstrap_confidence_interval(
                diff, n_bootstrap=2000, alpha=alpha, random_state=42
            )
        else:
            ci_lo, ci_hi = mean_diff, mean_diff

        # Nemenyi already controls FWER -- no Holm-Bonferroni needed.
        # Use raw p-value directly as the corrected value.
        results.append(
            PairwiseResult(
                system_a=system_names[i],
                system_b=system_names[j],
                test_name="nemenyi",
                statistic=np.nan,  # Nemenyi doesn't expose a single statistic per pair
                p_value_raw=raw_p,
                p_value_corrected=raw_p,
                is_significant=bool(raw_p < alpha),
                effect_size=delta,
                effect_size_label=delta_label,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                mean_diff=mean_diff,
            )
        )

    return results


def wilcoxon_signed_rank_pairwise(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
) -> PairwiseResult:
    """Wilcoxon signed-rank test for a single pair of systems.

    This is a non-parametric test for paired samples.  It tests whether
    the distribution of differences between the two systems' scores is
    symmetric about zero.

    Args:
        scores_a: scores from system A (one per task).
        scores_b: scores from system B (one per task).
        name_a: display name for system A.
        name_b: display name for system B.
        alpha: significance level.

    Returns:
        :class:`PairwiseResult` with test results and Cliff's Delta.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"Score arrays must have the same shape: "
            f"{scores_a.shape} vs {scores_b.shape}"
        )
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 paired observations")

    diff = scores_a - scores_b

    # If all differences are zero, Wilcoxon cannot be computed.
    if np.all(diff == 0):
        return PairwiseResult(
            system_a=name_a,
            system_b=name_b,
            test_name="wilcoxon",
            statistic=0.0,
            p_value_raw=1.0,
            p_value_corrected=1.0,
            is_significant=False,
            effect_size=0.0,
            effect_size_label="negligible",
            ci_lower=0.0,
            ci_upper=0.0,
            mean_diff=0.0,
        )

    # zero_method="pratt" includes zeros in the ranking (conservative)
    stat, p_value = stats.wilcoxon(
        scores_a, scores_b, zero_method="pratt", alternative="two-sided"
    )

    delta, delta_label = cliffs_delta(scores_a, scores_b)
    mean_diff = float(diff.mean())

    # Bootstrap CI on the mean difference
    _, ci_lo, ci_hi = bootstrap_confidence_interval(
        diff, n_bootstrap=2000, alpha=alpha, random_state=42
    )

    return PairwiseResult(
        system_a=name_a,
        system_b=name_b,
        test_name="wilcoxon",
        statistic=float(stat),
        p_value_raw=float(p_value),
        p_value_corrected=float(p_value),  # single test, no correction needed
        is_significant=bool(p_value < alpha),
        effect_size=delta,
        effect_size_label=delta_label,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        mean_diff=mean_diff,
    )


def wilcoxon_pairwise_all(
    score_matrix: np.ndarray,
    system_names: list[str],
    alpha: float = 0.05,
) -> list[PairwiseResult]:
    """Wilcoxon signed-rank tests for all pairs with Holm-Bonferroni correction.

    This is the recommended primary post-hoc test.  It runs a Wilcoxon
    signed-rank test for every pair of systems, collects all raw p-values,
    and applies Holm-Bonferroni correction as a batch to control FWER.
    Also computes Cliff's Delta and bootstrap CI for each pair.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        alpha: family-wise significance level.

    Returns:
        List of :class:`PairwiseResult` for every unique pair of systems,
        with Holm-Bonferroni corrected p-values.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    n_tasks, k_systems = score_matrix.shape

    if len(system_names) != k_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != "
            f"columns ({k_systems})"
        )

    # Collect all raw pairwise results
    pairs: list[tuple[int, int]] = list(combinations(range(k_systems), 2))
    raw_results: list[PairwiseResult] = []

    for i, j in pairs:
        result = wilcoxon_signed_rank_pairwise(
            score_matrix[:, i], score_matrix[:, j],
            system_names[i], system_names[j],
            alpha=alpha,
        )
        raw_results.append(result)

    # Collect all raw p-values and apply Holm-Bonferroni as a batch
    raw_p_values = [r.p_value_raw for r in raw_results]
    corrected_p_values = holm_bonferroni(raw_p_values, alpha=alpha)

    # Update each result with corrected p-values and significance
    final_results: list[PairwiseResult] = []
    for idx, r in enumerate(raw_results):
        final_results.append(
            PairwiseResult(
                system_a=r.system_a,
                system_b=r.system_b,
                test_name="wilcoxon_holm",
                statistic=r.statistic,
                p_value_raw=r.p_value_raw,
                p_value_corrected=corrected_p_values[idx],
                is_significant=bool(corrected_p_values[idx] < alpha),
                effect_size=r.effect_size,
                effect_size_label=r.effect_size_label,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                mean_diff=r.mean_diff,
            )
        )

    return final_results


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    The procedure sorts p-values in ascending order and multiplies each
    by ``(m - rank + 1)`` where ``m`` is the total number of tests.
    Corrected p-values are enforced to be monotonically non-decreasing
    and capped at 1.0.

    Args:
        p_values: raw (uncorrected) p-values.
        alpha: family-wise error rate (used only for documentation;
            the caller decides significance).

    Returns:
        List of corrected p-values in the *original* order.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return list(p_values)

    # Sort indices by ascending p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Holm step-down: multiply p_(i) by (m - i) where i is 0-based rank
    corrected = np.empty(m, dtype=float)
    for i in range(m):
        corrected[i] = sorted_p[i] * (m - i)

    # Enforce monotonically non-decreasing (step-down enforcement)
    for i in range(1, m):
        corrected[i] = max(corrected[i], corrected[i - 1])

    # Cap at 1.0
    corrected = np.minimum(corrected, 1.0)

    # Map back to original order
    result = np.empty(m, dtype=float)
    result[sorted_indices] = corrected

    return result.tolist()


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Attempts BCa (bias-corrected and accelerated) method via
    ``scipy.stats.bootstrap`` first, falling back to the percentile
    method if scipy's bootstrap is not available.

    Args:
        scores: 1-D array of scores.
        n_bootstrap: number of bootstrap iterations.
        alpha: significance level (e.g. 0.05 for 95% CI).
        random_state: seed for reproducibility.

    Returns:
        ``(mean, ci_lower, ci_upper)``
    """
    scores = np.asarray(scores, dtype=float).ravel()

    if len(scores) == 0:
        raise ValueError("Cannot compute bootstrap CI on empty array")

    if len(scores) == 1:
        val = float(scores[0])
        return val, val, val

    mean_val = float(scores.mean())

    # Try BCa via scipy.stats.bootstrap
    try:
        from scipy.stats import bootstrap as scipy_bootstrap

        def _mean_statistic(x, axis):
            return np.mean(x, axis=axis)

        res = scipy_bootstrap(
            (scores,), _mean_statistic, n_resamples=n_bootstrap,
            confidence_level=1 - alpha, method='BCa',
            random_state=np.random.default_rng(random_state),
        )
        ci_lo = float(res.confidence_interval.low)
        ci_hi = float(res.confidence_interval.high)
        if np.isnan(ci_lo) or np.isnan(ci_hi):
            raise ValueError("BCa returned NaN — falling back to percentile")
        return mean_val, ci_lo, ci_hi
    except (ImportError, AttributeError, Exception):
        pass

    # Fallback: percentile method
    rng = np.random.RandomState(random_state)
    n = len(scores)
    boot_means = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[b] = sample.mean()

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean_val, lo, hi


# ---------------------------------------------------------------------------
# Interquartile mean
# ---------------------------------------------------------------------------


def interquartile_mean(scores: np.ndarray) -> float:
    """Interquartile mean -- robust aggregate per Agarwal et al. (2021).

    Trims the bottom 25% and top 25% of scores, and averages the middle 50%.
    Uses linear interpolation at the quartile boundaries for fractional
    trim indices (equivalent to ``scipy.stats.trim_mean`` with
    ``proportiontocut=0.25``).

    Args:
        scores: 1-D array of scores.

    Returns:
        The interquartile mean.

    Raises:
        ValueError: if the array is empty.
    """
    scores = np.asarray(scores, dtype=float).ravel()

    if len(scores) == 0:
        raise ValueError("Cannot compute IQM on empty array")

    return float(stats.trim_mean(scores, proportiontocut=0.25))


def interquartile_mean_bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for interquartile mean.

    Attempts BCa method via ``scipy.stats.bootstrap`` first, falling back
    to the percentile method if not available.

    Args:
        scores: 1-D array of scores.
        n_bootstrap: number of bootstrap iterations.
        alpha: significance level.
        random_state: seed for reproducibility.

    Returns:
        ``(iqm, ci_lower, ci_upper)``
    """
    scores = np.asarray(scores, dtype=float).ravel()

    if len(scores) == 0:
        raise ValueError("Cannot compute IQM bootstrap CI on empty array")

    if len(scores) == 1:
        val = float(scores[0])
        return val, val, val

    iqm_val = interquartile_mean(scores)

    # Try BCa via scipy.stats.bootstrap
    try:
        from scipy.stats import bootstrap as scipy_bootstrap

        def _iqm_statistic(x, axis):
            # trim_mean works along axis for 2-D arrays
            return stats.trim_mean(x, proportiontocut=0.25, axis=axis)

        res = scipy_bootstrap(
            (scores,), _iqm_statistic, n_resamples=n_bootstrap,
            confidence_level=1 - alpha, method='BCa',
            random_state=np.random.default_rng(random_state),
        )
        ci_lo = float(res.confidence_interval.low)
        ci_hi = float(res.confidence_interval.high)
        if np.isnan(ci_lo) or np.isnan(ci_hi):
            raise ValueError("BCa returned NaN — falling back to percentile")
        return iqm_val, ci_lo, ci_hi
    except (ImportError, AttributeError, Exception):
        pass

    # Fallback: percentile method
    rng = np.random.RandomState(random_state)
    n = len(scores)
    boot_iqms = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_iqms[b] = stats.trim_mean(sample, proportiontocut=0.25)

    lo = float(np.percentile(boot_iqms, 100 * alpha / 2))
    hi = float(np.percentile(boot_iqms, 100 * (1 - alpha / 2)))

    return iqm_val, lo, hi


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Cliff's Delta non-parametric effect size.

    Cliff's Delta measures the degree of overlap between two distributions.
    It equals the probability that a randomly chosen value from *x* is
    greater than a randomly chosen value from *y*, minus the probability
    of the reverse.

    Args:
        x: 1-D array of scores from group A.
        y: 1-D array of scores from group B.

    Returns:
        ``(delta, label)`` where *label* is one of:

        - ``"negligible"`` if ``|d| < 0.147``
        - ``"small"`` if ``0.147 <= |d| < 0.33``
        - ``"medium"`` if ``0.33 <= |d| < 0.474``
        - ``"large"`` if ``|d| >= 0.474``

        Thresholds from Romano et al. (2006).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if len(x) == 0 or len(y) == 0:
        raise ValueError("Both arrays must be non-empty")

    n_x, n_y = len(x), len(y)

    # Count #(x_i > y_j) and #(x_i < y_j) over all pairs
    n_more = 0  # x > y
    n_less = 0  # x < y
    for xi in x:
        for yj in y:
            if xi > yj:
                n_more += 1
            elif xi < yj:
                n_less += 1

    delta = (n_more - n_less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        label = "negligible"
    elif abs_d < 0.33:
        label = "small"
    elif abs_d < 0.474:
        label = "medium"
    else:
        label = "large"

    return float(delta), label


# ---------------------------------------------------------------------------
# Concordance measures
# ---------------------------------------------------------------------------


def kendalls_w(rankings_matrix: np.ndarray) -> tuple[float, float]:
    """Kendall's W coefficient of concordance.

    Measures agreement among *m* raters who have each ranked *n* items.
    W = 1 indicates perfect agreement; W = 0 indicates no agreement.

    The significance is tested via the chi-squared approximation:
    ``chi2 = m * (n - 1) * W``, with ``df = n - 1``.

    Args:
        rankings_matrix: shape ``(m_raters, n_items)``.  Each row is one
            rater's ranking of the items (e.g. ``[3, 1, 2]`` means item 0
            is ranked 3rd, item 1 is ranked 1st, etc.).

    Returns:
        ``(W, p_value)`` where *p_value* is from the chi-squared test.

    Raises:
        ValueError: if the matrix has fewer than 2 raters or 2 items.
    """
    rankings_matrix = np.asarray(rankings_matrix, dtype=float)

    if rankings_matrix.ndim != 2:
        raise ValueError(
            f"rankings_matrix must be 2-D, got {rankings_matrix.ndim}-D"
        )

    m, n = rankings_matrix.shape  # m raters, n items

    if m < 2:
        raise ValueError(f"Need at least 2 raters, got {m}")
    if n < 2:
        raise ValueError(f"Need at least 2 items, got {n}")

    # Sum of ranks for each item (column sums)
    rank_sums = rankings_matrix.sum(axis=0)

    # Mean of rank sums
    mean_rank_sum = rank_sums.mean()

    # S = sum of squared deviations of rank sums from the mean
    s = float(np.sum((rank_sums - mean_rank_sum) ** 2))

    # Maximum possible S: m^2 * (n^3 - n) / 12
    s_max = (m ** 2) * (n ** 3 - n) / 12

    if s_max == 0:
        # Degenerate case: only 1 item or all identical
        return 0.0, 1.0

    w = s / s_max

    # Chi-squared approximation for significance
    chi2 = m * (n - 1) * w
    df = n - 1
    p_value = float(1.0 - stats.chi2.cdf(chi2, df))

    return float(w), p_value


def kendalls_tau(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
) -> tuple[float, float]:
    """Kendall's tau-b rank correlation with p-value.

    Args:
        ranking_a: 1-D ranking from rater A.
        ranking_b: 1-D ranking from rater B.

    Returns:
        ``(tau, p_value)``
    """
    ranking_a = np.asarray(ranking_a, dtype=float).ravel()
    ranking_b = np.asarray(ranking_b, dtype=float).ravel()

    if len(ranking_a) != len(ranking_b):
        raise ValueError(
            f"Rankings must have the same length: "
            f"{len(ranking_a)} vs {len(ranking_b)}"
        )
    if len(ranking_a) < 2:
        raise ValueError("Need at least 2 items to compute tau")

    tau, p_value = stats.kendalltau(ranking_a, ranking_b)
    return float(tau), float(p_value)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def compute_all_bootstrap_cis(
    score_matrix: np.ndarray,
    system_names: list[str],
    metric_name: str = "overall",
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> list[BootstrapCI]:
    """Compute bootstrap CIs for all systems.

    Uses BCa bootstrap via ``scipy.stats.bootstrap`` when available,
    falling back to the percentile method otherwise.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        metric_name: label for the metric being analysed.
        n_bootstrap: bootstrap iterations per system.
        alpha: significance level.
        random_state: base random seed (incremented per system for variety).

    Returns:
        List of :class:`BootstrapCI`, one per system.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)

    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D, got {score_matrix.ndim}-D"
        )

    n_tasks, k_systems = score_matrix.shape

    if len(system_names) != k_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != "
            f"columns ({k_systems})"
        )

    results: list[BootstrapCI] = []

    for j, name in enumerate(system_names):
        scores = score_matrix[:, j]
        seed = random_state + j

        mean, ci_lo, ci_hi = bootstrap_confidence_interval(
            scores, n_bootstrap=n_bootstrap, alpha=alpha, random_state=seed
        )
        iqm_val, iqm_lo, iqm_hi = interquartile_mean_bootstrap_ci(
            scores, n_bootstrap=n_bootstrap, alpha=alpha, random_state=seed
        )

        results.append(
            BootstrapCI(
                system=name,
                metric=metric_name,
                mean=mean,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                iqm=iqm_val,
                iqm_ci_lower=iqm_lo,
                iqm_ci_upper=iqm_hi,
                std=float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                n_samples=len(scores),
                n_bootstrap=n_bootstrap,
            )
        )

    return results


def ranking_concordance(
    rankings: dict[str, list[float]],
    method_names: list[str],
    system_names: list[str],
) -> ConcordanceResult:
    """Analyse concordance between multiple evaluation methods.

    Each evaluation method produces a score per system; we convert scores
    to ranks and compute Kendall's W across all methods, and pairwise
    Kendall's tau between each pair.

    Args:
        rankings: ``{method_name: [score_for_system_0, ..., score_for_system_k]}``.
            Higher scores are better.
        method_names: ordered list of method names (keys into *rankings*).
        system_names: ordered list of system names.

    Returns:
        :class:`ConcordanceResult`.
    """
    if len(method_names) < 2:
        raise ValueError("Need at least 2 methods for concordance analysis")
    if len(system_names) < 2:
        raise ValueError("Need at least 2 systems for concordance analysis")

    n_methods = len(method_names)
    n_systems = len(system_names)

    # Build rankings matrix: (n_methods, n_systems) of *ranks* (1 = best)
    rank_matrix = np.zeros((n_methods, n_systems), dtype=float)
    rankings_per_method: dict[str, list[str]] = {}

    for i, method in enumerate(method_names):
        scores = np.array(rankings[method], dtype=float)
        if len(scores) != n_systems:
            raise ValueError(
                f"Method '{method}' has {len(scores)} scores, "
                f"expected {n_systems}"
            )
        # Rank descending: highest score -> rank 1
        rank_matrix[i] = stats.rankdata(-scores, method="average")

        # Build ranked system name list
        order = np.argsort(-scores)
        rankings_per_method[method] = [system_names[idx] for idx in order]

    # Kendall's W
    w, w_p = kendalls_w(rank_matrix)

    # Pairwise Kendall's tau
    pairwise_tau: dict[tuple[str, str], tuple[float, float]] = {}
    for i, j in combinations(range(n_methods), 2):
        tau, tau_p = kendalls_tau(rank_matrix[i], rank_matrix[j])
        pair_key = (method_names[i], method_names[j])
        pairwise_tau[pair_key] = (tau, tau_p)

    return ConcordanceResult(
        kendalls_w=w,
        kendalls_w_p_value=w_p,
        pairwise_tau=pairwise_tau,
        rankings_per_method=rankings_per_method,
    )


# ---------------------------------------------------------------------------
# Power analysis (Fix #15)
# ---------------------------------------------------------------------------


def power_analysis(
    n_queries: int,
    k_systems: int,
    alpha: float = 0.05,
) -> dict:
    """Compute detectable effect sizes given n and k.

    Uses the relationship chi2_F = n*(k-1)*W where W is Kendall's W,
    and the non-central chi-squared distribution to find the minimum
    W detectable at 80% power.

    Args:
        n_queries: number of queries (tasks/observations).
        k_systems: number of systems being compared.
        alpha: significance level.

    Returns:
        dict with:
        - min_detectable_w: minimum Kendall's W detectable at 80% power
        - friedman_df: degrees of freedom for Friedman test
        - n_pairwise_comparisons: number of pairwise tests C(k, 2)
        - nemenyi_cd: critical difference for Nemenyi test
        - min_n_for_small_effect: sample size needed to detect W=0.1
    """
    from scipy.optimize import brentq
    from scipy.stats import ncx2

    df = k_systems - 1
    crit_value = float(stats.chi2.ppf(1 - alpha, df))
    target_power = 0.80

    # Find minimum noncentrality parameter lambda such that
    # P(ncx2(df, lambda) > crit_value) >= 0.80
    def _power_at_lambda(lam):
        # Power = P(X > crit_value) where X ~ ncx2(df, lam)
        return 1.0 - ncx2.cdf(crit_value, df, lam) - target_power

    # Search for lambda in a reasonable range
    try:
        min_lambda = brentq(_power_at_lambda, 0.01, 500.0, xtol=1e-8)
    except ValueError:
        # If the root is not in the interval, use a wider range
        try:
            min_lambda = brentq(_power_at_lambda, 1e-6, 5000.0, xtol=1e-8)
        except ValueError:
            min_lambda = float('nan')

    # Convert noncentrality to Kendall's W:
    # chi2_F = n*(k-1)*W, and under H1 the test stat is approximately
    # non-central chi2 with noncentrality = n*(k-1)*W
    # So min_W = min_lambda / (n * (k-1))
    if np.isfinite(min_lambda):
        min_w = min_lambda / (n_queries * (k_systems - 1))
    else:
        min_w = float('nan')

    # Compute Nemenyi CD
    try:
        cd = nemenyi_critical_difference(k_systems, n_queries, alpha=alpha)
    except (ValueError, Exception):
        cd = float('nan')

    # Compute minimum n for detecting W=0.1 at 80% power
    w_target = 0.1
    def _n_power(n_trial):
        lam_trial = n_trial * (k_systems - 1) * w_target
        return 1.0 - ncx2.cdf(crit_value, df, lam_trial) - target_power

    try:
        min_n = brentq(_n_power, 2.0, 10000.0, xtol=0.5)
        min_n = int(math.ceil(min_n))
    except ValueError:
        min_n = float('nan')

    n_pairwise = k_systems * (k_systems - 1) // 2

    return {
        "min_detectable_w": float(min_w),
        "friedman_df": df,
        "n_pairwise_comparisons": n_pairwise,
        "nemenyi_cd": float(cd),
        "min_n_for_small_effect": min_n,
    }


# ---------------------------------------------------------------------------
# Bootstrap rank distributions (Fix #19)
# ---------------------------------------------------------------------------


def bootstrap_rank_distribution(
    score_matrix: np.ndarray,
    system_names: list[str],
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> list[dict]:
    """Bootstrap the rank distribution for each system.

    Resamples rows (queries) with replacement, computes mean score per
    system in each bootstrap sample, ranks systems, and counts how often
    each system gets each rank.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        n_bootstrap: number of bootstrap iterations.
        random_state: seed for reproducibility.

    Returns:
        List of dicts (one per system):
        ``{system, mean_rank, rank_ci_lower, rank_ci_upper,
          prob_rank_1, rank_distribution}``
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    n_tasks, k_systems = score_matrix.shape

    if len(system_names) != k_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != "
            f"columns ({k_systems})"
        )

    rng = np.random.RandomState(random_state)

    # rank_counts[j][r] = number of times system j got rank r (1-indexed)
    rank_counts = np.zeros((k_systems, k_systems), dtype=int)
    boot_ranks = np.zeros((n_bootstrap, k_systems), dtype=float)

    for b in range(n_bootstrap):
        # Resample rows with replacement
        indices = rng.choice(n_tasks, size=n_tasks, replace=True)
        boot_sample = score_matrix[indices]

        # Compute mean score per system
        mean_scores = boot_sample.mean(axis=0)

        # Rank systems (higher score = rank 1)
        ranks = stats.rankdata(-mean_scores, method="average")
        boot_ranks[b] = ranks

        # Count rank assignments (convert to 0-indexed for counting)
        for j in range(k_systems):
            rank_idx = int(ranks[j]) - 1  # 0-indexed
            if 0 <= rank_idx < k_systems:
                rank_counts[j, rank_idx] += 1

    results: list[dict] = []
    for j in range(k_systems):
        mean_rank = float(boot_ranks[:, j].mean())
        ci_lo = float(np.percentile(boot_ranks[:, j], 2.5))
        ci_hi = float(np.percentile(boot_ranks[:, j], 97.5))
        prob_rank_1 = float(rank_counts[j, 0]) / n_bootstrap

        # Full rank distribution as probabilities
        rank_dist = (rank_counts[j].astype(float) / n_bootstrap).tolist()

        results.append({
            "system": system_names[j],
            "mean_rank": mean_rank,
            "rank_ci_lower": ci_lo,
            "rank_ci_upper": ci_hi,
            "prob_rank_1": prob_rank_1,
            "rank_distribution": rank_dist,
        })

    return results


# ---------------------------------------------------------------------------
# Stratified analysis (Fix #11)
# ---------------------------------------------------------------------------


def run_stratified_analysis(
    score_matrix: np.ndarray,
    system_names: list[str],
    query_strata: list[str],
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> dict[str, FullAnalysisResult]:
    """Run full analysis per stratum. Skips strata with < 5 queries.

    Groups rows by stratum label, runs ``run_full_analysis()`` on each
    sub-matrix, and returns a dict mapping stratum name to result.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        query_strata: stratum label per query (length ``n_tasks``).
        alpha: family-wise significance level.
        n_bootstrap: bootstrap iterations.

    Returns:
        Dict mapping stratum name to :class:`FullAnalysisResult`.
        Strata with fewer than 5 queries are omitted.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    n_tasks, k_systems = score_matrix.shape

    if len(query_strata) != n_tasks:
        raise ValueError(
            f"query_strata length ({len(query_strata)}) != "
            f"number of rows ({n_tasks})"
        )

    # Group row indices by stratum
    strata_indices: dict[str, list[int]] = {}
    for idx, stratum in enumerate(query_strata):
        strata_indices.setdefault(stratum, []).append(idx)

    results: dict[str, FullAnalysisResult] = {}
    for stratum, indices in sorted(strata_indices.items()):
        if len(indices) < 5:
            continue
        sub_matrix = score_matrix[indices]
        results[stratum] = run_full_analysis(
            sub_matrix, system_names,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )

    return results


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def run_full_analysis(
    score_matrix: np.ndarray,
    system_names: list[str],
    dimension_matrices: dict[str, np.ndarray] | None = None,
    dimension_names: list[str] | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> FullAnalysisResult:
    """Run the complete statistical analysis pipeline.

    Steps:

    1. Friedman omnibus test on overall scores (with Iman-Davenport correction).
    2. If significant:
       a. PRIMARY: Wilcoxon pairwise with Holm-Bonferroni correction.
       b. SECONDARY: Nemenyi post-hoc (no additional Holm correction).
    3. Bootstrap CIs for all systems (mean and IQM).
    4. Per-dimension Friedman + post-hoc (if dimension matrices provided).
    5. Nemenyi critical difference.
    6. Bootstrap rank distributions.
    7. Generate markdown summary.

    Args:
        score_matrix: shape ``(n_tasks, k_systems)``.
        system_names: names for each column.
        dimension_matrices: optional mapping ``{dim_name: (n_tasks, k_systems)}``.
        dimension_names: optional ordered list of dimension names.
        alpha: family-wise significance level.
        n_bootstrap: bootstrap iterations.

    Returns:
        :class:`FullAnalysisResult` with all sub-results and a markdown summary.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    n_tasks, k_systems = score_matrix.shape

    # 1. Friedman omnibus (now includes Iman-Davenport)
    omnibus = friedman_test(score_matrix, system_names, alpha=alpha)

    # 2. Post-hoc (only if significant)
    pairwise: list[PairwiseResult] = []
    nemenyi_pw: list[PairwiseResult] = []
    if omnibus.is_significant:
        # PRIMARY: Wilcoxon pairwise with Holm-Bonferroni
        pairwise = wilcoxon_pairwise_all(score_matrix, system_names, alpha=alpha)
        # SECONDARY: Nemenyi (already controls FWER, no Holm)
        nemenyi_pw = nemenyi_posthoc(score_matrix, system_names, alpha=alpha)

    # 3. Bootstrap CIs
    bootstrap_cis = compute_all_bootstrap_cis(
        score_matrix, system_names,
        metric_name="overall",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )

    # 4. Per-dimension analysis
    per_dim_omnibus: dict[str, OmnibusResult] = {}
    per_dim_pairwise: dict[str, list[PairwiseResult]] = {}
    per_dim_nemenyi: dict[str, list[PairwiseResult]] = {}

    if dimension_matrices is not None:
        dim_order = dimension_names or sorted(dimension_matrices.keys())
        # First pass: run Friedman test per dimension (uncorrected)
        for dim_name in dim_order:
            if dim_name not in dimension_matrices:
                continue
            dim_matrix = np.asarray(dimension_matrices[dim_name], dtype=float)
            dim_omni = friedman_test(dim_matrix, system_names, alpha=alpha)
            per_dim_omnibus[dim_name] = dim_omni

        # Apply Holm-Bonferroni correction across all per-dimension omnibus
        # p-values to control FWER when testing multiple dimensions.
        n_dim_tests = len(per_dim_omnibus)
        if n_dim_tests > 1:
            sorted_dims = sorted(
                per_dim_omnibus.keys(),
                key=lambda d: per_dim_omnibus[d].p_value,
            )
            for rank_idx, dim_name in enumerate(sorted_dims):
                corrected_alpha = alpha / (n_dim_tests - rank_idx)
                omni = per_dim_omnibus[dim_name]
                # Re-evaluate significance with Holm-corrected threshold
                per_dim_omnibus[dim_name] = OmnibusResult(
                    statistic=omni.statistic,
                    p_value=omni.p_value,
                    is_significant=omni.p_value < corrected_alpha,
                    df=omni.df,
                    n_systems=omni.n_systems,
                    n_tasks=omni.n_tasks,
                    avg_ranks=omni.avg_ranks,
                    iman_davenport_f=omni.iman_davenport_f,
                    iman_davenport_p=omni.iman_davenport_p,
                )

        # Second pass: run pairwise tests only for significant dimensions
        for dim_name in dim_order:
            if dim_name not in per_dim_omnibus:
                continue
            if per_dim_omnibus[dim_name].is_significant:
                dim_matrix = np.asarray(dimension_matrices[dim_name], dtype=float)
                per_dim_pairwise[dim_name] = wilcoxon_pairwise_all(
                    dim_matrix, system_names, alpha=alpha
                )
                per_dim_nemenyi[dim_name] = nemenyi_posthoc(
                    dim_matrix, system_names, alpha=alpha
                )
            else:
                per_dim_pairwise[dim_name] = []
                per_dim_nemenyi[dim_name] = []

    # 5. Critical difference
    try:
        cd = nemenyi_critical_difference(k_systems, n_tasks, alpha=alpha)
    except (ValueError, Exception):
        cd = 0.0

    # 6. Bootstrap rank distributions
    rank_dists = bootstrap_rank_distribution(
        score_matrix, system_names,
        n_bootstrap=n_bootstrap,
        random_state=42,
    )

    # 7. Assemble result
    result = FullAnalysisResult(
        omnibus=omnibus,
        pairwise=pairwise,
        bootstrap_cis=bootstrap_cis,
        per_dimension_omnibus=per_dim_omnibus,
        per_dimension_pairwise=per_dim_pairwise,
        summary_markdown="",
        nemenyi_pairwise=nemenyi_pw,
        per_dimension_nemenyi_pairwise=per_dim_nemenyi,
        critical_difference=cd,
        rank_distributions=rank_dists,
    )

    result.summary_markdown = generate_summary_markdown(result)
    return result


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def generate_summary_markdown(result: FullAnalysisResult) -> str:
    """Generate a publication-ready markdown summary of statistical results.

    The output includes:

    - Friedman omnibus test result (with Iman-Davenport correction)
    - Primary pairwise comparison table (Wilcoxon + Holm-Bonferroni)
    - Secondary pairwise comparison table (Nemenyi)
    - Bootstrap CI table for all systems
    - Rank distribution table
    - Critical difference
    - Per-dimension results (if available)

    Args:
        result: a :class:`FullAnalysisResult`.

    Returns:
        Multi-section markdown string.
    """
    lines: list[str] = []

    # -- Header --
    lines.append("# Statistical Analysis Summary")
    lines.append("")

    # -- Omnibus --
    lines.append("## Friedman Omnibus Test")
    lines.append("")
    o = result.omnibus
    sig_str = "**significant**" if o.is_significant else "not significant"
    lines.append(
        f"- Chi-squared statistic: {o.statistic:.4f} "
        f"(df = {o.df}, p = {o.p_value:.6f}) -- {sig_str}"
    )
    if o.iman_davenport_f > 0 or o.iman_davenport_p < 1.0:
        lines.append(
            f"- Iman-Davenport F: {o.iman_davenport_f:.4f} "
            f"(p = {o.iman_davenport_p:.6f})"
        )
    lines.append(f"- Systems: {o.n_systems}, Tasks: {o.n_tasks}")
    lines.append("")

    # Average ranks table
    lines.append("### Average Ranks (lower = better)")
    lines.append("")
    lines.append("| System | Avg Rank |")
    lines.append("|--------|----------|")
    for name, rank in sorted(o.avg_ranks.items(), key=lambda kv: kv[1]):
        lines.append(f"| {name} | {rank:.3f} |")
    lines.append("")

    # Critical difference
    if result.critical_difference > 0:
        lines.append(
            f"- Nemenyi Critical Difference (CD): {result.critical_difference:.4f}"
        )
        lines.append("")

    # -- Primary Pairwise (Wilcoxon + Holm) --
    if result.pairwise:
        lines.append("## Pairwise Comparisons (Wilcoxon + Holm-Bonferroni)")
        lines.append("")
        lines.append(
            "| System A | System B | p (raw) | p (corrected) | Sig? "
            "| Cliff's d | Effect | Mean Diff | 95% CI |"
        )
        lines.append(
            "|----------|----------|---------|---------------|------"
            "|-----------|--------|-----------|--------|"
        )
        for pw in result.pairwise:
            sig_mark = "Yes" if pw.is_significant else "No"
            lines.append(
                f"| {pw.system_a} | {pw.system_b} "
                f"| {pw.p_value_raw:.4f} | {pw.p_value_corrected:.4f} "
                f"| {sig_mark} | {pw.effect_size:.3f} | {pw.effect_size_label} "
                f"| {pw.mean_diff:+.4f} "
                f"| [{pw.ci_lower:.4f}, {pw.ci_upper:.4f}] |"
            )
        lines.append("")
    else:
        lines.append("## Pairwise Comparisons")
        lines.append("")
        lines.append(
            "Friedman test was not significant; post-hoc tests not performed."
        )
        lines.append("")

    # -- Secondary Pairwise (Nemenyi) --
    if result.nemenyi_pairwise:
        lines.append("## Nemenyi Post-Hoc (secondary)")
        lines.append("")
        lines.append(
            "| System A | System B | p (Nemenyi) | Sig? "
            "| Cliff's d | Effect | Mean Diff |"
        )
        lines.append(
            "|----------|----------|-------------|------"
            "|-----------|--------|-----------|"
        )
        for pw in result.nemenyi_pairwise:
            sig_mark = "Yes" if pw.is_significant else "No"
            lines.append(
                f"| {pw.system_a} | {pw.system_b} "
                f"| {pw.p_value_corrected:.4f} "
                f"| {sig_mark} | {pw.effect_size:.3f} | {pw.effect_size_label} "
                f"| {pw.mean_diff:+.4f} |"
            )
        lines.append("")

    # -- Bootstrap CIs --
    lines.append("## Bootstrap Confidence Intervals")
    lines.append("")
    lines.append(
        "| System | Mean | 95% CI | IQM | IQM 95% CI | Std | n |"
    )
    lines.append(
        "|--------|------|--------|-----|------------|-----|---|"
    )
    for ci in result.bootstrap_cis:
        lines.append(
            f"| {ci.system} | {ci.mean:.4f} "
            f"| [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}] "
            f"| {ci.iqm:.4f} "
            f"| [{ci.iqm_ci_lower:.4f}, {ci.iqm_ci_upper:.4f}] "
            f"| {ci.std:.4f} | {ci.n_samples} |"
        )
    lines.append("")

    # -- Rank Distributions --
    if result.rank_distributions:
        lines.append("## Bootstrap Rank Distributions")
        lines.append("")
        lines.append(
            "| System | Mean Rank | 95% CI | P(Rank 1) |"
        )
        lines.append(
            "|--------|-----------|--------|-----------|"
        )
        for rd in sorted(result.rank_distributions, key=lambda d: d["mean_rank"]):
            lines.append(
                f"| {rd['system']} | {rd['mean_rank']:.3f} "
                f"| [{rd['rank_ci_lower']:.2f}, {rd['rank_ci_upper']:.2f}] "
                f"| {rd['prob_rank_1']:.3f} |"
            )
        lines.append("")

    # -- Per-dimension --
    if result.per_dimension_omnibus:
        lines.append("## Per-Dimension Analysis")
        lines.append("")
        n_dims = len(result.per_dimension_omnibus)
        if n_dims > 1:
            lines.append(
                f"*Note: Significance thresholds are Holm-Bonferroni corrected "
                f"across {n_dims} dimensions to control FWER at alpha=0.05.*"
            )
            lines.append("")
        for dim_name, dim_omni in result.per_dimension_omnibus.items():
            sig_str = "significant" if dim_omni.is_significant else "not significant"
            lines.append(f"### {dim_name}")
            lines.append("")
            lines.append(
                f"- Friedman chi2 = {dim_omni.statistic:.4f}, "
                f"p = {dim_omni.p_value:.6f} ({sig_str})"
            )
            if dim_omni.iman_davenport_f > 0 or dim_omni.iman_davenport_p < 1.0:
                lines.append(
                    f"- Iman-Davenport F = {dim_omni.iman_davenport_f:.4f}, "
                    f"p = {dim_omni.iman_davenport_p:.6f}"
                )
            # Show significant pairwise results only (primary: Wilcoxon+Holm)
            dim_pw = result.per_dimension_pairwise.get(dim_name, [])
            sig_pairs = [p for p in dim_pw if p.is_significant]
            if sig_pairs:
                lines.append(
                    f"- Significant pairs ({len(sig_pairs)}):"
                )
                for pw in sig_pairs:
                    lines.append(
                        f"  - {pw.system_a} vs {pw.system_b}: "
                        f"p = {pw.p_value_corrected:.4f}, "
                        f"d = {pw.effect_size:.3f} ({pw.effect_size_label})"
                    )
            else:
                lines.append("- No significant pairwise differences.")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variance decomposition (two-factor ANOVA SS)
# ---------------------------------------------------------------------------


def variance_decomposition(
    score_matrix: np.ndarray,
    system_names: list[str],
    query_ids: list[str],
    query_metadata: dict[str, dict] | None = None,
) -> dict:
    """Decompose total score variance into system, query, and residual components.

    Two-factor ANOVA SS decomposition (no interaction term estimated separately).

    Uses a balanced two-way layout where each system is evaluated on every query.
    The decomposition is:

        SS_total = SS_system + SS_query + SS_residual

    where SS_residual captures system-query interaction plus noise.

    Args:
        score_matrix: shape (n_systems, n_queries) with scores. Each row is a
            system, each column is a query.
        system_names: list of system names, length must match n_systems.
        query_ids: list of query identifiers, length must match n_queries.
        query_metadata: optional dict mapping query_id to metadata dict.
            If the metadata contains "difficulty" or "domain" keys, the
            query component will be further decomposed by those facets.

    Returns:
        dict with keys:
            - ss_total, ss_system, ss_query, ss_residual (raw sum-of-squares)
            - system_pct, query_pct, residual_pct (percentage of total variance)
            - grand_mean (float)
            - system_means (dict[str, float])
            - query_means (dict[str, float])
            - by_difficulty (dict | None): per-difficulty SS breakdown
            - by_domain (dict | None): per-domain SS breakdown
    """
    mat = np.asarray(score_matrix, dtype=float)
    n_systems, n_queries = mat.shape

    if len(system_names) != n_systems:
        raise ValueError(
            f"system_names length ({len(system_names)}) != n_systems ({n_systems})"
        )
    if len(query_ids) != n_queries:
        raise ValueError(
            f"query_ids length ({len(query_ids)}) != n_queries ({n_queries})"
        )

    grand_mean = float(np.mean(mat))
    system_means_arr = np.mean(mat, axis=1)  # shape (n_systems,)
    query_means_arr = np.mean(mat, axis=0)   # shape (n_queries,)

    # Sum of squares
    ss_total = float(np.sum((mat - grand_mean) ** 2))
    ss_system = float(n_queries * np.sum((system_means_arr - grand_mean) ** 2))
    ss_query = float(n_systems * np.sum((query_means_arr - grand_mean) ** 2))
    ss_residual = ss_total - ss_system - ss_query

    # Clamp residual to zero in case of floating-point rounding
    if abs(ss_residual) < 1e-12:
        ss_residual = 0.0

    # Percentages
    if ss_total > 0:
        system_pct = 100.0 * ss_system / ss_total
        query_pct = 100.0 * ss_query / ss_total
        residual_pct = 100.0 * ss_residual / ss_total
    else:
        system_pct = 0.0
        query_pct = 0.0
        residual_pct = 0.0

    system_means = {
        name: float(system_means_arr[i]) for i, name in enumerate(system_names)
    }
    query_means = {
        qid: float(query_means_arr[j]) for j, qid in enumerate(query_ids)
    }

    # Optional: decompose query variance by difficulty / domain
    by_difficulty = None
    by_domain = None

    if query_metadata:
        by_difficulty = _decompose_by_facet(
            mat, query_ids, query_metadata, "difficulty", grand_mean, n_systems,
        )
        by_domain = _decompose_by_facet(
            mat, query_ids, query_metadata, "domain", grand_mean, n_systems,
        )

    return {
        "ss_total": ss_total,
        "ss_system": ss_system,
        "ss_query": ss_query,
        "ss_residual": ss_residual,
        "system_pct": system_pct,
        "query_pct": query_pct,
        "residual_pct": residual_pct,
        "grand_mean": grand_mean,
        "system_means": system_means,
        "query_means": query_means,
        "by_difficulty": by_difficulty,
        "by_domain": by_domain,
    }


def _decompose_by_facet(
    mat: np.ndarray,
    query_ids: list[str],
    query_metadata: dict[str, dict],
    facet_key: str,
    grand_mean: float,
    n_systems: int,
) -> dict | None:
    """Decompose query-level variance by a metadata facet (e.g. difficulty).

    Groups queries by their facet value and computes the between-group and
    within-group components of the query sum-of-squares.

    Returns None if the facet is not present in any query metadata.
    """
    # Group query indices by facet value
    groups: dict[str, list[int]] = {}
    for j, qid in enumerate(query_ids):
        meta = query_metadata.get(qid, {})
        if facet_key not in meta:
            continue
        val = str(meta[facet_key])
        groups.setdefault(val, []).append(j)

    if not groups:
        return None

    result: dict[str, dict] = {}
    for facet_val, indices in sorted(groups.items()):
        sub_mat = mat[:, indices]
        sub_mean = float(np.mean(sub_mat))
        sub_query_means = np.mean(sub_mat, axis=0)
        ss_within = float(n_systems * np.sum((sub_query_means - sub_mean) ** 2))
        result[facet_val] = {
            "n_queries": len(indices),
            "mean_score": sub_mean,
            "ss_query_within": ss_within,
        }

    return result


# ---------------------------------------------------------------------------
# Pareto frontier analysis
# ---------------------------------------------------------------------------


def pareto_frontier(
    system_metrics: dict[str, dict[str, float]],
    objectives: list[str] | None = None,
    minimize: list[str] | None = None,
) -> dict:
    """Identify Pareto-optimal systems across multiple objectives.

    A system is Pareto-optimal if no other system is at least as good on
    every objective and strictly better on at least one.

    Args:
        system_metrics: dict mapping system name to a dict of metric values.
            Example: {"P0": {"score": 0.45, "cost_usd": 1.2, "tokens": 5000}}.
        objectives: list of metric keys to consider. If None, all keys from the
            first system are used.
        minimize: list of metric keys that should be minimized (lower is better).
            Default: ["cost_usd", "tokens", "elapsed_seconds"]. All other
            objectives are maximized.

    Returns:
        dict with keys:
            - pareto_optimal: list of system names on the Pareto frontier.
            - dominated: dict mapping each dominated system to the list of
              systems that dominate it.
            - efficiency_scores: dict mapping each system to a float in [0, 1]
              representing normalized distance to the ideal point (1.0 = ideal).
    """
    if not system_metrics:
        return {
            "pareto_optimal": [],
            "dominated": {},
            "efficiency_scores": {},
        }

    if minimize is None:
        minimize = ["cost_usd", "tokens", "elapsed_seconds"]
    minimize_set = frozenset(minimize)

    names = sorted(system_metrics.keys())
    if objectives is None:
        objectives = sorted(system_metrics[names[0]].keys())

    n = len(names)

    # Build normalised matrix: after normalisation, higher is always better.
    # For minimised metrics we negate before comparison.
    raw = np.zeros((n, len(objectives)))
    for i, name in enumerate(names):
        for j, obj in enumerate(objectives):
            raw[i, j] = system_metrics[name].get(obj, 0.0)

    # Create comparison matrix where higher = better for all objectives
    oriented = raw.copy()
    for j, obj in enumerate(objectives):
        if obj in minimize_set:
            oriented[:, j] = -oriented[:, j]

    # Determine domination
    dominated: dict[str, list[str]] = {name: [] for name in names}
    pareto_set: set[str] = set(names)

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            # Check if k dominates i:
            # k >= i on all objectives AND k > i on at least one
            at_least_as_good = np.all(oriented[k] >= oriented[i])
            strictly_better = np.any(oriented[k] > oriented[i])
            if at_least_as_good and strictly_better:
                dominated[names[i]].append(names[k])
                pareto_set.discard(names[i])

    # Clean up dominated dict: only include actually dominated systems
    dominated = {name: doms for name, doms in dominated.items() if doms}

    # Efficiency scores: normalised distance to ideal point
    # Ideal = best observed value for each objective (after orientation)
    ideal = np.max(oriented, axis=0)
    anti_ideal = np.min(oriented, axis=0)
    span = ideal - anti_ideal

    efficiency_scores: dict[str, float] = {}
    for i, name in enumerate(names):
        if np.all(span == 0):
            # All systems identical on all objectives
            efficiency_scores[name] = 1.0
        else:
            # Normalise each dimension to [0, 1], compute Euclidean distance to ideal
            normed = np.where(span > 0, (oriented[i] - anti_ideal) / span, 1.0)
            # Distance to ideal in normalised space
            dist = float(np.linalg.norm(normed - 1.0))
            max_dist = float(np.sqrt(len(objectives)))  # distance from (0,...,0) to (1,...,1)
            if max_dist > 0:
                efficiency_scores[name] = 1.0 - dist / max_dist
            else:
                efficiency_scores[name] = 1.0

    return {
        "pareto_optimal": sorted(pareto_set),
        "dominated": dominated,
        "efficiency_scores": efficiency_scores,
    }
