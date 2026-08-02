"""Judge calibration infrastructure.

Detects and corrects systematic biases in LLM-as-judge scoring:
- Position bias (first/last criterion favored)
- Length bias (longer reports get higher scores)
- Severity bias (judges too lenient or too strict)
- Dimension bias (some dimensions systematically scored differently)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationResult:
    """Result of calibrating a judge's scoring patterns."""
    judge_id: str
    n_samples: int
    position_bias: float  # correlation between criterion position and pass rate
    length_bias: float  # correlation between report word count and overall score
    severity_score: float  # 0=very lenient, 0.5=balanced, 1=very strict
    dimension_biases: dict[str, float]  # per-dimension deviation from mean pass rate
    recommendations: list[str]


@dataclass
class CalibrationData:
    """Raw data for calibration analysis."""
    verdicts: list[dict]  # list of {criterion_index, verdict, dimension}
    report_word_counts: list[int]
    overall_scores: list[float]
    judge_id: str = "default"


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two sequences.

    Returns 0.0 if either sequence has zero variance or if inputs are empty.
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    sum_sq_x = sum((x - x_mean) ** 2 for x in xs)
    sum_sq_y = sum((y - y_mean) ** 2 for y in ys)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def detect_position_bias(verdicts_by_position: list[list[str]]) -> float:
    """Detect whether criterion position affects verdict.

    Args:
        verdicts_by_position: For each criterion position (0, 1, ...),
            a list of "SATISFIED"/"NOT_SATISFIED" verdicts across many runs.

    Returns:
        Pearson correlation between position and pass rate.
        Positive = later criteria get more SATISFIED.
        Negative = earlier criteria get more SATISFIED.
        Returns 0.0 for empty input or a single position.
    """
    if len(verdicts_by_position) < 2:
        return 0.0

    positions: list[float] = []
    pass_rates: list[float] = []

    for idx, verdicts in enumerate(verdicts_by_position):
        if not verdicts:
            # Skip positions with no data
            continue
        satisfied = sum(1 for v in verdicts if v == "SATISFIED")
        rate = satisfied / len(verdicts)
        positions.append(float(idx))
        pass_rates.append(rate)

    if len(positions) < 2:
        return 0.0

    return _pearson_correlation(positions, pass_rates)


def detect_length_bias(
    word_counts: list[int],
    scores: list[float],
) -> float:
    """Detect correlation between report length and score.

    Args:
        word_counts: Word count for each report.
        scores: Overall score for each report.

    Returns:
        Pearson correlation. Positive = longer reports get higher scores.
        Returns 0.0 for empty input or mismatched lengths.
    """
    if not word_counts or not scores or len(word_counts) != len(scores):
        return 0.0

    if len(word_counts) < 2:
        return 0.0

    return _pearson_correlation(
        [float(wc) for wc in word_counts],
        list(scores),
    )


def detect_severity(pass_rates: list[float]) -> float:
    """Compute severity score from distribution of pass rates.

    Args:
        pass_rates: Per-criterion pass rate across all reports.

    Returns:
        0.0 = very lenient (all passing), 0.5 = balanced, 1.0 = very strict (all failing).
        Returns 0.5 for empty input (no information defaults to balanced).
    """
    if not pass_rates:
        return 0.5

    mean_pass_rate = sum(pass_rates) / len(pass_rates)
    # Invert: high pass rate = lenient (low severity), low pass rate = strict (high severity)
    return 1.0 - mean_pass_rate


def detect_dimension_bias(
    dimension_pass_rates: dict[str, float],
) -> dict[str, float]:
    """Detect per-dimension scoring bias relative to overall mean.

    Args:
        dimension_pass_rates: Pass rate per dimension.

    Returns:
        Dict mapping dimension to deviation from mean (positive = easier, negative = harder).
        Returns empty dict for empty input.
    """
    if not dimension_pass_rates:
        return {}

    mean_rate = sum(dimension_pass_rates.values()) / len(dimension_pass_rates)
    return {dim: rate - mean_rate for dim, rate in dimension_pass_rates.items()}


def calibrate_scores(
    raw_scores: dict[str, float],
    dimension_biases: dict[str, float],
    severity_adjustment: float = 0.0,
) -> dict[str, float]:
    """Apply calibration corrections to raw dimension scores.

    Subtracts dimension bias (so an "easy" dimension is penalized) and
    adds the global severity adjustment. Results are clamped to [0, 1].

    Args:
        raw_scores: Raw per-dimension scores (0-1).
        dimension_biases: Per-dimension bias from detect_dimension_bias().
        severity_adjustment: Global severity correction (-1 to 1).
            Positive = judge was too lenient, so we lower scores.
            Negative = judge was too strict, so we raise scores.

    Returns:
        Calibrated scores, clamped to [0, 1].
    """
    if not raw_scores:
        return {}

    calibrated: dict[str, float] = {}
    for dim, raw in raw_scores.items():
        bias = dimension_biases.get(dim, 0.0)
        # Subtract bias: if a dimension is "easy" (positive bias), reduce score
        # Subtract severity_adjustment: if judge is lenient (positive adj), reduce score
        adjusted = raw - bias - severity_adjustment
        calibrated[dim] = max(0.0, min(1.0, adjusted))

    return calibrated


def run_calibration(data: CalibrationData) -> CalibrationResult:
    """Run full calibration analysis on judge data.

    Computes all bias metrics and generates recommendations.

    Args:
        data: CalibrationData with verdicts, word counts, scores, and judge_id.

    Returns:
        CalibrationResult with all computed metrics and recommendations.
    """
    # --- Position bias ---
    # Group verdicts by criterion_index
    position_map: dict[int, list[str]] = {}
    for v in data.verdicts:
        idx = v.get("criterion_index", 0)
        verdict = v.get("verdict", "")
        if idx not in position_map:
            position_map[idx] = []
        position_map[idx].append(verdict)

    if position_map:
        max_pos = max(position_map.keys())
        verdicts_by_position = [
            position_map.get(i, []) for i in range(max_pos + 1)
        ]
    else:
        verdicts_by_position = []

    position_bias = detect_position_bias(verdicts_by_position)

    # --- Length bias ---
    length_bias = detect_length_bias(data.report_word_counts, data.overall_scores)

    # --- Severity ---
    # Compute per-criterion pass rates across all verdicts
    criterion_verdicts: dict[int, list[str]] = {}
    for v in data.verdicts:
        idx = v.get("criterion_index", 0)
        verdict = v.get("verdict", "")
        if idx not in criterion_verdicts:
            criterion_verdicts[idx] = []
        criterion_verdicts[idx].append(verdict)

    criterion_pass_rates: list[float] = []
    for idx in sorted(criterion_verdicts.keys()):
        vs = criterion_verdicts[idx]
        if vs:
            satisfied = sum(1 for verdict in vs if verdict == "SATISFIED")
            criterion_pass_rates.append(satisfied / len(vs))

    severity_score = detect_severity(criterion_pass_rates)

    # --- Dimension bias ---
    dimension_verdicts: dict[str, list[str]] = {}
    for v in data.verdicts:
        dim = v.get("dimension", "unknown")
        verdict = v.get("verdict", "")
        if dim not in dimension_verdicts:
            dimension_verdicts[dim] = []
        dimension_verdicts[dim].append(verdict)

    dimension_pass_rates: dict[str, float] = {}
    for dim, vs in dimension_verdicts.items():
        if vs:
            satisfied = sum(1 for verdict in vs if verdict == "SATISFIED")
            dimension_pass_rates[dim] = satisfied / len(vs)

    dimension_biases = detect_dimension_bias(dimension_pass_rates)

    # --- Recommendations ---
    recommendations = _generate_recommendations(
        position_bias=position_bias,
        length_bias=length_bias,
        severity_score=severity_score,
        dimension_biases=dimension_biases,
    )

    n_samples = len(data.overall_scores) if data.overall_scores else 0

    return CalibrationResult(
        judge_id=data.judge_id,
        n_samples=n_samples,
        position_bias=position_bias,
        length_bias=length_bias,
        severity_score=severity_score,
        dimension_biases=dimension_biases,
        recommendations=recommendations,
    )


def _generate_recommendations(
    position_bias: float,
    length_bias: float,
    severity_score: float,
    dimension_biases: dict[str, float],
) -> list[str]:
    """Generate human-readable recommendations from calibration metrics."""
    recommendations: list[str] = []

    # Position bias thresholds
    if abs(position_bias) > 0.5:
        direction = "later" if position_bias > 0 else "earlier"
        recommendations.append(
            f"Strong position bias detected (r={position_bias:.3f}): "
            f"{direction} criteria are favored. Consider randomizing criterion order."
        )
    elif abs(position_bias) > 0.3:
        direction = "later" if position_bias > 0 else "earlier"
        recommendations.append(
            f"Moderate position bias detected (r={position_bias:.3f}): "
            f"{direction} criteria are slightly favored. Monitor across runs."
        )

    # Length bias thresholds
    if abs(length_bias) > 0.5:
        direction = "longer" if length_bias > 0 else "shorter"
        recommendations.append(
            f"Strong length bias detected (r={length_bias:.3f}): "
            f"{direction} reports receive higher scores. Consider length-normalizing."
        )
    elif abs(length_bias) > 0.3:
        direction = "longer" if length_bias > 0 else "shorter"
        recommendations.append(
            f"Moderate length bias detected (r={length_bias:.3f}): "
            f"{direction} reports receive slightly higher scores."
        )

    # Severity thresholds
    if severity_score < 0.2:
        recommendations.append(
            f"Judge is very lenient (severity={severity_score:.3f}). "
            f"Consider stricter rubric criteria or recalibrating."
        )
    elif severity_score > 0.8:
        recommendations.append(
            f"Judge is very strict (severity={severity_score:.3f}). "
            f"Consider whether rubric criteria are too demanding."
        )

    # Dimension bias thresholds
    for dim, bias in sorted(dimension_biases.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(bias) > 0.2:
            direction = "easy" if bias > 0 else "hard"
            recommendations.append(
                f"Dimension '{dim}' appears systematically {direction} "
                f"(bias={bias:+.3f}). Review rubric for this dimension."
            )

    if not recommendations:
        recommendations.append("No significant biases detected. Judge appears well-calibrated.")

    return recommendations
