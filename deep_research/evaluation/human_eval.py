"""Human evaluation data structures and agreement metrics.

Provides the data model for human evaluation results and functions
to compute inter-annotator agreement and judge-human concordance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HumanVerdict:
    """A single human evaluator's judgment on one criterion."""

    evaluator_id: str
    report_id: str
    criterion: str
    dimension: str
    verdict: str           # "SATISFIED", "NOT_SATISFIED", "UNCERTAIN"
    confidence: float      # 0-1, self-reported
    comment: str = ""
    time_seconds: int = 0


@dataclass
class HumanEvalResult:
    """Aggregated human evaluation for one report."""

    report_id: str
    pattern: str
    query_id: str
    evaluators: list[str]
    verdicts: list[HumanVerdict]
    # Agreement metrics
    fleiss_kappa: float = 0.0
    per_dimension_agreement: dict[str, float] = field(default_factory=dict)

    @property
    def n_evaluators(self) -> int:
        return len(self.evaluators)

    @property
    def avg_confidence(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.confidence for v in self.verdicts) / len(self.verdicts)

    def dimension_scores(self) -> dict[str, float]:
        """Compute dimension scores from majority vote of human verdicts."""
        by_dim: dict[str, list[bool]] = {}
        for v in self.verdicts:
            by_dim.setdefault(v.dimension, []).append(v.verdict == "SATISFIED")

        return {
            dim: sum(votes) / len(votes) if votes else 0.0
            for dim, votes in by_dim.items()
        }


@dataclass
class JudgeHumanAgreement:
    """Agreement metrics between LLM judge and human evaluators."""

    n_reports: int
    overall_kappa: float
    per_dimension_kappa: dict[str, float]
    overall_correlation: float     # Pearson r between judge and human scores
    dimension_correlation: dict[str, float]
    agreement_rate: float          # fraction of criteria where they agree
    judge_bias: float              # positive = judge rates higher than humans


def compute_judge_human_agreement(
    judge_scores: dict[str, dict[str, float]],
    human_results: list[HumanEvalResult],
) -> JudgeHumanAgreement:
    """Compute agreement between LLM judge and human evaluators.

    Compares per-dimension scores from the judge with majority-vote
    human dimension scores.

    Args:
        judge_scores: ``{report_id: {dimension: score}}``.
        human_results: List of :class:`HumanEvalResult`.

    Returns:
        :class:`JudgeHumanAgreement` with agreement metrics.
    """
    from deep_research.evaluation.multi_judge import cohens_kappa

    matched_reports: list[tuple[HumanEvalResult, dict[str, float]]] = []
    for hr in human_results:
        if hr.report_id in judge_scores:
            matched_reports.append((hr, judge_scores[hr.report_id]))

    if not matched_reports:
        return JudgeHumanAgreement(
            n_reports=0, overall_kappa=0.0, per_dimension_kappa={},
            overall_correlation=0.0, dimension_correlation={},
            agreement_rate=0.0, judge_bias=0.0,
        )

    # Collect dimension scores
    all_dims: set[str] = set()
    judge_dim_scores: dict[str, list[float]] = {}
    human_dim_scores: dict[str, list[float]] = {}

    for hr, js in matched_reports:
        hs = hr.dimension_scores()
        for dim in set(js.keys()) | set(hs.keys()):
            all_dims.add(dim)
            judge_dim_scores.setdefault(dim, []).append(js.get(dim, 0.0))
            human_dim_scores.setdefault(dim, []).append(hs.get(dim, 0.0))

    # Per-dimension correlation
    dim_corr: dict[str, float] = {}
    for dim in all_dims:
        if len(judge_dim_scores.get(dim, [])) >= 3:
            j = np.array(judge_dim_scores[dim])
            h = np.array(human_dim_scores[dim])
            if np.std(j) > 0 and np.std(h) > 0:
                dim_corr[dim] = float(np.corrcoef(j, h)[0, 1])
            else:
                dim_corr[dim] = 0.0

    # Overall correlation
    all_j: list[float] = []
    all_h: list[float] = []
    for dim in all_dims:
        all_j.extend(judge_dim_scores.get(dim, []))
        all_h.extend(human_dim_scores.get(dim, []))

    overall_corr = 0.0
    if len(all_j) >= 3:
        j_arr = np.array(all_j)
        h_arr = np.array(all_h)
        if np.std(j_arr) > 0 and np.std(h_arr) > 0:
            overall_corr = float(np.corrcoef(j_arr, h_arr)[0, 1])

    # Binary agreement (threshold at 0.5)
    j_binary = [1 if x >= 0.5 else 0 for x in all_j]
    h_binary = [1 if x >= 0.5 else 0 for x in all_h]
    agreement_rate = (
        sum(a == b for a, b in zip(j_binary, h_binary)) / len(j_binary)
        if j_binary else 0.0
    )

    # Cohen's kappa on binary
    overall_kappa = cohens_kappa(
        [bool(x) for x in j_binary],
        [bool(x) for x in h_binary],
    )

    # Per-dimension kappa
    per_dim_kappa: dict[str, float] = {}
    for dim in all_dims:
        j_d = [1 if x >= 0.5 else 0 for x in judge_dim_scores.get(dim, [])]
        h_d = [1 if x >= 0.5 else 0 for x in human_dim_scores.get(dim, [])]
        if len(j_d) >= 5:
            per_dim_kappa[dim] = cohens_kappa(
                [bool(x) for x in j_d],
                [bool(x) for x in h_d],
            )

    # Judge bias
    judge_bias = float(np.mean(all_j) - np.mean(all_h)) if all_j else 0.0

    return JudgeHumanAgreement(
        n_reports=len(matched_reports),
        overall_kappa=overall_kappa,
        per_dimension_kappa=per_dim_kappa,
        overall_correlation=overall_corr,
        dimension_correlation=dim_corr,
        agreement_rate=agreement_rate,
        judge_bias=judge_bias,
    )


def generate_human_eval_report(
    human_results: list[HumanEvalResult],
    agreement: JudgeHumanAgreement,
) -> str:
    """Generate markdown summary of human evaluation results.

    Args:
        human_results: List of all human evaluation results.
        agreement: Judge-human agreement metrics.

    Returns:
        Markdown-formatted summary string.
    """
    lines = ["# Human Evaluation Results\n"]

    lines.append("## Summary\n")
    lines.append(f"- Reports evaluated: {agreement.n_reports}")
    lines.append(f"- Overall Judge-Human Cohen's kappa: {agreement.overall_kappa:.3f}")
    lines.append(
        f"- Overall correlation (Pearson r): {agreement.overall_correlation:.3f}"
    )
    lines.append(f"- Agreement rate: {agreement.agreement_rate:.1%}")
    lines.append(
        f"- Judge bias: {agreement.judge_bias:+.3f} "
        "(positive = judge rates higher)\n"
    )

    if agreement.per_dimension_kappa:
        lines.append("## Per-Dimension Agreement\n")
        lines.append("| Dimension | Cohen's kappa | Correlation |")
        lines.append("|---|---|---|")
        for dim in sorted(agreement.per_dimension_kappa.keys()):
            kappa = agreement.per_dimension_kappa[dim]
            corr = agreement.dimension_correlation.get(dim, 0.0)
            lines.append(f"| {dim} | {kappa:.3f} | {corr:.3f} |")

    return "\n".join(lines)
