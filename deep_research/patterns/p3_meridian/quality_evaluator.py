"""MERIDIAN Role 4: Quality Evaluator — single-judge evaluation on 12 dimensions.

Runs one judge model (gpt-5.2), collects per-dimension scores and rationales.
Previously ran three identical judges in parallel, but since all three used the
same model at the same temperature, there was no real diversity. External
multi-judge evaluation (Phase 1 LLM-as-judge) handles proper cross-model scoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from deep_research.tools import LLMCaller
from deep_research.types import Section

from .rubric import DIMENSION_NAMES, build_judge_prompt
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# Single judge model (three identical judges provided no diversity)
JUDGE_MODELS: List[str] = [DEFAULT_MODEL]

# Minimum acceptable average overall score (below this triggers revision)
REVISION_THRESHOLD = 7.0

# Maximum revision iterations allowed
MAX_REVISIONS = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score and rationale for a single dimension from a single judge."""
    dimension: str
    score: int
    rationale: str
    judge_model: str


@dataclass
class JudgeResult:
    """Full evaluation result from one judge model."""
    model: str
    dimension_scores: Dict[str, DimensionScore] = field(default_factory=dict)
    overall_score: int = 0
    overall_rationale: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation across all judges."""
    judge_results: List[JudgeResult] = field(default_factory=list)
    averaged_scores: Dict[str, float] = field(default_factory=dict)
    averaged_overall: float = 0.0
    needs_revision: bool = False
    feedback_text: str = ""

    def summary_dict(self) -> Dict[str, Any]:
        """Return a serialisable summary of the evaluation."""
        return {
            "averaged_overall": round(self.averaged_overall, 2),
            "needs_revision": self.needs_revision,
            "dimension_scores": {
                k: round(v, 2) for k, v in self.averaged_scores.items()
            },
            "judges": [
                {
                    "model": jr.model,
                    "overall_score": jr.overall_score,
                    "overall_rationale": jr.overall_rationale,
                }
                for jr in self.judge_results
            ],
        }


# ── Single judge evaluation ───────────────────────────────────────────────────

async def _run_single_judge(
    query: str,
    report_text: str,
    model: str,
    llm: LLMCaller,
) -> JudgeResult:
    """Run evaluation with a single judge model."""
    prompt = build_judge_prompt(query, report_text)
    result = JudgeResult(model=model)

    try:
        raw = await llm.complete_json(
            prompt=prompt,
            model=model,
            system=(
                "You are an expert research-report evaluator. Score rigorously on a "
                "1-10 scale. 7 = good professional quality. 9-10 = exceptional."
            ),
            temperature=0.2,
            max_tokens=4096,
        )
        result.raw_response = raw
    except Exception as e:
        log.warning("quality_evaluator.judge_failed", model=model, error=str(e))
        # Return default scores on failure
        for dim_name in DIMENSION_NAMES:
            result.dimension_scores[dim_name] = DimensionScore(
                dimension=dim_name, score=5, rationale="Judge failed", judge_model=model
            )
        result.overall_score = 5
        result.overall_rationale = f"Judge {model} failed: {e}"
        return result

    # Parse dimension scores
    dimensions_raw = raw.get("dimensions", {})
    for dim_name in DIMENSION_NAMES:
        dim_data = dimensions_raw.get(dim_name, {})
        score = dim_data.get("score", 5)
        # Clamp to valid range
        score = max(1, min(10, int(score)))
        result.dimension_scores[dim_name] = DimensionScore(
            dimension=dim_name,
            score=score,
            rationale=dim_data.get("rationale", ""),
            judge_model=model,
        )

    # Parse overall score
    overall_raw = raw.get("overall", {})
    result.overall_score = max(1, min(10, int(overall_raw.get("score", 5))))
    result.overall_rationale = overall_raw.get("rationale", "")

    log.info(
        "quality_evaluator.judge_done",
        model=model,
        overall=result.overall_score,
    )
    return result


# ── Multi-judge aggregation ───────────────────────────────────────────────────

def _aggregate_results(judge_results: List[JudgeResult]) -> EvaluationResult:
    """Average scores across all judges and decide whether revision is needed."""
    n = len(judge_results)
    if n == 0:
        return EvaluationResult(needs_revision=True, feedback_text="No judges ran.")

    # Average dimension scores
    averaged: Dict[str, float] = {}
    for dim_name in DIMENSION_NAMES:
        scores = [
            jr.dimension_scores[dim_name].score
            for jr in judge_results
            if dim_name in jr.dimension_scores
        ]
        averaged[dim_name] = sum(scores) / len(scores) if scores else 5.0

    # Average overall
    overall_scores = [jr.overall_score for jr in judge_results]
    averaged_overall = sum(overall_scores) / len(overall_scores)

    needs_revision = averaged_overall < REVISION_THRESHOLD

    # Build feedback text for the writer (used if revision is needed)
    feedback_parts: List[str] = []
    feedback_parts.append(f"Average overall score: {averaged_overall:.1f}/10")
    feedback_parts.append(
        f"Revision {'REQUIRED' if needs_revision else 'not needed'} "
        f"(threshold: {REVISION_THRESHOLD})"
    )
    feedback_parts.append("")

    # Identify weak dimensions (below 7)
    weak_dims = [(name, score) for name, score in averaged.items() if score < 7.0]
    if weak_dims:
        feedback_parts.append("### Weak Dimensions (need improvement)")
        for name, score in sorted(weak_dims, key=lambda x: x[1]):
            rationales = [
                f"  - {jr.model}: {jr.dimension_scores[name].rationale}"
                for jr in judge_results
                if name in jr.dimension_scores and jr.dimension_scores[name].rationale
            ]
            feedback_parts.append(f"**{name}** ({score:.1f}/10)")
            feedback_parts.extend(rationales)
            feedback_parts.append("")

    # Include overall rationales from each judge
    feedback_parts.append("### Overall Assessments from Each Judge")
    for jr in judge_results:
        feedback_parts.append(
            f"- **{jr.model}** (score: {jr.overall_score}/10): {jr.overall_rationale}"
        )

    feedback_text = "\n".join(feedback_parts)

    return EvaluationResult(
        judge_results=judge_results,
        averaged_scores=averaged,
        averaged_overall=averaged_overall,
        needs_revision=needs_revision,
        feedback_text=feedback_text,
    )


# ── Top-level entrypoint ──────────────────────────────────────────────────────

async def evaluate_report(
    query: str,
    report_text: str,
    llm: LLMCaller,
    judge_models: Optional[List[str]] = None,
) -> EvaluationResult:
    """Run multi-judge evaluation in parallel and aggregate results.

    Args:
        query: The original research query.
        report_text: The full text of the report to evaluate.
        llm: Shared LLMCaller (carries the cost tracker).
        judge_models: Models to use as judges. Defaults to JUDGE_MODELS.

    Returns:
        EvaluationResult with averaged scores and revision decision.
    """
    models = judge_models or JUDGE_MODELS

    log.info("quality_evaluator.starting", judges=models)

    tasks = [
        _run_single_judge(query, report_text, model, llm)
        for model in models
    ]
    judge_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results: List[JudgeResult] = []
    for res in judge_results:
        if isinstance(res, JudgeResult):
            valid_results.append(res)
        else:
            log.warning("quality_evaluator.judge_exception", error=str(res))

    evaluation = _aggregate_results(valid_results)

    log.info(
        "quality_evaluator.complete",
        avg_overall=f"{evaluation.averaged_overall:.1f}",
        needs_revision=evaluation.needs_revision,
        judges_succeeded=len(valid_results),
    )
    return evaluation
