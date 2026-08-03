"""Meta-evaluator: evaluate coverage and decide next iteration.

Uses gpt-5.2 to assess whether the current research findings
sufficiently cover the original query, and whether another
width-depth iteration is warranted. Evaluates based on source
summaries and analyses, not chunk counts.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

META_EVAL_SYSTEM = """You are a research quality evaluator. Assess the completeness and
quality of research findings against the original query. Be critical and specific."""

META_EVAL_PROMPT = """Evaluate the current state of this research project.

Original Query: {query}

Research Plan Title: {plan_title}
Planned Subtopics: {planned_subtopics}

Completed Analyses (by subtopic):
{analyses_summary}

Current Statistics:
- Iteration step: {step} of {max_steps}
- Total documents retrieved: {total_docs}
- Relevant source summaries collected: {total_summaries}
- Budget spent: ${budget_spent:.4f} of ${budget_total:.4f}
- Average analysis confidence: {avg_confidence:.2f}

Evaluate and return JSON:
{{
    "coverage_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "should_continue": true/false,
    "rationale": "Why continue or stop",
    "covered_subtopics": ["subtopic1", "subtopic2"],
    "uncovered_subtopics": ["subtopic3"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "additional_queries": ["query for gap 1", "query for gap 2"]
}}"""


async def evaluate_progress(
    query: str,
    plan: Dict[str, Any],
    subtopic_analyses: List[Dict[str, Any]],
    step: int,
    max_steps: int,
    total_docs: int,
    total_summaries: int,
    budget_spent: float,
    budget_total: float,
    avg_confidence: float,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Evaluate research progress and decide whether to continue.

    Args:
        query: Original research query.
        plan: Research plan from planner.
        subtopic_analyses: List of completed subtopic analysis results.
        step: Current iteration step.
        max_steps: Maximum allowed steps.
        total_docs: Total documents retrieved so far.
        total_summaries: Total relevant source summaries collected so far.
        budget_spent: USD spent so far.
        budget_total: Total budget.
        avg_confidence: Average confidence score from depth analyses.
        llm: LLM caller.
        model: Model for evaluation.

    Returns:
        Dict with scores, should_continue flag, and suggestions.
    """
    log.info("meta_eval_start", step=step)

    # Format planned subtopics
    planned = plan.get("subtopics", [])
    planned_names = [s.get("name", "") for s in planned]
    planned_text = "\n".join(f"- {name}" for name in planned_names)

    # Format completed analyses
    analyses_parts = []
    for a in subtopic_analyses:
        subtopic = a.get("subtopic", "Unknown")
        synthesis = a.get("synthesis", a.get("summary", "No summary"))[:300]
        confidence = a.get("confidence", 0)
        findings_count = len(a.get("key_findings", []))
        gaps = a.get("remaining_gaps", a.get("gaps", []))[:3]
        analyses_parts.append(
            f"### {subtopic}\n"
            f"Confidence: {confidence:.2f} | Findings: {findings_count}\n"
            f"Summary: {synthesis}\n"
            f"Remaining gaps: {', '.join(gaps) if gaps else 'None identified'}"
        )
    analyses_text = "\n\n".join(analyses_parts) if analyses_parts else "No analyses completed yet."

    result = await llm.complete_json(
        META_EVAL_PROMPT.format(
            query=query,
            plan_title=plan.get("title", ""),
            planned_subtopics=planned_text,
            analyses_summary=analyses_text,
            step=step,
            max_steps=max_steps,
            total_docs=total_docs,
            total_summaries=total_summaries,
            budget_spent=budget_spent,
            budget_total=budget_total,
            avg_confidence=avg_confidence,
        ),
        model=model,
        system=META_EVAL_SYSTEM,
        temperature=0.2,
        max_tokens=1024,
    )

    # Force stop if at max steps or near budget limit
    budget_remaining_pct = (budget_total - budget_spent) / budget_total if budget_total > 0 else 0
    if step >= max_steps - 1:
        result["should_continue"] = False
        result["rationale"] = (
            result.get("rationale", "") + " [Forced stop: max steps reached]"
        )
    elif budget_remaining_pct < 0.20:
        result["should_continue"] = False
        result["rationale"] = (
            result.get("rationale", "") + " [Forced stop: <20% budget remaining]"
        )

    log.info(
        "meta_eval_complete",
        step=step,
        coverage=result.get("coverage_score", 0),
        quality=result.get("quality_score", 0),
        overall=result.get("overall_score", 0),
        should_continue=result.get("should_continue", False),
    )

    return result
