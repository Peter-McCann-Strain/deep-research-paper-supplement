"""Quality Gate: evaluates completeness of aggregated findings, identifies gaps."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import SourceExtraction, format_extractions_as_evidence
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# -- Prompt --------------------------------------------------------------------

QUALITY_GATE_PROMPT = """You are a research quality evaluator. Assess whether the gathered
evidence is sufficient to produce a comprehensive research report.

Original query: {query}

Research plan title: {title}

Sub-topics investigated:
{sub_topics}

Worker summaries:
{summaries}

Source evidence (numbered):
{evidence}

Evaluate on these criteria (score each 1-10):
1. **Coverage**: Are all major aspects of the query addressed?
2. **Depth**: Is there sufficient detail and specificity for each aspect?
3. **Source diversity**: Do findings come from multiple independent sources?
4. **Evidence quality**: Are claims supported by concrete data, numbers, or citations?
5. **Balance**: Are contrasting viewpoints or limitations represented?

Return JSON:
{{
  "scores": {{
    "coverage": <1-10>,
    "depth": <1-10>,
    "source_diversity": <1-10>,
    "evidence_quality": <1-10>,
    "balance": <1-10>
  }},
  "overall_score": <1-10>,
  "gaps": ["list of specific missing topics, weak areas, or unanswered questions"],
  "has_critical_gaps": <true if important aspects are entirely missing>,
  "feedback": "brief overall assessment with specific improvement suggestions"
}}
"""


async def evaluate_quality(
    query: str,
    plan_title: str,
    sub_topics: List[Dict[str, Any]],
    worker_summaries: Dict[str, str],
    source_summaries: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
    threshold: float = 7.0,
) -> QualityResult:
    """Evaluate the quality and completeness of gathered research.

    Args:
        query: original research query.
        plan_title: title from the supervisor's plan.
        sub_topics: list of sub-topic dicts from the plan.
        worker_summaries: dict mapping sub-topic query -> narrative summary text.
        source_summaries: list of SourceExtraction objects from SourceExtractor.
        llm: LLMCaller instance.
        model: model to use for evaluation.
        threshold: minimum overall score to pass the gate.

    Returns:
        QualityResult with scores, gaps, and pass/fail status.
    """
    sub_topics_str = "\n".join(
        f"- [{t.get('priority', '?')}] {t.get('query', '?')} ({t.get('intent', '')})"
        for t in sub_topics
    )

    summaries_str = ""
    for topic, summary in worker_summaries.items():
        summaries_str += f"\n### {topic}\n{summary[:3000]}\n"

    evidence_str = format_extractions_as_evidence(source_summaries[:20])

    result = await llm.complete_json(
        QUALITY_GATE_PROMPT.format(
            query=query,
            title=plan_title,
            sub_topics=sub_topics_str,
            summaries=summaries_str,
            evidence=evidence_str,
        ),
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )

    scores = result.get("scores", {})
    overall = result.get("overall_score", 5)
    gaps = result.get("gaps", [])
    has_critical_gaps = result.get("has_critical_gaps", False)
    feedback = result.get("feedback", "")

    passed = overall >= threshold and not has_critical_gaps

    log.info(
        "quality_gate",
        overall_score=overall,
        passed=passed,
        gaps=len(gaps),
        critical=has_critical_gaps,
        scores=scores,
    )

    return QualityResult(
        scores=scores,
        overall_score=overall,
        gaps=gaps,
        has_critical_gaps=has_critical_gaps,
        feedback=feedback,
        passed=passed,
    )


class QualityResult:
    """Container for quality gate evaluation output."""

    def __init__(
        self,
        scores: Dict[str, int],
        overall_score: float,
        gaps: List[str],
        has_critical_gaps: bool,
        feedback: str,
        passed: bool,
    ):
        self.scores = scores
        self.overall_score = overall_score
        self.gaps = gaps
        self.has_critical_gaps = has_critical_gaps
        self.feedback = feedback
        self.passed = passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": self.scores,
            "overall_score": self.overall_score,
            "gaps": self.gaps,
            "has_critical_gaps": self.has_critical_gaps,
            "feedback": self.feedback,
            "passed": self.passed,
        }
