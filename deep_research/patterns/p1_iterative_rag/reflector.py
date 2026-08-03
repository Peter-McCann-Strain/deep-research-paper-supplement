"""Stage 7: Self-reflection and gap identification."""

from __future__ import annotations

from typing import List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

REFLECT_PROMPT = """You are a research quality reviewer. Evaluate this research report and identify gaps.

Query: {query}

Report:
{report}

Evaluate on:
1. Coverage: Are all aspects of the query addressed?
2. Evidence quality: Are claims supported by specific citations?
3. Balance: Are multiple perspectives represented?
4. Specificity: Are there concrete numbers, dates, methods?
5. Coherence: Does the argument flow logically?

Return JSON:
{{
  "overall_score": <1-10>,
  "gaps": ["list of specific missing topics or weak areas"],
  "improvement_queries": ["new search queries to fill gaps"],
  "should_continue": <true if score < 7 and gaps exist>,
  "feedback": "brief overall assessment"
}}
"""


async def reflect(
    query: str,
    report: str,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Reflect on report quality and identify gaps."""
    result = await llm.complete_json(
        REFLECT_PROMPT.format(query=query, report=report),
        model=model,
    )
    log.info("reflection",
             score=result.get("overall_score"),
             gaps=len(result.get("gaps", [])),
             continue_=result.get("should_continue"))
    return result
