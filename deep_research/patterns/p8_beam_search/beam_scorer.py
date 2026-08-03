"""Beam scoring and selection — rank research directions by promise and prune.

Each direction is scored by an LLM on relevance, evidence availability, and
novelty.  The top-B directions survive; the rest are pruned.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller

from .hypothesis_generator import ResearchDirection

log = structlog.get_logger()


# ── Scoring prompt ───────────────────────────────────────────────────────────

_SCORE_PROMPT = """Rate this research direction for investigating the query below.

Query: {query}

Direction: {thesis}
Angle: {angle}
Key Questions: {key_questions}
Evidence found: {n_extractions} sources ({n_docs} documents retrieved)
Evidence summary:
{evidence_summary}

Rate each dimension on a scale of 1-10:
- relevance: How directly does this direction address the research question?
- evidence_availability: How much supporting evidence was found in the initial search?
- novelty: Does this direction offer unique insights not covered by a simple overview?

Then provide an overall_promise score (1-10) that reflects the direction's potential \
for producing a strong research report section.

Return JSON:
{{"relevance": <1-10>, "evidence_availability": <1-10>, "novelty": <1-10>, "overall_promise": <1-10>, "reasoning": "..."}}"""


# ── Score one direction ──────────────────────────────────────────────────────


async def _score_one(
    llm: LLMCaller,
    direction: ResearchDirection,
    query: str,
) -> float:
    """Score a single research direction, returning promise_score."""
    prompt = _SCORE_PROMPT.format(
        query=query,
        thesis=direction.thesis,
        angle=direction.angle,
        key_questions="; ".join(direction.key_questions),
        n_extractions=len(direction.extractions),
        n_docs=direction.n_docs_found,
        evidence_summary=direction.evidence_summary[:20000],
    )

    try:
        result = await llm.complete_json(
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=512,
            temperature=0.1,
        )
        overall = float(result.get("overall_promise", 5))
        overall = max(1.0, min(10.0, overall))
        reasoning = result.get("reasoning", "")
        log.info(
            "direction_scored",
            direction_id=direction.id,
            score=overall,
            relevance=result.get("relevance"),
            evidence=result.get("evidence_availability"),
            novelty=result.get("novelty"),
            reasoning=reasoning[:80],
        )
        return overall
    except Exception as exc:
        log.warning(
            "direction_score_failed",
            direction_id=direction.id,
            error=str(exc),
        )
        # Fallback: use a moderate score based on evidence count
        fallback = min(10.0, max(1.0, 3.0 + len(direction.extractions) * 0.5))
        return fallback


# ── Score all directions ─────────────────────────────────────────────────────


async def score_directions(
    llm: LLMCaller,
    directions: List[ResearchDirection],
    query: str,
) -> List[ResearchDirection]:
    """Score all directions and sort by promise_score descending.

    Mutates each direction's ``promise_score`` in place.

    Args:
        llm: LLM caller instance.
        directions: All candidate directions.
        query: The original research query.

    Returns:
        The same list sorted by promise_score (highest first).
    """
    import asyncio

    tasks = [_score_one(llm, d, query) for d in directions]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    for d, score in zip(directions, scores):
        if isinstance(score, Exception):
            log.warning("score_exception", direction_id=d.id, error=str(score))
            d.promise_score = 3.0
        else:
            d.promise_score = score

    # Sort descending by promise_score
    directions.sort(key=lambda d: d.promise_score, reverse=True)

    log.info(
        "directions_ranked",
        ranking=[(d.id, d.promise_score) for d in directions],
    )
    return directions


# ── Beam selection ───────────────────────────────────────────────────────────


def select_beam(
    directions: List[ResearchDirection],
    beam_width: int,
) -> List[ResearchDirection]:
    """Keep the top ``beam_width`` directions, mark the rest as pruned.

    Directions should already be sorted by score (call ``score_directions``
    first).

    Args:
        directions: Sorted list of directions (best first).
        beam_width: Number of beams to keep alive.

    Returns:
        List of surviving directions only.
    """
    surviving: List[ResearchDirection] = []
    pruned_ids: List[str] = []

    for i, d in enumerate(directions):
        if i < beam_width:
            d.is_alive = True
            surviving.append(d)
        else:
            d.is_alive = False
            pruned_ids.append(d.id)

    log.info(
        "beam_selected",
        beam_width=beam_width,
        surviving=[d.id for d in surviving],
        pruned=pruned_ids,
    )
    return surviving


# ── Re-scoring for second beam selection ─────────────────────────────────────

_RESCORE_PROMPT = """Re-evaluate this research direction after deep investigation.

Query: {query}

Direction: {thesis}
Angle: {angle}
Number of sources after deep search: {n_extractions}
Detailed analysis:
{detailed_analysis}

Rate the quality of evidence and analysis on a scale of 1-10:
- evidence_depth: How deep and well-supported are the findings?
- evidence_breadth: How many different facets of the direction are covered?
- analytical_quality: How insightful and well-reasoned is the analysis?
- contribution: How much does this direction contribute to the overall research?

Return JSON:
{{"evidence_depth": <1-10>, "evidence_breadth": <1-10>, "analytical_quality": <1-10>, "contribution": <1-10>, "overall_quality": <1-10>, "reasoning": "..."}}"""


async def rescore_directions(
    llm: LLMCaller,
    directions: List[ResearchDirection],
    query: str,
) -> List[ResearchDirection]:
    """Re-score surviving directions after deep investigation.

    Updates ``evidence_quality_score`` on each direction.

    Args:
        llm: LLM caller instance.
        directions: Surviving directions with detailed_analysis populated.
        query: The original research query.

    Returns:
        The same list sorted by evidence_quality_score (highest first).
    """
    import asyncio

    async def _rescore_one(d: ResearchDirection) -> float:
        prompt = _RESCORE_PROMPT.format(
            query=query,
            thesis=d.thesis,
            angle=d.angle,
            n_extractions=len(d.extractions),
            detailed_analysis=d.detailed_analysis[:20000],
        )
        try:
            result = await llm.complete_json(
                prompt,
                model=DEFAULT_MODEL,
                max_tokens=512,
                temperature=0.1,
            )
            score = float(result.get("overall_quality", 5))
            score = max(1.0, min(10.0, score))
            log.info(
                "direction_rescored",
                direction_id=d.id,
                quality_score=score,
                reasoning=result.get("reasoning", "")[:80],
            )
            return score
        except Exception as exc:
            log.warning("rescore_failed", direction_id=d.id, error=str(exc))
            return d.promise_score  # Fall back to initial promise score

    tasks = [_rescore_one(d) for d in directions]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    for d, score in zip(directions, scores):
        if isinstance(score, Exception):
            d.evidence_quality_score = d.promise_score
        else:
            d.evidence_quality_score = score

    directions.sort(key=lambda d: d.evidence_quality_score, reverse=True)

    log.info(
        "directions_rescored",
        ranking=[(d.id, d.evidence_quality_score) for d in directions],
    )
    return directions
