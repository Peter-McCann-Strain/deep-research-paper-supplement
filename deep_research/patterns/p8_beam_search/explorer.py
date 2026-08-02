"""Broad exploration — quick search and lightweight extraction for each beam direction.

Each direction gets a small number of search queries and limited results to
quickly gauge evidence availability before the first beam selection.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import (
    SourceExtraction,
    SourceExtractor,
    format_extractions_as_evidence,
)
from deep_research.types import Document

from .hypothesis_generator import ResearchDirection

log = structlog.get_logger()


async def explore_direction(
    direction: ResearchDirection,
    llm: LLMCaller,
    web_searcher,
    source_extractor: SourceExtractor,
    query: str,
) -> None:
    """Quick exploration of a single research direction.

    Mutates ``direction`` in place, populating evidence fields:
    - search_queries_used
    - n_docs_found
    - extractions
    - evidence_summary

    Args:
        direction: The research direction to explore.
        llm: LLM caller instance.
        web_searcher: Web search tool (BingSearcher or WebSearcher).
        source_extractor: Two-step source extractor.
        query: The original research query (for context).
    """
    # Build quick search queries from thesis + key questions
    search_queries: List[str] = [direction.thesis]
    for q in direction.key_questions[:2]:
        search_queries.append(q)
    direction.search_queries_used = list(search_queries)

    log.info(
        "explore_direction_start",
        direction_id=direction.id,
        angle=direction.angle,
        n_queries=len(search_queries),
    )

    # Quick search with limited results
    try:
        docs: List[Document] = await web_searcher.search_batch(
            search_queries, max_results_per=5
        )
    except Exception as exc:
        log.warning(
            "explore_search_failed",
            direction_id=direction.id,
            error=str(exc),
        )
        docs = []

    direction.n_docs_found = len(docs)

    if not docs:
        log.info(
            "explore_no_docs",
            direction_id=direction.id,
            thesis=direction.thesis[:60],
        )
        return

    # Lightweight two-step extraction (still uses the full pipeline, but on fewer docs)
    try:
        extractions: List[SourceExtraction] = await source_extractor.extract_batch(
            docs, direction.thesis, max_concurrent=3
        )
    except Exception as exc:
        log.warning(
            "explore_extraction_failed",
            direction_id=direction.id,
            error=str(exc),
        )
        extractions = []

    direction.extractions = extractions

    # Build brief evidence summary for scoring
    if extractions:
        evidence_text = format_extractions_as_evidence(extractions[:15])
        direction.evidence_summary = evidence_text[:20000]

    log.info(
        "explore_direction_done",
        direction_id=direction.id,
        docs_found=len(docs),
        extractions=len(extractions),
        evidence_chars=len(direction.evidence_summary),
    )
