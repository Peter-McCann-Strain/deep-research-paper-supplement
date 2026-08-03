"""Deep investigation — full search, extraction, and analysis for surviving beams.

After the first beam selection, surviving directions receive a much deeper
search budget: more queries, academic sources, full URL extraction, and
a detailed LLM-generated analysis.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.academic_search import AcademicSearcher
from deep_research.tools.source_extractor import (
    SourceExtraction,
    SourceExtractor,
    format_extractions_as_evidence,
)
from deep_research.tools.url_extractor import URLExtractor
from deep_research.types import Document

from .hypothesis_generator import ResearchDirection

log = structlog.get_logger()


# ── Query generation prompt ──────────────────────────────────────────────────

_DEEP_QUERIES_PROMPT = """Generate 6-8 specific search queries to deeply investigate this research direction.

Original Query: {query}

Direction being investigated:
- Thesis: {thesis}
- Key Questions: {key_questions}
- Angle: {angle}

Evidence already gathered: {n_existing_extractions} sources

Generate queries that would find:
- Specific data, statistics, and benchmarks
- Expert opinions and authoritative sources
- Case studies and real-world examples
- Counterarguments and alternative viewpoints
- Recent developments and updates

IMPORTANT: Make queries specific and varied. Avoid repeating the initial search queries.

Return JSON: {{"queries": ["q1", "q2", "q3", "q4", "q5", "q6"]}}"""


# ── Analysis prompt ──────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """Provide a detailed analysis for this research direction based on the evidence gathered.

Original Research Query: {query}

Direction: {thesis}
Angle: {angle}
Key Questions: {key_questions}

Source Evidence:
{evidence}

Write a thorough analysis covering:
1. Key findings from the evidence, with specific data points and citations
2. How this direction directly addresses the original research query
3. Areas of consensus across sources
4. Counterarguments, limitations, and knowledge gaps
5. The most important takeaway from this direction

Requirements:
- Be specific: cite source numbers [1], [2], etc.
- Be analytical, not just descriptive — interpret the evidence
- Highlight surprising or non-obvious findings
- Note the strength/weakness of the evidence base
- Aim for 800-1200 words"""


# ── Deep investigation ───────────────────────────────────────────────────────


async def investigate_direction(
    direction: ResearchDirection,
    llm: LLMCaller,
    web_searcher,
    academic_searcher: AcademicSearcher,
    url_extractor: URLExtractor,
    source_extractor: SourceExtractor,
    query: str,
) -> None:
    """Deep investigation of a surviving beam direction.

    Mutates ``direction`` in place, extending its extractions and populating
    ``detailed_analysis``.

    Args:
        direction: A surviving research direction.
        llm: LLM caller instance.
        web_searcher: Web search tool.
        academic_searcher: Academic search tool.
        url_extractor: URL content extractor.
        source_extractor: Two-step source extractor.
        query: The original research query.
    """
    log.info(
        "deep_investigate_start",
        direction_id=direction.id,
        thesis=direction.thesis[:60],
        existing_extractions=len(direction.extractions),
    )

    # ── Step 1: Generate deep search queries ─────────────────────────────
    deep_queries = await _generate_deep_queries(llm, direction, query)

    # ── Step 2: Web search with expanded budget ──────────────────────────
    web_docs = await _web_search(web_searcher, deep_queries)

    # ── Step 3: Academic search ──────────────────────────────────────────
    academic_docs = await _academic_search(academic_searcher, direction, query)

    # ── Step 4: Combine and deduplicate ──────────────────────────────────
    all_new_docs = web_docs + academic_docs
    existing_urls = {e.url for e in direction.extractions if e.url}
    unique_docs: List[Document] = []
    seen_urls: set = set(existing_urls)
    for doc in all_new_docs:
        key = doc.url or doc.id
        if key not in seen_urls:
            seen_urls.add(key)
            unique_docs.append(doc)

    log.info(
        "deep_docs_collected",
        direction_id=direction.id,
        web=len(web_docs),
        academic=len(academic_docs),
        unique_new=len(unique_docs),
    )

    # ── Step 5: URL enrichment for thin content ──────────────────────────
    urls_to_enrich = [
        doc.url for doc in unique_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_enrich:
        try:
            enriched = await url_extractor.extract_batch(urls_to_enrich[:20])
            url_to_content = {e.url: e.content for e in enriched if e.content}
            for doc in unique_docs:
                if doc.url in url_to_content:
                    doc.content = url_to_content[doc.url]
            log.info(
                "deep_url_enriched",
                direction_id=direction.id,
                attempted=len(urls_to_enrich),
                enriched=len(url_to_content),
            )
        except Exception as exc:
            log.warning(
                "deep_url_enrichment_failed",
                direction_id=direction.id,
                error=str(exc),
            )

    # ── Step 6: Full two-step source extraction ──────────────────────────
    if unique_docs:
        try:
            new_extractions = await source_extractor.extract_batch(
                unique_docs, direction.thesis, max_concurrent=5
            )
        except Exception as exc:
            log.warning(
                "deep_extraction_failed",
                direction_id=direction.id,
                error=str(exc),
            )
            new_extractions = []

        # Merge with existing extractions
        direction.extractions.extend(new_extractions)
        direction.n_docs_found += len(unique_docs)

        # Update search queries used
        direction.search_queries_used.extend(deep_queries)

    log.info(
        "deep_extraction_done",
        direction_id=direction.id,
        total_extractions=len(direction.extractions),
    )

    # ── Step 7: Generate detailed analysis ───────────────────────────────
    direction.detailed_analysis = await _generate_analysis(
        llm, direction, query
    )

    log.info(
        "deep_investigate_done",
        direction_id=direction.id,
        total_extractions=len(direction.extractions),
        analysis_chars=len(direction.detailed_analysis),
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


async def _generate_deep_queries(
    llm: LLMCaller,
    direction: ResearchDirection,
    query: str,
) -> List[str]:
    """Generate 6-8 deep search queries for a direction."""
    prompt = _DEEP_QUERIES_PROMPT.format(
        query=query,
        thesis=direction.thesis,
        key_questions="; ".join(direction.key_questions),
        angle=direction.angle,
        n_existing_extractions=len(direction.extractions),
    )

    try:
        result = await llm.complete_json(
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=512,
            temperature=0.5,
        )
        queries = result.get("queries", [])
        queries = [str(q) for q in queries if q][:12]
    except Exception as exc:
        log.warning("deep_query_gen_failed", error=str(exc))
        # Fallback: use thesis + key questions as queries
        queries = [direction.thesis] + direction.key_questions

    log.info(
        "deep_queries_generated",
        direction_id=direction.id,
        count=len(queries),
    )
    return queries


async def _web_search(
    web_searcher,
    queries: List[str],
) -> List[Document]:
    """Execute web search with expanded results budget."""
    try:
        docs = await web_searcher.search_batch(queries, max_results_per=8)
        return docs
    except Exception as exc:
        log.warning("deep_web_search_failed", error=str(exc))
        return []


async def _academic_search(
    academic_searcher: AcademicSearcher,
    direction: ResearchDirection,
    query: str,
) -> List[Document]:
    """Search academic sources for the direction."""
    academic_docs: List[Document] = []
    # Use thesis and first key question for academic search
    academic_queries = [direction.thesis]
    if direction.key_questions:
        academic_queries.append(direction.key_questions[0])

    for aq in academic_queries:
        try:
            results = await academic_searcher.search(aq, max_per_source=5)
            academic_docs.extend(results)
        except Exception as exc:
            log.warning("deep_academic_search_failed", query=aq[:60], error=str(exc))

    return academic_docs


async def _generate_analysis(
    llm: LLMCaller,
    direction: ResearchDirection,
    query: str,
) -> str:
    """Generate a detailed analysis of the direction's evidence."""
    if not direction.extractions:
        return f"No evidence was found for direction: {direction.thesis}"

    evidence = format_extractions_as_evidence(direction.extractions)

    prompt = _ANALYSIS_PROMPT.format(
        query=query,
        thesis=direction.thesis,
        angle=direction.angle,
        key_questions="; ".join(direction.key_questions),
        evidence=evidence,
    )

    try:
        analysis = await llm.complete(
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=3000,
            temperature=0.3,
        )
        return analysis.strip()
    except Exception as exc:
        log.error("deep_analysis_failed", direction_id=direction.id, error=str(exc))
        # Fallback: concatenate evidence summaries
        summaries = [e.summary for e in direction.extractions if e.summary]
        return "\n\n".join(summaries[:5]) if summaries else ""
