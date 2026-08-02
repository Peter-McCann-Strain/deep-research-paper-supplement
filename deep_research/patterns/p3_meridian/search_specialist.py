"""MERIDIAN Role 1: Search Specialist — generates diverse queries, searches, and
extracts structured information from sources using an LLM.

Uses gpt-5.2 (cheap, fast) to produce 25 diverse sub-queries from the user's
research question, fans out web + academic searches, then has SourceExtractor
read and extract structured evidence from every retrieved document using a
two-step approach (free-text analysis then structured JSON extraction).

No chunking, embedding, vector-store, BM25, reranker, or fusion infrastructure.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import structlog

from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtractor,
    SourceExtraction,
    URLExtractor,
    get_web_searcher,
)
from deep_research.types import Document, SubQuery
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

MODEL = DEFAULT_MODEL
N_QUERIES = 25


# -- Query generation ----------------------------------------------------------

_QUERY_GEN_SYSTEM = (
    "You are a search-query specialist. Given a research question, generate diverse "
    "search queries that together will surface all the information needed for a "
    "comprehensive research report. Include queries that target:\n"
    "- Core definitions and background\n"
    "- Recent developments and news\n"
    "- Academic and peer-reviewed research\n"
    "- Statistics, data, and quantitative evidence\n"
    "- Contrasting viewpoints and debates\n"
    "- Practical applications and case studies\n"
    "- Expert opinions and authoritative sources\n"
    "- Historical context and evolution\n"
    "- Future trends and predictions\n"
    "Vary phrasing, specificity, and angle to maximise retrieval diversity."
)


async def generate_queries(
    query: str,
    llm: LLMCaller,
    n_queries: int = N_QUERIES,
) -> List[SubQuery]:
    """Generate *n_queries* diverse sub-queries for the research question."""
    prompt = (
        f"Research question: {query}\n\n"
        f"Generate exactly {n_queries} search queries as a JSON object with key "
        f'"queries", where each element has "query" (the search string) and '
        f'"intent" (brief label: background / recent / academic / data / debate '
        f"/ case_study / expert / history / trends)."
    )
    result = await llm.complete_json(
        prompt=prompt,
        model=MODEL,
        system=_QUERY_GEN_SYSTEM,
        temperature=0.7,
        max_tokens=3000,
    )

    raw_queries = result.get("queries", [])
    sub_queries: List[SubQuery] = []
    for i, q in enumerate(raw_queries[:n_queries]):
        sub_queries.append(
            SubQuery(
                query=q.get("query", q) if isinstance(q, dict) else str(q),
                intent=q.get("intent", "general") if isinstance(q, dict) else "general",
                priority=i + 1,
            )
        )

    log.info("search_specialist.queries_generated", count=len(sub_queries))
    return sub_queries


# -- Search execution ---------------------------------------------------------

async def execute_searches(
    sub_queries: List[SubQuery],
    web_searcher,
    academic_searcher: AcademicSearcher,
    max_web_per_query: int = 5,
    max_academic_per_source: int = 5,
) -> List[Document]:
    """Execute web and academic searches for all sub-queries.

    Returns deduplicated documents from both sources.
    """
    # Separate academic-intent queries
    academic_queries = [sq for sq in sub_queries if sq.intent in ("academic", "data")]
    web_queries = [sq.query for sq in sub_queries]

    # Academic queries (up to 5 unique queries to avoid rate-limits)
    academic_query_strings = list({sq.query for sq in academic_queries[:5]})

    log.info(
        "search_specialist.executing",
        web_queries=len(web_queries),
        academic_queries=len(academic_query_strings),
    )

    # Fan out web + academic in parallel
    web_task = web_searcher.search_batch(
        web_queries, max_results_per=max_web_per_query
    )
    academic_tasks = [
        academic_searcher.search(q, max_per_source=max_academic_per_source)
        for q in academic_query_strings
    ]

    web_docs, *academic_results = await asyncio.gather(
        web_task, *academic_tasks, return_exceptions=True
    )

    # Collect all docs
    all_docs: List[Document] = []
    if isinstance(web_docs, list):
        all_docs.extend(web_docs)
    else:
        log.warning("search_specialist.web_error", error=str(web_docs))

    for result in academic_results:
        if isinstance(result, list):
            all_docs.extend(result)
        elif isinstance(result, Exception):
            log.warning("search_specialist.academic_error", error=str(result))

    # Deduplicate by URL
    seen_urls: set = set()
    deduped: List[Document] = []
    for doc in all_docs:
        key = doc.url or doc.id
        if key not in seen_urls:
            seen_urls.add(key)
            deduped.append(doc)

    log.info(
        "search_specialist.documents_collected",
        total_raw=len(all_docs),
        deduplicated=len(deduped),
    )
    return deduped


# -- LLM-based source extraction -----------------------------------------------

async def extract_sources(
    documents: List[Document],
    query: str,
    llm: LLMCaller,
    model: str = MODEL,
) -> List[SourceExtraction]:
    """Read every document with the LLM and extract structured evidence.

    Uses the two-step SourceExtractor (free-text analysis then structured JSON)
    to produce SourceExtraction objects with key_findings, relevance_score,
    confidence_notes, and optional rich fields.

    Returns a list of SourceExtraction objects sorted by relevance_score desc.
    Documents the LLM judges irrelevant are filtered out.
    """
    extractor = SourceExtractor(llm=llm, model=model)
    extractions = await extractor.extract_batch(documents, query)

    log.info(
        "search_specialist.sources_extracted",
        total_docs=len(documents),
        relevant_extractions=len(extractions),
    )
    return extractions


# -- Top-level convenience -----------------------------------------------------

async def run_search_specialist(
    query: str,
    llm: LLMCaller,
    cost_tracker: CostTracker,
    n_queries: int = N_QUERIES,
    max_web_per_query: int = 5,
    max_academic_per_source: int = 5,
) -> Tuple[List[SubQuery], List[Document], List[SourceExtraction]]:
    """Full Search-Specialist pipeline: generate queries -> search -> extract.

    Returns:
        sub_queries, documents, extractions
    """
    web_searcher = get_web_searcher()
    academic_searcher = AcademicSearcher()

    # Step 1: Generate queries
    sub_queries = await generate_queries(query, llm, n_queries=n_queries)

    # Step 2: Execute searches
    documents = await execute_searches(
        sub_queries,
        web_searcher,
        academic_searcher,
        max_web_per_query=max_web_per_query,
        max_academic_per_source=max_academic_per_source,
    )

    # Step 2b: URL enrichment — fetch full page content for docs with short/missing content
    url_extractor = URLExtractor()
    urls_to_extract = [
        doc.url for doc in documents if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in documents:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]
        log.info("search_specialist.url_enrichment", enriched=len(url_to_content),
                 attempted=len(urls_to_extract))

    # Step 3: LLM reads and extracts structured evidence from each source
    extractions = await extract_sources(documents, query, llm)

    log.info(
        "search_specialist.complete",
        queries=len(sub_queries),
        docs=len(documents),
        extractions=len(extractions),
    )
    return sub_queries, documents, extractions
