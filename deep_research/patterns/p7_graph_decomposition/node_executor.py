"""Execute a single graph node: generate search queries, search, extract, summarise.

Each node goes through:
    1. LLM generates 2-3 targeted search queries
    2. Web + academic search
    3. URL extraction for thin results
    4. Two-step source extraction
    5. LLM summarises findings into a node answer
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import (
    AcademicSearcher,
    SourceExtractor,
    SourceExtraction,
    URLExtractor,
    format_extractions_as_evidence,
)
from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import Document

from .graph import GraphNode, NodeStatus

log = structlog.get_logger()


SEARCH_QUERY_PROMPT = """Generate 2-3 effective web search queries to answer this research sub-question.
The queries should be specific, diverse, and likely to surface high-quality sources.

Original research topic: {original_query}
Sub-question to answer: {question}

Return ONLY valid JSON: {{"queries": ["query1", "query2", "query3"]}}
"""

SUMMARISE_PROMPT = """Based on the evidence below, provide a comprehensive answer to the sub-question.
Include specific facts, data points, and citations where available.
If the evidence is insufficient, note what information is missing.

Sub-question: {question}

Evidence:
{evidence}

Write a thorough 300-600 word answer:"""


async def _generate_search_queries(
    llm: LLMCaller,
    question: str,
    original_query: str,
) -> List[str]:
    """Ask the LLM for 2-3 targeted search queries for a sub-question."""
    try:
        result = await llm.complete_json(
            SEARCH_QUERY_PROMPT.format(
                original_query=original_query,
                question=question,
            ),
            model=DEFAULT_MODEL,
            max_tokens=512,
            temperature=0.3,
        )
        queries = result.get("queries", [])
        if isinstance(queries, list) and queries:
            return queries[:6]
    except Exception as exc:
        log.warning("search_query_gen_failed", error=str(exc))

    # Fallback: use the sub-question itself
    return [question]


async def execute_node(
    node: GraphNode,
    llm: LLMCaller,
    web_searcher,
    academic_searcher: AcademicSearcher,
    url_extractor: URLExtractor,
    source_extractor: SourceExtractor,
    query: str,
) -> None:
    """Execute a single graph node end-to-end.

    Modifies *node* in place (status, answer, extractions, etc.).
    Raises no exceptions — failures are recorded as NodeStatus.FAILED.
    """
    node.status = NodeStatus.IN_PROGRESS
    log.info("node_execute_start", node_id=node.id, question=node.question[:80])

    try:
        # ── 1. Generate search queries ───────────────────────────────
        search_queries = await _generate_search_queries(llm, node.question, query)
        node.search_queries_used = search_queries

        # ── 2. Web search ────────────────────────────────────────────
        web_docs: List[Document] = await web_searcher.search_batch(
            search_queries, max_results_per=5,
        )
        log.info("node_web_search", node_id=node.id, docs=len(web_docs))

        # ── 3. Academic search ───────────────────────────────────────
        academic_docs: List[Document] = await academic_searcher.search(
            node.question, max_per_source=3,
        )
        log.info("node_academic_search", node_id=node.id, docs=len(academic_docs))

        # ── 4. Deduplicate by URL ────────────────────────────────────
        seen_urls: Dict[str, bool] = {}
        all_docs: List[Document] = []
        for doc in web_docs + academic_docs:
            if doc.url and doc.url not in seen_urls:
                seen_urls[doc.url] = True
                all_docs.append(doc)
        node.n_docs_found = len(all_docs)

        # ── 5. Extract full page content where thin ──────────────────
        urls_to_extract = [
            doc.url for doc in all_docs
            if doc.url and len(doc.content) < 500
        ]
        if urls_to_extract:
            extracted = await url_extractor.extract_batch(urls_to_extract)
            url_to_content = {e.url: e.content for e in extracted if e.content}
            for doc in all_docs:
                if doc.url in url_to_content:
                    doc.content = url_to_content[doc.url]
            log.info("node_url_extract", node_id=node.id, extracted=len(url_to_content))

        # ── 6. Two-step source extraction ────────────────────────────
        extractions = await source_extractor.extract_batch(all_docs, node.question)
        node.extractions = extractions
        log.info("node_source_extract", node_id=node.id, sources=len(all_docs), relevant=len(extractions))

        # ── 7. Summarise into an answer ──────────────────────────────
        if extractions:
            evidence = format_extractions_as_evidence(extractions)
            answer = await llm.complete(
                SUMMARISE_PROMPT.format(
                    question=node.question,
                    evidence=evidence,
                ),
                model=DEFAULT_MODEL,
                max_tokens=2048,
                temperature=0.3,
            )
            node.answer = answer
        else:
            node.answer = (
                f"No relevant sources were found for: {node.question}. "
                "This sub-question may require more specialised search terms "
                "or the topic may not have sufficient publicly available coverage."
            )

        node.status = NodeStatus.COMPLETED
        log.info(
            "node_execute_done",
            node_id=node.id,
            status="completed",
            n_extractions=len(extractions),
        )

    except Exception as exc:
        node.status = NodeStatus.FAILED
        node.answer = f"Execution failed: {exc}"
        log.warning("node_execute_failed", node_id=node.id, error=str(exc))
