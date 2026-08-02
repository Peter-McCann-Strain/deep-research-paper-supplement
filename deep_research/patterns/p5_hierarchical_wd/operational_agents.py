"""Operational agents: search-and-extract workers + analysis helpers.

Provides reusable worker functions for both the width and depth phases:
- SearchWorker: executes web + academic search, then LLM-extracts sources
- analyze_evidence: synthesizes findings from source extractions
- detect_gaps: identifies missing information in current coverage

All vector/BM25/embedding/chunker/reranker infrastructure has been replaced
with SourceExtractor (two-step LLM extraction approach).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

from deep_research.tools import (
    LLMCaller,
    get_web_searcher,
    AcademicSearcher,
    SourceExtractor,
    SourceExtraction,
    URLExtractor,
    format_extractions_as_evidence,
)
from deep_research.tools.cost_tracker import CostTracker
from deep_research.types import Document
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()


# ── Search Worker ────────────────────────────────────────────────────────────

class SearchWorker:
    """Executes search queries across web and academic sources, then LLM-extracts results."""

    def __init__(
        self,
        llm: LLMCaller,
        cost_tracker: CostTracker,
        worker_id: str = "w0",
    ):
        self.web = get_web_searcher()
        self.academic = AcademicSearcher()
        self.extractor = SourceExtractor(llm, model=DEFAULT_MODEL)
        self.url_extractor = URLExtractor()
        self.cost_tracker = cost_tracker
        self.worker_id = worker_id

    async def search_and_summarize(
        self,
        queries: List[str],
        research_query: str,
        max_web_per_query: int = 5,
        include_academic: bool = True,
        max_academic_per_query: int = 5,
    ) -> Dict[str, Any]:
        """Execute searches and LLM-extract the retrieved documents.

        Args:
            queries: Search queries to execute.
            research_query: The overarching research query (for extraction context).
            max_web_per_query: Max web results per search query.
            include_academic: Whether to include academic search sources.
            max_academic_per_query: Max academic results per query.

        Returns:
            Dict with:
                - docs: all retrieved Document objects
                - summaries: list of SourceExtraction objects
        """
        log.info(
            "search_worker_start",
            worker=self.worker_id,
            queries=len(queries),
        )

        # Web search (batch)
        web_docs = await self.web.search_batch(
            queries, max_results_per=max_web_per_query
        )
        log.info("search_worker_web", worker=self.worker_id, docs=len(web_docs))

        # Academic search (if enabled)
        academic_docs: List[Document] = []
        if include_academic and queries:
            for q in queries[:2]:
                try:
                    results = await self.academic.search(
                        q, max_per_source=max_academic_per_query
                    )
                    academic_docs.extend(results)
                except Exception as e:
                    log.warning(
                        "search_worker_academic_error",
                        worker=self.worker_id,
                        error=str(e),
                    )

        all_docs = web_docs + academic_docs

        if not all_docs:
            log.warning("search_worker_no_docs", worker=self.worker_id)
            return {"docs": [], "summaries": []}

        # URL enrichment — fetch full page content for docs with short snippets
        urls_to_extract = [
            doc.url for doc in all_docs if doc.url and len(doc.content) < 500
        ]
        if urls_to_extract:
            extracted = await self.url_extractor.extract_batch(urls_to_extract)
            url_to_content = {e.url: e.content for e in extracted if e.content}
            for doc in all_docs:
                if doc.url in url_to_content:
                    doc.content = url_to_content[doc.url]
            log.info(
                "search_worker_url_enrichment",
                worker=self.worker_id,
                enriched=len(url_to_content),
                attempted=len(urls_to_extract),
            )

        # LLM-extract each source against the research query
        extractions = await self.extractor.extract_batch(all_docs, research_query)

        log.info(
            "search_worker_extracted",
            worker=self.worker_id,
            docs=len(all_docs),
            extractions=len(extractions),
        )

        return {"docs": all_docs, "summaries": extractions}


# ── Analysis Helper ──────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a research analyst. Synthesize the provided source summaries
into a focused analysis. Be precise, cite specific findings, and note gaps."""

ANALYSIS_PROMPT = """Analyze the following source summaries related to this research question.

Research Question: {question}
Subtopic Context: {context}

Evidence (from source summaries):
{evidence}

Provide a focused analysis in JSON:
{{
    "summary": "2-3 paragraph synthesis of key findings",
    "key_findings": ["finding1", "finding2", ...],
    "data_points": ["specific statistic or fact 1", ...],
    "gaps": ["what information is still missing"],
    "confidence": 0.0-1.0,
    "sources_used": ["source_title1", "source_title2"]
}}"""


async def analyze_evidence(
    question: str,
    context: str,
    summaries: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Synthesize source extractions into a structured analysis.

    Args:
        question: The specific research question to answer.
        context: Broader subtopic context.
        summaries: List of SourceExtraction objects from SourceExtractor.
        llm: LLM caller instance.
        model: Model to use for analysis.

    Returns:
        Dict with summary, key_findings, data_points, gaps, confidence, sources_used.
    """
    if not summaries:
        return {
            "summary": "No evidence found for this question.",
            "key_findings": [],
            "data_points": [],
            "gaps": [question],
            "confidence": 0.0,
            "sources_used": [],
        }

    # Format source extractions as numbered evidence blocks
    evidence_text = format_extractions_as_evidence(summaries)

    result = await llm.complete_json(
        ANALYSIS_PROMPT.format(
            question=question,
            context=context,
            evidence=evidence_text,
        ),
        model=model,
        system=ANALYSIS_SYSTEM,
        temperature=0.3,
        max_tokens=2048,
    )

    return result


# ── Gap Detector ─────────────────────────────────────────────────────────────

GAP_PROMPT = """Given the original research query and the analyses completed so far,
identify remaining gaps in coverage.

Original Query: {query}

Completed Analyses:
{analyses_summary}

Subtopics Not Yet Covered:
{uncovered}

Return JSON:
{{
    "gaps": [
        {{
            "topic": "gap topic",
            "description": "what is missing",
            "search_queries": ["suggested query 1", "suggested query 2"],
            "priority": 1
        }}
    ],
    "coverage_estimate": 0.0-1.0
}}"""


async def detect_gaps(
    query: str,
    completed_analyses: List[Dict[str, Any]],
    uncovered_subtopics: List[str],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Detect gaps in current research coverage.

    Returns:
        Dict with gaps list and coverage_estimate.
    """
    # Summarize completed analyses
    analyses_parts = []
    for i, analysis in enumerate(completed_analyses, 1):
        summary = analysis.get("summary", "No summary")[:200]
        findings = analysis.get("key_findings", [])[:3]
        analyses_parts.append(
            f"{i}. {summary}\n   Findings: {', '.join(findings)}"
        )
    analyses_summary = "\n".join(analyses_parts) if analyses_parts else "None yet."

    uncovered_text = "\n".join(
        f"- {s}" for s in uncovered_subtopics
    ) if uncovered_subtopics else "All subtopics have been addressed."

    result = await llm.complete_json(
        GAP_PROMPT.format(
            query=query,
            analyses_summary=analyses_summary,
            uncovered=uncovered_text,
        ),
        model=model,
        temperature=0.3,
        max_tokens=1024,
    )

    return result
