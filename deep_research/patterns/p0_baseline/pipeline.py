"""Pattern 0: Baseline — single LLM call with search context.

The simplest possible approach: search for sources, stuff them into a single
LLM prompt, and ask for a research report in one shot. No reflection, no
iteration, no multi-agent orchestration. This establishes the floor that
all other patterns must beat to justify their complexity.

Flow:
    Query -> Web search (top 10 results)
    -> Extract page content
    -> Two-step source extraction (same shared tool)
    -> Single LLM call: "write a research report using these sources"
    -> Parse into ResearchReport
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtraction,
    SourceExtractor,
    URLExtractor,
    get_web_searcher,
    format_extractions_as_evidence,
    StateManager,
)
from deep_research.types import ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

log = structlog.get_logger()


REPORT_PROMPT = """You are a research analyst. Write a comprehensive, well-structured research report answering the following query. Use ONLY the provided source evidence. Cite sources using inline numbered references like [1], [2], etc.

Research query: {query}

Source evidence:
{evidence}

Requirements:
- Start with a title (# Title)
- Include an abstract (## Abstract)
- Organize into logical sections (## Section Name)
- End with a References section listing all cited sources
- Be comprehensive, accurate, and balanced
- Use inline citations [1], [2], etc. throughout
- Aim for 2000-4000 words

Write the full research report:"""


async def run(query: str, budget_usd: float = MAX_COST_PER_RUN, **kwargs) -> ResearchReport:
    """Execute the baseline: search + single-shot report generation."""
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p0_baseline")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    trace = ProcessTrace(pattern_name="p0_baseline", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p0_start", query=query[:80])

    # ── Stage 1: Web search (single query, top 10) ───────────────────────
    log.info("stage_1_search")
    web_docs = await web.search_batch([query], max_results_per=10)
    trace.append(tool="search", input_args={"query": query, "max_results": 10},
                 output_summary=f"{len(web_docs)} web docs", n_results=len(web_docs))

    # Also grab a few academic results
    academic_docs = await academic.search(query, max_per_source=5)
    trace.append(tool="academic_search", input_args={"query": query, "max_per_source": 5},
                 output_summary=f"{len(academic_docs)} academic docs", n_results=len(academic_docs))

    # Deduplicate by URL
    seen_urls: set = set()
    all_docs = []
    for doc in web_docs + academic_docs:
        if doc.url and doc.url not in seen_urls:
            seen_urls.add(doc.url)
            all_docs.append(doc)

    log.info("search_done", web=len(web_docs), academic=len(academic_docs),
             deduped=len(all_docs))

    # ── Stage 2: Extract page content where missing ──────────────────────
    urls_to_extract = [
        doc.url for doc in all_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in all_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]
        log.info("page_extract_done", pages=len(url_to_content))
        trace.append(tool="extract", input_args={"n_urls": len(urls_to_extract)},
                     output_summary=f"{len(url_to_content)} pages extracted", n_results=len(url_to_content))

    # ── Stage 3: Two-step source extraction ──────────────────────────────
    log.info("stage_3_source_extraction")
    extractions = await source_extractor.extract_batch(all_docs, query)
    trace.append(tool="source_extract", input_args={"n_docs": len(all_docs)},
                 output_summary=f"{len(extractions)} relevant extractions", n_results=len(extractions))

    state.save("search", {
        "doc_count": len(all_docs),
        "extraction_count": len(extractions),
        "extractions": [e.to_evidence_dict() for e in extractions],
    })

    if not extractions:
        log.warning("no_relevant_sources")
        return ResearchReport(
            query=query,
            title=query,
            pattern_name="p0_baseline",
            total_cost_usd=tracker.total_cost,
            total_tokens=tracker.total_tokens,
        )

    # ── Stage 4: Single-shot report generation ───────────────────────────
    log.info("stage_4_generate", sources=len(extractions))
    evidence_text = format_extractions_as_evidence(extractions)

    tokens_before_gen = tracker.total_tokens
    report_md = await llm.complete(
        REPORT_PROMPT.format(query=query, evidence=evidence_text),
        model=DEFAULT_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )
    trace.append(tool="generate", input_args={"max_tokens": 8192},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    state.save("report", {"markdown": report_md})

    # ── Assemble ─────────────────────────────────────────────────────────
    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=extractions,
        pattern_name="p0_baseline",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
    )
    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len(seen_urls)
    state.save("trace", trace.model_dump(mode="json"))

    log.info("p0_complete",
             cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens,
             sections=len(report.sections))

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
    })

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = []
    report.metadata["search_queries_sent"] = [query]
    report.metadata["n_documents_retrieved"] = len(web_docs) + len(academic_docs)
    report.metadata["n_documents_after_dedup"] = len(all_docs)
    report.metadata["n_extractions"] = len(extractions)

    return report


def _assemble_report(
    query: str,
    markdown: str,
    extractions: List[SourceExtraction],
    cost_usd: float,
    total_tokens: int,
) -> ResearchReport:
    """Parse markdown into structured ResearchReport.

    .. deprecated:: 0.2
        Use :func:`deep_research.utils.markdown_parser.parse_markdown_report`
        directly. This wrapper is kept for backward compatibility with tests.
    """
    return parse_markdown_report(
        query=query,
        markdown=markdown,
        extractions=extractions,
        pattern_name="p0_baseline",
        cost_usd=cost_usd,
        total_tokens=total_tokens,
    )
