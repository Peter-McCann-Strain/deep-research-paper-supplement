"""Pattern 4: Perspective-Driven STORM — main orchestration pipeline.

Architecture:
    Query -> Discover 4-6 perspectives (gpt-5.2)
    -> Search for each perspective (Tavily + academic) -> Two-step source extraction (gpt-5.2)
    -> Simulated expert conversations using extractions as context (gpt-5.2, parallel pairs)
    -> Build mind map from conversations (gpt-5.2)
    -> Triangulate claims across perspectives (gpt-5.2)
    -> Synthesize final report (gpt-5.2)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import structlog

from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtractor,
    SourceExtraction,
    StateManager,
    URLExtractor,
    get_web_searcher,
)
from deep_research.types import (
    Document,
    Perspective,
    ProcessTrace,
    ResearchReport,
)

from .perspective_discovery import discover_perspectives, generate_search_queries
from .conversation_sim import run_all_conversations, extract_all_conversation_text
from .mind_map import build_mind_map, build_outline, extract_triangulation_queries
from .triangulator import triangulate
from .synthesizer import synthesize, assemble_report
from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN

log = structlog.get_logger()


async def _search_and_extract(
    queries: List[str],
    query: str,
    web,
    academic: AcademicSearcher,
    extractor: SourceExtractor,
    max_web_per_query: int = 5,
    max_academic_per_query: int = 5,
    do_academic: bool = True,
    max_academic_queries: int = 5,
) -> tuple[List[Document], List[SourceExtraction]]:
    """Search web and academic sources, then extract structured information.

    Args:
        queries: Search queries to execute.
        query: The main research query (used for relevance-focused extraction).
        web: Web search tool.
        academic: Academic search tool.
        extractor: Two-step source extractor (free-text analysis then structured JSON).
        max_web_per_query: Max web results per query.
        max_academic_per_query: Max academic results per query.
        do_academic: Whether to also search academic sources.

    Returns:
        Tuple of (all_docs, extractions) where extractions is a list of
        SourceExtraction objects with structured fields.
    """
    # Web search
    web_docs = await web.search_batch(queries, max_results_per=max_web_per_query)
    log.info("web_search_done", queries=len(queries), docs=len(web_docs))

    # Academic search (only for first few queries to conserve budget)
    academic_docs: List[Document] = []
    if do_academic:
        academic_queries = queries[:max_academic_queries]
        for q in academic_queries:
            results = await academic.search(q, max_per_source=max_academic_per_query)
            academic_docs.extend(results)
        log.info("academic_search_done", docs=len(academic_docs))

    all_docs = web_docs + academic_docs

    if not all_docs:
        log.warning("no_docs_from_search")
        return all_docs, []

    # Deduplicate documents by URL
    seen_urls: set = set()
    unique_docs: List[Document] = []
    for doc in all_docs:
        key = doc.url or doc.id
        if key not in seen_urls:
            seen_urls.add(key)
            unique_docs.append(doc)

    log.info("deduped_docs", before=len(all_docs), after=len(unique_docs))

    # URL enrichment — fetch full page content for docs with short/missing content
    url_extractor = URLExtractor()
    urls_to_extract = [
        doc.url for doc in unique_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in unique_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]
        log.info("url_enrichment_done", enriched=len(url_to_content),
                 attempted=len(urls_to_extract))

    # Two-step extraction: free-text analysis then structured JSON
    extractions = await extractor.extract_batch(unique_docs, query)
    log.info("sources_extracted", docs=len(unique_docs), relevant=len(extractions))

    return unique_docs, extractions


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    skip_conversations: bool = False,
    skip_triangulation: bool = False,
    fixed_perspectives: bool = False,
    n_perspectives: int = 5,
    **kwargs,
) -> ResearchReport:
    """Execute the Perspective-Driven STORM pipeline.

    Stages:
        1. Discover 4-6 expert perspectives (gpt-5.2)
        2. Search for each perspective (Tavily + academic) and extract sources (gpt-5.2)
        3. Simulate expert conversations using extractions as context (gpt-5.2, parallel)
        4. Build mind map from conversation insights (gpt-5.2)
        5. Triangulate claims across perspectives and sources (gpt-5.2)
        6. Synthesize findings into final report (gpt-5.2)

    Args:
        query: The research question.
        budget_usd: Maximum budget for this run in USD.
        skip_conversations: if True, skip conversation simulation, go from search to synthesis.
        skip_triangulation: if True, skip triangulation, pass mind_map directly to synthesizer.
        fixed_perspectives: if True, use 3 fixed generic perspectives instead of LLM-discovered.
        n_perspectives: number of perspectives to discover (default 5).
        **kwargs: Absorbs unknown ablation parameters gracefully.

    Returns:
        A structured ResearchReport with pattern_name="p4_perspective_storm".
    """
    t0 = time.time()
    tracker = CostTracker(budget_usd=budget_usd)
    import os as _os
    if _os.environ.get("DR_LOCAL_LLM"):
        from deep_research.tools.local_llm_caller import LocalLLMCaller
        llm = LocalLLMCaller(model_id=_os.environ.get("DR_LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                             cost_tracker=tracker, quantize_4bit=True)
    else:
        llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p4_perspective_storm")
    extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    trace = ProcessTrace(pattern_name="p4_perspective_storm", query=query, query_id=kwargs.get("query_id", ""))

    # Shared search tools
    web = get_web_searcher()
    academic = AcademicSearcher()

    log.info("p4_start", query=query[:80], budget=budget_usd,
             skip_conversations=skip_conversations,
             skip_triangulation=skip_triangulation,
             fixed_perspectives=fixed_perspectives,
             n_perspectives=n_perspectives)

    # == Stage 1: Discover perspectives ========================================
    if fixed_perspectives:
        log.info("stage_1_fixed_perspectives")
        perspectives = [
            Perspective(
                name="Technical Expert",
                description="Deep technical knowledge of the subject matter, "
                            "focusing on implementation details and mechanisms.",
                focus_areas=["technical details", "implementation", "mechanisms"],
            ),
            Perspective(
                name="Industry Practitioner",
                description="Real-world industry experience with practical applications, "
                            "market dynamics, and operational considerations.",
                focus_areas=["practical applications", "market dynamics", "operations"],
            ),
            Perspective(
                name="Academic Researcher",
                description="Scholarly perspective emphasising theoretical frameworks, "
                            "empirical evidence, and research methodology.",
                focus_areas=["theory", "empirical evidence", "methodology"],
            ),
        ]
    else:
        log.info("stage_1_perspective_discovery")
        perspectives = await discover_perspectives(query, llm, n_perspectives=n_perspectives)
    trace.append(tool="decompose",
                 input_args={"stage": "perspective_discovery", "n_perspectives": n_perspectives},
                 output_summary=f"{len(perspectives)} perspectives: {[p.name for p in perspectives]}",
                 n_results=len(perspectives))
    state.save("perspectives", {
        "perspectives": [p.model_dump() for p in perspectives],
    })

    # == Stage 2: Search + source extraction ====================================
    log.info("stage_2_search_and_extract")
    search_plan = await generate_search_queries(query, perspectives, llm)
    state.save("search_plan", search_plan)

    # Collect all queries
    all_search_queries: List[str] = list(search_plan.get("general_queries", []))
    perspective_queries_map: Dict[str, List[str]] = search_plan.get("perspective_queries", {})
    for pname, pqueries in perspective_queries_map.items():
        all_search_queries.extend(pqueries)

    # Deduplicate while preserving order
    seen_queries: set = set()
    unique_queries: List[str] = []
    for q in all_search_queries:
        q_lower = q.lower().strip()
        if q_lower not in seen_queries:
            seen_queries.add(q_lower)
            unique_queries.append(q)

    log.info("search_queries_ready", total=len(unique_queries))

    # Execute searches and extract structured source information
    all_docs, all_extractions = await _search_and_extract(
        unique_queries,
        query=query,
        web=web,
        academic=academic,
        extractor=extractor,
        max_web_per_query=5,
        max_academic_per_query=5,
        do_academic=True,
    )
    trace.append(tool="search",
                 input_args={"n_queries": len(unique_queries)},
                 output_summary=f"{len(all_docs)} docs",
                 n_results=len(all_docs))
    trace.append(tool="source_extract",
                 input_args={"n_docs": len(all_docs)},
                 output_summary=f"{len(all_extractions)} relevant extractions",
                 n_results=len(all_extractions))
    state.save("search_results", {
        "doc_count": len(all_docs),
        "extraction_count": len(all_extractions),
        "extractions": [e.to_evidence_dict() for e in all_extractions],
    })

    # Build perspective-specific extraction sets for conversation context.
    # Each perspective gets extractions from its own queries plus general queries.
    general_queries_lower = {
        gq.lower().strip() for gq in search_plan.get("general_queries", [])
    }

    # Build a mapping: doc_id -> extraction for easy lookup
    extraction_by_id: Dict[str, SourceExtraction] = {}
    for e in all_extractions:
        extraction_by_id[e.doc_id or e.url] = e

    # For simplicity, give every perspective access to all extractions.
    # The conversation prompt already focuses through the perspective lens.
    perspective_extractions: Dict[str, List[SourceExtraction]] = {}
    for perspective in perspectives:
        perspective_extractions[perspective.name] = list(all_extractions)

    log.info("perspective_extractions_built",
             perspectives={k: len(v) for k, v in perspective_extractions.items()})

    # == Stage 3: Simulated conversations ======================================
    if skip_conversations:
        log.info("stage_3_conversations_skipped")
        # Build a minimal conversations_text from extractions directly
        extraction_summaries = []
        for e in all_extractions:
            findings = "; ".join(e.key_findings) if e.key_findings else ""
            extraction_summaries.append(
                f"Source: {e.title}\nSummary: {e.summary}\n"
                f"Key Findings: {findings}"
            )
        conversations_text = "\n\n".join(extraction_summaries)
        conversations = []
        state.save("conversations", {"count": 0, "skipped": True})
    else:
        log.info("stage_3_conversations")
        tokens_before_conv = tracker.total_tokens
        conversations = await run_all_conversations(
            perspectives=perspectives,
            query=query,
            perspective_extractions=perspective_extractions,
            llm=llm,
            model=DEFAULT_MODEL,
            n_turns=3,
        )
        trace.append(tool="tool_call",
                     input_args={"stage": "conversations", "n_perspectives": len(perspectives), "n_turns": 3},
                     output_summary=f"{len(conversations)} conversation pairs",
                     n_results=len(conversations),
                     tokens_used=tracker.total_tokens - tokens_before_conv)
        state.save("conversations", {
            "count": len(conversations),
            "pairs": [(c["interviewer"], c["expert"]) for c in conversations],
        })

        # Extract full conversation text for mind mapping
        conversations_text = extract_all_conversation_text(conversations)

    # == Stage 4: Mind map and outline =========================================
    log.info("stage_4_mind_map")
    mind_map = await build_mind_map(
        query=query,
        perspectives=perspectives,
        conversations_text=conversations_text,
        llm=llm,
        model=DEFAULT_MODEL,
    )
    state.save("mind_map", mind_map)

    outline = await build_outline(
        query=query,
        mind_map=mind_map,
        llm=llm,
        model=DEFAULT_MODEL,
    )
    state.save("outline", outline)

    # == Triangulation pre-search and triangulation ============================
    triangulation_extractions: List[SourceExtraction] = []
    triangulation_result: Dict[str, Any] = {}

    if skip_triangulation:
        log.info("stage_5_triangulate_skipped")
        # Extract claims from mind map's topic_clusters[].key_claims[]
        # and convert to the verified_claims format the synthesizer expects.
        extracted_claims = []
        for cluster in mind_map.get("topic_clusters", []):
            for claim in cluster.get("key_claims", []):
                extracted_claims.append({
                    "claim": claim.get("claim", ""),
                    "verdict": "uncertain",
                    "evidence_strength": "weak",
                    "supporting_sources": [],
                    "num_independent_sources": 0,
                    "perspectives_supporting": claim.get("supporting_perspectives", []),
                    "caveats": "Not triangulated (triangulation skipped)",
                    "confidence_score": {"high": 0.7, "medium": 0.5, "low": 0.3}.get(
                        claim.get("confidence", "medium"), 0.5
                    ),
                })
        triangulation_result = {
            "skipped": True,
            "verified_claims": extracted_claims,
            "unverified_claims": [],
            "contradictions": [],
        }
        state.save("triangulation", triangulation_result)
    else:
        triangulation_queries = extract_triangulation_queries(mind_map)

        if triangulation_queries:
            log.info("triangulation_pre_search", queries=len(triangulation_queries))
            _tri_docs, tri_extractions = await _search_and_extract(
                triangulation_queries,
                query=query,
                web=web,
                academic=academic,
                extractor=extractor,
                max_web_per_query=3,
                do_academic=False,
            )

            # Deduplicate against primary extractions
            existing_ids = {e.doc_id or e.url for e in all_extractions}
            for e in tri_extractions:
                eid = e.doc_id or e.url
                if eid not in existing_ids:
                    triangulation_extractions.append(e)
                    existing_ids.add(eid)

            log.info("triangulation_extractions", new=len(triangulation_extractions))

        # == Stage 5: Triangulate ==================================================
        log.info("stage_5_triangulate")
        tokens_before_tri = tracker.total_tokens
        triangulation_result = await triangulate(
            query=query,
            mind_map=mind_map,
            evidence_extractions=all_extractions,
            triangulation_extractions=triangulation_extractions,
            llm=llm,
            model=DEFAULT_MODEL,
        )
        trace.append(tool="triangulate",
                     input_args={"n_evidence": len(all_extractions),
                                 "n_triangulation": len(triangulation_extractions)},
                     output_summary=f"{len(triangulation_result.get('verified_claims', []))} verified claims",
                     n_results=len(triangulation_result.get("verified_claims", [])),
                     tokens_used=tracker.total_tokens - tokens_before_tri)
        state.save("triangulation", triangulation_result)

    # == Stage 6: Synthesize ===================================================
    log.info("stage_6_synthesize")

    # Combine primary + triangulation extractions for the synthesizer
    combined_extractions = list(all_extractions) + triangulation_extractions

    tokens_before_gen = tracker.total_tokens
    report_md = await synthesize(
        query=query,
        outline=outline,
        triangulation=triangulation_result,
        perspectives=perspectives,
        evidence_extractions=combined_extractions,
        llm=llm,
        model=DEFAULT_MODEL,
    )
    trace.append(tool="generate",
                 input_args={"stage": "synthesize", "n_extractions": len(combined_extractions)},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)
    state.save("report_markdown", {"markdown": report_md})

    # == Assemble final report =================================================
    elapsed = time.time() - t0
    report = assemble_report(
        query=query,
        markdown=report_md,
        evidence_extractions=combined_extractions,
        perspectives=perspectives,
        triangulation=triangulation_result,
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )

    log.info("p4_complete",
             cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens,
             sections=len(report.sections),
             elapsed=f"{elapsed:.1f}s")
    log.info("p4_cost_breakdown", summary=tracker.summary_text())

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed_seconds": elapsed,
        "sections": len(report.sections),
        "citations": len(report.citations),
    })

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [p.name for p in perspectives]
    report.metadata["search_queries_sent"] = unique_queries
    report.metadata["n_documents_retrieved"] = len(all_docs)
    report.metadata["n_documents_after_dedup"] = len(all_docs)
    report.metadata["n_extractions"] = len(combined_extractions)

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len({d.url for d in all_docs if d.url})
    state.save("trace", trace.model_dump(mode="json"))

    return report
