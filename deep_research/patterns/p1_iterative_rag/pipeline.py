"""Pattern 1: Iterative RAG Pipeline — search, extract, generate, reflect.

Flow:
    Query -> Decompose (gpt-5.2)
    -> Search (Tavily web + academic) -> Extract page content
    -> Two-step source extraction (gpt-5.2)
    -> Generate report (gpt-5.2)
    -> Reflect on quality (gpt-5.2) -> loop or finish
    -> Final Report
"""

from __future__ import annotations

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.tools import CostTracker, LLMCaller, StateManager
from deep_research.types import ProcessTrace, ResearchReport, SubQuery

from .generator import generate_report
from .query_decomposer import decompose_query
from .reflector import reflect
from .report_assembler import assemble_report
from .retriever import Retriever

log = structlog.get_logger()

MAX_REFLECTION_LOOPS = 3


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    max_iterations: int = MAX_REFLECTION_LOOPS,
    **kwargs,
) -> ResearchReport:
    """Execute the full iterative RAG pipeline.

    Args:
        query: The research question.
        budget_usd: Maximum budget in USD.
        max_iterations: Maximum reflection loop iterations (default 3).
        **kwargs: Absorbs unknown ablation parameters gracefully.
    """
    tracker = CostTracker(budget_usd=budget_usd)
    # B2 external-validity switch: run the orchestration on a local 7B backbone when
    # DR_LOCAL_LLM is set, so P1-vs-P9 isolates the orchestration premium at 7B.
    import os as _os
    if _os.environ.get("DR_LOCAL_LLM"):
        from deep_research.tools.local_llm_caller import LocalLLMCaller
        llm = LocalLLMCaller(model_id=_os.environ.get("DR_LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                             cost_tracker=tracker, quantize_4bit=True)
    else:
        llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p1_iterative_rag")
    retriever = Retriever(llm=llm, cost_tracker=tracker)
    trace = ProcessTrace(pattern_name="p1_iterative_rag", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p1_start", query=query[:80], max_iterations=max_iterations)

    # ── Stage 1: Decompose query ─────────────────────────────────────────
    log.info("stage_1_decompose")
    tokens_before_decompose = tracker.total_tokens
    sub_queries = await decompose_query(query, llm, n_queries=25)
    trace.append(tool="decompose", input_args={"n_queries": 25},
                 output_summary=f"{len(sub_queries)} sub-queries",
                 n_results=len(sub_queries),
                 tokens_used=tracker.total_tokens - tokens_before_decompose)
    state.save("decompose", {"queries": [sq.model_dump() for sq in sub_queries]})

    # ── Stage 2: Search + Extract ────────────────────────────────────────
    log.info("stage_2_search_and_extract")
    extractions = await retriever.search_and_summarize(sub_queries, query)
    trace.append(tool="search", input_args={"n_sub_queries": len(sub_queries)},
                 output_summary=f"{len(extractions)} extractions across sub-queries",
                 n_results=len(extractions))
    trace.append(tool="source_extract", input_args={"n_sub_queries": len(sub_queries)},
                 output_summary=f"{len(extractions)} relevant extractions",
                 n_results=len(extractions))
    state.save("search", {
        "extraction_count": len(extractions),
        "extractions": [e.to_evidence_dict() for e in extractions],
    })

    if not extractions:
        log.warning("no_relevant_sources_found")

    # ── Reflection loop ──────────────────────────────────────────────────
    report_md = ""
    n_iterations = 0

    for loop_i in range(max_iterations + 1):
        log.info("iteration", loop=loop_i)
        n_iterations += 1

        # Stage 3: Generate report
        log.info("stage_3_generate", sources=len(extractions))
        tokens_before_gen = tracker.total_tokens
        report_md = await generate_report(query, extractions, llm)
        trace.append(tool="generate", input_args={"loop": loop_i, "n_extractions": len(extractions)},
                     output_summary=f"{len(report_md)}-char report (loop {loop_i})",
                     tokens_used=tracker.total_tokens - tokens_before_gen)
        state.save(f"report_v{loop_i}", {"markdown": report_md})

        # Stage 4: Reflect (skip on last possible iteration)
        if loop_i < max_iterations:
            log.info("stage_4_reflect")
            tokens_before_reflect = tracker.total_tokens
            reflection = await reflect(query, report_md, llm)
            trace.append(tool="reflect", input_args={"loop": loop_i},
                         output_summary=f"score={reflection.get('overall_score')}, continue={reflection.get('should_continue', False)}",
                         tokens_used=tracker.total_tokens - tokens_before_reflect)
            state.save(f"reflection_v{loop_i}", reflection)

            if not reflection.get("should_continue", False):
                log.info(
                    "reflection_satisfied",
                    score=reflection.get("overall_score"),
                )
                break

            # Gap-fill: search for missing topics
            gap_queries = reflection.get("improvement_queries", [])
            if gap_queries:
                log.info("gap_fill", new_queries=len(gap_queries))
                gap_subs = [
                    SubQuery(query=q, intent="gap_fill", priority=1)
                    for q in gap_queries
                ]
                new_extractions = await retriever.search_and_summarize(
                    gap_subs, query
                )
                trace.append(tool="search", input_args={"n_gap_queries": len(gap_queries), "loop": loop_i},
                             output_summary=f"{len(new_extractions)} gap-fill extractions",
                             n_results=len(new_extractions))
                extractions.extend(new_extractions)
                log.info("gap_fill_done", new_sources=len(new_extractions),
                         total_sources=len(extractions))

    # ── Assemble final report ────────────────────────────────────────────
    report = assemble_report(
        query=query,
        markdown=report_md,
        extractions=extractions,
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
    )

    log.info(
        "p1_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        sections=len(report.sections),
    )
    state.save("final", {"cost": tracker.total_cost, "tokens": tracker.total_tokens})

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [sq.query for sq in sub_queries]
    report.metadata["search_queries_sent"] = [sq.query for sq in sub_queries]
    report.metadata["n_documents_retrieved"] = len(extractions)
    report.metadata["n_extractions"] = len(extractions)

    trace.final_report_word_count = len(report_md.split())
    trace.n_iterations = n_iterations
    state.save("trace", trace.model_dump(mode="json"))

    return report
