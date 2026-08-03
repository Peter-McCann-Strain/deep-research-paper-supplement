"""Pattern 2: Supervisor + Parallel Workers Pipeline.

Architecture:
    Query -> Supervisor plans research subtopics (gpt-5.2)
    -> Dispatch N parallel workers (each worker: search -> read sources -> LLM extract, gpt-5.2)
    -> Aggregate worker extractions -> Quality gate (gpt-5.2, check for gaps)
    -> Gap-fill if needed (1 iteration) -> Compress findings (gpt-5.2)
    -> Write final report (gpt-5.2)

No embedding, FAISS, BM25, chunking, or reranking — pure LLM-reads-and-extracts approach.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.tools import CostTracker, LLMCaller, StateManager
from deep_research.types import ProcessTrace, ResearchReport

from .supervisor import plan_research, plan_gap_fill
from .search_worker import SearchWorker
from .aggregator import Aggregator
from .quality_gate import evaluate_quality
from .compressor import compress_findings, generate_abstract
from .report_writer import write_report, assemble_research_report

log = structlog.get_logger()

pattern_name = "p2_supervisor_parallel"

# Maximum gap-fill iterations before forcing report generation
MAX_GAP_FILL_ITERATIONS = 1

# Default number of parallel workers
DEFAULT_N_WORKERS = 5

# Number of gap-fill workers per iteration
GAP_FILL_WORKERS = 2


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    max_workers: int = DEFAULT_N_WORKERS,
    skip_quality_gate: bool = False,
    **kwargs,
) -> ResearchReport:
    """Execute the full Supervisor + Parallel Workers pipeline.

    Stages:
        1. Supervisor plans research strategy (gpt-5.2)
        2. N workers search, read sources, and LLM-extract in parallel (gpt-5.2)
        3. Aggregator deduplicates and collects source extractions
        4. Quality gate evaluates completeness (gpt-5.2)
        5. Optional gap-fill: supervisor generates new tasks, workers fill gaps (1 iteration)
        6. Compressor distils findings into sections (gpt-5.2)
        7. Report writer produces final output (gpt-5.2)

    Args:
        query: the research question to investigate.
        budget_usd: maximum cost budget for the entire run.
        max_workers: number of parallel workers (default 5). Set to 1 for sequential.
        skip_quality_gate: if True, skip quality gate evaluation and gap-fill loop.
        **kwargs: Absorbs unknown ablation parameters gracefully.

    Returns:
        ResearchReport with pattern_name="p2_supervisor_parallel".
    """
    t0 = time.time()
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager(pattern_name)
    aggregator = Aggregator()
    trace = ProcessTrace(pattern_name=pattern_name, query=query, query_id=kwargs.get("query_id", ""))

    log.info("p2_start", query=query[:80], budget=budget_usd,
             max_workers=max_workers, skip_quality_gate=skip_quality_gate)

    # -- Stage 1: Supervisor plans research ------------------------------------
    log.info("stage_1_plan")
    tokens_before_plan = tracker.total_tokens
    plan = await plan_research(query, llm, n_workers=DEFAULT_N_WORKERS)
    sub_topics = plan["sub_topics"]
    plan_title = plan["title"]
    trace.append(tool="decompose", input_args={"n_workers": DEFAULT_N_WORKERS},
                 output_summary=f"{len(sub_topics)} sub-topics: {plan_title[:60]}",
                 n_results=len(sub_topics),
                 tokens_used=tracker.total_tokens - tokens_before_plan)

    state.save("plan", plan)
    log.info("stage_1_done", title=plan_title, sub_topics=len(sub_topics))

    # -- Stage 2: Dispatch parallel workers ------------------------------------
    log.info("stage_2_workers", count=len(sub_topics), max_workers=max_workers)
    worker_results = await _dispatch_workers(
        sub_topics, tracker, max_concurrency=max_workers,
    )
    trace.append(tool="search",
                 input_args={"n_workers": len(sub_topics), "max_concurrency": max_workers},
                 output_summary=f"{len(worker_results)} workers; {sum(len(wr.documents) for wr in worker_results)} docs",
                 n_results=sum(len(wr.documents) for wr in worker_results))
    trace.append(tool="source_extract",
                 input_args={"n_workers": len(worker_results)},
                 output_summary=f"{sum(len(wr.source_summaries) for wr in worker_results)} extractions",
                 n_results=sum(len(wr.source_summaries) for wr in worker_results))

    state.save("workers", {
        "results": [wr.to_dict() for wr in worker_results],
        "total_docs": sum(len(wr.documents) for wr in worker_results),
        "total_extractions": sum(len(wr.source_summaries) for wr in worker_results),
    })

    # -- Stage 3: Aggregate and deduplicate ------------------------------------
    log.info("stage_3_aggregate")
    agg_output = await aggregator.aggregate(worker_results, query)

    state.save("aggregate", {
        **agg_output.to_dict(),
        "source_extractions": [s.to_evidence_dict() for s in agg_output.source_summaries],
    })
    log.info(
        "stage_3_done",
        docs=len(agg_output.documents),
        source_extractions=len(agg_output.source_summaries),
    )

    # -- Stage 4: Quality gate -------------------------------------------------
    gap_iteration = 0
    if skip_quality_gate:
        log.info("stage_4_quality_gate_skipped")
        # Create a minimal quality object for downstream bookkeeping
        from types import SimpleNamespace
        quality = SimpleNamespace(
            passed=True, overall_score=0.0, gaps=[], feedback="",
            to_dict=lambda: {"passed": True, "skipped": True, "overall_score": 0.0},
        )
        state.save("quality_gate", quality.to_dict())
    else:
        log.info("stage_4_quality_gate")
        tokens_before_quality = tracker.total_tokens
        quality = await evaluate_quality(
            query=query,
            plan_title=plan_title,
            sub_topics=sub_topics,
            worker_summaries=agg_output.worker_summaries,
            source_summaries=agg_output.source_summaries,
            llm=llm,
        )
        trace.append(tool="reflect",
                     input_args={"stage": "quality_gate"},
                     output_summary=f"score={quality.overall_score:.2f}, passed={quality.passed}, gaps={len(quality.gaps)}",
                     tokens_used=tracker.total_tokens - tokens_before_quality)

        state.save("quality_gate", quality.to_dict())

        # -- Stage 5: Gap-fill (if quality gate fails, up to 1 iteration) ----------
        while not quality.passed and gap_iteration < MAX_GAP_FILL_ITERATIONS:
            gap_iteration += 1
            log.info("stage_5_gap_fill", iteration=gap_iteration,
                     gaps=len(quality.gaps), score=quality.overall_score)

            # Supervisor generates gap-fill sub-topics
            covered_topics = [t["query"] for t in sub_topics]
            gap_topics = await plan_gap_fill(
                query=query,
                plan_title=plan_title,
                covered_topics=covered_topics,
                gaps=quality.gaps,
                feedback=quality.feedback,
                llm=llm,
                n_gap_workers=GAP_FILL_WORKERS,
            )

            if not gap_topics:
                log.info("no_gap_topics_generated, proceeding")
                break

            # Dispatch gap-fill workers
            gap_results = await _dispatch_workers(
                gap_topics, tracker, worker_id_offset=len(sub_topics),
                max_concurrency=max_workers,
            )
            trace.append(tool="search",
                         input_args={"stage": "gap_fill", "iteration": gap_iteration, "n_gap_topics": len(gap_topics)},
                         output_summary=f"{len(gap_results)} gap-fill workers completed",
                         n_results=sum(len(wr.documents) for wr in gap_results))

            state.save(f"gap_fill_{gap_iteration}", {
                "gap_topics": gap_topics,
                "results": [wr.to_dict() for wr in gap_results],
            })

            # Add gap topics to the overall plan for tracking
            sub_topics.extend(gap_topics)

            # Re-aggregate with new results
            agg_output = await aggregator.aggregate_additional(
                new_results=gap_results,
                query=query,
                existing_output=agg_output,
            )

            # Re-evaluate quality
            quality = await evaluate_quality(
                query=query,
                plan_title=plan_title,
                sub_topics=sub_topics,
                worker_summaries=agg_output.worker_summaries,
                source_summaries=agg_output.source_summaries,
                llm=llm,
            )

            state.save(f"quality_gate_{gap_iteration}", quality.to_dict())
            log.info("gap_fill_quality", iteration=gap_iteration,
                     score=quality.overall_score, passed=quality.passed)

    # -- Stage 6: Compress findings --------------------------------------------
    log.info("stage_6_compress")
    compressed = await compress_findings(
        query=query,
        worker_summaries=agg_output.worker_summaries,
        source_summaries=agg_output.source_summaries,
        llm=llm,
        n_sections=5,
    )

    abstract = await generate_abstract(
        query=query,
        sections=compressed.sections,
        llm=llm,
    )

    state.save("compress", {
        "sections": compressed.to_dict(),
        "abstract_length": len(abstract),
    })

    # -- Stage 7: Write final report -------------------------------------------
    log.info("stage_7_report")
    tokens_before_gen = tracker.total_tokens
    report_md = await write_report(
        query=query,
        abstract=abstract,
        compressed=compressed,
        source_summaries=agg_output.source_summaries,
        llm=llm,
    )
    trace.append(tool="generate",
                 input_args={"n_sections": len(compressed.sections)},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    elapsed = time.time() - t0

    # Assemble structured report
    report = assemble_research_report(
        query=query,
        report_markdown=report_md,
        source_summaries=agg_output.source_summaries,
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )

    # -- Final bookkeeping -----------------------------------------------------
    state.save("final", {
        "cost_usd": tracker.total_cost,
        "total_tokens": tracker.total_tokens,
        "elapsed_seconds": elapsed,
        "sections": len(report.sections),
        "citations": len(report.citations),
        "quality_score": quality.overall_score,
        "gap_fill_iterations": gap_iteration,
    })

    log.info(
        "p2_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        elapsed=f"{elapsed:.1f}s",
        sections=len(report.sections),
        citations=len(report.citations),
        quality_score=quality.overall_score,
    )

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [t["query"] for t in sub_topics]
    report.metadata["search_queries_sent"] = [t["query"] for t in sub_topics]
    report.metadata["n_documents_retrieved"] = sum(len(wr.documents) for wr in worker_results)
    report.metadata["n_documents_after_dedup"] = len(agg_output.documents)
    report.metadata["n_extractions"] = len(agg_output.source_summaries)

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len(agg_output.documents)
    trace.n_iterations = gap_iteration
    state.save("trace", trace.model_dump(mode="json"))

    return report


async def _dispatch_workers(
    sub_topics: list,
    cost_tracker: CostTracker,
    worker_id_offset: int = 0,
    max_concurrency: int = 0,
) -> list:
    """Dispatch parallel search workers for the given sub-topics.

    Creates one SearchWorker per sub-topic and runs them concurrently
    via asyncio.gather. Exceptions from individual workers are caught
    and logged rather than crashing the entire pipeline.

    Args:
        sub_topics: list of sub-topic dicts from the supervisor plan.
        cost_tracker: shared CostTracker for budget enforcement.
        worker_id_offset: starting worker ID (for gap-fill workers).
        max_concurrency: max parallel workers. 0 = unlimited (all parallel).

    Returns:
        List of WorkerResult objects (failed workers are excluded).
    """
    from .search_worker import WorkerResult

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def _run_one(worker_id: int, topic: dict) -> WorkerResult | None:
        worker = SearchWorker(worker_id=worker_id, cost_tracker=cost_tracker)
        try:
            if semaphore:
                async with semaphore:
                    return await worker.run(topic)
            return await worker.run(topic)
        except Exception as exc:
            log.warning("worker_failed", worker_id=worker_id, error=str(exc))
            return None

    tasks = [
        _run_one(i + worker_id_offset, topic)
        for i, topic in enumerate(sub_topics)
    ]

    results = await asyncio.gather(*tasks)

    # Filter out None (failed workers)
    successful = [r for r in results if r is not None]
    log.info("workers_done", dispatched=len(tasks), successful=len(successful))
    return successful
