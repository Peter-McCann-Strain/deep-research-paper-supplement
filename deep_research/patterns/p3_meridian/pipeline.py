"""Pattern 3: MERIDIAN 4-Role Pipeline — orchestration module.

Flow:
    Query
    -> Search Specialist (gpt-5.2: 25 queries, search, LLM-extract sources)
    -> Topic Miner (gpt-5.2: cluster extractions into topics)
    -> Writer (gpt-5.2: write report from topic clusters)
    -> Evaluator (3 judges in parallel: gpt-5.2 + gpt-5.2 + gpt-5.2)
    -> revise (if avg < 7) or finalise with quality scores
"""

from __future__ import annotations

import time
from typing import Any, Dict

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.tools import CostTracker, LLMCaller, StateManager
from deep_research.types import ProcessTrace, ResearchReport

from .search_specialist import run_search_specialist
from .topic_miner import run_topic_miner
from .research_writer import revise_report, write_report
from .quality_evaluator import (
    MAX_REVISIONS,
    REVISION_THRESHOLD,
    evaluate_report,
)

log = structlog.get_logger()

PATTERN_NAME = "p3_meridian"


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    skip_evaluation: bool = False,
    skip_topic_mining: bool = False,
    **kwargs,
) -> ResearchReport:
    """Execute the full MERIDIAN 4-role pipeline.

    Args:
        query: The user's research question.
        budget_usd: Maximum dollar spend for the entire run.
        skip_evaluation: if True, skip quality evaluation/revision, take first draft.
        skip_topic_mining: if True, skip topic miner, create a single catch-all topic.
        **kwargs: Absorbs unknown ablation parameters gracefully.

    Returns:
        A ResearchReport with pattern_name="p3_meridian" and quality metadata.
    """
    t0 = time.time()

    # -- Shared infrastructure -------------------------------------------------
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager(PATTERN_NAME)
    trace = ProcessTrace(pattern_name=PATTERN_NAME, query=query, query_id=kwargs.get("query_id", ""))

    log.info("p3_start", query=query[:100], budget=f"${budget_usd:.2f}",
             skip_evaluation=skip_evaluation, skip_topic_mining=skip_topic_mining)

    # -- Role 1: Search Specialist ---------------------------------------------
    log.info("p3_role1_search_specialist")
    tokens_before_role1 = tracker.total_tokens
    sub_queries, documents, extractions = await run_search_specialist(
        query=query,
        llm=llm,
        cost_tracker=tracker,
        n_queries=25,
        max_web_per_query=5,
        max_academic_per_source=5,
    )
    trace.append(tool="decompose", input_args={"n_queries": 25},
                 output_summary=f"{len(sub_queries)} sub-queries",
                 n_results=len(sub_queries))
    trace.append(tool="search", input_args={"n_sub_queries": len(sub_queries)},
                 output_summary=f"{len(documents)} documents retrieved",
                 n_results=len(documents))
    trace.append(tool="source_extract", input_args={"n_docs": len(documents)},
                 output_summary=f"{len(extractions)} relevant extractions",
                 n_results=len(extractions),
                 tokens_used=tracker.total_tokens - tokens_before_role1)
    state.save(
        "search_specialist",
        {
            "n_queries": len(sub_queries),
            "n_documents": len(documents),
            "n_extractions": len(extractions),
            "queries": [sq.model_dump() for sq in sub_queries],
            "extractions": [ext.to_evidence_dict() for ext in extractions],
        },
    )
    log.info(
        "p3_role1_complete",
        queries=len(sub_queries),
        docs=len(documents),
        extractions=len(extractions),
        cost=f"${tracker.total_cost:.4f}",
    )

    # -- Role 2: Topic Miner --------------------------------------------------
    if skip_topic_mining:
        log.info("p3_role2_topic_miner_skipped")
        from deep_research.types import TopicCluster
        from .topic_miner import build_extraction_map
        # Create a single catch-all topic cluster containing all extractions
        clusters = [
            TopicCluster(
                topic="General Research Findings",
                summary=f"All {len(extractions)} source extractions grouped into a single topic.",
                source_ids=[ext.doc_id for ext in extractions if ext.doc_id],
                importance=1.0,
            )
        ]
        extraction_map = build_extraction_map(extractions)
        state.save(
            "topic_miner",
            {
                "n_topics": len(clusters),
                "topics": [c.model_dump() for c in clusters],
                "n_extractions": len(extractions),
                "skipped": True,
            },
        )
        log.info("p3_role2_skipped_complete", extractions_used=len(extractions))
    else:
        log.info("p3_role2_topic_miner")
        tokens_before_cluster = tracker.total_tokens
        clusters, extraction_map = await run_topic_miner(
            query=query,
            extractions=extractions,
            llm=llm,
        )
        trace.append(tool="cluster", input_args={"n_extractions": len(extractions)},
                     output_summary=f"{len(clusters)} topic clusters",
                     n_results=len(clusters),
                     tokens_used=tracker.total_tokens - tokens_before_cluster)
        state.save(
            "topic_miner",
            {
                "n_topics": len(clusters),
                "topics": [c.model_dump() for c in clusters],
                "n_extractions": len(extractions),
            },
        )
        log.info(
            "p3_role2_complete",
            topics=len(clusters),
            extractions_used=len(extractions),
            cost=f"${tracker.total_cost:.4f}",
        )

    # -- Role 3: Writer --------------------------------------------------------
    log.info("p3_role3_writer")
    tokens_before_write = tracker.total_tokens
    title, abstract, sections, citations = await write_report(
        query=query,
        clusters=clusters,
        extraction_map=extraction_map,
        llm=llm,
    )
    trace.append(tool="generate", input_args={"role": "writer", "n_clusters": len(clusters)},
                 output_summary=f"{len(sections)} sections, {len(citations)} citations",
                 tokens_used=tracker.total_tokens - tokens_before_write)
    state.save(
        "writer_v0",
        {
            "title": title,
            "abstract": abstract,
            "n_sections": len(sections),
            "n_citations": len(citations),
        },
    )
    log.info(
        "p3_role3_complete",
        title=title[:60],
        sections=len(sections),
        citations=len(citations),
        cost=f"${tracker.total_cost:.4f}",
    )

    # -- Role 4: Evaluator (with optional revision loop) -----------------------
    evaluation_metadata: Dict[str, Any] = {}
    revision_round = 0

    if skip_evaluation:
        log.info("p3_role4_evaluator_skipped")
        # Use a lightweight namespace so downstream code can reference
        # evaluation.averaged_overall and evaluation.averaged_scores.
        from types import SimpleNamespace
        evaluation = SimpleNamespace(
            averaged_overall=0.0,
            averaged_scores={},
            needs_revision=False,
            feedback_text="",
            summary_dict=lambda: {"skipped": True, "averaged_overall": 0.0},
        )
        evaluation_metadata["evaluation_skipped"] = {"skipped": True}
    else:
        for revision_round in range(MAX_REVISIONS + 1):
            log.info("p3_role4_evaluator", round=revision_round)

            # Build report text for evaluation
            report = _assemble_report(
                query=query,
                title=title,
                abstract=abstract,
                sections=sections,
                citations=citations,
                tracker=tracker,
                elapsed=time.time() - t0,
            )
            report_text = report.full_text()

            # Evaluate
            tokens_before_eval = tracker.total_tokens
            evaluation = await evaluate_report(
                query=query,
                report_text=report_text,
                llm=llm,
            )
            trace.append(tool="reflect",
                         input_args={"role": "evaluator", "round": revision_round},
                         output_summary=f"avg_score={evaluation.averaged_overall:.2f}, needs_revision={evaluation.needs_revision}",
                         tokens_used=tracker.total_tokens - tokens_before_eval)

            state.save(
                f"evaluation_v{revision_round}",
                evaluation.summary_dict(),
            )
            evaluation_metadata[f"evaluation_v{revision_round}"] = (
                evaluation.summary_dict()
            )

            log.info(
                "p3_role4_eval_done",
                round=revision_round,
                avg_overall=f"{evaluation.averaged_overall:.1f}",
                needs_revision=evaluation.needs_revision,
                cost=f"${tracker.total_cost:.4f}",
            )

            # Decide: revise or accept
            if not evaluation.needs_revision:
                log.info("p3_report_accepted", round=revision_round)
                break

            if revision_round < MAX_REVISIONS:
                log.info("p3_role3_revising", round=revision_round + 1)
                tokens_before_revise = tracker.total_tokens
                title, abstract, sections, citations = await revise_report(
                    query=query,
                    current_title=title,
                    current_abstract=abstract,
                    current_sections=sections,
                    evaluation_feedback=evaluation.feedback_text,
                    clusters=clusters,
                    extraction_map=extraction_map,
                    llm=llm,
                )
                trace.append(tool="generate",
                             input_args={"role": "revise", "round": revision_round + 1},
                             output_summary=f"revised {len(sections)} sections",
                             tokens_used=tracker.total_tokens - tokens_before_revise)
                state.save(
                    f"writer_v{revision_round + 1}",
                    {
                        "title": title,
                        "abstract": abstract,
                        "n_sections": len(sections),
                        "n_citations": len(citations),
                    },
                )
            else:
                log.info(
                    "p3_max_revisions_reached",
                    final_score=f"{evaluation.averaged_overall:.1f}",
                )

    # -- Final assembly --------------------------------------------------------
    elapsed = time.time() - t0
    report = _assemble_report(
        query=query,
        title=title,
        abstract=abstract,
        sections=sections,
        citations=citations,
        tracker=tracker,
        elapsed=elapsed,
        extra_metadata={
            "evaluations": evaluation_metadata,
            "final_avg_score": evaluation.averaged_overall,
            "final_dimension_scores": evaluation.averaged_scores,
            "revision_rounds": revision_round,
            "n_topics": len(clusters),
            "n_source_documents": len(documents),
            "n_extractions": len(extractions),
            "topic_names": [c.topic for c in clusters],
        },
    )

    state.save(
        "final",
        {
            "cost": tracker.total_cost,
            "tokens": tracker.total_tokens,
            "elapsed": elapsed,
            "avg_score": evaluation.averaged_overall,
        },
    )

    log.info(
        "p3_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        elapsed=f"{elapsed:.1f}s",
        sections=len(report.sections),
        citations=len(report.citations),
        avg_score=f"{evaluation.averaged_overall:.1f}",
    )
    log.info("p3_cost_breakdown", summary=tracker.summary_text())

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [sq.query for sq in sub_queries]
    report.metadata["search_queries_sent"] = [sq.query for sq in sub_queries]
    report.metadata["n_documents_retrieved"] = len(documents)
    report.metadata["n_extractions"] = len(extractions)

    trace.final_report_word_count = len(report.full_text().split())
    trace.n_unique_urls_visited = len({d.url for d in documents if d.url})
    trace.n_iterations = revision_round
    state.save("trace", trace.model_dump(mode="json"))

    return report


# -- Helpers -------------------------------------------------------------------


def _assemble_report(
    query: str,
    title: str,
    abstract: str,
    sections: list,
    citations: list,
    tracker: CostTracker,
    elapsed: float,
    extra_metadata: dict | None = None,
) -> ResearchReport:
    """Assemble a ResearchReport from its components."""
    metadata: Dict[str, Any] = {"cost_summary": tracker.summary_text()}
    if extra_metadata:
        metadata.update(extra_metadata)

    return ResearchReport(
        query=query,
        title=title,
        abstract=abstract,
        sections=sections,
        citations=citations,
        metadata=metadata,
        pattern_name=PATTERN_NAME,
        total_cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )
