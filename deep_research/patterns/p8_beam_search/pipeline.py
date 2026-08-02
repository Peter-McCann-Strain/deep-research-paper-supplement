"""Pattern 8: Beam Search Exploration — main orchestration pipeline.

Architecture (DeepDiver / AI-Scientist inspired):
    Query -> Generate K=6 diverse research directions (hypotheses)
    -> Broad exploration: quick search + score each direction
    -> First beam selection: keep top B=3
    -> Deep investigation: full search + extraction + analysis on survivors
    -> Second beam selection (optional): keep top B'=2
    -> Synthesis: merge surviving directions into coherent report

Unlike patterns that commit to one research plan upfront, P8 explores
multiple competing directions in parallel, scores them on evidence
availability, prunes the weakest, and deepens the strongest — like
beam search over the space of possible research.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import structlog

from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtraction,
    SourceExtractor,
    StateManager,
    URLExtractor,
    get_web_searcher,
)
from deep_research.types import ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

from .hypothesis_generator import ResearchDirection, generate_hypotheses
from .explorer import explore_direction
from .beam_scorer import score_directions, select_beam, rescore_directions
from .deep_investigator import investigate_direction
from .beam_synthesizer import synthesize_beams

log = structlog.get_logger()


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    **kwargs,
) -> ResearchReport:
    """Execute the Beam Search Exploration pipeline.

    Stages:
        1. Hypothesis generation: LLM produces K diverse research directions
        2. Broad exploration: quick search + lightweight extraction per direction
        3. First beam selection: score and keep top B directions
        4. Deep investigation: full search + extraction + analysis on survivors
        5. Second beam selection (optional): re-score and keep top B' directions
        6. Synthesis: merge surviving directions into final report

    Args:
        query: The research question.
        budget_usd: Maximum budget for this run in USD.
        **kwargs: Additional parameters:
            n_hypotheses (int): Number of initial directions (default 6).
            beam_width (int): Beams kept after first selection (default 3).
            final_beam_width (int): Beams kept after second selection (default 2).
            skip_second_selection (bool): Skip the second beam round (default False).

    Returns:
        A structured ResearchReport with pattern_name="p8_beam_search".
    """
    t0 = time.time()

    # ── Configuration ────────────────────────────────────────────────────
    n_hypotheses = kwargs.get("n_hypotheses", 6)
    beam_width = kwargs.get("beam_width", 3)
    final_beam_width = kwargs.get("final_beam_width", 2)
    skip_second_selection = kwargs.get("skip_second_selection", False)

    # ── Shared infrastructure ────────────────────────────────────────────
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p8_beam_search")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    trace = ProcessTrace(pattern_name="p8_beam_search", query=query, query_id=kwargs.get("query_id", ""))

    log.info(
        "p8_start",
        query=query[:80],
        budget=budget_usd,
        n_hypotheses=n_hypotheses,
        beam_width=beam_width,
        final_beam_width=final_beam_width,
        skip_second_selection=skip_second_selection,
    )

    # == Stage 1: Hypothesis Generation ====================================
    log.info("stage_1_hypothesis_generation")
    tokens_before_hypo = tracker.total_tokens
    directions = await generate_hypotheses(llm, query, n_hypotheses)
    trace.append(tool="decompose",
                 input_args={"stage": "hypothesis_generation", "n_hypotheses": n_hypotheses},
                 output_summary=f"{len(directions)} research directions",
                 n_results=len(directions),
                 tokens_used=tracker.total_tokens - tokens_before_hypo)
    state.save("hypotheses", {
        "directions": [d.to_dict() for d in directions],
    })

    # == Stage 2: Broad Exploration (parallel) =============================
    log.info("stage_2_broad_exploration", n_directions=len(directions))
    tokens_before_explore = tracker.total_tokens
    explore_tasks = [
        explore_direction(d, llm, web, source_extractor, query)
        for d in directions
    ]
    explore_results = await asyncio.gather(*explore_tasks, return_exceptions=True)
    trace.append(tool="search",
                 input_args={"stage": "broad_exploration", "n_directions": len(directions)},
                 output_summary=f"{sum(d.n_docs_found for d in directions)} docs across {len(directions)} directions",
                 n_results=sum(len(d.extractions) for d in directions),
                 tokens_used=tracker.total_tokens - tokens_before_explore)

    # Log any exploration failures (directions remain with empty evidence)
    for i, result in enumerate(explore_results):
        if isinstance(result, Exception):
            log.warning(
                "exploration_task_failed",
                direction_id=directions[i].id,
                error=str(result),
            )

    state.save("exploration", {
        "directions_explored": len(directions),
        "per_direction": [
            {
                "id": d.id,
                "thesis": d.thesis[:80],
                "docs_found": d.n_docs_found,
                "extractions": len(d.extractions),
            }
            for d in directions
        ],
    })

    # == Stage 3: First Beam Selection =====================================
    log.info("stage_3_first_beam_selection")
    tokens_before_score = tracker.total_tokens
    ranked = await score_directions(llm, directions, query)
    surviving = select_beam(ranked, beam_width)
    trace.append(tool="beam_select",
                 input_args={"stage": "first_selection", "beam_width": beam_width,
                             "n_candidates": len(directions)},
                 output_summary=f"{len(surviving)} surviving / {len(directions)} candidates",
                 n_results=len(surviving),
                 tokens_used=tracker.total_tokens - tokens_before_score)

    state.save("beam_1", {
        "surviving": [
            {"id": d.id, "thesis": d.thesis, "score": d.promise_score}
            for d in surviving
        ],
        "pruned": [
            {"id": d.id, "thesis": d.thesis, "score": d.promise_score}
            for d in directions if not d.is_alive
        ],
    })

    # == Stage 4: Deep Investigation (parallel on surviving beams) =========
    log.info("stage_4_deep_investigation", n_surviving=len(surviving))
    tokens_before_invest = tracker.total_tokens
    investigate_tasks = [
        investigate_direction(
            d, llm, web, academic, url_extractor, source_extractor, query
        )
        for d in surviving
    ]
    investigate_results = await asyncio.gather(
        *investigate_tasks, return_exceptions=True
    )
    trace.append(tool="source_extract",
                 input_args={"stage": "deep_investigation", "n_surviving": len(surviving)},
                 output_summary=f"deep-investigated {len(surviving)} directions; "
                                 f"{sum(len(d.extractions) for d in surviving)} total extractions",
                 n_results=sum(len(d.extractions) for d in surviving),
                 tokens_used=tracker.total_tokens - tokens_before_invest)

    for i, result in enumerate(investigate_results):
        if isinstance(result, Exception):
            log.warning(
                "investigation_task_failed",
                direction_id=surviving[i].id,
                error=str(result),
            )

    state.save("deep_investigation", {
        "per_direction": [
            {
                "id": d.id,
                "thesis": d.thesis[:80],
                "total_extractions": len(d.extractions),
                "analysis_chars": len(d.detailed_analysis),
            }
            for d in surviving
        ],
    })

    # == Stage 5: Second Beam Selection (optional) =========================
    if (
        not skip_second_selection
        and len(surviving) > final_beam_width
        and final_beam_width > 0
    ):
        log.info(
            "stage_5_second_beam_selection",
            n_surviving=len(surviving),
            target=final_beam_width,
        )
        tokens_before_rescore = tracker.total_tokens
        surviving = await rescore_directions(llm, surviving, query)
        # Prune to final_beam_width
        for i, d in enumerate(surviving):
            if i >= final_beam_width:
                d.is_alive = False
        surviving = [d for d in surviving if d.is_alive]
        trace.append(tool="beam_select",
                     input_args={"stage": "second_selection", "final_beam_width": final_beam_width},
                     output_summary=f"{len(surviving)} final beams",
                     n_results=len(surviving),
                     tokens_used=tracker.total_tokens - tokens_before_rescore)

        state.save("beam_2", {
            "final_surviving": [
                {
                    "id": d.id,
                    "thesis": d.thesis,
                    "promise_score": d.promise_score,
                    "quality_score": d.evidence_quality_score,
                }
                for d in surviving
            ],
        })
    else:
        log.info("stage_5_second_beam_selection_skipped")
        # Populate evidence_quality_score from promise_score for metadata
        for d in surviving:
            d.evidence_quality_score = d.promise_score

    state.save("final_beams", {
        "final_directions": [d.to_dict() for d in surviving],
    })

    # == Stage 6: Synthesis ================================================
    log.info("stage_6_synthesis", n_surviving=len(surviving))

    # Collect all extractions from surviving directions, deduplicated
    all_extractions: List[SourceExtraction] = []
    seen_ids: set = set()
    for d in surviving:
        for ext in d.extractions:
            ext_key = ext.doc_id or ext.url
            if ext_key and ext_key not in seen_ids:
                seen_ids.add(ext_key)
                all_extractions.append(ext)

    tokens_before_gen = tracker.total_tokens
    report_md = await synthesize_beams(
        llm, surviving, all_extractions, query,
        n_directions_total=len(directions),
    )
    trace.append(tool="generate",
                 input_args={"n_surviving": len(surviving), "n_extractions": len(all_extractions)},
                 output_summary=f"{len(report_md)}-char synthesized report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    state.save("report_markdown", {"markdown": report_md[:5000]})

    # == Assemble final report =============================================
    elapsed = time.time() - t0

    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=all_extractions,
        pattern_name="p8_beam_search",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )

    # Attach rich metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [
        q for d in directions for q in d.search_queries_used
    ]
    report.metadata["search_queries_sent"] = [
        q for d in directions for q in d.search_queries_used
    ]
    report.metadata["n_documents_retrieved"] = sum(
        d.n_docs_found for d in directions
    )
    report.metadata["n_extractions"] = len(all_extractions)
    report.metadata["n_hypotheses_generated"] = len(directions)
    report.metadata["n_surviving_beams"] = len(surviving)
    report.metadata["beam_scores"] = {
        d.id: d.promise_score for d in directions
    }
    report.metadata["beam_quality_scores"] = {
        d.id: d.evidence_quality_score for d in surviving
    }
    report.metadata["pruned_directions"] = [
        d.thesis for d in directions if not d.is_alive
    ]
    report.metadata["surviving_directions"] = [
        d.thesis for d in surviving
    ]

    log.info(
        "p8_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        sections=len(report.sections),
        citations=len(report.citations),
        elapsed=f"{elapsed:.1f}s",
        n_hypotheses=len(directions),
        n_surviving=len(surviving),
    )
    log.info("p8_cost_breakdown", summary=tracker.summary_text())

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed_seconds": elapsed,
        "sections": len(report.sections),
        "citations": len(report.citations),
        "n_hypotheses": len(directions),
        "n_surviving": len(surviving),
    })

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len({ext.url for ext in all_extractions if ext.url})
    state.save("trace", trace.model_dump(mode="json"))

    return report
