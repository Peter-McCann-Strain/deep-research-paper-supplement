"""Pattern 5: Hierarchical Width-Depth Pipeline.

Architecture:
    Query -> Planner (gpt-5.2) -> [loop: Width phase (gpt-5.2 workers, parallel)
    -> Depth phase (gpt-5.2) -> Meta-eval (gpt-5.2) -> Budget rebalance (gpt-5.2)]
    -> Citation verify (gpt-5.2) -> Report (gpt-5.2)

Width schedule: W(t) = max(W_min, W_0 * alpha^t)
    Starts broad with many parallel workers, narrows to depth over iterations.

All data flows through source extractions (two-step LLM extraction) instead
of chunk/embed/vector infrastructure.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import structlog

from deep_research.tools import (
    CostTracker,
    LLMCaller,
    StateManager,
    SourceExtraction,
    format_extractions_as_evidence,
)
from deep_research.types import (
    ProcessTrace,
    ResearchReport,
    Citation,
    SubQuery,
)
from deep_research.utils.markdown_parser import parse_markdown_report

from .planner import create_research_plan, extract_sub_queries, extract_depth_questions
from .wd_schedule import WDSchedule
from .width_controller import WidthController
from .depth_controller import DepthController
from .meta_evaluator import evaluate_progress
from .budget_allocator import BudgetAllocator
from .citation_verifier import CitationVerifier
from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN

log = structlog.get_logger()

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_W0 = 4
DEFAULT_ALPHA = 0.5
DEFAULT_W_MIN = 1
DEFAULT_MAX_STEPS = 3

# ── Report generation ────────────────────────────────────────────────────────

REPORT_SYSTEM = """You are an expert research report writer. Produce a well-structured,
comprehensive research report. Use inline citations [1], [2], etc. Be specific and
include data points, comparisons, and nuanced analysis."""

REPORT_PROMPT = """Write a comprehensive research report based on the following
analyses and evidence.

Research Query: {query}
Research Title: {title}

Subtopic Analyses:
{analyses}

Cross-cutting Themes: {themes}
Key Controversies: {controversies}

Source Evidence:
{evidence}

Requirements:
- Start with a markdown title (# Title)
- Include an ## Abstract (150-250 words)
- Organize into clear ## sections covering each major subtopic
- Use inline citations [1], [2], etc. referencing the numbered sources above
- Include specific data points, statistics, and comparisons where available
- Discuss limitations and areas of disagreement
- End with a ## Conclusion
- Aim for 2000-4000 words

Write the report in markdown format."""


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    w_0: int = DEFAULT_W0,
    alpha: float = DEFAULT_ALPHA,
    w_min: int = DEFAULT_W_MIN,
    max_steps: int = DEFAULT_MAX_STEPS,
    skip_meta_eval: bool = False,
    skip_citation_verify: bool = False,
    **kwargs,
) -> ResearchReport:
    """Execute the full Hierarchical Width-Depth pipeline.

    Args:
        query: The research question to investigate.
        budget_usd: Maximum budget in USD.
        w_0: Initial width workers (default 4).
        alpha: Width decay factor (default 0.5).
        w_min: Minimum width workers (default 1).
        max_steps: Maximum iteration steps (default 3).
        skip_meta_eval: if True, skip meta-evaluation and budget rebalancing.
        skip_citation_verify: if True, skip citation verification step.
        **kwargs: Absorbs unknown ablation parameters gracefully.

    Returns:
        ResearchReport with pattern_name="p5_hierarchical_wd".
    """
    start_time = time.time()

    # ── Initialize shared resources ──────────────────────────────────────
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p5_hierarchical_wd")
    trace = ProcessTrace(pattern_name="p5_hierarchical_wd", query=query, query_id=kwargs.get("query_id", ""))

    schedule = WDSchedule(
        w_0=w_0,
        alpha=alpha,
        w_min=w_min,
        max_steps=max_steps,
        total_budget=budget_usd,
    )

    width_ctrl = WidthController(
        llm=llm,
        cost_tracker=tracker,
    )
    depth_ctrl = DepthController(
        llm=llm,
        cost_tracker=tracker,
    )
    budget_alloc = BudgetAllocator(
        schedule=schedule,
        cost_tracker=tracker,
        llm=llm,
    )

    log.info("p5_start", query=query[:80], budget=budget_usd)

    # ── Stage 1: Planning (gpt-5.2) ─────────────────────────────────────
    log.info("p5_stage_plan")
    tokens_before_plan = tracker.total_tokens
    plan = await create_research_plan(query, llm, model=DEFAULT_MODEL)
    state.save("plan", plan)

    all_sub_queries = extract_sub_queries(plan)
    depth_questions = extract_depth_questions(plan)
    trace.append(tool="decompose",
                 input_args={"stage": "plan"},
                 output_summary=f"{len(all_sub_queries)} sub-queries, {len(depth_questions)} depth groups",
                 n_results=len(all_sub_queries),
                 tokens_used=tracker.total_tokens - tokens_before_plan)

    log.info(
        "p5_plan_ready",
        title=plan.get("title", "")[:60],
        sub_queries=len(all_sub_queries),
        depth_groups=len(depth_questions),
    )

    # ── Stage 2-5: Iterative Width-Depth Loop ────────────────────────────
    all_subtopic_analyses: List[Dict[str, Any]] = []
    all_summaries: List[SourceExtraction] = []
    total_docs = 0
    step = 0
    current_allocation = await budget_alloc.initial_allocation(step=0)
    meta_eval_result: Dict[str, Any] = {
        "should_continue": True,
        "overall_score": 0.0,
        "rationale": "initial iteration",
        "additional_queries": [],
        "uncovered_subtopics": [],
        "covered_subtopics": [],
        "improvement_suggestions": [],
    }

    for step in range(max_steps):
        log.info("p5_iteration_start", step=step)

        try:
            tracker.check_budget()
        except Exception:
            log.warning("p5_budget_exceeded_before_step", step=step)
            break

        # ── Width Phase: parallel broad search ───────────────────────
        log.info("p5_width_phase", step=step, workers=current_allocation.width_workers)

        # On later iterations, use gap-fill queries from meta-eval
        if step == 0:
            step_queries = all_sub_queries
        else:
            # Use additional queries from meta-eval + remaining sub-queries
            step_queries = _get_step_queries(
                all_sub_queries, meta_eval_result, step
            )

        tokens_before_width = tracker.total_tokens
        width_result = await width_ctrl.run_width_phase(
            sub_queries=step_queries,
            allocation=current_allocation,
            research_query=query,
            include_academic=(step <= 1),  # academic on steps 0 and 1
        )
        total_docs += width_result.get("total_docs", 0)

        # Accumulate source summaries across iterations
        all_summaries.extend(width_result.get("summaries", []))
        trace.append(tool="widen",
                     input_args={"step": step, "workers": current_allocation.width_workers,
                                 "n_queries": len(step_queries)},
                     output_summary=f"{width_result.get('total_docs', 0)} docs, {width_result.get('total_summaries', 0)} summaries",
                     n_results=width_result.get("total_summaries", 0),
                     tokens_used=tracker.total_tokens - tokens_before_width)

        state.save(f"width_step{step}", {
            "total_docs": width_result["total_docs"],
            "total_summaries": width_result["total_summaries"],
            "worker_results": width_result["worker_results"],
        })

        # ── Depth Phase: focused analysis ────────────────────────────
        log.info("p5_depth_phase", step=step, iterations=current_allocation.depth_iterations)

        tokens_before_depth = tracker.total_tokens
        depth_result = await depth_ctrl.run_depth_phase(
            depth_questions=depth_questions,
            allocation=current_allocation,
            query=query,
            summaries=all_summaries,
        )
        all_subtopic_analyses.extend(depth_result.get("subtopic_analyses", []))
        trace.append(tool="deepen",
                     input_args={"step": step, "iterations": current_allocation.depth_iterations,
                                 "n_depth_groups": len(depth_questions)},
                     output_summary=f"{len(depth_result.get('subtopic_analyses', []))} subtopic analyses",
                     n_results=len(depth_result.get("subtopic_analyses", [])),
                     tokens_used=tracker.total_tokens - tokens_before_depth)

        state.save(f"depth_step{step}", {
            "subtopic_analyses": [
                {k: v for k, v in a.items()}
                for a in depth_result.get("subtopic_analyses", [])
            ],
            "avg_confidence": depth_result.get("avg_confidence", 0),
        })

        # ── Meta-Evaluation ──────────────────────────────────────────
        if skip_meta_eval:
            log.info("p5_meta_eval_skipped", step=step)
            # Without meta-eval, we have no signal to stop or rebalance.
            # Use a default meta_eval_result so _get_step_queries still works.
            meta_eval_result = {
                "should_continue": step < max_steps - 1,
                "overall_score": 0.0,
                "rationale": "meta-eval skipped",
                "additional_queries": [],
                "uncovered_subtopics": [],
                "covered_subtopics": [],
                "improvement_suggestions": [],
            }
            state.save(f"meta_eval_step{step}", meta_eval_result)
            if step >= max_steps - 1:
                break
        else:
            log.info("p5_meta_eval", step=step)

            tokens_before_meta = tracker.total_tokens
            meta_eval_result = await evaluate_progress(
                query=query,
                plan=plan,
                subtopic_analyses=all_subtopic_analyses,
                step=step,
                max_steps=max_steps,
                total_docs=total_docs,
                total_summaries=len(all_summaries),
                budget_spent=tracker.total_cost,
                budget_total=budget_usd,
                avg_confidence=depth_result.get("avg_confidence", 0.5),
                llm=llm,
            )
            trace.append(tool="reflect",
                         input_args={"stage": "meta_eval", "step": step},
                         output_summary=f"score={meta_eval_result.get('overall_score', 0):.2f}, continue={meta_eval_result.get('should_continue', False)}",
                         tokens_used=tracker.total_tokens - tokens_before_meta)

            state.save(f"meta_eval_step{step}", meta_eval_result)

            if not meta_eval_result.get("should_continue", False):
                log.info(
                    "p5_stopping",
                    step=step,
                    overall=meta_eval_result.get("overall_score", 0),
                    reason=meta_eval_result.get("rationale", "")[:100],
                )
                break

            # ── Budget Rebalance ─────────────────────────────────────────
            log.info("p5_budget_rebalance", step=step)

            current_allocation = await budget_alloc.rebalance(
                step=step + 1,
                meta_eval=meta_eval_result,
                current_allocation=current_allocation,
            )

            # Update depth questions with uncovered subtopics
            depth_questions = _update_depth_questions(
                depth_questions, meta_eval_result, plan
            )

    # ── Stage 6: Report Generation (gpt-5.2) ────────────────────────────
    log.info("p5_report_generation")

    tokens_before_gen = tracker.total_tokens
    report_md, all_sources = await _generate_report(
        query=query,
        plan=plan,
        subtopic_analyses=all_subtopic_analyses,
        summaries=all_summaries,
        llm=llm,
    )
    trace.append(tool="generate",
                 input_args={"n_analyses": len(all_subtopic_analyses), "n_summaries": len(all_summaries)},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    state.save("report_draft", {"markdown_length": len(report_md)})

    # ── Stage 7: Citation Verification (gpt-5.2) ────────────────────────
    citations = _build_citations(all_sources)

    if skip_citation_verify:
        log.info("p5_citation_verify_skipped")
        verify_result = {
            "skipped": True,
            "accuracy_rate": 0.0,
            "checked": 0,
            "flagged_claims": [],
        }
    else:
        log.info("p5_citation_verify")
        verifier = CitationVerifier(
            llm=llm,
            cost_tracker=tracker,
        )
        verify_result = await verifier.verify_report(
            report_text=report_md,
            citations=citations,
            extractions=all_summaries,
            max_checks=5,
            model=DEFAULT_MODEL,
        )

    state.save("citation_verify", {
        "accuracy_rate": verify_result.get("accuracy_rate", 0),
        "checked": verify_result.get("checked", 0),
        "flagged": len(verify_result.get("flagged_claims", [])),
    })

    # ── Assemble Final Report ────────────────────────────────────────────
    elapsed = time.time() - start_time

    report = _assemble_report(
        query=query,
        plan=plan,
        report_md=report_md,
        citations=citations,
        verify_result=verify_result,
        tracker=tracker,
        elapsed=elapsed,
        schedule=schedule,
        total_summaries=len(all_summaries),
    )

    log.info(
        "p5_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        sections=len(report.sections),
        elapsed=f"{elapsed:.1f}s",
        citation_accuracy=f"{verify_result.get('accuracy_rate', 0):.2f}",
        total_summaries=len(all_summaries),
    )

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed": elapsed,
        "sections": len(report.sections),
        "citations": len(report.citations),
        "citation_accuracy": verify_result.get("accuracy_rate", 0),
        "total_summaries": len(all_summaries),
    })

    # Persist cost breakdown and search metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [sq.query for sq in all_sub_queries]
    report.metadata["search_queries_sent"] = [sq.query for sq in all_sub_queries]
    report.metadata["n_documents_retrieved"] = total_docs
    report.metadata["n_extractions"] = len(all_summaries)

    trace.final_report_word_count = len(report_md.split())
    trace.n_iterations = step + 1
    state.save("trace", trace.model_dump(mode="json"))

    return report


# ── Fuzzy matching ────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "this", "that", "these", "those", "what", "which", "who", "how",
})


def _tokenize(text: str) -> set[str]:
    """Tokenize text into meaningful words (lowered, stop words removed)."""
    words = set(text.lower().split())
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _fuzzy_match(candidate: str, targets: set[str], threshold: float = 0.35) -> bool:
    """Check if candidate fuzzy-matches any target using token overlap (Jaccard).

    Uses lowercased word sets with stop words removed.
    """
    candidate_tokens = _tokenize(candidate)
    if not candidate_tokens:
        return False

    for target in targets:
        target_tokens = _tokenize(target)
        if not target_tokens:
            continue
        intersection = candidate_tokens & target_tokens
        union = candidate_tokens | target_tokens
        jaccard = len(intersection) / len(union) if union else 0
        if jaccard >= threshold:
            return True
    return False


# ── Helper Functions ──────────────────────────────────────────────────────────


def _get_step_queries(
    all_sub_queries: List[SubQuery],
    meta_eval: Dict[str, Any],
    step: int,
) -> List[SubQuery]:
    """Get queries for a non-initial step, incorporating gap-fill suggestions."""
    # Start with additional queries from meta-eval
    additional = meta_eval.get("additional_queries", [])
    queries: List[SubQuery] = [
        SubQuery(query=q, intent="gap_fill", priority=1)
        for q in additional
    ]

    # Add uncovered subtopic queries from original plan (fuzzy match)
    uncovered = set(meta_eval.get("uncovered_subtopics", []))
    for sq in all_sub_queries:
        if _fuzzy_match(sq.intent, uncovered):
            queries.append(sq)

    # If still few queries, add remaining original queries by priority
    if len(queries) < 3:
        remaining = [
            sq for sq in all_sub_queries
            if sq not in queries
        ]
        remaining.sort(key=lambda sq: sq.priority)
        queries.extend(remaining[:5])

    return queries


def _update_depth_questions(
    current_questions: List[Dict[str, Any]],
    meta_eval: Dict[str, Any],
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Update depth questions based on meta-eval, prioritizing uncovered topics."""
    uncovered = set(meta_eval.get("uncovered_subtopics", []))
    covered = set(meta_eval.get("covered_subtopics", []))

    updated = []
    for q in current_questions:
        subtopic = q.get("subtopic", "")
        if _fuzzy_match(subtopic, uncovered):
            # Boost priority of uncovered subtopics
            q = dict(q)
            q["priority"] = 1
            updated.append(q)
        elif not _fuzzy_match(subtopic, covered):
            updated.append(q)

    # Add improvement suggestions as new depth questions
    suggestions = meta_eval.get("improvement_suggestions", [])
    if suggestions:
        updated.append({
            "subtopic": "Improvements",
            "description": "Addressing evaluator suggestions",
            "priority": 1,
            "questions": suggestions[:3],
        })

    updated.sort(key=lambda x: x.get("priority", 2))
    return updated


async def _generate_report(
    query: str,
    plan: Dict[str, Any],
    subtopic_analyses: List[Dict[str, Any]],
    summaries: List[SourceExtraction],
    llm: LLMCaller,
) -> tuple[str, List[Dict[str, Any]]]:
    """Generate the final report from all analyses and source extractions.

    Returns:
        Tuple of (report_markdown, sources_list).
    """
    # Format analyses
    analyses_parts = []
    for a in subtopic_analyses:
        subtopic = a.get("subtopic", "Unknown")
        synthesis = a.get("synthesis", a.get("summary", "No summary"))
        findings = a.get("key_findings", [])
        data_points = a.get("data_points", [])
        contradictions = a.get("contradictions", [])

        analyses_parts.append(
            f"### {subtopic}\n"
            f"Synthesis: {synthesis}\n"
            f"Key Findings: {'; '.join(findings[:8])}\n"
            f"Data Points: {'; '.join(data_points[:5])}\n"
            f"Contradictions: {'; '.join(contradictions[:3]) if contradictions else 'None'}"
        )

    analyses_text = "\n\n".join(analyses_parts) if analyses_parts else "No analyses available."

    # Build numbered source list from extractions
    all_sources: List[Dict[str, Any]] = []
    for i, s in enumerate(summaries, 1):
        all_sources.append({
            "index": i,
            "title": s.title or "Unknown",
            "url": s.url or "",
            "doc_id": s.doc_id or "",
        })

    # Format evidence from source extractions
    evidence_text = format_extractions_as_evidence(summaries) if summaries else "No source evidence."

    themes = plan.get("cross_cutting_themes", [])
    controversies = plan.get("key_controversies", [])

    report_md = await llm.complete(
        REPORT_PROMPT.format(
            query=query,
            title=plan.get("title", query),
            analyses=analyses_text,
            themes=", ".join(themes) if themes else "None identified",
            controversies=", ".join(controversies) if controversies else "None identified",
            evidence=evidence_text,
        ),
        model=DEFAULT_MODEL,
        system=REPORT_SYSTEM,
        temperature=0.3,
        max_tokens=8192,
    )

    return report_md, all_sources


def _build_citations(sources: List[Dict[str, Any]]) -> List[Citation]:
    """Build Citation objects from the sources list."""
    citations = []
    for s in sources:
        citations.append(Citation(
            claim=f"[{s['index']}]",
            source_id=s.get("doc_id", str(s.get("index", ""))),
            source_title=s.get("title", ""),
            source_url=s.get("url", ""),
            relevance_score=0.0,
        ))
    return citations


def _assemble_report(
    query: str,
    plan: Dict[str, Any],
    report_md: str,
    citations: List[Citation],
    verify_result: Dict[str, Any],
    tracker: CostTracker,
    elapsed: float,
    schedule: WDSchedule,
    total_summaries: int = 0,
) -> ResearchReport:
    """Parse markdown and assemble the final ResearchReport."""
    # Build P5-specific metadata
    metadata = {
        "pattern_params": {
            "w_0": schedule.w_0,
            "alpha": schedule.alpha,
            "w_min": schedule.w_min,
            "max_steps": schedule.max_steps,
        },
        "wd_history": [a.model_dump() for a in schedule.history],
        "citation_verification": {
            "accuracy_rate": verify_result.get("accuracy_rate", 0),
            "checked": verify_result.get("checked", 0),
            "flagged": len(verify_result.get("flagged_claims", [])),
        },
        "total_summaries": total_summaries,
        "cost_summary": tracker.summary_text(),
    }

    # Use the shared parser for title/abstract/sections extraction.
    # P5 builds citations separately via _build_citations(), so we pass
    # empty extractions and override citations on the result.
    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=[],
        pattern_name="p5_hierarchical_wd",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
        metadata=metadata,
    )

    # Override with pre-built citations (P5 builds these from a different
    # source list via _build_citations, not from SourceExtractions directly).
    report.citations = citations

    return report
