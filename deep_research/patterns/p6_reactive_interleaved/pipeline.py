"""Pattern 6: Reactive Interleaved Agent — WebThinker-inspired reasoning loop.

Unlike patterns P0-P5 which follow fixed pipeline stages, P6 uses a single
autonomous reasoning loop where the model decides at each iteration what to
do next: search for more information, deep-read specific URLs, draft or
revise a report section, reflect on gaps, or finalise the report.

Flow::

    Loop (max 15 iterations, or until FINALIZE or budget low):
        1. Build context from accumulated state
        2. LLM decides next action (structured JSON)
           - SEARCH   -> web + academic search, extract sources
           - DEEP_READ -> fetch and extract specific URLs
           - DRAFT     -> write or revise a report section
           - REFLECT   -> analyse gaps, contradictions, missing angles
           - FINALIZE  -> assemble final report
        3. Execute the chosen action
        4. Update state, checkpoint, loop
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

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

from .action_router import (
    DEEP_READ,
    DRAFT,
    FINALIZE,
    REFLECT,
    SEARCH,
    parse_action,
)
from .state_tracker import AgentState

log = structlog.get_logger()

DEFAULT_MAX_ITERATIONS = 15
BUDGET_FORCE_FINALIZE_PCT = 0.20  # Force finalize when budget falls below 20%


# ── Prompts ──────────────────────────────────────────────────────────────────

AGENT_PROMPT = """\
You are an autonomous research agent. Given the current research state, decide your next action.

Research Query: {query}

Current State:
- Iteration: {iteration}/{max_iterations}
- Searches completed: {n_searches} ({search_queries_summary})
- Sources extracted: {n_extractions}
- Sections drafted: {draft_section_titles}
- Budget remaining: {budget_pct}%
- Last reflection: {last_reflection}

Available Actions:
1. SEARCH - Search for more information (provide 2-5 search queries)
2. DEEP_READ - Fetch and extract specific URLs you have seen in search results
3. DRAFT - Write or revise a report section using current evidence
4. REFLECT - Analyze what's missing, contradictions, or gaps
5. FINALIZE - Complete the report from current sections

Strategy guidance:
- Start with SEARCH to gather initial evidence, then REFLECT to identify gaps.
- Use DEEP_READ when you know specific URLs that need full extraction.
- Use DRAFT once you have sufficient evidence for a section.
- Use REFLECT periodically to check coverage before drafting more.
- Use FINALIZE when all major sections are drafted and evidence is sufficient.
- If this is the last iteration, you MUST choose FINALIZE.

Respond with JSON:
{{"action": "SEARCH|DEEP_READ|DRAFT|REFLECT|FINALIZE", "params": {{...}}}}

For SEARCH: {{"action": "SEARCH", "params": {{"queries": ["query1", "query2"]}}}}
For DEEP_READ: {{"action": "DEEP_READ", "params": {{"urls": ["url1", "url2"]}}}}
For DRAFT: {{"action": "DRAFT", "params": {{"section_title": "...", "instruction": "Write about..."}}}}
For REFLECT: {{"action": "REFLECT", "params": {{}}}}
For FINALIZE: {{"action": "FINALIZE", "params": {{}}}}
"""

DRAFT_PROMPT = """\
Write a section for a research report.

Research Query: {query}
Section Title: {section_title}
Instruction: {instruction}

Available Evidence:
{evidence_text}

Previously drafted sections:
{existing_sections_summary}

Write the section content with inline citations [1], [2], etc. Be comprehensive and analytical. \
Include specific data points, comparisons, and nuanced discussion where the evidence supports it. \
Do NOT include the section heading itself — just the body text."""

REFLECT_PROMPT = """\
Analyze the current state of this research and identify gaps.

Research Query: {query}
Evidence collected from {n_sources} sources covering: {topics_summary}
Sections drafted: {section_titles}

Identify:
1. What key aspects of the query are NOT yet covered?
2. Any contradictions in the evidence?
3. What additional searches would fill gaps?
4. Are any drafted sections weak or missing citations?

Provide your analysis concisely."""

FINALIZE_SINGLE_SHOT_PROMPT = """\
You are a research analyst. Write a comprehensive, well-structured research report \
answering the following query. Use ONLY the provided source evidence. Cite sources \
using inline numbered references like [1], [2], etc.

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

FINALIZE_ASSEMBLY_PROMPT = """\
You are assembling a research report. Given the sections below, produce a final \
markdown report with a title (# Title) and abstract (## Abstract), then include \
each section with ## headings, and end with a ## References section.

Research query: {query}

Drafted sections:
{sections_text}

Available evidence for references:
{evidence_summary}

Requirements:
- Write a concise, informative title
- Write a 150-250 word abstract summarizing the key findings
- Include each drafted section under its ## heading (you may lightly edit for flow)
- End with a ## References section listing all cited sources with their URLs
- Ensure citation numbers [1], [2], etc. are consistent throughout

Write the complete report:"""


# ── Action executors ─────────────────────────────────────────────────────────


async def _execute_search(
    params: Dict[str, Any],
    state: AgentState,
    web_searcher: Any,
    academic: AcademicSearcher,
    url_extractor: URLExtractor,
    source_extractor: SourceExtractor,
    query: str,
) -> str:
    """Execute a SEARCH action: web + academic search, then extract sources."""
    queries = params.get("queries", [])
    if not queries:
        return "No search queries provided."
    # Cap queries per action
    queries = queries[:8]

    log.info("action_search", n_queries=len(queries), queries=queries)

    # Run web + academic searches
    web_docs = await web_searcher.search_batch(queries, max_results_per=10)

    # Academic search on the first query only (to save budget)
    academic_docs = await academic.search(queries[0], max_per_source=5)

    # Deduplicate
    seen_urls = state._seen_urls.copy()
    all_docs = []
    for doc in web_docs + academic_docs:
        if doc.url and doc.url not in seen_urls:
            seen_urls.add(doc.url)
            all_docs.append(doc)

    log.info("search_results", web=len(web_docs), academic=len(academic_docs),
             new_docs=len(all_docs))

    if not all_docs:
        state.record_searches(queries)
        return f"Searched {len(queries)} queries but found no new documents."

    # Extract page content for docs with thin content
    urls_to_extract = [
        doc.url for doc in all_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted_pages = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted_pages if e.content}
        for doc in all_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]

    # Two-step source extraction
    extractions = await source_extractor.extract_batch(all_docs, query)

    added = state.add_extractions(extractions)
    state.record_searches(queries)

    summary = (
        f"Searched {len(queries)} queries -> {len(all_docs)} new docs "
        f"-> {len(extractions)} extractions ({added} new after dedup)."
    )
    log.info("search_complete", summary=summary)
    return summary


async def _execute_deep_read(
    params: Dict[str, Any],
    state: AgentState,
    url_extractor: URLExtractor,
    source_extractor: SourceExtractor,
    query: str,
) -> str:
    """Execute a DEEP_READ action: fetch specific URLs and extract."""
    urls = params.get("urls", [])
    if not urls:
        return "No URLs provided for deep read."
    # Cap URLs per deep-read action
    urls = urls[:10]

    log.info("action_deep_read", n_urls=len(urls))

    extracted_pages = await url_extractor.extract_batch(urls)
    if not extracted_pages:
        return f"Failed to extract content from {len(urls)} URLs."

    extractions = await source_extractor.extract_batch(extracted_pages, query)
    added = state.add_extractions(extractions)

    summary = (
        f"Deep-read {len(urls)} URLs -> {len(extracted_pages)} pages fetched "
        f"-> {len(extractions)} extractions ({added} new)."
    )
    log.info("deep_read_complete", summary=summary)
    return summary


async def _execute_draft(
    params: Dict[str, Any],
    state: AgentState,
    llm: LLMCaller,
) -> str:
    """Execute a DRAFT action: write or revise a report section."""
    section_title = params.get("section_title", "Untitled Section")
    instruction = params.get("instruction", f"Write a comprehensive section about {section_title}")

    log.info("action_draft", section=section_title)

    evidence_text = format_extractions_as_evidence(state.extractions)

    # Summarise existing sections for context
    if state.draft_sections:
        existing_parts = []
        for title, content in state.draft_sections.items():
            preview = content[:5000] + "..." if len(content) > 5000 else content
            existing_parts.append(f"### {title}\n{preview}")
        existing_sections_summary = "\n\n".join(existing_parts)
    else:
        existing_sections_summary = "No sections drafted yet."

    prompt = DRAFT_PROMPT.format(
        query=state.query,
        section_title=section_title,
        instruction=instruction,
        evidence_text=evidence_text,
        existing_sections_summary=existing_sections_summary,
    )

    section_content = await llm.complete(
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=4096,
        temperature=0.3,
    )

    state.draft_sections[section_title] = section_content.strip()

    summary = f"Drafted section '{section_title}' ({len(section_content)} chars)."
    log.info("draft_complete", section=section_title, chars=len(section_content))
    return summary


async def _execute_reflect(
    state: AgentState,
    llm: LLMCaller,
) -> str:
    """Execute a REFLECT action: analyse gaps and weaknesses."""
    log.info("action_reflect")

    # Build topics summary from extraction titles
    topics = list({e.title for e in state.extractions if e.title})[:20]
    topics_summary = ", ".join(topics) if topics else "no sources yet"

    prompt = REFLECT_PROMPT.format(
        query=state.query,
        n_sources=len(state.extractions),
        topics_summary=topics_summary,
        section_titles=state.draft_section_titles(),
    )

    reflection = await llm.complete(
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=2048,
        temperature=0.2,
    )

    reflection_text = reflection.strip()
    state.reflections.append(reflection_text)

    log.info("reflect_complete", chars=len(reflection_text))
    return reflection_text


# ── Main entry point ─────────────────────────────────────────────────────────


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    skip_reflect: bool = False,
    force_sequential_draft: bool = False,
    **kwargs,
) -> ResearchReport:
    """Execute the reactive interleaved agent loop.

    Args:
        query: The research question.
        budget_usd: Maximum budget in USD.
        max_iterations: Maximum reasoning-loop iterations (default 15).
        skip_reflect: If True, the agent never chooses REFLECT (ablation).
        force_sequential_draft: If True, drafts must follow Introduction ->
            body -> Conclusion order (ablation).
        **kwargs: Absorbs unknown ablation parameters gracefully.
    """
    t0 = time.monotonic()

    # ── Initialise tools ─────────────────────────────────────────────────
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state_mgr = StateManager("p6_reactive_interleaved")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)

    state = AgentState(query=query)
    action_history: List[Tuple[str, int]] = []
    all_docs_seen: set = set()  # track all doc URLs seen across actions
    trace = ProcessTrace(pattern_name="p6_reactive_interleaved", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p6_start", query=query[:80], max_iterations=max_iterations,
             skip_reflect=skip_reflect, force_sequential_draft=force_sequential_draft)

    # ── Reasoning loop ───────────────────────────────────────────────────
    for iteration in range(max_iterations):
        state.iteration = iteration + 1

        # Budget check — force finalize if low
        budget_remaining_pct = _budget_remaining_pct(tracker, budget_usd)
        if budget_remaining_pct < BUDGET_FORCE_FINALIZE_PCT and iteration > 0:
            log.warning("budget_low_forcing_finalize",
                        budget_pct=f"{budget_remaining_pct:.0%}")
            action_history.append((FINALIZE, state.iteration))
            break

        # Force FINALIZE on last iteration
        is_last_iteration = (iteration == max_iterations - 1)

        # Build the agent prompt
        agent_prompt = AGENT_PROMPT.format(
            query=query,
            iteration=state.iteration,
            max_iterations=max_iterations,
            n_searches=state.total_search_queries,
            search_queries_summary=state.search_queries_summary(),
            n_extractions=len(state.extractions),
            draft_section_titles=state.draft_section_titles(),
            budget_pct=f"{budget_remaining_pct * 100:.0f}",
            last_reflection=state.last_reflection()[:3000],
        )

        # Ask the model what to do next
        raw_response = await llm.complete(
            agent_prompt,
            model=DEFAULT_MODEL,
            max_tokens=1024,
            temperature=0.2,
        )

        action_type, params = parse_action(raw_response)

        # Ablation overrides
        if skip_reflect and action_type == REFLECT:
            log.info("skip_reflect_override", original=REFLECT)
            action_type = SEARCH
            params = {"queries": [query]}

        if is_last_iteration and action_type != FINALIZE:
            log.info("last_iteration_forcing_finalize", original=action_type)
            action_type = FINALIZE
            params = {}

        # Enforce sequential drafting if enabled
        if force_sequential_draft and action_type == DRAFT:
            params = _enforce_sequential_draft(params, state)

        log.info("iteration_action", iteration=state.iteration, action=action_type)
        action_history.append((action_type, state.iteration))

        # ── Dispatch action ──────────────────────────────────────────────
        tokens_before_action = tracker.total_tokens
        if action_type == SEARCH:
            result = await _execute_search(
                params, state, web, academic, url_extractor,
                source_extractor, query,
            )
            all_docs_seen.update(state._seen_urls)
            trace.append(tool="search",
                         input_args={"queries": params.get("queries", []), "iteration": state.iteration},
                         output_summary=result[:300] if isinstance(result, str) else "",
                         n_results=len(state.extractions),
                         tokens_used=tracker.total_tokens - tokens_before_action)

        elif action_type == DEEP_READ:
            result = await _execute_deep_read(
                params, state, url_extractor, source_extractor, query,
            )
            all_docs_seen.update(state._seen_urls)
            trace.append(tool="extract",
                         input_args={"urls": params.get("urls", []), "iteration": state.iteration},
                         output_summary=result[:300] if isinstance(result, str) else "",
                         tokens_used=tracker.total_tokens - tokens_before_action)

        elif action_type == DRAFT:
            result = await _execute_draft(params, state, llm)
            trace.append(tool="generate",
                         input_args={"section_title": params.get("section_title", ""),
                                     "iteration": state.iteration},
                         output_summary=result[:300] if isinstance(result, str) else "",
                         tokens_used=tracker.total_tokens - tokens_before_action)

        elif action_type == REFLECT:
            result = await _execute_reflect(state, llm)
            trace.append(tool="reflect",
                         input_args={"iteration": state.iteration},
                         output_summary=result[:300] if isinstance(result, str) else "",
                         tokens_used=tracker.total_tokens - tokens_before_action)

        elif action_type == FINALIZE:
            break

        else:
            log.warning("unhandled_action", action=action_type)
            break

        log.info("action_result", iteration=state.iteration, action=action_type,
                 result=result[:200] if isinstance(result, str) else str(result)[:200])

        # Checkpoint
        state_mgr.save(f"iter_{state.iteration:02d}_{action_type.lower()}", {
            "action": action_type,
            "params": params,
            "result_summary": result[:500] if isinstance(result, str) else "",
            **state.to_checkpoint_dict(),
        })

    # ── Finalise: assemble report ────────────────────────────────────────
    log.info("p6_finalize", iterations_used=state.iteration,
             sections=len(state.draft_sections),
             extractions=len(state.extractions))

    tokens_before_final = tracker.total_tokens
    report_md = await _build_final_report(state, llm)
    trace.append(tool="generate",
                 input_args={"stage": "finalize", "n_sections": len(state.draft_sections)},
                 output_summary=f"{len(report_md)}-char final report",
                 tokens_used=tracker.total_tokens - tokens_before_final)

    state_mgr.save("final_report", {"markdown": report_md})

    # Parse into ResearchReport
    elapsed = time.monotonic() - t0
    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=state.extractions,
        pattern_name="p6_reactive_interleaved",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )

    # Populate metadata
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = state.searches_done
    report.metadata["search_queries_sent"] = state.searches_done
    report.metadata["n_documents_retrieved"] = len(all_docs_seen)
    report.metadata["n_extractions"] = len(state.extractions)
    report.metadata["iterations_used"] = state.iteration
    report.metadata["actions_taken"] = action_history

    log.info("p6_complete",
             cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens,
             sections=len(report.sections),
             iterations=state.iteration,
             elapsed=f"{elapsed:.1f}s")

    state_mgr.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed_seconds": elapsed,
        "iterations_used": state.iteration,
        "actions_taken": action_history,
    })

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len(all_docs_seen)
    trace.n_iterations = state.iteration
    state_mgr.save("trace", trace.model_dump(mode="json"))

    return report


# ── Helpers ──────────────────────────────────────────────────────────────────


def _budget_remaining_pct(tracker: CostTracker, budget_usd: float) -> float:
    """Return remaining budget as a fraction (0.0 to 1.0).

    For PTU deployments where cost is always 0, return 1.0 to avoid
    premature finalization.
    """
    if budget_usd <= 0:
        return 1.0
    spent = tracker.total_cost
    if spent <= 0:
        return 1.0
    remaining = max(0.0, budget_usd - spent) / budget_usd
    return remaining


async def _build_final_report(state: AgentState, llm: LLMCaller) -> str:
    """Assemble the final markdown report from state.

    If no sections have been drafted, falls back to a single-shot generation
    (like P0) using all accumulated extractions.
    """
    evidence_text = format_extractions_as_evidence(state.extractions)

    if not state.draft_sections:
        # Single-shot fallback
        log.info("finalize_single_shot", extractions=len(state.extractions))
        report_md = await llm.complete(
            FINALIZE_SINGLE_SHOT_PROMPT.format(
                query=state.query,
                evidence=evidence_text,
            ),
            model=DEFAULT_MODEL,
            max_tokens=8192,
            temperature=0.3,
        )
        return report_md

    # Assemble from drafted sections
    sections_parts = []
    for title, content in state.draft_sections.items():
        sections_parts.append(f"## {title}\n{content}")
    sections_text = "\n\n".join(sections_parts)

    # Build a compact evidence summary for references
    evidence_summary_parts = []
    for i, e in enumerate(state.extractions, 1):
        evidence_summary_parts.append(f"[{i}] {e.title} — {e.url}")
    evidence_summary = "\n".join(evidence_summary_parts)

    prompt = FINALIZE_ASSEMBLY_PROMPT.format(
        query=state.query,
        sections_text=sections_text,
        evidence_summary=evidence_summary,
    )

    report_md = await llm.complete(
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )
    return report_md


def _enforce_sequential_draft(
    params: Dict[str, Any],
    state: AgentState,
) -> Dict[str, Any]:
    """When force_sequential_draft is enabled, constrain draft order.

    Order: Introduction -> body sections -> Conclusion.
    """
    section_title = params.get("section_title", "")
    has_intro = any(
        "introduction" in t.lower() for t in state.draft_sections
    )
    has_conclusion = any(
        "conclusion" in t.lower() for t in state.draft_sections
    )

    if not has_intro and "introduction" not in section_title.lower():
        log.info("sequential_draft_override_intro")
        params["section_title"] = "Introduction"
        params["instruction"] = (
            "Write an introduction that frames the research query, provides "
            "background context, and outlines what the report will cover."
        )
    elif has_intro and "conclusion" in section_title.lower() and len(state.draft_sections) < 3:
        # Don't allow conclusion too early; force a body section
        log.info("sequential_draft_defer_conclusion")
        params["section_title"] = section_title.replace("Conclusion", "Analysis")
        params["instruction"] = params.get(
            "instruction", f"Write an analysis section about {state.query}"
        )

    return params
