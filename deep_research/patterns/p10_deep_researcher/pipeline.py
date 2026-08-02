"""Pattern 10: DeepResearcher-7b — RL-trained agentic search agent.

Uses GAIR/DeepResearcher-7b, a Qwen2.5-7B-Instruct fine-tuned with GRPO
reinforcement learning in real web search environments. The model learned
emergent search behaviors: planning, cross-validation, self-reflection,
and honest uncertainty acknowledgment.

Purpose: Comparing P10 vs P9 (same 7B base, no RL) isolates the effect of
RL training on search behavior and information gathering quality.

The model's native task is factoid QA with tool calls. We adapt it for
research report generation by:
  1. Using its RL-trained search loop to gather information autonomously
  2. Collecting all search results and extracted information
  3. Using the same model to write a research report from gathered evidence

Flow:
    Query → RL agent decides search actions autonomously
    → Agentic search loop (model decides when/what to search)
    → Collect all retrieved sources
    → Two-step source extraction on gathered documents
    → Report generation from extracted evidence
    → Parse into ResearchReport
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    SourceExtraction,
    SourceExtractor,
    URLExtractor,
    get_web_searcher,
    format_extractions_as_evidence,
    StateManager,
)
from deep_research.tools.local_llm_caller import LocalLLMCaller
from deep_research.types import Document, ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

log = structlog.get_logger()

# DeepResearcher-7b model — RL-trained on Qwen2.5-7B-Instruct
DEEP_RESEARCHER_MODEL = "GAIR/DeepResearcher-7b"

# ── System prompt matching DeepResearcher's training format ──────────────────

AGENT_SYSTEM_PROMPT = """You are a research agent. Your task is to thoroughly research a topic by searching the web, analyzing results, and collecting evidence before writing a final answer.

You MUST use web_search before answering. Do NOT answer without searching first.

You should perform at least 3-5 searches on different aspects of the topic to gather comprehensive evidence. Think about what subtopics and angles need investigation.

Think through your research plan in <think> tags. After gathering sufficient evidence from multiple searches, provide your comprehensive findings in <answer> tags.

IMPORTANT: Always search first. Never provide an answer without at least 2-3 web searches."""

# Qwen2.5 native tool format — triggers proper tool-call behavior from RL-trained model
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic. Use this to find relevant articles, papers, and data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

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


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output."""
    calls = []
    # Match <tool_call>...</tool_call> pattern
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        try:
            call = json.loads(match.group(1))
            calls.append(call)
        except json.JSONDecodeError:
            continue

    # Also try JSON code blocks (alternative format the model might use)
    if not calls:
        pattern = r'```json\s*\n?(.*?)\n?```'
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match.group(1))
                if "name" in call or "query" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue

    # Try to find inline search requests
    if not calls:
        search_patterns = [
            r'web_search\("([^"]+)"\)',
            r'search\("([^"]+)"\)',
            r'Search for:?\s*["\']([^"\']+)["\']',
        ]
        for sp in search_patterns:
            for match in re.finditer(sp, text):
                calls.append({"name": "web_search", "arguments": {"query": match.group(1)}})

    return calls


def _extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_thinking(text: str) -> str:
    """Extract thinking from <think> tags."""
    matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    return "\n".join(m.strip() for m in matches)


async def _agentic_search_loop(
    llm: LocalLLMCaller,
    web_searcher,
    query: str,
    max_iterations: int = 8,
    min_searches: int = 2,
) -> tuple[list[Document], list[str], str]:
    """Run the RL agent's autonomous search loop.

    The model decides when to search, what to search for, and when to stop.
    Requires at least min_searches before accepting an answer.

    Returns:
        (gathered_documents, search_queries_used, agent_findings)
    """
    all_docs: list[Document] = []
    search_queries: list[str] = []
    seen_urls: set[str] = set()

    # Build conversation
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research this topic thoroughly: {query}"},
    ]

    for iteration in range(max_iterations):
        log.info("p10_agent_iteration", iteration=iteration)

        # Generate model response (pass tools for Qwen2.5 native tool format)
        response = await llm.complete_messages(
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            tools=AGENT_TOOLS,
        )

        # Check for answer (agent decided to stop)
        answer = _extract_answer(response)
        if answer:
            if len(search_queries) >= min_searches:
                log.info("p10_agent_answered", iteration=iteration,
                         total_searches=len(search_queries))
                return all_docs, search_queries, answer
            else:
                # Not enough searches yet — redirect model to search first
                log.info("p10_answer_too_early", iteration=iteration,
                         searches=len(search_queries), min_required=min_searches)
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "You haven't searched enough yet. Please use web_search "
                        "to find specific evidence before answering. Search for "
                        "different aspects of the topic."
                    ),
                })
                continue

        # Extract tool calls
        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            # Model didn't make a tool call or answer — might be done
            # Check if the response itself is a summary
            if iteration > 0 and len(response) > 500:
                log.info("p10_agent_implicit_answer", iteration=iteration)
                return all_docs, search_queries, response
            # Prompt it to continue
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Continue your research. Use web_search to find more information, or provide your findings in <answer> tags if you have enough."})
            continue

        # Execute search tool calls
        messages.append({"role": "assistant", "content": response})

        for call in tool_calls:
            search_query = ""
            if isinstance(call.get("arguments"), dict):
                search_query = call["arguments"].get("query", "")
            elif isinstance(call.get("query"), str):
                search_query = call["query"]

            if not search_query:
                continue

            search_queries.append(search_query)
            log.info("p10_agent_search", query=search_query[:80], iteration=iteration)

            # Execute search
            try:
                docs = await web_searcher.search_batch([search_query], max_results_per=10)
                new_docs = []
                for doc in docs:
                    if doc.url and doc.url not in seen_urls:
                        seen_urls.add(doc.url)
                        all_docs.append(doc)
                        new_docs.append(doc)

                # Format results for the model
                results_text = "\n".join(
                    f"- [{doc.title}]({doc.url}): {doc.content[:200]}"
                    for doc in new_docs[:5]
                ) or "No results found."

                messages.append({
                    "role": "user",
                    "content": f"Search results for '{search_query}':\n{results_text}\n\nAnalyze these results. Search for more information or provide your findings in <answer> tags.",
                })
            except Exception as e:
                log.warning("p10_search_error", error=str(e)[:100])
                messages.append({
                    "role": "user",
                    "content": f"Search failed: {str(e)[:100]}. Try a different query or provide your findings.",
                })

    # Max iterations reached — ask for final answer
    log.info("p10_max_iterations", total_searches=len(search_queries))
    messages.append({
        "role": "user",
        "content": "You've done enough research. Please provide your comprehensive findings in <answer> tags now.",
    })
    response = await llm.complete_messages(messages=messages, max_tokens=2048, temperature=0.3, tools=AGENT_TOOLS)
    answer = _extract_answer(response) or response

    return all_docs, search_queries, answer


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    **kwargs,
) -> ResearchReport:
    """Execute DeepResearcher-7b: RL-trained agentic search + report generation."""
    start = time.time()

    model_id = kwargs.get("model_id", DEEP_RESEARCHER_MODEL)
    quantize_4bit = kwargs.get("quantize_4bit", True)
    max_search_iterations = kwargs.get("max_search_iterations", 8)
    evidence_word_limit = kwargs.get("evidence_word_limit", 6000)

    tracker = CostTracker(budget_usd=budget_usd)
    llm = LocalLLMCaller(
        model_id=model_id,
        cost_tracker=tracker,
        quantize_4bit=quantize_4bit,
    )
    state = StateManager("p10_deep_researcher")
    web = get_web_searcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(
        llm=llm, model=model_id,
        max_content_per_call=15_000,  # 15K chars ≈ 4K tokens, fits 16GB VRAM with 4-bit 7B
    )
    trace = ProcessTrace(pattern_name="p10_deep_researcher", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p10_start", query=query[:80], model=model_id)

    # ── Stage 1: Agentic search loop (RL-trained) ────────────────────────
    log.info("p10_stage_1_agentic_search")
    gathered_docs, search_queries, agent_findings = await _agentic_search_loop(
        llm=llm,
        web_searcher=web,
        query=query,
        max_iterations=max_search_iterations,
    )
    # Log each agent search query as a search step (the RL agent decides each)
    for i, sq in enumerate(search_queries):
        trace.append(tool="search",
                     input_args={"query": sq, "agent_iteration": i},
                     output_summary=f"agentic search {i+1}",
                     n_results=0)
    trace.append(tool="tool_call",
                 input_args={"stage": "agentic_loop", "max_iterations": max_search_iterations},
                 output_summary=f"{len(gathered_docs)} docs gathered, {len(search_queries)} searches, "
                                 f"findings={len(agent_findings)} chars",
                 n_results=len(gathered_docs))

    state.save("agentic_search", {
        "docs_gathered": len(gathered_docs),
        "search_queries": search_queries,
        "findings_length": len(agent_findings),
    })

    log.info("p10_search_complete",
             docs=len(gathered_docs),
             queries=len(search_queries))

    # ── Stage 2: Enrich documents (fetch full page content) ──────────────
    urls_to_extract = [
        doc.url for doc in gathered_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract[:15])
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in gathered_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]
        log.info("p10_page_extract_done", pages=len(url_to_content))
        trace.append(tool="extract",
                     input_args={"n_urls": min(15, len(urls_to_extract))},
                     output_summary=f"{len(url_to_content)} pages extracted",
                     n_results=len(url_to_content))

    # ── Stage 3: Two-step source extraction ──────────────────────────────
    log.info("p10_stage_3_extraction")
    extractions = []
    if gathered_docs:
        extractions = await source_extractor.extract_batch(gathered_docs, query)
    trace.append(tool="source_extract",
                 input_args={"n_docs": len(gathered_docs)},
                 output_summary=f"{len(extractions)} relevant extractions",
                 n_results=len(extractions))

    state.save("extraction", {
        "extraction_count": len(extractions),
    })

    if not extractions and not agent_findings:
        log.warning("p10_no_sources")
        return ResearchReport(
            query=query,
            title=query,
            pattern_name="p10_deep_researcher",
            total_cost_usd=tracker.total_cost,
            total_tokens=tracker.total_tokens,
            elapsed_seconds=time.time() - start,
        )

    # ── Stage 4: Report generation ───────────────────────────────────────
    log.info("p10_stage_4_report", extractions=len(extractions))

    # Combine extraction evidence with agent's own findings
    evidence_text = ""
    if extractions:
        evidence_text = format_extractions_as_evidence(extractions)
    if agent_findings:
        evidence_text += f"\n\n--- Agent Research Findings ---\n{agent_findings}"

    # Truncate for 7B context
    words = evidence_text.split()
    if len(words) > evidence_word_limit:
        evidence_text = " ".join(words[:evidence_word_limit]) + "\n\n[... evidence truncated ...]"

    tokens_before_gen = tracker.total_tokens
    report_md = await llm.complete(
        REPORT_PROMPT.format(query=query, evidence=evidence_text),
        max_tokens=4096,
        temperature=0.3,
    )
    trace.append(tool="generate",
                 input_args={"max_tokens": 4096, "model": model_id},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    state.save("report", {"markdown_length": len(report_md)})

    # ── Assemble ─────────────────────────────────────────────────────────
    elapsed = time.time() - start

    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=extractions,
        pattern_name="p10_deep_researcher",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
    )
    report.elapsed_seconds = elapsed

    log.info("p10_complete",
             cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens,
             sections=len(report.sections),
             elapsed=f"{elapsed:.1f}s")

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed": elapsed,
    })

    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = search_queries
    report.metadata["search_queries_sent"] = search_queries
    report.metadata["n_documents_retrieved"] = len(gathered_docs)
    report.metadata["n_extractions"] = len(extractions)
    report.metadata["local_model"] = model_id
    report.metadata["quantize_4bit"] = quantize_4bit
    report.metadata["agent_search_iterations"] = len(search_queries)
    report.metadata["agent_findings_length"] = len(agent_findings)

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len({d.url for d in gathered_docs if d.url})
    trace.n_iterations = len(search_queries)
    state.save("trace", trace.model_dump(mode="json"))

    return report
