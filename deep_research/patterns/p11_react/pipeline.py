"""Pattern 11: Canonical ReAct (Yao et al. 2022, arXiv:2210.03629).

Single-agent Thought→Action→Observation loop with three action primitives
(search, read, academic_search) plus a finish() emitter. No orchestration,
no perspective expansion, no quality eval — this is the clean ReAct
baseline that Step-DeepResearch [stepdeepresearch2025] argues outperforms
multi-agent decomposition.

Loop semantics:
  - Up to 8 turns (configurable). At each turn the model produces
    `Thought:` + `Action:`; the system parses the action, executes it,
    and feeds the resulting `Observation:` back as the next user turn.
  - When the model emits `Action: finish()`, the loop ends and the model
    is prompted in a separate call to write the final markdown report
    using the accumulated trace as context.

Differs from P6 (Reactive Interleaved): P6 follows the WebThinker design
where the model interleaves search / read / write within a single
streaming generation; P11 is the strict 1972-vintage Thought→Action→
Observation loop with discrete tool dispatches.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtraction,
    SourceExtractor,
    URLExtractor,
    format_extractions_as_evidence,
    get_web_searcher,
    StateManager,
)
from deep_research.types import Document, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report
from deep_research.patterns.p11_react.prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_USER_INITIAL,
    REACT_USER_OBSERVATION,
    REACT_FINAL_REPORT_PROMPT,
)

log = structlog.get_logger()


MAX_TURNS = 8
OBSERVATION_TRUNCATE = 2000  # chars per observation; full content cached separately
SEARCH_RESULTS_PER_QUERY = 5
ACADEMIC_RESULTS_PER_QUERY = 3
DEFAULT_BUDGET = MAX_COST_PER_RUN


@dataclass
class TraceTurn:
    """One turn of the ReAct loop."""
    turn: int
    thought: str
    action_kind: str  # "search" | "read" | "academic_search" | "finish" | "invalid"
    action_arg: str
    observation: str  # truncated for context, full saved separately


_ACTION_RE = re.compile(
    r"Action:\s*(search|read|academic_search|finish)\s*\(\s*(?:\"([^\"]*)\"|\'([^\']*)\'|)?\s*\)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", flags=re.IGNORECASE | re.DOTALL)


def _parse_thought_action(text: str) -> tuple[str, str, str]:
    """Extract (thought, action_kind, action_arg) from a model response."""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    action_m = _ACTION_RE.search(text)
    if not action_m:
        return thought, "invalid", ""
    kind = action_m.group(1).lower()
    arg = action_m.group(2) or action_m.group(3) or ""
    return thought, kind, arg.strip()


def _truncate_obs(text: str, limit: int = OBSERVATION_TRUNCATE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[…truncated {len(text) - limit} chars]"


def _configured_max_turns(kwargs: dict) -> int:
    """Resolve the ReAct turn budget from kwargs or P11_MAX_TURNS."""
    raw = kwargs.get("max_turns") or os.environ.get("P11_MAX_TURNS") or MAX_TURNS
    try:
        turns = int(raw)
    except (TypeError, ValueError):
        turns = MAX_TURNS
    return max(1, turns)


async def run(query: str, budget_usd: float = DEFAULT_BUDGET, **kwargs) -> ResearchReport:
    """Execute the canonical ReAct loop on a research query."""
    max_turns = _configured_max_turns(kwargs)
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p11_react")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()

    log.info("p11_start", query=query[:80], max_turns=max_turns)

    trace: List[TraceTurn] = []
    cited_documents: dict[str, Document] = {}  # url -> doc; preserves all evidence
    citation_order: List[str] = []  # urls in first-seen order

    # Build the running messages history for the LLM (system + alternating user/assistant)
    history: List[dict] = []
    history.append({"role": "user", "content": REACT_USER_INITIAL.format(query=query)})

    finished = False
    system_prompt = REACT_SYSTEM_PROMPT.format(max_turns=max_turns)
    for turn in range(1, max_turns + 1):
        # Convert history into a single prompt using the LLMCaller system+prompt API
        # (LLMCaller.complete takes one prompt string + system; we reconstruct dialog)
        prompt = _format_history_as_prompt(history)
        try:
            response = await llm.complete(
                prompt,
                model=DEFAULT_MODEL,
                system=system_prompt,
                temperature=0.3,
                max_tokens=512,
            )
        except Exception as e:
            log.error("p11_llm_error", turn=turn, error=str(e)[:120])
            break

        thought, kind, arg = _parse_thought_action(response)
        log.info("p11_turn", turn=turn, kind=kind, arg=arg[:60])
        observation = ""

        if kind == "finish":
            trace.append(TraceTurn(turn=turn, thought=thought, action_kind="finish",
                                   action_arg="", observation="(loop terminated)"))
            history.append({"role": "assistant", "content": response})
            finished = True
            break

        if kind == "search":
            try:
                docs = await web.search(arg or query, max_results=SEARCH_RESULTS_PER_QUERY)
                lines = []
                for i, d in enumerate(docs, 1):
                    if d.url and d.url not in cited_documents:
                        cited_documents[d.url] = d
                        citation_order.append(d.url)
                    cidx = citation_order.index(d.url) + 1 if d.url in citation_order else 0
                    preview = (d.content or "")[:200]
                    lines.append(f"  [{cidx}] {(d.title or '(no title)')[:90]}\n      {d.url}\n      {preview}")
                observation = f"Web search '{arg}' returned {len(docs)} results:\n" + "\n".join(lines)
            except Exception as e:
                observation = f"Search error: {str(e)[:200]}"

        elif kind == "academic_search":
            try:
                docs = await academic.search(arg or query, max_per_source=ACADEMIC_RESULTS_PER_QUERY)
                lines = []
                for d in docs:
                    if d.url and d.url not in cited_documents:
                        cited_documents[d.url] = d
                        citation_order.append(d.url)
                    cidx = citation_order.index(d.url) + 1 if d.url in citation_order else 0
                    preview = (d.content or "")[:300]
                    lines.append(f"  [{cidx}] {(d.title or '(no title)')[:100]}\n      {d.url}\n      {preview}")
                observation = f"Academic search '{arg}' returned {len(docs)} papers:\n" + "\n".join(lines)
            except Exception as e:
                observation = f"Academic search error: {str(e)[:200]}"

        elif kind == "read":
            url = arg.strip()
            if not url:
                observation = "read() requires a URL argument"
            else:
                # Check cache first
                if url in cited_documents and cited_documents[url].content and len(cited_documents[url].content) > 500:
                    observation = _truncate_obs(cited_documents[url].content)
                else:
                    try:
                        results = await url_extractor.extract_batch([url])
                        if results and results[0].content:
                            content = results[0].content
                            # Cache as a Document if not already
                            if url not in cited_documents:
                                cited_documents[url] = Document(
                                    url=url, title=results[0].title or url,
                                    content=content,
                                )
                                citation_order.append(url)
                            else:
                                cited_documents[url].content = content
                            observation = _truncate_obs(content)
                        else:
                            observation = f"Could not extract content from {url}"
                    except Exception as e:
                        observation = f"Read error: {str(e)[:200]}"

        else:
            # invalid action format
            observation = ("Invalid action. Output exactly:\n"
                           '  Thought: <reasoning>\n'
                           '  Action: search("...") | read("...") | academic_search("...") | finish()')

        trace.append(TraceTurn(turn=turn, thought=thought, action_kind=kind,
                               action_arg=arg, observation=observation))
        history.append({"role": "assistant", "content": response})
        history.append({"role": "user", "content": REACT_USER_OBSERVATION.format(observation=observation)})

    # ── Persist intermediate state ───────────────────────────────────────
    state.save("react_trace", {
        "turns": [
            {"turn": t.turn, "thought": t.thought, "action_kind": t.action_kind,
             "action_arg": t.action_arg, "observation_chars": len(t.observation)}
            for t in trace
        ],
        "n_turns": len(trace),
        "max_turns": max_turns,
        "finished_naturally": finished,
        "n_documents": len(cited_documents),
    })

    if not cited_documents:
        log.warning("p11_no_documents")
        return ResearchReport(
            query=query, title=query, pattern_name="p11_react",
            total_cost_usd=tracker.total_cost, total_tokens=tracker.total_tokens,
        )

    # ── Build extractions via the canonical SourceExtractor for evidence formatting ──
    docs_in_order = [cited_documents[u] for u in citation_order]
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    extractions = await source_extractor.extract_batch(docs_in_order, query)

    evidence_text = format_extractions_as_evidence(extractions)
    trace_text = _format_trace_for_report(trace)

    log.info("p11_generating_final", cost=f"${tracker.total_cost:.4f}",
             trace_turns=len(trace), n_docs=len(cited_documents))
    final_md = await llm.complete(
        REACT_FINAL_REPORT_PROMPT.format(query=query, trace=trace_text, evidence=evidence_text),
        model=DEFAULT_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    state.save("report", {"markdown": final_md})

    report = parse_markdown_report(
        query=query, markdown=final_md, extractions=extractions,
        pattern_name="p11_react",
        cost_usd=tracker.total_cost, total_tokens=tracker.total_tokens,
    )
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [t.action_arg for t in trace if t.action_kind in ("search", "academic_search")]
    report.metadata["search_queries_sent"] = [t.action_arg for t in trace if t.action_kind == "search"]
    report.metadata["n_documents_retrieved"] = len(cited_documents)
    report.metadata["n_extractions"] = len(extractions)
    report.metadata["react_turns"] = len(trace)
    report.metadata["react_max_turns"] = max_turns
    report.metadata["react_finished_naturally"] = finished

    log.info("p11_complete", cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens, sections=len(report.sections),
             turns=len(trace), max_turns=max_turns, finished_naturally=finished)
    state.save("final", {"cost": tracker.total_cost, "tokens": tracker.total_tokens})
    return report


def _format_history_as_prompt(history: List[dict]) -> str:
    """Render alternating user/assistant turns as a single prompt for LLMCaller.complete()."""
    parts: List[str] = []
    for msg in history:
        role = msg["role"].upper()
        parts.append(f"{role}:\n{msg['content']}")
    return "\n\n".join(parts)


def _format_trace_for_report(trace: List[TraceTurn]) -> str:
    blocks: List[str] = []
    for t in trace:
        action_str = (
            f"{t.action_kind}(\"{t.action_arg}\")" if t.action_kind in ("search", "read", "academic_search")
            else f"{t.action_kind}()"
        )
        blocks.append(
            f"Turn {t.turn}:\n"
            f"  Thought: {t.thought}\n"
            f"  Action: {action_str}\n"
            f"  Observation: {t.observation[:600]}{'…' if len(t.observation) > 600 else ''}"
        )
    return "\n\n".join(blocks)
