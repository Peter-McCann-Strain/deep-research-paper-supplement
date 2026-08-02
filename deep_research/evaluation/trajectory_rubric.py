"""E8: Process-trajectory rubric.

Scores a pattern's ProcessTrace on 4 dimensions complementary to the V2
outcome rubric. Two dimensions are computable purely from trace metadata
(retrieval_diversity, tool_efficiency); two require an LLM judge to score
qualitatively against the trace+report (reasoning_coherence,
iterative_refinement).

Used to test whether top-cluster patterns that score equivalently on the
final-report rubric differ on process quality.

Outputs match the existing per-dimension rubric shape so they can be
joined into df_scores.parquet alongside V2 dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from deep_research.types import ProcessTrace


PROCESS_DIMENSIONS = [
    "retrieval_diversity",
    "reasoning_coherence",
    "iterative_refinement",
    "tool_efficiency",
]


@dataclass
class ProcessScore:
    """Per-dimension score in [0, 1] plus a human-readable rationale."""
    dimension: str
    score: float
    rationale: str
    is_metadata_only: bool  # True if score derived from trace metadata, no LLM call


# ── Metadata-only scorers ───────────────────────────────────────────────────

def score_retrieval_diversity(trace: ProcessTrace) -> ProcessScore:
    """Diversity of retrieval calls.

    Encodes:
      - Number of distinct search queries (more is better up to a cap)
      - Penalty for repeating identical queries
      - Bonus for academic_search vs only web search

    Returns score in [0, 1] using a logarithmic scale capped at 8 unique queries.
    """
    search_calls = [c for c in trace.tool_calls
                    if c.tool in ("search", "academic_search")]
    if not search_calls:
        return ProcessScore("retrieval_diversity", 0.0,
                            "No search calls in trace.", True)

    unique_q = {(c.tool, c.input_args.get("query", "").lower().strip())
                for c in search_calls}
    n_unique = len(unique_q)
    # Logarithmic scale: 1 unique = 0.2, 4 unique = 0.7, 8+ = 1.0
    import math
    diversity = min(1.0, math.log2(n_unique + 1) / 3.17)

    # Bonus if academic_search is used at least once
    has_academic = any(c.tool == "academic_search" for c in search_calls)
    if has_academic:
        diversity = min(1.0, diversity + 0.05)

    # Penalty if repeats are common
    repeat_ratio = 1 - n_unique / len(search_calls)
    if repeat_ratio > 0.5:
        diversity *= 0.7

    rationale = (f"{n_unique} unique queries across {len(search_calls)} search calls; "
                 f"{'with' if has_academic else 'no'} academic_search; "
                 f"repeat ratio {repeat_ratio:.2f}")
    return ProcessScore("retrieval_diversity", round(diversity, 3), rationale, True)


def score_tool_efficiency(trace: ProcessTrace) -> ProcessScore:
    """Tokens used relative to research-report yield.

    Lower tokens-per-citation and lower wall-clock per section indicate efficiency.
    Normalised against expected ranges from the existing 990-report distribution.
    """
    total_tokens = sum(c.tokens_used for c in trace.tool_calls)
    if total_tokens <= 0 or trace.final_report_word_count == 0:
        return ProcessScore("tool_efficiency", 0.0,
                            "No tokens or empty final report.", True)

    # Words-per-token: higher is better; cap at 0.5 (typical English)
    yield_ratio = trace.final_report_word_count / total_tokens
    # Normalise: typical good is 0.05-0.10, bad is <0.01
    eff = min(1.0, yield_ratio / 0.10)

    rationale = (f"{total_tokens:,} tokens for {trace.final_report_word_count}-word report "
                 f"(ratio {yield_ratio:.4f})")
    return ProcessScore("tool_efficiency", round(eff, 3), rationale, True)


# ── LLM-judged scorers ──────────────────────────────────────────────────────

REASONING_COHERENCE_PROMPT = """You are evaluating the reasoning coherence of a research-agent's tool-call trace. Score the trace on a 0.0–1.0 scale.

A coherent trace:
  - Each search query builds on prior observations (not random)
  - Read calls are aimed at sources discovered by earlier searches
  - The final report integrates evidence retrieved during the trace
  - There are no obvious dead ends or unmotivated detours

An incoherent trace:
  - Search queries are unrelated to each other or the original query
  - Read calls fetch URLs that are never cited in the final report
  - Long detours that don't contribute to the final answer
  - Repeated near-identical operations without progress

Output JSON only: {{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}.

Original query: {query}

Tool-call trace:
{trace_summary}

Final report excerpt (first 2000 chars):
{report_excerpt}"""


ITERATIVE_REFINEMENT_PROMPT = """You are evaluating whether a research-agent's tool-call trace shows productive iterative refinement. Score on 0.0–1.0.

Productive refinement:
  - Later searches narrow or refine based on early findings
  - Reflection/critique steps lead to gap-filling searches
  - Multiple rounds with each round measurably improving evidence base
  - Quality-evaluator or self-critique calls trigger meaningful corrections

Unproductive (low score):
  - Just one search round with no refinement
  - "Refinement" loops that don't change behaviour
  - Iterations that re-fetch the same sources without adding new ones

Output JSON only: {{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}.

Original query: {query}

Tool-call trace:
{trace_summary}"""


def trace_to_summary(trace: ProcessTrace, max_calls: int = 30) -> str:
    """Compact text summary of the tool-call trace for LLM scoring."""
    calls = trace.tool_calls[:max_calls]
    lines = []
    for c in calls:
        arg = c.input_args.get("query") or c.input_args.get("url") or ""
        arg_short = str(arg)[:80]
        lines.append(f"  step {c.step_idx}: {c.tool}({arg_short!r}) -> "
                     f"{c.output_summary[:80]} (n_results={c.n_results}, tok={c.tokens_used})")
    if len(trace.tool_calls) > max_calls:
        lines.append(f"  ... ({len(trace.tool_calls) - max_calls} more steps)")
    return "\n".join(lines)


async def score_with_judge(
    trace: ProcessTrace,
    report_text: str,
    llm_caller,
    model: str,
    dimension: str,
) -> ProcessScore:
    """Send a trace+report to an LLM judge for a qualitative process score."""
    if dimension == "reasoning_coherence":
        prompt_template = REASONING_COHERENCE_PROMPT
        prompt = prompt_template.format(
            query=trace.query,
            trace_summary=trace_to_summary(trace),
            report_excerpt=report_text[:2000],
        )
    elif dimension == "iterative_refinement":
        prompt = ITERATIVE_REFINEMENT_PROMPT.format(
            query=trace.query,
            trace_summary=trace_to_summary(trace),
        )
    else:
        raise ValueError(f"Unknown LLM-judged dimension: {dimension}")

    raw = await llm_caller.complete_json(
        prompt, model=model, temperature=0.1, max_tokens=300,
    )
    if isinstance(raw, str):
        import json
        try:
            data = json.loads(raw)
        except Exception:
            data = {"score": 0.0, "rationale": "parse error"}
    else:
        data = raw or {}
    return ProcessScore(
        dimension=dimension,
        score=float(data.get("score", 0.0)),
        rationale=str(data.get("rationale", ""))[:300],
        is_metadata_only=False,
    )


def score_metadata_only(trace: ProcessTrace) -> dict[str, ProcessScore]:
    """Compute the two metadata-only process dimensions (no LLM cost)."""
    return {
        "retrieval_diversity": score_retrieval_diversity(trace),
        "tool_efficiency": score_tool_efficiency(trace),
    }


async def score_full(
    trace: ProcessTrace,
    report_text: str,
    llm_caller,
    model: str,
) -> dict[str, ProcessScore]:
    """Compute all 4 process dimensions (uses LLM for 2 of them)."""
    out = score_metadata_only(trace)
    out["reasoning_coherence"] = await score_with_judge(
        trace, report_text, llm_caller, model, "reasoning_coherence"
    )
    out["iterative_refinement"] = await score_with_judge(
        trace, report_text, llm_caller, model, "iterative_refinement"
    )
    return out
