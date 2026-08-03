"""Stage 1: Decompose the research query into root sub-questions.

Produces 3-5 independent sub-questions that together cover the query from
complementary angles.  Unlike P1 which generates 25 flat queries, P7 starts
small and grows the graph dynamically.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller

log = structlog.get_logger()

DECOMPOSE_PROMPT = """Decompose this research query into {n} independent sub-questions \
that together would comprehensively answer the main query.  Each sub-question should \
explore a distinct aspect (e.g. background/theory, current state-of-the-art, \
applications, challenges, future directions).

Research Query: {query}

Return ONLY valid JSON in this format:
{{"sub_questions": ["question 1", "question 2", "question 3"]}}
"""


async def decompose_query(
    llm: LLMCaller,
    query: str,
    n_roots: int = 4,
) -> List[str]:
    """Decompose *query* into *n_roots* root sub-questions.

    Falls back to a simple three-way split if the LLM response cannot be parsed.
    """
    log.info("decompose_start", query=query[:80], n_roots=n_roots)

    try:
        result = await llm.complete_json(
            DECOMPOSE_PROMPT.format(query=query, n=n_roots),
            model=DEFAULT_MODEL,
            max_tokens=1024,
            temperature=0.4,
        )
        questions = result.get("sub_questions", [])
        if isinstance(questions, list) and len(questions) >= 2:
            log.info("decompose_done", n_questions=len(questions))
            return questions[:n_roots + 2]  # allow slight overshoot
    except Exception as exc:
        log.warning("decompose_json_failed", error=str(exc))

    # ── Fallback: ask for plain text and split by newlines ────────────
    log.info("decompose_fallback")
    try:
        raw = await llm.complete(
            f"List {n_roots} independent research sub-questions for: {query}\n"
            "Number each question on its own line.",
            model=DEFAULT_MODEL,
            max_tokens=512,
            temperature=0.4,
        )
        lines = [
            line.lstrip("0123456789.-) ").strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 10
        ]
        if lines:
            return lines[:n_roots + 2]
    except Exception as exc:
        log.warning("decompose_fallback_failed", error=str(exc))

    # ── Last resort: echo the original query as a single root ─────────
    return [query]
