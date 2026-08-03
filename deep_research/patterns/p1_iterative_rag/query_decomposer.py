"""Stage 1: Decompose query into sub-queries."""

from __future__ import annotations

from typing import List

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import SubQuery
from deep_research.config import DEFAULT_MODEL

DECOMPOSE_PROMPT = """Given this research query, generate {n} diverse search sub-queries that would help
comprehensively answer it. Include variations: synonyms, related concepts, specific aspects,
comparison angles, and methodological questions.

Research query: {query}

Return JSON: {{"queries": [{{"query": "...", "intent": "...", "priority": 1-3}}]}}
"""


async def decompose_query(
    query: str,
    llm: LLMCaller,
    n_queries: int = 25,
    model: str = DEFAULT_MODEL,
) -> List[SubQuery]:
    """Decompose a research query into diverse sub-queries."""
    result = await llm.complete_json(
        DECOMPOSE_PROMPT.format(query=query, n=n_queries),
        model=model,
    )

    sub_queries = []
    for item in result.get("queries", []):
        sub_queries.append(SubQuery(
            query=item.get("query", ""),
            intent=item.get("intent", ""),
            priority=item.get("priority", 1),
        ))

    return sub_queries
