"""Stage 1: Discover 4-6 diverse research perspectives for the query."""

from __future__ import annotations

from typing import List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import Perspective
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

DISCOVER_PROMPT = """You are a research methodology expert. Given the following research query,
identify {n} diverse expert perspectives that would provide comprehensive coverage of the topic.

Each perspective should represent a distinct domain, methodology, or stakeholder viewpoint.
Ensure the perspectives are complementary — together they should cover theoretical foundations,
empirical evidence, practical applications, critical analysis, and emerging trends.

Research query: {query}

Return JSON:
{{
  "perspectives": [
    {{
      "name": "Short perspective title (e.g., 'Computational Neuroscientist')",
      "description": "2-3 sentence description of this expert's background and how they approach the topic",
      "focus_areas": ["area1", "area2", "area3"]
    }}
  ]
}}
"""

SEARCH_QUERIES_PROMPT = """You are a research librarian. Given a research query and a set of expert
perspectives, generate targeted search queries that would help each expert prepare for a
research discussion.

Research query: {query}

Perspectives:
{perspectives_text}

Generate 2-3 search queries per perspective, plus 2-3 general queries that span all perspectives.

Return JSON:
{{
  "perspective_queries": {{
    "perspective_name": ["query1", "query2", "query3"]
  }},
  "general_queries": ["query1", "query2", "query3"]
}}
"""


async def discover_perspectives(
    query: str,
    llm: LLMCaller,
    n_perspectives: int = 5,
    model: str = DEFAULT_MODEL,
) -> List[Perspective]:
    """Discover diverse research perspectives for the query.

    Args:
        query: The main research question.
        llm: LLM caller instance.
        n_perspectives: Target number of perspectives (4-6).
        model: Model to use for discovery.

    Returns:
        List of Perspective objects representing expert viewpoints.
    """
    log.info("discovering_perspectives", query=query[:80], target=n_perspectives)

    result = await llm.complete_json(
        DISCOVER_PROMPT.format(query=query, n=n_perspectives),
        model=model,
        temperature=0.5,
    )

    perspectives: List[Perspective] = []
    for item in result.get("perspectives", []):
        perspectives.append(Perspective(
            name=item.get("name", "Unknown Expert"),
            description=item.get("description", ""),
            focus_areas=item.get("focus_areas", []),
        ))

    # Ensure we have at least 4 perspectives
    if len(perspectives) < 4:
        log.warning("few_perspectives", count=len(perspectives))

    log.info("perspectives_discovered", count=len(perspectives),
             names=[p.name for p in perspectives])
    return perspectives


async def generate_search_queries(
    query: str,
    perspectives: List[Perspective],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Generate targeted search queries for each perspective.

    Args:
        query: The main research question.
        perspectives: Discovered perspectives.
        llm: LLM caller instance.
        model: Model to use.

    Returns:
        Dict with 'perspective_queries' mapping perspective names to query lists,
        and 'general_queries' as a list of cross-cutting queries.
    """
    perspectives_text = "\n".join(
        f"- {p.name}: {p.description} (Focus: {', '.join(p.focus_areas)})"
        for p in perspectives
    )

    result = await llm.complete_json(
        SEARCH_QUERIES_PROMPT.format(
            query=query,
            perspectives_text=perspectives_text,
        ),
        model=model,
        temperature=0.3,
    )

    perspective_queries = result.get("perspective_queries", {})
    general_queries = result.get("general_queries", [])

    total_queries = sum(len(qs) for qs in perspective_queries.values()) + len(general_queries)
    log.info("search_queries_generated", total=total_queries,
             per_perspective={k: len(v) for k, v in perspective_queries.items()})

    return {
        "perspective_queries": perspective_queries,
        "general_queries": general_queries,
    }
