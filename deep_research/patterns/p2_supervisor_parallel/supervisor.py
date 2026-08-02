"""Supervisor: plans research strategy, dispatches workers, manages gap-fill."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import SubQuery
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# ── Prompts ──────────────────────────────────────────────────────────────────

PLAN_PROMPT = """You are a research supervisor. Given a user query, produce a detailed research
plan consisting of {n_workers} independent sub-topics that can be investigated in parallel.

Each sub-topic should be:
- Self-contained (a worker can search and summarise it alone)
- Complementary (together they cover the full query)
- Diverse (cover different dimensions: definitions, mechanisms, evidence, applications,
  controversies, recent developments, comparisons, limitations)

Research query: {query}

Return JSON:
{{
  "title": "short research title",
  "sub_topics": [
    {{
      "query": "precise search query for this sub-topic",
      "intent": "what this sub-topic should cover",
      "priority": 1-3,
      "search_type": "web" | "academic" | "both"
    }}
  ]
}}
"""

GAP_FILL_PROMPT = """You are a research supervisor reviewing partial findings for gaps.

Original query: {query}

Plan title: {title}

Sub-topics already covered:
{covered_topics}

Quality gate feedback:
{feedback}

Identified gaps:
{gaps}

Generate {n_gap_workers} new sub-topic assignments to fill these gaps.
Each should target a specific gap that was NOT adequately covered.

Return JSON:
{{
  "gap_sub_topics": [
    {{
      "query": "precise search query for this gap",
      "intent": "what gap this fills",
      "priority": 1,
      "search_type": "web" | "academic" | "both"
    }}
  ]
}}
"""


async def plan_research(
    query: str,
    llm: LLMCaller,
    n_workers: int = 5,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Create a research plan with N parallel sub-topic assignments.

    Returns:
        dict with keys: "title", "sub_topics" (list of dicts).
    """
    result = await llm.complete_json(
        PLAN_PROMPT.format(query=query, n_workers=n_workers),
        model=model,
        temperature=0.4,
        max_tokens=4096,
    )

    title = result.get("title", query[:80])
    raw_topics = result.get("sub_topics", [])

    # Validate and normalise
    sub_topics: List[Dict[str, Any]] = []
    for t in raw_topics:
        sub_topics.append({
            "query": t.get("query", ""),
            "intent": t.get("intent", ""),
            "priority": t.get("priority", 2),
            "search_type": t.get("search_type", "both"),
        })

    log.info("supervisor_plan", title=title, sub_topics=len(sub_topics))
    return {"title": title, "sub_topics": sub_topics}


def plan_to_sub_queries(plan: Dict[str, Any]) -> List[SubQuery]:
    """Convert plan sub-topics to typed SubQuery objects."""
    return [
        SubQuery(
            query=t["query"],
            intent=t["intent"],
            priority=t["priority"],
        )
        for t in plan.get("sub_topics", [])
    ]


async def plan_gap_fill(
    query: str,
    plan_title: str,
    covered_topics: List[str],
    gaps: List[str],
    feedback: str,
    llm: LLMCaller,
    n_gap_workers: int = 2,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Generate new sub-topic assignments to fill identified gaps.

    Returns:
        List of sub-topic dicts (same schema as plan sub_topics).
    """
    covered_str = "\n".join(f"- {t}" for t in covered_topics)
    gaps_str = "\n".join(f"- {g}" for g in gaps)

    result = await llm.complete_json(
        GAP_FILL_PROMPT.format(
            query=query,
            title=plan_title,
            covered_topics=covered_str,
            feedback=feedback,
            gaps=gaps_str,
            n_gap_workers=n_gap_workers,
        ),
        model=model,
        temperature=0.3,
        max_tokens=2048,
    )

    gap_topics = []
    for t in result.get("gap_sub_topics", []):
        gap_topics.append({
            "query": t.get("query", ""),
            "intent": t.get("intent", ""),
            "priority": t.get("priority", 1),
            "search_type": t.get("search_type", "both"),
        })

    log.info("supervisor_gap_fill", new_sub_topics=len(gap_topics))
    return gap_topics
