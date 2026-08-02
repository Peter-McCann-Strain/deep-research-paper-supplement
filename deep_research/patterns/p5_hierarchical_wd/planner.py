"""Planner: Generate initial research plan with subtopics and search strategy.

Uses gpt-5.2 to decompose the research query into a structured plan of
subtopics, search queries, and prioritized investigation areas.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import SubQuery
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

PLAN_SYSTEM = """You are a research planning expert. Your job is to decompose a complex
research query into a structured investigation plan. Be thorough and systematic."""

PLAN_PROMPT = """Analyze this research query and create a comprehensive research plan.

Research Query: {query}

Create a plan with:
1. A title for the research
2. 4-8 major subtopics that together cover the query comprehensively
3. For each subtopic: 2-4 specific search queries (both web and academic)
4. Priority ranking (1=highest, 3=lowest) for each subtopic
5. Key aspects to investigate for depth analysis

Return JSON:
{{
    "title": "Research title",
    "subtopics": [
        {{
            "name": "Subtopic name",
            "description": "What this subtopic covers",
            "priority": 1,
            "search_queries": ["query1", "query2", "query3"],
            "depth_questions": ["What specific mechanisms...", "How does X compare to Y..."]
        }}
    ],
    "cross_cutting_themes": ["theme1", "theme2"],
    "key_controversies": ["controversy1"]
}}"""


async def create_research_plan(
    query: str,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate a structured research plan from the query.

    Returns:
        Dict with keys: title, subtopics, cross_cutting_themes, key_controversies.
        Each subtopic has: name, description, priority, search_queries, depth_questions.
    """
    log.info("planner_start", query=query[:80])

    result = await llm.complete_json(
        PLAN_PROMPT.format(query=query),
        model=model,
        system=PLAN_SYSTEM,
        temperature=0.4,
        max_tokens=4096,
    )

    subtopics = result.get("subtopics", [])
    log.info(
        "planner_complete",
        title=result.get("title", "")[:60],
        subtopics=len(subtopics),
        total_queries=sum(len(s.get("search_queries", [])) for s in subtopics),
    )

    return result


def extract_sub_queries(plan: Dict[str, Any]) -> List[SubQuery]:
    """Extract flat list of SubQuery objects from the research plan."""
    queries: List[SubQuery] = []
    for subtopic in plan.get("subtopics", []):
        priority = subtopic.get("priority", 2)
        intent = subtopic.get("name", "general")
        for q in subtopic.get("search_queries", []):
            queries.append(SubQuery(
                query=q,
                intent=intent,
                priority=priority,
            ))
    return queries


def extract_depth_questions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract depth questions grouped by subtopic for the depth phase."""
    grouped: List[Dict[str, Any]] = []
    for subtopic in plan.get("subtopics", []):
        questions = subtopic.get("depth_questions", [])
        if questions:
            grouped.append({
                "subtopic": subtopic.get("name", ""),
                "description": subtopic.get("description", ""),
                "priority": subtopic.get("priority", 2),
                "questions": questions,
            })
    # Sort by priority (ascending = highest priority first)
    grouped.sort(key=lambda x: x["priority"])
    return grouped
