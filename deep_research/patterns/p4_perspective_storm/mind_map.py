"""Stage 3: Build a mind map / structured outline from conversation insights."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import Perspective
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

MIND_MAP_PROMPT = """You are a research analyst creating a structured mind map from expert
conversation transcripts. Analyze the conversations below and organize the key insights
into a hierarchical topic structure.

Research query: {query}

Expert perspectives consulted:
{perspectives_text}

Conversation transcripts:
{conversations_text}

Create a mind map that:
1. Identifies 5-8 major topic clusters that emerged across conversations
2. For each cluster, lists 3-5 key claims or findings with the perspectives that support them
3. Notes areas of agreement and disagreement between perspectives
4. Identifies cross-cutting themes that span multiple clusters
5. Flags claims that need triangulation (mentioned by only one perspective)

Return JSON:
{{
  "topic_clusters": [
    {{
      "topic": "Cluster title",
      "summary": "1-2 sentence summary of this cluster",
      "key_claims": [
        {{
          "claim": "Specific factual claim or finding",
          "supporting_perspectives": ["Perspective Name 1", "Perspective Name 2"],
          "confidence": "high|medium|low",
          "evidence_type": "empirical|theoretical|anecdotal|consensus"
        }}
      ],
      "sub_topics": ["sub-topic 1", "sub-topic 2"]
    }}
  ],
  "cross_cutting_themes": [
    {{
      "theme": "Theme description",
      "related_clusters": ["Cluster title 1", "Cluster title 2"],
      "insight": "What this theme reveals when looking across clusters"
    }}
  ],
  "agreements": [
    {{
      "claim": "Claim that multiple perspectives agree on",
      "perspectives": ["Perspective 1", "Perspective 2"],
      "strength": "strong|moderate|weak"
    }}
  ],
  "disagreements": [
    {{
      "claim": "Point of contention",
      "positions": [
        {{"perspective": "Perspective 1", "position": "Their stance"}},
        {{"perspective": "Perspective 2", "position": "Their stance"}}
      ],
      "nature": "factual|methodological|interpretive|values-based"
    }}
  ],
  "needs_triangulation": [
    {{
      "claim": "Claim from a single perspective needing verification",
      "source_perspective": "Perspective Name",
      "suggested_search_queries": ["query1", "query2"]
    }}
  ]
}}
"""

OUTLINE_PROMPT = """Based on the following mind map of a research topic, create a detailed
report outline suitable for a comprehensive research report.

Research query: {query}

Mind map:
{mind_map_json}

Create a report outline with:
- A compelling title
- An abstract outline (key points to cover)
- 5-8 sections, each with:
  - A section title
  - Key points to address
  - Which perspective insights to draw from
  - Which claims to include (with their confidence levels)

Return JSON:
{{
  "title": "Report title",
  "abstract_points": ["point1", "point2", "point3"],
  "sections": [
    {{
      "title": "Section title",
      "key_points": ["point1", "point2"],
      "perspectives_to_cite": ["Perspective 1", "Perspective 2"],
      "claims_to_include": ["claim text 1", "claim text 2"],
      "notes": "Any special instructions for this section"
    }}
  ]
}}
"""


async def build_mind_map(
    query: str,
    perspectives: List[Perspective],
    conversations_text: str,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Analyze conversation transcripts and build a structured mind map.

    Args:
        query: The research query.
        perspectives: All perspectives that participated.
        conversations_text: Concatenated conversation transcripts.
        llm: LLM caller instance.
        model: Model to use for analysis.

    Returns:
        Dict containing topic_clusters, cross_cutting_themes, agreements,
        disagreements, and needs_triangulation.
    """
    perspectives_text = "\n".join(
        f"- {p.name}: {p.description}" for p in perspectives
    )

    # Truncate conversations if too long (keep within context limits)
    max_conv_chars = 200_000
    if len(conversations_text) > max_conv_chars:
        log.warning("truncating_conversations",
                    original=len(conversations_text), target=max_conv_chars)
        conversations_text = conversations_text[:max_conv_chars] + "\n[... truncated ...]"

    log.info("building_mind_map", perspectives=len(perspectives),
             conv_length=len(conversations_text))

    mind_map = await llm.complete_json(
        MIND_MAP_PROMPT.format(
            query=query,
            perspectives_text=perspectives_text,
            conversations_text=conversations_text,
        ),
        model=model,
        temperature=0.3,
        max_tokens=4096,
    )

    # Validate and log structure
    n_clusters = len(mind_map.get("topic_clusters", []))
    n_agreements = len(mind_map.get("agreements", []))
    n_disagreements = len(mind_map.get("disagreements", []))
    n_triangulate = len(mind_map.get("needs_triangulation", []))

    log.info("mind_map_built",
             clusters=n_clusters,
             agreements=n_agreements,
             disagreements=n_disagreements,
             needs_triangulation=n_triangulate)

    return mind_map


async def build_outline(
    query: str,
    mind_map: Dict[str, Any],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Create a report outline from the mind map.

    Args:
        query: The research query.
        mind_map: The structured mind map.
        llm: LLM caller instance.
        model: Model to use.

    Returns:
        Dict containing the report outline with title, abstract_points, and sections.
    """
    import json

    mind_map_json = json.dumps(mind_map, indent=2, default=str)

    # Truncate if the mind map serialization is very large
    max_map_chars = 100_000
    if len(mind_map_json) > max_map_chars:
        mind_map_json = mind_map_json[:max_map_chars] + "\n... [truncated]"

    log.info("building_outline", mind_map_size=len(mind_map_json))

    outline = await llm.complete_json(
        OUTLINE_PROMPT.format(
            query=query,
            mind_map_json=mind_map_json,
        ),
        model=model,
        temperature=0.3,
        max_tokens=3072,
    )

    n_sections = len(outline.get("sections", []))
    log.info("outline_built", title=outline.get("title", "")[:60],
             sections=n_sections)

    return outline


def extract_triangulation_queries(mind_map: Dict[str, Any]) -> List[str]:
    """Extract search queries for claims that need triangulation.

    Args:
        mind_map: The structured mind map.

    Returns:
        List of search queries for claims needing verification.
    """
    queries: List[str] = []
    for item in mind_map.get("needs_triangulation", []):
        queries.extend(item.get("suggested_search_queries", []))
    return queries
