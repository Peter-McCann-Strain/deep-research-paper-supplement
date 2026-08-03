"""Hypothesis generation — query to K candidate research directions.

Generates diverse research directions from a query, maximising coverage
of different angles (technical, economic, historical, etc.) so the beam
search can explore the most promising ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import SourceExtraction

log = structlog.get_logger()


# ── Data structure ───────────────────────────────────────────────────────────


@dataclass
class ResearchDirection:
    """A candidate research direction (beam) being explored."""

    id: str                                   # "dir_0", "dir_1", etc.
    thesis: str                               # The research angle / hypothesis
    key_questions: List[str]                  # Questions this direction answers
    angle: str                                # Brief perspective description
    expected_source_types: str                # "academic", "industry", "mixed"

    # Populated during broad exploration (Stage 2)
    promise_score: float = 0.0                # 1-10 from scorer
    evidence_summary: str = ""
    extractions: List[SourceExtraction] = field(default_factory=list)
    search_queries_used: List[str] = field(default_factory=list)
    n_docs_found: int = 0

    # Populated during deep investigation (Stage 4)
    detailed_analysis: str = ""
    evidence_quality_score: float = 0.0

    # Beam state
    is_alive: bool = True                     # False after pruning

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for checkpointing (omits non-serialisable fields)."""
        return {
            "id": self.id,
            "thesis": self.thesis,
            "key_questions": self.key_questions,
            "angle": self.angle,
            "expected_source_types": self.expected_source_types,
            "promise_score": self.promise_score,
            "evidence_summary": self.evidence_summary[:500],
            "n_docs_found": self.n_docs_found,
            "n_extractions": len(self.extractions),
            "search_queries_used": self.search_queries_used,
            "detailed_analysis": self.detailed_analysis[:500],
            "evidence_quality_score": self.evidence_quality_score,
            "is_alive": self.is_alive,
        }


# ── Generation ───────────────────────────────────────────────────────────────

_HYPOTHESIS_PROMPT = """Generate {n_hypotheses} diverse research directions for investigating this query. \
Each direction should explore a fundamentally different angle or perspective.

Research Query: {query}

For each direction provide:
- thesis: A specific research angle or hypothesis to investigate
- key_questions: 2-3 specific questions this direction would answer
- angle: Brief description of the perspective (e.g., "technical mechanisms", "economic impact", "historical context", "regulatory landscape", "comparative analysis")
- expected_source_types: "academic", "industry", "news", or "mixed"

IMPORTANT: Maximise diversity. Directions should be as different from each other as possible — \
covering technical, economic, social, historical, practical, theoretical, and other angles as appropriate.

Return JSON:
{{"directions": [
  {{"thesis": "...", "key_questions": ["q1", "q2"], "angle": "...", "expected_source_types": "..."}},
  ...
]}}"""


async def generate_hypotheses(
    llm: LLMCaller,
    query: str,
    n_hypotheses: int = 6,
) -> List[ResearchDirection]:
    """Generate K diverse research directions from a query.

    Args:
        llm: LLM caller instance.
        query: The original research query.
        n_hypotheses: Number of directions to generate (default 6).

    Returns:
        List of ResearchDirection objects, one per hypothesis.
    """
    prompt = _HYPOTHESIS_PROMPT.format(n_hypotheses=n_hypotheses, query=query)

    try:
        result = await llm.complete_json(
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=2048,
            temperature=0.7,
        )
    except Exception as exc:
        log.error("hypothesis_generation_json_failed", error=str(exc))
        # Fallback: try plain completion and parse JSON manually
        raw = await llm.complete(
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=2048,
            temperature=0.7,
        )
        result = _parse_json_fallback(raw)

    directions = _parse_directions(result, n_hypotheses)
    log.info(
        "hypotheses_generated",
        query=query[:60],
        count=len(directions),
        angles=[d.angle for d in directions],
    )
    return directions


def _parse_directions(
    data: Any,
    expected_count: int,
) -> List[ResearchDirection]:
    """Parse LLM output into ResearchDirection objects with fallbacks."""
    directions: List[ResearchDirection] = []

    raw_dirs = []
    if isinstance(data, dict):
        raw_dirs = data.get("directions", [])
    elif isinstance(data, list):
        raw_dirs = data

    for i, d in enumerate(raw_dirs):
        if not isinstance(d, dict):
            continue
        try:
            direction = ResearchDirection(
                id=f"dir_{i}",
                thesis=str(d.get("thesis", f"Direction {i}")),
                key_questions=[str(q) for q in d.get("key_questions", [])],
                angle=str(d.get("angle", "general")),
                expected_source_types=str(d.get("expected_source_types", "mixed")),
            )
            directions.append(direction)
        except Exception as exc:
            log.warning("direction_parse_error", index=i, error=str(exc))

    if not directions:
        log.warning("no_directions_parsed, creating fallback")
        directions = [
            ResearchDirection(
                id="dir_0",
                thesis="General investigation of the topic",
                key_questions=["What are the key aspects?", "What is the current state?"],
                angle="general overview",
                expected_source_types="mixed",
            )
        ]

    return directions


def _parse_json_fallback(raw: str) -> dict:
    """Attempt to extract JSON from a raw LLM response."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"directions": []}
