"""Assemble final ResearchReport from generated markdown and source extractions."""

from __future__ import annotations

from typing import List

from deep_research.tools import SourceExtraction
from deep_research.types import ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report


def assemble_report(
    query: str,
    markdown: str,
    extractions: List[SourceExtraction],
    pattern_name: str = "p1_iterative_rag",
    cost_usd: float = 0.0,
    total_tokens: int = 0,
) -> ResearchReport:
    """Parse markdown report into structured ResearchReport."""
    return parse_markdown_report(
        query=query,
        markdown=markdown,
        extractions=extractions,
        pattern_name=pattern_name,
        cost_usd=cost_usd,
        total_tokens=total_tokens,
    )
