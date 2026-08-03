"""Report Writer: generates final polished report from compressed findings and source summaries."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import SourceExtraction, format_extractions_as_evidence
from deep_research.types import ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

from .compressor import CompressedOutput
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# -- Prompt --------------------------------------------------------------------

REPORT_PROMPT = """You are a senior research report writer. Using the compressed research
sections and source evidence below, produce a polished, comprehensive research report.

Original query: {query}

Abstract:
{abstract}

Compressed sections:
{sections}

Source evidence (numbered):
{evidence}

Requirements:
- Start with a title (# Title) and the abstract (## Abstract)
- Expand each section with smooth prose, logical transitions, and inline citations [1], [2], etc.
- Add a ## Conclusion section summarising key findings and implications
- Be specific: preserve all numbers, dates, method names, and benchmark results
- Acknowledge limitations and areas where evidence is conflicting or sparse
- Aim for 2000-4000 words
- Use markdown formatting

Write the complete report.
"""


async def write_report(
    query: str,
    abstract: str,
    compressed: CompressedOutput,
    source_summaries: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate the final polished report markdown.

    Args:
        query: original research query.
        abstract: pre-generated abstract.
        compressed: compressed findings with sections.
        source_summaries: list of SourceExtraction objects for evidence citation.
        llm: LLMCaller instance.
        model: model for report writing (heavy model for quality).

    Returns:
        Full report as markdown string.
    """
    sections_text = compressed.sections_as_text()
    evidence_text = format_extractions_as_evidence(source_summaries[:50])

    report_md = await llm.complete(
        REPORT_PROMPT.format(
            query=query,
            abstract=abstract,
            sections=sections_text,
            evidence=evidence_text,
        ),
        model=model,
        temperature=0.3,
        max_tokens=8192,
    )

    log.info("report_written", length=len(report_md))
    return report_md


def assemble_research_report(
    query: str,
    report_markdown: str,
    source_summaries: List[SourceExtraction],
    cost_usd: float = 0.0,
    total_tokens: int = 0,
    elapsed_seconds: float = 0.0,
) -> ResearchReport:
    """Parse the generated markdown into a structured ResearchReport.

    Args:
        query: original research query.
        report_markdown: full report in markdown format.
        source_summaries: list of SourceExtraction objects for citation building.
        cost_usd: total cost of the run.
        total_tokens: total tokens used.
        elapsed_seconds: total elapsed time.

    Returns:
        Structured ResearchReport object.
    """
    report = parse_markdown_report(
        query=query,
        markdown=report_markdown,
        extractions=source_summaries,
        pattern_name="p2_supervisor_parallel",
        cost_usd=cost_usd,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
        metadata={
            "section_count": 0,  # filled below
            "citation_count": 0,  # filled below
        },
    )

    # Update metadata with actual counts
    report.metadata["section_count"] = len(report.sections)
    report.metadata["citation_count"] = len(report.citations)

    log.info("report_assembled", title=report.title[:60], sections=len(report.sections),
             citations=len(report.citations))
    return report
