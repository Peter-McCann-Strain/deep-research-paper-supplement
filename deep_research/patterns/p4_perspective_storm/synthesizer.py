"""Stage 5: Synthesize triangulated findings into a final research report.

Uses two-step source extractions as the evidence base instead of
retrieved/reranked chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import format_extractions_as_evidence, SourceExtraction
from deep_research.types import (
    Perspective,
    ResearchReport,
)
from deep_research.utils.markdown_parser import parse_markdown_report

log = structlog.get_logger()

SYNTHESIS_PROMPT = """You are an expert research synthesizer. Write a comprehensive, well-structured
research report based on triangulated findings from multiple expert perspectives.

Research query: {query}

== REPORT OUTLINE ==
{outline_text}

== TRIANGULATED FINDINGS ==

Verified claims (with confidence levels):
{verified_claims_text}

Resolved disagreements:
{resolved_text}

Unresolved disagreements:
{unresolved_text}

Novel insights from sources:
{novel_text}

Evidence gaps:
{gaps_text}

== EXPERT PERSPECTIVES CONSULTED ==
{perspectives_text}

== SOURCE EVIDENCE ==
{evidence_text}

Write a comprehensive research report following the outline above. Requirements:

1. **Structure**: Use the outline sections, but adapt as needed for flow and completeness.
   Start with a title (# Title) and abstract (## Abstract).

2. **Multi-perspective integration**: Weave insights from different expert perspectives
   throughout. When perspectives agree, present as established understanding.
   When they disagree, present the debate fairly and note which position has stronger evidence.

3. **Evidence-based**: Ground all claims in the triangulated evidence. Use inline citations
   [1], [2], etc. corresponding to the numbered sources.

4. **Confidence signaling**: Indicate when claims are well-established vs. tentative.
   Use language like "strong evidence suggests" vs. "preliminary findings indicate."

5. **Nuance**: Acknowledge limitations, caveats, and areas where evidence is thin.
   Discuss unresolved debates and evidence gaps honestly.

6. **Specificity**: Include concrete data, methods, statistics, dates, and examples
   wherever the evidence provides them.

7. **Length**: Aim for 2000-4000 words for a thorough report.

8. **Conclusion**: End with a synthesis of key findings and identified directions for
   future research.

Write the report in markdown format.
"""


def _format_outline(outline: Dict[str, Any]) -> str:
    """Format the report outline for the synthesis prompt."""
    parts = [f"Title: {outline.get('title', 'Research Report')}"]

    abstract_points = outline.get("abstract_points", [])
    if abstract_points:
        parts.append("\nAbstract should cover:")
        for pt in abstract_points:
            parts.append(f"  - {pt}")

    for sec in outline.get("sections", []):
        parts.append(f"\n### {sec.get('title', 'Section')}")
        for pt in sec.get("key_points", []):
            parts.append(f"  - {pt}")
        perspectives = sec.get("perspectives_to_cite", [])
        if perspectives:
            parts.append(f"  Draw from: {', '.join(perspectives)}")
        notes = sec.get("notes", "")
        if notes:
            parts.append(f"  Note: {notes}")

    return "\n".join(parts)


def _format_verified_claims(triangulation: Dict[str, Any]) -> str:
    """Format verified claims from triangulation."""
    parts = []
    for claim in triangulation.get("verified_claims", []):
        verdict = claim.get("verdict", "unknown")
        strength = claim.get("evidence_strength", "unknown")
        confidence = claim.get("confidence_score", 0.5)
        supporters = ", ".join(claim.get("perspectives_supporting", []))
        n_sources = claim.get("num_independent_sources", 0)
        caveats = claim.get("caveats", "")

        parts.append(
            f"- [{verdict}|{strength}|conf={confidence:.2f}] {claim.get('claim', '')}\n"
            f"  Perspectives: {supporters} | Independent sources: {n_sources}\n"
            f"  {f'Caveats: {caveats}' if caveats else ''}"
        )
    return "\n".join(parts) if parts else "No verified claims available."


def _format_resolved(triangulation: Dict[str, Any]) -> str:
    """Format resolved disagreements."""
    parts = []
    for item in triangulation.get("resolved_disagreements", []):
        parts.append(
            f"- {item.get('topic', '')}: {item.get('resolution', '')}\n"
            f"  Winner: {item.get('winning_position', '')}\n"
            f"  Basis: {item.get('evidence_basis', '')}"
        )
    return "\n".join(parts) if parts else "No disagreements resolved."


def _format_unresolved(triangulation: Dict[str, Any]) -> str:
    """Format unresolved disagreements."""
    parts = []
    for item in triangulation.get("unresolved_disagreements", []):
        perspectives = ", ".join(item.get("perspectives", []))
        parts.append(
            f"- {item.get('topic', '')}: {item.get('reason', '')}\n"
            f"  Perspectives: {perspectives}"
        )
    return "\n".join(parts) if parts else "All disagreements resolved."


def _format_novel(triangulation: Dict[str, Any]) -> str:
    """Format novel insights."""
    parts = []
    for item in triangulation.get("novel_insights", []):
        parts.append(
            f"- {item.get('insight', '')}\n"
            f"  Source: {item.get('source', '')} | Relevance: {item.get('relevance', '')}"
        )
    return "\n".join(parts) if parts else "No novel insights identified."


def _format_gaps(triangulation: Dict[str, Any]) -> str:
    """Format evidence gaps."""
    parts = []
    for item in triangulation.get("evidence_gaps", []):
        parts.append(
            f"- [{item.get('importance', 'unknown')}] {item.get('topic', '')}\n"
            f"  Suggestion: {item.get('suggestion', '')}"
        )
    return "\n".join(parts) if parts else "No significant evidence gaps."


async def synthesize(
    query: str,
    outline: Dict[str, Any],
    triangulation: Dict[str, Any],
    perspectives: List[Perspective],
    evidence_extractions: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> str:
    """Synthesize all findings into a markdown research report.

    Args:
        query: The research query.
        outline: The report outline from mind mapping.
        triangulation: Triangulated and verified findings.
        perspectives: All expert perspectives consulted.
        evidence_extractions: All source extractions (SourceExtraction objects
            with structured fields including key_findings, data_points, etc.).
        llm: LLM caller instance.
        model: Model to use for synthesis (should be high-capability).

    Returns:
        The complete research report as markdown text.
    """
    outline_text = _format_outline(outline)
    verified_claims_text = _format_verified_claims(triangulation)
    resolved_text = _format_resolved(triangulation)
    unresolved_text = _format_unresolved(triangulation)
    novel_text = _format_novel(triangulation)
    gaps_text = _format_gaps(triangulation)
    perspectives_text = "\n".join(
        f"- {p.name}: {p.description}" for p in perspectives
    )
    evidence_text = (
        format_extractions_as_evidence(evidence_extractions)
        if evidence_extractions
        else "No source material available."
    )

    log.info("synthesizing_report",
             n_verified=len(triangulation.get("verified_claims", [])),
             n_evidence=len(evidence_extractions),
             n_perspectives=len(perspectives))

    report_md = await llm.complete(
        SYNTHESIS_PROMPT.format(
            query=query,
            outline_text=outline_text,
            verified_claims_text=verified_claims_text,
            resolved_text=resolved_text,
            unresolved_text=unresolved_text,
            novel_text=novel_text,
            gaps_text=gaps_text,
            perspectives_text=perspectives_text,
            evidence_text=evidence_text,
        ),
        model=model,
        max_tokens=8192,
        temperature=0.3,
    )

    log.info("synthesis_complete", report_length=len(report_md))
    return report_md


def assemble_report(
    query: str,
    markdown: str,
    evidence_extractions: List[SourceExtraction],
    perspectives: List[Perspective],
    triangulation: Dict[str, Any],
    cost_usd: float = 0.0,
    total_tokens: int = 0,
    elapsed_seconds: float = 0.0,
) -> ResearchReport:
    """Parse the synthesized markdown into a structured ResearchReport.

    Args:
        query: The research query.
        markdown: The synthesized markdown report.
        evidence_extractions: Source extractions used as evidence.
        perspectives: Expert perspectives consulted.
        triangulation: Triangulation results.
        cost_usd: Total cost for the run.
        total_tokens: Total tokens used.
        elapsed_seconds: Wall-clock time for the run.

    Returns:
        A structured ResearchReport object.
    """
    # Build P4-specific metadata
    verified_claims = triangulation.get("verified_claims", [])
    avg_confidence = 0.0
    if verified_claims:
        avg_confidence = sum(
            c.get("confidence_score", 0.5) for c in verified_claims
        ) / len(verified_claims)

    metadata = {
        "perspectives": [p.model_dump() for p in perspectives],
        "n_perspectives": len(perspectives),
        "n_verified_claims": len(verified_claims),
        "avg_claim_confidence": round(avg_confidence, 3),
        "n_resolved_disagreements": len(triangulation.get("resolved_disagreements", [])),
        "n_unresolved_disagreements": len(triangulation.get("unresolved_disagreements", [])),
        "n_novel_insights": len(triangulation.get("novel_insights", [])),
        "n_evidence_gaps": len(triangulation.get("evidence_gaps", [])),
        "n_source_extractions": len(evidence_extractions),
    }

    return parse_markdown_report(
        query=query,
        markdown=markdown,
        extractions=evidence_extractions,
        pattern_name="p4_perspective_storm",
        cost_usd=cost_usd,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
        metadata=metadata,
        max_citations=30,
    )
