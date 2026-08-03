"""MERIDIAN Role 3: Research Writer — produces the report from topic clusters of
source extractions.

Uses gpt-5.2 (heavy model) to synthesise a comprehensive, well-cited research
report from the topic clusters and their underlying structured SourceExtraction
objects, leveraging key findings, data points, and confidence notes for richer
evidence presentation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import LLMCaller, SourceExtraction
from deep_research.types import (
    Citation,
    Section,
    TopicCluster,
)

log = structlog.get_logger()

MODEL = DEFAULT_MODEL


# -- Prompt construction -------------------------------------------------------

_WRITER_SYSTEM = (
    "You are an expert research writer. You produce comprehensive, well-structured "
    "research reports that synthesise evidence from multiple sources. Your writing "
    "is clear, precise, and appropriately technical. You always cite your sources "
    "using [N] reference markers. Every factual claim must be grounded in the "
    "provided source material."
)


MAX_EVIDENCE_CHARS = 300_000  # ~75K tokens — fits in 128K context with prompt overhead


def _build_topic_blocks(
    clusters: List[TopicCluster],
    extraction_map: Dict[str, SourceExtraction],
) -> Tuple[str, Dict[str, int]]:
    """Render each topic cluster and its source extractions into a text block.

    Applies a character budget to prevent context-length overflows when many
    full-page sources are available.

    Returns:
        (topic_text, source_registry) where source_registry maps doc_id -> [N] number.
    """
    blocks: List[str] = []
    source_index = 1
    source_registry: Dict[str, int] = {}  # doc_id -> citation number
    total_chars = 0
    truncated = False

    for cl in clusters:
        lines = [f"### Topic: {cl.topic} (importance: {cl.importance:.2f})"]
        lines.append(f"Synthesis: {cl.summary}")
        lines.append("")
        for doc_id in cl.source_ids:
            if truncated:
                break
            ext = extraction_map.get(doc_id)
            if not ext:
                continue
            if doc_id not in source_registry:
                source_registry[doc_id] = source_index
                source_index += 1
            ref = source_registry[doc_id]

            source_lines = []
            source_lines.append(
                f"[Source {ref}] (title: {ext.title} | url: {ext.url} "
                f"| type: {ext.source_type.value} | relevance: {ext.relevance_score}/10)"
            )
            source_lines.append(ext.summary)
            if ext.key_findings:
                source_lines.append("Key Findings:")
                for finding in ext.key_findings:
                    source_lines.append(f"  - {finding}")
            if ext.data_points:
                source_lines.append("Data Points:")
                for dp in ext.data_points:
                    source_lines.append(f"  - {dp}")
            if ext.methodology:
                source_lines.append(f"Methodology: {ext.methodology}")
            if ext.limitations:
                source_lines.append(f"Limitations: {ext.limitations}")
            if ext.competing_perspectives:
                source_lines.append("Competing Perspectives:")
                for persp in ext.competing_perspectives:
                    source_lines.append(f"  - {persp}")
            if ext.confidence_notes:
                source_lines.append(f"Confidence: {ext.confidence_notes}")
            source_lines.append("")

            source_text = "\n".join(source_lines)
            if total_chars + len(source_text) > MAX_EVIDENCE_CHARS:
                log.info("writer_evidence_truncated",
                         sources_included=source_index - 1,
                         chars=total_chars)
                truncated = True
                break

            lines.extend(source_lines)
            total_chars += len(source_text)

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks), source_registry


def _build_writer_prompt(
    query: str,
    clusters: List[TopicCluster],
    extraction_map: Dict[str, SourceExtraction],
) -> Tuple[str, Dict[str, int]]:
    """Build the full prompt for the writer model."""
    topic_text, source_registry = _build_topic_blocks(clusters, extraction_map)

    prompt = f"""\
Write a comprehensive research report answering the following query.

## Query
{query}

## Source Material (organised by topic)
{topic_text}

## Instructions
1. Start with a concise but informative title.
2. Write an abstract (3-5 sentences) summarising the key findings.
3. Organise the body into logical sections with ## headers.
4. Cite sources using [N] markers corresponding to the source numbers above.
5. Include concrete data, statistics, and specific examples where available.
6. Present multiple perspectives and note areas of disagreement or uncertainty.
7. End with a section on implications, future directions, or recommendations.
8. Ensure every factual claim is backed by at least one citation.
9. Aim for 2000-4000 words for the body.

Format the output as JSON:
{{
  "title": "<report title>",
  "abstract": "<abstract text>",
  "sections": [
    {{
      "title": "<section title>",
      "content": "<section content with [N] citations>"
    }},
    ...
  ]
}}
"""
    return prompt, source_registry


# -- Citation extraction -------------------------------------------------------

def _extract_citations(
    sections: List[Section],
    source_registry: Dict[str, int],
    extraction_map: Dict[str, SourceExtraction],
) -> List[Citation]:
    """Parse [N] markers from section content and build Citation objects."""
    # Invert registry: number -> doc_id
    num_to_doc_id: Dict[int, str] = {v: k for k, v in source_registry.items()}

    # Collect all referenced numbers
    all_text = " ".join(s.content for s in sections)
    ref_numbers = set(int(m) for m in re.findall(r"\[(\d+)\]", all_text))

    citations: List[Citation] = []
    for ref_num in sorted(ref_numbers):
        doc_id = num_to_doc_id.get(ref_num)
        if not doc_id:
            continue
        ext = extraction_map.get(doc_id)
        if not ext:
            continue
        citations.append(
            Citation(
                claim=f"[{ref_num}]",
                source_id=doc_id,
                source_title=ext.title,
                source_url=ext.url,
                relevance_score=ext.relevance_score / 10.0,
            )
        )

    return citations


def _link_section_citations(
    section: Section,
    source_registry: Dict[str, int],
    extraction_map: Dict[str, SourceExtraction],
) -> List[Citation]:
    """Build citation objects for a single section."""
    num_to_doc_id: Dict[int, str] = {v: k for k, v in source_registry.items()}
    ref_numbers = set(int(m) for m in re.findall(r"\[(\d+)\]", section.content))
    cites: List[Citation] = []
    for ref_num in sorted(ref_numbers):
        doc_id = num_to_doc_id.get(ref_num)
        if not doc_id:
            continue
        ext = extraction_map.get(doc_id)
        if not ext:
            continue
        cites.append(
            Citation(
                claim=f"[{ref_num}]",
                source_id=doc_id,
                source_title=ext.title,
                source_url=ext.url,
                relevance_score=ext.relevance_score / 10.0,
            )
        )
    return cites


# -- Writing entrypoint -------------------------------------------------------

async def write_report(
    query: str,
    clusters: List[TopicCluster],
    extraction_map: Dict[str, SourceExtraction],
    llm: LLMCaller,
) -> Tuple[str, str, List[Section], List[Citation]]:
    """Generate the research report from topic clusters of source extractions.

    Returns:
        (title, abstract, sections, citations)
    """
    prompt, source_registry = _build_writer_prompt(query, clusters, extraction_map)

    result = await llm.complete_json(
        prompt=prompt,
        model=MODEL,
        system=_WRITER_SYSTEM,
        temperature=0.3,
        max_tokens=8192,
    )

    title = result.get("title", "Research Report")
    abstract = result.get("abstract", "")
    raw_sections = result.get("sections", [])

    sections: List[Section] = []
    for raw in raw_sections:
        sec = Section(
            title=raw.get("title", ""),
            content=raw.get("content", ""),
        )
        sec.citations = _link_section_citations(sec, source_registry, extraction_map)
        sections.append(sec)

    # Aggregate all citations
    all_citations = _extract_citations(sections, source_registry, extraction_map)

    log.info(
        "research_writer.report_generated",
        title=title[:60],
        sections=len(sections),
        citations=len(all_citations),
    )
    return title, abstract, sections, all_citations


# -- Revision entrypoint -------------------------------------------------------

async def revise_report(
    query: str,
    current_title: str,
    current_abstract: str,
    current_sections: List[Section],
    evaluation_feedback: str,
    clusters: List[TopicCluster],
    extraction_map: Dict[str, SourceExtraction],
    llm: LLMCaller,
) -> Tuple[str, str, List[Section], List[Citation]]:
    """Revise a report based on evaluator feedback.

    Returns the same tuple as write_report: (title, abstract, sections, citations).
    """
    from .rubric import build_revision_prompt

    # Reconstruct current report text for context
    report_text_parts = [f"# {current_title}\n", f"## Abstract\n{current_abstract}\n"]
    for sec in current_sections:
        report_text_parts.append(f"## {sec.title}\n{sec.content}\n")
    report_text = "\n".join(report_text_parts)

    # Build topic context so the writer has source material available
    topic_text, source_registry = _build_topic_blocks(clusters, extraction_map)

    revision_prompt = build_revision_prompt(query, report_text, evaluation_feedback)
    revision_prompt += (
        f"\n\n## Available Source Material\n{topic_text}\n\n"
        "Respond in JSON with the same format:\n"
        '{"title": "...", "abstract": "...", "sections": [{"title": "...", "content": "..."}]}'
    )

    result = await llm.complete_json(
        prompt=revision_prompt,
        model=MODEL,
        system=_WRITER_SYSTEM,
        temperature=0.3,
        max_tokens=8192,
    )

    title = result.get("title", current_title)
    abstract = result.get("abstract", current_abstract)
    raw_sections = result.get("sections", [])

    sections: List[Section] = []
    for raw in raw_sections:
        sec = Section(
            title=raw.get("title", ""),
            content=raw.get("content", ""),
        )
        sec.citations = _link_section_citations(sec, source_registry, extraction_map)
        sections.append(sec)

    all_citations = _extract_citations(sections, source_registry, extraction_map)

    log.info(
        "research_writer.report_revised",
        title=title[:60],
        sections=len(sections),
        citations=len(all_citations),
    )
    return title, abstract, sections, all_citations
