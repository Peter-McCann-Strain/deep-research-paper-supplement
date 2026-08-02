"""Compressor: compresses aggregated findings into coherent sections for report writing."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import SourceExtraction, format_extractions_as_evidence
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# -- Prompts -------------------------------------------------------------------

COMPRESS_PROMPT = """You are a research synthesiser. Compress the following worker summaries and
source evidence into a coherent set of report sections.

Original query: {query}

Worker summaries:
{summaries}

Source evidence (numbered):
{evidence}

Requirements:
- Organise the information into {n_sections} logical sections.
- Merge overlapping content; remove pure duplication.
- Keep all specific facts: numbers, dates, author names, method names, results.
- Preserve attribution: note which sources support which claims using [source_N] markers.
- Each section should be 300-600 words of dense, factual prose.
- Suggest a short, descriptive title for each section.

Return JSON:
{{
  "sections": [
    {{
      "title": "section title",
      "content": "compressed section content with [source_N] citations",
      "key_sources": ["list of important source URLs or titles used"]
    }}
  ]
}}
"""

ABSTRACT_PROMPT = """Based on the following section summaries, write a concise abstract
(150-250 words) for a research report on this topic.

Query: {query}

Sections:
{section_summaries}

Write the abstract directly (no JSON wrapper).
"""


async def compress_findings(
    query: str,
    worker_summaries: Dict[str, str],
    source_summaries: List[SourceExtraction],
    llm: LLMCaller,
    n_sections: int = 5,
    model: str = DEFAULT_MODEL,
) -> CompressedOutput:
    """Compress worker summaries and source evidence into coherent report sections.

    Args:
        query: original research query.
        worker_summaries: dict mapping sub-topic -> narrative summary text.
        source_summaries: list of SourceExtraction objects from SourceExtractor.
        llm: LLMCaller instance.
        n_sections: target number of sections.
        model: model to use for compression (cost-efficient).

    Returns:
        CompressedOutput with sections and metadata.
    """
    summaries_str = ""
    for topic, summary in worker_summaries.items():
        summaries_str += f"\n### {topic}\n{summary}\n"

    evidence_str = format_extractions_as_evidence(source_summaries[:50])

    result = await llm.complete_json(
        COMPRESS_PROMPT.format(
            query=query,
            summaries=summaries_str,
            evidence=evidence_str,
            n_sections=n_sections,
        ),
        model=model,
        temperature=0.3,
        max_tokens=8192,
    )

    sections = []
    for s in result.get("sections", []):
        sections.append({
            "title": s.get("title", "Untitled Section"),
            "content": s.get("content", ""),
            "key_sources": s.get("key_sources", []),
        })

    log.info("compressor_done", sections=len(sections),
             total_chars=sum(len(s["content"]) for s in sections))

    return CompressedOutput(sections=sections)


async def generate_abstract(
    query: str,
    sections: List[Dict[str, Any]],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a concise abstract from compressed sections.

    Args:
        query: original research query.
        sections: list of compressed section dicts.
        llm: LLMCaller instance.
        model: model for abstract generation.

    Returns:
        Abstract text.
    """
    section_summaries = "\n".join(
        f"## {s['title']}\n{s['content'][:2000]}..."
        for s in sections
    )

    abstract = await llm.complete(
        ABSTRACT_PROMPT.format(query=query, section_summaries=section_summaries),
        model=model,
        temperature=0.3,
        max_tokens=1024,
    )

    log.info("abstract_generated", length=len(abstract))
    return abstract.strip()


class CompressedOutput:
    """Container for compressed findings ready for report writing."""

    def __init__(self, sections: List[Dict[str, Any]]):
        self.sections = sections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_count": len(self.sections),
            "sections": [
                {"title": s["title"], "content_length": len(s["content"])}
                for s in self.sections
            ],
        }

    def sections_as_text(self) -> str:
        """Format sections as markdown text for downstream use."""
        parts = []
        for s in self.sections:
            parts.append(f"## {s['title']}\n{s['content']}")
        return "\n\n".join(parts)
