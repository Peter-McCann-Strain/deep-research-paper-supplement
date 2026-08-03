"""Shared markdown-to-ResearchReport parser.

Every pattern that generates a markdown research report needs to parse it into
a structured ``ResearchReport``.  The regex logic was duplicated across P0, P1,
P2, P4, and P5 with only minor variations.  This module consolidates all of
them into a single, tested implementation.

Supported variations (observed across patterns):
    - Title extracted from ``# Title`` heading (falls back to *query*).
    - Abstract extracted from ``## Abstract`` section (case-insensitive).
    - Body sections extracted from ``## Heading`` markers.
    - Configurable set of section titles to skip (default: Abstract,
      References, Sources).
    - Citation building from ``SourceExtraction`` objects with 1-based
      numbering and ``relevance_score / 10`` normalisation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from deep_research.tools.source_extractor import SourceExtraction
from deep_research.types import Citation, ResearchReport, Section


# ── Configurable defaults ────────────────────────────────────────────────────

_DEFAULT_SKIP_SECTIONS: Set[str] = {"abstract", "references", "sources"}

# ── Title extraction ─────────────────────────────────────────────────────────

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _extract_title(markdown: str) -> str | None:
    """Extract the title from the first ``# Heading`` in *markdown*.

    Returns ``None`` when no heading-1 is found so the caller can decide on
    a fallback (typically the original query).
    """
    match = _TITLE_RE.search(markdown)
    if match:
        return match.group(1).strip()
    return None


# ── Abstract extraction ──────────────────────────────────────────────────────

_ABSTRACT_RE = re.compile(
    r"##\s*Abstract\s*\n(.*?)(?=\n##(?!#)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_abstract(markdown: str) -> str:
    """Extract the content of the ``## Abstract`` section.

    Returns an empty string when no abstract section is found.
    """
    match = _ABSTRACT_RE.search(markdown)
    if match:
        return match.group(1).strip()
    return ""


# ── Section extraction ───────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"##\s+(.+?)\n(.*?)(?=\n##(?!#)|\Z)", re.DOTALL)


def _extract_sections(
    markdown: str,
    skip: Set[str] | None = None,
) -> List[Tuple[str, str]]:
    """Extract ``## Section Title`` blocks as ``(title, content)`` pairs.

    Parameters
    ----------
    markdown:
        The full markdown text.
    skip:
        Lowercase section titles to exclude (e.g. ``{"abstract",
        "references"}``).  Defaults to :data:`_DEFAULT_SKIP_SECTIONS`.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of ``(title, content)`` with leading/trailing whitespace
        stripped.  Subsection headings (``###``, ``####``, ...) within
        a section's content are preserved.
    """
    if skip is None:
        skip = _DEFAULT_SKIP_SECTIONS

    sections: List[Tuple[str, str]] = []
    for match in _SECTION_RE.finditer(markdown):
        sec_title = match.group(1).strip()
        sec_content = match.group(2).strip()
        if sec_title.lower() in skip:
            continue
        sections.append((sec_title, sec_content))
    return sections


# ── Citation building ────────────────────────────────────────────────────────


def build_citations_from_extractions(
    extractions: List[SourceExtraction],
    *,
    max_citations: int | None = None,
) -> List[Citation]:
    """Convert :class:`SourceExtraction` objects into :class:`Citation` objects.

    Each extraction is assigned a 1-based index (``[1]``, ``[2]``, ...) and
    its ``relevance_score`` (0-10 integer) is normalised to a 0-1 float.

    Parameters
    ----------
    extractions:
        The source extractions to convert.
    max_citations:
        Optional cap on the number of citations produced.  ``None`` means
        no limit (all extractions are converted).
    """
    capped = extractions[:max_citations] if max_citations is not None else extractions

    citations: List[Citation] = []
    for i, e in enumerate(capped, 1):
        citations.append(
            Citation(
                claim=f"[{i}]",
                source_id=e.doc_id,
                source_title=e.title,
                source_url=e.url,
                relevance_score=e.relevance_score / 10.0,
            )
        )
    return citations


# ── Main entry point ─────────────────────────────────────────────────────────


def parse_markdown_report(
    query: str,
    markdown: str,
    extractions: List[SourceExtraction],
    pattern_name: str,
    cost_usd: float,
    total_tokens: int,
    elapsed_seconds: float = 0.0,
    metadata: Dict[str, Any] | None = None,
    *,
    skip_sections: Set[str] | None = None,
    max_citations: int | None = None,
) -> ResearchReport:
    """Parse a markdown research report into a structured :class:`ResearchReport`.

    This is the single replacement for the duplicated ``_assemble_report`` /
    ``assemble_report`` / ``assemble_research_report`` functions that existed
    in P0, P1, P2, P4, and P5.

    Parameters
    ----------
    query:
        The original research query.
    markdown:
        The LLM-generated markdown report text.
    extractions:
        Source extractions used as evidence.  Converted to :class:`Citation`
        objects with 1-based numbering.
    pattern_name:
        Pattern identifier (e.g. ``"p0_baseline"``).
    cost_usd:
        Total cost in USD for the run.
    total_tokens:
        Total token count for the run.
    elapsed_seconds:
        Wall-clock time for the run.
    metadata:
        Optional extra metadata to attach to the report.
    skip_sections:
        Section titles (lowercase) to exclude from the body.  Defaults to
        ``{"abstract", "references", "sources"}``.
    max_citations:
        Optional cap on citations.  ``None`` means no limit.

    Returns
    -------
    ResearchReport
        A fully populated report object.
    """
    title = _extract_title(markdown) or query
    abstract = _extract_abstract(markdown)

    raw_sections = _extract_sections(markdown, skip=skip_sections)
    sections = [
        Section(title=sec_title, content=sec_content)
        for sec_title, sec_content in raw_sections
    ]

    citations = build_citations_from_extractions(
        extractions,
        max_citations=max_citations,
    )

    return ResearchReport(
        query=query,
        title=title,
        abstract=abstract,
        sections=sections,
        citations=citations,
        metadata=metadata or {},
        pattern_name=pattern_name,
        total_cost_usd=cost_usd,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
    )
