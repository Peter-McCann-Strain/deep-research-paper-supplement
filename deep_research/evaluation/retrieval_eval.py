"""Retrieval-generation separation evaluation.

Evaluates retrieval quality and synthesis quality independently,
following DeepResearchBench's RACE+FACT model and SurGE's three-level
citation accuracy.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger()


# ── Academic domain patterns ─────────────────────────────────────────────────

_ACADEMIC_DOMAINS = frozenset({
    "arxiv.org",
    "scholar.google.com",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "doi.org",
    "dx.doi.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "acm.org",
    "dl.acm.org",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "plos.org",
    "wiley.com",
    "jstor.org",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "researchgate.net",
    "openreview.net",
    "aclanthology.org",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
})

_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")
_SECTION_PATTERN = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    """Metrics for the retrieval component of a pattern run."""

    total_sources_retrieved: int
    unique_urls: int
    urls_with_full_content: int
    academic_sources: int
    web_sources: int
    source_diversity: float  # Shannon entropy of domain distribution
    avg_content_length: int  # characters per source
    median_content_length: int
    domain_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class SynthesisMetrics:
    """Metrics for the synthesis/generation component."""

    total_sections: int
    total_words: int
    total_claims: int  # from citation verifier
    attributed_claims: int
    attribution_rate: float
    unique_sources_cited: int
    citation_density: float  # citations per 1000 words
    has_abstract: bool
    has_conclusion: bool
    avg_section_length: int  # words per section


@dataclass
class ThreeLevelCitationResult:
    """SurGE three-level citation accuracy.

    Doc-Acc: Is the cited source thematically relevant to the report topic?
    Sec-Acc: Is the citation in an appropriate section?
    Sent-Acc: Does the source support the specific sentence's claim?
    """

    doc_accuracy: float
    sec_accuracy: float
    sent_accuracy: float
    n_citations_evaluated: int


@dataclass
class RetrievalGenerationReport:
    """Combined retrieval + generation evaluation."""

    pattern: str
    query_id: str
    retrieval: RetrievalMetrics
    synthesis: SynthesisMetrics
    three_level: ThreeLevelCitationResult | None = None


# ── Pure Python metrics ──────────────────────────────────────────────────────


def extract_domain(url: str) -> str:
    """Extract domain from URL for diversity analysis.

    Handles various URL formats including missing schemes.
    Returns the hostname or 'unknown' for invalid URLs.
    """
    if not url or not url.strip():
        return "unknown"

    url = url.strip()

    # Add scheme if missing
    if not url.startswith(("http://", "https://", "//")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return "unknown"
        # Remove www. prefix for cleaner grouping
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()
    except Exception:
        return "unknown"


def compute_source_diversity(domains: list[str]) -> float:
    """Shannon entropy of domain distribution.

    Higher entropy = more diverse sources.
    H = -sum(p_i * log2(p_i))

    Returns 0.0 for empty or single-domain lists.
    """
    if not domains or len(domains) <= 1:
        return 0.0

    counts = Counter(domains)
    total = len(domains)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_citation_density(report_text: str) -> float:
    """Citations per 1000 words.

    Counts unique citation markers (e.g., [1], [2]) in the text
    and divides by word count / 1000.
    """
    if not report_text or not report_text.strip():
        return 0.0

    # Count citation markers (all occurrences, not just unique)
    markers = _CITATION_MARKER_PATTERN.findall(report_text)
    citation_count = len(markers)

    words = report_text.split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    return (citation_count / word_count) * 1000


def _is_academic_url(url: str) -> bool:
    """Check if a URL belongs to an academic domain."""
    domain = extract_domain(url)
    if domain == "unknown":
        return False
    # Check exact match or suffix match (e.g., "papers.nips.cc")
    for acad in _ACADEMIC_DOMAINS:
        if domain == acad or domain.endswith("." + acad):
            return True
    return False


def _extract_sections(report_text: str) -> list[dict[str, str]]:
    """Extract sections from markdown report text.

    Returns list of dicts with 'title' and 'content' keys.
    """
    sections: list[dict[str, str]] = []
    matches = list(_SECTION_PATTERN.finditer(report_text))

    if not matches:
        # No markdown headers found -- treat entire text as one section
        return [{"title": "Body", "content": report_text}]

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report_text)
        content = report_text[start:end].strip()
        sections.append({"title": title, "content": content})

    return sections


def compute_retrieval_metrics(
    sources: list,  # list[SourceExtraction] or list[Document] or list of dicts
) -> RetrievalMetrics:
    """Compute retrieval quality metrics from sources.

    Extracts URL domains, computes diversity entropy, categorizes
    academic vs web sources.

    Sources can be SourceExtraction, Document, or dicts with 'url' and 'content' keys.
    """
    if not sources:
        return RetrievalMetrics(
            total_sources_retrieved=0,
            unique_urls=0,
            urls_with_full_content=0,
            academic_sources=0,
            web_sources=0,
            source_diversity=0.0,
            avg_content_length=0,
            median_content_length=0,
            domain_distribution={},
        )

    urls: list[str] = []
    content_lengths: list[int] = []
    urls_with_content = 0
    academic = 0
    web = 0

    for source in sources:
        # Extract URL -- handle various source types
        url = ""
        content = ""

        if isinstance(source, dict):
            url = source.get("url", "")
            content = source.get("content", "") or source.get("summary", "")
        elif hasattr(source, "url"):
            url = getattr(source, "url", "")
            content = getattr(source, "content", "") or getattr(source, "summary", "")
        else:
            continue

        urls.append(url)

        content_len = len(content) if content else 0
        content_lengths.append(content_len)

        if content_len > 50:
            urls_with_content += 1

        if _is_academic_url(url):
            academic += 1
        else:
            web += 1

    # Compute domain distribution
    domains = [extract_domain(u) for u in urls]
    domain_counts = dict(Counter(domains))

    # Compute diversity
    diversity = compute_source_diversity(domains)

    # Content length stats
    unique_url_set = set(u for u in urls if u)
    sorted_lengths = sorted(content_lengths) if content_lengths else [0]
    avg_length = int(sum(content_lengths) / len(content_lengths)) if content_lengths else 0
    mid = len(sorted_lengths) // 2
    if len(sorted_lengths) % 2 == 0 and len(sorted_lengths) > 1:
        median_length = int((sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2)
    else:
        median_length = sorted_lengths[mid]

    return RetrievalMetrics(
        total_sources_retrieved=len(sources),
        unique_urls=len(unique_url_set),
        urls_with_full_content=urls_with_content,
        academic_sources=academic,
        web_sources=web,
        source_diversity=diversity,
        avg_content_length=avg_length,
        median_content_length=median_length,
        domain_distribution=domain_counts,
    )


def compute_synthesis_metrics(
    report_text: str,
    citations: list | None = None,
    claim_count: int = 0,
    attributed_count: int = 0,
) -> SynthesisMetrics:
    """Compute synthesis quality metrics from report text.

    Args:
        report_text: The full markdown report text.
        citations: list of Citation objects (for unique sources count).
        claim_count: Total atomic claims (from citation verifier).
        attributed_count: Claims with citations (from citation verifier).
    """
    if not report_text or not report_text.strip():
        return SynthesisMetrics(
            total_sections=0,
            total_words=0,
            total_claims=claim_count,
            attributed_claims=attributed_count,
            attribution_rate=0.0,
            unique_sources_cited=0,
            citation_density=0.0,
            has_abstract=False,
            has_conclusion=False,
            avg_section_length=0,
        )

    citations = citations or []

    # Count words
    words = report_text.split()
    total_words = len(words)

    # Extract sections
    sections = _extract_sections(report_text)
    total_sections = len(sections)

    # Section word lengths
    section_word_lengths = [len(s["content"].split()) for s in sections]
    avg_section_length = (
        int(sum(section_word_lengths) / len(section_word_lengths))
        if section_word_lengths
        else 0
    )

    # Check for abstract and conclusion
    text_lower = report_text.lower()
    has_abstract = bool(re.search(r"#+\s*abstract", text_lower))
    has_conclusion = bool(
        re.search(r"#+\s*(conclusion|summary|closing remarks)", text_lower)
    )

    # Unique sources cited
    unique_source_urls: set[str] = set()
    for cit in citations:
        url = ""
        if isinstance(cit, dict):
            url = cit.get("source_url", "")
        elif hasattr(cit, "source_url"):
            url = getattr(cit, "source_url", "")
        if url:
            unique_source_urls.add(url)

    # Citation density
    density = compute_citation_density(report_text)

    # Attribution rate
    attribution_rate = (
        attributed_count / claim_count if claim_count > 0 else 0.0
    )

    return SynthesisMetrics(
        total_sections=total_sections,
        total_words=total_words,
        total_claims=claim_count,
        attributed_claims=attributed_count,
        attribution_rate=attribution_rate,
        unique_sources_cited=len(unique_source_urls),
        citation_density=density,
        has_abstract=has_abstract,
        has_conclusion=has_conclusion,
        avg_section_length=avg_section_length,
    )


# ── Three-level citation accuracy (requires LLM) ────────────────────────────

_THREE_LEVEL_DOC_PROMPT = """Is the following source thematically relevant to the research report topic?

Report topic: {report_topic}
Source title: {source_title}
Source summary (first 500 chars): {source_summary}

Return JSON:
{{
    "relevant": true,
    "reasoning": "Brief explanation"
}}

Only set "relevant" to false if the source has no thematic connection to the report topic."""

_THREE_LEVEL_SEC_PROMPT = """Is this citation placed in an appropriate section of the report?

Section title: {section_title}
Section content (excerpt, 200 chars): {section_excerpt}
Cited source title: {source_title}
Cited source summary (first 300 chars): {source_summary}

Return JSON:
{{
    "appropriate": true,
    "reasoning": "Brief explanation"
}}

Only set "appropriate" to false if the source is clearly misplaced in this section."""

_THREE_LEVEL_SENT_PROMPT = """Does the cited source support the specific claim in the sentence?

Sentence containing citation: {sentence}
Source title: {source_title}
Source content (excerpt): {source_content}

Return JSON:
{{
    "supports": true,
    "reasoning": "Brief explanation"
}}

Set "supports" to false if the source does not entail or confirm the claim in the sentence."""


async def three_level_citation_accuracy(
    report_text: str,
    citations: list,  # list[Citation]
    source_extractions: list,  # list[SourceExtraction]
    llm_caller,
    max_citations: int = 20,
) -> ThreeLevelCitationResult:
    """SurGE three-level citation accuracy evaluation.

    For each citation in the report:
    1. Doc-Acc: Is the cited source thematically relevant? (LLM check)
    2. Sec-Acc: Is the citation placed in a topically appropriate section? (LLM check)
    3. Sent-Acc: Does the cited source support the specific claim? (NLI check)

    Args:
        report_text: Full markdown report text.
        citations: List of Citation objects.
        source_extractions: List of SourceExtraction objects.
        llm_caller: LLMCaller instance for LLM checks.
        max_citations: Maximum number of citations to evaluate.

    Returns:
        ThreeLevelCitationResult with doc/sec/sent accuracy scores.
    """
    if not citations or not report_text:
        return ThreeLevelCitationResult(
            doc_accuracy=0.0,
            sec_accuracy=0.0,
            sent_accuracy=0.0,
            n_citations_evaluated=0,
        )

    # Build source lookup: by URL, doc_id, title
    source_map: dict[str, Any] = {}
    for ext in source_extractions:
        url = getattr(ext, "url", "")
        doc_id = getattr(ext, "doc_id", "")
        title = getattr(ext, "title", "")
        if url:
            source_map[url] = ext
        if doc_id:
            source_map[doc_id] = ext
        if title:
            source_map[title] = ext

    # Extract report topic from first line or title
    first_line = report_text.strip().split("\n")[0]
    report_topic = re.sub(r"^#+\s*", "", first_line).strip() or "Unknown topic"

    # Extract sections with their content
    sections = _extract_sections(report_text)

    # Find citation markers in sections and map to sentences
    citation_contexts: list[dict[str, Any]] = []
    for i, cit in enumerate(citations):
        if i >= max_citations:
            break

        ref_num = i + 1
        marker = f"[{ref_num}]"

        # Find which section contains this citation
        section_title = ""
        section_content = ""
        sentence = ""

        for sec in sections:
            if marker in sec["content"]:
                section_title = sec["title"]
                section_content = sec["content"]
                # Extract sentence containing the marker
                for sent in re.split(r"(?<=[.!?])\s+", sec["content"]):
                    if marker in sent:
                        sentence = sent
                        break
                break

        # Find matching source extraction
        source = None
        source_url = getattr(cit, "source_url", "") if hasattr(cit, "source_url") else ""
        source_id = getattr(cit, "source_id", "") if hasattr(cit, "source_id") else ""
        source_title = getattr(cit, "source_title", "") if hasattr(cit, "source_title") else ""

        source = source_map.get(source_url) or source_map.get(source_id) or source_map.get(source_title)

        citation_contexts.append({
            "citation": cit,
            "ref_num": ref_num,
            "section_title": section_title,
            "section_content": section_content,
            "sentence": sentence,
            "source": source,
            "source_title": source_title,
        })

    if not citation_contexts:
        return ThreeLevelCitationResult(
            doc_accuracy=0.0,
            sec_accuracy=0.0,
            sent_accuracy=0.0,
            n_citations_evaluated=0,
        )

    # Evaluate each citation at all three levels
    doc_correct = 0
    sec_correct = 0
    sent_correct = 0
    evaluated = 0

    sem = asyncio.Semaphore(5)

    async def _evaluate_one(ctx: dict) -> tuple[bool, bool, bool]:
        async with sem:
            source = ctx["source"]
            s_title = ctx["source_title"]
            s_summary = ""
            s_content = ""

            if source:
                s_summary = getattr(source, "summary", "")[:500]
                s_content = getattr(source, "summary", "") or getattr(source, "content", "")
                s_title = getattr(source, "title", s_title)

            doc_ok = False
            sec_ok = False
            sent_ok = False

            # Level 1: Doc-Acc
            try:
                result = await llm_caller.complete_json(
                    _THREE_LEVEL_DOC_PROMPT.format(
                        report_topic=report_topic,
                        source_title=s_title,
                        source_summary=s_summary[:500] if s_summary else "No summary available",
                    ),
                    temperature=0.1,
                    max_tokens=512,
                )
                doc_ok = bool(result.get("relevant", False))
            except Exception as e:
                logger.warning("three_level_doc_error", error=str(e))

            # Level 2: Sec-Acc
            if ctx["section_title"] and s_summary:
                try:
                    result = await llm_caller.complete_json(
                        _THREE_LEVEL_SEC_PROMPT.format(
                            section_title=ctx["section_title"],
                            section_excerpt=ctx["section_content"][:200],
                            source_title=s_title,
                            source_summary=s_summary[:300],
                        ),
                        temperature=0.1,
                        max_tokens=512,
                    )
                    sec_ok = bool(result.get("appropriate", False))
                except Exception as e:
                    logger.warning("three_level_sec_error", error=str(e))

            # Level 3: Sent-Acc
            if ctx["sentence"] and s_content:
                try:
                    result = await llm_caller.complete_json(
                        _THREE_LEVEL_SENT_PROMPT.format(
                            sentence=ctx["sentence"],
                            source_title=s_title,
                            source_content=s_content[:2000],
                        ),
                        temperature=0.1,
                        max_tokens=512,
                    )
                    sent_ok = bool(result.get("supports", False))
                except Exception as e:
                    logger.warning("three_level_sent_error", error=str(e))

            return (doc_ok, sec_ok, sent_ok)

    results = await asyncio.gather(
        *[_evaluate_one(ctx) for ctx in citation_contexts],
        return_exceptions=True,
    )

    for r in results:
        if isinstance(r, tuple):
            d, s, st = r
            evaluated += 1
            if d:
                doc_correct += 1
            if s:
                sec_correct += 1
            if st:
                sent_correct += 1
        elif isinstance(r, Exception):
            logger.warning("three_level_eval_error", error=str(r))

    doc_acc = doc_correct / evaluated if evaluated > 0 else 0.0
    sec_acc = sec_correct / evaluated if evaluated > 0 else 0.0
    sent_acc = sent_correct / evaluated if evaluated > 0 else 0.0

    logger.info(
        "three_level_citation_accuracy",
        doc_acc=f"{doc_acc:.2f}",
        sec_acc=f"{sec_acc:.2f}",
        sent_acc=f"{sent_acc:.2f}",
        evaluated=evaluated,
    )

    return ThreeLevelCitationResult(
        doc_accuracy=doc_acc,
        sec_accuracy=sec_acc,
        sent_accuracy=sent_acc,
        n_citations_evaluated=evaluated,
    )
