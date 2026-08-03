"""Two-step source extraction — free-text analysis then lightweight structured extraction.

Based on research findings:
- PaperQA2 achieves superhuman performance with just summary + relevance_score
- Forcing JSON output degrades LLM reasoning (Tam et al. 2024)
- Two-step approach (SLOT framework) preserves reasoning quality

Step 1: LLM reads source freely, writes analysis with no format constraints
Step 2: LLM converts that analysis into minimal structured JSON
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import Document
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

MAX_CONTENT_CHARS = 50_000

# Chunk size for local models with limited VRAM.
# 15K chars ≈ 4K tokens → ~6 GB VRAM (safe on 16 GB GPU with 4-bit model).
# Pages longer than this are split into overlapping chunks, each analyzed
# separately in Step 1, then combined for Step 2 (no information loss).
LOCAL_CHUNK_CHARS = 15_000
LOCAL_CHUNK_OVERLAP = 500


# ── Schema ────────────────────────────────────────────────────────────────────


class ExtractedSourceType(str, Enum):
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    DOCUMENTATION = "documentation"
    OPINION_PIECE = "opinion_piece"
    FORUM_DISCUSSION = "forum_discussion"
    OFFICIAL_REPORT = "official_report"
    OTHER = "other"


class SourceExtraction(BaseModel):
    """Structured extraction from a single source document."""

    # Identity (carried through from Document)
    doc_id: str = ""
    title: str = ""
    url: str = ""
    original_source_type: str = ""

    # Core fields (always populated)
    summary: str = Field(
        default="",
        description="Query-focused summary, 200-400 words. "
        "Preserves key quotes and data points verbatim.",
    )
    relevance_score: int = Field(
        default=0,
        ge=0,
        le=10,
        description="1-10 relevance to the research query. 0 means not relevant.",
    )
    source_type: ExtractedSourceType = ExtractedSourceType.OTHER
    key_findings: List[str] = Field(
        default_factory=list,
        description="Atomic findings that can be independently cross-referenced.",
    )
    confidence_notes: str = Field(
        default="",
        description="Caveats about reliability, recency, or completeness.",
    )

    # Adaptive fields (populated only when applicable)
    methodology: Optional[str] = None
    data_points: Optional[List[str]] = None
    limitations: Optional[str] = None
    competing_perspectives: Optional[List[str]] = None
    practical_implications: Optional[str] = None
    temporal_context: Optional[str] = None

    def to_evidence_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for evidence formatting."""
        d: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type.value,
            "summary": self.summary,
            "relevance_score": self.relevance_score,
            "key_findings": self.key_findings,
            "confidence_notes": self.confidence_notes,
        }
        if self.methodology:
            d["methodology"] = self.methodology
        if self.data_points:
            d["data_points"] = self.data_points
        if self.limitations:
            d["limitations"] = self.limitations
        if self.competing_perspectives:
            d["competing_perspectives"] = self.competing_perspectives
        if self.practical_implications:
            d["practical_implications"] = self.practical_implications
        if self.temporal_context:
            d["temporal_context"] = self.temporal_context
        return d


# ── Prompts ───────────────────────────────────────────────────────────────────

STEP1_ANALYSIS_PROMPT = """You are a research analyst. Read the following source document carefully and write a thorough analysis of how it relates to the research query.

Include in your analysis:
- All relevant findings, claims, and arguments
- Specific data points, statistics, benchmark results, dates, and names (preserve these verbatim)
- The methodology used (if it's a research study)
- Any stated or obvious limitations
- Your assessment of how reliable and current this source is
- Any competing viewpoints mentioned
- Practical implications or actionable takeaways

Be thorough but focused on what's relevant to the query. If the source has no relevance to the query, simply write "NOT RELEVANT" and nothing else.

Research query: {query}

Source title: {title}
Source URL: {url}
Content:
{content}

Write your analysis:"""

STEP2_EXTRACTION_PROMPT = """Convert the following analysis into structured JSON. Return ONLY valid JSON, no other text.

Analysis:
{analysis}

Source title: {title}
Source URL: {url}

Return this exact JSON structure. Fill all core fields. Only include optional fields if they are clearly present in the analysis — omit them entirely if not applicable.

{{
  "summary": "Query-focused summary of relevant information, 200-400 words. Preserve key quotes and data points verbatim.",
  "relevance_score": <1-10 integer, how relevant to the research query>,
  "source_type": "<one of: research_paper, news_article, blog_post, documentation, opinion_piece, forum_discussion, official_report, other>",
  "key_findings": ["atomic finding 1", "atomic finding 2", "...each finding should be a single verifiable claim or fact"],
  "confidence_notes": "Any caveats about the reliability, recency, or completeness of this source",

  "methodology": "How the research was conducted (ONLY if this is a research study, omit otherwise)",
  "data_points": ["specific statistic or benchmark 1", "..."] ,
  "limitations": "Stated or obvious limitations (ONLY if applicable, omit otherwise)",
  "competing_perspectives": ["views that disagree with this source's findings"],
  "practical_implications": "Actionable takeaways or implementation details (ONLY if applicable)",
  "temporal_context": "When this information was current, any known updates (ONLY if relevant)"
}}

JSON:"""


# ── Extractor ─────────────────────────────────────────────────────────────────


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks at paragraph boundaries.

    Tries to split at double-newlines (paragraph breaks), falling back
    to single newlines, then hard splits at chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to split at a paragraph boundary (double newline)
        split_region = text[end - 500:end + 200]
        para_break = split_region.rfind("\n\n")
        if para_break >= 0:
            end = (end - 500) + para_break + 2
        else:
            # Fall back to single newline
            nl_break = split_region.rfind("\n")
            if nl_break >= 0:
                end = (end - 500) + nl_break + 1

        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)  # overlap for context continuity

    return chunks


def _cleanup_gpu_memory():
    """Free unused GPU memory between extractions."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class SourceExtractor:
    """Two-step source extraction: free-text analysis then structured JSON."""

    def __init__(
        self,
        llm: LLMCaller,
        model: str = DEFAULT_MODEL,
        max_content_per_call: int = MAX_CONTENT_CHARS,
    ):
        self.llm = llm
        self.model = model
        self.max_content_per_call = max_content_per_call

    async def _analyze_chunk(
        self, chunk: str, query: str, title: str, url: str, chunk_idx: int, n_chunks: int,
    ) -> str:
        """Run Step 1 analysis on a single content chunk."""
        chunk_header = ""
        if n_chunks > 1:
            chunk_header = f"[Chunk {chunk_idx + 1} of {n_chunks}]\n"

        analysis = await self.llm.complete(
            STEP1_ANALYSIS_PROMPT.format(
                query=query,
                title=title,
                url=url,
                content=chunk_header + chunk,
            ),
            model=self.model,
            max_tokens=2048,
            temperature=0.1,
        )
        _cleanup_gpu_memory()
        return analysis

    async def extract_one(
        self,
        doc: Document,
        query: str,
    ) -> Optional[SourceExtraction]:
        """Extract structured information from a single source document.

        Step 1: Free-text analysis (no format constraints — preserves reasoning)
               For long content, splits into chunks and analyzes each separately.
        Step 2: Structured JSON extraction from the (combined) analysis
        """
        content = doc.content[:MAX_CONTENT_CHARS]
        if not content.strip():
            return None

        try:
            # ── Step 1: Free-text analysis ────────────────────────────
            # Chunk if content exceeds per-call limit (VRAM-safe for local models)
            if len(content) > self.max_content_per_call:
                chunks = _split_into_chunks(
                    content, self.max_content_per_call, LOCAL_CHUNK_OVERLAP,
                )
                log.debug(
                    "chunked_extraction",
                    title=doc.title[:60],
                    content_len=len(content),
                    n_chunks=len(chunks),
                )

                # Analyze each chunk sequentially (GPU can only do one at a time)
                chunk_analyses = []
                for i, chunk in enumerate(chunks):
                    chunk_analysis = await self._analyze_chunk(
                        chunk, query, doc.title, doc.url, i, len(chunks),
                    )
                    # If any chunk says NOT RELEVANT, still process others
                    if "NOT RELEVANT" not in chunk_analysis.strip().upper()[:50]:
                        chunk_analyses.append(chunk_analysis)

                if not chunk_analyses:
                    log.debug("source_not_relevant", title=doc.title[:60])
                    return None

                # Combine chunk analyses for Step 2
                analysis = "\n\n---\n\n".join(chunk_analyses)
            else:
                analysis = await self.llm.complete(
                    STEP1_ANALYSIS_PROMPT.format(
                        query=query,
                        title=doc.title,
                        url=doc.url,
                        content=content,
                    ),
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.1,
                )
                _cleanup_gpu_memory()

            if "NOT RELEVANT" in analysis.strip().upper()[:50]:
                log.debug("source_not_relevant", title=doc.title[:60])
                return None

            # ── Step 2: Structured extraction ─────────────────────────
            json_str = await self.llm.complete(
                STEP2_EXTRACTION_PROMPT.format(
                    analysis=analysis,
                    title=doc.title,
                    url=doc.url,
                ),
                model=self.model,
                max_tokens=2048,
                temperature=0.0,
            )
            _cleanup_gpu_memory()

            extracted = _parse_extraction_json(json_str)
            if extracted is None:
                # Fallback: use the free-text analysis as summary
                log.warning("json_parse_failed_using_fallback", title=doc.title[:60])
                return SourceExtraction(
                    doc_id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    original_source_type=doc.source_type.value,
                    summary=analysis.strip()[:2000],
                    relevance_score=5,
                    key_findings=[],
                    confidence_notes="Structured extraction failed; raw analysis used.",
                )

            # Attach document identity
            extracted.doc_id = doc.id
            extracted.title = doc.title
            extracted.url = doc.url
            extracted.original_source_type = doc.source_type.value

            return extracted

        except Exception as e:
            log.warning("extract_error", title=doc.title[:60], error=str(e))
            _cleanup_gpu_memory()
            return None

    async def extract_batch(
        self,
        docs: List[Document],
        query: str,
        max_concurrent: int = 5,
        min_relevance: int = 0,
    ) -> List[SourceExtraction]:
        """Extract from multiple documents concurrently.

        Args:
            docs: Source documents to extract from.
            query: The research query for relevance-focused extraction.
            max_concurrent: Max parallel extractions (each does 2 LLM calls).
            min_relevance: Filter out sources scoring below this threshold (0 = keep all).

        Returns:
            List of SourceExtraction objects, sorted by relevance_score descending.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _extract(doc: Document) -> Optional[SourceExtraction]:
            async with sem:
                return await self.extract_one(doc, query)

        results = await asyncio.gather(
            *[_extract(d) for d in docs],
            return_exceptions=True,
        )

        extractions: List[SourceExtraction] = []
        for r in results:
            if isinstance(r, SourceExtraction):
                if r.relevance_score >= min_relevance:
                    extractions.append(r)
            elif isinstance(r, Exception):
                log.warning("extract_batch_error", error=str(r))

        # Sort by relevance (highest first)
        extractions.sort(key=lambda e: e.relevance_score, reverse=True)

        log.info(
            "sources_extracted",
            total=len(docs),
            relevant=len(extractions),
            avg_relevance=(
                sum(e.relevance_score for e in extractions) / len(extractions)
                if extractions
                else 0
            ),
        )
        return extractions


# ── Formatting ────────────────────────────────────────────────────────────────


def format_extractions_as_evidence(
    extractions: List[SourceExtraction],
    max_chars: int = 300_000,
) -> str:
    """Format extractions into numbered evidence blocks for report generation.

    Produces richer evidence than the old format_summaries_as_evidence by
    including key findings, data points, and confidence notes.

    Args:
        extractions: List of SourceExtraction objects, ideally pre-sorted by relevance.
        max_chars: Maximum total characters for the evidence text. Extractions are
                   added in order until the budget is exhausted. This prevents
                   context-length overflows when many full-page sources are used.
    """
    parts = []
    total_chars = 0

    for i, e in enumerate(extractions, 1):
        block = (
            f"[{i}] Source: {e.title}\n"
            f"URL: {e.url}\n"
            f"Type: {e.source_type.value} | Relevance: {e.relevance_score}/10\n"
            f"Summary: {e.summary}\n"
        )

        if e.key_findings:
            findings = "\n".join(f"  - {f}" for f in e.key_findings)
            block += f"Key Findings:\n{findings}\n"

        if e.data_points:
            points = "\n".join(f"  - {d}" for d in e.data_points)
            block += f"Data Points:\n{points}\n"

        if e.methodology:
            block += f"Methodology: {e.methodology}\n"

        if e.limitations:
            block += f"Limitations: {e.limitations}\n"

        if e.competing_perspectives:
            persp = "\n".join(f"  - {p}" for p in e.competing_perspectives)
            block += f"Competing Views:\n{persp}\n"

        if e.confidence_notes:
            block += f"Confidence: {e.confidence_notes}\n"

        if total_chars + len(block) > max_chars:
            log.info("evidence_truncated", included=len(parts), total=len(extractions),
                     chars=total_chars)
            break

        parts.append(block)
        total_chars += len(block)

    return "\n---\n".join(parts)


# Backward-compatible aliases
def format_summaries_as_evidence(summaries: List[dict]) -> str:
    """Legacy wrapper — accepts old-style summary dicts."""
    parts = []
    for i, s in enumerate(summaries, 1):
        parts.append(
            f"[{i}] Source: {s.get('title', 'Unknown')}\n"
            f"URL: {s.get('url', '')}\n"
            f"{s.get('summary', '')}\n"
        )
    return "\n---\n".join(parts)


# ── JSON parsing ──────────────────────────────────────────────────────────────


def _parse_extraction_json(raw: str) -> Optional[SourceExtraction]:
    """Parse LLM JSON output into SourceExtraction, handling common issues."""
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate and coerce source_type
    source_type_raw = data.get("source_type", "other")
    try:
        source_type = ExtractedSourceType(source_type_raw)
    except ValueError:
        source_type = ExtractedSourceType.OTHER

    # Build extraction
    try:
        return SourceExtraction(
            summary=data.get("summary", ""),
            relevance_score=max(0, min(10, int(data.get("relevance_score", 5)))),
            source_type=source_type,
            key_findings=data.get("key_findings", []),
            confidence_notes=data.get("confidence_notes", ""),
            methodology=data.get("methodology"),
            data_points=data.get("data_points"),
            limitations=data.get("limitations"),
            competing_perspectives=data.get("competing_perspectives"),
            practical_implications=data.get("practical_implications"),
            temporal_context=data.get("temporal_context"),
        )
    except Exception:
        return None
