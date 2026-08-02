"""Component-level evaluation for research pipeline stages.

Evaluates individual components (search, extraction, synthesis) to identify
where quality breaks down in the pipeline, rather than only evaluating
the final report output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from deep_research.tools import LLMCaller
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()


@dataclass
class SearchEvalResult:
    """Evaluation of the search/retrieval component."""
    query_id: str
    pattern: str
    queries_issued: list[str]
    n_results_retrieved: int
    n_unique_domains: int
    n_academic_sources: int
    diversity_score: float  # 0-1, ratio of unique domains to total results
    relevance_scores: list[float]  # per-result relevance (0-1)
    mean_relevance: float
    coverage_gaps: list[str]  # topics not covered by search results
    metadata: dict = field(default_factory=dict)

    @property
    def retrieval_quality(self) -> float:
        """Composite retrieval quality score (0-1)."""
        if not self.n_results_retrieved:
            return 0.0
        return 0.5 * self.mean_relevance + 0.3 * self.diversity_score + 0.2 * min(1.0, self.n_academic_sources / 3)


@dataclass
class ExtractionEvalResult:
    """Evaluation of the source extraction component."""
    query_id: str
    pattern: str
    n_sources_processed: int
    n_extractions_produced: int
    extraction_rate: float  # extractions / sources
    mean_faithfulness: float  # 0-1, how faithful extractions are to source
    mean_informativeness: float  # 0-1, how much useful info extracted
    hallucination_count: int  # claims in extraction not in source
    metadata: dict = field(default_factory=dict)

    @property
    def extraction_quality(self) -> float:
        """Composite extraction quality score (0-1)."""
        if not self.n_sources_processed:
            return 0.0
        return 0.4 * self.mean_faithfulness + 0.3 * self.mean_informativeness + 0.3 * self.extraction_rate


@dataclass
class SynthesisEvalResult:
    """Evaluation of the synthesis/writing component."""
    query_id: str
    pattern: str
    n_sources_cited: int
    n_sources_available: int
    source_utilization: float  # cited / available
    n_claims_total: int
    n_claims_supported: int  # claims with source backing
    support_rate: float  # supported / total
    coherence_score: float  # 0-1, structural coherence
    metadata: dict = field(default_factory=dict)

    @property
    def synthesis_quality(self) -> float:
        """Composite synthesis quality score (0-1)."""
        return 0.4 * self.support_rate + 0.3 * self.source_utilization + 0.3 * self.coherence_score


@dataclass
class ComponentEvalResult:
    """Complete component-level evaluation for a single run."""
    query_id: str
    pattern: str
    search: SearchEvalResult | None = None
    extraction: ExtractionEvalResult | None = None
    synthesis: SynthesisEvalResult | None = None
    bottleneck: str = ""  # which component is weakest

    def identify_bottleneck(self) -> str:
        """Identify the weakest component in the pipeline."""
        scores = {}
        if self.search:
            scores["search"] = self.search.retrieval_quality
        if self.extraction:
            scores["extraction"] = self.extraction.extraction_quality
        if self.synthesis:
            scores["synthesis"] = self.synthesis.synthesis_quality
        if not scores:
            return "unknown"
        self.bottleneck = min(scores, key=scores.get)
        return self.bottleneck


async def evaluate_search_component(
    query_text: str,
    query_id: str,
    pattern: str,
    search_queries: list[str],
    retrieved_docs: list[dict],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> SearchEvalResult:
    """Evaluate search/retrieval quality.

    Args:
        query_text: The original research query.
        query_id: Query identifier.
        pattern: Pattern name.
        search_queries: The search queries that were issued.
        retrieved_docs: List of dicts with keys: url, title, content, source_type.
        llm: LLM caller for relevance assessment.
        model: Model to use.

    Returns:
        SearchEvalResult with relevance and diversity metrics.
    """
    if not retrieved_docs:
        return SearchEvalResult(
            query_id=query_id, pattern=pattern,
            queries_issued=search_queries,
            n_results_retrieved=0, n_unique_domains=0, n_academic_sources=0,
            diversity_score=0.0, relevance_scores=[], mean_relevance=0.0,
            coverage_gaps=[query_text],
        )

    # Count unique domains
    domains = set()
    academic_count = 0
    for doc in retrieved_docs:
        url = doc.get("url", "")
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                domains.add(domain)
            except Exception:
                pass
        if doc.get("source_type") in ("academic", "arxiv", "semantic_scholar", "pubmed"):
            academic_count += 1

    n_total = len(retrieved_docs)
    diversity = len(domains) / n_total if n_total else 0.0

    # Batch relevance scoring via LLM
    relevance_scores = await _batch_relevance_score(
        query_text, retrieved_docs, llm, model
    )
    mean_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    # Detect coverage gaps
    gaps = await _detect_search_gaps(query_text, retrieved_docs, llm, model)

    return SearchEvalResult(
        query_id=query_id, pattern=pattern,
        queries_issued=search_queries,
        n_results_retrieved=n_total,
        n_unique_domains=len(domains),
        n_academic_sources=academic_count,
        diversity_score=diversity,
        relevance_scores=relevance_scores,
        mean_relevance=mean_rel,
        coverage_gaps=gaps,
    )


async def evaluate_extraction_component(
    query_text: str,
    query_id: str,
    pattern: str,
    sources: list[dict],
    extractions: list[dict],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> ExtractionEvalResult:
    """Evaluate extraction quality by checking faithfulness and informativeness.

    Args:
        query_text: The original research query.
        query_id: Query identifier.
        pattern: Pattern name.
        sources: Original source documents (list of dicts with content).
        extractions: Extracted summaries (list of dicts with summary, key_facts).
        llm: LLM caller for faithfulness assessment.
        model: Model to use.

    Returns:
        ExtractionEvalResult with faithfulness and informativeness metrics.
    """
    n_sources = len(sources)
    n_extractions = len(extractions)

    if not n_sources:
        return ExtractionEvalResult(
            query_id=query_id, pattern=pattern,
            n_sources_processed=0, n_extractions_produced=0,
            extraction_rate=0.0, mean_faithfulness=0.0,
            mean_informativeness=0.0, hallucination_count=0,
        )

    extraction_rate = n_extractions / n_sources if n_sources else 0.0

    # Sample up to 10 extractions for faithfulness check
    sample = extractions[:10]
    source_map = {i: sources[i] if i < len(sources) else {} for i in range(len(sample))}

    faithfulness_scores = []
    informativeness_scores = []
    hallucination_count = 0

    for i, ext in enumerate(sample):
        source = source_map.get(i, {})
        faith, info, halluc = await _check_extraction_quality(
            query_text, source, ext, llm, model
        )
        faithfulness_scores.append(faith)
        informativeness_scores.append(info)
        hallucination_count += halluc

    mean_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    mean_info = sum(informativeness_scores) / len(informativeness_scores) if informativeness_scores else 0.0

    return ExtractionEvalResult(
        query_id=query_id, pattern=pattern,
        n_sources_processed=n_sources,
        n_extractions_produced=n_extractions,
        extraction_rate=extraction_rate,
        mean_faithfulness=mean_faith,
        mean_informativeness=mean_info,
        hallucination_count=hallucination_count,
    )


async def evaluate_synthesis_component(
    query_text: str,
    query_id: str,
    pattern: str,
    report_text: str,
    available_sources: int,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> SynthesisEvalResult:
    """Evaluate synthesis quality of the final report.

    Args:
        query_text: The original research query.
        query_id: Query identifier.
        pattern: Pattern name.
        report_text: Full report text.
        available_sources: Number of sources that were available to the synthesizer.
        llm: LLM caller.
        model: Model to use.

    Returns:
        SynthesisEvalResult with source utilization and claim support metrics.
    """
    import re

    if not report_text:
        return SynthesisEvalResult(
            query_id=query_id, pattern=pattern,
            n_sources_cited=0, n_sources_available=available_sources,
            source_utilization=0.0, n_claims_total=0, n_claims_supported=0,
            support_rate=0.0, coherence_score=0.0,
        )

    # Count cited sources from reference numbers
    citation_refs = set(int(m) for m in re.findall(r"\[(\d+)\]", report_text))
    n_cited = len(citation_refs)
    utilization = n_cited / available_sources if available_sources else 0.0

    # Use LLM to assess claim support and coherence
    synthesis_eval = await _evaluate_synthesis_quality(
        query_text, report_text, llm, model
    )

    n_claims = synthesis_eval.get("n_claims_total", 0)
    n_supported = synthesis_eval.get("n_claims_supported", 0)
    coherence = synthesis_eval.get("coherence_score", 0.5)
    support_rate = n_supported / n_claims if n_claims else 0.0

    return SynthesisEvalResult(
        query_id=query_id, pattern=pattern,
        n_sources_cited=n_cited,
        n_sources_available=available_sources,
        source_utilization=min(1.0, utilization),
        n_claims_total=n_claims,
        n_claims_supported=n_supported,
        support_rate=support_rate,
        coherence_score=coherence,
    )


def aggregate_component_results(
    results: list[ComponentEvalResult],
) -> dict[str, dict[str, float]]:
    """Aggregate component evaluation results by pattern.

    Args:
        results: List of ComponentEvalResult from individual runs.

    Returns:
        Dict mapping pattern name to dict of aggregated metrics.
    """
    by_pattern: dict[str, list[ComponentEvalResult]] = {}
    for r in results:
        by_pattern.setdefault(r.pattern, []).append(r)

    aggregated: dict[str, dict[str, float]] = {}
    for pattern, runs in by_pattern.items():
        search_scores = [r.search.retrieval_quality for r in runs if r.search]
        extract_scores = [r.extraction.extraction_quality for r in runs if r.extraction]
        synth_scores = [r.synthesis.synthesis_quality for r in runs if r.synthesis]

        # Count bottlenecks
        bottlenecks = [r.identify_bottleneck() for r in runs]
        bottleneck_counts = {}
        for b in bottlenecks:
            bottleneck_counts[b] = bottleneck_counts.get(b, 0) + 1

        aggregated[pattern] = {
            "mean_search_quality": sum(search_scores) / len(search_scores) if search_scores else 0.0,
            "mean_extraction_quality": sum(extract_scores) / len(extract_scores) if extract_scores else 0.0,
            "mean_synthesis_quality": sum(synth_scores) / len(synth_scores) if synth_scores else 0.0,
            "n_runs": len(runs),
            "primary_bottleneck": max(bottleneck_counts, key=bottleneck_counts.get) if bottleneck_counts else "unknown",
            "bottleneck_counts": bottleneck_counts,
        }

    return aggregated


# ── Internal helpers ──────────────────────────────────────────────────────

RELEVANCE_PROMPT = """Rate the relevance of this search result to the research query.

Research Query: {query}
Search Result Title: {title}
Search Result Content (first 500 chars): {content}

Return JSON:
{{"relevance": 0.0-1.0, "reasoning": "brief explanation"}}"""


async def _batch_relevance_score(
    query: str,
    docs: list[dict],
    llm: LLMCaller,
    model: str,
) -> list[float]:
    """Score relevance of each document to the query."""
    # Sample at most 15 docs for efficiency
    sample = docs[:15]

    async def _score_one(doc: dict) -> float:
        title = doc.get("title", "")[:200]
        content = doc.get("content", "")[:500]
        try:
            result = await llm.complete_json(
                RELEVANCE_PROMPT.format(query=query, title=title, content=content),
                model=model,
                temperature=0.1,
                max_tokens=256,
            )
            return max(0.0, min(1.0, float(result.get("relevance", 0.5))))
        except Exception:
            return 0.5  # Default on error

    scores = await asyncio.gather(*[_score_one(d) for d in sample])
    return list(scores)


GAP_DETECTION_PROMPT = """Given this research query and the titles of retrieved search results, identify any important subtopics NOT covered by the search results.

Research Query: {query}

Search Result Titles:
{titles}

Return JSON:
{{"gaps": ["subtopic not covered 1", "subtopic not covered 2", ...]}}"""


async def _detect_search_gaps(
    query: str,
    docs: list[dict],
    llm: LLMCaller,
    model: str,
) -> list[str]:
    """Detect gaps in search result coverage."""
    titles = "\n".join(f"- {d.get('title', 'Untitled')}" for d in docs[:20])
    try:
        result = await llm.complete_json(
            GAP_DETECTION_PROMPT.format(query=query, titles=titles),
            model=model,
            temperature=0.2,
            max_tokens=512,
        )
        return result.get("gaps", [])
    except Exception:
        return []


EXTRACTION_QUALITY_PROMPT = """Compare the extraction against the original source content.

Research Query: {query}
Source Content (first 800 chars): {source}
Extraction Summary: {extraction}

Return JSON:
{{
    "faithfulness": 0.0-1.0,
    "informativeness": 0.0-1.0,
    "hallucinated_claims": 0
}}"""


async def _check_extraction_quality(
    query: str,
    source: dict,
    extraction: dict,
    llm: LLMCaller,
    model: str,
) -> tuple[float, float, int]:
    """Check faithfulness and informativeness of a single extraction."""
    source_content = (source.get("content", "") or "")[:800]
    ext_summary = extraction.get("summary", extraction.get("text", ""))[:500]

    try:
        result = await llm.complete_json(
            EXTRACTION_QUALITY_PROMPT.format(
                query=query, source=source_content, extraction=ext_summary,
            ),
            model=model,
            temperature=0.1,
            max_tokens=256,
        )
        faith = max(0.0, min(1.0, float(result.get("faithfulness", 0.5))))
        info = max(0.0, min(1.0, float(result.get("informativeness", 0.5))))
        halluc = max(0, int(result.get("hallucinated_claims", 0)))
        return faith, info, halluc
    except Exception:
        return 0.5, 0.5, 0


SYNTHESIS_QUALITY_PROMPT = """Evaluate the synthesis quality of this research report.

Research Query: {query}
Report (first 2000 chars): {report}

Assess:
1. How many factual claims does the report make?
2. How many of those claims are supported by cited sources?
3. Rate the structural coherence (0-1).

Return JSON:
{{
    "n_claims_total": integer,
    "n_claims_supported": integer,
    "coherence_score": 0.0-1.0
}}"""


async def _evaluate_synthesis_quality(
    query: str,
    report_text: str,
    llm: LLMCaller,
    model: str,
) -> dict:
    """Evaluate synthesis quality of a report."""
    try:
        result = await llm.complete_json(
            SYNTHESIS_QUALITY_PROMPT.format(
                query=query, report=report_text[:2000],
            ),
            model=model,
            temperature=0.2,
            max_tokens=256,
        )
        return result
    except Exception:
        return {"n_claims_total": 0, "n_claims_supported": 0, "coherence_score": 0.5}
