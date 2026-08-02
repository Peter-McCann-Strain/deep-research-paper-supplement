"""Scoring functions for evaluating research reports."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

from deep_research.types import ResearchReport
from deep_research.evaluation.test_queries import TestQuery


@dataclass
class EvalResult:
    """Evaluation result for a single pattern × query run."""
    pattern_name: str
    query_id: str
    coverage_score: float = 0.0        # % of expected elements found
    coverage_details: Dict[str, bool] = field(default_factory=dict)
    citation_count: int = 0
    unique_sources: int = 0
    report_length_words: int = 0
    section_count: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern_name,
            "query": self.query_id,
            "coverage": f"{self.coverage_score:.1%}",
            "citations": self.citation_count,
            "sources": self.unique_sources,
            "words": self.report_length_words,
            "sections": self.section_count,
            "cost": f"${self.cost_usd:.4f}",
            "tokens": self.total_tokens,
            "latency": f"{self.latency_seconds:.1f}s",
            "overall": f"{self.overall_score:.2f}",
        }


def score_coverage(report: ResearchReport, test_query: TestQuery) -> tuple[float, Dict[str, bool]]:
    """Score what percentage of expected elements are present in the report."""
    full_text = report.full_text().lower()
    details: Dict[str, bool] = {}

    for element in test_query.expected_elements:
        # Check for key terms from the element
        key_terms = [t.strip().lower() for t in re.split(r"[,/:()\[\]]", element) if len(t.strip()) > 3]
        # Element is "found" if majority of key terms appear
        found_count = sum(1 for term in key_terms if term in full_text)
        threshold = max(1, len(key_terms) // 2)
        details[element] = found_count >= threshold

    found = sum(1 for v in details.values() if v)
    total = max(len(details), 1)
    return found / total, details


def count_citations(report: ResearchReport) -> int:
    """Count inline citations in the report text."""
    full_text = report.full_text()
    # Match [1], [2], etc.
    citations = re.findall(r"\[\d+\]", full_text)
    return len(set(citations))


def count_unique_sources(report: ResearchReport) -> int:
    """Count unique source URLs in citations."""
    urls = set()
    for c in report.citations:
        if c.source_url:
            urls.add(c.source_url)
    return len(urls)


def evaluate_report(
    report: ResearchReport,
    test_query: TestQuery,
) -> EvalResult:
    """Full evaluation of a report against a test query."""
    coverage, coverage_details = score_coverage(report, test_query)
    full_text = report.full_text()

    result = EvalResult(
        pattern_name=report.pattern_name,
        query_id=test_query.id,
        coverage_score=coverage,
        coverage_details=coverage_details,
        citation_count=count_citations(report),
        unique_sources=count_unique_sources(report),
        report_length_words=len(full_text.split()),
        section_count=len(report.sections),
        cost_usd=report.total_cost_usd,
        total_tokens=report.total_tokens,
        latency_seconds=report.elapsed_seconds,
    )

    # Overall score: weighted combination
    result.overall_score = (
        coverage * 0.4
        + min(result.citation_count / 15, 1.0) * 0.2
        + min(result.report_length_words / 2000, 1.0) * 0.1
        + min(result.section_count / 5, 1.0) * 0.1
        + max(0, 1.0 - result.cost_usd / 2.0) * 0.1
        + max(0, 1.0 - result.latency_seconds / 300) * 0.1
    )

    return result
