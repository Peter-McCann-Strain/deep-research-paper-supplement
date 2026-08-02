"""ScholarQABench integration (OpenScholar / AI2).

Evaluates scientific literature synthesis with citations.
2,967 queries + 208 expert-written long-form answers across
CS, Physics, Neuroscience, Biomedicine.

Published in Nature (2025). Expert-validated evaluation.

Dataset: https://github.com/AkariAsai/ScholarQABench
HuggingFace: https://huggingface.co/OpenScholar

Scoring dimensions:
- Correctness (vs. expert rubrics/responses)
- Citation Precision (each citation supports its statement)
- Citation Recall (all citation-worthy statements have citations)
- Coverage, Relevance, Organization
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from deep_research.benchmarks.base import (
    BenchmarkDataset,
    BenchmarkLoadError,
    BenchmarkQuery,
    BenchmarkResult,
)
from deep_research.config import DATA_DIR
from deep_research.types import ResearchReport

log = structlog.get_logger()

_CACHE_DIR = DATA_DIR / "benchmarks" / "scholar_qa"


class ScholarQABenchmark(BenchmarkDataset):
    """ScholarQABench: scientific literature synthesis evaluation."""

    @property
    def name(self) -> str:
        return "ScholarQABench"

    @property
    def description(self) -> str:
        return "Scientific literature synthesis with citation evaluation (Nature 2025)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load ScholarQABench dataset."""
        cache_path = _CACHE_DIR / "scholar_qa_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("scholar_qa_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            if queries:
                cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("scholar_qa_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download ScholarQABench from GitHub (not on HuggingFace).

        Source: https://github.com/AkariAsai/ScholarQABench
        The dataset is hosted as JSON files in the GitHub repository.
        """
        import aiohttp

        base_url = "https://raw.githubusercontent.com/AkariAsai/ScholarQABench/main/data"
        queries = []

        for subset in ["cs", "multi"]:
            url = f"{base_url}/{subset}_test.json"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            log.warning(
                                "scholar_qa_download_failed",
                                subset=subset,
                                status=resp.status,
                            )
                            continue
                        data = await resp.json(content_type=None)

                if not isinstance(data, list):
                    data = [data]

                for i, row in enumerate(data):
                    query_text = row.get("input", row.get("question", ""))
                    reference = row.get("output", row.get("answer", ""))
                    citations = row.get("ctxs", [])

                    queries.append(
                        BenchmarkQuery(
                            id=f"sqb_{subset}_{i:04d}",
                            query=query_text,
                            domain=subset,
                            difficulty="scholarly",
                            reference_answer=reference,
                            expected_citations=[c.get("title", "") for c in citations]
                            if isinstance(citations, list)
                            else [],
                            metadata={
                                "subset": subset,
                                "citation_count": (
                                    len(citations) if isinstance(citations, list) else 0
                                ),
                            },
                        )
                    )
            except Exception as e:
                log.warning("scholar_qa_subset_failed", subset=subset, error=str(e))

        if not queries:
            raise BenchmarkLoadError("ScholarQABench download failed for all public subsets")
        return queries

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a report against ScholarQABench criteria."""
        report_text = report.full_text()
        scores: Dict[str, float] = {}

        # 1. Coverage: How much of the reference answer is covered
        if query.reference_answer:
            scores["coverage"] = self._score_coverage(report_text, query.reference_answer)

        # 2. Citation quality
        citation_scores = self._score_citations(report, query)
        scores.update(citation_scores)

        # 3. Relevance: Is the response on-topic
        scores["relevance"] = self._score_relevance(report_text, query.query)

        # 4. Organization: Section structure quality
        scores["organization"] = self._score_organization(report)

        # Overall: weighted combination
        weights = {
            "coverage": 0.30,
            "citation_precision": 0.20,
            "citation_recall": 0.15,
            "relevance": 0.15,
            "organization": 0.10,
            "citation_count_score": 0.10,
        }
        overall = sum(scores.get(k, 0) * w for k, w in weights.items())

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
        )

    def _score_coverage(self, report_text: str, reference: str) -> float:
        """Score coverage of reference answer content."""
        ref_lower = reference.lower()
        report_lower = report_text.lower()

        # Extract key phrases (3+ word segments)
        ref_sentences = [s.strip() for s in ref_lower.split(".") if len(s.strip()) > 20]
        if not ref_sentences:
            return 0.0

        covered = 0
        for sentence in ref_sentences:
            # Extract key terms from each sentence
            terms = [t for t in sentence.split() if len(t) > 3][:5]
            if terms:
                found = sum(1 for t in terms if t in report_lower)
                if found >= max(1, len(terms) // 2):
                    covered += 1

        return covered / len(ref_sentences)

    def _score_citations(self, report: ResearchReport, query: BenchmarkQuery) -> Dict[str, float]:
        """Score citation quality."""
        scores = {}

        # Count inline citations in text
        text = report.full_text()
        inline_refs = set(re.findall(r"\[\d+\]", text))
        scores["citation_count_score"] = min(1.0, len(inline_refs) / 10)

        # Citation precision: what fraction of our citations have content
        if report.citations:
            valid = sum(1 for c in report.citations if c.source_url)
            scores["citation_precision"] = valid / len(report.citations)
        else:
            scores["citation_precision"] = 0.0

        # Citation recall: do we cite the expected sources
        if query.expected_citations:
            found = 0
            for expected in query.expected_citations:
                if expected and any(
                    expected.lower() in c.source_title.lower() for c in report.citations
                ):
                    found += 1
            scores["citation_recall"] = found / len(query.expected_citations)
        else:
            scores["citation_recall"] = scores["citation_precision"]

        return scores

    def _score_relevance(self, report_text: str, query: str) -> float:
        """Score topical relevance."""
        query_terms = [t.lower() for t in query.split() if len(t) > 3]
        if not query_terms:
            return 0.0
        report_lower = report_text.lower()
        found = sum(1 for t in query_terms if t in report_lower)
        return found / len(query_terms)

    def _score_organization(self, report: ResearchReport) -> float:
        """Score structural organization."""
        score = 0.0
        # Has sections
        if report.sections:
            score += 0.3
        # Has 3+ sections
        if len(report.sections) >= 3:
            score += 0.2
        # Has abstract
        if report.abstract:
            score += 0.2
        # Has title
        if report.title:
            score += 0.1
        # Has citations
        if report.citations:
            score += 0.2
        return score


def _query_to_dict(q: BenchmarkQuery) -> Dict[str, Any]:
    return {
        "id": q.id,
        "query": q.query,
        "domain": q.domain,
        "difficulty": q.difficulty,
        "rubric": q.rubric,
        "reference_answer": q.reference_answer,
        "expected_citations": q.expected_citations,
        "metadata": q.metadata,
    }
