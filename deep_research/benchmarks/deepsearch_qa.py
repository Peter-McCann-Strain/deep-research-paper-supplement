"""DeepSearchQA benchmark integration (Google).

Evaluates exhaustive multi-step retrieval across 17 fields.
900 prompts with expert-validated answers requiring deep web search.

Dataset: https://huggingface.co/datasets/google/deepsearchqa
Schema: {problem, problem_category, answer, answer_type}
answer_type: "Single Answer" | "List" | etc.

Designed to test whether systems can find answers that require
synthesizing information across multiple sources.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

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

_CACHE_DIR = DATA_DIR / "benchmarks" / "deepsearch_qa"


class DeepSearchQABenchmark(BenchmarkDataset):
    """DeepSearchQA: multi-step retrieval evaluation (Google)."""

    @property
    def name(self) -> str:
        return "DeepSearchQA"

    @property
    def description(self) -> str:
        return "Multi-step retrieval across 17 fields (Google)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load DeepSearchQA dataset."""
        cache_path = _CACHE_DIR / "deepsearch_qa_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("deepsearch_qa_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            if queries:
                cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("deepsearch_qa_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download DeepSearchQA from HuggingFace.

        Schema: {problem, problem_category, answer, answer_type}
        Split: eval (900 examples)
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("google/deepsearchqa", split="eval")

            queries = []
            for i, row in enumerate(ds):
                answer = row.get("answer", "")
                answer_type = row.get("answer_type", "")
                category = row.get("problem_category", "")

                queries.append(
                    BenchmarkQuery(
                        id=f"dsqa_{i:04d}",
                        query=row.get("problem", ""),
                        domain=category,
                        difficulty="multi-step",
                        reference_answer=answer,
                        rubric={
                            "answer_type": answer_type,
                            "expected_answer": answer,
                        },
                        metadata={
                            "problem_category": category,
                            "answer_type": answer_type,
                        },
                    )
                )

            return queries

        except ImportError as exc:
            log.error("datasets_not_installed")
            raise BenchmarkLoadError(
                "DeepSearchQA requires the optional datasets package. Install the datasets package or use the public API workflow."
            ) from exc
        except Exception as e:
            log.error("deepsearch_qa_download_failed", error=str(e))
            raise BenchmarkLoadError(
                "DeepSearchQA download failed; check network access and dataset availability"
            ) from e

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a report against DeepSearchQA expected answers.

        DeepSearchQA is primarily about finding the correct answer through
        multi-step retrieval. Scoring checks:
        1. Answer correctness (does the report contain the expected answer)
        2. Supporting evidence (does it explain how it arrived at the answer)
        3. Source quality (are citations provided)
        """
        report_text = report.full_text()
        scores: Dict[str, float] = {}

        # 1. Answer correctness
        expected = query.rubric.get("expected_answer", "")
        if expected:
            scores["answer_found"] = self._check_answer(
                report_text, expected, query.rubric.get("answer_type", "")
            )
        else:
            scores["answer_found"] = 0.0

        # 2. Supporting evidence / reasoning
        scores["evidence_quality"] = self._score_evidence(report)

        # 3. Source quality
        scores["source_quality"] = self._score_sources(report)

        # Overall: answer correctness is most important
        weights = {
            "answer_found": 0.50,
            "evidence_quality": 0.30,
            "source_quality": 0.20,
        }
        overall = sum(scores.get(k, 0) * w for k, w in weights.items())

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
        )

    def _check_answer(self, report_text: str, expected: str, answer_type: str) -> float:
        """Check if the expected answer appears in the report."""
        report_lower = report_text.lower()
        expected_lower = expected.lower().strip()

        if not expected_lower:
            return 0.0

        # Direct containment check
        if expected_lower in report_lower:
            return 1.0

        # For list-type answers, check each element
        if answer_type == "List":
            items = [item.strip() for item in re.split(r"[,;\n]", expected_lower) if item.strip()]
            if items:
                found = sum(1 for item in items if item in report_lower)
                return found / len(items)

        # Fuzzy: check if key terms from the answer appear
        terms = [t for t in expected_lower.split() if len(t) > 3]
        if terms:
            found = sum(1 for t in terms if t in report_lower)
            ratio = found / len(terms)
            if ratio >= 0.8:
                return 0.8
            elif ratio >= 0.5:
                return 0.5
            elif ratio >= 0.3:
                return 0.3

        return 0.0

    def _score_evidence(self, report: ResearchReport) -> float:
        """Score the quality of supporting evidence."""
        score = 0.0
        text = report.full_text()
        words = len(text.split())

        # Has substantial content
        if words >= 1000:
            score += 0.3
        elif words >= 500:
            score += 0.2

        # Has sections (structured reasoning)
        if report.sections:
            score += 0.2
        if len(report.sections) >= 3:
            score += 0.1

        # Has inline citations
        inline_refs = set(re.findall(r"\[\d+\]", text))
        if inline_refs:
            score += 0.2

        # Has reasoning markers
        for marker in [
            "because",
            "therefore",
            "evidence",
            "according to",
            "based on",
            "suggests that",
            "indicates",
        ]:
            if marker in text.lower():
                score += 0.05
                if score >= 1.0:
                    break

        return min(1.0, score)

    def _score_sources(self, report: ResearchReport) -> float:
        """Score source quality."""
        score = 0.0

        if report.citations:
            score += 0.3
            # Sources with URLs
            with_url = sum(1 for c in report.citations if c.source_url)
            score += 0.3 * (with_url / len(report.citations))
            # Diverse sources
            unique_domains = set()
            for c in report.citations:
                if c.source_url:
                    parts = c.source_url.split("/")
                    if len(parts) > 2:
                        unique_domains.add(parts[2])
            score += min(0.2, len(unique_domains) * 0.04)
            # Quantity
            if len(report.citations) >= 5:
                score += 0.2
            elif len(report.citations) >= 3:
                score += 0.1

        return min(1.0, score)


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
