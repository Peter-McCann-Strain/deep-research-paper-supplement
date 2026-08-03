"""ResearchQA benchmark integration.

Evaluates research report quality using expert-crafted rubric items.
21,414 queries across 8 fields with 160K+ rubric items authored by 31 PhD annotators.

Dataset: https://huggingface.co/datasets/realliyifei/ResearchQA
Paper: Li et al. (2025)

Schema: {id, general_domain, subdomain, field, query, date, rubric}
Rubric items: [{citation_metadata, rubric_item (question), type: [str]}]

Leaderboard reference:
    Perplexity Sonar DR: 75.29%
    OpenAI O4-Mini DR:   72.69%
    Claude Sonnet+Search: 69.18%
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

_CACHE_DIR = DATA_DIR / "benchmarks" / "research_qa"


class ResearchQABenchmark(BenchmarkDataset):
    """ResearchQA: PhD-annotated rubric evaluation across 8 academic fields."""

    @property
    def name(self) -> str:
        return "ResearchQA"

    @property
    def description(self) -> str:
        return "PhD-annotated rubric evaluation across 8 fields (Li et al. 2025)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load ResearchQA dataset."""
        cache_path = _CACHE_DIR / "research_qa_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("research_qa_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            if queries:
                cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("research_qa_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download ResearchQA from HuggingFace.

        Schema: {id, general_domain, subdomain, field, query, date, rubric}
        Rubric: list of {citation_metadata, rubric_item, type}
        Splits: valid, test (3750), full (21414), test_mini (776)
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("realliyifei/ResearchQA", split="test")

            queries = []
            for i, row in enumerate(ds):
                rubric_items = row.get("rubric", [])
                if not isinstance(rubric_items, list):
                    rubric_items = []

                # Convert rubric items to our format
                rubric_criteria = []
                rubric_types = set()
                for item in rubric_items:
                    rubric_criteria.append(
                        {
                            "question": item.get("rubric_item", ""),
                            "type": item.get("type", []),
                            "citation_metadata": item.get("citation_metadata"),
                        }
                    )
                    for t in item.get("type", []):
                        rubric_types.add(t)

                queries.append(
                    BenchmarkQuery(
                        id=row.get("id", f"rqa_{i:04d}"),
                        query=row.get("query", ""),
                        domain=row.get("field", row.get("general_domain", "")),
                        difficulty="research",
                        rubric={
                            "criteria": rubric_criteria,
                            "types": list(rubric_types),
                        },
                        metadata={
                            "general_domain": row.get("general_domain", ""),
                            "subdomain": row.get("subdomain", ""),
                            "field": row.get("field", ""),
                            "date": str(row.get("date", "")),
                            "total_rubric_items": len(rubric_criteria),
                        },
                    )
                )

            return queries

        except ImportError as exc:
            log.error("datasets_not_installed")
            raise BenchmarkLoadError(
                "ResearchQA requires the optional datasets package. Install the datasets package or use the public API workflow."
            ) from exc
        except Exception as e:
            log.error("research_qa_download_failed", error=str(e))
            raise BenchmarkLoadError(
                "ResearchQA download failed; check network access and dataset availability"
            ) from e

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a report against ResearchQA rubric items.

        Each rubric item is a yes/no question about the report content.
        The overall score is the fraction of rubric items satisfied.
        """
        report_text = report.full_text()
        scores: Dict[str, float] = {}

        rubric = query.rubric
        criteria = rubric.get("criteria", [])

        if not criteria:
            return BenchmarkResult(
                benchmark_name=self.name,
                pattern_name=report.pattern_name,
                query_id=query.id,
                scores={},
                overall_score=0.0,
            )

        # Score each rubric item
        met_count = 0
        type_scores: Dict[str, List[bool]] = {}

        for item in criteria:
            question = item.get("question", "")
            item_types = item.get("type", [])
            met = self._check_rubric_item(report_text, question)

            if met:
                met_count += 1

            for t in item_types:
                if t not in type_scores:
                    type_scores[t] = []
                type_scores[t].append(met)

        # Overall rubric coverage
        scores["rubric_coverage"] = met_count / len(criteria)

        # Per-type scores
        for type_name, results in type_scores.items():
            key = f"type_{type_name.lower().replace(' ', '_')}"
            scores[key] = sum(results) / len(results)

        # Citation quality
        scores["citation_quality"] = self._score_citation_quality(report)

        # Organization
        scores["organization"] = self._score_organization(report)

        # Overall: primarily rubric coverage (matching ResearchQA's evaluation)
        weights = {
            "rubric_coverage": 0.60,
            "citation_quality": 0.25,
            "organization": 0.15,
        }
        overall = sum(scores.get(k, 0) * w for k, w in weights.items())

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
        )

    def _check_rubric_item(self, report_text: str, question: str) -> bool:
        """Check if a rubric item (yes/no question) is addressed in the report.

        Uses keyword matching as a fast baseline. For production accuracy,
        replace with LLM-as-judge.
        """
        text_lower = report_text.lower()
        # Extract key terms from the rubric question
        # Skip common question words
        stop_words = {
            "does",
            "the",
            "response",
            "provide",
            "explain",
            "describe",
            "discuss",
            "mention",
            "include",
            "address",
            "detail",
            "explore",
            "how",
            "what",
            "whether",
            "about",
            "with",
            "that",
            "this",
            "from",
            "into",
            "their",
            "between",
            "which",
            "have",
            "been",
        }
        terms = [t for t in re.findall(r"\b[a-z]{4,}\b", question.lower()) if t not in stop_words]
        if not terms:
            return False

        # Use a higher threshold for rubric items (they're specific)
        found = sum(1 for t in terms if t in text_lower)
        return found >= max(2, len(terms) // 2)

    def _score_citation_quality(self, report: ResearchReport) -> float:
        """Score citation presence and quality."""
        text = report.full_text()
        inline_refs = set(re.findall(r"\[\d+\]", text))

        score = 0.0
        if inline_refs:
            score += 0.3
        n_refs = len(inline_refs)
        if n_refs >= 15:
            score += 0.3
        elif n_refs >= 10:
            score += 0.25
        elif n_refs >= 5:
            score += 0.15

        if report.citations:
            with_url = sum(1 for c in report.citations if c.source_url)
            score += 0.2 * (with_url / len(report.citations))
            # Diversity bonus
            unique_domains = set()
            for c in report.citations:
                if c.source_url:
                    parts = c.source_url.split("/")
                    if len(parts) > 2:
                        unique_domains.add(parts[2])
            score += min(0.2, len(unique_domains) * 0.03)

        return min(1.0, score)

    def _score_organization(self, report: ResearchReport) -> float:
        """Score structural organization."""
        score = 0.0
        if report.sections:
            score += 0.3
        if len(report.sections) >= 3:
            score += 0.2
        if report.abstract:
            score += 0.2
        if report.title:
            score += 0.1
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
