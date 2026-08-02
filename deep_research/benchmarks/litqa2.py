"""LitQA2 benchmark integration (FutureHouse / LAB-Bench).

Multiple-choice scientific literature comprehension questions.
199 expert-crafted questions requiring full-text paper understanding.
PaperQA2 achieved superhuman precision (85.2% vs human 73.8%).

Dataset: https://huggingface.co/datasets/futurehouse/lab-bench (LitQA2 subset)
Paper: Skarlinski et al. (2024)

Schema: {id, question, ideal, distractors, sources, key-passage, ...}
Scoring: Multiple-choice accuracy (exact match)

Leaderboard reference:
    PaperQA2: Precision 85.2%, Accuracy 66.0%
    Human experts: Precision 73.8%, Accuracy 67.7%
"""

from __future__ import annotations

import json
import re
import random
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

_CACHE_DIR = DATA_DIR / "benchmarks" / "litqa2"

# The "Insufficient Information" option that PaperQA2 uses
INSUFFICIENT_INFO = "Insufficient information to answer this question"


class LitQA2Benchmark(BenchmarkDataset):
    """LitQA2: scientific literature MCQ with superhuman baseline."""

    @property
    def name(self) -> str:
        return "LitQA2"

    @property
    def description(self) -> str:
        return "Scientific literature MCQ, superhuman PaperQA2 baseline (FutureHouse)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load LitQA2 dataset."""
        cache_path = _CACHE_DIR / "litqa2_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("litqa2_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            if queries:
                cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("litqa2_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download LitQA2 from HuggingFace (futurehouse/lab-bench, LitQA2 subset).

        Schema: {id, question, ideal, distractors, sources, key-passage,
                 canary, tag, version, is_opensource, subtask, split}
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("futurehouse/lab-bench", name="LitQA2", split="train")

            queries = []
            for i, row in enumerate(ds):
                ideal = row.get("ideal", "")
                distractors_raw = row.get("distractors", "[]")

                # Parse distractors (stored as JSON string)
                if isinstance(distractors_raw, str):
                    try:
                        distractors = json.loads(distractors_raw)
                    except json.JSONDecodeError:
                        distractors = [distractors_raw]
                elif isinstance(distractors_raw, list):
                    distractors = distractors_raw
                else:
                    distractors = []

                # Build answer options (shuffled, with insufficient info option)
                options = distractors + [ideal]
                # Add "Insufficient Information" as an option (like PaperQA2 does)
                options.append(INSUFFICIENT_INFO)

                # Parse sources
                sources_raw = row.get("sources", "[]")
                if isinstance(sources_raw, str):
                    try:
                        sources = json.loads(sources_raw)
                    except json.JSONDecodeError:
                        sources = [sources_raw] if sources_raw else []
                elif isinstance(sources_raw, list):
                    sources = sources_raw
                else:
                    sources = []

                queries.append(
                    BenchmarkQuery(
                        id=row.get("id", f"litqa2_{i:04d}"),
                        query=row.get("question", ""),
                        domain="scientific_literature",
                        difficulty="expert",
                        reference_answer=ideal,
                        rubric={
                            "ideal": ideal,
                            "distractors": distractors,
                            "options": options,
                        },
                        expected_citations=sources,
                        metadata={
                            "key_passage": row.get("key-passage", ""),
                            "is_opensource": row.get("is_opensource", False),
                            "tag": row.get("tag", ""),
                            "sources": sources,
                            "n_options": len(options),
                        },
                    )
                )

            return queries

        except ImportError as exc:
            log.error("datasets_not_installed")
            raise BenchmarkLoadError(
                "LitQA2 requires the optional datasets package. Install the datasets package or use the public API workflow."
            ) from exc
        except Exception as e:
            log.error("litqa2_download_failed", error=str(e))
            raise BenchmarkLoadError(
                "LitQA2 download failed; check network access and dataset availability"
            ) from e

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a report against LitQA2 expected answer.

        LitQA2 is MCQ — we check if the report contains the correct answer
        and doesn't prefer a distractor. Metrics:
        - answer_correct: 1.0 if correct answer found, 0.0 otherwise
        - answer_attempted: 1.0 if any answer given (not "insufficient info")
        - precision: correct / attempted
        """
        report_text = report.full_text()
        ideal = query.rubric.get("ideal", "")
        distractors = query.rubric.get("distractors", [])
        scores: Dict[str, float] = {}

        if not ideal:
            return BenchmarkResult(
                benchmark_name=self.name,
                pattern_name=report.pattern_name,
                query_id=query.id,
                scores={},
                overall_score=0.0,
            )

        # Check if the correct answer is present
        report_lower = report_text.lower()
        ideal_lower = ideal.lower().strip()
        correct_found = ideal_lower in report_lower

        # Check if any distractor is more prominent
        distractor_found = False
        for dist in distractors:
            if dist.lower().strip() in report_lower:
                distractor_found = True
                break

        # Determine the result
        if correct_found and not distractor_found:
            scores["answer_correct"] = 1.0
            scores["answer_attempted"] = 1.0
        elif correct_found and distractor_found:
            # Both found — check which appears first / more prominently
            ideal_pos = report_lower.find(ideal_lower)
            dist_positions = [
                report_lower.find(d.lower().strip())
                for d in distractors
                if d.lower().strip() in report_lower
            ]
            if ideal_pos <= min(dist_positions):
                scores["answer_correct"] = 1.0
            else:
                scores["answer_correct"] = 0.0
            scores["answer_attempted"] = 1.0
        elif distractor_found:
            scores["answer_correct"] = 0.0
            scores["answer_attempted"] = 1.0
        else:
            # No answer found — treated as "insufficient information"
            scores["answer_correct"] = 0.0
            scores["answer_attempted"] = 0.0

        # Precision: correct / attempted (match PaperQA2's key metric)
        if scores["answer_attempted"] > 0:
            scores["precision"] = scores["answer_correct"]
        else:
            scores["precision"] = 0.0  # Abstained

        # Source quality bonus
        scores["source_quality"] = self._score_sources(report, query)

        # Overall: accuracy (correct/total)
        overall = scores["answer_correct"]

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
        )

    def _score_sources(self, report: ResearchReport, query: BenchmarkQuery) -> float:
        """Score whether the system found relevant source papers."""
        expected_sources = query.expected_citations
        if not expected_sources:
            return 0.5  # Neutral if no expected sources

        report_text = report.full_text().lower()
        found = 0
        for source in expected_sources:
            # Sources are like "arxiv:2407.10362"
            source_lower = source.lower()
            # Check if arxiv ID or DOI appears
            if source_lower in report_text:
                found += 1
            # Also check the numeric part
            elif ":" in source_lower:
                num_part = source_lower.split(":")[-1]
                if num_part in report_text:
                    found += 1

        return found / len(expected_sources) if expected_sources else 0.0


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
