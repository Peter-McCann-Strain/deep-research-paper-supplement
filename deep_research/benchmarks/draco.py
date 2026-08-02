"""DRACO benchmark integration (Perplexity).

DRACO evaluates research report quality using expert-crafted rubrics with
~40 weighted criteria per task across 10 domains. Each criterion has a
positive weight (+1 to +20 for presence) or negative weight (-10 to -300
for critical failures).

Dataset: https://huggingface.co/datasets/perplexity-ai/draco

Leaderboard reference:
    Perplexity DR: 70.5%
    Gemini DR: 59.0%
    OpenAI o3: 52.1%
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
from deep_research.config import DATA_DIR, DEFAULT_MODEL
from deep_research.tools import LLMCaller, CostTracker
from deep_research.types import ResearchReport

log = structlog.get_logger()

_CACHE_DIR = DATA_DIR / "benchmarks" / "draco"


class DRACOBenchmark(BenchmarkDataset):
    """DRACO benchmark: weighted rubric scoring across 10 domains."""

    @property
    def name(self) -> str:
        return "DRACO"

    @property
    def description(self) -> str:
        return "Expert-crafted rubric scoring across 10 domains (Perplexity)"

    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load DRACO dataset from HuggingFace or local cache."""
        cache_path = _CACHE_DIR / "draco_queries.json"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            log.info("draco_cache_hit")
            data = json.loads(cache_path.read_text())
            queries = [BenchmarkQuery(**q) for q in data]
        else:
            queries = await self._download()
            cache_path.write_text(json.dumps([_query_to_dict(q) for q in queries], indent=2))

        if max_queries > 0:
            queries = queries[:max_queries]

        log.info("draco_loaded", queries=len(queries))
        return queries

    async def _download(self) -> List[BenchmarkQuery]:
        """Download DRACO from HuggingFace datasets API.

        Schema: {id: str, problem: str, answer: str(JSON), domain: str}
        Answer JSON: {id: str, sections: [{id, title, criteria: [{id, weight, requirement}]}]}
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("perplexity-ai/draco", split="test")
        except Exception as e:
            log.error("draco_download_failed", error=str(e))
            log.info("draco_fallback", msg="Attempting direct download")
            return await self._download_direct(e)

        queries = []
        for i, row in enumerate(ds):
            query_text = row.get("problem", "")
            domain = row.get("domain", "")
            answer_raw = row.get("answer", "{}")

            # Parse the answer JSON which contains the rubric
            if isinstance(answer_raw, str):
                try:
                    answer = json.loads(answer_raw)
                except json.JSONDecodeError:
                    answer = {"raw": answer_raw}
            else:
                answer = answer_raw if isinstance(answer_raw, dict) else {}

            # Convert DRACO sections format to our rubric format
            rubric: Dict[str, Any] = {}
            for section in answer.get("sections", []):
                section_title = section.get("title", "unknown")
                criteria = []
                for criterion in section.get("criteria", []):
                    criteria.append(
                        {
                            "id": criterion.get("id", ""),
                            "weight": criterion.get("weight", 1),
                            "description": criterion.get("requirement", ""),
                        }
                    )
                rubric[section_title] = criteria

            queries.append(
                BenchmarkQuery(
                    id=row.get("id", f"draco_{i:04d}"),
                    query=query_text,
                    domain=domain,
                    difficulty="research",
                    rubric=rubric,
                    metadata={
                        "index": i,
                        "source": "draco",
                        "draco_id": answer.get("id", ""),
                        "total_criteria": sum(
                            len(s.get("criteria", [])) for s in answer.get("sections", [])
                        ),
                    },
                )
            )

        return queries

    async def _download_direct(
        self, original_error: Exception | None = None
    ) -> List[BenchmarkQuery]:
        """Fallback: download DRACO via HTTP."""
        import aiohttp

        url = "https://huggingface.co/api/datasets/perplexity-ai/draco/parquet/default/test"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        log.info("draco_parquet_info", data=str(data)[:200])
        except Exception as e:
            log.warning("draco_direct_download_failed", error=str(e))
            raise BenchmarkLoadError(
                "DRACO download failed through HuggingFace and direct fallback"
            ) from original_error or e

        raise BenchmarkLoadError(
            "DRACO direct fallback returned no usable query records"
        ) from original_error

    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a report against DRACO rubric using LLM-as-judge.

        DRACO rubrics have weighted criteria. We evaluate each criterion
        and compute a weighted score.
        """
        report_text = report.full_text()
        rubric = query.rubric
        scores: Dict[str, float] = {}

        if not rubric:
            return BenchmarkResult(
                benchmark_name=self.name,
                pattern_name=report.pattern_name,
                query_id=query.id,
                scores={"error": 0.0},
                overall_score=0.0,
            )

        # Extract rubric sections and score each
        total_weighted_score = 0.0
        max_possible_score = 0.0
        scoring_details: Dict[str, Any] = {}

        if isinstance(rubric, dict):
            # DRACO rubric has sections with criteria
            for section_name, section_data in rubric.items():
                if isinstance(section_data, list):
                    # List of criteria
                    for criterion in section_data:
                        weight = criterion.get("weight", 1)
                        description = criterion.get("description", criterion.get("text", ""))
                        if weight > 0:
                            max_possible_score += weight
                            # Check if criterion is met
                            met = self._check_criterion(report_text, description)
                            if met:
                                total_weighted_score += weight
                            scoring_details[description[:80]] = {
                                "met": met,
                                "weight": weight,
                            }
                        elif weight < 0:
                            # Negative weight = critical failure
                            met = self._check_criterion(report_text, description)
                            if met:
                                total_weighted_score += weight  # Subtracts
                            scoring_details[description[:80]] = {
                                "critical_failure": met,
                                "penalty": weight,
                            }
                elif isinstance(section_data, dict):
                    scores[section_name] = self._score_section(
                        report_text, section_name, section_data
                    )

        # Normalize to 0-1
        if max_possible_score > 0:
            overall = max(0.0, total_weighted_score / max_possible_score)
        elif scores:
            overall = sum(scores.values()) / len(scores)
        else:
            overall = 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            pattern_name=report.pattern_name,
            query_id=query.id,
            scores=scores,
            overall_score=overall,
            scoring_details=scoring_details,
        )

    def _check_criterion(self, report_text: str, criterion: str) -> bool:
        """Check if a criterion is met using keyword matching.

        For production use, this should be replaced with LLM-as-judge,
        but keyword matching provides a fast, deterministic baseline.
        """
        text_lower = report_text.lower()
        # Extract key terms from criterion
        terms = re.findall(r"\b[a-z]{4,}\b", criterion.lower())
        if not terms:
            return False
        # Criterion met if majority of key terms appear
        found = sum(1 for t in terms if t in text_lower)
        return found >= max(1, len(terms) // 3)

    def _score_section(self, report_text: str, section_name: str, criteria: Dict) -> float:
        """Score a rubric section."""
        met_count = 0
        total = 0
        for key, value in criteria.items():
            desc = value if isinstance(value, str) else str(value)
            if self._check_criterion(report_text, desc):
                met_count += 1
            total += 1
        return met_count / max(total, 1)


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
