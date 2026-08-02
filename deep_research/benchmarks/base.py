"""Base classes for benchmark dataset integration.

Provides a unified interface for loading, running, and scoring
benchmark datasets against our research patterns.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from deep_research.config import REPORTS_DIR
from deep_research.types import ResearchReport

log = structlog.get_logger()


class BenchmarkLoadError(RuntimeError):
    """Raised when a benchmark dataset cannot be loaded safely."""


# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class BenchmarkQuery:
    """A single query from a benchmark dataset."""

    id: str
    query: str
    domain: str = ""
    difficulty: str = ""
    # Rubric / expected answer elements
    rubric: Dict[str, Any] = field(default_factory=dict)
    reference_answer: str = ""
    expected_citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of evaluating a single pattern × query against a benchmark."""

    benchmark_name: str
    pattern_name: str
    query_id: str
    # Scores (benchmark-specific, normalized to 0-1 where possible)
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    # Resource usage
    cost_usd: float = 0.0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    # Details
    report_text: str = ""
    scoring_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "pattern": self.pattern_name,
            "query_id": self.query_id,
            "overall_score": round(self.overall_score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "cost_usd": round(self.cost_usd, 4),
            "tokens": self.total_tokens,
            "latency_s": round(self.latency_seconds, 1),
        }


# ── Abstract benchmark dataset ──────────────────────────────────────────────


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark dataset integrations.

    Each benchmark implements:
    - load(): Download/cache the dataset and return queries
    - score(): Evaluate a generated report against a benchmark query
    - name: Human-readable name for the benchmark
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this benchmark evaluates."""

    @abstractmethod
    async def load(self, max_queries: int = 0) -> List[BenchmarkQuery]:
        """Load benchmark queries.

        Args:
            max_queries: Maximum queries to load (0 = all).

        Returns:
            List of BenchmarkQuery objects.
        """

    @abstractmethod
    async def score(
        self,
        query: BenchmarkQuery,
        report: ResearchReport,
    ) -> BenchmarkResult:
        """Score a generated report against a benchmark query.

        Args:
            query: The benchmark query that was asked.
            report: The generated research report to evaluate.

        Returns:
            BenchmarkResult with scores and details.
        """

    async def score_batch(
        self,
        queries: List[BenchmarkQuery],
        reports: List[ResearchReport],
    ) -> List[BenchmarkResult]:
        """Score multiple reports. Default: sequential."""
        results = []
        for q, r in zip(queries, reports):
            result = await self.score(q, r)
            results.append(result)
        return results


# ── Pattern registry ─────────────────────────────────────────────────────────

PATTERN_MODULES = {
    "p0_baseline": "deep_research.patterns.p0_baseline.pipeline",
    "p1_iterative_rag": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2_supervisor_parallel": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3_meridian": "deep_research.patterns.p3_meridian.pipeline",
    "p4_perspective_storm": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5_hierarchical_wd": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6_reactive_interleaved": "deep_research.patterns.p6_reactive_interleaved.pipeline",
}


# ── Benchmark suite runner ───────────────────────────────────────────────────


class BenchmarkSuite:
    """Runs patterns against benchmark datasets and collects results."""

    def __init__(
        self,
        benchmarks: List[BenchmarkDataset],
        patterns: Optional[List[str]] = None,
        budget_usd: float = 5.0,
        max_queries_per_benchmark: int = 10,
    ):
        self.benchmarks = benchmarks
        self.patterns = patterns or list(PATTERN_MODULES.keys())
        self.budget_usd = budget_usd
        self.max_queries = max_queries_per_benchmark
        self.results: List[BenchmarkResult] = []

    async def run_pattern(
        self,
        pattern_name: str,
        query: BenchmarkQuery,
    ) -> ResearchReport:
        """Run a single pattern against a single query."""
        mod = importlib.import_module(PATTERN_MODULES[pattern_name])
        start = time.time()
        report = await mod.run(query.query, budget_usd=self.budget_usd)
        report.elapsed_seconds = time.time() - start
        report.pattern_name = pattern_name
        return report

    async def run_all(self) -> List[BenchmarkResult]:
        """Run all patterns × all benchmark queries."""
        all_results: List[BenchmarkResult] = []

        for benchmark in self.benchmarks:
            log.info("benchmark_load", name=benchmark.name)
            queries = await benchmark.load(max_queries=self.max_queries)
            log.info("benchmark_loaded", name=benchmark.name, queries=len(queries))

            for pattern_name in self.patterns:
                if pattern_name not in PATTERN_MODULES:
                    log.warning("unknown_pattern", pattern=pattern_name)
                    continue

                for query in queries:
                    log.info(
                        "benchmark_run",
                        benchmark=benchmark.name,
                        pattern=pattern_name,
                        query=query.id,
                    )
                    try:
                        report = await self.run_pattern(pattern_name, query)
                        result = await benchmark.score(query, report)
                        result.pattern_name = pattern_name
                        result.cost_usd = report.total_cost_usd
                        result.total_tokens = report.total_tokens
                        result.latency_seconds = report.elapsed_seconds
                        result.report_text = report.full_text()
                        all_results.append(result)

                        log.info(
                            "benchmark_scored",
                            benchmark=benchmark.name,
                            pattern=pattern_name,
                            query=query.id,
                            score=f"{result.overall_score:.3f}",
                        )

                    except Exception as e:
                        log.error(
                            "benchmark_error",
                            benchmark=benchmark.name,
                            pattern=pattern_name,
                            query=query.id,
                            error=str(e),
                        )
                        all_results.append(
                            BenchmarkResult(
                                benchmark_name=benchmark.name,
                                pattern_name=pattern_name,
                                query_id=query.id,
                            )
                        )

        self.results = all_results
        return all_results

    def generate_report(self) -> str:
        """Generate a markdown comparison report from results."""
        if not self.results:
            return "No benchmark results to report."

        lines = ["# Benchmark Evaluation Results\n"]

        # Group by benchmark
        by_benchmark: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_benchmark.setdefault(r.benchmark_name, []).append(r)

        for bench_name, bench_results in sorted(by_benchmark.items()):
            lines.append(f"\n## {bench_name}\n")

            # Summary by pattern
            by_pattern: Dict[str, List[BenchmarkResult]] = {}
            for r in bench_results:
                by_pattern.setdefault(r.pattern_name, []).append(r)

            lines.append("| Pattern | Avg Score | Avg Cost | Avg Latency | Queries |")
            lines.append("|---------|-----------|----------|-------------|---------|")

            for pname, presults in sorted(
                by_pattern.items(), key=lambda x: -_avg(x[1], "overall_score")
            ):
                avg_score = _avg(presults, "overall_score")
                avg_cost = _avg(presults, "cost_usd")
                avg_lat = _avg(presults, "latency_seconds")
                lines.append(
                    f"| {pname} | {avg_score:.3f} | ${avg_cost:.3f} "
                    f"| {avg_lat:.0f}s | {len(presults)} |"
                )

            # Per-query breakdown
            lines.append(f"\n### Per-Query Results\n")
            lines.append("| Query | Pattern | Score | Cost | Latency |")
            lines.append("|-------|---------|-------|------|---------|")

            for r in sorted(bench_results, key=lambda x: (x.query_id, -x.overall_score)):
                lines.append(
                    f"| {r.query_id} | {r.pattern_name} "
                    f"| {r.overall_score:.3f} | ${r.cost_usd:.3f} "
                    f"| {r.latency_seconds:.0f}s |"
                )

            # Dimension breakdown if available
            all_dims = set()
            for r in bench_results:
                all_dims.update(r.scores.keys())
            if all_dims:
                lines.append(f"\n### Score Dimensions\n")
                dim_list = sorted(all_dims)
                header = "| Pattern | " + " | ".join(dim_list) + " |"
                sep = "|---------|" + "|".join("------" for _ in dim_list) + "|"
                lines.append(header)
                lines.append(sep)

                for pname, presults in sorted(by_pattern.items()):
                    cells = []
                    for dim in dim_list:
                        vals = [r.scores.get(dim, 0) for r in presults]
                        avg = sum(vals) / len(vals) if vals else 0
                        cells.append(f"{avg:.3f}")
                    lines.append(f"| {pname} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def save_results(self, path: Optional[Path] = None) -> Path:
        """Save results to JSON and markdown."""
        out_dir = path or REPORTS_DIR / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out_dir / "benchmark_results.json"
        json_path.write_text(
            json.dumps(
                [r.to_dict() for r in self.results],
                indent=2,
            )
        )

        # Markdown
        md_path = out_dir / "benchmark_report.md"
        md_path.write_text(self.generate_report())

        log.info("benchmark_results_saved", json=str(json_path), md=str(md_path))
        return out_dir


def _avg(results: List[BenchmarkResult], field: str) -> float:
    vals = [getattr(r, field, 0) for r in results]
    return sum(vals) / len(vals) if vals else 0.0
