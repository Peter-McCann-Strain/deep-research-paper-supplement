"""Benchmark dataset integrations for evaluating research patterns.

Supports multiple evaluation benchmarks with a unified interface:
- DRACO (Perplexity): Weighted rubric scoring across 10 domains
- ResearchQA: PhD-annotated rubric evaluation across 8 fields
- ScholarQABench (OpenScholar/AI2): Scientific literature synthesis
- FreshWiki (STORM): Wikipedia-article generation quality
- DeepSearchQA (Google): Multi-step retrieval across 17 fields
- LitQA2 (FutureHouse): Scientific literature MCQ with superhuman baseline
"""

from deep_research.benchmarks.base import (
    BenchmarkDataset,
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkSuite,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkQuery",
    "BenchmarkResult",
    "BenchmarkSuite",
]
