"""Run all patterns against all test queries."""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Dict, List

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.evaluation.test_queries import get_all_queries, TestQuery
from deep_research.evaluation.metrics import EvalResult, evaluate_report
from deep_research.types import ResearchReport

log = structlog.get_logger()

PATTERN_MODULES = {
    "p0_baseline": "deep_research.patterns.p0_baseline.pipeline",
    "p1_iterative_rag": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2_supervisor_parallel": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3_meridian": "deep_research.patterns.p3_meridian.pipeline",
    "p4_perspective_storm": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5_hierarchical_wd": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6_reactive_interleaved": "deep_research.patterns.p6_reactive_interleaved.pipeline",
    "p7_graph_decomposition": "deep_research.patterns.p7_graph_decomposition.pipeline",
    "p8_beam_search": "deep_research.patterns.p8_beam_search.pipeline",
    "p11_react": "deep_research.patterns.p11_react.pipeline",
    "p12_rl_trained": "deep_research.patterns.p12_rl_trained.pipeline",
}


async def run_single(
    pattern_name: str,
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
) -> ResearchReport:
    """Run a single pattern against a single query."""
    mod = importlib.import_module(PATTERN_MODULES[pattern_name])
    start = time.time()
    report = await mod.run(query, budget_usd=budget_usd)
    report.elapsed_seconds = time.time() - start
    return report


async def run_pattern_suite(
    pattern_name: str,
    queries: List[TestQuery],
    budget_usd: float = MAX_COST_PER_RUN,
) -> List[EvalResult]:
    """Run one pattern against all queries."""
    results = []
    for tq in queries:
        log.info("eval_run", pattern=pattern_name, query=tq.id)
        try:
            report = await run_single(pattern_name, tq.query, budget_usd)
            result = evaluate_report(report, tq)
            results.append(result)
            log.info("eval_done", pattern=pattern_name, query=tq.id,
                     score=f"{result.overall_score:.2f}",
                     cost=f"${result.cost_usd:.4f}")
        except Exception as e:
            log.error("eval_error", pattern=pattern_name, query=tq.id, error=str(e))
            results.append(EvalResult(
                pattern_name=pattern_name,
                query_id=tq.id,
            ))
    return results


async def run_all_evaluations(
    budget_usd: float = MAX_COST_PER_RUN,
    patterns: List[str] | None = None,
    query_ids: List[str] | None = None,
) -> List[EvalResult]:
    """Run all patterns × all queries (25 runs)."""
    queries = get_all_queries()
    if query_ids:
        queries = [q for q in queries if q.id in query_ids]

    target_patterns = patterns or list(PATTERN_MODULES.keys())

    all_results: List[EvalResult] = []
    for pname in target_patterns:
        if pname not in PATTERN_MODULES:
            log.warning("unknown_pattern", pattern=pname)
            continue
        results = await run_pattern_suite(pname, queries, budget_usd)
        all_results.extend(results)

    return all_results
