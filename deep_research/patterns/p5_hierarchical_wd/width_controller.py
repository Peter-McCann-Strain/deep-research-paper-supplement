"""Width controller: parallel broad search phase.

Dispatches multiple SearchWorker instances in parallel, each handling
a different subtopic. Workers search web/academic sources and produce
LLM-extracted source extractions (no chunking/embedding).

The number of parallel workers is determined by the W(t) schedule
from wd_schedule.py.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.cost_tracker import CostTracker
from deep_research.tools.source_extractor import SourceExtraction
from deep_research.types import Document, SubQuery, WDAllocation

from .operational_agents import SearchWorker

log = structlog.get_logger()


class WidthController:
    """Manages the width (broad search) phase of each iteration.

    Dispatches parallel SearchWorker instances, each responsible for
    a subset of the search queries grouped by subtopic. Workers return
    source extractions instead of indexed chunks.
    """

    def __init__(
        self,
        llm: LLMCaller,
        cost_tracker: CostTracker,
    ):
        self.llm = llm
        self.cost_tracker = cost_tracker

    async def run_width_phase(
        self,
        sub_queries: List[SubQuery],
        allocation: WDAllocation,
        research_query: str,
        include_academic: bool = True,
    ) -> Dict[str, Any]:
        """Execute the width phase: parallel broad searches with LLM extraction.

        Args:
            sub_queries: All search queries to distribute across workers.
            allocation: Current width-depth allocation with worker count.
            research_query: The original research query (for extraction context).
            include_academic: Whether to include academic search sources.

        Returns:
            Dict with:
                - docs: all retrieved documents
                - summaries: all source extractions (list of SourceExtraction)
                - worker_results: per-worker extraction counts
                - total_docs: total documents retrieved
                - total_summaries: total relevant extractions produced
        """
        n_workers = allocation.width_workers
        log.info(
            "width_phase_start",
            step=allocation.step,
            workers=n_workers,
            queries=len(sub_queries),
        )

        # Sort queries by priority (highest first)
        sorted_queries = sorted(sub_queries, key=lambda sq: sq.priority)

        # Distribute queries across workers (round-robin by priority groups)
        worker_queries: List[List[str]] = [[] for _ in range(n_workers)]
        for i, sq in enumerate(sorted_queries):
            worker_idx = i % n_workers
            worker_queries[worker_idx].append(sq.query)

        # Remove empty worker slots
        worker_queries = [wq for wq in worker_queries if wq]

        # Cap queries per worker to limit cost
        max_per_worker = max(3, 15 // max(n_workers, 1))
        worker_queries = [wq[:max_per_worker] for wq in worker_queries]

        # Launch parallel workers
        tasks = []
        for idx, queries in enumerate(worker_queries):
            worker = SearchWorker(
                llm=self.llm,
                cost_tracker=self.cost_tracker,
                worker_id=f"w{allocation.step}_{idx}",
            )
            tasks.append(
                worker.search_and_summarize(
                    queries=queries,
                    research_query=research_query,
                    max_web_per_query=5,
                    include_academic=include_academic and idx < 2,  # academic for first 2 workers
                    max_academic_per_query=5,
                )
            )

        # Gather with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs: List[Document] = []
        all_summaries: List[SourceExtraction] = []
        worker_results: List[Dict[str, Any]] = []

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(
                    "width_worker_error",
                    worker=idx,
                    error=str(result),
                )
                worker_results.append({"worker": idx, "docs": 0, "summaries": 0, "error": str(result)})
            else:
                docs = result.get("docs", [])
                summaries = result.get("summaries", [])
                all_docs.extend(docs)
                all_summaries.extend(summaries)
                worker_results.append({"worker": idx, "docs": len(docs), "summaries": len(summaries)})

        log.info(
            "width_phase_complete",
            step=allocation.step,
            total_docs=len(all_docs),
            total_summaries=len(all_summaries),
            workers_succeeded=sum(1 for r in worker_results if "error" not in r),
        )

        return {
            "docs": all_docs,
            "summaries": all_summaries,
            "worker_results": worker_results,
            "total_docs": len(all_docs),
            "total_summaries": len(all_summaries),
        }
