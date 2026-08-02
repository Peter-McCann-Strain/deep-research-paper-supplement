"""Aggregator: collects and deduplicates source extractions from multiple parallel workers.

No FAISS, BM25, RRF, or reranking — simply merges source extractions by URL and
collects worker-level narrative summaries for downstream compression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import structlog

from deep_research.tools.source_extractor import SourceExtraction
from deep_research.types import Document

from .search_worker import WorkerResult

log = structlog.get_logger()


class Aggregator:
    """Aggregates source extractions from multiple workers, deduplicating by URL."""

    async def aggregate(
        self,
        worker_results: List[WorkerResult],
        query: str,
    ) -> AggregatedOutput:
        """Merge results from all workers: deduplicate docs and source extractions.

        Args:
            worker_results: list of WorkerResult from parallel workers.
            query: the original research query (kept for interface consistency).

        Returns:
            AggregatedOutput with documents, source_summaries, and worker_summaries.
        """
        # -- 1. Deduplicate documents by URL ------------------------------------
        seen_urls: Set[str] = set()
        all_docs: List[Document] = []
        for wr in worker_results:
            for doc in wr.documents:
                if doc.url and doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    all_docs.append(doc)
                elif not doc.url:
                    all_docs.append(doc)

        log.info(
            "aggregator_docs",
            total=len(all_docs),
            deduplicated_from=sum(len(wr.documents) for wr in worker_results),
        )

        # -- 2. Deduplicate source extractions by URL --------------------------
        seen_extraction_urls: Set[str] = set()
        all_extractions: List[SourceExtraction] = []
        for wr in worker_results:
            for s in wr.source_summaries:
                url = s.url
                if url and url not in seen_extraction_urls:
                    seen_extraction_urls.add(url)
                    all_extractions.append(s)
                elif not url:
                    all_extractions.append(s)

        log.info(
            "aggregator_extractions",
            total=len(all_extractions),
            deduplicated_from=sum(len(wr.source_summaries) for wr in worker_results),
        )

        # -- 3. Collect worker-level narrative summaries (keyed by sub-topic) ---
        worker_summaries = _collect_worker_summaries(worker_results)

        return AggregatedOutput(
            documents=all_docs,
            source_summaries=all_extractions,
            worker_summaries=worker_summaries,
        )

    async def aggregate_additional(
        self,
        new_results: List[WorkerResult],
        query: str,
        existing_output: "AggregatedOutput",
    ) -> "AggregatedOutput":
        """Incrementally merge gap-fill results into existing output.

        Args:
            new_results: gap-fill worker results.
            query: original research query.
            existing_output: output from the initial aggregation.

        Returns:
            Merged AggregatedOutput.
        """
        # Start from existing
        combined_docs = list(existing_output.documents)
        combined_extractions = list(existing_output.source_summaries)
        combined_worker_summaries = dict(existing_output.worker_summaries)

        seen_urls = {d.url for d in combined_docs if d.url}
        seen_extraction_urls = {s.url for s in combined_extractions if s.url}

        for wr in new_results:
            for doc in wr.documents:
                if doc.url and doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    combined_docs.append(doc)
            for s in wr.source_summaries:
                url = s.url
                if url and url not in seen_extraction_urls:
                    seen_extraction_urls.add(url)
                    combined_extractions.append(s)
                elif not url:
                    combined_extractions.append(s)
            key = wr.sub_topic.get("query", f"gap_worker_{wr.worker_id}")
            combined_worker_summaries[key] = _worker_narrative(wr)

        log.info(
            "aggregator_incremental_done",
            docs=len(combined_docs),
            extractions=len(combined_extractions),
        )

        return AggregatedOutput(
            documents=combined_docs,
            source_summaries=combined_extractions,
            worker_summaries=combined_worker_summaries,
        )


class AggregatedOutput:
    """Container for the aggregated results from all workers."""

    def __init__(
        self,
        documents: List[Document],
        source_summaries: List[SourceExtraction],
        worker_summaries: Dict[str, str],
    ):
        self.documents = documents
        self.source_summaries = source_summaries
        self.worker_summaries = worker_summaries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_count": len(self.documents),
            "source_summary_count": len(self.source_summaries),
            "worker_summary_count": len(self.worker_summaries),
        }


def _worker_narrative(wr: WorkerResult) -> str:
    """Build a narrative from a worker's source extractions."""
    if not wr.source_summaries:
        return "No relevant sources found for this sub-topic."
    parts = []
    for s in wr.source_summaries:
        parts.append(f"[{s.title or 'Unknown'}]: {s.summary}")
    return "\n\n".join(parts)


def _collect_worker_summaries(worker_results: List[WorkerResult]) -> Dict[str, str]:
    """Collect worker narratives keyed by sub-topic query."""
    summaries: Dict[str, str] = {}
    for wr in worker_results:
        key = wr.sub_topic.get("query", f"worker_{wr.worker_id}")
        summaries[key] = _worker_narrative(wr)
    return summaries
