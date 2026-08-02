"""Search Worker: individual async worker that searches, reads, and LLM-extracts a sub-topic.

No chunking, embedding, or indexing — each worker collects documents and uses
SourceExtractor to distil each source into a query-relevant structured extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools import (
    get_web_searcher,
    AcademicSearcher,
    URLExtractor,
    SourceExtractor,
    SourceExtraction,
    LLMCaller,
)
from deep_research.tools.cost_tracker import CostTracker
from deep_research.types import Document
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()


class SearchWorker:
    """An individual parallel worker that searches, reads sources, and LLM-extracts a sub-topic."""

    def __init__(
        self,
        worker_id: int,
        cost_tracker: CostTracker,
    ):
        self.worker_id = worker_id
        self.web = get_web_searcher()
        self.academic = AcademicSearcher()
        self.url_extractor = URLExtractor()
        self.llm = LLMCaller(cost_tracker=cost_tracker)
        self.extractor = SourceExtractor(llm=self.llm, model=DEFAULT_MODEL)
        self._log = log.bind(worker_id=worker_id)

    async def run(
        self,
        sub_topic: Dict[str, Any],
        max_web_results: int = 5,
        max_academic_results: int = 5,
    ) -> WorkerResult:
        """Execute the full search-read-extract cycle for a single sub-topic.

        Args:
            sub_topic: dict with keys query, intent, priority, search_type.
            max_web_results: max results from web search.
            max_academic_results: max results from academic search.

        Returns:
            WorkerResult with documents and source extractions.
        """
        query = sub_topic["query"]
        intent = sub_topic.get("intent", query)
        search_type = sub_topic.get("search_type", "both")

        self._log.info("worker_start", query=query[:60], search_type=search_type)

        # -- 1. Search ----------------------------------------------------------
        docs: List[Document] = []

        if search_type in ("web", "both"):
            try:
                web_docs = await self.web.search(query, max_results=max_web_results)
                docs.extend(web_docs)
                self._log.info("worker_web_done", count=len(web_docs))
            except Exception as exc:
                self._log.warning("worker_web_error", error=str(exc))

        if search_type in ("academic", "both"):
            try:
                acad_docs = await self.academic.search(query, max_per_source=max_academic_results)
                docs.extend(acad_docs)
                self._log.info("worker_academic_done", count=len(acad_docs))
            except Exception as exc:
                self._log.warning("worker_academic_error", error=str(exc))

        if not docs:
            self._log.warning("worker_no_docs")
            return WorkerResult(
                worker_id=self.worker_id,
                sub_topic=sub_topic,
                documents=[],
                source_summaries=[],
            )

        # -- 2. Extract full content for web docs that lack it -----------------
        docs = await self._enrich_documents(docs)

        # -- 3. LLM-extract each source ----------------------------------------
        source_extractions = await self.extractor.extract_batch(docs, query)

        self._log.info(
            "worker_done",
            docs=len(docs),
            relevant_extractions=len(source_extractions),
        )

        return WorkerResult(
            worker_id=self.worker_id,
            sub_topic=sub_topic,
            documents=docs,
            source_summaries=source_extractions,
        )

    async def _enrich_documents(self, docs: List[Document]) -> List[Document]:
        """For documents with short content, attempt URL extraction to get full text."""
        urls_to_extract: List[str] = []
        doc_index_map: Dict[str, int] = {}

        for i, doc in enumerate(docs):
            if doc.url and len(doc.content) < 500:
                urls_to_extract.append(doc.url)
                doc_index_map[doc.url] = i

        if not urls_to_extract:
            return docs

        self._log.info("enriching_docs", count=len(urls_to_extract))
        extracted = await self.url_extractor.extract_batch(urls_to_extract)

        for ext_doc in extracted:
            if ext_doc.url in doc_index_map:
                idx = doc_index_map[ext_doc.url]
                # Replace content with richer extraction if substantially longer
                if len(ext_doc.content) > len(docs[idx].content):
                    docs[idx] = docs[idx].model_copy(update={"content": ext_doc.content})

        return docs


class WorkerResult:
    """Container for the output of a single search worker."""

    def __init__(
        self,
        worker_id: int,
        sub_topic: Dict[str, Any],
        documents: List[Document],
        source_summaries: List[SourceExtraction],
    ):
        self.worker_id = worker_id
        self.sub_topic = sub_topic
        self.documents = documents
        self.source_summaries = source_summaries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "sub_topic": self.sub_topic,
            "doc_count": len(self.documents),
            "source_summary_count": len(self.source_summaries),
        }
