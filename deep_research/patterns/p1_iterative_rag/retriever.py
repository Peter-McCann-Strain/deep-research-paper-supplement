"""Stage 2: Multi-source retrieval with two-step source extraction.

Replaces the old chunking/embedding/BM25/FAISS pipeline with a simpler
search -> extract page content -> LLM extract flow.
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from deep_research.tools import (
    AcademicSearcher,
    SourceExtractor,
    SourceExtraction,
    URLExtractor,
    get_web_searcher,
)
from deep_research.tools.cost_tracker import CostTracker
from deep_research.tools.llm_caller import LLMCaller
from deep_research.types import Document, SubQuery
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()


class Retriever:
    """Search multiple sources, extract page content, and LLM-extract each source."""

    def __init__(self, llm: LLMCaller, cost_tracker: CostTracker):
        self.web = get_web_searcher()
        self.academic = AcademicSearcher()
        self.extractor = URLExtractor()
        self.extractor_tool = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
        self._seen_urls: Dict[str, bool] = {}

    async def search_and_summarize(
        self,
        sub_queries: List[SubQuery],
        query: str,
        max_web_per_query: int = 5,
        max_academic_per_query: int = 5,
    ) -> List[SourceExtraction]:
        """Search all sources, extract content, and return structured extractions.

        Returns a list of SourceExtraction objects with fields:
            doc_id, title, url, summary, relevance_score, source_type,
            key_findings, confidence_notes, and optional adaptive fields.
        """
        # ── 1. Web search (batch) ────────────────────────────────────────
        web_queries = [sq.query for sq in sub_queries[:15]]
        web_docs = await self.web.search_batch(
            web_queries, max_results_per=max_web_per_query
        )
        log.info("web_search_done", docs=len(web_docs))

        # ── 2. Academic search (top-priority queries) ────────────────────
        academic_queries = [
            sq.query for sq in sub_queries if sq.priority <= 2
        ][:5]
        academic_docs: List[Document] = []
        for q in academic_queries:
            results = await self.academic.search(
                q, max_per_source=max_academic_per_query
            )
            academic_docs.extend(results)
        log.info("academic_search_done", docs=len(academic_docs))

        # ── 3. Deduplicate by URL ────────────────────────────────────────
        all_docs: List[Document] = []
        for doc in web_docs + academic_docs:
            if doc.url and doc.url not in self._seen_urls:
                self._seen_urls[doc.url] = True
                all_docs.append(doc)
        log.info("deduped_docs", total=len(all_docs))

        # ── 4. Extract full page content where missing ───────────────────
        urls_to_extract = [
            doc.url for doc in all_docs
            if doc.url and len(doc.content) < 500
        ]
        if urls_to_extract:
            extracted = await self.extractor.extract_batch(urls_to_extract)
            url_to_content = {e.url: e.content for e in extracted if e.content}
            # B2 memory guard: on the local 7B run, cap per-source content so the synthesis
            # context fits the constrained VRAM (the GPU is shared with another process).
            import os as _os
            _cap = int(_os.environ.get("DR_LOCAL_CONTENT_CAP", "0"))
            for doc in all_docs:
                if doc.url in url_to_content:
                    c = url_to_content[doc.url]
                    doc.content = c[:_cap] if _cap and len(c) > _cap else c
            log.info("extraction_done", extracted=len(url_to_content))

        # ── 5. Two-step extraction: analyse then structure each source ────
        extractions = await self.extractor_tool.extract_batch(all_docs, query)
        log.info("extraction_done_sources", sources=len(all_docs), relevant=len(extractions))

        return extractions
