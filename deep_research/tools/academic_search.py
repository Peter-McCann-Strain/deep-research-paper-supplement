"""Semantic Scholar + arXiv academic search."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import arxiv
import structlog

from deep_research.config import CACHE_DIR
from deep_research.types import Document, SourceType

log = structlog.get_logger()

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,url,year,authors,citationCount,externalIds"

_CACHE_DIR = CACHE_DIR / "academic_search"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prefix: str, query: str) -> str:
    """Generate cache key from prefix and query."""
    return hashlib.sha256(f"{prefix}:{query}".encode()).hexdigest()[:16]


class AcademicSearcher:
    """Combined Semantic Scholar + arXiv search with caching."""

    def __init__(self, max_concurrent: int = 2, use_cache: bool = True):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.use_cache = use_cache

    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Document]:
        """Search Semantic Scholar API with caching."""
        import os
        if os.environ.get("SEARCH_BACKEND") == "oracle":
            return []  # oracle mode: academic evidence is pooled into the frozen corpus
        # Check cache
        ck = _cache_key("s2", query)
        cache_path = _CACHE_DIR / f"{ck}.json"
        if self.use_cache and cache_path.exists():
            log.debug("s2_cache_hit", query=query[:60])
            data_list = json.loads(cache_path.read_text())
            return [Document(**d) for d in data_list]

        async with self._semaphore:
            try:
                # Semantic Scholar API key (optional but recommended): raises the rate limit
                # far above the keyless free tier, which 429s under load (the academic channel
                # was S2-down all through the Mar-Apr corpus generation for exactly this reason).
                # Add SEMANTIC_SCHOLAR_API_KEY to .env and S2 retrieval becomes healthy — no code change.
                import asyncio as _asyncio, random as _random
                _s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY") or ""
                _s2_headers = {"x-api-key": _s2_key} if _s2_key else {}
                async with aiohttp.ClientSession() as session:
                    params = {"query": query, "limit": limit, "fields": S2_FIELDS}
                    # Rate-limit mitigation: S2's keyless tier 429s under load. Retry with
                    # exponential backoff, honouring Retry-After; retry on 429/5xx only.
                    _MAX_RETRIES = 5
                    data = None
                    for _attempt in range(_MAX_RETRIES):
                        async with session.get(S2_API, params=params, headers=_s2_headers,
                                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                break
                            if resp.status in (429, 500, 502, 503, 504) and _attempt < _MAX_RETRIES - 1:
                                _ra = resp.headers.get("Retry-After")
                                _wait = (float(_ra) if (_ra and _ra.replace('.', '', 1).isdigit())
                                         else min(2.0 ** _attempt, 16.0)) + _random.uniform(0, 0.75)
                                log.warning("s2_retry", status=resp.status, attempt=_attempt + 1, wait_s=round(_wait, 1))
                                await _asyncio.sleep(_wait)
                                continue
                            log.warning("s2_error", status=resp.status)
                            return []
                    if data is None:
                        log.warning("s2_error", status="retries_exhausted")
                        return []
            except Exception as e:
                log.warning("s2_exception", error=str(e))
                return []

        docs = []
        for p in data.get("data", []):
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
            content = p.get("abstract") or ""
            docs.append(Document(
                id=p.get("paperId", ""),
                title=p.get("title", ""),
                content=content,
                url=p.get("url", ""),
                source_type=SourceType.SEMANTIC_SCHOLAR,
                metadata={
                    "year": p.get("year"),
                    "authors": authors,
                    "citations": p.get("citationCount", 0),
                    "arxiv_id": (p.get("externalIds") or {}).get("ArXiv"),
                },
            ))
        log.info("s2_results", query=query[:60], count=len(docs))

        # Save cache
        if self.use_cache and docs:
            cache_path.write_text(
                json.dumps([d.model_dump(mode="json") for d in docs], default=str)
            )

        return docs

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Document]:
        """Search arXiv with caching."""
        import os
        if os.environ.get("SEARCH_BACKEND") == "oracle":
            return []  # oracle mode: academic evidence is pooled into the frozen corpus
        # Check cache
        ck = _cache_key("arxiv", query)
        cache_path = _CACHE_DIR / f"{ck}.json"
        if self.use_cache and cache_path.exists():
            log.debug("arxiv_cache_hit", query=query[:60])
            data_list = json.loads(cache_path.read_text())
            return [Document(**d) for d in data_list]

        try:
            client = arxiv.Client(num_retries=1, delay_seconds=2.0)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            # arxiv library is synchronous — run in thread, HARD-CAPPED at 20s so a
            # transient arxiv 500 can't stall the query (was a ~135s default-retry stall).
            results = await asyncio.wait_for(
                asyncio.to_thread(lambda: list(client.results(search))), timeout=20.0,
            )
        except Exception as e:
            log.warning("arxiv_exception", error=str(e))
            return []

        docs = []
        for r in results:
            authors = ", ".join(a.name for a in r.authors[:3])
            docs.append(Document(
                id=r.entry_id,
                title=r.title,
                content=r.summary,
                url=r.entry_id,
                source_type=SourceType.ARXIV,
                metadata={
                    "year": r.published.year if r.published else None,
                    "authors": authors,
                    "categories": r.categories,
                },
            ))
        log.info("arxiv_results", query=query[:60], count=len(docs))

        # Save cache
        if self.use_cache and docs:
            cache_path.write_text(
                json.dumps([d.model_dump(mode="json") for d in docs], default=str)
            )

        return docs

    async def search(
        self,
        query: str,
        max_per_source: int = 10,
    ) -> List[Document]:
        """Search both sources, deduplicate by title similarity."""
        s2_task = self.search_semantic_scholar(query, limit=max_per_source)
        arxiv_task = self.search_arxiv(query, max_results=max_per_source)
        s2_docs, arxiv_docs = await asyncio.gather(s2_task, arxiv_task)

        # Deduplicate by normalized title
        seen_titles: set = set()
        combined: List[Document] = []
        for doc in s2_docs + arxiv_docs:
            norm = doc.title.lower().strip()
            if norm not in seen_titles:
                seen_titles.add(norm)
                combined.append(doc)

        return combined
