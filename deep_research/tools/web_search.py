"""Tavily web search wrapper."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from tavily import AsyncTavilyClient

from deep_research.config import CACHE_DIR, TAVILY_API_KEY
from deep_research.types import Document, SourceType

log = structlog.get_logger()

_CACHE_DIR = CACHE_DIR / "tavily_search"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


class WebSearcher:
    """Tavily-powered web search with caching."""

    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 3):
        self._key = api_key or TAVILY_API_KEY
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_client(self) -> AsyncTavilyClient:
        return AsyncTavilyClient(api_key=self._key)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_raw_content: bool = True,
        use_cache: bool = True,
    ) -> List[Document]:
        """Search Tavily, return Documents."""
        # Check cache
        ck = _cache_key(query)
        cache_path = _CACHE_DIR / f"{ck}.json"
        if use_cache and cache_path.exists():
            log.debug("tavily_cache_hit", query=query[:60])
            data = json.loads(cache_path.read_text())
            return [Document(**d) for d in data]

        async with self._semaphore:
            client = self._get_client()
            log.info("tavily_search", query=query[:60], max_results=max_results)
            raw = await client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )

        docs: List[Document] = []
        for r in raw.get("results", []):
            doc = Document(
                id=_cache_key(r.get("url", "")),
                title=r.get("title", ""),
                content=r.get("raw_content") or r.get("content", ""),
                url=r.get("url", ""),
                source_type=SourceType.WEB,
                metadata={"score": r.get("score", 0)},
            )
            docs.append(doc)

        # Save cache
        if use_cache:
            cache_path.write_text(
                json.dumps([d.model_dump(mode="json") for d in docs], default=str)
            )

        log.info("tavily_results", query=query[:60], count=len(docs))
        return docs

    async def search_batch(
        self,
        queries: List[str],
        max_results_per: int = 5,
    ) -> List[Document]:
        """Search multiple queries, deduplicate by URL."""
        tasks = [self.search(q, max_results=max_results_per) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set = set()
        docs: List[Document] = []
        for result in results:
            if isinstance(result, Exception):
                log.warning("tavily_batch_error", error=str(result))
                continue
            for doc in result:
                if doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    docs.append(doc)

        return docs
