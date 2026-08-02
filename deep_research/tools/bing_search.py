"""Azure OpenAI web search via Responses API with web_search_preview tool.

Pipeline: Search (Responses API) → Discover URLs → Fetch full pages (trafilatura) → Return Documents
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import structlog
from openai import AsyncAzureOpenAI

from deep_research.config import (
    SEARCH_OPENAI_API_KEY,
    SEARCH_OPENAI_ENDPOINT,
    SEARCH_MODEL,
    CACHE_DIR,
)
from deep_research.tools.url_extractor import URLExtractor
from deep_research.types import Document, SourceType

log = structlog.get_logger()

_CACHE_DIR = CACHE_DIR / "bing_search"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Responses API needs a newer API version
_RESPONSES_API_VERSION = "2025-03-01-preview"


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


class BingSearcher:
    """Web search via Azure OpenAI Responses API + full page extraction.

    Two-level pipeline:
      1. Responses API (web_search_preview) discovers relevant URLs
      2. URLExtractor (trafilatura) fetches full page content for each URL

    This gives us Tavily-quality raw content (~8-20K chars per page)
    at zero API cost (PTU + free web search).
    """

    def __init__(
        self,
        model: str = SEARCH_MODEL,
        max_concurrent: int = 3,
        fetch_pages: bool = True,
        max_page_fetch: int = 10,
    ):
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        import httpx
        self._client = AsyncAzureOpenAI(
            api_key=SEARCH_OPENAI_API_KEY,
            azure_endpoint=SEARCH_OPENAI_ENDPOINT,
            api_version=_RESPONSES_API_VERSION,
            max_retries=5,  # SDK-level retries for transient errors
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        )
        self._fetch_pages = fetch_pages
        self._max_page_fetch = max_page_fetch
        self._extractor = URLExtractor()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> List[Document]:
        """Search the web via Responses API, then fetch full page content."""
        ck = _cache_key(query)
        cache_path = _CACHE_DIR / f"{ck}.json"
        if use_cache and cache_path.exists():
            log.debug("bing_cache_hit", query=query[:60])
            data = json.loads(cache_path.read_text())
            return [Document(**d) for d in data]

        # ── Step 1: Responses API discovers URLs ───────────────────────
        async with self._semaphore:
            log.info("bing_search", query=query[:60], max_results=max_results)
            prompt = (
                f"Search the web for: {query}\n\n"
                f"Return up to {max_results} relevant results. "
                f"For each result, provide the title, URL, and a detailed "
                f"summary of the key content (2-3 sentences). "
                f"Cite all sources with URLs."
            )
            response = None
            for attempt in range(5):
                try:
                    response = await self._client.responses.create(
                        model=self._model,
                        tools=[{"type": "web_search_preview"}],
                        input=prompt,
                    )
                    break
                except (ConnectionError, OSError) as e:
                    # BrokenPipeError, ConnectionResetError, etc.
                    wait = min(2 ** attempt + random.uniform(0, 2), 30)
                    log.warning("bing_connection_retry", error=str(e),
                                attempt=attempt + 1, wait=f"{wait:.1f}s")
                    await asyncio.sleep(wait)
                except Exception as e:
                    log.warning("bing_exception", error=str(e))
                    return []
            if response is None:
                log.warning("bing_retries_exhausted", query=query[:60])
                return []

        # The full grounded synthesis from the model
        full_text = response.output_text or ""

        # Extract URLs and their synthesis context from annotations
        url_info: Dict[str, Dict] = {}  # url -> {title, context_paragraphs}

        for item in response.output:
            if item.type != "message":
                continue
            content_data = item.model_dump() if hasattr(item, "model_dump") else {}
            for content_block in content_data.get("content", []):
                block_text = content_block.get("text", "")
                # Split into paragraphs for context extraction
                paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
                para_offsets: List[tuple] = []
                pos = 0
                for para in paragraphs:
                    idx = block_text.find(para, pos)
                    if idx >= 0:
                        para_offsets.append((idx, idx + len(para), para))
                        pos = idx + len(para)

                for ann in content_block.get("annotations", []):
                    if ann.get("type") != "url_citation":
                        continue
                    url = ann.get("url", "")
                    if not url:
                        continue

                    # Find the paragraph containing this citation
                    cite_pos = ann.get("start_index", 0)
                    context_para = ""
                    for p_start, p_end, p_text in para_offsets:
                        if p_start <= cite_pos <= p_end:
                            context_para = p_text
                            break
                    if not context_para:
                        start = max(0, cite_pos - 250)
                        end = min(len(block_text), cite_pos + 250)
                        context_para = block_text[start:end].strip()

                    if url not in url_info:
                        url_info[url] = {
                            "title": ann.get("title", ""),
                            "contexts": [],
                        }
                    if context_para and context_para not in url_info[url]["contexts"]:
                        url_info[url]["contexts"].append(context_para)

        # ── Step 2: Fetch full page content for each URL ───────────────
        urls_to_fetch = list(url_info.keys())[:self._max_page_fetch]
        fetched_content: Dict[str, str] = {}

        if self._fetch_pages and urls_to_fetch:
            log.info("bing_fetching_pages", count=len(urls_to_fetch))
            fetched_docs = await self._extractor.extract_batch(
                urls_to_fetch, max_concurrent=5
            )
            for doc in fetched_docs:
                fetched_content[doc.url] = doc.content
            log.info(
                "bing_pages_fetched",
                attempted=len(urls_to_fetch),
                succeeded=len(fetched_content),
            )

        # ── Step 3: Build final documents ──────────────────────────────
        docs: List[Document] = []

        for url, info in url_info.items():
            # Use full page content if fetched, otherwise fall back to
            # the synthesis context paragraph from the Responses API
            page_content = fetched_content.get(url, "")
            synthesis_context = "\n\n".join(info["contexts"])

            if page_content:
                content = page_content
                source_type = SourceType.WEB
            elif synthesis_context:
                content = synthesis_context
                source_type = SourceType.WEB
            else:
                continue  # skip docs with no content at all

            docs.append(Document(
                id=_cache_key(url),
                title=info["title"],
                content=content,
                url=url,
                source_type=source_type,
                metadata={
                    "search_query": query,
                    "content_source": "page_fetch" if page_content else "synthesis",
                    "page_content_len": len(page_content),
                    "synthesis_context_len": len(synthesis_context),
                },
            ))

        # Also add the full synthesis as its own document
        if full_text.strip():
            docs.insert(0, Document(
                id=_cache_key(f"synthesis:{query}"),
                title=f"Web Search Synthesis: {query[:80]}",
                content=full_text,
                url="",
                source_type=SourceType.WEB,
                metadata={
                    "search_query": query,
                    "is_synthesis": True,
                    "content_source": "synthesis",
                },
            ))

        # Save cache
        if use_cache and docs:
            cache_path.write_text(
                json.dumps([d.model_dump(mode="json") for d in docs], default=str)
            )

        log.info("bing_results", query=query[:60], count=len(docs),
                 pages_fetched=len(fetched_content))
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
                log.warning("bing_batch_error", error=str(result))
                continue
            for doc in result:
                key = doc.url or doc.id
                if key not in seen_urls:
                    seen_urls.add(key)
                    docs.append(doc)

        return docs
