"""URL content extraction with trafilatura + BS4 fallback."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Optional

import structlog
import trafilatura
from bs4 import BeautifulSoup
from trafilatura.settings import use_config

from deep_research.types import Document, SourceType

log = structlog.get_logger()

# Hard wall-clock ceiling for a single blocking fetch+extract. trafilatura's
# fetch_url runs synchronous urllib3 I/O that asyncio.wait_for/to_thread CANNOT
# cancel (the executor thread keeps running even after the await is cancelled).
# A slow / trickling / non-responding server therefore wedges the whole pipeline
# indefinitely. We bound it two ways:
#   1. a socket-level DOWNLOAD_TIMEOUT passed to trafilatura (per read), and
#   2. a hard future.result(timeout=...) on a DAEMON ThreadPoolExecutor so a
#      genuinely hung fetch is ABANDONED (we move on; the orphan thread cannot
#      block process exit and the source is simply skipped — safe + corpus-faithful).
_FETCH_HARD_TIMEOUT_S = 30        # hard wall-clock per URL (abandon if exceeded)
_TRAFILATURA_SOCKET_TIMEOUT_S = 20  # socket read timeout handed to trafilatura

# Daemon pool: threads here never block interpreter shutdown, so an abandoned
# (still-hung) fetch thread is harmless. Sized for the batch concurrency.
_FETCH_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="url-fetch"
)

# Trafilatura config with a bounded socket timeout (default config has 30s but
# leaves no caller-visible bound; we set it explicitly and lower).
_TRAFI_CONFIG = use_config()
_TRAFI_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(_TRAFILATURA_SOCKET_TIMEOUT_S))


class URLExtractor:
    """Extract clean text content from URLs."""

    async def extract(self, url: str, timeout: int = 15) -> Optional[Document]:
        """Download and extract content from a URL."""
        try:
            loop = asyncio.get_running_loop()
            fut = _FETCH_POOL.submit(self._extract_sync, url, timeout)
            try:
                # Hard wall-clock bound. asyncio.wait_for around an awaited
                # to_thread cannot interrupt a blocking syscall; a real future
                # with a timeout abandons the hung worker and lets us continue.
                content = await asyncio.wait_for(
                    asyncio.wrap_future(fut, loop=loop),
                    timeout=_FETCH_HARD_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                fut.cancel()  # best-effort; orphan thread is a daemon, harmless
                log.warning("url_extract_timeout", url=url[:80],
                            timeout_s=_FETCH_HARD_TIMEOUT_S)
                return None
            if not content:
                return None

            return Document(
                id=url,
                title=self._extract_title(content) or url,
                content=content,
                url=url,
                source_type=SourceType.URL_EXTRACT,
            )
        except Exception as e:
            log.warning("url_extract_error", url=url[:80], error=str(e))
            return None

    def _extract_sync(self, url: str, timeout: int) -> Optional[str]:
        """Synchronous extraction: trafilatura first, BS4 fallback."""
        # Try trafilatura. Pass a config carrying a bounded socket DOWNLOAD_TIMEOUT
        # so a slow server cannot block this thread indefinitely at the I/O layer.
        # (The outer extract() additionally enforces a hard wall-clock ceiling.)
        downloaded = trafilatura.fetch_url(url, config=_TRAFI_CONFIG)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if text and len(text) > 100:
                log.debug("trafilatura_ok", url=url[:60], chars=len(text))
                return text

        # BS4 fallback
        if downloaded:
            try:
                soup = BeautifulSoup(downloaded, "html.parser")
                # Remove script/style
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text and len(text) > 100:
                    log.debug("bs4_fallback_ok", url=url[:60], chars=len(text))
                    return text
            except Exception:
                pass

        return None

    def _extract_title(self, content: str) -> str:
        """Extract first line as title."""
        for line in content.split("\n"):
            line = line.strip()
            if line and len(line) > 5:
                return line[:200]
        return ""

    async def extract_batch(
        self, urls: list[str], max_concurrent: int = 5
    ) -> list[Document]:
        """Extract from multiple URLs concurrently."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _extract_one(url: str) -> Optional[Document]:
            async with sem:
                return await self.extract(url)

        results = await asyncio.gather(
            *[_extract_one(u) for u in urls], return_exceptions=True
        )
        return [r for r in results if isinstance(r, Document)]
