"""Oracle / fixed-corpus retriever.

Returns an IDENTICAL frozen source set per query to every pattern, regardless of the
sub-query string the pattern emits. This isolates *synthesis* from *retrieval variance*:
the evidence is held constant, so any remaining cross-pattern difference is attributable
to orchestration, not to what each pattern happened to retrieve.

Wiring (mirrors the Protocol-A backend swap):
  - `SEARCH_BACKEND=oracle` selects this searcher via `get_web_searcher()`.
  - `ORACLE_CORPUS_PATH` points at a JSON file `{query_id: [Document-dict, ...]}`
    (built per tier by `scripts/build_oracle_corpus.py`).
  - `ORACLE_QUERY_ID` is set per run by the runner so the searcher knows which
    query's frozen corpus to serve.
  - `AcademicSearcher` returns `[]` under oracle mode (academic evidence is already
    pooled into the frozen corpus), so ALL evidence comes from the fixed set.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import structlog

from deep_research.config import DATA_DIR
from deep_research.types import Document, SourceType

log = structlog.get_logger()


@lru_cache(maxsize=4)
def _load_corpus(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning("oracle_corpus_missing", path=path)
        return {}
    return json.loads(p.read_text())


class OracleSearcher:
    """Fixed-corpus searcher: every `search()` returns the same frozen Documents."""

    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = corpus_path or os.environ.get(
            "ORACLE_CORPUS_PATH", str(DATA_DIR / "oracle_corpus_t1.json")
        )
        self.max_docs = int(os.environ.get("ORACLE_MAX_DOCS", "30"))

    def _docs(self) -> List[Document]:
        # Read query_id dynamically so it always reflects the current run.
        query_id = os.environ.get("ORACLE_QUERY_ID", "")
        corpus = _load_corpus(self.corpus_path)
        raw = corpus.get(query_id, [])
        if not raw:
            log.warning("oracle_no_docs", query_id=query_id, corpus=self.corpus_path)
        docs: List[Document] = []
        for d in raw[: self.max_docs]:
            try:
                docs.append(Document(**d))
            except Exception:
                docs.append(
                    Document(
                        id=str(d.get("id", "")),
                        title=str(d.get("title", "")),
                        content=str(d.get("content", "")),
                        url=str(d.get("url", "")),
                        source_type=SourceType(d.get("source_type", "web"))
                        if d.get("source_type") in {s.value for s in SourceType}
                        else SourceType.WEB,
                        metadata={"oracle": True},
                    )
                )
        return docs

    async def search(self, query: str, max_results: int = 10, **kwargs) -> List[Document]:
        # The sub-query is intentionally ignored: the oracle serves the same fixed
        # evidence to every call so all patterns see an identical source set.
        return self._docs()

    async def search_batch(
        self, queries: List[str], max_results_per: int = 5, **kwargs
    ) -> List[Document]:
        return self._docs()
