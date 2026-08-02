"""Shared tool layer — re-exports for convenience."""

from deep_research.tools.cost_tracker import CostTracker, BudgetExceeded
from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.web_search import WebSearcher
from deep_research.tools.bing_search import BingSearcher
from deep_research.tools.academic_search import AcademicSearcher
from deep_research.tools.url_extractor import URLExtractor
from deep_research.tools.source_extractor import (
    SourceExtractor,
    SourceExtraction,
    format_extractions_as_evidence,
    format_summaries_as_evidence,  # backward-compat alias
)
from deep_research.tools.state_manager import StateManager


def get_web_searcher(backend: str | None = None):
    """Return the configured web searcher (BingSearcher or WebSearcher).

    Args:
        backend: Override the SEARCH_BACKEND env config. Pass "bing" or "tavily"
                 to force a specific retriever (used by Protocol A and other
                 backend-comparison experiments). Defaults to env-configured.
    """
    if backend is None:
        from deep_research.config import SEARCH_BACKEND
        backend = SEARCH_BACKEND
    if backend == "oracle":
        from deep_research.tools.oracle_search import OracleSearcher
        return OracleSearcher()
    if backend == "bing":
        return BingSearcher()
    return WebSearcher()


__all__ = [
    "CostTracker", "BudgetExceeded",
    "LLMCaller",
    "WebSearcher",
    "BingSearcher",
    "get_web_searcher",
    "AcademicSearcher",
    "URLExtractor",
    "SourceExtractor", "SourceExtraction",
    "format_extractions_as_evidence", "format_summaries_as_evidence",
    "StateManager",
]
