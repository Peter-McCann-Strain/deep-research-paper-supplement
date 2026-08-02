"""Shared data models used by public pattern, tool, and evaluation modules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    URL_EXTRACT = "url_extract"


class Document(BaseModel):
    """A retrieved source document."""

    id: str = ""
    title: str = ""
    content: str = ""
    url: str = ""
    source_type: SourceType = SourceType.WEB
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def short_id(self) -> str:
        return self.id[:12] if self.id else self.url[:40]


class Citation(BaseModel):
    """A citation linking a claim to a source."""

    claim: str = ""
    source_id: str = ""
    source_title: str = ""
    source_url: str = ""
    relevance_score: float = 0.0


class SubQuery(BaseModel):
    """A decomposed sub-query."""

    query: str
    intent: str = ""
    priority: int = 1


class Section(BaseModel):
    """A section of a generated research report."""

    title: str
    content: str
    citations: list[Citation] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final output of any research pattern."""

    query: str
    title: str = ""
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    pattern_name: str = ""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def full_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}\n")
        if self.abstract:
            parts.append(f"## Abstract\n{self.abstract}\n")
        for section in self.sections:
            parts.append(f"## {section.title}\n{section.content}\n")
        if self.citations:
            parts.append("## References\n")
            for idx, citation in enumerate(self.citations, 1):
                parts.append(f"[{idx}] {citation.source_title} - {citation.source_url}")
        return "\n".join(parts)


class LLMUsage(BaseModel):
    """Single LLM call usage record."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    call_type: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Perspective(BaseModel):
    """A research perspective used by STORM-style patterns."""

    name: str
    description: str
    focus_areas: list[str] = Field(default_factory=list)


class TopicCluster(BaseModel):
    """A cluster of related information used by MERIDIAN-style patterns."""

    topic: str
    summary: str = ""
    source_ids: list[str] = Field(default_factory=list)
    importance: float = 0.0


class WDAllocation(BaseModel):
    """Width-depth budget allocation."""

    step: int
    width_budget: float
    depth_budget: float
    width_workers: int
    depth_iterations: int


class ToolCall(BaseModel):
    """One step of a pattern's tool/reasoning trace."""

    step_idx: int
    tool: str
    input_args: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    n_results: int | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_seconds: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcessTrace(BaseModel):
    """A pattern's full tool-call sequence for one query."""

    pattern_name: str
    query: str
    query_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    n_search_queries: int = 0
    n_unique_urls_visited: int = 0
    n_iterations: int = 0
    final_report_word_count: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def append(self, **kwargs: Any) -> ToolCall:
        """Build and append a ``ToolCall`` at the next step index."""
        kwargs.setdefault("step_idx", len(self.tool_calls))
        call = ToolCall(**kwargs)
        self.tool_calls.append(call)
        if call.tool in {"search", "academic_search"}:
            self.n_search_queries += 1
        return call
