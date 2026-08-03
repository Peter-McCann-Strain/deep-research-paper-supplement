"""Accumulated agent state for the reactive interleaved loop.

Tracks everything the agent has learned and produced across iterations:
search queries executed, source extractions, drafted report sections,
and gap-analysis reflections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from deep_research.tools.source_extractor import SourceExtraction


@dataclass
class AgentState:
    """Mutable state carried across reasoning-loop iterations."""

    query: str
    searches_done: List[str] = field(default_factory=list)
    extractions: List[SourceExtraction] = field(default_factory=list)
    draft_sections: Dict[str, str] = field(default_factory=dict)  # title -> content
    reflections: List[str] = field(default_factory=list)
    iteration: int = 0
    total_search_queries: int = 0

    # ── Deduplication helpers ────────────────────────────────────────────

    _seen_urls: set = field(default_factory=set, repr=False)

    def add_extractions(self, new: List[SourceExtraction]) -> int:
        """Merge *new* extractions, deduplicating by URL.

        Returns the number of genuinely new extractions added.
        """
        added = 0
        for ext in new:
            key = ext.url or ext.doc_id
            if key and key in self._seen_urls:
                continue
            if key:
                self._seen_urls.add(key)
            self.extractions.append(ext)
            added += 1
        return added

    def record_searches(self, queries: List[str]) -> None:
        """Record search queries that were executed."""
        self.searches_done.extend(queries)
        self.total_search_queries += len(queries)

    # ── Serialisation for checkpoints ────────────────────────────────────

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "iteration": self.iteration,
            "searches_done": self.searches_done,
            "total_search_queries": self.total_search_queries,
            "n_extractions": len(self.extractions),
            "extractions": [e.to_evidence_dict() for e in self.extractions],
            "draft_sections": self.draft_sections,
            "reflections": self.reflections,
        }

    # ── Context summaries for the agent prompt ───────────────────────────

    def search_queries_summary(self, max_display: int = 10) -> str:
        """Short summary of searches done so far."""
        if not self.searches_done:
            return "none"
        shown = self.searches_done[-max_display:]
        text = "; ".join(f'"{q}"' for q in shown)
        if len(self.searches_done) > max_display:
            text = f"... and {len(self.searches_done) - max_display} earlier queries; " + text
        return text

    def draft_section_titles(self) -> str:
        if not self.draft_sections:
            return "none"
        return ", ".join(self.draft_sections.keys())

    def last_reflection(self) -> str:
        if not self.reflections:
            return "none"
        return self.reflections[-1]
