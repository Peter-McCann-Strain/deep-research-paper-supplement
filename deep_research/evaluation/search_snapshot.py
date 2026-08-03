"""Search result snapshots for evaluation reproducibility.

Captures and replays search results so that evaluation runs can be
reproduced exactly, even if live search APIs return different results
over time.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class SearchResultSnapshot:
    """Snapshot of search results for a single query."""
    query: str
    search_queries: list[str]
    results: list[dict]  # list of serialized Document dicts
    timestamp: str = ""
    source_types: list[str] = field(default_factory=list)
    n_results: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        self.n_results = len(self.results)


@dataclass
class EvalRunSnapshot:
    """Complete search snapshot for an evaluation run."""
    run_id: str
    pattern: str
    query_id: str
    snapshots: list[SearchResultSnapshot] = field(default_factory=list)
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def total_results(self) -> int:
        return sum(s.n_results for s in self.snapshots)


class SnapshotStore:
    """Manages saving and loading search result snapshots."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, pattern: str, query_id: str) -> Path:
        return self.snapshot_dir / pattern / f"{query_id}.json"

    def save(self, snapshot: EvalRunSnapshot) -> Path:
        """Save a run snapshot to disk.

        Args:
            snapshot: The snapshot to save.

        Returns:
            Path where the snapshot was saved.
        """
        path = self._snapshot_path(snapshot.pattern, snapshot.query_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": snapshot.run_id,
            "pattern": snapshot.pattern,
            "query_id": snapshot.query_id,
            "timestamp": snapshot.timestamp,
            "metadata": snapshot.metadata,
            "total_results": snapshot.total_results,
            "snapshots": [asdict(s) for s in snapshot.snapshots],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        log.debug("snapshot_saved", path=str(path), n_snapshots=len(snapshot.snapshots))
        return path

    def load(self, pattern: str, query_id: str) -> EvalRunSnapshot | None:
        """Load a saved snapshot.

        Args:
            pattern: Pattern name.
            query_id: Query identifier.

        Returns:
            The loaded snapshot, or None if not found.
        """
        path = self._snapshot_path(pattern, query_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            snapshots = [
                SearchResultSnapshot(**s) for s in data.get("snapshots", [])
            ]
            return EvalRunSnapshot(
                run_id=data["run_id"],
                pattern=data["pattern"],
                query_id=data["query_id"],
                snapshots=snapshots,
                timestamp=data.get("timestamp", ""),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("snapshot_load_error", path=str(path), error=str(e))
            return None

    def exists(self, pattern: str, query_id: str) -> bool:
        """Check if a snapshot exists."""
        return self._snapshot_path(pattern, query_id).exists()

    def list_snapshots(self, pattern: str | None = None) -> list[tuple[str, str]]:
        """List all available snapshots as (pattern, query_id) tuples.

        Args:
            pattern: If provided, only list snapshots for this pattern.

        Returns:
            List of (pattern, query_id) tuples.
        """
        results: list[tuple[str, str]] = []
        if pattern:
            pattern_dir = self.snapshot_dir / pattern
            if pattern_dir.exists():
                for f in sorted(pattern_dir.glob("*.json")):
                    results.append((pattern, f.stem))
        else:
            for pattern_dir in sorted(self.snapshot_dir.iterdir()):
                if pattern_dir.is_dir():
                    for f in sorted(pattern_dir.glob("*.json")):
                        results.append((pattern_dir.name, f.stem))
        return results

    def snapshot_stats(self) -> dict[str, Any]:
        """Get summary statistics of all stored snapshots.

        Returns:
            Dict with counts and size info.
        """
        all_snapshots = self.list_snapshots()
        total_size = 0
        by_pattern: dict[str, int] = {}

        for pattern, query_id in all_snapshots:
            path = self._snapshot_path(pattern, query_id)
            if path.exists():
                total_size += path.stat().st_size
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1

        return {
            "total_snapshots": len(all_snapshots),
            "by_pattern": by_pattern,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


def capture_search_results(
    query: str,
    search_queries: list[str],
    documents: list,
) -> SearchResultSnapshot:
    """Capture search results as a snapshot.

    Args:
        query: The original research query.
        search_queries: The search queries that were issued.
        documents: List of Document objects (will be serialized).

    Returns:
        SearchResultSnapshot with serialized results.
    """
    results = []
    source_types = set()

    for doc in documents:
        if hasattr(doc, "model_dump"):
            results.append(doc.model_dump(mode="json"))
        elif hasattr(doc, "__dict__"):
            results.append({k: str(v) for k, v in doc.__dict__.items()})
        else:
            results.append({"content": str(doc)})

        if hasattr(doc, "source_type"):
            source_types.add(str(doc.source_type))

    return SearchResultSnapshot(
        query=query,
        search_queries=search_queries,
        results=results,
        source_types=sorted(source_types),
    )
