"""Research graph data structure for dynamic sub-question decomposition.

Each node represents a sub-question.  Root nodes are created during initial
decomposition; child nodes are added dynamically as answers reveal new angles.
The graph supports bottom-up traversal for synthesis and serialisation for
checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class NodeStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GraphNode:
    """A single node in the research graph (one sub-question)."""

    id: str                                          # "node_0", "node_1", ...
    question: str                                    # The sub-question text
    parent_id: Optional[str] = None                  # None for root nodes
    depth: int = 0
    status: NodeStatus = NodeStatus.PENDING
    answer: str = ""                                 # Summary answer after execution
    extractions: list = field(default_factory=list)  # SourceExtraction objects
    children_ids: List[str] = field(default_factory=list)
    search_queries_used: List[str] = field(default_factory=list)
    n_docs_found: int = 0


class ResearchGraph:
    """Directed acyclic graph of research sub-questions.

    Nodes are added incrementally — first root questions, then children as the
    graph expander decides deeper investigation is warranted.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self._next_id: int = 0

    # ── Mutation ─────────────────────────────────────────────────────────

    def add_node(
        self,
        question: str,
        parent_id: Optional[str] = None,
    ) -> GraphNode:
        """Create a new node and wire it to its parent (if any)."""
        node_id = f"node_{self._next_id}"
        self._next_id += 1
        depth = 0 if parent_id is None else self.nodes[parent_id].depth + 1
        node = GraphNode(id=node_id, question=question, parent_id=parent_id, depth=depth)
        self.nodes[node_id] = node
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)
        return node

    # ── Queries ──────────────────────────────────────────────────────────

    def get_leaves(self, status: Optional[NodeStatus] = None) -> List[GraphNode]:
        """Return leaf nodes (no children), optionally filtered by status."""
        leaves = [n for n in self.nodes.values() if not n.children_ids]
        if status is not None:
            leaves = [n for n in leaves if n.status == status]
        return leaves

    def get_pending_leaves(self) -> List[GraphNode]:
        """Convenience shortcut for pending leaf nodes."""
        return self.get_leaves(NodeStatus.PENDING)

    def bottom_up_traversal(self) -> List[GraphNode]:
        """Return all nodes ordered deepest-first, then by id for determinism."""
        return sorted(self.nodes.values(), key=lambda n: (-n.depth, n.id))

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise the graph for checkpointing (answers truncated)."""
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "question": n.question,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "status": n.status.value,
                    "answer": n.answer[:500],
                    "children_ids": n.children_ids,
                    "search_queries_used": n.search_queries_used,
                    "n_docs_found": n.n_docs_found,
                }
                for nid, n in self.nodes.items()
            },
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
        }
