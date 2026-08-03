"""Graph expansion: decide whether a completed node should spawn child nodes.

After a node is executed, the expander examines its answer and decides whether
deeper investigation would improve the research.  Expansion is suppressed when
the graph has reached its depth or node-count limits.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools.llm_caller import LLMCaller

from .graph import GraphNode, ResearchGraph

log = structlog.get_logger()

EXPANSION_PROMPT = """A research sub-question has been answered.  Decide whether deeper investigation is needed.

Original Research Query: {original_query}

Sub-Question: {question}
Answer Summary (truncated): {answer}

Existing questions already in the research graph:
{existing_questions}

Based on this answer, are there important follow-up questions that would \
meaningfully deepen the research?  Only suggest follow-ups if the answer \
reveals significant new angles NOT already covered by the existing questions.

Return ONLY valid JSON:
{{"needs_expansion": true, "follow_up_questions": ["question 1", "question 2"]}}
or
{{"needs_expansion": false, "follow_up_questions": []}}

Rules:
- Suggest at most {max_children} follow-ups.
- Return empty list if the answer is already comprehensive.
- Do NOT duplicate questions that are already in the graph.
"""


async def expand_node(
    llm: LLMCaller,
    node: GraphNode,
    graph: ResearchGraph,
    original_query: str,
    max_children: int = 3,
    max_total_nodes: int = 20,
    max_depth: int = 3,
) -> List[GraphNode]:
    """Decide if *node* should spawn child nodes and create them.

    Returns the list of newly created child nodes (may be empty).
    """
    # ── Guard: don't expand beyond limits ────────────────────────────
    if node.depth >= max_depth - 1:
        log.debug("expand_skip_depth", node_id=node.id, depth=node.depth)
        return []
    if graph.total_nodes >= max_total_nodes:
        log.debug("expand_skip_max_nodes", node_id=node.id, total=graph.total_nodes)
        return []

    # ── Guard: nodes with very short answers are unlikely to benefit ──
    if len(node.answer) < 50:
        return []

    existing_questions = "\n".join(
        f"  - {n.question}" for n in graph.nodes.values()
    )

    # How many new nodes we can still create
    remaining_budget = max_total_nodes - graph.total_nodes
    effective_max_children = min(max_children, remaining_budget)
    if effective_max_children <= 0:
        return []

    try:
        result = await llm.complete_json(
            EXPANSION_PROMPT.format(
                original_query=original_query,
                question=node.question,
                answer=node.answer[:5000],
                existing_questions=existing_questions,
                max_children=effective_max_children,
            ),
            model=DEFAULT_MODEL,
            max_tokens=1024,
            temperature=0.3,
        )

        needs_expansion = result.get("needs_expansion", False)
        follow_ups = result.get("follow_up_questions", [])

        if not needs_expansion or not isinstance(follow_ups, list) or not follow_ups:
            log.info("expand_none", node_id=node.id)
            return []

        # Create child nodes
        new_nodes: List[GraphNode] = []
        for q in follow_ups[:effective_max_children]:
            if isinstance(q, str) and len(q.strip()) > 5:
                child = graph.add_node(question=q.strip(), parent_id=node.id)
                new_nodes.append(child)

        log.info(
            "expand_done",
            node_id=node.id,
            new_children=len(new_nodes),
            total_nodes=graph.total_nodes,
        )
        return new_nodes

    except Exception as exc:
        log.warning("expand_failed", node_id=node.id, error=str(exc))
        return []
