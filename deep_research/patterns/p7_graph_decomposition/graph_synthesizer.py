"""Stage 3: Traverse the completed research graph and synthesise a final report.

Bottom-up traversal ensures that child findings feed into the parent context,
producing a coherent narrative that respects the graph structure.
"""

from __future__ import annotations

from typing import List

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import SourceExtraction, format_extractions_as_evidence
from deep_research.tools.llm_caller import LLMCaller

from .graph import ResearchGraph

log = structlog.get_logger()

SYNTHESIS_PROMPT = """Write a comprehensive, well-structured research report based on the \
following structured research findings.

Research Query: {query}

Research Graph Findings (ordered from deepest sub-questions to root questions):
{node_summaries}

Source Evidence:
{evidence}

Requirements:
- Start with a single # Title that captures the research topic
- Include a ## Abstract (150-250 words) summarising key findings
- Organise the body into logical ## sections — do NOT mirror the sub-question \
  structure directly; instead synthesise across findings into a coherent narrative
- End with ## References listing all cited sources
- Use inline citations [1], [2], etc. that reference the numbered sources above
- Synthesise across findings — identify themes, tensions, and connections
- Where sub-question answers conflict, discuss both sides
- Aim for 2000-4000 words (excluding references)
- Write in an objective, academic tone
"""


async def synthesize_graph(
    llm: LLMCaller,
    graph: ResearchGraph,
    query: str,
    all_extractions: List[SourceExtraction],
) -> str:
    """Traverse the completed graph and synthesise into a markdown report."""
    log.info(
        "synthesis_start",
        total_nodes=graph.total_nodes,
        max_depth=graph.max_depth,
        n_extractions=len(all_extractions),
    )

    # ── Build structured summary from bottom-up traversal ────────────
    traversal = graph.bottom_up_traversal()

    node_summaries_parts: List[str] = []
    for node in traversal:
        children_text = ""
        if node.children_ids:
            child_lines = []
            for cid in node.children_ids:
                child_node = graph.nodes[cid]
                child_lines.append(
                    f"  - {child_node.question}: {child_node.answer[:5000]}"
                )
            children_text = "\nSub-findings:\n" + "\n".join(child_lines)

        depth_label = "Root" if node.depth == 0 else f"Depth-{node.depth}"
        node_summaries_parts.append(
            f"### [{depth_label}] {node.question}\n"
            f"{node.answer[:8000]}"
            f"{children_text}"
        )

    node_summaries = "\n\n".join(node_summaries_parts)

    # ── Format source evidence ───────────────────────────────────────
    evidence = format_extractions_as_evidence(all_extractions)

    # ── Final synthesis call ─────────────────────────────────────────
    report_md = await llm.complete(
        SYNTHESIS_PROMPT.format(
            query=query,
            node_summaries=node_summaries,
            evidence=evidence,
        ),
        model=DEFAULT_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )

    log.info("synthesis_done", report_length=len(report_md))
    return report_md
