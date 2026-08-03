"""Pattern 7: Dynamic Graph Decomposition Pipeline (MindSearch-inspired).

Unlike P1 which decomposes into a flat list of 25 sub-queries upfront, P7
starts with a few root questions and dynamically grows the research graph as
answers reveal new connections.

Flow:
    Stage 1: Query -> LLM generates 3-5 root sub-questions (graph nodes)
    Stage 2: Iterative graph execution + expansion
        - Execute all pending leaf nodes in parallel (search + extract + summarise)
        - For each completed node, the graph expander decides: add 0-3 child nodes?
        - New child nodes become the next iteration's leaves
        - Repeat until no pending leaves, node limit, or depth limit
    Stage 3: Bottom-up graph synthesis -> final research report
"""

from __future__ import annotations

import asyncio
import time

import structlog

from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtraction,
    SourceExtractor,
    StateManager,
    URLExtractor,
    get_web_searcher,
)
from deep_research.types import ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

from .graph import ResearchGraph, NodeStatus
from .graph_expander import expand_node
from .graph_synthesizer import synthesize_graph
from .initial_decomposer import decompose_query
from .node_executor import execute_node

log = structlog.get_logger()


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    **kwargs,
) -> ResearchReport:
    """Execute the full dynamic graph decomposition pipeline.

    Args:
        query: The research question.
        budget_usd: Maximum budget in USD.
        **kwargs: Optional overrides —
            max_depth (int, default 3): Maximum graph depth.
            max_nodes (int, default 20): Maximum total graph nodes.
            n_roots (int, default 4): Number of initial root sub-questions.
            skip_expansion (bool, default False): If True, only execute root
                nodes with no graph expansion (useful for ablation).
    """
    t0 = time.perf_counter()

    max_depth: int = kwargs.get("max_depth", 3)
    max_nodes: int = kwargs.get("max_nodes", 20)
    n_roots: int = kwargs.get("n_roots", 4)
    skip_expansion: bool = kwargs.get("skip_expansion", False)

    # ── Initialise tools ─────────────────────────────────────────────
    tracker = CostTracker(budget_usd=budget_usd)
    llm = LLMCaller(cost_tracker=tracker)
    state = StateManager("p7_graph_decomposition")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    graph = ResearchGraph()
    trace = ProcessTrace(pattern_name="p7_graph_decomposition", query=query, query_id=kwargs.get("query_id", ""))

    log.info(
        "p7_start",
        query=query[:80],
        max_depth=max_depth,
        max_nodes=max_nodes,
        n_roots=n_roots,
        skip_expansion=skip_expansion,
    )

    # ── Stage 1: Initial decomposition ───────────────────────────────
    log.info("stage_1_decompose")
    tokens_before_decompose = tracker.total_tokens
    root_questions = await decompose_query(llm, query, n_roots=n_roots)
    for q in root_questions:
        graph.add_node(q)
    trace.append(tool="decompose",
                 input_args={"n_roots": n_roots},
                 output_summary=f"{len(root_questions)} root questions",
                 n_results=len(root_questions),
                 tokens_used=tracker.total_tokens - tokens_before_decompose)

    state.save("decomposition", {
        "root_questions": root_questions,
        "graph": graph.to_dict(),
    })

    # ── Stage 2: Iterative execution + expansion ─────────────────────
    log.info("stage_2_graph_execution")
    iteration = 0
    # Limit concurrent node execution to avoid lxml/trafilatura segfaults
    # under heavy parallel HTML parsing load
    node_semaphore = asyncio.Semaphore(3)

    async def _execute_with_limit(node):
        async with node_semaphore:
            return await execute_node(
                node, llm, web, academic, url_extractor, source_extractor, query,
            )

    while True:
        pending = graph.get_pending_leaves()
        if not pending:
            log.info("graph_loop_done", reason="no_pending_leaves", iteration=iteration)
            break
        if graph.total_nodes >= max_nodes:
            log.info("graph_loop_done", reason="max_nodes_reached", iteration=iteration,
                     total_nodes=graph.total_nodes)
            # Still execute any remaining pending nodes before stopping expansion
            pass

        log.info(
            "graph_iteration",
            iteration=iteration,
            pending=len(pending),
            total_nodes=graph.total_nodes,
            max_depth_seen=graph.max_depth,
        )

        # Execute pending leaves with concurrency limit
        tokens_before_iter = tracker.total_tokens
        tasks = [_execute_with_limit(node) for node in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any unexpected exceptions from gather
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(
                    "node_gather_exception",
                    node_id=pending[i].id,
                    error=str(result),
                )

        n_extractions_this_iter = sum(len(n.extractions) for n in pending)
        trace.append(tool="search",
                     input_args={"iteration": iteration, "n_pending_nodes": len(pending)},
                     output_summary=f"executed {len(pending)} nodes; {n_extractions_this_iter} extractions",
                     n_results=n_extractions_this_iter,
                     tokens_used=tracker.total_tokens - tokens_before_iter)

        state.save(f"iteration_{iteration}", {"graph": graph.to_dict()})

        # Expand completed nodes (unless expansion is disabled or we hit the cap)
        if not skip_expansion and graph.total_nodes < max_nodes:
            tokens_before_expand = tracker.total_tokens
            nodes_before = graph.total_nodes
            for node in pending:
                if node.status == NodeStatus.COMPLETED:
                    await expand_node(
                        llm, node, graph, query,
                        max_total_nodes=max_nodes,
                        max_depth=max_depth,
                    )
            new_nodes = graph.total_nodes - nodes_before
            trace.append(tool="decompose",
                         input_args={"iteration": iteration, "stage": "expand"},
                         output_summary=f"added {new_nodes} child nodes (total {graph.total_nodes})",
                         n_results=new_nodes,
                         tokens_used=tracker.total_tokens - tokens_before_expand)

        iteration += 1

        # Safety: if we somehow get stuck, break after many iterations
        if iteration > max_nodes:
            log.warning("graph_loop_safety_break", iteration=iteration)
            break

    # Execute any final pending leaves that were added in the last expansion
    final_pending = graph.get_pending_leaves()
    if final_pending:
        log.info("graph_final_execution", pending=len(final_pending))
        tasks = [_execute_with_limit(node) for node in final_pending]
        await asyncio.gather(*tasks, return_exceptions=True)
        state.save("final_execution", {"graph": graph.to_dict()})

    # ── Stage 3: Synthesis ───────────────────────────────────────────
    log.info("stage_3_synthesis")

    # Collect and deduplicate all extractions across nodes
    all_extractions: list[SourceExtraction] = []
    seen_doc_ids: set[str] = set()
    for node in graph.nodes.values():
        for ext in node.extractions:
            if ext.doc_id not in seen_doc_ids:
                seen_doc_ids.add(ext.doc_id)
                all_extractions.append(ext)

    tokens_before_gen = tracker.total_tokens
    report_md = await synthesize_graph(llm, graph, query, all_extractions)
    trace.append(tool="generate",
                 input_args={"n_nodes": graph.total_nodes, "n_extractions": len(all_extractions)},
                 output_summary=f"{len(report_md)}-char report from graph",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    # ── Assemble ResearchReport ──────────────────────────────────────
    elapsed = time.perf_counter() - t0

    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=all_extractions,
        pattern_name="p7_graph_decomposition",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
        elapsed_seconds=elapsed,
    )

    # ── Attach metadata ──────────────────────────────────────────────
    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = [n.question for n in graph.nodes.values()]
    report.metadata["search_queries_sent"] = [
        q for n in graph.nodes.values() for q in n.search_queries_used
    ]
    report.metadata["n_documents_retrieved"] = sum(
        n.n_docs_found for n in graph.nodes.values()
    )
    report.metadata["n_extractions"] = len(all_extractions)
    report.metadata["graph"] = graph.to_dict()
    report.metadata["graph_depth"] = graph.max_depth
    report.metadata["graph_total_nodes"] = graph.total_nodes
    report.metadata["graph_iterations"] = iteration
    report.metadata["skip_expansion"] = skip_expansion

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed_seconds": elapsed,
        "graph": graph.to_dict(),
    })

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len({ext.url for ext in all_extractions if ext.url})
    trace.n_iterations = iteration
    state.save("trace", trace.model_dump(mode="json"))

    log.info(
        "p7_complete",
        cost=f"${tracker.total_cost:.4f}",
        tokens=tracker.total_tokens,
        sections=len(report.sections),
        graph_nodes=graph.total_nodes,
        graph_depth=graph.max_depth,
        iterations=iteration,
        elapsed=f"{elapsed:.1f}s",
    )

    return report
