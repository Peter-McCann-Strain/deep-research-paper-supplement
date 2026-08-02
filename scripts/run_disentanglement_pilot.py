#!/usr/bin/env python3
"""Run retrieval-vs-synthesis disentanglement pilot cells.

This runner creates new experiment IDs only:

  - disentangle_matched_p{1,4,7}
  - disentangle_fixed_retrieval_p{1,4,7}
  - disentangle_fixed_synthesis_p{1,4,7}

It intentionally does not overwrite `base_*` or `protocol_a_*` outputs. Evidence
is cached by retrieval policy so the fixed-retrieval condition truly reuses the
same source extractions across synthesis styles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deep_research.config import DEFAULT_MODEL, MAX_COST_PER_RUN  # noqa: E402
from deep_research.patterns.p1_iterative_rag.generator import generate_report as p1_generate  # noqa: E402
from deep_research.patterns.p1_iterative_rag.query_decomposer import decompose_query as p1_decompose  # noqa: E402
from deep_research.patterns.p1_iterative_rag.retriever import Retriever as P1Retriever  # noqa: E402
from deep_research.patterns.p4_perspective_storm.perspective_discovery import generate_search_queries  # noqa: E402
from deep_research.patterns.p4_perspective_storm.pipeline import _search_and_extract as p4_search_extract  # noqa: E402
from deep_research.patterns.p7_graph_decomposition.graph import ResearchGraph  # noqa: E402
from deep_research.patterns.p7_graph_decomposition.initial_decomposer import decompose_query as p7_decompose  # noqa: E402
from deep_research.patterns.p7_graph_decomposition.node_executor import execute_node  # noqa: E402
from deep_research.tools import (  # noqa: E402
    AcademicSearcher,
    CostTracker,
    LLMCaller,
    SourceExtraction,
    SourceExtractor,
    URLExtractor,
    format_extractions_as_evidence,
    get_web_searcher,
)
from deep_research.types import Document, Perspective  # noqa: E402

RESULTS_DIR = ROOT / "results" / "experiments"
CHECKPOINT_DIR = ROOT / "checkpoints" / "experiments"
EVIDENCE_CACHE = ROOT / "checkpoints" / "disentanglement_evidence"
DEFAULT_QUERY_IDS = ROOT / "data" / "variance_stratified.json"

PATTERNS = ("p1", "p4", "p7")
CONDITIONS = ("matched", "fixed_retrieval", "fixed_synthesis")

COMMON_SYNTH_PROMPT = """You are a research analyst. Write a comprehensive, well-structured
research report answering the query using only the provided source evidence.

Research query: {query}

Source evidence:
{evidence}

Requirements:
- Start with a title (# Title) and include a ## Abstract.
- Organize the body into coherent ## sections.
- Use inline numbered citations [1], [2] tied to the evidence list.
- Include concrete dates, methods, metrics, and limitations when present.
- End with ## References.
- Aim for 2000-4000 words.

Write the report in markdown format."""

P4_STYLE_PROMPT = """You are an expert multi-perspective research synthesizer. Use the same
evidence base to write a report that explicitly integrates technical, empirical,
practitioner, and skeptical perspectives.

Research query: {query}

Source evidence:
{evidence}

Requirements:
- Start with # Title and ## Abstract.
- Present points of agreement, disagreement, evidence gaps, and confidence levels.
- Weave perspectives throughout the report instead of listing them mechanically.
- Ground claims in inline numbered citations [1], [2].
- End with ## References.
- Aim for 2000-4000 words.

Write the report in markdown format."""

P7_STYLE_PROMPT = """Write a comprehensive research report by synthesizing the evidence as
if it came from a dynamic research graph: identify root themes, sub-question
answers, tensions between branches, and cross-cutting conclusions.

Research query: {query}

Source evidence:
{evidence}

Requirements:
- Start with # Title and ## Abstract.
- Organize into logical ## sections; do not merely list sources.
- Discuss themes, tensions, unresolved gaps, and practical implications.
- Use inline numbered citations [1], [2] tied to the evidence list.
- End with ## References.
- Aim for 2000-4000 words.

Write the report in markdown format."""


def _checkpoint_path(experiment_id: str, query_id: str) -> Path:
    return CHECKPOINT_DIR / experiment_id / f"{query_id}.json"


def _result_path(experiment_id: str, query_id: str) -> Path:
    return RESULTS_DIR / experiment_id / f"{query_id}.md"


def _is_completed(experiment_id: str, query_id: str) -> bool:
    path = _checkpoint_path(experiment_id, query_id)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "success"
    except Exception:  # noqa: BLE001
        return False


def _save_checkpoint(experiment_id: str, query_id: str, payload: dict[str, Any]) -> None:
    path = _checkpoint_path(experiment_id, query_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_report(experiment_id: str, query_id: str, markdown: str) -> None:
    path = _result_path(experiment_id, query_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def _load_queries(query_ids_file: Path) -> list[dict[str, Any]]:
    ids_payload = json.loads(query_ids_file.read_text(encoding="utf-8"))
    query_ids = ids_payload["query_ids"] if isinstance(ids_payload, dict) else ids_payload
    wanted = set(query_ids)
    all_queries = json.loads((ROOT / "data" / "eval_queries_v2.json").read_text(encoding="utf-8"))["queries"]
    by_id = {q["id"]: q for q in all_queries}
    return [by_id[qid] for qid in query_ids if qid in by_id and qid in wanted]


def _evidence_cache_path(policy: str, query_id: str) -> Path:
    return EVIDENCE_CACHE / policy / f"{query_id}.json"


def _source_from_dict(data: dict[str, Any]) -> SourceExtraction:
    try:
        return SourceExtraction.model_validate(data)
    except AttributeError:
        return SourceExtraction(**data)


def _dedupe_docs(docs: Iterable[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = doc.url or doc.id
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


async def _fixed_retrieval(query: str, llm: LLMCaller) -> list[SourceExtraction]:
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    web_docs = await web.search_batch([query], max_results_per=10)
    academic_docs = await academic.search(query, max_per_source=5)
    docs = _dedupe_docs(web_docs + academic_docs)
    thin_urls = [doc.url for doc in docs if doc.url and len(doc.content) < 500]
    if thin_urls:
        extracted = await url_extractor.extract_batch(thin_urls)
        by_url = {doc.url: doc.content for doc in extracted if doc.content}
        for doc in docs:
            if doc.url in by_url:
                doc.content = by_url[doc.url]
    return await source_extractor.extract_batch(docs, query)


async def _p1_retrieval(query: str, llm: LLMCaller, tracker: CostTracker) -> list[SourceExtraction]:
    sub_queries = await p1_decompose(query, llm, n_queries=25)
    retriever = P1Retriever(llm=llm, cost_tracker=tracker)
    return await retriever.search_and_summarize(sub_queries, query)


async def _p4_retrieval(query: str, llm: LLMCaller) -> list[SourceExtraction]:
    perspectives = [
        Perspective(
            name="Technical Expert",
            description="Technical mechanisms, implementation details, and measurement.",
            focus_areas=["mechanisms", "implementation", "measurement"],
        ),
        Perspective(
            name="Empirical Researcher",
            description="Evidence quality, datasets, methodology, and limitations.",
            focus_areas=["evidence", "methods", "limitations"],
        ),
        Perspective(
            name="Practitioner",
            description="Operational implications, deployment constraints, and tradeoffs.",
            focus_areas=["practice", "deployment", "tradeoffs"],
        ),
        Perspective(
            name="Skeptic",
            description="Failure modes, contested claims, and missing evidence.",
            focus_areas=["failure modes", "controversies", "gaps"],
        ),
    ]
    plan = await generate_search_queries(query, perspectives, llm)
    queries: list[str] = list(plan.get("general_queries", []))
    for vals in plan.get("perspective_queries", {}).values():
        queries.extend(vals)
    seen: set[str] = set()
    unique_queries: list[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique_queries.append(q)
    _, extractions = await p4_search_extract(
        unique_queries,
        query=query,
        web=get_web_searcher(),
        academic=AcademicSearcher(),
        extractor=SourceExtractor(llm=llm, model=DEFAULT_MODEL),
        max_web_per_query=5,
        max_academic_per_query=5,
        do_academic=True,
    )
    return extractions


async def _p7_retrieval(query: str, llm: LLMCaller) -> list[SourceExtraction]:
    graph = ResearchGraph()
    root_questions = await p7_decompose(llm, query, n_roots=4)
    for question in root_questions:
        graph.add_node(question)
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(llm=llm, model=DEFAULT_MODEL)
    for node in graph.get_pending_leaves():
        await execute_node(node, llm, web, academic, url_extractor, source_extractor, query)
    seen: set[str] = set()
    out: list[SourceExtraction] = []
    for node in graph.nodes.values():
        for ext in node.extractions:
            key = ext.doc_id or ext.url
            if key and key not in seen:
                seen.add(key)
                out.append(ext)
    return out


async def _load_or_retrieve(policy: str, query: dict[str, Any], budget: float, resume: bool) -> tuple[list[SourceExtraction], dict[str, Any]]:
    cache_path = _evidence_cache_path(policy, query["id"])
    if resume and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [_source_from_dict(d) for d in payload.get("extractions", [])], payload.get("metadata", {})

    tracker = CostTracker(budget_usd=budget)
    llm = LLMCaller(cost_tracker=tracker)
    t0 = time.time()
    if policy == "fixed":
        extractions = await _fixed_retrieval(query["query"], llm)
    elif policy == "p1":
        extractions = await _p1_retrieval(query["query"], llm, tracker)
    elif policy == "p4":
        extractions = await _p4_retrieval(query["query"], llm)
    elif policy == "p7":
        extractions = await _p7_retrieval(query["query"], llm)
    else:
        raise ValueError(f"Unknown retrieval policy: {policy}")

    metadata = {
        "policy": policy,
        "query_id": query["id"],
        "elapsed_seconds": time.time() - t0,
        "total_tokens": tracker.total_tokens,
        "total_cost_usd": tracker.total_cost,
        "n_extractions": len(extractions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({
            "metadata": metadata,
            "extractions": [e.model_dump(mode="json") for e in extractions],
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return extractions, metadata


async def _synthesize(policy: str, query: str, extractions: list[SourceExtraction], llm: LLMCaller) -> str:
    if policy == "p1":
        return await p1_generate(query, extractions, llm, model=DEFAULT_MODEL)
    evidence = format_extractions_as_evidence(extractions) if extractions else "No source evidence."
    if policy == "p4":
        prompt = P4_STYLE_PROMPT
    elif policy == "p7":
        prompt = P7_STYLE_PROMPT
    elif policy == "common":
        prompt = COMMON_SYNTH_PROMPT
    else:
        raise ValueError(f"Unknown synthesis policy: {policy}")
    return await llm.complete(
        prompt.format(query=query, evidence=evidence),
        model=DEFAULT_MODEL,
        max_tokens=8192,
        temperature=0.3,
    )


def _experiment_id(condition: str, pattern: str) -> str:
    if condition == "matched":
        return f"disentangle_matched_{pattern}"
    if condition == "fixed_retrieval":
        return f"disentangle_fixed_retrieval_{pattern}"
    if condition == "fixed_synthesis":
        return f"disentangle_fixed_synthesis_{pattern}"
    raise ValueError(condition)


def _policies(condition: str, pattern: str) -> tuple[str, str]:
    if condition == "matched":
        return pattern, pattern
    if condition == "fixed_retrieval":
        return "fixed", pattern
    if condition == "fixed_synthesis":
        return pattern, "common"
    raise ValueError(condition)


async def _run_cell(condition: str, pattern: str, query: dict[str, Any], budget: float, resume: bool) -> dict[str, Any]:
    experiment_id = _experiment_id(condition, pattern)
    query_id = query["id"]
    retrieval_policy, synthesis_policy = _policies(condition, pattern)
    t0 = time.time()
    try:
        extractions, retrieval_meta = await _load_or_retrieve(retrieval_policy, query, budget, resume=resume)
        synth_tracker = CostTracker(budget_usd=budget)
        llm = LLMCaller(cost_tracker=synth_tracker)
        markdown = await _synthesize(synthesis_policy, query["query"], extractions, llm)
        _save_report(experiment_id, query_id, markdown)
        result = {
            "status": "success",
            "experiment_id": experiment_id,
            "pattern": pattern,
            "condition": condition,
            "retrieval_policy": retrieval_policy,
            "synthesis_policy": synthesis_policy,
            "query_id": query_id,
            "elapsed_seconds": time.time() - t0,
            "retrieval": retrieval_meta,
            "total_tokens": retrieval_meta.get("total_tokens", 0) + synth_tracker.total_tokens,
            "total_cost_usd": retrieval_meta.get("total_cost_usd", 0.0) + synth_tracker.total_cost,
            "synthesis_tokens": synth_tracker.total_tokens,
            "synthesis_cost_usd": synth_tracker.total_cost,
            "n_extractions": len(extractions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_checkpoint(experiment_id, query_id, result)
        return result
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "experiment_id": experiment_id,
            "pattern": pattern,
            "condition": condition,
            "query_id": query_id,
            "elapsed_seconds": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _save_checkpoint(experiment_id, query_id, result)
        return result


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-ids-file", type=Path, default=DEFAULT_QUERY_IDS)
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--budget", type=float, default=MAX_COST_PER_RUN)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown_patterns = sorted(set(patterns) - set(PATTERNS))
    unknown_conditions = sorted(set(conditions) - set(CONDITIONS))
    if unknown_patterns or unknown_conditions:
        raise SystemExit(f"Unknown patterns={unknown_patterns} conditions={unknown_conditions}")

    queries = _load_queries(args.query_ids_file)
    plan: list[tuple[str, str, dict[str, Any]]] = []
    for condition in conditions:
        for pattern in patterns:
            experiment_id = _experiment_id(condition, pattern)
            for query in queries:
                if args.resume and _is_completed(experiment_id, query["id"]):
                    continue
                plan.append((condition, pattern, query))

    print("Disentanglement pilot")
    print(f"  queries: {len(queries)} from {args.query_ids_file}")
    print(f"  patterns: {', '.join(patterns)}")
    print(f"  conditions: {', '.join(conditions)}")
    print(f"  pending cells: {len(plan)}")
    if args.dry_run:
        for condition, pattern, query in plan:
            retrieval_policy, synthesis_policy = _policies(condition, pattern)
            print(f"  {_experiment_id(condition, pattern)} / {query['id']} "
                  f"retrieval={retrieval_policy} synthesis={synthesis_policy}")
        return 0

    successes = 0
    errors = 0
    t0 = time.time()
    for idx, (condition, pattern, query) in enumerate(plan, 1):
        print(f"[{idx}/{len(plan)}] {_experiment_id(condition, pattern)} / {query['id']}")
        result = await _run_cell(condition, pattern, query, args.budget, args.resume)
        if result["status"] == "success":
            successes += 1
            print(f"  OK extractions={result['n_extractions']} tokens={result['total_tokens']}")
        else:
            errors += 1
            print(f"  ERROR {result.get('error', '')}")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "query_ids_file": str(args.query_ids_file),
        "patterns": patterns,
        "conditions": conditions,
        "successes": successes,
        "errors": errors,
        "elapsed_seconds": time.time() - t0,
    }
    manifest_path = RESULTS_DIR / f"disentanglement_manifest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete: successes={successes} errors={errors}")
    print(f"Manifest: {manifest_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
