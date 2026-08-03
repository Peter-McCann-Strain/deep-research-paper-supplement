"""Pattern 12: RL-trained 7B agent — Qwen2.5-7B-Instruct + P12 LoRA via GRPO with DR-Judge reward (E10).

Same single-pass RAG architecture as P0/P9 (search → extract → single LLM call),
but using a local Qwen2.5-7B-Instruct base model with a P12 LoRA adapter trained
via GRPO using DR-Judge as the reward signal (E10).

Purpose: Isolate the effect of RL fine-tuning on top of the same 7B backbone used
by P9. Comparing P12 vs P9 measures pure RL training gain at fixed scaffolding +
fixed model scale; comparing P12 vs P10 contrasts our adapter against the
GAIR/DeepResearcher-7b RL-trained checkpoint.

Flow:
    Query → Web search (top 10 results)
    → Extract page content
    → Two-step source extraction (using local 7B + P12 LoRA)
    → Single LLM call: "write a research report using these sources" (using local 7B + P12 LoRA)
    → Parse into ResearchReport
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import List

import structlog

from deep_research.config import MAX_COST_PER_RUN
from deep_research.tools import (
    AcademicSearcher,
    CostTracker,
    SourceExtraction,
    SourceExtractor,
    URLExtractor,
    get_web_searcher,
    format_extractions_as_evidence,
    StateManager,
)
from deep_research.tools.local_llm_caller import LocalLLMCaller
from deep_research.types import ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

log = structlog.get_logger()

# Default local model — Qwen2.5-7B-Instruct (same base as P9 and as DeepResearcher)
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Default LoRA adapter — RL-trained P12 adapter (E10 GRPO with DR-Judge reward)
DEFAULT_LORA_ADAPTER_PATH = str(
    Path(__file__).resolve().parents[3] / "models" / "P12-RL-LoRA-v2"
)

REPORT_PROMPT = """You are a research analyst. Write a comprehensive, well-structured research report answering the following query. Use ONLY the provided source evidence. Cite sources using inline numbered references like [1], [2], etc.

Research query: {query}

Source evidence:
{evidence}

Requirements:
- Start with a title (# Title)
- Include an abstract (## Abstract)
- Organize into logical sections (## Section Name)
- End with a References section listing all cited sources
- Be comprehensive, accurate, and balanced
- Use inline citations [1], [2], etc. throughout
- Aim for 2000-4000 words

Write the full research report:"""


async def run(
    query: str,
    budget_usd: float = MAX_COST_PER_RUN,
    **kwargs,
) -> ResearchReport:
    """Execute P12 RL-trained agent: P0/P9 architecture with Qwen2.5-7B + P12 LoRA."""
    start = time.time()

    model_id = kwargs.get("model_id", DEFAULT_LOCAL_MODEL)
    quantize_4bit = kwargs.get("quantize_4bit", True)
    evidence_word_limit = kwargs.get("evidence_word_limit", 6000)
    lora_adapter_path = (
        kwargs.get("lora_adapter_path")
        or os.environ.get("P12_LORA_ADAPTER_PATH")
        or DEFAULT_LORA_ADAPTER_PATH
    )

    tracker = CostTracker(budget_usd=budget_usd)
    llm = LocalLLMCaller(
        model_id=model_id,
        cost_tracker=tracker,
        quantize_4bit=quantize_4bit,
        lora_adapter_path=lora_adapter_path,
    )
    state = StateManager("p12_rl_trained")
    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(
        llm=llm, model=model_id,
        max_content_per_call=15_000,  # 15K chars ≈ 4K tokens, fits 16GB VRAM with 4-bit 7B
    )
    trace = ProcessTrace(pattern_name="p12_rl_trained", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p12_start", query=query[:80], model=model_id, lora=lora_adapter_path)

    # ── Stage 1: Web search (single query, top 10) ───────────────────────
    log.info("p12_stage_1_search")
    web_docs = await web.search_batch([query], max_results_per=10)
    trace.append(tool="search", input_args={"query": query, "max_results": 10},
                 output_summary=f"{len(web_docs)} web docs", n_results=len(web_docs))

    # Also grab a few academic results
    academic_docs = await academic.search(query, max_per_source=5)
    trace.append(tool="academic_search", input_args={"query": query, "max_per_source": 5},
                 output_summary=f"{len(academic_docs)} academic docs", n_results=len(academic_docs))

    # Deduplicate by URL
    seen_urls: set = set()
    all_docs = []
    for doc in web_docs + academic_docs:
        if doc.url and doc.url not in seen_urls:
            seen_urls.add(doc.url)
            all_docs.append(doc)

    log.info("p12_search_done", web=len(web_docs), academic=len(academic_docs),
             deduped=len(all_docs))

    # ── Stage 2: Extract page content where missing ──────────────────────
    urls_to_extract = [
        doc.url for doc in all_docs if doc.url and len(doc.content) < 500
    ]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in all_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]
        log.info("p12_page_extract_done", pages=len(url_to_content))
        trace.append(tool="extract", input_args={"n_urls": len(urls_to_extract)},
                     output_summary=f"{len(url_to_content)} pages extracted",
                     n_results=len(url_to_content))

    # ── Stage 3: Two-step source extraction (using local 7B + P12 LoRA) ───
    log.info("p12_stage_3_source_extraction")
    tokens_before_extract = tracker.total_tokens
    extractions = await source_extractor.extract_batch(all_docs, query)
    trace.append(tool="source_extract", input_args={"n_docs": len(all_docs)},
                 output_summary=f"{len(extractions)} relevant extractions",
                 n_results=len(extractions),
                 tokens_used=tracker.total_tokens - tokens_before_extract)

    state.save("search", {
        "doc_count": len(all_docs),
        "extraction_count": len(extractions),
    })

    if not extractions:
        log.warning("p12_no_relevant_sources")
        return ResearchReport(
            query=query,
            title=query,
            pattern_name="p12_rl_trained",
            total_cost_usd=tracker.total_cost,
            total_tokens=tracker.total_tokens,
            elapsed_seconds=time.time() - start,
        )

    # ── Stage 4: Single-shot report generation (using local 7B + P12 LoRA) ─
    log.info("p12_stage_4_generate", sources=len(extractions))
    evidence_text = format_extractions_as_evidence(extractions)

    # Truncate evidence if too long for 7B context
    # Qwen2.5-7B supports 128K, but generation quality degrades with very long context
    words = evidence_text.split()
    if len(words) > evidence_word_limit:
        evidence_text = " ".join(words[:evidence_word_limit]) + "\n\n[... evidence truncated ...]"
        log.info("p12_evidence_truncated", original_words=len(words), truncated_to=evidence_word_limit)

    tokens_before_gen = tracker.total_tokens
    report_md = await llm.complete(
        REPORT_PROMPT.format(query=query, evidence=evidence_text),
        max_tokens=4096,  # 7B generates slower, keep manageable
        temperature=0.3,
    )
    trace.append(tool="generate", input_args={"max_tokens": 4096, "model": model_id},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    state.save("report", {"markdown_length": len(report_md)})

    # ── Assemble ─────────────────────────────────────────────────────────
    elapsed = time.time() - start

    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=extractions,
        pattern_name="p12_rl_trained",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
    )
    report.elapsed_seconds = elapsed

    log.info("p12_complete",
             cost=f"${tracker.total_cost:.4f}",
             tokens=tracker.total_tokens,
             sections=len(report.sections),
             elapsed=f"{elapsed:.1f}s")

    state.save("final", {
        "cost": tracker.total_cost,
        "tokens": tracker.total_tokens,
        "elapsed": elapsed,
    })

    report.metadata["cost_breakdown"] = tracker.to_dict()
    report.metadata["sub_queries"] = []
    report.metadata["search_queries_sent"] = [query]
    report.metadata["n_documents_retrieved"] = len(web_docs) + len(academic_docs)
    report.metadata["n_documents_after_dedup"] = len(all_docs)
    report.metadata["n_extractions"] = len(extractions)
    report.metadata["local_model"] = model_id
    report.metadata["quantize_4bit"] = quantize_4bit
    report.metadata["lora_adapter_path"] = lora_adapter_path

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len(seen_urls)
    state.save("trace", trace.model_dump(mode="json"))

    return report
