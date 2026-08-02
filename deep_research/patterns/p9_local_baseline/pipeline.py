"""Pattern 9: Local 7B Baseline — Qwen2.5-7B-Instruct with P0 architecture.

Same single-pass RAG architecture as P0 (search → extract → single LLM call),
but using a local Qwen2.5-7B-Instruct model instead of GPT-4o.

Purpose: Control for model scale.  Comparing P9 vs P0 isolates the effect of
model capability (7B vs GPT-4o).  Comparing P9 vs P10 isolates the effect of
RL training (same 7B base, with vs without RL).

Flow:
    Query → Web search (top 10 results)
    → Extract page content
    → Two-step source extraction (using local 7B)
    → Single LLM call: "write a research report using these sources" (using local 7B)
    → Parse into ResearchReport
"""

from __future__ import annotations

import time
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
from deep_research.types import ProcessTrace, ResearchReport
from deep_research.utils.markdown_parser import parse_markdown_report

log = structlog.get_logger()

# Default local model — Qwen2.5-7B-Instruct (same base as DeepResearcher)
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

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
    """Execute local 7B baseline: P0 architecture with Qwen2.5-7B-Instruct."""
    start = time.time()

    model_id = kwargs.get("model_id", DEFAULT_LOCAL_MODEL)
    quantize_4bit = kwargs.get("quantize_4bit", True)
    evidence_word_limit = kwargs.get("evidence_word_limit", 6000)

    tracker = CostTracker(budget_usd=budget_usd)

    # Allow an injected, pre-built LLM caller (e.g. a llama.cpp GGUF backend for
    # the P17 14B arm) so callers can swap the backbone WITHOUT P9 importing
    # transformers. If no caller is injected, default to the transformers
    # LocalLLMCaller exactly as before — P9/P14 behaviour is byte-for-byte
    # unchanged. An injected caller is rebound to this run's CostTracker so
    # token/cost accounting stays per-run and identical to the default path.
    llm = kwargs.get("llm")
    if llm is None:
        from deep_research.tools.local_llm_caller import LocalLLMCaller

        llm = LocalLLMCaller(
            model_id=model_id,
            cost_tracker=tracker,
            quantize_4bit=quantize_4bit,
        )
    else:
        try:
            llm.cost_tracker = tracker
        except Exception:
            pass
        # Prefer the injected caller's provenance label for metadata/logging.
        model_id = getattr(llm, "model_id", model_id)

    # ── FROZEN-SOURCE injection (E8 frozen_vintage experiment) ───────────────
    # Mirrors the kwargs['llm'] injection above, but for the RETRIEVAL+EXTRACTION
    # side.  When a frozen evidence string (or a frozen-corpus dir + query_id) is
    # supplied, Stages 1-3 (web/academic search, url extraction, model-dependent
    # source extraction) are SKIPPED entirely and the generator reads a byte-
    # identical evidence_text from disk.  This makes the ONLY variable the
    # injected `llm` backbone — extraction quality is held constant by whatever
    # model produced the frozen store (the canonical GPT-4o extractor).
    #
    # frozen['evidence_text'] is the ALREADY-truncated string (the literal Stage-4
    # input), so we do NOT re-apply evidence_word_limit here — every arm, whatever
    # its own limit, sees the same text.  When absent, kwargs are unchanged and the
    # live retrieval path below runs exactly as before (P9/P14/P17 byte-for-byte).
    frozen = kwargs.get("frozen_evidence")
    if frozen is None and kwargs.get("frozen_corpus_dir"):
        import json as _json
        from pathlib import Path as _Path

        _fp = _Path(kwargs["frozen_corpus_dir"]) / f"{kwargs.get('query_id', '')}.json"
        frozen = _json.loads(_fp.read_text())
    if frozen is not None and isinstance(frozen, str):
        # A bare evidence string was injected; wrap it so the access below is uniform.
        frozen = {"evidence_text": frozen, "extractions": []}

    # Generation temperature: P9 historically hardcoded 0.3.  The frozen runner
    # overrides to 0.0 for greedy determinism via gen_temperature; the live path
    # keeps 0.3 by default so non-frozen behaviour is unchanged.
    gen_temperature = float(kwargs.get("gen_temperature", 0.3))

    state = StateManager("p9_local_baseline")
    trace = ProcessTrace(pattern_name="p9_local_baseline", query=query, query_id=kwargs.get("query_id", ""))

    log.info("p9_start", query=query[:80], model=model_id, frozen=frozen is not None)

    if frozen is not None:
        # ── FROZEN PATH: skip Stages 1-3; load evidence_text + extractions ───
        evidence_text = frozen["evidence_text"]
        extractions = [SourceExtraction(**d) for d in frozen.get("extractions", [])]
        # Optional integrity assertion: the frozen string must hash to the stored
        # corpus_sha256 so we can prove the generation input is the frozen one.
        expected_sha = frozen.get("corpus_sha256")
        if expected_sha:
            import hashlib as _hashlib

            actual_sha = _hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError(
                    "frozen_corpus integrity check failed for query_id="
                    f"{kwargs.get('query_id', '')}: evidence_text sha256 {actual_sha} "
                    f"!= stored {expected_sha}"
                )
        web_docs = []
        academic_docs = []
        all_docs = []
        seen_urls = set(frozen.get("urls", []) or [])
        trace.append(tool="frozen_corpus",
                     input_args={"query_id": kwargs.get("query_id", ""),
                                 "corpus_sha256": frozen.get("corpus_sha256", "")},
                     output_summary=f"{len(extractions)} frozen extractions, "
                                    f"{len(evidence_text.split())} evidence words",
                     n_results=len(extractions))
        log.info("p9_frozen_loaded", extractions=len(extractions),
                 evidence_words=len(evidence_text.split()),
                 sha=str(frozen.get("corpus_sha256", ""))[:12])
    else:
        web = get_web_searcher()
        academic = AcademicSearcher()
        url_extractor = URLExtractor()
        source_extractor = SourceExtractor(
            llm=llm, model=model_id,
            max_content_per_call=15_000,  # 15K chars ≈ 4K tokens, fits 16GB VRAM with 4-bit 7B
        )

    if frozen is None:
        # ── Stage 1: Web search (single query, top 10) ───────────────────────
        log.info("p9_stage_1_search")
        web_docs = await web.search_batch([query], max_results_per=10)
        trace.append(tool="search", input_args={"query": query, "max_results": 10},
                     output_summary=f"{len(web_docs)} web docs", n_results=len(web_docs))

        # Also grab a few academic results
        academic_docs = await academic.search(query, max_per_source=5)
        trace.append(tool="academic_search", input_args={"query": query, "max_per_source": 5},
                     output_summary=f"{len(academic_docs)} academic docs", n_results=len(academic_docs))

        # Deduplicate by URL
        seen_urls = set()
        all_docs = []
        for doc in web_docs + academic_docs:
            if doc.url and doc.url not in seen_urls:
                seen_urls.add(doc.url)
                all_docs.append(doc)

        log.info("p9_search_done", web=len(web_docs), academic=len(academic_docs),
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
            log.info("p9_page_extract_done", pages=len(url_to_content))
            trace.append(tool="extract", input_args={"n_urls": len(urls_to_extract)},
                         output_summary=f"{len(url_to_content)} pages extracted",
                         n_results=len(url_to_content))

        # ── Stage 3: Two-step source extraction (using local 7B) ─────────────
        log.info("p9_stage_3_source_extraction")
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
        log.warning("p9_no_relevant_sources")
        return ResearchReport(
            query=query,
            title=query,
            pattern_name="p9_local_baseline",
            total_cost_usd=tracker.total_cost,
            total_tokens=tracker.total_tokens,
            elapsed_seconds=time.time() - start,
        )

    # ── Stage 4: Single-shot report generation (using local 7B) ──────────
    log.info("p9_stage_4_generate", sources=len(extractions))
    if frozen is None:
        evidence_text = format_extractions_as_evidence(extractions)

        # Truncate evidence if too long for 7B context.  Qwen2.5-7B supports 128K,
        # but generation quality degrades with very long context.  In the FROZEN
        # path this truncation is DELIBERATELY skipped: evidence_text is the
        # already-truncated string from the frozen store, so every arm (whose
        # evidence_word_limit could differ) sees a byte-identical context.
        words = evidence_text.split()
        if len(words) > evidence_word_limit:
            evidence_text = " ".join(words[:evidence_word_limit]) + "\n\n[... evidence truncated ...]"
            log.info("p9_evidence_truncated", original_words=len(words), truncated_to=evidence_word_limit)

    tokens_before_gen = tracker.total_tokens
    report_md = await llm.complete(
        REPORT_PROMPT.format(query=query, evidence=evidence_text),
        max_tokens=4096,  # 7B generates slower, keep manageable
        temperature=gen_temperature,
    )
    trace.append(tool="generate", input_args={"max_tokens": 4096, "model": model_id},
                 output_summary=f"{len(report_md)}-char report",
                 tokens_used=tracker.total_tokens - tokens_before_gen)

    # Reasoning models (e.g. DeepSeek-R1-Distill) emit <think>...</think> chains
    # before the report. Strip them BEFORE parsing so the # Title/## Abstract
    # detection and citation/organization scoring see the report only, not the
    # chain-of-thought. Off by default (default kwarg absent) so P9/P14/P17 live
    # behaviour is byte-for-byte unchanged; the frozen runner enables it
    # identically for every arm so it cannot bias one arm relative to another.
    if kwargs.get("strip_think"):
        import re as _re

        report_md = _re.sub(r"<think>.*?</think>", "", report_md, flags=_re.DOTALL).strip()
        # Some reasoning models open <think> but never close it (truncation): drop a
        # dangling open tag and everything before the first markdown heading.
        if "<think>" in report_md:
            m = _re.search(r"^#\s", report_md, flags=_re.MULTILINE)
            if m:
                report_md = report_md[m.start():]
            else:
                report_md = report_md.replace("<think>", "").strip()

    state.save("report", {"markdown_length": len(report_md)})

    # ── Assemble ─────────────────────────────────────────────────────────
    elapsed = time.time() - start

    report = parse_markdown_report(
        query=query,
        markdown=report_md,
        extractions=extractions,
        pattern_name="p9_local_baseline",
        cost_usd=tracker.total_cost,
        total_tokens=tracker.total_tokens,
    )
    report.elapsed_seconds = elapsed

    log.info("p9_complete",
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

    trace.final_report_word_count = len(report_md.split())
    trace.n_unique_urls_visited = len(seen_urls)
    state.save("trace", trace.model_dump(mode="json"))

    return report
