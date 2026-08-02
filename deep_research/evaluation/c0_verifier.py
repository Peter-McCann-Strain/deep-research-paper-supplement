"""E9: C0 — Citation-Grounded verification system.

A FActScore/SAFE-inspired post-processor that scores a research report's
*verified* factual accuracy: every atomic claim in the report is extracted,
its citation is checked against the source, and an entailment LLM judges
whether the citation actually supports the claim.

Pipeline:
  1. Atomic-fact extraction — split the report's prose into claim units
     each tagged with the citation index it relies on.
  2. Source resolution — for each (claim, citation_idx) pair, look up the
     extraction text from the report's source pool.
  3. Entailment check — ask GPT-4o-on-PTU "does this source support this
     claim?" and parse the JSON verdict.
  4. Aggregate per-report verified-factual-accuracy = supported / total.

The pattern's per-pattern verified-factual-accuracy can then be compared to
the LLM-judge factual_accuracy score in the existing parquet pipeline, and
the §6.2 confound becomes testable at the system level: if the patterns
that score high on judge.factual_accuracy *also* score high on C0
verified-fact rate, the confound is benign; if the high-judge-FA patterns
score *low* on C0, the confound is real and consequential.

Note: PTU is free, so this is essentially compute-bound on the rate gate.
For 990 reports × ~25 claims/report × 1 entailment call = ~25k PTU calls,
which is ~2 hours wall-clock at the 200 rpm rate gate.

Reuses:
  - LLMCaller for atomic-fact extraction + entailment
  - ResearchReport.citations for the citation index
  - Existing source extractions (from results/experiments/*/{qid}.md)
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Optional

import structlog

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import LLMCaller

log = structlog.get_logger()


# ── Stage 1: atomic-fact extraction ──────────────────────────────────────────

ATOMIC_EXTRACT_PROMPT = """You will split a research report excerpt into atomic factual claims.

For each claim:
  - Quote a single, verifiable assertion (≤ 25 words)
  - Note which inline citation (e.g. [3]) the claim depends on, if any
  - Skip topic sentences, transitions, opinions, hedging, and questions

Return strict JSON only:
{{"claims": [
  {{"text": "<atomic claim, ≤ 25 words>", "citation_idx": <int or null>}},
  ...
]}}

Aim for 5–15 claims per excerpt. Skip excerpts that contain no factual content.

Excerpt:
{excerpt}"""


@dataclass
class AtomicClaim:
    text: str
    citation_idx: Optional[int]
    source_index: int  # index into the report's section list


async def extract_claims(
    llm: LLMCaller,
    report_excerpt: str,
    source_index: int = 0,
    model: str = DEFAULT_MODEL,
) -> List[AtomicClaim]:
    if len(report_excerpt.strip()) < 100:
        return []
    raw = await llm.complete_json(
        ATOMIC_EXTRACT_PROMPT.format(excerpt=report_excerpt[:6000]),
        model=model, temperature=0.1, max_tokens=1024,
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out = []
    for c in (raw or {}).get("claims", []) or []:
        text = (c.get("text") or "").strip()
        if not text or len(text) > 250:
            continue
        idx = c.get("citation_idx")
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                idx = None
        out.append(AtomicClaim(text=text, citation_idx=idx, source_index=source_index))
    return out


# ── Stage 2: entailment check ────────────────────────────────────────────────

ENTAILMENT_PROMPT = """You are a fact-checker. Given an evidence excerpt and a single claim, decide if the evidence supports the claim.

Rules:
  - "supports" only if the evidence directly states the claim or its essential equivalent
  - "neutral" if the evidence is on the right topic but does not state the claim
  - "contradicts" if the evidence directly contradicts the claim

Output JSON only:
{{"verdict": "supports" | "neutral" | "contradicts", "evidence_quote": "<quote from evidence ≤ 30 words>"}}

Claim: {claim}

Evidence:
{evidence}"""


@dataclass
class EntailmentVerdict:
    claim: str
    citation_idx: Optional[int]
    verdict: str  # "supports" | "neutral" | "contradicts" | "no_source"
    evidence_quote: str = ""


async def check_entailment(
    llm: LLMCaller,
    claim: AtomicClaim,
    evidence_text: str,
    model: str = DEFAULT_MODEL,
) -> EntailmentVerdict:
    if not evidence_text or len(evidence_text.strip()) < 50:
        return EntailmentVerdict(claim=claim.text, citation_idx=claim.citation_idx,
                                  verdict="no_source")
    raw = await llm.complete_json(
        ENTAILMENT_PROMPT.format(claim=claim.text, evidence=evidence_text[:4000]),
        model=model, temperature=0.0, max_tokens=200,
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return EntailmentVerdict(claim=claim.text, citation_idx=claim.citation_idx,
                                      verdict="no_source")
    verdict = (raw or {}).get("verdict", "no_source")
    if verdict not in ("supports", "neutral", "contradicts"):
        verdict = "no_source"
    return EntailmentVerdict(
        claim=claim.text, citation_idx=claim.citation_idx, verdict=verdict,
        evidence_quote=str((raw or {}).get("evidence_quote", ""))[:200],
    )


# ── Stage 3: per-report aggregation ──────────────────────────────────────────

@dataclass
class C0Result:
    pattern: str
    query_id: str
    n_claims: int
    n_supports: int
    n_neutral: int
    n_contradicts: int
    n_no_source: int
    verified_factual_accuracy: float  # supports / max(1, n_claims)
    verdicts: List[EntailmentVerdict]


async def verify_report(
    llm: LLMCaller,
    pattern: str,
    query_id: str,
    report_markdown: str,
    sources_by_citation: dict[int, str],
    model: str = DEFAULT_MODEL,
    max_claims: int = 30,
) -> C0Result:
    """End-to-end C0 score for one report."""
    # Split report into sections; extract claims from each substantive paragraph
    sections = re.split(r"\n#+\s+", report_markdown)
    all_claims: List[AtomicClaim] = []
    for i, sec in enumerate(sections):
        if len(sec.strip()) < 200:
            continue
        sec_claims = await extract_claims(llm, sec, source_index=i, model=model)
        all_claims.extend(sec_claims)
        if len(all_claims) >= max_claims:
            break
    all_claims = all_claims[:max_claims]
    log.info("c0_extracted_claims", pattern=pattern, query_id=query_id, n=len(all_claims))

    # Entailment check per claim
    verdicts: List[EntailmentVerdict] = []
    # When a claim has no citation_idx, fall back to a concatenation of all sources
    # (capped at 6000 chars). This catches claims that are factually grounded but
    # not directly cited inline in the report.
    all_sources_concat = "\n\n---\n\n".join(
        f"[{idx}] {text}" for idx, text in list(sources_by_citation.items())[:6]
    )[:6000]
    for claim in all_claims:
        if claim.citation_idx and claim.citation_idx in sources_by_citation:
            evidence = sources_by_citation[claim.citation_idx]
        else:
            evidence = all_sources_concat
        v = await check_entailment(llm, claim, evidence, model=model)
        verdicts.append(v)

    n = len(verdicts)
    n_sup = sum(1 for v in verdicts if v.verdict == "supports")
    n_neu = sum(1 for v in verdicts if v.verdict == "neutral")
    n_con = sum(1 for v in verdicts if v.verdict == "contradicts")
    n_ns = sum(1 for v in verdicts if v.verdict == "no_source")

    return C0Result(
        pattern=pattern, query_id=query_id, n_claims=n,
        n_supports=n_sup, n_neutral=n_neu, n_contradicts=n_con, n_no_source=n_ns,
        verified_factual_accuracy=n_sup / max(1, n),
        verdicts=verdicts,
    )
