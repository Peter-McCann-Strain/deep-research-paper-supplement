"""Agentic citation verification for research reports.

Follows the SAFE (Google, 2024) and FActScore approaches:
1. Decompose report into atomic factual claims
2. For each claim with a citation: fetch the cited source, check if it supports the claim
3. For each claim without a citation: search the web, check if evidence exists
4. Compute citation precision, recall, and attribution accuracy

Also provides a lighter-weight NLI-only mode (``nli_verify_claim`` /
``nli_verify_batch``) that checks entailment/neutral/contradiction
between a claim and source text without requiring URL fetching or web search.
This mode is useful for faster evaluation when source content is already available.

This is separate from the rubric-based judge -- it provides ground-truth
citation quality scores rather than LLM-estimated ones.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from deep_research.types import Citation

logger = structlog.get_logger()


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class AtomicClaim:
    """A single atomic factual claim extracted from a report."""

    text: str
    section: str
    has_citation: bool
    citation_markers: list[str] = field(default_factory=list)  # e.g., ["[1]", "[3]"]
    cited_urls: list[str] = field(default_factory=list)


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: AtomicClaim
    verdict: str  # "supported", "not_supported", "unverifiable", "source_unavailable"
    confidence: float  # 0-1
    evidence: str  # supporting/contradicting evidence found
    source_url: str = ""  # URL that was checked
    verification_method: str = ""  # "citation_check", "web_search", "parametric"


@dataclass
class CitationVerificationResult:
    """Complete citation verification for a report."""

    report_id: str
    pattern: str
    total_claims: int
    claims_with_citations: int
    claims_without_citations: int

    # Verification outcomes
    supported: int
    not_supported: int
    unverifiable: int
    source_unavailable: int

    # Computed metrics
    citation_precision: float  # supported / (supported + not_supported)
    citation_recall: float  # claims_with_citations / total_claims
    attribution_accuracy: float  # supported / claims_with_citations
    source_availability: float  # (total - source_unavailable) / total

    # DOI tracking (for LitQA2)
    dois_found: list[str] = field(default_factory=list)
    doi_recall: float = 0.0  # populated when reference DOIs available

    # Detail
    verifications: list[ClaimVerification] = field(default_factory=list)


# ── Prompts ──────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM = (
    "You are a factual claim extractor. Your job is to decompose research "
    "report text into atomic, independently verifiable factual claims. "
    "Skip opinions, hedged statements, transitions, and meta-commentary."
)

CLAIM_EXTRACTION_PROMPT = """Decompose the following report text into atomic factual claims.

Rules:
- Each claim must be a single, independently verifiable factual statement.
- Preserve any inline citation markers (e.g., [1], [2]) in the claim text.
- Record which section the claim appears in.
- Skip opinions, subjective assessments, hedged statements (e.g., "may", "could"), transitions, and meta-commentary (e.g., "This report discusses...").
- If a sentence contains multiple facts, split them into separate claims.
- Maximum {max_claims} claims.

Report text:
{report_text}

Return a JSON object with exactly this structure:
{{
    "claims": [
        {{
            "text": "The specific factual claim",
            "section": "Section title where this appears",
            "citation_markers": ["[1]", "[3]"]
        }}
    ]
}}"""

NLI_SYSTEM = (
    "You are a natural language inference (NLI) judge. Your job is to determine "
    "whether a given piece of evidence supports, refutes, or is irrelevant to "
    "a factual claim. Be strict: the evidence must clearly entail the claim, "
    "not merely be topically related."
)

NLI_PROMPT = """Determine whether the following evidence supports or refutes the claim.

Claim: {claim}

Evidence (from source):
{evidence}

Analyse step by step, then provide your verdict.

Return a JSON object with exactly this structure:
{{
    "verdict": "supported" or "not_supported" or "unverifiable",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "Step-by-step explanation of your determination"
}}

Definitions:
- "supported": The evidence clearly entails or confirms the claim.
- "not_supported": The evidence contradicts the claim or provides information that is inconsistent with it.
- "unverifiable": The evidence is topically related but does not contain enough information to confirm or deny the claim."""

THREE_LEVEL_DOC_PROMPT = """Is the following source thematically relevant to the research report topic?

Report topic: {report_topic}
Source title: {source_title}
Source summary (first 500 chars): {source_summary}

Return JSON:
{{
    "relevant": true or false,
    "reasoning": "Brief explanation"
}}"""

THREE_LEVEL_SEC_PROMPT = """Is this citation placed in an appropriate section of the report?

Section title: {section_title}
Section content (excerpt): {section_excerpt}
Cited source title: {source_title}
Cited source summary (first 300 chars): {source_summary}

Return JSON:
{{
    "appropriate": true or false,
    "reasoning": "Brief explanation"
}}"""


# ── DOI regex ────────────────────────────────────────────────────────────────

_DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/[^\s,;\"')\]}>]+",
    re.IGNORECASE,
)

_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


# ── AgenticCitationVerifier ─────────────────────────────────────────────────


class AgenticCitationVerifier:
    """SAFE-style citation verification with source access."""

    def __init__(
        self,
        llm_caller,  # LLMCaller instance for NLI checks
        url_extractor,  # URLExtractor for fetching sources
        web_searcher=None,  # Optional web searcher for uncited claims
        max_claims: int = 50,
        max_concurrent: int = 5,
    ):
        self.llm = llm_caller
        self.url_extractor = url_extractor
        self.web_searcher = web_searcher
        self.max_claims = max_claims
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_report(
        self,
        report_text: str,
        report_id: str,
        pattern: str,
        citations: list | None = None,  # list[Citation] from the ResearchReport
        reference_dois: list[str] | None = None,  # for DOI recall
    ) -> CitationVerificationResult:
        """Full citation verification pipeline.

        1. Extract atomic claims from report
        2. Match claims to citations
        3. For cited claims: fetch source -> NLI check
        4. For uncited claims: web search -> NLI check (if web_searcher available)
        5. Compute metrics
        """
        if not report_text or not report_text.strip():
            logger.warning("verify_report_empty", report_id=report_id)
            return CitationVerificationResult(
                report_id=report_id,
                pattern=pattern,
                total_claims=0,
                claims_with_citations=0,
                claims_without_citations=0,
                supported=0,
                not_supported=0,
                unverifiable=0,
                source_unavailable=0,
                citation_precision=0.0,
                citation_recall=0.0,
                attribution_accuracy=0.0,
                source_availability=0.0,
                dois_found=[],
                doi_recall=0.0,
                verifications=[],
            )

        citations = citations or []

        # Step 1: Extract atomic claims
        claims = await self._extract_atomic_claims(report_text)
        if not claims:
            logger.warning("no_claims_extracted", report_id=report_id)
            return CitationVerificationResult(
                report_id=report_id,
                pattern=pattern,
                total_claims=0,
                claims_with_citations=0,
                claims_without_citations=0,
                supported=0,
                not_supported=0,
                unverifiable=0,
                source_unavailable=0,
                citation_precision=0.0,
                citation_recall=0.0,
                attribution_accuracy=0.0,
                source_availability=0.0,
                dois_found=[],
                doi_recall=0.0,
                verifications=[],
            )

        # Step 2: Match claims to citations
        self._match_claims_to_citations(claims, citations)

        # Step 3 & 4: Verify all claims concurrently
        verification_tasks = []
        for claim in claims:
            if claim.has_citation and claim.cited_urls:
                verification_tasks.append(self._verify_cited_claim(claim))
            elif self.web_searcher is not None:
                verification_tasks.append(self._verify_uncited_claim(claim))
            else:
                # No web searcher and no citation -- mark as unverifiable
                verification_tasks.append(
                    self._make_unverifiable(claim, "No citation and no web searcher available")
                )

        verifications = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_verifications: list[ClaimVerification] = []
        for v in verifications:
            if isinstance(v, ClaimVerification):
                valid_verifications.append(v)
            elif isinstance(v, Exception):
                logger.warning("verification_error", error=str(v))

        # Step 5: Compute metrics
        metrics = self._compute_metrics(valid_verifications, reference_dois)

        # Extract DOIs
        dois_found = self._extract_dois_from_report(report_text)

        claims_with_cit = sum(1 for c in claims if c.has_citation)
        claims_without_cit = len(claims) - claims_with_cit

        return CitationVerificationResult(
            report_id=report_id,
            pattern=pattern,
            total_claims=len(claims),
            claims_with_citations=claims_with_cit,
            claims_without_citations=claims_without_cit,
            supported=metrics["supported"],
            not_supported=metrics["not_supported"],
            unverifiable=metrics["unverifiable"],
            source_unavailable=metrics["source_unavailable"],
            citation_precision=metrics["citation_precision"],
            citation_recall=metrics["citation_recall"],
            attribution_accuracy=metrics["attribution_accuracy"],
            source_availability=metrics["source_availability"],
            dois_found=dois_found,
            doi_recall=metrics["doi_recall"],
            verifications=valid_verifications,
        )

    async def _extract_atomic_claims(
        self,
        report_text: str,
    ) -> list[AtomicClaim]:
        """Decompose report into atomic factual claims using LLM.

        Follows FActScore approach: break each sentence into independent,
        verifiable factual claims. Skip opinions, analyses, and hedged statements.
        """
        # Truncate to avoid exceeding context window
        truncated = report_text[:12000]

        prompt = CLAIM_EXTRACTION_PROMPT.format(
            max_claims=self.max_claims,
            report_text=truncated,
        )

        try:
            result = await self.llm.complete_json(
                prompt,
                system=CLAIM_EXTRACTION_SYSTEM,
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning("claim_extraction_llm_error", error=str(e))
            return []

        # Parse the response
        raw_claims = result.get("claims", [])
        if not isinstance(raw_claims, list):
            logger.warning("claim_extraction_bad_format", type=type(raw_claims).__name__)
            return []

        claims: list[AtomicClaim] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue

            markers = item.get("citation_markers", [])
            if not isinstance(markers, list):
                markers = []
            # Also extract markers from the text itself
            text_markers = _CITATION_MARKER_PATTERN.findall(text)
            for m in text_markers:
                marker = f"[{m}]"
                if marker not in markers:
                    markers.append(marker)

            claims.append(
                AtomicClaim(
                    text=text,
                    section=item.get("section", ""),
                    has_citation=len(markers) > 0,
                    citation_markers=markers,
                )
            )

            if len(claims) >= self.max_claims:
                break

        logger.info("claims_extracted", total=len(claims))
        return claims

    def _match_claims_to_citations(
        self,
        claims: list[AtomicClaim],
        citations: list,  # list[Citation]
    ) -> None:
        """Match citation markers [N] in claims to Citation objects.

        Populates claim.cited_urls from the matching citations.
        """
        # Build a mapping: reference number -> Citation
        # Citations are typically ordered [1], [2], ...
        citation_map: dict[int, Citation] = {}
        for i, cit in enumerate(citations):
            citation_map[i + 1] = cit

        for claim in claims:
            urls: list[str] = []
            for marker in claim.citation_markers:
                # Extract the number from [N]
                match = re.match(r"\[(\d+)\]", marker)
                if match:
                    ref_num = int(match.group(1))
                    cit = citation_map.get(ref_num)
                    if cit and cit.source_url and cit.source_url not in urls:
                        urls.append(cit.source_url)
            claim.cited_urls = urls
            # Update has_citation based on whether we found actual URLs
            if urls:
                claim.has_citation = True

    async def _verify_cited_claim(
        self,
        claim: AtomicClaim,
    ) -> ClaimVerification:
        """Verify a claim against its cited source.

        1. Fetch the cited URL
        2. Extract relevant content
        3. NLI check: does source entail the claim?
        """
        async with self._semaphore:
            # Try each cited URL until we get content
            for url in claim.cited_urls:
                try:
                    doc = await self.url_extractor.extract(url)
                    if doc and doc.content and len(doc.content.strip()) > 50:
                        # We have source content -- run NLI check
                        verdict, confidence = await self._nli_check(
                            claim.text, doc.content[:5000]
                        )
                        return ClaimVerification(
                            claim=claim,
                            verdict=verdict,
                            confidence=confidence,
                            evidence=doc.content[:500],
                            source_url=url,
                            verification_method="citation_check",
                        )
                except Exception as e:
                    logger.warning(
                        "source_fetch_error",
                        url=url[:80],
                        error=str(e),
                    )

            # All URLs failed to fetch
            return ClaimVerification(
                claim=claim,
                verdict="source_unavailable",
                confidence=0.0,
                evidence="Could not fetch any cited source URL.",
                source_url=claim.cited_urls[0] if claim.cited_urls else "",
                verification_method="citation_check",
            )

    async def _verify_uncited_claim(
        self,
        claim: AtomicClaim,
    ) -> ClaimVerification:
        """Attempt to verify an uncited claim via web search.

        1. Search for the claim text
        2. Check top results for supporting evidence
        """
        async with self._semaphore:
            if self.web_searcher is None:
                return ClaimVerification(
                    claim=claim,
                    verdict="unverifiable",
                    confidence=0.0,
                    evidence="No web searcher available.",
                    verification_method="web_search",
                )

            try:
                # Search for the claim text
                results = await self.web_searcher.search(claim.text, max_results=3)
                if not results:
                    return ClaimVerification(
                        claim=claim,
                        verdict="unverifiable",
                        confidence=0.0,
                        evidence="No web search results found.",
                        verification_method="web_search",
                    )

                # Check top results for supporting evidence
                for result in results:
                    content = result.get("content", "") or result.get("snippet", "")
                    if content and len(content.strip()) > 30:
                        verdict, confidence = await self._nli_check(
                            claim.text, content[:5000]
                        )
                        if verdict == "supported":
                            return ClaimVerification(
                                claim=claim,
                                verdict="supported",
                                confidence=confidence,
                                evidence=content[:500],
                                source_url=result.get("url", ""),
                                verification_method="web_search",
                            )

                # No search result supported the claim
                return ClaimVerification(
                    claim=claim,
                    verdict="unverifiable",
                    confidence=0.3,
                    evidence="Web search results did not clearly support the claim.",
                    verification_method="web_search",
                )

            except Exception as e:
                logger.warning("web_search_error", error=str(e))
                return ClaimVerification(
                    claim=claim,
                    verdict="unverifiable",
                    confidence=0.0,
                    evidence=f"Web search failed: {e}",
                    verification_method="web_search",
                )

    async def _make_unverifiable(
        self,
        claim: AtomicClaim,
        reason: str,
    ) -> ClaimVerification:
        """Create an unverifiable result without any external calls."""
        return ClaimVerification(
            claim=claim,
            verdict="unverifiable",
            confidence=0.0,
            evidence=reason,
            verification_method="parametric",
        )

    async def _nli_check(
        self,
        claim: str,
        evidence: str,
    ) -> tuple[str, float]:
        """Natural language inference check using LLM.

        Returns (verdict, confidence) where verdict is
        "supported", "not_supported", or "unverifiable".
        """
        prompt = NLI_PROMPT.format(claim=claim, evidence=evidence[:5000])

        try:
            result = await self.llm.complete_json(
                prompt,
                system=NLI_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as e:
            logger.warning("nli_check_error", error=str(e))
            return ("unverifiable", 0.0)

        verdict = result.get("verdict", "unverifiable")
        if verdict not in ("supported", "not_supported", "unverifiable"):
            verdict = "unverifiable"

        confidence = result.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return (verdict, confidence)

    def _compute_metrics(
        self,
        verifications: list[ClaimVerification],
        reference_dois: list[str] | None,
    ) -> dict[str, Any]:
        """Compute precision, recall, attribution accuracy from verifications."""
        if not verifications:
            return {
                "supported": 0,
                "not_supported": 0,
                "unverifiable": 0,
                "source_unavailable": 0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "attribution_accuracy": 0.0,
                "source_availability": 0.0,
                "doi_recall": 0.0,
            }

        supported = sum(1 for v in verifications if v.verdict == "supported")
        not_supported = sum(1 for v in verifications if v.verdict == "not_supported")
        unverifiable = sum(1 for v in verifications if v.verdict == "unverifiable")
        source_unavailable = sum(
            1 for v in verifications if v.verdict == "source_unavailable"
        )
        total = len(verifications)

        # Citation precision: of claims that were checked (supported or not_supported),
        # what fraction were supported?
        checked = supported + not_supported
        citation_precision = supported / checked if checked > 0 else 0.0

        # Citation recall: fraction of claims that have citations
        claims_with_cit = sum(1 for v in verifications if v.claim.has_citation)
        citation_recall = claims_with_cit / total if total > 0 else 0.0

        # Attribution accuracy: of cited claims, how many are supported?
        cited_verifications = [v for v in verifications if v.claim.has_citation]
        cited_supported = sum(1 for v in cited_verifications if v.verdict == "supported")
        attribution_accuracy = (
            cited_supported / len(cited_verifications)
            if cited_verifications
            else 0.0
        )

        # Source availability: fraction that were not source_unavailable
        source_availability = (total - source_unavailable) / total if total > 0 else 0.0

        # DOI recall: if reference DOIs are provided
        doi_recall = 0.0
        if reference_dois:
            # Collect all DOIs from cited URLs and evidence
            found_dois: set[str] = set()
            for v in verifications:
                for url in v.claim.cited_urls:
                    for doi in _DOI_PATTERN.findall(url):
                        found_dois.add(doi.lower().rstrip("."))
                for doi in _DOI_PATTERN.findall(v.evidence):
                    found_dois.add(doi.lower().rstrip("."))
                if v.source_url:
                    for doi in _DOI_PATTERN.findall(v.source_url):
                        found_dois.add(doi.lower().rstrip("."))

            ref_dois_lower = {d.lower().rstrip(".") for d in reference_dois}
            matched = found_dois & ref_dois_lower
            doi_recall = len(matched) / len(ref_dois_lower) if ref_dois_lower else 0.0

        return {
            "supported": supported,
            "not_supported": not_supported,
            "unverifiable": unverifiable,
            "source_unavailable": source_unavailable,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "attribution_accuracy": attribution_accuracy,
            "source_availability": source_availability,
            "doi_recall": doi_recall,
        }

    def _extract_dois_from_report(self, report_text: str) -> list[str]:
        """Extract DOI strings from report text."""
        matches = _DOI_PATTERN.findall(report_text)
        # Deduplicate while preserving order, clean trailing punctuation
        seen: set[str] = set()
        dois: list[str] = []
        for m in matches:
            cleaned = m.rstrip(".,;:\"')")
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                dois.append(cleaned)
        return dois


# ── NLI-based citation verification ────────────────────────────────────────
#
# A lighter-weight alternative to the full agentic pipeline. Requires
# source content to be provided directly (no URL fetching or web search).
# Useful when checkpoint data already contains retrieved source text.


_NLI_LABELS = ("entailment", "neutral", "contradiction")

NLI_VERIFICATION_SYSTEM = (
    "You are a textual entailment classifier. Given a premise (source text) "
    "and a hypothesis (claim from a research report), classify the relationship "
    "as 'entailment', 'neutral', or 'contradiction'. Be strict: 'entailment' "
    "requires the premise to clearly support the hypothesis; 'contradiction' "
    "requires the premise to clearly refute it; otherwise use 'neutral'."
)

NLI_VERIFICATION_PROMPT = """Classify the relationship between the premise and hypothesis.

Premise (source text):
{premise}

Hypothesis (claim):
{hypothesis}

Return a JSON object with exactly this structure:
{{
    "label": "entailment" or "neutral" or "contradiction",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "Brief explanation"
}}"""


@dataclass
class NLIVerificationResult:
    """Result of NLI-based verification for a single claim-source pair."""

    claim_text: str
    source_text: str
    label: str  # "entailment", "neutral", or "contradiction"
    confidence: float  # 0-1
    reasoning: str = ""

    @property
    def is_supported(self) -> bool:
        """Whether the source entails the claim."""
        return self.label == "entailment"

    @property
    def is_contradicted(self) -> bool:
        """Whether the source contradicts the claim."""
        return self.label == "contradiction"


async def nli_verify_claim(
    claim_text: str,
    source_text: str,
    llm_caller,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> NLIVerificationResult:
    """Verify a single claim against source text using NLI.

    This is a lightweight alternative to the full agentic verification.
    It directly checks entailment between the claim and the provided
    source content without any URL fetching or web search.

    Args:
        claim_text: The factual claim to verify.
        source_text: The source/premise text to check against.
        llm_caller: LLM caller with ``complete_json()`` method.
        model: Model to use for NLI.
        temperature: Sampling temperature (low for determinism).
        max_tokens: Max tokens for the response.

    Returns:
        NLIVerificationResult with label, confidence, and reasoning.
    """
    if not claim_text or not source_text:
        return NLIVerificationResult(
            claim_text=claim_text or "",
            source_text=source_text or "",
            label="neutral",
            confidence=0.0,
            reasoning="Empty claim or source text.",
        )

    prompt = NLI_VERIFICATION_PROMPT.format(
        premise=source_text[:5000],
        hypothesis=claim_text[:1000],
    )

    try:
        result = await llm_caller.complete_json(
            prompt,
            system=NLI_VERIFICATION_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("nli_verify_claim_error", error=str(e))
        return NLIVerificationResult(
            claim_text=claim_text,
            source_text=source_text[:200],
            label="neutral",
            confidence=0.0,
            reasoning=f"LLM call failed: {e}",
        )

    label = result.get("label", "neutral")
    if label not in _NLI_LABELS:
        label = "neutral"

    confidence = result.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = result.get("reasoning", "")

    return NLIVerificationResult(
        claim_text=claim_text,
        source_text=source_text[:200],  # truncate for storage
        label=label,
        confidence=confidence,
        reasoning=reasoning,
    )


async def nli_verify_batch(
    claim_source_pairs: list[tuple[str, str]],
    llm_caller,
    model: str = "gpt-4o",
    max_concurrent: int = 5,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> list[NLIVerificationResult]:
    """Batch NLI verification for multiple claim-source pairs.

    Runs verification concurrently with a semaphore to limit parallelism.

    Args:
        claim_source_pairs: List of (claim_text, source_text) tuples.
        llm_caller: LLM caller with ``complete_json()`` method.
        model: Model to use for NLI.
        max_concurrent: Maximum concurrent LLM calls.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.

    Returns:
        List of NLIVerificationResult, one per input pair.
    """
    if not claim_source_pairs:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _verify_one(pair: tuple[str, str]) -> NLIVerificationResult:
        async with semaphore:
            return await nli_verify_claim(
                claim_text=pair[0],
                source_text=pair[1],
                llm_caller=llm_caller,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    results = await asyncio.gather(
        *[_verify_one(p) for p in claim_source_pairs],
        return_exceptions=True,
    )

    verified: list[NLIVerificationResult] = []
    for i, r in enumerate(results):
        if isinstance(r, NLIVerificationResult):
            verified.append(r)
        elif isinstance(r, Exception):
            logger.warning("nli_batch_error", index=i, error=str(r))
            verified.append(NLIVerificationResult(
                claim_text=claim_source_pairs[i][0],
                source_text=claim_source_pairs[i][1][:200],
                label="neutral",
                confidence=0.0,
                reasoning=f"Batch error: {r}",
            ))

    return verified


def compute_nli_metrics(
    results: list[NLIVerificationResult],
) -> dict[str, float]:
    """Compute aggregate NLI metrics from a batch of results.

    Args:
        results: List of NLIVerificationResult.

    Returns:
        Dictionary with aggregate metrics.
    """
    if not results:
        return {
            "n_total": 0,
            "n_entailment": 0,
            "n_neutral": 0,
            "n_contradiction": 0,
            "entailment_rate": 0.0,
            "contradiction_rate": 0.0,
            "mean_confidence": 0.0,
        }

    n_total = len(results)
    n_entailment = sum(1 for r in results if r.label == "entailment")
    n_neutral = sum(1 for r in results if r.label == "neutral")
    n_contradiction = sum(1 for r in results if r.label == "contradiction")
    mean_conf = sum(r.confidence for r in results) / n_total

    return {
        "n_total": n_total,
        "n_entailment": n_entailment,
        "n_neutral": n_neutral,
        "n_contradiction": n_contradiction,
        "entailment_rate": n_entailment / n_total,
        "contradiction_rate": n_contradiction / n_total,
        "mean_confidence": mean_conf,
    }


class CitationVerifier:
    """Unified citation verifier supporting both agentic and NLI modes.

    Wraps ``AgenticCitationVerifier`` for full verification and adds an
    ``nli_mode`` for lighter-weight NLI-only verification when source
    content is already available.

    Args:
        llm_caller: LLM caller with ``complete_json()`` method.
        url_extractor: URL extractor for fetching sources (agentic mode).
        web_searcher: Optional web searcher (agentic mode).
        max_claims: Maximum claims to extract.
        max_concurrent: Concurrency limit.
        nli_mode: If True, default to NLI-only verification.
        nli_model: Model to use for NLI verification.
    """

    def __init__(
        self,
        llm_caller,
        url_extractor=None,
        web_searcher=None,
        max_claims: int = 50,
        max_concurrent: int = 5,
        nli_mode: bool = False,
        nli_model: str = "gpt-4o",
    ):
        self.llm = llm_caller
        self.url_extractor = url_extractor
        self.web_searcher = web_searcher
        self.max_claims = max_claims
        self.max_concurrent = max_concurrent
        self.nli_mode = nli_mode
        self.nli_model = nli_model

        # Build the agentic verifier if dependencies are provided
        self._agentic: AgenticCitationVerifier | None = None
        if url_extractor is not None:
            self._agentic = AgenticCitationVerifier(
                llm_caller=llm_caller,
                url_extractor=url_extractor,
                web_searcher=web_searcher,
                max_claims=max_claims,
                max_concurrent=max_concurrent,
            )

    async def verify_report(
        self,
        report_text: str,
        report_id: str,
        pattern: str,
        citations: list | None = None,
        reference_dois: list[str] | None = None,
    ) -> CitationVerificationResult:
        """Full agentic verification (delegates to AgenticCitationVerifier).

        Raises RuntimeError if url_extractor was not provided at init.
        """
        if self._agentic is None:
            raise RuntimeError(
                "Agentic verification requires url_extractor. "
                "Pass url_extractor at construction or use nli_verify_claims() instead."
            )
        return await self._agentic.verify_report(
            report_text=report_text,
            report_id=report_id,
            pattern=pattern,
            citations=citations,
            reference_dois=reference_dois,
        )

    async def nli_verify_claims(
        self,
        claim_source_pairs: list[tuple[str, str]],
    ) -> list[NLIVerificationResult]:
        """NLI-only verification for pre-extracted claim-source pairs.

        This is the lightweight path: no URL fetching, no web search.
        Just entailment classification between each claim and its source.

        Args:
            claim_source_pairs: List of (claim_text, source_text) tuples.

        Returns:
            List of NLIVerificationResult.
        """
        return await nli_verify_batch(
            claim_source_pairs=claim_source_pairs,
            llm_caller=self.llm,
            model=self.nli_model,
            max_concurrent=self.max_concurrent,
        )

    async def nli_verify_single(
        self,
        claim_text: str,
        source_text: str,
    ) -> NLIVerificationResult:
        """NLI verification for a single claim-source pair.

        Args:
            claim_text: The factual claim.
            source_text: The source text to check against.

        Returns:
            NLIVerificationResult.
        """
        return await nli_verify_claim(
            claim_text=claim_text,
            source_text=source_text,
            llm_caller=self.llm,
            model=self.nli_model,
        )
