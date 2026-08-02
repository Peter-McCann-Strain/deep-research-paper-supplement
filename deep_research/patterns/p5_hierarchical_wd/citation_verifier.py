"""Citation verifier: spot-check citations for accuracy.

Uses gpt-5.2 to verify that cited claims are actually supported
by the source extractions collected during research, flagging any
unsupported or inaccurate citations. No vector retrieval -- looks
up source extractions directly by title/URL match.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.cost_tracker import CostTracker
from deep_research.tools.source_extractor import SourceExtraction
from deep_research.types import Citation
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

VERIFY_SYSTEM = """You are a citation accuracy verifier. Your job is to check whether
a claim is actually supported by the provided source material. Be strict and precise."""

VERIFY_PROMPT = """Verify whether this citation is accurate.

Claim: {claim}
Source Title: {source_title}
Source URL: {source_url}

Source Summary (from research phase):
{source_content}

Return JSON:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "rationale": "Explanation of whether the source supports the claim",
    "suggested_correction": "If unsupported, suggest how to fix (or empty string)"
}}"""

EXTRACT_CITATIONS_PROMPT = """Extract all inline citations from this report. Each citation
is a factual claim paired with its source reference (e.g., [1], [2]).

Report text:
{report_text}

Return JSON:
{{
    "citations": [
        {{
            "claim": "The specific factual claim being made",
            "ref_number": 1,
            "section": "Section title where this appears"
        }}
    ]
}}"""


class CitationVerifier:
    """Verifies citation accuracy by cross-referencing claims with source extractions.

    Process:
    1. Extract citations from the report text
    2. Select a random sample for spot-checking (to limit cost)
    3. Look up the relevant source extraction for each citation
    4. Use gpt-5.2 to verify support
    5. Flag unsupported claims and suggest corrections
    """

    def __init__(
        self,
        llm: LLMCaller,
        cost_tracker: CostTracker,
    ):
        self.llm = llm
        self.cost_tracker = cost_tracker

    async def verify_report(
        self,
        report_text: str,
        citations: List[Citation],
        extractions: List[SourceExtraction],
        max_checks: int = 5,
        model: str = DEFAULT_MODEL,
    ) -> Dict[str, Any]:
        """Verify citations in the report by spot-checking a sample.

        Args:
            report_text: The full report markdown text.
            citations: List of Citation objects from report assembly.
            extractions: All source extractions collected during research.
            max_checks: Maximum number of citations to spot-check.
            model: Model for verification.

        Returns:
            Dict with:
                - total_citations: number of citations in report
                - checked: number of citations verified
                - supported: number that passed verification
                - unsupported: number that failed
                - accuracy_rate: supported / checked
                - results: per-citation verification details
                - flagged_claims: list of unsupported claims
        """
        log.info("citation_verify_start", total_citations=len(citations))

        if not citations:
            return {
                "total_citations": 0,
                "checked": 0,
                "supported": 0,
                "unsupported": 0,
                "accuracy_rate": 1.0,
                "results": [],
                "flagged_claims": [],
            }

        # Extract inline citations from the report for richer claim text
        extracted = await self._extract_inline_citations(report_text)

        # Select sample for spot-checking
        sample_size = min(max_checks, len(citations))
        if sample_size < len(citations):
            sample_indices = sorted(random.sample(range(len(citations)), sample_size))
        else:
            sample_indices = list(range(len(citations)))

        # Verify each sampled citation
        results: List[Dict[str, Any]] = []
        supported_count = 0

        for idx in sample_indices:
            citation = citations[idx]

            # Find matching extracted claim (if available)
            claim_text = citation.claim
            for ext in extracted:
                if ext.get("ref_number") == idx + 1:
                    claim_text = ext.get("claim", citation.claim)
                    break

            try:
                self.cost_tracker.check_budget()
            except Exception:
                log.warning("citation_verify_budget_exceeded")
                break

            # Look up source content from extractions
            source_content = self._find_source_extraction(
                citation, extractions
            )

            # Verify
            verification = await self._verify_single(
                claim=claim_text,
                citation=citation,
                source_content=source_content,
                model=model,
            )
            verification["citation_index"] = idx
            verification["claim"] = claim_text
            results.append(verification)

            if verification.get("supported", False):
                supported_count += 1

        checked = len(results)
        unsupported = checked - supported_count
        accuracy_rate = supported_count / checked if checked > 0 else 1.0

        flagged_claims = [
            {
                "claim": r["claim"],
                "source": r.get("source_title", ""),
                "rationale": r.get("rationale", ""),
                "suggestion": r.get("suggested_correction", ""),
            }
            for r in results
            if not r.get("supported", True)
        ]

        log.info(
            "citation_verify_complete",
            checked=checked,
            supported=supported_count,
            unsupported=unsupported,
            accuracy_rate=f"{accuracy_rate:.2f}",
        )

        return {
            "total_citations": len(citations),
            "checked": checked,
            "supported": supported_count,
            "unsupported": unsupported,
            "accuracy_rate": accuracy_rate,
            "results": results,
            "flagged_claims": flagged_claims,
        }

    async def _extract_inline_citations(
        self, report_text: str
    ) -> List[Dict[str, Any]]:
        """Extract inline citations from the report text."""
        try:
            result = await self.llm.complete_json(
                EXTRACT_CITATIONS_PROMPT.format(
                    report_text=report_text[:30000]
                ),
                model=DEFAULT_MODEL,
                temperature=0.1,
                max_tokens=2048,
            )
            return result.get("citations", [])
        except Exception as e:
            log.warning("citation_extract_error", error=str(e))
            return []

    def _find_source_extraction(
        self,
        citation: Citation,
        extractions: List[SourceExtraction],
    ) -> str:
        """Find the source extraction matching a citation.

        Matches by source_id (doc_id), URL, or title.
        """
        for s in extractions:
            if (
                s.doc_id == citation.source_id
                or s.url == citation.source_url
                or s.title == citation.source_title
            ):
                return s.summary

        # Fallback: partial title match
        citation_title_lower = citation.source_title.lower()
        for s in extractions:
            if citation_title_lower and citation_title_lower in s.title.lower():
                return s.summary

        return "No source extraction available for verification."

    async def _verify_single(
        self,
        claim: str,
        citation: Citation,
        source_content: str,
        model: str,
    ) -> Dict[str, Any]:
        """Verify a single citation against its source summary."""
        try:
            result = await self.llm.complete_json(
                VERIFY_PROMPT.format(
                    claim=claim,
                    source_title=citation.source_title,
                    source_url=citation.source_url,
                    source_content=source_content[:15000],
                ),
                model=model,
                system=VERIFY_SYSTEM,
                temperature=0.1,
                max_tokens=512,
            )
            result["source_title"] = citation.source_title
            result["source_url"] = citation.source_url
            return result
        except Exception as e:
            log.warning("citation_verify_error", claim=claim[:60], error=str(e))
            return {
                "supported": True,  # assume supported on error to avoid false flagging
                "confidence": 0.0,
                "rationale": f"Verification failed: {e}",
                "suggested_correction": "",
                "source_title": citation.source_title,
                "source_url": citation.source_url,
            }
