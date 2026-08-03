"""Stage 4: Cross-reference and triangulate claims across perspectives.

Uses two-step source extractions as the evidence base instead of
retrieved/reranked chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.source_extractor import format_extractions_as_evidence, SourceExtraction
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

TRIANGULATE_PROMPT = """You are a critical research analyst specializing in evidence triangulation.
Your task is to cross-reference claims from multiple expert perspectives against source
summaries to assess their validity, consistency, and evidential support.

Research query: {query}

== CLAIMS FROM EXPERT CONVERSATIONS ==

Topic clusters and claims:
{claims_text}

Areas of agreement between experts:
{agreements_text}

Areas of disagreement between experts:
{disagreements_text}

== SOURCE MATERIAL FOR VERIFICATION ==

{evidence_text}

== ADDITIONAL VERIFICATION SOURCES (for previously unverified claims) ==

{triangulation_evidence}

Perform rigorous triangulation:
1. For each major claim, assess how many independent sources support it
2. Classify the strength of evidence (strong/moderate/weak/unverified)
3. Identify claims that are well-supported vs. those that rest on thin evidence
4. Resolve disagreements where the evidence clearly favors one position
5. Flag any claims that appear to be incorrect based on the evidence
6. Note important nuances or caveats that the experts may have missed

Return JSON:
{{
  "verified_claims": [
    {{
      "claim": "The specific claim",
      "verdict": "confirmed|likely|uncertain|disputed|refuted",
      "evidence_strength": "strong|moderate|weak",
      "supporting_sources": ["brief source description 1", "brief source description 2"],
      "num_independent_sources": 3,
      "perspectives_supporting": ["Perspective 1", "Perspective 2"],
      "caveats": "Any important nuances or limitations",
      "confidence_score": 0.85
    }}
  ],
  "resolved_disagreements": [
    {{
      "topic": "The disagreement topic",
      "resolution": "Which position the evidence supports and why",
      "winning_position": "The better-supported position",
      "evidence_basis": "Key evidence that resolves this"
    }}
  ],
  "unresolved_disagreements": [
    {{
      "topic": "The disagreement topic",
      "reason": "Why evidence is insufficient to resolve",
      "perspectives": ["Perspective 1", "Perspective 2"]
    }}
  ],
  "novel_insights": [
    {{
      "insight": "Something important found in sources that no expert mentioned",
      "source": "Where this was found",
      "relevance": "Why it matters for the research query"
    }}
  ],
  "evidence_gaps": [
    {{
      "topic": "What remains poorly evidenced",
      "importance": "high|medium|low",
      "suggestion": "What kind of evidence would help"
    }}
  ]
}}
"""


def _format_claims_from_mind_map(mind_map: Dict[str, Any]) -> str:
    """Format claims from the mind map into readable text."""
    parts = []
    for cluster in mind_map.get("topic_clusters", []):
        parts.append(f"\n### {cluster.get('topic', 'Unknown Topic')}")
        parts.append(f"Summary: {cluster.get('summary', '')}")
        for claim in cluster.get("key_claims", []):
            supporters = ", ".join(claim.get("supporting_perspectives", []))
            confidence = claim.get("confidence", "unknown")
            parts.append(
                f"  - [{confidence}] {claim.get('claim', '')} "
                f"(from: {supporters})"
            )
    return "\n".join(parts)


def _format_agreements(mind_map: Dict[str, Any]) -> str:
    """Format areas of agreement."""
    parts = []
    for agreement in mind_map.get("agreements", []):
        perspectives = ", ".join(agreement.get("perspectives", []))
        parts.append(
            f"- [{agreement.get('strength', 'unknown')}] "
            f"{agreement.get('claim', '')} (agreed by: {perspectives})"
        )
    return "\n".join(parts) if parts else "No explicit agreements identified."


def _format_disagreements(mind_map: Dict[str, Any]) -> str:
    """Format areas of disagreement."""
    parts = []
    for disagreement in mind_map.get("disagreements", []):
        parts.append(f"- {disagreement.get('claim', '')} [{disagreement.get('nature', '')}]")
        for pos in disagreement.get("positions", []):
            parts.append(f"    {pos.get('perspective', '')}: {pos.get('position', '')}")
    return "\n".join(parts) if parts else "No explicit disagreements identified."


async def triangulate(
    query: str,
    mind_map: Dict[str, Any],
    evidence_extractions: List[SourceExtraction],
    triangulation_extractions: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Cross-reference claims against source extractions.

    Args:
        query: The research query.
        mind_map: The structured mind map with claims and agreements/disagreements.
        evidence_extractions: Primary source extractions from initial search+extract.
        triangulation_extractions: Additional extractions retrieved specifically for
            claims that needed triangulation.
        llm: LLM caller instance.
        model: Model to use for triangulation (should be high-capability).

    Returns:
        Dict containing verified_claims, resolved_disagreements,
        unresolved_disagreements, novel_insights, and evidence_gaps.
    """
    claims_text = _format_claims_from_mind_map(mind_map)
    agreements_text = _format_agreements(mind_map)
    disagreements_text = _format_disagreements(mind_map)
    evidence_text = (
        format_extractions_as_evidence(evidence_extractions)
        if evidence_extractions
        else "No source material available."
    )
    triangulation_evidence = (
        format_extractions_as_evidence(triangulation_extractions)
        if triangulation_extractions
        else "No additional verification sources available."
    )

    log.info("triangulating",
             n_clusters=len(mind_map.get("topic_clusters", [])),
             n_evidence=len(evidence_extractions),
             n_triangulation=len(triangulation_extractions))

    result = await llm.complete_json(
        TRIANGULATE_PROMPT.format(
            query=query,
            claims_text=claims_text,
            agreements_text=agreements_text,
            disagreements_text=disagreements_text,
            evidence_text=evidence_text,
            triangulation_evidence=triangulation_evidence,
        ),
        model=model,
        temperature=0.2,
        max_tokens=6144,
    )

    # Log summary
    n_verified = len(result.get("verified_claims", []))
    n_resolved = len(result.get("resolved_disagreements", []))
    n_unresolved = len(result.get("unresolved_disagreements", []))
    n_novel = len(result.get("novel_insights", []))
    n_gaps = len(result.get("evidence_gaps", []))

    # Compute aggregate confidence
    verified = result.get("verified_claims", [])
    avg_confidence = 0.0
    if verified:
        avg_confidence = sum(
            c.get("confidence_score", 0.5) for c in verified
        ) / len(verified)

    log.info("triangulation_complete",
             verified_claims=n_verified,
             resolved_disagreements=n_resolved,
             unresolved_disagreements=n_unresolved,
             novel_insights=n_novel,
             evidence_gaps=n_gaps,
             avg_confidence=f"{avg_confidence:.2f}")

    return result
