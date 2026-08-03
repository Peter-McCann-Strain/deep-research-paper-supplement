"""Depth controller: deep analysis on promising leads.

Uses gpt-5.2 to perform focused analysis on source extractions collected
during the width phase, iterating through depth questions for each
high-priority subtopic. No vector retrieval -- reads extractions directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.cost_tracker import CostTracker
from deep_research.tools.source_extractor import SourceExtraction
from deep_research.types import WDAllocation

from .operational_agents import analyze_evidence
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

# Stop words for keyword matching (shared with pipeline._fuzzy_match)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "this", "that", "these", "those", "what", "which", "who", "how",
})


def _tokenize(text: str) -> set[str]:
    """Tokenize text into meaningful words (lowered, stop words removed)."""
    words = set(text.lower().split())
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


DEPTH_SYNTHESIS_SYSTEM = """You are a senior research analyst conducting deep analysis.
Synthesize multiple analysis passes into a coherent, detailed assessment."""

DEPTH_SYNTHESIS_PROMPT = """Given the following analyses from multiple depth iterations
on this subtopic, create a comprehensive synthesis.

Subtopic: {subtopic}
Research Context: {context}

Individual Analyses:
{analyses}

Return JSON:
{{
    "synthesis": "Detailed 3-5 paragraph synthesis integrating all findings",
    "key_findings": ["consolidated finding 1", "finding 2", ...],
    "data_points": ["specific statistic or fact 1", ...],
    "remaining_gaps": ["what is still unclear"],
    "confidence": 0.0-1.0,
    "contradictions": ["any conflicting findings"],
    "sources_used": ["source1", "source2"]
}}"""


class DepthController:
    """Manages the depth (focused analysis) phase of each iteration.

    For each high-priority subtopic, runs multiple analysis iterations:
    1. Select relevant source summaries for the depth question
    2. Analyze summaries with gpt-5.2
    3. Identify gaps and refine queries
    4. Synthesize multiple passes into a cohesive analysis
    """

    def __init__(
        self,
        llm: LLMCaller,
        cost_tracker: CostTracker,
    ):
        self.llm = llm
        self.cost_tracker = cost_tracker

    async def run_depth_phase(
        self,
        depth_questions: List[Dict[str, Any]],
        allocation: WDAllocation,
        query: str,
        summaries: List[SourceExtraction],
    ) -> Dict[str, Any]:
        """Execute the depth phase: focused analysis on top subtopics.

        Args:
            depth_questions: List of subtopic dicts with questions, sorted by priority.
                Each dict: {subtopic, description, priority, questions}.
            allocation: Current allocation controlling depth iterations.
            query: The original research query for context.
            summaries: All source extractions collected during width phases.

        Returns:
            Dict with:
                - subtopic_analyses: list of per-subtopic synthesis results
                - total_analyses: count of individual analysis passes
                - avg_confidence: average confidence across subtopics
        """
        max_iterations = allocation.depth_iterations
        log.info(
            "depth_phase_start",
            step=allocation.step,
            subtopics=len(depth_questions),
            max_iterations=max_iterations,
            available_summaries=len(summaries),
        )

        subtopic_analyses: List[Dict[str, Any]] = []
        total_passes = 0

        # Process top subtopics (limited by depth iterations)
        topics_to_process = depth_questions[:max(max_iterations, 2)]

        for topic_info in topics_to_process:
            subtopic = topic_info.get("subtopic", "")
            description = topic_info.get("description", "")
            questions = topic_info.get("questions", [])

            if not questions:
                continue

            log.info(
                "depth_subtopic_start",
                subtopic=subtopic[:50],
                questions=len(questions),
            )

            # Select summaries relevant to this subtopic
            relevant_summaries = self._select_relevant_summaries(
                summaries, subtopic, description
            )

            # Run analysis iterations for this subtopic
            iteration_analyses: List[Dict[str, Any]] = []
            for iter_idx, question in enumerate(questions[:max_iterations]):
                try:
                    self.cost_tracker.check_budget()
                except Exception:
                    log.warning("depth_budget_exceeded", subtopic=subtopic[:40])
                    break

                # Analyze source summaries directly
                analysis = await analyze_evidence(
                    question=question,
                    context=f"{subtopic}: {description}",
                    summaries=relevant_summaries,
                    llm=self.llm,
                    model=DEFAULT_MODEL,
                )
                analysis["question"] = question
                analysis["summaries_used"] = len(relevant_summaries)
                iteration_analyses.append(analysis)
                total_passes += 1

                log.debug(
                    "depth_iteration",
                    subtopic=subtopic[:40],
                    iter=iter_idx,
                    confidence=analysis.get("confidence", 0),
                )

            # Synthesize across iterations for this subtopic
            if iteration_analyses:
                synthesis = await self._synthesize_analyses(
                    subtopic=subtopic,
                    description=description,
                    analyses=iteration_analyses,
                )
                synthesis["subtopic"] = subtopic
                synthesis["description"] = description
                synthesis["iteration_count"] = len(iteration_analyses)
                subtopic_analyses.append(synthesis)

        # Compute average confidence
        confidences = [
            a.get("confidence", 0.0) for a in subtopic_analyses
        ]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        log.info(
            "depth_phase_complete",
            step=allocation.step,
            subtopics_analyzed=len(subtopic_analyses),
            total_passes=total_passes,
            avg_confidence=f"{avg_confidence:.2f}",
        )

        return {
            "subtopic_analyses": subtopic_analyses,
            "total_analyses": total_passes,
            "avg_confidence": avg_confidence,
        }

    def _select_relevant_summaries(
        self,
        summaries: List[SourceExtraction],
        subtopic: str,
        description: str,
    ) -> List[SourceExtraction]:
        """Select source extractions most relevant to a subtopic.

        Uses Jaccard similarity on tokenized word sets (stop words removed,
        words > 2 chars) -- same approach as pipeline._fuzzy_match.
        Sorts by relevance and caps at 20 to fit context windows.
        """
        if not summaries:
            return []

        subtopic_tokens = _tokenize(f"{subtopic} {description}")

        if not subtopic_tokens:
            # Fallback: return first 20 summaries if no meaningful tokens
            return summaries[:20]

        def relevance_score(s: SourceExtraction) -> float:
            text = f"{s.title} {s.summary}"
            source_tokens = _tokenize(text)
            if not source_tokens:
                return 0.0
            intersection = subtopic_tokens & source_tokens
            union = subtopic_tokens | source_tokens
            return len(intersection) / len(union) if union else 0.0

        # Sort by Jaccard relevance, highest first
        scored = sorted(summaries, key=relevance_score, reverse=True)

        # Cap at a reasonable number to fit context windows
        return scored[:20]

    async def _synthesize_analyses(
        self,
        subtopic: str,
        description: str,
        analyses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize multiple analysis passes into a cohesive result."""
        if len(analyses) == 1:
            # Single analysis: no need to synthesize
            return analyses[0]

        # Format analyses for synthesis prompt
        analyses_parts = []
        for i, a in enumerate(analyses, 1):
            q = a.get("question", "N/A")
            summary = a.get("summary", "No summary")
            findings = a.get("key_findings", [])
            analyses_parts.append(
                f"--- Analysis {i} (Q: {q}) ---\n"
                f"Summary: {summary}\n"
                f"Findings: {', '.join(findings[:5])}"
            )
        analyses_text = "\n\n".join(analyses_parts)

        try:
            synthesis = await self.llm.complete_json(
                DEPTH_SYNTHESIS_PROMPT.format(
                    subtopic=subtopic,
                    context=description,
                    analyses=analyses_text,
                ),
                model=DEFAULT_MODEL,
                system=DEPTH_SYNTHESIS_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            log.warning("depth_synthesis_error", subtopic=subtopic[:40], error=str(e))
            # Fallback: merge findings from all analyses
            all_findings = []
            all_gaps = []
            for a in analyses:
                all_findings.extend(a.get("key_findings", []))
                all_gaps.extend(a.get("gaps", []))
            synthesis = {
                "synthesis": analyses[0].get("summary", ""),
                "key_findings": all_findings[:25],
                "data_points": [],
                "remaining_gaps": all_gaps[:10],
                "confidence": sum(
                    a.get("confidence", 0) for a in analyses
                ) / len(analyses),
                "contradictions": [],
                "sources_used": [],
            }

        return synthesis
