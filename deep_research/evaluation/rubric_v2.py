"""V2 Rubric System for research report evaluation.

Improvements over V1:
- Factual accuracy: 3 -> 8 criteria (30% weight dimension no longer bimodal)
- Instruction following: 1 -> 4 criteria (no longer binary 0/100%)
- Citation quality: redesigned for verifiability without URL access
- Task-adaptive criteria: LLM generates 10-15 per query
- Compatible with DRACO weighted scoring
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Criterion:
    """A single evaluation criterion."""

    text: str
    dimension: str
    weight: float = 1.0  # for DRACO compatibility
    source: str = "general"  # "general", "task_specific", "draco", "benchmark"


@dataclass
class RubricV2:
    """Complete evaluation rubric for a single query."""

    query_id: str
    query_text: str
    criteria: list[Criterion]
    dimension_weights: dict[str, float]

    def get_criteria_by_dimension(self, dimension: str) -> list[Criterion]:
        """Get all criteria for a specific dimension."""
        return [c for c in self.criteria if c.dimension == dimension]

    def get_dimensions(self) -> list[str]:
        """Get all unique dimensions."""
        seen: dict[str, None] = {}
        for c in self.criteria:
            if c.dimension not in seen:
                seen[c.dimension] = None
        return list(seen.keys())

    @property
    def total_criteria(self) -> int:
        return len(self.criteria)


# Dimension weights - V2.1 rebalanced with new dimensions from DRB-II + systematic survey
DIMENSION_WEIGHTS_V2: dict[str, float] = {
    "information_recall": 0.20,
    "factual_accuracy": 0.20,
    "coverage": 0.10,
    "analytical_depth": 0.15,
    "citation_quality": 0.10,
    "logical_coherence": 0.05,
    "organization": 0.05,
    "instruction_following": 0.10,
    "attribution_quality": 0.05,
}

# Source-type-specific dimension weight overrides
# All entries must include every dimension key and sum to 1.0
DIMENSION_WEIGHTS_BY_SOURCE: dict[str, dict[str, float]] = {
    "default": {
        "information_recall": 0.20,
        "factual_accuracy": 0.20,
        "coverage": 0.10,
        "analytical_depth": 0.15,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.10,
        "attribution_quality": 0.05,
    },
    "litqa2": {
        "information_recall": 0.20,
        "factual_accuracy": 0.30,
        "coverage": 0.05,
        "analytical_depth": 0.15,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
    "deepsearchqa": {
        "information_recall": 0.25,
        "factual_accuracy": 0.20,
        "coverage": 0.10,
        "analytical_depth": 0.15,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
    "draco": {
        "information_recall": 0.15,
        "factual_accuracy": 0.20,
        "coverage": 0.20,
        "analytical_depth": 0.15,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
    "researchqa": {
        "information_recall": 0.20,
        "factual_accuracy": 0.20,
        "coverage": 0.10,
        "analytical_depth": 0.20,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
    "research_rubrics": {
        "information_recall": 0.15,
        "factual_accuracy": 0.20,
        "coverage": 0.15,
        "analytical_depth": 0.15,
        "citation_quality": 0.10,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.10,
        "attribution_quality": 0.05,
    },
    "drb2": {
        "information_recall": 0.35,
        "factual_accuracy": 0.15,
        "coverage": 0.10,
        "analytical_depth": 0.15,
        "citation_quality": 0.05,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
    "drbench": {
        "information_recall": 0.20,
        "factual_accuracy": 0.20,
        "coverage": 0.10,
        "analytical_depth": 0.15,
        "citation_quality": 0.15,
        "logical_coherence": 0.05,
        "organization": 0.05,
        "instruction_following": 0.05,
        "attribution_quality": 0.05,
    },
}


def get_dimension_weights(source_type: str = "default") -> dict[str, float]:
    """Get dimension weights for a query source type.

    Falls back to the ``"default"`` weights if *source_type* is not
    recognized.

    Args:
        source_type: One of ``"default"``, ``"litqa2"``, ``"deepsearchqa"``,
            ``"draco"``, ``"researchqa"``.

    Returns:
        A dict mapping dimension names to weights that sum to 1.0.
    """
    return DIMENSION_WEIGHTS_BY_SOURCE.get(
        source_type, DIMENSION_WEIGHTS_BY_SOURCE["default"]
    ).copy()


# ===== EXPANDED GENERAL CRITERIA =====

FACTUAL_ACCURACY_CRITERIA: list[Criterion] = [
    Criterion(
        "Factual claims are accurate and consistent with current knowledge",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Technical terminology is used correctly and precisely",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Specific numbers, dates, or benchmarks cited are accurate",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Historical timeline and chronology of developments is correct",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Comparison claims (X outperforms Y, X predates Y) are supported by cited evidence",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "No internal contradictions between different sections of the report",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Limitations and caveats of described methods or findings are accurately represented",
        "factual_accuracy",
        source="general",
    ),
    Criterion(
        "Current state-of-the-art is correctly identified where relevant",
        "factual_accuracy",
        source="general",
    ),
]

COVERAGE_CRITERIA: list[Criterion] = [
    Criterion(
        "The report covers the major aspects of the topic",
        "coverage",
        source="general",
    ),
    Criterion(
        "Both advantages and limitations of approaches are discussed",
        "coverage",
        source="general",
    ),
    Criterion(
        "Recent developments (within the last 2 years) are included",
        "coverage",
        source="general",
    ),
    Criterion(
        "Multiple perspectives or schools of thought are represented",
        "coverage",
        source="general",
    ),
    Criterion(
        "Practical implications or applications are addressed",
        "coverage",
        source="general",
    ),
]

ANALYTICAL_DEPTH_CRITERIA: list[Criterion] = [
    Criterion(
        "The report synthesizes across sources rather than merely summarizing each",
        "analytical_depth",
        source="general",
    ),
    Criterion(
        "Analysis goes beyond surface-level description to identify patterns, trade-offs, or mechanisms",
        "analytical_depth",
        source="general",
    ),
    Criterion(
        "Connections between different aspects of the topic are drawn",
        "analytical_depth",
        source="general",
    ),
    Criterion(
        "The report distinguishes between well-established findings and emerging or contested claims",
        "analytical_depth",
        source="general",
    ),
]

CITATION_CRITERIA: list[Criterion] = [
    Criterion(
        "Claims are attributed to named sources with inline citations",
        "citation_quality",
        source="general",
    ),
    Criterion(
        "Citations are formatted consistently throughout the report",
        "citation_quality",
        source="general",
    ),
    Criterion(
        "The number of distinct sources cited is appropriate for the topic scope (minimum 5)",
        "citation_quality",
        source="general",
    ),
    Criterion(
        "Different sections draw from different sources rather than relying on a single source",
        "citation_quality",
        source="general",
    ),
]

ORGANIZATION_CRITERIA: list[Criterion] = [
    Criterion(
        "The report has a clear introduction that frames the topic",
        "organization",
        source="general",
    ),
    Criterion(
        "Sections follow a logical progression",
        "organization",
        source="general",
    ),
    Criterion(
        "The report has a conclusion that synthesizes key findings",
        "organization",
        source="general",
    ),
    Criterion(
        "Paragraphs are focused and transitions between topics are smooth",
        "organization",
        source="general",
    ),
]

INSTRUCTION_FOLLOWING_CRITERIA: list[Criterion] = [
    Criterion(
        "The report directly addresses the specific research question asked",
        "instruction_following",
        source="general",
    ),
    Criterion(
        "The scope of the report is appropriate to the query (not too narrow or too broad)",
        "instruction_following",
        source="general",
    ),
    Criterion(
        "The report addresses all sub-questions or dimensions implied by the query",
        "instruction_following",
        source="general",
    ),
    Criterion(
        "The format and structure match what the query implies (e.g., comparison queries produce comparative analysis)",
        "instruction_following",
        source="general",
    ),
]

ATTRIBUTION_CRITERIA: list[Criterion] = [
    Criterion(
        "Each major claim or finding is traceable to a named source",
        "attribution_quality",
        source="general",
    ),
    Criterion(
        "The report clearly distinguishes between the author's analysis and source material",
        "attribution_quality",
        source="general",
    ),
]

# New dimensions from DRB-II (information recall) and systematic survey (logical coherence)

INFORMATION_RECALL_CRITERIA: list[Criterion] = [
    Criterion(
        "The report retrieves and includes the key facts needed to answer the research question",
        "information_recall",
        source="general",
    ),
    Criterion(
        "Important quantitative data (statistics, measurements, benchmarks) relevant to the query are present",
        "information_recall",
        source="general",
    ),
    Criterion(
        "The report identifies and includes the most authoritative or seminal sources on the topic",
        "information_recall",
        source="general",
    ),
    Criterion(
        "No critical pieces of widely-available evidence on the topic are omitted",
        "information_recall",
        source="general",
    ),
]

LOGICAL_COHERENCE_CRITERIA: list[Criterion] = [
    Criterion(
        "The report does not contain internal contradictions between sections or claims",
        "logical_coherence",
        source="general",
    ),
    Criterion(
        "Conclusions follow logically from the evidence and arguments presented",
        "logical_coherence",
        source="general",
    ),
    Criterion(
        "Comparative claims and causal arguments are supported by valid reasoning",
        "logical_coherence",
        source="general",
    ),
]


def build_general_criteria() -> list[Criterion]:
    """Build the full set of general (non-task-specific) criteria."""
    return (
        INFORMATION_RECALL_CRITERIA
        + FACTUAL_ACCURACY_CRITERIA
        + COVERAGE_CRITERIA
        + ANALYTICAL_DEPTH_CRITERIA
        + CITATION_CRITERIA
        + LOGICAL_COHERENCE_CRITERIA
        + ORGANIZATION_CRITERIA
        + INSTRUCTION_FOLLOWING_CRITERIA
        + ATTRIBUTION_CRITERIA
    )


def build_rubric_v2(
    query_id: str,
    query_text: str,
    task_specific_criteria: list[Criterion] | None = None,
    coverage_criteria: list[Criterion] | None = None,
    dimension_weights: dict[str, float] | None = None,
    source_type: str | None = None,
) -> RubricV2:
    """Build a complete V2 rubric for a query.

    Args:
        query_id: Unique query identifier
        query_text: The research query text
        task_specific_criteria: Optional task-specific criteria (from LLM or benchmark)
        coverage_criteria: Optional query-specific coverage criteria (replaces generic coverage)
        dimension_weights: Optional custom weights (defaults to DIMENSION_WEIGHTS_V2).
            Takes precedence over *source_type* if both are provided.
        source_type: Optional source type (e.g. ``"litqa2"``, ``"draco"``) to
            select source-specific default weights via
            :func:`get_dimension_weights`.  Ignored when *dimension_weights*
            is explicitly provided.

    Returns:
        Complete RubricV2 with 30-50 criteria
    """
    criteria = build_general_criteria()

    if coverage_criteria:
        # Replace generic coverage with query-specific
        criteria = [c for c in criteria if c.dimension != "coverage"] + coverage_criteria

    if task_specific_criteria:
        criteria.extend(task_specific_criteria)

    if dimension_weights is not None:
        weights = dimension_weights
    elif source_type is not None:
        weights = get_dimension_weights(source_type)
    else:
        weights = DIMENSION_WEIGHTS_V2.copy()

    return RubricV2(
        query_id=query_id,
        query_text=query_text,
        criteria=criteria,
        dimension_weights=weights,
    )


def build_rubric_from_test_query(test_query) -> RubricV2:
    """Build V2 rubric from an existing TestQuery (backward compat with test_queries.py).

    Converts expected_elements into coverage criteria.  The TestQuery's
    ``expected_elements`` list is translated into per-element coverage criteria
    that replace the generic coverage set.  All other general criteria are
    included unchanged.

    Args:
        test_query: A ``TestQuery`` instance from ``test_queries.py``.

    Returns:
        A ``RubricV2`` with the standard general criteria plus coverage criteria
        derived from the test query's expected elements.
    """
    coverage: list[Criterion] = [
        Criterion(
            text=f"The report covers: {element}",
            dimension="coverage",
            source="task_specific",
        )
        for element in test_query.expected_elements
    ]

    return build_rubric_v2(
        query_id=test_query.id,
        query_text=test_query.query,
        coverage_criteria=coverage,
    )


def build_rubric_from_draco(
    query_id: str,
    query_text: str,
    draco_sections: list[dict],
    draco_criteria: list[dict],
) -> RubricV2:
    """Convert DRACO rubric format to V2 rubric, preserving weights.

    DRACO rubrics contain *sections* (each with a title) and *criteria* (each
    with a ``weight``, ``requirement``/``description``, and optional ``id``).
    Positive-weight criteria are converted to ``Criterion`` objects; negative-
    weight criteria (critical failures) are also included with their original
    weight so the downstream scorer can apply the penalty.

    Args:
        query_id: Unique identifier for the query.
        query_text: The research query text.
        draco_sections: List of section dicts, each having at least a
            ``"title"`` key.  Used to assign a section label to criteria.
        draco_criteria: List of criterion dicts, each having ``"weight"``
            (int), ``"requirement"`` or ``"description"`` (str), and
            optionally ``"section"`` or ``"section_title"`` to link the
            criterion to a section.

    Returns:
        A ``RubricV2`` that includes DRACO criteria (with their original
        weights) plus the standard V2 general criteria.
    """
    # Build a section-title lookup for labelling
    section_titles: dict[str, str] = {}
    for sec in draco_sections:
        sec_id = sec.get("id", sec.get("title", ""))
        sec_title = sec.get("title", sec_id)
        section_titles[sec_id] = sec_title

    task_criteria: list[Criterion] = []
    for crit in draco_criteria:
        description = crit.get("requirement", crit.get("description", ""))
        weight = crit.get("weight", 1)
        # Try to resolve a human-readable section label
        sec_ref = crit.get("section", crit.get("section_title", ""))
        section_label = section_titles.get(sec_ref, sec_ref)

        # Map DRACO criteria into the most appropriate V2 dimension.
        # All DRACO criteria are topic-coverage or factual checks, so we
        # default to "coverage" but allow callers to reclassify later.
        dimension = "coverage"

        task_criteria.append(
            Criterion(
                text=description,
                dimension=dimension,
                weight=float(weight),
                source="draco",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
    )


def rubric_to_judge_prompt(rubric: RubricV2) -> str:
    """Convert V2 rubric to the prompt format expected by the LLM judge.

    Produces numbered criteria list grouped by dimension with instruction text.
    The output is suitable for embedding in the user message of a judge LLM
    call.

    Returns:
        A multi-line string with numbered criteria grouped by dimension.
    """
    # Group criteria by dimension while preserving insertion order
    grouped: dict[str, list[Criterion]] = {}
    for c in rubric.criteria:
        grouped.setdefault(c.dimension, []).append(c)

    lines: list[str] = []
    lines.append(
        "Evaluate the following research report against each criterion below."
    )
    lines.append(
        "For EACH criterion, provide a verdict of SATISFIED or NOT_SATISFIED,"
    )
    lines.append("along with brief evidence and one-sentence reasoning.")
    lines.append("")

    global_idx = 0
    for dimension, criteria in grouped.items():
        weight = rubric.dimension_weights.get(dimension, 0.0)
        weight_pct = weight * 100
        lines.append(
            f"### {dimension.replace('_', ' ').title()} (weight: {weight_pct:.0f}%)"
        )
        for crit in criteria:
            lines.append(f"  {global_idx}. [{dimension}] {crit.text}")
            global_idx += 1
        lines.append("")

    lines.append(f"Total criteria: {global_idx}")
    lines.append("")
    lines.append("Return JSON in this exact format:")
    lines.append("{")
    lines.append('  "evaluations": [')
    lines.append("    {")
    lines.append('      "criterion_index": 0,')
    lines.append('      "verdict": "SATISFIED" or "NOT_SATISFIED",')
    lines.append('      "evidence": "brief quote or reference from the report",')
    lines.append('      "reasoning": "one sentence explanation"')
    lines.append("    },")
    lines.append("    ...")
    lines.append("  ]")
    lines.append("}")

    return "\n".join(lines)


def rubric_to_judge_prompt_with_mapping(
    rubric: RubricV2, seed: int
) -> tuple[str, list[int]]:
    """Convert V2 rubric to judge prompt with shuffled criteria order.

    Shuffles the flat list of criteria using the given seed so that different
    judge passes see criteria in different orders (mitigating position bias).
    Returns both the prompt string and a mapping from shuffled index to the
    original ``rubric.criteria`` index so that downstream verdict parsing can
    recover the correct criterion.

    Args:
        rubric: The rubric to convert.
        seed: Deterministic RNG seed for reproducible shuffling.

    Returns:
        A tuple of ``(prompt_str, mapping)`` where ``mapping[i]`` is the
        original ``rubric.criteria`` index for shuffled position ``i``.
    """
    rng = random.Random(seed)

    # Build (original_index, criterion) pairs and shuffle
    indexed: list[tuple[int, Criterion]] = list(enumerate(rubric.criteria))
    rng.shuffle(indexed)

    # mapping[shuffled_pos] = original_criteria_index
    mapping: list[int] = [orig_idx for orig_idx, _ in indexed]

    lines: list[str] = []
    lines.append(
        "Evaluate the following research report against each criterion below."
    )
    lines.append(
        "For EACH criterion, provide a verdict of SATISFIED or NOT_SATISFIED,"
    )
    lines.append("along with brief evidence and one-sentence reasoning.")
    lines.append("")

    for shuffled_idx, (orig_idx, crit) in enumerate(indexed):
        weight_str = ""
        if crit.weight != 1.0:
            weight_str = f" (weight: {crit.weight})"
        lines.append(
            f"  {shuffled_idx}. [orig={orig_idx}] [{crit.dimension}] {crit.text}{weight_str}"
        )

    lines.append("")
    lines.append(f"Total criteria: {len(indexed)}")
    lines.append("")
    lines.append("Return JSON in this exact format:")
    lines.append("{")
    lines.append('  "evaluations": [')
    lines.append("    {")
    lines.append('      "criterion_index": 0,')
    lines.append('      "verdict": "SATISFIED" or "NOT_SATISFIED",')
    lines.append('      "evidence": "brief quote or reference from the report",')
    lines.append('      "reasoning": "one sentence explanation"')
    lines.append("    },")
    lines.append("    ...")
    lines.append("  ]")
    lines.append("}")

    return "\n".join(lines), mapping
