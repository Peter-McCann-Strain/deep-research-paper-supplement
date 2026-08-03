"""Rubric format converters for benchmark datasets.

Converts DRACO, ResearchQA, DeepSearchQA, LitQA2, ResearchRubrics,
DRB-II, and DR.BENCH rubric formats to the unified RubricV2 format,
preserving benchmark-specific weights and criteria semantics.
"""

from __future__ import annotations

from deep_research.evaluation.rubric_v2 import (
    Criterion,
    RubricV2,
    DIMENSION_WEIGHTS_V2,
    build_general_criteria,
    build_rubric_v2,
    build_rubric_from_test_query,
)


# ── DRACO section-name to V2 dimension mapping ──────────────────────────────

DRACO_DIMENSION_MAP: dict[str, str] = {
    # Exact section titles from cached DRACO data
    "factual accuracy": "factual_accuracy",
    "factual_accuracy": "factual_accuracy",
    "accuracy": "factual_accuracy",
    "breadth and depth of analysis": "coverage",
    "breadth_and_depth": "coverage",
    "breadth and depth": "coverage",
    "breadth": "coverage",
    "depth": "analytical_depth",
    "analysis": "analytical_depth",
    "depth of analysis": "analytical_depth",
    "presentation quality": "organization",
    "presentation": "organization",
    "presentation_quality": "organization",
    "citation quality": "citation_quality",
    "citation": "citation_quality",
    "citation_quality": "citation_quality",
    "completeness": "coverage",
    "relevance": "instruction_following",
    "organization": "organization",
    "structure": "organization",
}


def _infer_dimension(description: str) -> str:
    """Infer the V2 dimension from criterion text when section mapping fails."""
    desc_lower = description.lower()
    # Check factual_accuracy early since "factual" is unambiguous
    if any(w in desc_lower for w in ["factual", "accura", "correct", "error", "wrong", "false"]):
        return "factual_accuracy"
    if any(w in desc_lower for w in ["cite", "source", "reference", "bibliography"]):
        return "citation_quality"
    if any(w in desc_lower for w in ["attribut", "traceab"]):
        return "attribution_quality"
    if any(w in desc_lower for w in ["recall", "retrieve"]):
        return "information_recall"
    # Use " logical " with spaces or "coherenc" to avoid matching "geological"
    if any(w in desc_lower for w in ["contradict", "coherent", "coherenc", "inconsisten"]):
        return "logical_coherence"
    if " logical " in f" {desc_lower} ":
        return "logical_coherence"
    if any(w in desc_lower for w in ["analyze", "compar", "evaluat", "synthesiz", "depth"]):
        return "analytical_depth"
    if any(w in desc_lower for w in ["organiz", "structur", "section", "format", "present"]):
        return "organization"
    if any(w in desc_lower for w in ["follow", "instruct", "address", "scope"]):
        return "instruction_following"
    return "coverage"


def draco_to_rubric_v2(
    query_id: str,
    query_text: str,
    sections: list[dict],
) -> RubricV2:
    """Convert DRACO rubric to V2, preserving criterion weights.

    DRACO criteria have integer weights (positive for requirements, negative
    for penalties).  These are preserved in the Criterion.weight field.

    DRACO sections map to V2 dimensions via ``DRACO_DIMENSION_MAP``.
    Negative-weight criteria (penalties) are mapped to ``factual_accuracy``
    since they represent errors the report should avoid.

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        sections: DRACO rubric in cached format -- a list of dicts, but
            typically presented as ``{section_title: [criteria]}`` when read
            from the ``BenchmarkQuery.rubric`` dict.  This function also
            accepts the raw dict directly (auto-detected).

    Returns:
        RubricV2 combining DRACO task-specific criteria with V2 general
        criteria.
    """
    task_criteria: list[Criterion] = []

    # The DRACO rubric in the cache is stored as a dict:
    #   {"Factual Accuracy": [{"id": ..., "weight": ..., "description": ...}, ...], ...}
    # Accept both dict (from BenchmarkQuery.rubric) and list-of-dicts forms.
    if isinstance(sections, dict):
        section_items: list[tuple[str, list[dict]]] = list(sections.items())
    else:
        # list-of-dicts form: [{"name": ..., "criteria": [...], "weight": ...}]
        section_items = [
            (sec.get("name", sec.get("title", "")), sec.get("criteria", []))
            for sec in sections
        ]

    for section_name, criteria_list in section_items:
        if not isinstance(criteria_list, list):
            continue

        # Resolve the V2 dimension for this section
        section_key = section_name.lower().strip()
        dimension = DRACO_DIMENSION_MAP.get(section_key, "")

        for crit in criteria_list:
            description = crit.get("description", crit.get("requirement", crit.get("text", "")))
            weight = crit.get("weight", 1)

            if not description:
                continue

            # For negative-weight criteria (penalties), force factual_accuracy
            if weight < 0:
                crit_dimension = "factual_accuracy"
                text = f"The report avoids: {description}"
            else:
                crit_dimension = dimension or _infer_dimension(description)
                text = f"The report addresses: {description}"

            task_criteria.append(
                Criterion(
                    text=text,
                    dimension=crit_dimension,
                    weight=float(weight),
                    source="draco",
                )
            )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
    )


def research_qa_to_rubric_v2(
    query_id: str,
    query_text: str,
    rubric_items: list[dict],
) -> RubricV2:
    """Convert ResearchQA rubric items to V2.

    ResearchQA rubric items are yes/no questions about report content, each
    with a ``type`` list (e.g. ``["Citation", "Comparison"]``) and optional
    citation metadata.  The type tags are used to map each item to the most
    appropriate V2 dimension.

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        rubric_items: List of dicts with ``question``, ``type``, and optional
            ``citation_metadata`` fields.

    Returns:
        RubricV2 with ResearchQA items as task-specific criteria.
    """
    # Type tag -> V2 dimension mapping
    type_to_dimension: dict[str, str] = {
        "citation": "citation_quality",
        "comparison": "analytical_depth",
        "example": "coverage",
        "impact": "coverage",
        "limitation": "analytical_depth",
        "other": "coverage",
    }

    task_criteria: list[Criterion] = []
    for item in rubric_items:
        question = item.get("question", "")
        if not question:
            continue

        item_types = item.get("type", [])
        if not isinstance(item_types, list):
            item_types = [str(item_types)]

        # Determine dimension from type tags (first match wins)
        dimension = "coverage"  # default
        for t in item_types:
            mapped = type_to_dimension.get(t.lower().strip())
            if mapped:
                dimension = mapped
                break

        task_criteria.append(
            Criterion(
                text=question,
                dimension=dimension,
                weight=1.0,
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
    )


def deepsearch_qa_to_rubric_v2(
    query_id: str,
    query_text: str,
    reference_answer: str,
    evaluation_criteria: list[str],
) -> RubricV2:
    """Convert DeepSearchQA evaluation criteria to V2.

    DeepSearchQA queries have an expected answer and answer type but no
    per-criterion rubric.  We synthesise criteria from the expected answer
    and generic deep-search evaluation axes (correctness, evidence,
    multi-source synthesis, citation quality).

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        reference_answer: The expected answer string.
        evaluation_criteria: Additional criterion strings (e.g. answer_type
            specific).  May be empty, in which case only synthesised criteria
            are used.

    Returns:
        RubricV2 with answer-correctness and evidence criteria.
    """
    task_criteria: list[Criterion] = []

    # Core answer-correctness criteria
    if reference_answer:
        truncated = reference_answer[:500]
        task_criteria.append(
            Criterion(
                text=f"The report contains or directly addresses the expected answer: {truncated}",
                dimension="factual_accuracy",
                weight=2.0,  # Higher weight -- primary goal
                source="benchmark",
            )
        )
        task_criteria.append(
            Criterion(
                text="The report provides the correct factual answer to the research question",
                dimension="factual_accuracy",
                weight=1.5,
                source="benchmark",
            )
        )

    # Evidence and synthesis criteria
    task_criteria.extend([
        Criterion(
            text="The report provides supporting evidence and reasoning for its answer",
            dimension="analytical_depth",
            source="benchmark",
        ),
        Criterion(
            text="The report synthesizes information from multiple sources",
            dimension="analytical_depth",
            source="benchmark",
        ),
        Criterion(
            text="The report cites specific, verifiable sources",
            dimension="citation_quality",
            source="benchmark",
        ),
    ])

    # Additional criteria from caller (e.g. answer-type specific)
    for crit_text in evaluation_criteria:
        task_criteria.append(
            Criterion(
                text=crit_text,
                dimension=_infer_dimension(crit_text),
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
    )


def litqa2_to_rubric_v2(
    query_id: str,
    query_text: str,
    correct_answer: str,
    distractors: list[str],
) -> RubricV2:
    """Convert LitQA2 MCQ format to V2.

    LitQA2 is multiple-choice, so criteria focus on whether the report
    identifies the correct answer and provides supporting evidence from
    the scientific literature.

    Args:
        query_id: Unique query identifier.
        query_text: The research question.
        correct_answer: The ideal/correct answer option.
        distractors: The incorrect answer options.

    Returns:
        RubricV2 with MCQ-specific task criteria.
    """
    task_criteria: list[Criterion] = [
        Criterion(
            text=f"The report identifies the correct answer: {correct_answer}",
            dimension="factual_accuracy",
            weight=2.0,
            source="benchmark",
        ),
        Criterion(
            text="The report provides scientific reasoning supporting its answer",
            dimension="analytical_depth",
            source="benchmark",
        ),
        Criterion(
            text="The report cites relevant scientific literature",
            dimension="citation_quality",
            source="benchmark",
        ),
        Criterion(
            text="The report demonstrates understanding of the underlying science",
            dimension="coverage",
            source="benchmark",
        ),
    ]

    if distractors:
        dist_text = ", ".join(distractors[:3])
        task_criteria.append(
            Criterion(
                text=(
                    f"The report correctly distinguishes the answer from "
                    f"plausible alternatives ({dist_text})"
                ),
                dimension="analytical_depth",
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
    )


def test_query_to_rubric_v2(test_query) -> RubricV2:
    """Convert an existing TestQuery to V2 (backward compatibility).

    Delegates to :func:`build_rubric_from_test_query` from ``rubric_v2``,
    which turns ``expected_elements`` into coverage criteria.

    Args:
        test_query: A ``TestQuery`` instance from ``test_queries.py``.

    Returns:
        RubricV2 built from the test query's expected elements.
    """
    return build_rubric_from_test_query(test_query)


# ── DRB-II category to V2 dimension mapping ──────────────────────────────

DRB2_CATEGORY_MAP: dict[str, str] = {
    "information_recall": "information_recall",
    "factual_accuracy": "factual_accuracy",
    "completeness": "coverage",
    "organization": "organization",
    "citation": "citation_quality",
}


def research_rubrics_to_rubric_v2(
    query_id: str,
    query_text: str,
    criteria_list: list[dict],
) -> RubricV2:
    """Convert ScaleAI ResearchRubrics format to V2.

    ResearchRubrics provides per-prompt criteria with a ``dimension_hint``
    and ``weight`` for each criterion.  The dimension hint is mapped to V2
    dimensions using :func:`_infer_dimension` when the hint does not match
    a known V2 dimension name directly.

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        criteria_list: List of dicts, each with ``criterion_text``,
            ``dimension_hint`` (str), and ``weight`` (float).

    Returns:
        RubricV2 with ResearchRubrics criteria as task-specific criteria,
        using ``research_rubrics`` source-type dimension weights.
    """
    task_criteria: list[Criterion] = []

    for item in criteria_list:
        text = item.get("criterion_text", "")
        if not text:
            continue

        hint = item.get("dimension_hint", "")
        weight = float(item.get("weight", 1.0))

        # Try direct match to V2 dimension names first
        known_dimensions = {
            "information_recall", "factual_accuracy", "coverage",
            "analytical_depth", "citation_quality", "logical_coherence",
            "organization", "instruction_following", "attribution_quality",
        }
        if hint.lower().strip() in known_dimensions:
            dimension = hint.lower().strip()
        else:
            # Try hint first, fall back to criterion text for inference
            dimension = _infer_dimension(hint)
            if dimension == "coverage" and text:
                # "coverage" is the default fallback; try criterion text for a better match
                text_dimension = _infer_dimension(text)
                if text_dimension != "coverage":
                    dimension = text_dimension

        task_criteria.append(
            Criterion(
                text=text,
                dimension=dimension,
                weight=weight,
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
        source_type="research_rubrics",
    )


def drb2_to_rubric_v2(
    query_id: str,
    query_text: str,
    rubric_items: list[dict],
) -> RubricV2:
    """Convert DRB-II binary rubric format to V2.

    DRB-II rubric items are binary yes/no questions about report content,
    each with a ``category`` and ``weight``.  Categories are mapped to V2
    dimensions via :data:`DRB2_CATEGORY_MAP`.

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        rubric_items: List of dicts, each with ``question`` (str, binary
            yes/no phrasing), ``category`` (str), and ``weight`` (float).

    Returns:
        RubricV2 with DRB-II items as task-specific criteria, using
        ``drb2`` source-type dimension weights.
    """
    task_criteria: list[Criterion] = []

    for item in rubric_items:
        question = item.get("question", "")
        if not question:
            continue

        category = item.get("category", "").lower().strip()
        weight = float(item.get("weight", 1.0))

        dimension = DRB2_CATEGORY_MAP.get(category, "")
        if not dimension:
            dimension = _infer_dimension(question)

        task_criteria.append(
            Criterion(
                text=question,
                dimension=dimension,
                weight=weight,
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
        source_type="drb2",
    )


def drbench_to_rubric_v2(
    query_id: str,
    query_text: str,
    reference_facts: list[str],
    evaluation_axes: list[str],
) -> RubricV2:
    """Convert DR.BENCH format to V2.

    DR.BENCH tasks include reference facts that the report should contain
    and evaluation axes that describe the quality dimensions to assess.
    Each reference fact becomes a ``factual_accuracy`` criterion, and each
    evaluation axis becomes a criterion with its dimension inferred from
    the axis text.

    Args:
        query_id: Unique query identifier.
        query_text: The research query text.
        reference_facts: List of factual statements the report should include.
        evaluation_axes: List of evaluation axis descriptions.

    Returns:
        RubricV2 with DR.BENCH items as task-specific criteria, using
        ``drbench`` source-type dimension weights.
    """
    task_criteria: list[Criterion] = []

    for fact in reference_facts:
        if not fact:
            continue
        task_criteria.append(
            Criterion(
                text=f"The report includes the fact: {fact}",
                dimension="factual_accuracy",
                weight=1.0,
                source="benchmark",
            )
        )

    for axis in evaluation_axes:
        if not axis:
            continue
        task_criteria.append(
            Criterion(
                text=axis,
                dimension=_infer_dimension(axis),
                weight=1.0,
                source="benchmark",
            )
        )

    return build_rubric_v2(
        query_id=query_id,
        query_text=query_text,
        task_specific_criteria=task_criteria,
        source_type="drbench",
    )
