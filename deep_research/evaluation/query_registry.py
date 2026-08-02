"""Unified query registry for evaluation across all benchmark sources.

Provides a single interface for loading, filtering, and managing evaluation
queries from custom test queries, DRACO, DeepSearchQA, ResearchQA, and LitQA2.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from deep_research.evaluation.rubric_v2 import RubricV2, Criterion

log = logging.getLogger(__name__)


# ── Unified query model ──────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    """Unified evaluation query from any source."""

    id: str
    query: str
    source: str  # "custom", "draco", "deepsearch_qa", "research_qa", "litqa2"
    domain: str  # subject domain
    difficulty: str  # "simple", "moderate", "complex"
    rubric: RubricV2
    expected_elements: list[str] = field(default_factory=list)
    reference_answer: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict (for manifests)."""
        return {
            "id": self.id,
            "query": self.query,
            "source": self.source,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "expected_elements": self.expected_elements,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
            "rubric": {
                "query_id": self.rubric.query_id,
                "query_text": self.rubric.query_text,
                "dimension_weights": self.rubric.dimension_weights,
                "criteria": [
                    {
                        "text": c.text,
                        "dimension": c.dimension,
                        "weight": c.weight,
                        "source": c.source,
                    }
                    for c in self.rubric.criteria
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalQuery:
        """Reconstruct from a serialised dict (manifest loading)."""
        rubric_data = data["rubric"]
        criteria = [
            Criterion(
                text=c["text"],
                dimension=c["dimension"],
                weight=c.get("weight", 1.0),
                source=c.get("source", "general"),
            )
            for c in rubric_data["criteria"]
        ]
        rubric = RubricV2(
            query_id=rubric_data["query_id"],
            query_text=rubric_data["query_text"],
            criteria=criteria,
            dimension_weights=rubric_data["dimension_weights"],
        )
        return cls(
            id=data["id"],
            query=data["query"],
            source=data["source"],
            domain=data["domain"],
            difficulty=data["difficulty"],
            rubric=rubric,
            expected_elements=data.get("expected_elements", []),
            reference_answer=data.get("reference_answer", ""),
            metadata=data.get("metadata", {}),
        )


# ── Difficulty classification heuristic ──────────────────────────────────────

_COMPLEX_INDICATORS = [
    "compare and contrast",
    "analyze the tradeoffs",
    "evaluate the implications",
    "multi-step",
    "discuss the advantages and disadvantages",
    "critically assess",
    "how does .* affect",
    "what are the tradeoffs",
]

_SIMPLE_INDICATORS = [
    "what is",
    "who is",
    "define",
    "when did",
    "where is",
    "name the",
    "list the",
    "how many",
]


def classify_difficulty(query: str, *, default: str = "moderate") -> str:
    """Classify a query as simple, moderate, or complex.

    Uses a keyword heuristic:
    - Multi-part questions, comparisons, tradeoff analysis -> complex
    - Single factual questions -> simple
    - Everything else -> moderate

    If the query already carries a difficulty label that is one of
    ``simple``, ``moderate``, or ``complex``, that label is returned unchanged.
    """
    if default in ("simple", "moderate", "complex"):
        pass  # valid default

    q_lower = query.lower()
    word_count = len(query.split())

    # Multi-part questions (contains multiple question marks)
    if query.count("?") >= 2:
        return "complex"

    for indicator in _COMPLEX_INDICATORS:
        if indicator in q_lower:
            return "complex"

    for indicator in _SIMPLE_INDICATORS:
        if q_lower.startswith(indicator):
            if word_count < 20:
                return "simple"
            return "moderate"

    # Long queries tend to be more complex
    if word_count > 40:
        return "complex"
    if word_count < 15:
        return "simple"

    return default


def validate_difficulty_classification(queries: list[EvalQuery]) -> dict:
    """Validate the difficulty classifier against query properties.

    Runs heuristic checks to flag potential misclassifications where the
    assigned difficulty seems inconsistent with observable query features
    (word count, number of sub-questions, etc.).

    Args:
        queries: List of :class:`EvalQuery` objects to validate.

    Returns:
        A dict with keys:
        - ``distribution``: count per difficulty level.
        - ``flagged``: list of dicts describing potentially misclassified queries.
        - ``total``: total number of queries checked.
        - ``note``: caveat about the heuristic nature of these checks.
    """
    distribution = Counter(q.difficulty for q in queries)
    flagged: list[dict] = []

    for q in queries:
        word_count = len(q.query.split())
        question_marks = q.query.count("?")

        # Short queries (< 15 words) classified as complex
        if q.difficulty == "complex" and word_count < 15 and question_marks < 2:
            flagged.append({
                "id": q.id,
                "query": q.query,
                "difficulty": q.difficulty,
                "word_count": word_count,
                "reason": "Short query (< 15 words, single question) classified as complex",
            })

        # Single-question queries classified as complex (no complex indicators)
        if q.difficulty == "complex" and question_marks <= 1 and word_count < 20:
            q_lower = q.query.lower()
            has_complex_indicator = any(
                ind in q_lower for ind in _COMPLEX_INDICATORS
            )
            if not has_complex_indicator:
                # Avoid duplicate if already flagged above
                already = any(f["id"] == q.id for f in flagged)
                if not already:
                    flagged.append({
                        "id": q.id,
                        "query": q.query,
                        "difficulty": q.difficulty,
                        "word_count": word_count,
                        "reason": (
                            "Single-question query without complex indicators "
                            "classified as complex"
                        ),
                    })

        # Long multi-part queries classified as simple
        if q.difficulty == "simple" and (word_count > 30 or question_marks >= 2):
            flagged.append({
                "id": q.id,
                "query": q.query,
                "difficulty": q.difficulty,
                "word_count": word_count,
                "reason": (
                    f"Long/multi-part query (words={word_count}, "
                    f"questions={question_marks}) classified as simple"
                ),
            })

    return {
        "distribution": dict(distribution),
        "flagged": flagged,
        "total": len(queries),
        "note": (
            "Difficulty classification uses keyword heuristics and word-count "
            "thresholds. Flagged items are not necessarily wrong -- they "
            "indicate cases where the heuristic may have produced a "
            "surprising result and manual review is recommended."
        ),
    }


# ── Stratified sampling helpers ──────────────────────────────────────────────

def _stratified_sample(
    items: list,
    max_items: int,
    key_fn,
    seed: int = 42,
) -> list:
    """Sample *max_items* from *items* with equal representation per group.

    Groups are defined by ``key_fn(item)``.  If a group has fewer than its
    share, the surplus is redistributed to larger groups.
    """
    if max_items <= 0 or not items:
        return []
    if len(items) <= max_items:
        return list(items)

    rng = random.Random(seed)

    # Group items
    groups: dict[str, list] = {}
    for item in items:
        k = key_fn(item)
        groups.setdefault(k, []).append(item)

    n_groups = len(groups)
    per_group = max(1, max_items // n_groups)
    selected: list = []

    # First pass: take per_group from each
    remaining_budget = max_items
    leftover: list = []
    for g_items in groups.values():
        rng.shuffle(g_items)
        take = min(per_group, len(g_items), remaining_budget)
        selected.extend(g_items[:take])
        remaining_budget -= take
        leftover.extend(g_items[take:])

    # Second pass: fill remaining budget from leftover
    if remaining_budget > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:remaining_budget])

    return selected


# ── Query registry ───────────────────────────────────────────────────────────

class QueryRegistry:
    """Central registry of all evaluation queries."""

    def __init__(self, data_dir: Path | None = None):
        self._queries: list[EvalQuery] = []
        self._data_dir = data_dir or Path("data")

    # ── Loading individual sources ───────────────────────────────────────

    def load_custom_queries(self) -> list[EvalQuery]:
        """Load the 5 existing custom test queries."""
        from deep_research.evaluation.test_queries import TEST_QUERIES
        from deep_research.evaluation.rubric_converters import test_query_to_rubric_v2

        loaded: list[EvalQuery] = []
        for tq in TEST_QUERIES:
            rubric = test_query_to_rubric_v2(tq)
            eq = EvalQuery(
                id=tq.id,
                query=tq.query,
                source="custom",
                domain="nlp",  # all 5 custom queries are NLP-related
                difficulty=tq.difficulty,
                rubric=rubric,
                expected_elements=list(tq.expected_elements),
                reference_answer="",
                metadata={
                    "description": tq.description,
                    "expected_sources": list(tq.expected_sources),
                },
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d custom queries", len(loaded))
        return loaded

    def load_draco_queries(
        self, max_queries: int = 40, stratified: bool = True
    ) -> list[EvalQuery]:
        """Load DRACO queries with stratified sampling across domains.

        If ``stratified=True``, samples equally from each of the 10 DRACO
        domains.  Falls back gracefully if the cache file is missing.
        """
        from deep_research.evaluation.rubric_converters import draco_to_rubric_v2

        cache_path = self._data_dir / "benchmarks" / "draco" / "draco_queries.json"
        if not cache_path.exists():
            log.warning("DRACO cache not found at %s -- skipping", cache_path)
            return []

        raw = json.loads(cache_path.read_text())

        # Stratified or simple truncation
        if stratified:
            sampled = _stratified_sample(
                raw, max_queries, key_fn=lambda q: q.get("domain", "")
            )
        else:
            sampled = raw[:max_queries]

        loaded: list[EvalQuery] = []
        for bq in sampled:
            query_id = bq["id"]
            query_text = bq["query"]
            rubric_data = bq.get("rubric", {})
            domain = bq.get("domain", "")

            rubric = draco_to_rubric_v2(query_id, query_text, rubric_data)

            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="draco",
                domain=domain,
                difficulty=difficulty,
                rubric=rubric,
                reference_answer=bq.get("reference_answer", ""),
                metadata=bq.get("metadata", {}),
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d DRACO queries (stratified=%s)", len(loaded), stratified)
        return loaded

    def load_deepsearch_queries(self, max_queries: int = 20) -> list[EvalQuery]:
        """Load DeepSearchQA queries with field diversity."""
        from deep_research.evaluation.rubric_converters import deepsearch_qa_to_rubric_v2

        cache_path = (
            self._data_dir / "benchmarks" / "deepsearch_qa" / "deepsearch_qa_queries.json"
        )
        if not cache_path.exists():
            log.warning("DeepSearchQA cache not found at %s -- skipping", cache_path)
            return []

        raw = json.loads(cache_path.read_text())

        # Stratified sample across problem_category (domain)
        sampled = _stratified_sample(
            raw, max_queries, key_fn=lambda q: q.get("domain", "")
        )

        loaded: list[EvalQuery] = []
        for bq in sampled:
            query_id = bq["id"]
            query_text = bq["query"]
            ref_answer = bq.get("reference_answer", "")
            rubric_raw = bq.get("rubric", {})
            answer_type = rubric_raw.get("answer_type", "")

            # Build answer-type-specific criteria
            extra_criteria: list[str] = []
            if answer_type == "Set Answer":
                extra_criteria.append(
                    "The report provides a comprehensive list addressing all "
                    "parts of the question"
                )
            else:
                extra_criteria.append(
                    "The report directly addresses the specific question asked"
                )

            rubric = deepsearch_qa_to_rubric_v2(
                query_id, query_text, ref_answer, extra_criteria
            )

            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="deepsearch_qa",
                domain=bq.get("domain", ""),
                difficulty=difficulty,
                rubric=rubric,
                reference_answer=ref_answer,
                metadata=bq.get("metadata", {}),
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d DeepSearchQA queries", len(loaded))
        return loaded

    def load_research_qa_queries(self, max_queries: int = 15) -> list[EvalQuery]:
        """Load ResearchQA queries across academic fields.

        Samples from distinct ``field`` values in the metadata for maximum
        domain diversity (75 fields available, 50 queries each).
        """
        from deep_research.evaluation.rubric_converters import research_qa_to_rubric_v2

        cache_path = (
            self._data_dir / "benchmarks" / "research_qa" / "research_qa_queries.json"
        )
        if not cache_path.exists():
            log.warning("ResearchQA cache not found at %s -- skipping", cache_path)
            return []

        raw = json.loads(cache_path.read_text())

        # Stratified sample across field metadata
        sampled = _stratified_sample(
            raw,
            max_queries,
            key_fn=lambda q: q.get("metadata", {}).get("field", q.get("domain", "")),
        )

        loaded: list[EvalQuery] = []
        for bq in sampled:
            query_id = bq["id"]
            query_text = bq["query"]
            rubric_raw = bq.get("rubric", {})
            rubric_items = rubric_raw.get("criteria", [])

            rubric = research_qa_to_rubric_v2(query_id, query_text, rubric_items)

            field_name = bq.get("metadata", {}).get("field", bq.get("domain", ""))
            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="research_qa",
                domain=field_name,
                difficulty=difficulty,
                rubric=rubric,
                reference_answer=bq.get("reference_answer", ""),
                metadata=bq.get("metadata", {}),
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d ResearchQA queries", len(loaded))
        return loaded

    def load_litqa2_queries(self, max_queries: int = 10) -> list[EvalQuery]:
        """Load LitQA2 scientific literature queries.

        Random sample since all are scientific literature MCQs.
        """
        from deep_research.evaluation.rubric_converters import litqa2_to_rubric_v2

        cache_path = self._data_dir / "benchmarks" / "litqa2" / "litqa2_queries.json"
        if not cache_path.exists():
            log.warning("LitQA2 cache not found at %s -- skipping", cache_path)
            return []

        raw = json.loads(cache_path.read_text())

        # Simple random sample (no stratification -- single domain)
        rng = random.Random(42)
        if len(raw) > max_queries > 0:
            sampled = rng.sample(raw, max_queries)
        else:
            sampled = raw

        loaded: list[EvalQuery] = []
        for bq in sampled:
            query_id = bq["id"]
            query_text = bq["query"]
            rubric_raw = bq.get("rubric", {})
            correct_answer = rubric_raw.get("ideal", bq.get("reference_answer", ""))
            distractors = rubric_raw.get("distractors", [])

            rubric = litqa2_to_rubric_v2(
                query_id, query_text, correct_answer, distractors
            )

            difficulty = classify_difficulty(query_text, default="complex")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="litqa2",
                domain="scientific_literature",
                difficulty=difficulty,
                rubric=rubric,
                reference_answer=correct_answer,
                expected_elements=[correct_answer] if correct_answer else [],
                metadata=bq.get("metadata", {}),
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d LitQA2 queries", len(loaded))
        return loaded

    # ── New benchmark loaders ────────────────────────────────────────────

    def load_research_rubrics_queries(
        self, data_path: Path | None = None, max_queries: int = 20
    ) -> list[EvalQuery]:
        """Load ScaleAI ResearchRubrics queries.

        Reads a JSON file with format::

            [{"id": str, "prompt": str, "criteria": [
                {"criterion_text": str, "dimension_hint": str, "weight": float}
            ]}]

        Criteria are attached as metadata for later rubric conversion via
        :func:`research_rubrics_to_rubric_v2`.

        Args:
            data_path: Path to JSON file. Defaults to
                ``<data_dir>/benchmarks/research_rubrics/research_rubrics_queries.json``.
            max_queries: Maximum number of queries to load.

        Returns:
            List of loaded :class:`EvalQuery` objects.
        """
        from deep_research.evaluation.rubric_converters import research_rubrics_to_rubric_v2

        if data_path is None:
            data_path = (
                self._data_dir
                / "benchmarks"
                / "research_rubrics"
                / "research_rubrics_queries.json"
            )

        if not data_path.exists():
            log.warning(
                "ResearchRubrics data not found at %s -- skipping", data_path
            )
            return []

        try:
            raw = json.loads(data_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read ResearchRubrics data: %s", exc)
            return []

        sampled = _stratified_sample(
            raw, max_queries, key_fn=lambda q: q.get("domain", "general")
        )

        loaded: list[EvalQuery] = []
        for item in sampled:
            query_id = item.get("id", "")
            query_text = item.get("prompt", "")
            criteria = item.get("criteria", [])

            if not query_id or not query_text:
                continue

            rubric = research_rubrics_to_rubric_v2(query_id, query_text, criteria)
            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="research_rubrics",
                domain=item.get("domain", "general"),
                difficulty=difficulty,
                rubric=rubric,
                metadata={"criteria": criteria},
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d ResearchRubrics queries", len(loaded))
        return loaded

    def load_drb2_queries(
        self, data_path: Path | None = None, max_queries: int = 20
    ) -> list[EvalQuery]:
        """Load DRB-II benchmark queries.

        Reads a JSON file with format::

            [{"task_id": str, "query": str, "rubric": [
                {"question": str, "category": str, "weight": float}
            ], "category": str}]

        Args:
            data_path: Path to JSON file. Defaults to
                ``<data_dir>/benchmarks/drb2/drb2_queries.json``.
            max_queries: Maximum number of queries to load.

        Returns:
            List of loaded :class:`EvalQuery` objects.
        """
        from deep_research.evaluation.rubric_converters import drb2_to_rubric_v2

        if data_path is None:
            data_path = (
                self._data_dir / "benchmarks" / "drb2" / "drb2_queries.json"
            )

        if not data_path.exists():
            log.warning("DRB-II data not found at %s -- skipping", data_path)
            return []

        try:
            raw = json.loads(data_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read DRB-II data: %s", exc)
            return []

        sampled = _stratified_sample(
            raw, max_queries, key_fn=lambda q: q.get("category", "general")
        )

        loaded: list[EvalQuery] = []
        for item in sampled:
            task_id = item.get("task_id", "")
            query_text = item.get("query", "")
            rubric_items = item.get("rubric", [])
            category = item.get("category", "general")

            if not task_id or not query_text:
                continue

            rubric = drb2_to_rubric_v2(task_id, query_text, rubric_items)
            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=task_id,
                query=query_text,
                source="drb2",
                domain=category,
                difficulty=difficulty,
                rubric=rubric,
                metadata={"category": category},
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d DRB-II queries", len(loaded))
        return loaded

    def load_drbench_queries(
        self, data_path: Path | None = None, max_queries: int = 20
    ) -> list[EvalQuery]:
        """Load DR.BENCH benchmark queries.

        Reads a JSON file with format::

            [{"id": str, "query": str, "reference_facts": [str],
              "evaluation_axes": [str]}]

        Args:
            data_path: Path to JSON file. Defaults to
                ``<data_dir>/benchmarks/drbench/drbench_queries.json``.
            max_queries: Maximum number of queries to load.

        Returns:
            List of loaded :class:`EvalQuery` objects.
        """
        from deep_research.evaluation.rubric_converters import drbench_to_rubric_v2

        if data_path is None:
            data_path = (
                self._data_dir / "benchmarks" / "drbench" / "drbench_queries.json"
            )

        if not data_path.exists():
            log.warning("DR.BENCH data not found at %s -- skipping", data_path)
            return []

        try:
            raw = json.loads(data_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read DR.BENCH data: %s", exc)
            return []

        sampled = _stratified_sample(
            raw, max_queries, key_fn=lambda q: q.get("domain", "general")
        )

        loaded: list[EvalQuery] = []
        for item in sampled:
            query_id = item.get("id", "")
            query_text = item.get("query", "")
            ref_facts = item.get("reference_facts", [])
            eval_axes = item.get("evaluation_axes", [])

            if not query_id or not query_text:
                continue

            rubric = drbench_to_rubric_v2(
                query_id, query_text, ref_facts, eval_axes
            )
            difficulty = classify_difficulty(query_text, default="moderate")

            eq = EvalQuery(
                id=query_id,
                query=query_text,
                source="drbench",
                domain=item.get("domain", "general"),
                difficulty=difficulty,
                rubric=rubric,
                reference_answer="",
                metadata={
                    "reference_facts": ref_facts,
                    "evaluation_axes": eval_axes,
                },
            )
            loaded.append(eq)

        self._queries.extend(loaded)
        log.info("Loaded %d DR.BENCH queries", len(loaded))
        return loaded

    # ── Unified loading ──────────────────────────────────────────────────

    def load_all(
        self,
        custom: int = 5,
        draco: int = 40,
        deepsearch: int = 20,
        research_qa: int = 15,
        litqa2: int = 10,
    ) -> list[EvalQuery]:
        """Load all queries from all sources with specified counts.

        Returns unified list of EvalQuery objects.
        Target total: custom + draco + deepsearch + research_qa + litqa2 = 90

        Sources whose cache files are missing are silently skipped.
        """
        # Reset to avoid duplicates on repeated calls
        self._queries = []

        if custom > 0:
            self.load_custom_queries()
        if draco > 0:
            self.load_draco_queries(max_queries=draco)
        if deepsearch > 0:
            self.load_deepsearch_queries(max_queries=deepsearch)
        if research_qa > 0:
            self.load_research_qa_queries(max_queries=research_qa)
        if litqa2 > 0:
            self.load_litqa2_queries(max_queries=litqa2)

        log.info("Total queries loaded: %d", len(self._queries))
        return list(self._queries)

    # ── Filtering ────────────────────────────────────────────────────────

    def get_by_source(self, source: str) -> list[EvalQuery]:
        """Filter loaded queries by source."""
        return [q for q in self._queries if q.source == source]

    def get_by_domain(self, domain: str) -> list[EvalQuery]:
        """Filter loaded queries by domain."""
        return [q for q in self._queries if q.domain == domain]

    def get_by_difficulty(self, difficulty: str) -> list[EvalQuery]:
        """Filter loaded queries by difficulty level."""
        return [q for q in self._queries if q.difficulty == difficulty]

    @property
    def queries(self) -> list[EvalQuery]:
        """All currently loaded queries."""
        return list(self._queries)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_manifest(self, path: Path) -> None:
        """Save query selection to JSON manifest for reproducibility."""
        manifest = {
            "version": "2.0",
            "total_queries": len(self._queries),
            "summary": self.summary,
            "queries": [q.to_dict() for q in self._queries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest saved to %s (%d queries)", path, len(self._queries))

    @classmethod
    def from_manifest(cls, path: Path) -> QueryRegistry:
        """Load a previously saved query selection."""
        data = json.loads(path.read_text())
        registry = cls()
        registry._queries = [EvalQuery.from_dict(q) for q in data["queries"]]
        log.info("Loaded %d queries from manifest %s", len(registry._queries), path)
        return registry

    # ── Summary statistics ───────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        """Summary statistics: counts by source, domain, difficulty."""
        by_source = Counter(q.source for q in self._queries)
        by_domain = Counter(q.domain for q in self._queries)
        by_difficulty = Counter(q.difficulty for q in self._queries)
        return {
            "total": len(self._queries),
            "by_source": dict(by_source),
            "by_domain": dict(by_domain),
            "by_difficulty": dict(by_difficulty),
        }
