"""MERIDIAN Role 2: Topic Miner — clusters source extractions into coherent topics.

Uses gpt-5.2 (medium-tier) to analyse the structured SourceExtraction objects and
produce a set of TopicCluster objects, each grouping semantically related
extractions under a named topic with an importance score.

Leverages the richer structured data (key_findings, relevance_score, data_points)
from SourceExtractor for higher-quality clustering.

No vector retrieval, BM25, reranker, or fusion — the LLM reads all extractions
directly and performs the clustering in a single pass.
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from deep_research.tools import LLMCaller, SourceExtraction
from deep_research.types import TopicCluster
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

MODEL = DEFAULT_MODEL


# -- Clustering via LLM -------------------------------------------------------

_CLUSTER_SYSTEM = (
    "You are a topic-clustering expert. Given a set of source extractions retrieved "
    "for a research query, identify the distinct topics/themes they cover. Group the "
    "extractions by topic, name each topic, write a brief synthesis, and assign an "
    "importance score (0.0-1.0) reflecting how central the topic is to answering "
    "the query. Every extraction must be assigned to exactly one topic. Pay close "
    "attention to the key findings, relevance scores, and data points provided for "
    "each source to improve clustering accuracy."
)


def _build_cluster_prompt(query: str, extractions: List[SourceExtraction]) -> str:
    """Construct the clustering prompt with numbered extraction excerpts."""
    extraction_blocks = []
    for i, ext in enumerate(extractions):
        block = (
            f"[Source {i} | doc_id={ext.doc_id} | relevance={ext.relevance_score}/10]\n"
            f"Title: {ext.title}\n"
            f"URL: {ext.url}\n"
            f"Type: {ext.source_type.value}\n"
            f"Summary: {ext.summary}\n"
        )
        if ext.key_findings:
            findings = "; ".join(ext.key_findings)
            block += f"Key Findings: {findings}\n"
        if ext.data_points:
            points = "; ".join(ext.data_points)
            block += f"Data Points: {points}\n"
        if ext.confidence_notes:
            block += f"Confidence: {ext.confidence_notes}\n"
        extraction_blocks.append(block)

    extraction_text = "\n\n".join(extraction_blocks)

    return f"""\
Research query: {query}

Below are {len(extractions)} source extractions with structured findings. Cluster them into coherent topics.

{extraction_text}

Respond in JSON:
{{
  "topics": [
    {{
      "topic": "<topic name>",
      "summary": "<1-3 sentence synthesis of this topic drawing on the sources>",
      "summary_indices": [<int>, ...],
      "importance": <float 0.0 to 1.0>
    }},
    ...
  ]
}}

Rules:
- summary_indices are the zero-based indices of the source extractions (0, 1, 2, ...).
- Every index must appear in exactly one topic.
- Order topics from most to least important.
- Aim for 4-10 topics (merge very small groups, split overly broad ones).
- importance values should sum to roughly 1.0.
- Use the relevance scores and key findings to inform importance weights.
"""


async def cluster_extractions(
    query: str,
    extractions: List[SourceExtraction],
    llm: LLMCaller,
) -> List[TopicCluster]:
    """Use LLM to cluster source extractions into TopicCluster objects.

    Each cluster's ``source_ids`` field stores the string doc_ids of the sources
    assigned to that topic.
    """
    if not extractions:
        log.warning("topic_miner.no_extractions_to_cluster")
        return []

    prompt = _build_cluster_prompt(query, extractions)
    result = await llm.complete_json(
        prompt=prompt,
        model=MODEL,
        system=_CLUSTER_SYSTEM,
        temperature=0.2,
        max_tokens=4096,
    )

    raw_topics = result.get("topics", [])
    valid_indices = set(range(len(extractions)))

    clusters: List[TopicCluster] = []
    assigned_indices: set = set()

    for raw in raw_topics:
        indices = [
            idx for idx in raw.get("summary_indices", [])
            if isinstance(idx, int) and idx in valid_indices
        ]
        if not indices:
            continue

        assigned_indices.update(indices)

        doc_ids = [extractions[idx].doc_id or str(idx) for idx in indices]

        clusters.append(
            TopicCluster(
                topic=raw.get("topic", "Untitled"),
                summary=raw.get("summary", ""),
                source_ids=doc_ids,
                importance=float(raw.get("importance", 0.5)),
            )
        )

    # Assign any orphan extractions to a catch-all cluster
    orphan_indices = valid_indices - assigned_indices
    if orphan_indices:
        orphan_doc_ids = [
            extractions[idx].doc_id or str(idx) for idx in sorted(orphan_indices)
        ]
        clusters.append(
            TopicCluster(
                topic="Additional Findings",
                summary="Sources that did not fit neatly into the main topic clusters.",
                source_ids=orphan_doc_ids,
                importance=0.05,
            )
        )

    # Normalise importance
    total_imp = sum(c.importance for c in clusters) or 1.0
    for c in clusters:
        c.importance = c.importance / total_imp

    # Sort by importance descending
    clusters.sort(key=lambda c: c.importance, reverse=True)

    log.info(
        "topic_miner.clustered",
        n_topics=len(clusters),
        topic_names=[c.topic for c in clusters],
    )
    return clusters


# -- Extraction lookup helpers -------------------------------------------------


def build_extraction_map(extractions: List[SourceExtraction]) -> Dict[str, SourceExtraction]:
    """Return a dict mapping doc_id -> SourceExtraction for quick access."""
    return {ext.doc_id: ext for ext in extractions if ext.doc_id}


# -- Top-level convenience -----------------------------------------------------


async def run_topic_miner(
    query: str,
    extractions: List[SourceExtraction],
    llm: LLMCaller,
) -> tuple[List[TopicCluster], Dict[str, SourceExtraction]]:
    """Full Topic-Miner pipeline: cluster extractions into topics.

    Returns:
        (clusters, extraction_map)
    """
    clusters = await cluster_extractions(query, extractions, llm)
    extraction_map = build_extraction_map(extractions)

    log.info(
        "topic_miner.complete",
        topics=len(clusters),
        extractions_used=len(extractions),
    )
    return clusters, extraction_map
