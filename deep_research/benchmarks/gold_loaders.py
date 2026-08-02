"""Per-dataset adapters that load each human-label set into ONE common schema.

Common schema row (a plain ``dict``)::

    {
        "item_id":     str,    # stable identifier for the labelled unit
        "dimension":   str,    # one of our rubric_v2 dimensions OR a class-label axis
        "human_label": float | str,  # NORMALISED 0-1 for graded/binary axes;
                                     #   the raw class string for multi-class axes
                                     #   (keep_classes=True keeps the raw class too)
        "label_class": str | None,   # canonical class string when the axis is categorical
                                     #   (so a downstream consumer can rebuild the class
                                     #    without re-parsing); None for purely numeric axes
        "text":        str,    # the graded span / claim / criterion / report-pointer
        "source":      str,    # dataset key: "drb_race" | "healthbench" | "expertqa" |
                               #   "deepfactbench" | "draco_full"
        "meta":        dict,   # provenance + dedup keys + anything a consumer needs
    }

Design notes (binding on every adapter)
--------------------------------------
* Adapters are GENERATORS over the on-disk normalised files inspected 2026-06-15;
  the true keys were read off the real files, not the paper.
* NORMALISATION: 0-100 -> /100; binary met/not-met -> {1.0, 0.0}; 3-class and 5-point
  categorical axes are mapped to a 0-1 score AND the raw class is preserved in
  ``label_class`` so a consumer can choose to treat the axis as ordinal-numeric or
  as a classification target. Where a class is genuinely unscoreable (ExpertQA
  "Unsure"/None, support "N/A") the numeric ``human_label`` is ``None`` while the
  class string is preserved — never silently coerced to 0.
* DEDUP CAUTIONS (register §"Cross-cutting cautions", HUMAN_LABEL_ASSETS.md):
  ExpertQA ⊂ AttributionBench and HAGRID ⊂ AttributionBench, so each ExpertQA row
  carries ``meta["dedup_family"]="expertqa"`` so a pooled statistic can exclude the
  AttributionBench copy. HealthBench has MULTIPLE physician verdicts per
  (completion, criterion) pair (27,339 pairs × ~2.2 physicians); rows are emitted
  PER PHYSICIAN with ``meta["pair_key"]`` and ``meta["physician_id"]`` so a
  consumer can aggregate to physician-consensus WITHOUT double counting a pair.
  DRB-RACE has 150 annotation records over 50 tasks (id 1..50, ~3 replicates each);
  ``meta["task_id"]`` + ``meta["annotation_id"]`` let a consumer cluster on the task.
  DRACO-full criteria are the EXPERT RUBRIC POOL (MET-targets), not per-report
  grades — see ``load_draco_full`` docstring.

This module makes ZERO network calls and reads only the on-disk normalised files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

# Repo-root-anchored data dir (works regardless of cwd).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HL = _REPO_ROOT / "data" / "human_labels"
_DRB = _REPO_ROOT / "data" / "benchmarks" / "drb1"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(
    item_id: str,
    dimension: str,
    human_label,
    text: str,
    source: str,
    meta: dict,
    label_class: Optional[str] = None,
) -> dict:
    return {
        "item_id": item_id,
        "dimension": dimension,
        "human_label": human_label,
        "label_class": label_class,
        "text": text,
        "source": source,
        "meta": meta,
    }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# DRB-RACE (DeepResearch-Bench human RACE annotations)
# ---------------------------------------------------------------------------
# Raw: 150 annotation records, each scoring 4 commercial DRA systems on 4
# dimensions (0-100). 50 unique task ids (1..50), ~3 annotation replicates each.
# Mapping (per task spec):
#   Comprehensiveness     -> coverage  AND  information_recall  (one source axis -> two rubric dims)
#   depth                 -> analytical_depth
#   instruction following -> instruction_following
#   readability           -> organization
DRB_DIMENSION_MAP: dict[str, list[str]] = {
    "Comprehensiveness": ["coverage", "information_recall"],
    "depth": ["analytical_depth"],
    "instruction following": ["instruction_following"],
    "readability": ["organization"],
}

# How many top-level annotation records the file is expected to carry (the
# self-test asserts on THIS, mirroring the register's "150 expert records").
DRB_N_SOURCE_RECORDS = 150


def load_drb_race(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield common-schema rows from DRB-RACE expert dimension scores.

    Source axis (0-100) -> rubric dimension(s), normalised to 0-1. One source
    record explodes into (n_systems × Σ mapped-dims) rows; ``Comprehensiveness``
    maps to BOTH coverage and information_recall, so its 0-1 value is duplicated
    across those two rows (the consumer may average or treat separately).

    ``meta`` carries ``task_id`` (the 1..50 task), ``annotation_id``,
    ``system`` (which DRA was scored), ``raw_dimension`` (source label),
    ``raw_score`` (0-100), and ``overall_score`` (0-100, the record's overall
    rating for that system). The text field is a pointer string (the released
    reports live in ``drb1_system_reports.json``, keyed by system+task — not
    inlined here to keep the registry light).
    """
    path = path or (_DRB / "drb1_human_annotations.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    for rec_idx, rec in enumerate(records):
        ann_id = rec.get("annotation_id")
        task_id = rec.get("id")
        dim_scores = rec.get("dimension_scores", {})
        overall = rec.get("overall_scores", {})
        for system, dims in dim_scores.items():
            sys_overall = overall.get(system)
            for raw_dim, raw_score in dims.items():
                rubric_dims = DRB_DIMENSION_MAP.get(raw_dim)
                if not rubric_dims:
                    continue  # unmapped source axis -> skip (logged by count delta)
                if raw_score is None:
                    continue
                norm = float(raw_score) / 100.0
                for rdim in rubric_dims:
                    # record_index makes item_id unique even where annotation_id
                    # collides (one annotation_id is shared by two source records).
                    item_id = f"drb_{rec_idx}_{system}_{rdim}"
                    yield _row(
                        item_id=item_id,
                        dimension=rdim,
                        human_label=round(norm, 4),
                        text=f"DRB-RACE report: system={system}, task_id={task_id} "
                             f"(report in drb1_system_reports.json)",
                        source="drb_race",
                        meta={
                            "record_index": rec_idx,
                            "task_id": task_id,
                            "annotation_id": ann_id,
                            "system": system,
                            "raw_dimension": raw_dim,
                            "raw_score": raw_score,
                            "overall_score": sys_overall,
                            "scale": "0-100->0-1",
                            "dedup_family": "drb_race",
                        },
                    )


# ---------------------------------------------------------------------------
# HealthBench meta_eval (OpenAI) — physician binary met/not-met
# ---------------------------------------------------------------------------
# Raw: 60,896 physician verdicts; 27,339 distinct (completion, criterion) pairs,
# ~2.2 physicians/pair. One row per physician verdict (do NOT pre-aggregate —
# the per-physician granularity is needed for inter-annotator / consensus work).
def load_healthbench(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield one common-schema row per physician met/not-met verdict.

    ``dimension`` is fixed to ``factual_accuracy`` (the binary met/not-met grade
    is a fulfilment/correctness judgement, the closest rubric_v2 axis to our
    248k binary-criterion corpus). ``human_label`` is 1.0 (met) / 0.0 (not met).
    ``meta["pair_key"]`` = (completion_id, rubric_criterion) so a consumer can
    aggregate to physician-consensus per pair without double counting; the
    per-physician identity is in ``meta["physician_id"]``.
    """
    path = path or (_HL / "healthbench" / "healthbench_normalised.jsonl")
    for i, rec in enumerate(_iter_jsonl(path)):
        cid = rec.get("completion_id", "")
        crit = rec.get("rubric_criterion", "")
        met = rec.get("human_met")
        label = 1.0 if met is True else (0.0 if met is False else None)
        yield _row(
            item_id=f"hb_{cid}_{i}",
            dimension="factual_accuracy",
            human_label=label,
            text=crit,
            source="healthbench",
            meta={
                "completion_id": cid,
                "prompt_id": rec.get("prompt_id"),
                "physician_id": rec.get("physician_id"),
                "category": rec.get("category"),
                # pair_key groups the multiple physician verdicts on one
                # (completion, criterion) pair for consensus aggregation.
                "pair_key": f"{cid}||{crit[:80]}",
                "label_type": "binary_met",
                "dedup_family": "healthbench",
            },
            label_class="met" if met is True else ("not_met" if met is False else None),
        )


# ---------------------------------------------------------------------------
# ExpertQA — per-claim factuality (5-point) AND attribution (support); SEPARATE
# ---------------------------------------------------------------------------
# Raw: 12,598 claims, 2,177 questions. correctness = 5-point factuality;
# support = attribution-completeness. Per task spec these are kept SEPARATE:
# each claim yields TWO rows (factual_accuracy from correctness,
# attribution_quality from support). The self-test asserts on the CLAIM count
# (12,598) via load_expertqa_claims (one row/claim, both labels attached) so the
# "12598" register number is reproduced exactly; load_expertqa() is the
# exploded two-row-per-claim common-schema generator.

# 5-point factuality -> 0-1 (ordinal). "Unsure"/None are unscoreable -> None.
EXPERTQA_CORRECTNESS_MAP: dict[str, Optional[float]] = {
    "Definitely correct": 1.0,
    "Probably correct": 0.75,
    "Unsure": None,
    "Likely incorrect": 0.25,
    "Definitely incorrect": 0.0,
}
# attribution support -> 0-1. "N/A"/None unscoreable -> None.
EXPERTQA_SUPPORT_MAP: dict[str, Optional[float]] = {
    "Complete": 1.0,
    "Partial": 0.5,
    "Incomplete": 0.25,
    "Missing": 0.0,
    "N/A": None,
}


def _expertqa_meta(rec: dict, axis: str) -> dict:
    return {
        "question": rec.get("question"),
        "field": rec.get("field"),
        "specific_field": rec.get("specific_field"),
        "answer_model": rec.get("answer_model"),
        "claim_id": rec.get("claim_id"),
        "axis": axis,
        # ExpertQA ⊂ AttributionBench / HAGRID ⊂ AttributionBench: never pool the
        # AttributionBench copy with these rows in one statistic.
        "dedup_family": "expertqa",
    }


def load_expertqa(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield TWO common-schema rows per ExpertQA claim (factuality + attribution).

    factual_accuracy  <- ``correctness`` (5-point ordinal -> 0-1, label_class kept)
    attribution_quality <- ``support``  (completeness -> 0-1, label_class kept)
    Unscoreable classes ("Unsure"/None, "N/A"/None) yield ``human_label=None`` with
    the raw class preserved in ``label_class``.
    """
    path = path or (_HL / "expertqa" / "expertqa_claims_normalised.jsonl")
    for rec in _iter_jsonl(path):
        claim_id = rec.get("claim_id", "")
        claim = rec.get("claim_string", "")
        corr = rec.get("correctness")
        supp = rec.get("support")
        yield _row(
            item_id=f"eqa_{claim_id}_factual",
            dimension="factual_accuracy",
            human_label=EXPERTQA_CORRECTNESS_MAP.get(corr) if corr is not None else None,
            text=claim,
            source="expertqa",
            meta=_expertqa_meta(rec, "correctness"),
            label_class=corr,
        )
        yield _row(
            item_id=f"eqa_{claim_id}_attribution",
            dimension="attribution_quality",
            human_label=EXPERTQA_SUPPORT_MAP.get(supp) if supp is not None else None,
            text=claim,
            source="expertqa",
            meta=_expertqa_meta(rec, "support"),
            label_class=supp,
        )


def load_expertqa_claims(path: Optional[Path] = None) -> Iterator[dict]:
    """One common-schema row per CLAIM (count == 12,598), both labels in meta.

    Used by the self-test to reproduce the register's claim count exactly while
    still carrying the separate factuality/attribution labels (in ``meta``).
    ``dimension`` is ``factual_accuracy`` and ``human_label`` is the factuality
    score; ``meta["attribution_label"]`` / ``meta["attribution_class"]`` carry the
    attribution side so the SEPARATE labels are not lost.
    """
    path = path or (_HL / "expertqa" / "expertqa_claims_normalised.jsonl")
    for rec in _iter_jsonl(path):
        claim_id = rec.get("claim_id", "")
        corr = rec.get("correctness")
        supp = rec.get("support")
        meta = _expertqa_meta(rec, "claim")
        meta["correctness_class"] = corr
        meta["support_class"] = supp
        meta["attribution_label"] = (
            EXPERTQA_SUPPORT_MAP.get(supp) if supp is not None else None
        )
        meta["attribution_class"] = supp
        yield _row(
            item_id=f"eqa_{claim_id}",
            dimension="factual_accuracy",
            human_label=EXPERTQA_CORRECTNESS_MAP.get(corr) if corr is not None else None,
            text=rec.get("claim_string", ""),
            source="expertqa",
            meta=meta,
            label_class=corr,
        )


# ---------------------------------------------------------------------------
# DeepFact-Bench — claim-level SUPPORTED / CONTRADICTORY / INCONCLUSIVE
# ---------------------------------------------------------------------------
# Raw: 621 claim verdicts on actual deep-research report sentences. 3-class
# human verdict. Mapped to 0-1 (supported=1, inconclusive=0.5, contradictory=0)
# AND raw class preserved. This is the only IN-GENRE (deep-research report)
# factuality set, so dimension = factual_accuracy.
DEEPFACT_VERDICT_MAP: dict[str, float] = {
    "supported": 1.0,
    "inconclusive": 0.5,
    "contradictory": 0.0,
}


def load_deepfactbench(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield one common-schema row per DeepFact-Bench claim verdict.

    factual_accuracy <- human_verdict (SUPPORTED/INCONCLUSIVE/CONTRADICTORY ->
    1.0/0.5/0.0; raw class in label_class). ``meta`` carries the report_id,
    domain, the model's own agent_verdict (for human-vs-agent agreement), and the
    relevance grade (1-5). In-genre deep-research report sentences.
    """
    path = path or (_HL / "deepfactbench" / "deepfactbench_normalised.jsonl")
    for rec in _iter_jsonl(path):
        verdict = rec.get("human_verdict")
        norm = DEEPFACT_VERDICT_MAP.get(verdict)
        rel_raw = rec.get("relevance")
        try:
            rel = int(rel_raw) if rel_raw is not None else None
        except (TypeError, ValueError):
            rel = None
        yield _row(
            item_id=f"dfb_{rec.get('claim_id', '')}",
            dimension="factual_accuracy",
            human_label=norm,
            text=rec.get("sentence", ""),
            source="deepfactbench",
            meta={
                "report_id": rec.get("report_id"),
                "domain": rec.get("domain"),
                "agent_verdict": rec.get("agent_verdict"),
                "relevance": rel,
                "split": rec.get("split"),
                "in_genre": True,
                "dedup_family": "deepfactbench",
            },
            label_class=verdict,
        )


# ---------------------------------------------------------------------------
# DRACO-full — expert MET/UNMET criteria (the rubric POOL, weighted)
# ---------------------------------------------------------------------------
# Raw: 3,934 expert-curated criteria over 100 real-user deep-research tasks,
# grouped into 4 sections (factual-accuracy / breadth-and-depth-of-analysis /
# presentation-quality / citation-quality). Each criterion carries a weight;
# 3,519 positive (a MET-target: a report SHOULD satisfy it) and 415 NEGATIVE
# (a critical-failure: a report should NOT trigger it -> an UNMET-target).
#
# IMPORTANT: this normalised file is the expert RUBRIC POOL, not per-report
# grades. So the "human_label" here is the EXPERT TARGET POLARITY of the
# criterion: 1.0 = should-be-MET (positive weight), 0.0 = should-be-UNMET /
# critical-failure (negative weight). This is the same format our harness
# generates (binary MET/UNMET criteria), used as gold criterion targets.
DRACO_SECTION_DIMENSION_MAP: dict[str, str] = {
    "factual-accuracy": "factual_accuracy",
    "breadth-and-depth-of-analysis": "analytical_depth",
    "presentation-quality": "organization",
    "citation-quality": "citation_quality",
}


def load_draco_full(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield one common-schema row per DRACO expert criterion (count == 3,934).

    ``dimension`` is mapped from the criterion's section. ``human_label`` is the
    expert TARGET POLARITY: 1.0 for a positive-weight MET-target, 0.0 for a
    negative-weight critical-failure (UNMET-target). The criterion ``weight`` and
    ``section`` are preserved in ``meta`` for DRACO-weighted downstream scoring.
    """
    path = path or (_HL / "draco_full" / "draco_criteria_normalised.jsonl")
    for rec in _iter_jsonl(path):
        section = rec.get("section", "")
        weight = rec.get("weight", 0)
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.0
        polarity = 0.0 if w < 0 else 1.0
        yield _row(
            item_id=f"draco_{rec.get('task_id', '')}_{rec.get('criterion_id', '')}",
            dimension=DRACO_SECTION_DIMENSION_MAP.get(section, "factual_accuracy"),
            human_label=polarity,
            text=rec.get("requirement", ""),
            source="draco_full",
            meta={
                "task_id": rec.get("task_id"),
                "criterion_id": rec.get("criterion_id"),
                "domain": rec.get("domain"),
                "section": section,
                "weight": w,
                "is_critical_failure": w < 0,
                "label_type": "expert_target_polarity",
                "dedup_family": "draco_full",
            },
            label_class="met_target" if w >= 0 else "unmet_critical_failure",
        )


# ---------------------------------------------------------------------------
# registry + self-test
# ---------------------------------------------------------------------------

LOADERS = {
    "drb_race": load_drb_race,
    "healthbench": load_healthbench,
    "expertqa": load_expertqa,            # exploded: 2 rows/claim
    "expertqa_claims": load_expertqa_claims,  # 1 row/claim (count anchor)
    "deepfactbench": load_deepfactbench,
    "draco_full": load_draco_full,
}

# The register "row count" the self-test asserts per dataset. For DRB-RACE the
# anchor is the 150 SOURCE annotation records (not the exploded row count); for
# ExpertQA the anchor is the 12,598 CLAIM count (via load_expertqa_claims).
EXPECTED_COUNTS = {
    "drb_race_source_records": DRB_N_SOURCE_RECORDS,  # 150
    "healthbench": 60896,
    "expertqa_claims": 12598,
    "deepfactbench": 621,
    "draco_full": 3934,
}


def _selftest() -> int:
    """Load each adapter, print counts, assert the register numbers. Returns exit code."""
    ok = True
    print("=" * 72)
    print("gold_loaders self-test (real on-disk human labels)")
    print("=" * 72)

    # DRB-RACE: assert on SOURCE records (file list length), also report exploded
    # rows. record_index is the stable per-source-record key (annotation_id has one
    # duplicate, so it under-counts; the register's "150 records" = file length).
    drb_rows = list(load_drb_race())
    src_records = len({r["meta"]["record_index"] for r in drb_rows})
    exp = EXPECTED_COUNTS["drb_race_source_records"]
    status = "OK" if src_records == exp else "FAIL"
    ok &= src_records == exp
    dims = sorted({r["dimension"] for r in drb_rows})
    print(f"[drb_race]      source_records={src_records} (expected {exp}) [{status}]  "
          f"exploded_rows={len(drb_rows)} dims={dims}")

    # HealthBench
    hb = sum(1 for _ in load_healthbench())
    exp = EXPECTED_COUNTS["healthbench"]
    status = "OK" if hb == exp else "FAIL"
    ok &= hb == exp
    print(f"[healthbench]   rows={hb} (expected {exp}) [{status}]")

    # ExpertQA — claim anchor + exploded
    eqa_claims = sum(1 for _ in load_expertqa_claims())
    eqa_exploded = sum(1 for _ in load_expertqa())
    exp = EXPECTED_COUNTS["expertqa_claims"]
    status = "OK" if eqa_claims == exp else "FAIL"
    ok &= eqa_claims == exp
    print(f"[expertqa]      claims={eqa_claims} (expected {exp}) [{status}]  "
          f"exploded_rows={eqa_exploded} (factual+attribution, expected {exp*2})")
    ok &= eqa_exploded == exp * 2

    # DeepFactBench
    dfb = sum(1 for _ in load_deepfactbench())
    exp = EXPECTED_COUNTS["deepfactbench"]
    status = "OK" if dfb == exp else "FAIL"
    ok &= dfb == exp
    print(f"[deepfactbench] rows={dfb} (expected {exp}) [{status}]")

    # DRACO-full
    draco = sum(1 for _ in load_draco_full())
    exp = EXPECTED_COUNTS["draco_full"]
    status = "OK" if draco == exp else "FAIL"
    ok &= draco == exp
    print(f"[draco_full]    rows={draco} (expected {exp}) [{status}]")

    # Schema conformance check on a sample row from each source.
    print("-" * 72)
    required_keys = {"item_id", "dimension", "human_label", "label_class",
                     "text", "source", "meta"}
    for name in ("drb_race", "healthbench", "expertqa", "deepfactbench", "draco_full"):
        gen = LOADERS[name]()
        sample = next(gen)
        missing = required_keys - set(sample.keys())
        sstatus = "OK" if not missing else f"FAIL missing={missing}"
        ok &= not missing
        hl = sample["human_label"]
        print(f"[schema:{name:13s}] keys OK [{sstatus}]  "
              f"dim={sample['dimension']!r} label={hl!r} class={sample['label_class']!r}")

    print("=" * 72)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
