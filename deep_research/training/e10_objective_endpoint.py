#!/usr/bin/env python3
"""E10 noise-RL — JUDGE-FREE objective endpoint (anti-Goodhart).

PURE CPU, NO LLM, NO PAID API, NO CANONICAL WRITE. This computes a
key-fact / answer-match score for the ANSWER-CHECKABLE query slice and is
logged EVERY eval step ALONGSIDE the 7B DR-Judge reward. It is NEVER the
training reward.

WHY
---
The E10 training reward is a 7B DR-Judge LoRA — a signal the policy can hack.
If the judge reward rises while this judge-free objective score is flat or
falls, the arm is Goodharting the 7B judge rather than improving the report.
This metric is the divergence detector. It is a NOISY PROXY (string/entity
match over gold key-facts), not ground truth; the prereg states it monitors
divergence and must never become the reward.

ANSWER-CHECKABLE SLICE (from data/eval_queries_v2.json)
-------------------------------------------------------
A query is answer-checkable iff it carries gold key-facts in EITHER:
  * ``reference_answer`` (non-empty string) — 29 queries, OR
  * ``expected_elements`` (non-empty list)  — 15 queries.
Their UNION is the answer-checkable slice. In the current manifest the two
signals overlap on 10 queries, so the distinct slice is 34 queries (NOT 44 —
the 44 in the original design double-counted the overlap; this module uses the
real, deduplicated 34 and records the count in its output so the prereg n is
auditable). When a query has both signals, ``expected_elements`` is preferred
(it is an explicit, atomised gold-fact list); otherwise the reference answer is
sentence-segmented into pseudo key-facts.

SCORE
-----
For each held-out answer-checkable query:
    objective_score(report) = (# gold key-facts present in report) / (# gold key-facts)
"present" = deterministic normalised string / entity containment match (see
``_fact_present``). The per-query scores are averaged over the held-out slice.

OUTPUT
------
``emit_objective_trace`` appends one JSONL row per eval step to
``results/e10/<arm>/objective_trace.jsonl`` with the step, judge-reward (passed
in by the trainer), the judge-free objective mean, per-query breakdown, and the
manifest hash so a divergence plot is reproducible.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Gold key-fact extraction
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Light stopword set for entity-overlap matching (deterministic, fixed).
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to "
    "with was were will would can could should this these those their there which "
    "who whom whose into than then they them you your our we us also but not".split()
)


def _normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _content_tokens(s: str) -> List[str]:
    return [t for t in _WORD_RE.findall(_normalise(s)) if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    difficulty: str
    key_facts: List[str]          # atomised gold facts (strings)
    source: str                   # 'expected_elements' or 'reference_answer'


def extract_gold(query: dict) -> Optional[GoldQuery]:
    """Return a GoldQuery if the query is answer-checkable, else None.

    Prefers ``expected_elements`` (explicit atomised facts). Falls back to
    sentence-segmenting a non-empty ``reference_answer``.
    """
    qid = query["id"]
    diff = query.get("difficulty", "moderate")
    exp = query.get("expected_elements")
    if isinstance(exp, list) and exp:
        facts = [str(e).strip() for e in exp if str(e).strip()]
        if facts:
            return GoldQuery(qid, diff, facts, "expected_elements")
    ref = query.get("reference_answer")
    if isinstance(ref, str) and ref.strip():
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(ref.strip()) if s.strip()]
        if sents:
            return GoldQuery(qid, diff, sents, "reference_answer")
    return None


def load_answer_checkable(queries_path: str | Path) -> Dict[str, GoldQuery]:
    """Load the deduplicated answer-checkable slice, keyed by query_id, sorted."""
    data = json.loads(Path(queries_path).read_text())
    out: Dict[str, GoldQuery] = {}
    for q in sorted(data["queries"], key=lambda r: r["id"]):
        g = extract_gold(q)
        if g is not None:
            out[g.query_id] = g
    return out


# --------------------------------------------------------------------------- #
# Fact-presence matching (deterministic, no LLM)
# --------------------------------------------------------------------------- #
def _fact_present(fact: str, report_norm: str, report_tokens: set,
                  token_overlap_threshold: float = 0.6) -> bool:
    """Deterministic key-fact containment.

    A gold fact counts as present if EITHER:
      (1) its full normalised string is a substring of the normalised report
          (exact phrase match), OR
      (2) at least ``token_overlap_threshold`` of its content tokens appear in
          the report's token set (entity-overlap match) — handles paraphrase
          while staying deterministic.
    """
    fnorm = _normalise(fact)
    if fnorm and fnorm in report_norm:
        return True
    ftoks = _content_tokens(fact)
    if not ftoks:
        return False
    hits = sum(1 for t in ftoks if t in report_tokens)
    return (hits / len(ftoks)) >= token_overlap_threshold


def objective_score_one(report: str, gold: GoldQuery,
                        token_overlap_threshold: float = 0.6) -> float:
    """Fraction of gold key-facts present in the report."""
    report_norm = _normalise(report)
    report_tokens = set(_content_tokens(report))
    if not gold.key_facts:
        return float("nan")
    present = sum(
        1 for f in gold.key_facts
        if _fact_present(f, report_norm, report_tokens, token_overlap_threshold)
    )
    return present / len(gold.key_facts)


@dataclass
class ObjectiveResult:
    mean_score: float
    n_queries: int
    per_query: Dict[str, float]
    per_query_source: Dict[str, str]
    token_overlap_threshold: float


def evaluate_objective(
    reports_by_qid: Dict[str, str],
    gold_slice: Dict[str, GoldQuery],
    token_overlap_threshold: float = 0.6,
) -> ObjectiveResult:
    """Score a set of generated reports against the answer-checkable gold slice.

    ``reports_by_qid`` maps query_id -> generated report text. Only query_ids
    present in BOTH ``reports_by_qid`` and ``gold_slice`` are scored (this is
    expected to be the HELD-OUT eval slice from e10_split.json). The mean is
    over the scored queries, sorted for determinism.
    """
    per_query: Dict[str, float] = {}
    per_src: Dict[str, str] = {}
    for qid in sorted(gold_slice):
        if qid not in reports_by_qid:
            continue
        g = gold_slice[qid]
        per_query[qid] = objective_score_one(
            reports_by_qid[qid], g, token_overlap_threshold
        )
        per_src[qid] = g.source
    scores = [v for v in per_query.values() if v == v]  # drop NaN
    mean = sum(scores) / len(scores) if scores else float("nan")
    return ObjectiveResult(
        mean_score=mean,
        n_queries=len(scores),
        per_query=per_query,
        per_query_source=per_src,
        token_overlap_threshold=token_overlap_threshold,
    )


# --------------------------------------------------------------------------- #
# Trace emission (anti-Goodhart divergence log)
# --------------------------------------------------------------------------- #
def emit_objective_trace(
    out_dir: str | Path,
    arm: str,
    step: int,
    judge_reward_mean: float,
    objective: ObjectiveResult,
    manifest_hash: str,
    extra: Optional[dict] = None,
) -> Path:
    """Append one JSONL row to results/e10/<arm>/objective_trace.jsonl.

    Logs judge reward and judge-free objective TOGETHER so a post-hoc plot can
    detect Goodharting (judge up, objective flat/down). Never writes canonical.
    """
    out_dir = Path(out_dir)
    trace_dir = out_dir / "results" / "e10" / arm if (out_dir / "results").exists() \
        else out_dir / arm
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "objective_trace.jsonl"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "step": int(step),
        "judge_reward_mean": float(judge_reward_mean),
        "objective_score_mean": (
            float(objective.mean_score) if objective.mean_score == objective.mean_score
            else None
        ),
        "objective_n_queries": objective.n_queries,
        "token_overlap_threshold": objective.token_overlap_threshold,
        "objective_per_query": objective.per_query,
        "objective_per_query_source": objective.per_query_source,
        "manifest_hash": manifest_hash,
    }
    if extra:
        row.update(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI self-test (CPU, no LLM)
# --------------------------------------------------------------------------- #
def _selftest(queries_path: str) -> int:
    gold = load_answer_checkable(queries_path)
    import collections
    by_src = collections.Counter(g.source for g in gold.values())
    by_diff = collections.Counter(g.difficulty for g in gold.values())
    print(f"[selftest] answer-checkable slice n={len(gold)}")
    print(f"[selftest]   by source:   {dict(by_src)}")
    print(f"[selftest]   by difficulty: {dict(by_diff)}")
    # synthetic perfect vs empty report
    some_qid = sorted(gold)[0]
    g = gold[some_qid]
    perfect = " ".join(g.key_facts)
    empty = "nothing relevant here"
    sp = objective_score_one(perfect, g)
    se = objective_score_one(empty, g)
    print(f"[selftest] qid={some_qid} src={g.source} n_facts={len(g.key_facts)} "
          f"perfect={sp:.3f} empty={se:.3f}")
    assert sp >= 0.99, "perfect report should score ~1.0"
    assert se <= 0.5, "empty report should score low"
    res = evaluate_objective({some_qid: perfect}, gold)
    assert res.n_queries == 1 and abs(res.mean_score - sp) < 1e-9
    print("[selftest] PASS — objective endpoint is deterministic and in-range.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E10 judge-free objective endpoint self-test.")
    ap.add_argument("--queries", default="data/eval_queries_v2.json")
    a = ap.parse_args()
    raise SystemExit(_selftest(a.queries))
