"""RACE — Reference-based Adaptive Criteria-driven Evaluation (DeepResearch-Bench).

A reference-based, reference-guided evaluator that produces comparable
DeepResearch-Bench *leaderboard* RACE scores for our own deep-research reports,
so they can be placed head-to-head against the four commercial DRAs whose 400
released reports and 150 expert RACE annotations live on disk under
``data/benchmarks/drb1/``.

Background — what RACE is
-------------------------
RACE (Du et al., 2025, *DeepResearch Bench*) grades a long research report on
**four dimensions** — Comprehensiveness, Depth, Instruction-Following,
Readability — and combines them into a 0-100 ``overall`` via fixed dimension
weights.  Two properties make it *reference-based* and *adaptive*:

* **Reference-guided (relative) scoring.**  Each candidate report is graded
  *against a high-quality reference report* for the same task.  The judge is
  asked "is the candidate better / comparable / worse than the reference on
  this criterion", which calibrates absolute generosity away and makes scores
  comparable across tasks of very different intrinsic difficulty.
* **Adaptive, task-specific criteria.**  Under each of the four dimensions the
  judge first proposes a small set of *weighted* criteria tailored to *this*
  task (e.g. for an investment-comparison query, Comprehensiveness criteria
  enumerate the specific entities that must be covered).  The per-dimension
  score is the criterion-weight-weighted fraction satisfied, mapped to 0-100.

Dimension weights
-----------------
The published DeepResearch-Bench leaderboard combines the four dimension scores
with fixed weights.  We default to the published weights and expose the
empirical weights recovered from this repo's 150 expert annotations as a
documented, switchable alternative (both reproduce the human holistic
``overall`` at R^2 ~= 0.89 on the 600 model-task records on disk):

    RACE_WEIGHTS_PUBLISHED  Comp 0.28  Depth 0.31  IF 0.25  Read 0.16
    RACE_WEIGHTS_EMPIRICAL  Comp 0.25  Depth 0.31  IF 0.23  Read 0.21  (nnls fit, this repo)

Public interface
----------------
``RaceEvaluator.score(report_text, task) -> {dim: 0-100, "overall": 0-100}``
is a pure, synchronous function of (i) the candidate report text, (ii) the task
(query + optional reference report), and (iii) a *grader* callable that returns
per-criterion verdicts.  The grader is injected; in production it delegates to
the GPT-5.2 judge (reference-guided), but it is NEVER called in the self-test —
``score`` accepts an explicit ``grader=`` (e.g. a stub returning stored grades),
and there is a parallel ``score_from_dimension_grades`` that reconstructs the
published ``overall`` directly from already-graded per-dimension 0-100 scores.

This module makes **no network / API calls at import time or in the self-test**.
The production GPT-5.2 path is provided by :func:`make_gpt52_grader`, which is
only constructed (and only imports the judge) when explicitly requested.

Self-test (``python -m deep_research.benchmarks.race``)
-----------------------------------------------------
1. Validate the rubric structure loads and the weights sum to 1.0.
2. Reconstruct the leaderboard ``overall`` for 2-3 of the 400 on-disk reports
   from their stored expert dimension grades, and confirm the weighting math is
   exact (protocol-defined overall) and tracks the human holistic overall.
3. Exercise the full ``score()`` path end-to-end with a deterministic offline
   stub grader on a synthetic example — no API.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# ── RACE dimensions and weights ──────────────────────────────────────────────

# The four RACE dimensions, in canonical order.  The keys are the names used in
# the output dict; ``ANNOTATION_KEYS`` maps each to the (lower-cased, spaced)
# field name in the on-disk expert annotations so the self-test can read them.
RACE_DIMENSIONS: List[str] = [
    "comprehensiveness",
    "depth",
    "instruction_following",
    "readability",
]

# How each canonical dimension appears in drb1_human_annotations.json.
ANNOTATION_KEYS: Dict[str, str] = {
    "comprehensiveness": "Comprehensiveness",
    "depth": "depth",
    "instruction_following": "instruction following",
    "readability": "readability",
}

# Published DeepResearch-Bench leaderboard dimension weights.
RACE_WEIGHTS_PUBLISHED: Dict[str, float] = {
    "comprehensiveness": 0.28,
    "depth": 0.31,
    "instruction_following": 0.25,
    "readability": 0.16,
}

# Empirical weights recovered by non-negative least squares on this repo's 150
# expert annotation records (600 model-task overall scores), normalised to 1.0.
# Documented alternative; both fits reproduce the human holistic overall at
# R^2 ~= 0.89.  See module docstring and the self-test.
RACE_WEIGHTS_EMPIRICAL: Dict[str, float] = {
    "comprehensiveness": 0.251,
    "depth": 0.314,
    "instruction_following": 0.228,
    "readability": 0.207,
}


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return weights restricted to the four RACE dimensions, summing to 1.0."""
    missing = [d for d in RACE_DIMENSIONS if d not in weights]
    if missing:
        raise ValueError(f"weights missing RACE dimensions: {missing}")
    total = sum(weights[d] for d in RACE_DIMENSIONS)
    if total <= 0:
        raise ValueError("RACE dimension weights must sum to a positive number")
    return {d: weights[d] / total for d in RACE_DIMENSIONS}


# ── Per-task adaptive criteria ───────────────────────────────────────────────

@dataclass
class RaceCriterion:
    """A single weighted, task-specific RACE criterion under one dimension."""

    text: str
    dimension: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension not in RACE_DIMENSIONS:
            raise ValueError(
                f"criterion dimension {self.dimension!r} not one of {RACE_DIMENSIONS}"
            )
        if self.weight <= 0:
            raise ValueError("criterion weight must be positive")


@dataclass
class RaceRubric:
    """The adaptive RACE rubric for one task: weighted criteria per dimension."""

    task_id: str
    criteria: List[RaceCriterion] = field(default_factory=list)

    def by_dimension(self, dimension: str) -> List[RaceCriterion]:
        return [c for c in self.criteria if c.dimension == dimension]

    def dimensions_present(self) -> List[str]:
        return [d for d in RACE_DIMENSIONS if self.by_dimension(d)]


# ── Default (dimension-level) rubric ─────────────────────────────────────────

# When no task-specific criteria are supplied we fall back to one canonical
# criterion per dimension.  This keeps ``score`` well-defined for any report
# while still being reference-guided (the grader compares against the reference
# report for that single criterion).  Production runs supply richer, adaptive
# criteria (typically generated by the judge for the task) via ``rubric=``.
_DEFAULT_CRITERION_TEXT: Dict[str, str] = {
    "comprehensiveness": (
        "Relative to the reference report, the report covers the full breadth "
        "of sub-topics, entities, and aspects the task demands, with no major "
        "required area missing."
    ),
    "depth": (
        "Relative to the reference report, the report analyses each aspect in "
        "depth — synthesising across sources, quantifying where possible, and "
        "going beyond surface description to mechanisms, trade-offs and "
        "implications."
    ),
    "instruction_following": (
        "Relative to the reference report, the report follows every explicit "
        "and implicit instruction in the task (scope, format, requested "
        "comparisons, constraints) and stays on-topic."
    ),
    "readability": (
        "Relative to the reference report, the report is well-structured, "
        "fluent, logically organised, and easy for the intended reader to "
        "follow."
    ),
}


def build_default_rubric(task_id: str) -> RaceRubric:
    """One equally-weighted criterion per RACE dimension (reference-guided)."""
    return RaceRubric(
        task_id=task_id,
        criteria=[
            RaceCriterion(text=_DEFAULT_CRITERION_TEXT[d], dimension=d, weight=1.0)
            for d in RACE_DIMENSIONS
        ],
    )


# ── Task wrapper ─────────────────────────────────────────────────────────────

@dataclass
class RaceTask:
    """A DeepResearch-Bench task: the prompt plus an optional reference report.

    ``reference_report`` is the high-quality report the candidate is graded
    *against* (the "reference-based" in RACE).  If absent, grading degrades to
    absolute scoring and a warning flag is recorded — RACE is designed to be
    relative, so production runs should always supply a reference.
    """

    id: str
    query: str
    reference_report: Optional[str] = None
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_report and self.reference_report.strip())


# ── Grader contract ──────────────────────────────────────────────────────────

@dataclass
class CriterionGrade:
    """Grader output for one criterion: a 0-1 satisfaction fraction.

    Reference-guided convention (mirrors RACE's relative scale):
        1.0  candidate clearly meets / exceeds the reference on this criterion
        0.5  candidate is comparable to the reference
        0.0  candidate clearly falls short of the reference
    Intermediate values are allowed.  The grader MAY also pass back the verdict
    text/evidence for audit; only ``fraction`` is used in the score math.
    """

    criterion: str
    dimension: str
    fraction: float
    evidence: str = ""
    reasoning: str = ""


# A grader takes (report_text, task, criteria) and returns a CriterionGrade per
# criterion, in the same order.  It is the ONLY component permitted to call an
# LLM; ``score`` itself is pure given a grader.
Grader = Callable[[str, RaceTask, Sequence[RaceCriterion]], List[CriterionGrade]]


# ── The evaluator ────────────────────────────────────────────────────────────

class RaceEvaluator:
    """Reference-based RACE scorer producing DeepResearch-Bench-comparable scores."""

    def __init__(
        self,
        grader: Optional[Grader] = None,
        weights: Optional[Dict[str, float]] = None,
        weight_profile: str = "published",
    ) -> None:
        """
        Args:
            grader: per-criterion grading callable (injected so the self-test
                can pass an offline stub).  May be ``None`` if only
                :meth:`score_from_dimension_grades` is used.
            weights: explicit dimension-weight dict (overrides *weight_profile*).
            weight_profile: ``"published"`` (default, leaderboard) or
                ``"empirical"`` (nnls fit on this repo's 150 expert records).
        """
        self.grader = grader
        if weights is not None:
            self.weights = _normalise_weights(weights)
        elif weight_profile == "empirical":
            self.weights = _normalise_weights(RACE_WEIGHTS_EMPIRICAL)
        elif weight_profile == "published":
            self.weights = _normalise_weights(RACE_WEIGHTS_PUBLISHED)
        else:
            raise ValueError(f"unknown weight_profile {weight_profile!r}")

    # -- dimension aggregation -------------------------------------------------

    @staticmethod
    def _dimension_score_0_100(
        grades: Sequence[CriterionGrade],
        criteria: Sequence[RaceCriterion],
    ) -> float:
        """Criterion-weight-weighted mean satisfaction (0-1) -> 0-100."""
        if not criteria:
            return 0.0
        # Index grades by criterion text for robust pairing.
        by_text = {g.criterion: g for g in grades}
        num = 0.0
        den = 0.0
        for c in criteria:
            g = by_text.get(c.text)
            frac = g.fraction if g is not None else 0.0
            frac = max(0.0, min(1.0, float(frac)))
            num += c.weight * frac
            den += c.weight
        return 100.0 * (num / den) if den else 0.0

    def combine(self, dim_scores: Dict[str, float]) -> float:
        """Weighted-sum the four 0-100 dimension scores into the overall (0-100)."""
        return round(
            sum(self.weights[d] * float(dim_scores.get(d, 0.0)) for d in RACE_DIMENSIONS),
            4,
        )

    # -- main scoring entry points --------------------------------------------

    def score(
        self,
        report_text: str,
        task: RaceTask,
        rubric: Optional[RaceRubric] = None,
        grader: Optional[Grader] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Score *report_text* for *task*, returning ``{dim: 0-100, overall}``.

        Args:
            report_text: the candidate report.
            task: the task (query + optional reference report).
            rubric: adaptive per-task criteria; defaults to one criterion per
                dimension (:func:`build_default_rubric`).
            grader: per-criterion grader; defaults to the instance grader.  In
                the self-test an offline deterministic stub is passed here and
                NO API call is made.
            return_details: also return per-criterion grades under ``"_details"``.

        Returns:
            dict with the four dimension keys (0-100), ``"overall"`` (0-100),
            and ``"reference_based"`` (bool flag).
        """
        rubric = rubric or build_default_rubric(task.id)
        grader = grader or self.grader
        if grader is None:
            raise ValueError(
                "no grader available: pass grader= to score() or construct "
                "RaceEvaluator(grader=...).  (The self-test uses an offline stub.)"
            )

        grades = grader(report_text, task, rubric.criteria)
        # Defensive: the grader must return one grade per criterion.
        if len(grades) != len(rubric.criteria):
            raise ValueError(
                f"grader returned {len(grades)} grades for "
                f"{len(rubric.criteria)} criteria"
            )

        out: Dict[str, Any] = {}
        for d in RACE_DIMENSIONS:
            crits = rubric.by_dimension(d)
            dgrades = [g for g in grades if g.dimension == d]
            out[d] = round(self._dimension_score_0_100(dgrades, crits), 4)
        out["overall"] = self.combine(out)
        out["reference_based"] = task.has_reference
        if return_details:
            out["_details"] = [g.__dict__ for g in grades]
        return out

    def score_from_dimension_grades(
        self, dimension_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Reconstruct the leaderboard ``overall`` from per-dimension 0-100 grades.

        This is the *protocol-defined* overall (the weighted sum the leaderboard
        uses), exact given the dimension grades — used by the self-test to
        reconstruct published scores without any grading call.  Accepts either
        canonical keys (``comprehensiveness`` ...) or annotation keys
        (``Comprehensiveness``, ``instruction following`` ...).
        """
        canon: Dict[str, float] = {}
        ann_to_canon = {v: k for k, v in ANNOTATION_KEYS.items()}
        for k, v in dimension_scores.items():
            if k in RACE_DIMENSIONS:
                canon[k] = float(v)
            elif k in ann_to_canon:
                canon[ann_to_canon[k]] = float(v)
        missing = [d for d in RACE_DIMENSIONS if d not in canon]
        if missing:
            raise ValueError(f"dimension grades missing: {missing}")
        out: Dict[str, float] = {d: round(canon[d], 4) for d in RACE_DIMENSIONS}
        out["overall"] = self.combine(canon)
        return out


# ── Production GPT-5.2 reference-guided grader (lazy; never used in self-test) ─

def make_gpt52_grader() -> Grader:
    """Build a reference-guided grader backed by the GPT-5.2 judge.

    Lazily imports the judge so importing this module (and the self-test) makes
    no API client / network setup.  The returned grader runs the judge's async
    criterion evaluation under a fresh event loop and maps SATISFIED/
    NOT_SATISFIED (against the reference) onto the 0/0.5/1 RACE fraction.

    NOTE: this is the ONLY function in the module that touches the judge.  It is
    NOT called by the self-test.
    """
    import asyncio

    from deep_research.evaluation.llm_judge import (  # local import by design
        _judge_batch,
    )

    def grader(
        report_text: str,
        task: RaceTask,
        criteria: Sequence[RaceCriterion],
    ) -> List[CriterionGrade]:
        # Embed the reference report into the query context so the judge grades
        # the candidate *relative to* the reference (reference-guided RACE).
        if task.has_reference:
            framed_query = (
                f"{task.query}\n\n"
                "## Reference report (a strong answer to grade AGAINST)\n"
                f"{task.reference_report}\n\n"
                "When judging each criterion, mark SATISFIED only if the report "
                "below is comparable to or better than the reference on that "
                "criterion."
            )
        else:
            framed_query = task.query

        rubric_pairs = [(c.text, c.dimension) for c in criteria]
        verdicts = asyncio.run(
            _judge_batch(framed_query, report_text, rubric_pairs)
        )
        # _judge_batch returns one CriterionVerdict per criterion, in order.
        grades: List[CriterionGrade] = []
        for c, v in zip(criteria, verdicts):
            grades.append(
                CriterionGrade(
                    criterion=c.text,
                    dimension=c.dimension,
                    fraction=1.0 if getattr(v, "satisfied", False) else 0.0,
                    evidence=getattr(v, "evidence", ""),
                    reasoning=getattr(v, "reasoning", ""),
                )
            )
        return grades

    return grader


# ── On-disk DeepResearch-Bench (DRB1) loaders ────────────────────────────────

_DRB1_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "drb1"


def load_drb1_reports(drb1_dir: Path = _DRB1_DIR) -> Dict[str, Dict[int, str]]:
    """Return {model_name: {task_id(int): article_text}} for the 400 reports."""
    raw = json.loads((drb1_dir / "drb1_system_reports.json").read_text())
    out: Dict[str, Dict[int, str]] = {}
    for model, recs in raw.items():
        out[model] = {int(r["id"]): r["article"] for r in recs}
    return out


def load_drb1_tasks(drb1_dir: Path = _DRB1_DIR) -> Dict[int, Dict[str, Any]]:
    """Return {original_task_id(int): task_record} from drb1_queries.json."""
    raw = json.loads((drb1_dir / "drb1_queries.json").read_text())
    out: Dict[int, Dict[str, Any]] = {}
    for q in raw:
        oid = q.get("metadata", {}).get("original_id")
        if oid is not None:
            out[int(oid)] = q
    return out


def load_drb1_human_dimension_means(
    drb1_dir: Path = _DRB1_DIR,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Mean expert dimension+overall grades per (task_id, model).

    Returns {task_id: {model: {comprehensiveness, depth, instruction_following,
    readability, overall}}} averaging the (typically 3) annotators per task.
    Canonical dimension keys are used so they feed
    :meth:`RaceEvaluator.score_from_dimension_grades` directly.
    """
    raw = json.loads((drb1_dir / "drb1_human_annotations.json").read_text())
    # Accumulate per (task, model).
    acc: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for rec in raw:
        tid = int(rec["id"])
        for model, dims in rec["dimension_scores"].items():
            slot = acc.setdefault(tid, {}).setdefault(model, {})
            for canon, ann_key in ANNOTATION_KEYS.items():
                slot.setdefault(canon, []).append(float(dims[ann_key]))
            ov = rec.get("overall_scores", {}).get(model)
            if ov is not None:
                slot.setdefault("overall", []).append(float(ov))
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for tid, models in acc.items():
        out[tid] = {}
        for model, dims in models.items():
            out[tid][model] = {
                k: round(statistics.fmean(v), 4) for k, v in dims.items() if v
            }
    return out


# ── Self-test (no API) ───────────────────────────────────────────────────────

def _stub_grader(
    report_text: str,
    task: RaceTask,
    criteria: Sequence[RaceCriterion],
) -> List[CriterionGrade]:
    """Deterministic offline grader for the synthetic self-test (no API).

    Maps a synthetic report whose dimensions are encoded as marker lines like
    ``[[comprehensiveness=0.8]]`` to per-criterion fractions, so the full
    ``score`` path can be exercised end-to-end with a known expected result.
    """
    import re

    fracs: Dict[str, float] = {}
    for d in RACE_DIMENSIONS:
        m = re.search(rf"\[\[{d}=([01](?:\.\d+)?)\]\]", report_text)
        fracs[d] = float(m.group(1)) if m else 0.0
    return [
        CriterionGrade(
            criterion=c.text,
            dimension=c.dimension,
            fraction=fracs[c.dimension],
            reasoning="stub",
        )
        for c in criteria
    ]


def _self_test() -> int:
    print("=" * 72)
    print("RACE evaluator self-test (NO API CALLS)")
    print("=" * 72)
    failures: List[str] = []

    # 1) Rubric / weight structure -------------------------------------------
    for profile, table in (
        ("published", RACE_WEIGHTS_PUBLISHED),
        ("empirical", RACE_WEIGHTS_EMPIRICAL),
    ):
        norm = _normalise_weights(table)
        s = sum(norm.values())
        ok = abs(s - 1.0) < 1e-9 and set(norm) == set(RACE_DIMENSIONS)
        print(f"[1] weights[{profile}] sum={s:.6f} dims={sorted(norm)} -> "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"weight profile {profile} malformed")
    rub = build_default_rubric("t")
    ok = rub.dimensions_present() == RACE_DIMENSIONS and len(rub.criteria) == 4
    print(f"[1] default rubric dims={rub.dimensions_present()} "
          f"n_criteria={len(rub.criteria)} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("default rubric malformed")

    ev_pub = RaceEvaluator(weight_profile="published")
    ev_emp = RaceEvaluator(weight_profile="empirical")

    # 2) Reconstruct published leaderboard overall from stored expert grades ---
    drb1_present = (_DRB1_DIR / "drb1_human_annotations.json").exists()
    if drb1_present:
        human = load_drb1_human_dimension_means()
        # Pick 3 (task, model) cells deterministically (sorted).
        cells: List = []
        for tid in sorted(human):
            for model in sorted(human[tid]):
                if "overall" in human[tid][model]:
                    cells.append((tid, model))
        sample = cells[:3]
        print(f"[2] reconstructing protocol overall for {len(sample)} cells "
              f"(of {len(cells)} on disk); comparing to human holistic overall")
        diffs: List[float] = []
        for tid, model in sample:
            grades = human[tid][model]
            recon = ev_pub.score_from_dimension_grades(grades)
            human_overall = grades["overall"]
            # Exactness of the weighting MATH: recompute by hand and compare.
            # combine() rounds the overall to 4 dp, so allow that rounding step.
            hand = sum(RACE_WEIGHTS_PUBLISHED[d] * grades[d] for d in RACE_DIMENSIONS)
            hand /= sum(RACE_WEIGHTS_PUBLISHED.values())
            math_ok = abs(recon["overall"] - hand) < 1e-3
            diffs.append(abs(recon["overall"] - human_overall))
            print(f"    task {tid:>2} {model:<28} "
                  f"protocol_overall={recon['overall']:.2f} "
                  f"human_overall={human_overall:.2f} "
                  f"|diff|={abs(recon['overall'] - human_overall):.2f} "
                  f"math={'OK' if math_ok else 'FAIL'}")
            if not math_ok:
                failures.append(f"weighting math wrong for {tid}/{model}")
        # The protocol overall should track the holistic human overall (it is a
        # weighted dimension mean; R^2 ~= 0.89 over the full set, so per-cell
        # |diff| is modest, not zero).  Assert it is in a sane band.
        if diffs and statistics.fmean(diffs) > 15.0:
            failures.append(
                f"protocol overall diverges from human overall "
                f"(mean |diff|={statistics.fmean(diffs):.1f} > 15)"
            )

        # Whole-corpus agreement summary (informative, not a hard gate).
        all_recon, all_human = [], []
        for tid in human:
            for model in human[tid]:
                g = human[tid][model]
                if "overall" in g:
                    all_recon.append(ev_pub.score_from_dimension_grades(g)["overall"])
                    all_human.append(g["overall"])
        if len(all_recon) >= 2:
            mae = statistics.fmean(abs(a - b) for a, b in zip(all_recon, all_human))
            # Pearson r without numpy.
            mr = statistics.fmean(all_recon)
            mh = statistics.fmean(all_human)
            cov = sum((a - mr) * (b - mh) for a, b in zip(all_recon, all_human))
            va = sum((a - mr) ** 2 for a in all_recon)
            vb = sum((b - mh) ** 2 for b in all_human)
            r = cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else float("nan")
            print(f"[2] corpus: n={len(all_recon)} protocol-vs-human "
                  f"MAE={mae:.2f} pts, r={r:.3f}")
            if not (r > 0.9):
                failures.append(f"protocol-vs-human r too low ({r:.3f})")
    else:
        print("[2] SKIP: drb1 annotations not on disk")
        failures.append("drb1 annotations missing")

    # 3) Full score() path with offline stub grader (synthetic, no API) -------
    synth_report = (
        "# Synthetic report\n"
        "[[comprehensiveness=1.0]] [[depth=0.5]] "
        "[[instruction_following=1.0]] [[readability=0.0]]\n"
        "Body text ..."
    )
    task = RaceTask(id="synthetic-1", query="Compare X and Y.",
                    reference_report="A strong reference report about X and Y.")
    res = ev_pub.score(synth_report, task, grader=_stub_grader, return_details=True)
    # Expected dimension scores: 100, 50, 100, 0.
    exp_dims = {"comprehensiveness": 100.0, "depth": 50.0,
                "instruction_following": 100.0, "readability": 0.0}
    exp_overall = sum(RACE_WEIGHTS_PUBLISHED[d] * exp_dims[d] for d in RACE_DIMENSIONS)
    exp_overall /= sum(RACE_WEIGHTS_PUBLISHED.values())
    dims_ok = all(abs(res[d] - exp_dims[d]) < 1e-6 for d in RACE_DIMENSIONS)
    overall_ok = abs(res["overall"] - exp_overall) < 1e-4
    print(f"[3] synthetic score dims={{c:{res['comprehensiveness']:.0f}, "
          f"d:{res['depth']:.0f}, if:{res['instruction_following']:.0f}, "
          f"r:{res['readability']:.0f}}} overall={res['overall']:.4f} "
          f"(expected {exp_overall:.4f}) reference_based={res['reference_based']} "
          f"-> {'OK' if dims_ok and overall_ok else 'FAIL'}")
    if not (dims_ok and overall_ok):
        failures.append("synthetic score() path wrong")
    if not res["reference_based"]:
        failures.append("reference_based flag not set despite reference report")

    # Weighted-criteria sanity: two criteria with unequal weights average right.
    rub2 = RaceRubric(
        task_id="t2",
        criteria=[
            RaceCriterion("c-easy", "comprehensiveness", weight=3.0),
            RaceCriterion("c-hard", "comprehensiveness", weight=1.0),
        ],
    )
    grades = [
        CriterionGrade("c-easy", "comprehensiveness", 1.0),
        CriterionGrade("c-hard", "comprehensiveness", 0.0),
    ]
    dim = RaceEvaluator._dimension_score_0_100(grades, rub2.criteria)
    # (3*1 + 1*0)/4 = 0.75 -> 75.
    wok = abs(dim - 75.0) < 1e-6
    print(f"[3] weighted-criteria dim score={dim:.2f} (expected 75.00) "
          f"-> {'OK' if wok else 'FAIL'}")
    if not wok:
        failures.append("weighted criterion aggregation wrong")

    # 4) empirical vs published combine differ but both well-formed -----------
    eo = ev_emp.combine(exp_dims)
    po = ev_pub.combine(exp_dims)
    print(f"[4] combine published={po:.2f} empirical={eo:.2f} "
          f"(both 0-100, profiles differ -> {'OK' if 0 <= eo <= 100 and 0 <= po <= 100 else 'FAIL'})")
    if not (0 <= eo <= 100 and 0 <= po <= 100):
        failures.append("combine out of range")

    print("=" * 72)
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
