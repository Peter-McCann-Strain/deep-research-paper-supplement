#!/usr/bin/env python3
"""E11 P14-VTR — Verify-then-refine at matched budget (PTU generation harness).

WHAT THIS IS (RESEARCH_PLAN_2026H2.md §4, E11, priority 7)
----------------------------------------------------------
Tests the ONE architecture class the bounded-returns study never tested: sequential
*verification* compute instead of parallel *orchestration* compute.  For each
variance-stratified query we take an existing un-refined draft (P0-base and/or
P4-base, the SAME gpt-4o/PTU corpus reports), run a verify-then-refine loop, and
measure the VTR NET EFFECT (refined overall vs the un-refined draft) under the
authoritative GPT-5.2 judge.

VTR loop (per query, per base, per arm), token budget matched to the top-cluster median
-------------------------------------------------------------------------------------
  draft_0  := existing results/experiments/base_p{0,4}/{qid}.md   (turn 0; verdicts EXIST)
  for round in 1..R (default R=2):
    VERIFIER scores draft_{r-1} against the 38 pre-generated rubric_v2 general criteria,
      emitting per-FAILED-criterion targeted fix instructions  (verifier = gpt-4o on PTU);
    GENERATOR (gpt-4o on PTU) REFINES draft_{r-1} -> draft_r, addressing only the failed
      criteria, holding everything else, within the matched token budget.
  emit draft_R as the refined report.

ARMS (n = 30 variance queries x {2 bases} x {arms below})
  vtr_gpt4o   : verifier = gpt-4o (PTU)            -- the primary VTR arm
  control     : UNGUIDED revision (no verifier; "improve this report" prompt, same
                generator + same budget)            -- isolates verification signal vs
                                                       plain extra-compute revision
  vtr_drjudge : verifier = local DR-Judge-7B-LoRA (QLoRA Qwen2.5-7B-Instruct adapter,
                local GPU) -- ON (adapter on disk at models/DR-Judge-7B-LoRA/). DR-Judge
                is ONLY the verifier; the refiner stays gpt-4o on PTU and GPT-5.2 remains
                the sole authoritative judge. The SATISFIED/NOT_SATISFIED decision is
                pinned to the canonical signed-Youden-J per-dimension operating point.
                Forces concurrency=1 (single 16GB card; one shared, pre-loaded caller).

CONSISTENCY (HARD)
------------------
* ALL generation (draft refinement AND the gpt-4o verifier) uses the default backbone
  gpt-4o on PTU deployment "sthree-ptu-02" via config DEFAULT_MODEL — the SAME backbone
  as the 248k-report corpus.  This script NEVER overrides the generation model.
* ALL AUTHORITATIVE judging is GPT-5.2, performed SEPARATELY by the corpus-safe runner
  scripts/run_gpt52_judge_namespaced.py pointed at this script's NEW output dir.  This
  harness performs NO judging; it does not even import the judge endpoint.  The un-refined
  (turn-0) GPT-5.2 verdicts already exist under results/judge_gpt52/base_p{0,4}/ so the
  net-effect baseline is free.
* SEARCH_MODEL stays gpt-4o-mini as in the corpus (irrelevant here: VTR re-uses the
  drafts' already-retrieved evidence; no new web search is performed).

SAFETY (hard guards; refuses to run if tripped)
-----------------------------------------------
* Source drafts under results/experiments/ are READ-ONLY (opened read-only, never written).
* ALL writes land under NEW dirs only:
    results/experiments_e11_vtr/   (refined .md reports, ready for the namespaced judge)
    reports/e11_vtr/               (run manifest, per-report VTR sidecars, cost ledger)
  Neither overlaps the protected corpus (results/judge_gpt52, results/experiments,
  data/analysis/*.parquet, reports/eval_v2/verdicts).  A startup guard refuses any output
  path that resolves into a protected location.
* sys.path.insert(0, repo_root) near the top so `python scripts/run_e11_vtr.py` never
  ModuleNotFound-crashes.
* --dry-run / --limit make ZERO API calls (no LLMCaller is constructed) and write nothing
  outside an explicit scratch dir; they print the full plan + a per-report budget estimate.

OUTPUT LAYOUT (so run_gpt52_judge_namespaced.py ingests it for free)
--------------------------------------------------------------------
  results/experiments_e11_vtr/<arm_pattern>/<query_id>.md       <- the refined report
     arm_pattern examples: e11_vtr_p0_gpt4o, e11_vtr_p4_gpt4o, e11_ctrl_p0, e11_ctrl_p4
  reports/e11_vtr/manifest.json                                  <- run plan + estimates
  reports/e11_vtr/sidecars/<arm_pattern>/<query_id>.json         <- per-round VTR telemetry

The human then judges with (writes ONLY to a new judge root, NEVER the corpus):
  python scripts/run_gpt52_judge_namespaced.py \
      --judge-out results/judge_gpt52_e11 \
      --patterns-raw e11_vtr_p0_gpt4o,e11_vtr_p4_gpt4o,e11_ctrl_p0,e11_ctrl_p4
  (NB the namespaced runner reads results/experiments/<pattern>/ — see "JUDGE WIRING"
   in the manifest for the exact, corpus-safe invocation against THIS dir.)

USAGE
-----
  # zero-API smoke test (no LLMCaller built, nothing written outside scratch):
  python scripts/run_e11_vtr.py --dry-run
  python scripts/run_e11_vtr.py --dry-run --limit 2 --bases p0 --arms vtr_gpt4o

  # full paid run (launched separately by the human; NOT launched here):
  python scripts/run_e11_vtr.py --run --bases p0,p4 --arms vtr_gpt4o,control --rounds 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# MUST be present so `python scripts/run_e11_vtr.py` does not crash with
# ModuleNotFoundError.  Do NOT remove.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
# READ-ONLY source corpus of un-refined drafts (the matched gpt-4o reports).
RESULTS_BASE = _REPO_ROOT / "results" / "experiments"
# NEW write roots — deliberately distinct top-level dirs, NOT the corpus.
OUT_REPORTS = _REPO_ROOT / "results" / "experiments_e11_vtr"   # refined .md reports
OUT_META = _REPO_ROOT / "reports" / "e11_vtr"                  # manifests + sidecars
SCRATCH = _REPO_ROOT / "results" / "_scratch_e11_dryrun"       # dry-run only

VARIANCE_SET = _REPO_ROOT / "data" / "variance_stratified.json"
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"

# Quarantined from the GPT-5.2 / Claude judge panels (reproducible AUP false-positive).
# We still GENERATE its refined draft (judge-specific exclusion), but flag it so the
# net-effect analysis can drop it from the GPT-5.2 contrast.
QUARANTINE_QID = "82de3e92-abe2-46ac-ad17-23417b9c4da7"

# ── Protected (never-write) paths — the irreplaceable corpus ──────────────────
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "results" / "experiments_e4_cite",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

# Budget accountant: the matched per-report token budget.  We match the existing
# corpus's single-shot generation cap (8192 completion tokens, as used by P0/P4
# generation) so refinement compute per round is on the same scale.  The full VTR
# spend per (query, base, arm) is then approximately:
#   R verifier passes (verifier reads draft + 38 criteria) + R generator refines.
GEN_MAX_TOKENS = 8192          # matched to P0/P4 single-shot generation cap
VERIFY_MAX_TOKENS = 4096       # verifier emits structured per-criterion fix list
DEFAULT_ROUNDS = 2
REPORT_TRUNCATE_WORDS = 12000  # same truncation the judge uses, so verifier sees same text

# ── Imports that touch config/LLM are done lazily inside _run() so --dry-run and
#    --limit make ZERO API calls and don't even require a configured endpoint. ──


# ── Safety guards ─────────────────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_out(path: Path) -> Path:
    """Refuse any output root that equals/sits inside/parents a protected path."""
    p = path.resolve()
    for protected in PROTECTED_PATHS:
        prot = protected.resolve()
        if p == prot:
            raise SystemExit(
                f"REFUSING: output path {p} IS protected corpus path {prot}."
            )
        if _is_relative_to(p, prot):
            raise SystemExit(
                f"REFUSING: output path {p} is INSIDE protected path {prot}."
            )
        if _is_relative_to(prot, p):
            raise SystemExit(
                f"REFUSING: output path {p} is a PARENT of protected path {prot}; "
                f"a run rooted there could traverse into the corpus."
            )
    return p


def assert_safe_arm_name(name: str) -> None:
    """Belt-and-braces: refuse arm/pattern names with traversal aliases."""
    target = (OUT_REPORTS / name).resolve()
    for protected in PROTECTED_PATHS:
        prot = protected.resolve()
        if target == prot or _is_relative_to(target, prot):
            raise SystemExit(
                f"REFUSING: arm name {name!r} resolves write target {target} "
                f"into protected path {prot}."
            )


# ── Rubric (the 38 pre-generated general criteria the verifier scores against) ─

def load_general_criteria():
    """The 38 general rubric_v2 criteria, pre-generated (verifier targets these)."""
    from deep_research.evaluation.rubric_v2 import build_general_criteria
    return build_general_criteria()


def load_variance_qids() -> list[str]:
    data = json.loads(VARIANCE_SET.read_text())
    return list(data["query_ids"])


def load_queries() -> dict[str, dict]:
    data = json.loads(EVAL_QUERIES.read_text())
    return {q["id"]: q for q in data["queries"]}


# ── Prompts (ALL run on gpt-4o / PTU = DEFAULT_MODEL) ──────────────────────────

VERIFIER_SYSTEM = """You are a meticulous research-report VERIFIER. You assess a draft report \
against explicit evaluation criteria and, for each criterion the draft does NOT fully satisfy, \
write a concrete, targeted fix the author can apply using ONLY information already present in \
the draft and its cited sources. Never invent new facts or sources. Respond with valid JSON only."""


def verifier_prompt(query: str, draft: str, criteria) -> str:
    crit_lines = "\n".join(
        f"  {i}. [{c.dimension}] {c.text}" for i, c in enumerate(criteria)
    )
    return f"""## Research Query
{query}

## Draft Report
{draft}

## Evaluation Criteria
{crit_lines}

For EACH criterion, decide SATISFIED or NOT_SATISFIED for the draft above. For every \
NOT_SATISFIED criterion, write one concrete, targeted fix instruction that the author can \
apply WITHOUT introducing any new external facts or sources (reorganise, make an existing \
implicit claim explicit, add an inline citation to an ALREADY-cited source, surface a \
synthesis already supported by the draft's evidence, resolve an internal contradiction, etc.).

Return JSON exactly:
{{
  "evaluations": [
    {{"criterion_index": 0, "verdict": "SATISFIED" or "NOT_SATISFIED",
      "fix": "targeted instruction, or empty string if SATISFIED"}}
  ]
}}"""


REFINE_SYSTEM = """You are a research analyst REVISING your own report. Apply the requested \
targeted fixes precisely. Preserve everything already correct; do not remove substantive \
content; do not introduce new external facts or sources beyond what the draft already cites. \
Keep the same overall structure (title, abstract, sections, references). Output the full \
revised report in Markdown."""


def refine_prompt(query: str, draft: str, fixes: list[str]) -> str:
    fix_block = "\n".join(f"  - {f}" for f in fixes) if fixes else "  (none)"
    return f"""Research query: {query}

Current draft report:
{draft}

A verifier identified these targeted fixes to apply. Apply ALL of them, changing only what \
each fix requires and leaving the rest of the report intact:
{fix_block}

Rewrite the FULL report incorporating every fix above. Keep the title, ## Abstract, logical \
## Section headings, and a References section. Use inline numbered citations [1], [2], ... as \
in the draft. Do not add facts or sources not already present. Output only the revised report."""


CONTROL_SYSTEM = """You are a research analyst REVISING your own report to improve its overall \
quality. Do not introduce new external facts or sources beyond what the draft already cites. \
Keep the same overall structure. Output the full revised report in Markdown."""


def control_prompt(query: str) -> str:
    return f"""Research query: {query}

Below is a draft research report. Revise it to improve its overall quality: clarity, \
completeness, analytical depth, citation discipline, organisation, and internal consistency. \
Do not add facts or sources not already present in the draft. Keep the title, ## Abstract, \
## Section headings, and a References section, with inline numbered citations [1], [2], ... \
Output only the revised report."""


# ── Arm / pattern naming ──────────────────────────────────────────────────────

def arm_pattern(base: str, arm: str) -> str:
    """base in {p0,p4}; arm in {vtr_gpt4o, control}. -> write subdir name."""
    if arm == "vtr_gpt4o":
        return f"e11_vtr_{base}_gpt4o"
    if arm == "control":
        return f"e11_ctrl_{base}"
    if arm == "vtr_drjudge":
        return f"e11_vtr_{base}_drjudge"
    raise SystemExit(f"unknown arm {arm!r}")


def turn0_verdict_dir(base: str) -> Path:
    """Where the EXISTING un-refined GPT-5.2 verdicts live (read-only baseline)."""
    return _REPO_ROOT / "results" / "judge_gpt52" / f"base_{base}"


# ── DR-Judge verifier: fixed Youden-J operating point ─────────────────────────

# New canonical location (post-0a80ba6); resolved, not hardcoded to the stale path.
CANON_PATH = (_REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
              / "canonical_numbers.json")
DRJUDGE_CHECKPOINT = "checkpoint-7617"  # git-pinned adapter checkpoint on disk


def load_drjudge_youden_op() -> dict:
    """Load the FIXED per-dimension signed-Youden-J operating point for DR-Judge-7B.

    Returns {dimension: {youden_j_signed, tpr, fpr, phase}, ...} from canonical key
    `drjudge_youden_j`. This pins the DR-Judge SATISFIED/NOT_SATISFIED decision to a
    reproducible operating point instead of an ad-hoc threshold. A dimension whose
    signed J <= j_zero_epsilon is at/below chance (no informedness): the verifier's
    NOT_SATISFIED flags on that dimension carry no signal and are NOT acted on.
    """
    cn = json.loads(CANON_PATH.read_text())
    yj = cn["drjudge_youden_j"]
    eps = float(yj.get("j_zero_epsilon", 0.05))
    per_dim = yj["judges"]["DR-Judge-7B"]["per_dimension"]
    op = {"_j_zero_epsilon": eps, "per_dimension": {}}
    for dim, v in per_dim.items():
        op["per_dimension"][dim] = {
            "youden_j_signed": v.get("youden_j_signed"),
            "tpr": v.get("tpr"),
            "fpr": v.get("fpr"),
            "phase": v.get("phase"),
            "actionable": (v.get("youden_j_signed") is not None
                           and float(v["youden_j_signed"]) > eps),
        }
    return op


def _drjudge_fixes(evals, criteria, op: dict | None):
    """Map DR-Judge per-criterion verdicts -> (fixes, n_failed) at the fixed Youden-J op.

    The DR-Judge model emits a native binary verdict per criterion at temperature 0.1
    (its argmax decision IS the operating point the canonical J was measured at). We
    additionally GATE on the per-dimension signed-J: a NOT_SATISFIED flag on a dimension
    whose J <= epsilon (at/below chance, no informedness) is dropped, because acting on a
    chance-level detector is exactly the verifier-quality confound the 3-arm design tests.
    This keeps the operating point fixed and reproducible, never tuned post-hoc.
    """
    per_dim = (op or {}).get("per_dimension", {})
    fixes: list[str] = []
    n_failed = 0
    for ev in evals:
        if str(ev.get("verdict", "")).upper() != "NOT_SATISFIED":
            continue
        idx = ev.get("criterion_index")
        dim = None
        if isinstance(idx, int) and 0 <= idx < len(criteria):
            dim = criteria[idx].dimension
        # Gate: only act on dimensions where DR-Judge has measured informedness (J>eps).
        if dim is not None and per_dim:
            if not per_dim.get(dim, {}).get("actionable", True):
                continue
        n_failed += 1
        fix = str(ev.get("fix", "")).strip()
        if fix:
            fixes.append(fix)
    return fixes, n_failed


def _flagged_criteria(evals, criteria) -> list[dict]:
    """The NOT_SATISFIED criteria the verifier flagged, with index + dimension + text.

    Recorded per round in the sidecar so the landing script can ask, for each flagged
    criterion, whether the GPT-5.2 verdict on that dimension improved (repair) or
    degraded (regression) — the per-criterion repair-vs-regression map.
    """
    out: list[dict] = []
    for ev in evals:
        if str(ev.get("verdict", "")).upper() != "NOT_SATISFIED":
            continue
        idx = ev.get("criterion_index")
        dim = None
        text = None
        if isinstance(idx, int) and 0 <= idx < len(criteria):
            dim = criteria[idx].dimension
            text = criteria[idx].text
        out.append({"criterion_index": idx, "dimension": dim, "criterion": text})
    return out


# ── Per-report VTR loop (only called under --run) ─────────────────────────────

def _truncate_words(text: str, n: int) -> str:
    w = text.split()
    if len(w) <= n:
        return text
    return " ".join(w[:n]) + "\n\n[... report truncated for verification ...]"


async def vtr_one(llm, query: str, draft0: str, criteria, arm: str,
                  rounds: int, drjudge=None, drjudge_op=None) -> tuple[str, dict]:
    """Run the VTR (or control) loop. Returns (refined_md, telemetry).

    arm:
      control      -> unguided revision, gpt-4o generator, no verifier signal.
      vtr_gpt4o    -> gpt-4o verifier (PTU)  -> gpt-4o refiner (PTU).
      vtr_drjudge  -> DR-Judge-7B verifier (LOCAL GPU) -> gpt-4o refiner (PTU).
                      DR-Judge is ONLY the verifier; the authoritative judge is
                      ALWAYS GPT-5.2 run separately. The SATISFIED/NOT_SATISFIED
                      mapping is pinned to the canonical signed-Youden-J operating
                      point (drjudge_op), not an ad-hoc threshold.
    """
    from deep_research.config import DEFAULT_MODEL  # gpt-4o; NEVER overridden

    if arm == "vtr_drjudge" and drjudge is None:
        raise SystemExit("vtr_drjudge requires a constructed DRJudgeCaller (drjudge=...).")

    rounds_log: list[dict] = []
    current = draft0
    for r in range(1, rounds + 1):
        seen = _truncate_words(current, REPORT_TRUNCATE_WORDS)

        if arm == "control":
            # Unguided revision: no verifier signal, same generator + budget.
            refined = await llm.complete(
                control_prompt(query) + "\n\nDraft report:\n" + seen,
                model=DEFAULT_MODEL,
                system=CONTROL_SYSTEM,
                max_tokens=GEN_MAX_TOKENS,
                temperature=0.3,
            )
            rounds_log.append({
                "round": r, "arm": arm,
                "n_failed_criteria": None, "fixes": [],
                "len_before": len(current), "len_after": len(refined),
            })
            current = refined
            continue

        # VTR: verify -> targeted refine.
        if arm == "vtr_drjudge":
            # VERIFIER = DR-Judge-7B on the LOCAL GPU (NOT the authoritative judge).
            # The model emits a per-criterion verdict + fix; we re-map the verdict to
            # SATISFIED/NOT_SATISFIED through the FIXED canonical signed-Youden-J
            # per-dimension operating point so the decision is reproducible, not ad-hoc.
            vres = await drjudge.complete_json(
                verifier_prompt(query, seen, criteria),
                system=VERIFIER_SYSTEM,
                max_tokens=VERIFY_MAX_TOKENS,
                temperature=0.1,
            )
            evals = vres.get("evaluations", []) if isinstance(vres, dict) else []
            fixes, n_failed = _drjudge_fixes(evals, criteria, drjudge_op)
        else:
            vres = await llm.complete_json(
                verifier_prompt(query, seen, criteria),
                model=DEFAULT_MODEL,          # gpt-4o verifier on PTU
                system=VERIFIER_SYSTEM,
                max_tokens=VERIFY_MAX_TOKENS,
                temperature=0.1,
            )
            evals = vres.get("evaluations", []) if isinstance(vres, dict) else []
            fixes = []
            n_failed = 0
            for ev in evals:
                if str(ev.get("verdict", "")).upper() == "NOT_SATISFIED":
                    n_failed += 1
                    fix = str(ev.get("fix", "")).strip()
                    if fix:
                        fixes.append(fix)

        # The per-criterion flagged-NOT_SATISFIED list (criterion text + dimension),
        # recorded in the sidecar so build_e11_vtr.py can compute the per-criterion
        # repair-vs-regression map against the GPT-5.2 verdicts for free.
        flagged = _flagged_criteria(evals, criteria)

        if not fixes:
            rounds_log.append({
                "round": r, "arm": arm, "n_failed_criteria": n_failed,
                "fixes": [], "flagged_criteria": flagged,
                "len_before": len(current), "len_after": len(current),
                "note": "no actionable fixes; refinement skipped this round",
            })
            break  # converged: verifier had nothing to repair

        refined = await llm.complete(
            refine_prompt(query, seen, fixes),
            model=DEFAULT_MODEL,          # gpt-4o generator on PTU
            system=REFINE_SYSTEM,
            max_tokens=GEN_MAX_TOKENS,
            temperature=0.3,
        )
        rounds_log.append({
            "round": r, "arm": arm, "n_failed_criteria": n_failed,
            "fixes": fixes, "flagged_criteria": flagged,
            "len_before": len(current), "len_after": len(refined),
        })
        current = refined

    telemetry = {
        "arm": arm, "rounds_run": len(rounds_log), "rounds": rounds_log,
        "len_draft0": len(draft0), "len_final": len(current),
    }
    return current, telemetry


# ── Work-list construction (shared by dry-run and run) ────────────────────────

def build_worklist(bases: list[str], arms: list[str], qids: list[str],
                   queries: dict[str, dict]) -> list[dict]:
    work: list[dict] = []
    for base in bases:
        src_dir = RESULTS_BASE / f"base_{base}"
        for qid in qids:
            md = src_dir / f"{qid}.md"
            if not md.exists():
                continue
            if qid not in queries:
                continue
            for arm in arms:
                work.append({
                    "base": base, "arm": arm, "qid": qid,
                    "arm_pattern": arm_pattern(base, arm),
                    "src_md": md,
                    "quarantined": (qid == QUARANTINE_QID),
                })
    return work


def estimate(work: list[dict], rounds: int) -> dict:
    """Cost/call estimate. GPT-4o is on PTU ($0 marginal). GPT-5.2 judging is the
    only paid step and is done LATER by the namespaced runner."""
    n_items = len(work)
    # gpt-4o generation calls (PTU, $0): per VTR item = R verify + up-to-R refine;
    # per control item = R refine.
    gen_calls = 0
    for w in work:
        if w["arm"] == "control":
            gen_calls += rounds
        else:
            gen_calls += rounds * 2  # verify + refine each round (upper bound)
    # GPT-5.2 judge calls the human will run NEXT: one per refined report
    # (one call per report under the namespaced DRACO single-call path).
    n_judged = sum(1 for w in work if not w["quarantined"])
    # GPT-5.2 cost: namespaced runner estimates ~$0.08/report.
    est_gpt52_calls = n_judged
    est_cost_usd = round(n_judged * 0.08, 2)
    return {
        "n_items": n_items,
        "gpt4o_ptu_gen_calls_upper_bound": gen_calls,
        "gpt4o_ptu_cost_usd": 0.0,
        "est_gpt52_judge_calls": est_gpt52_calls,
        "est_gpt52_cost_usd": est_cost_usd,
        "note": "GPT-4o generation is PTU ($0 marginal). GPT-5.2 judging (the only paid "
                "step) is performed SEPARATELY by run_gpt52_judge_namespaced.py and costs "
                "~$0.08/report; quarantined query excluded from the judged count.",
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def judge_wiring(arm_patterns: list[str]) -> dict:
    """The exact corpus-safe path to judge THIS dir's refined reports with GPT-5.2.

    run_gpt52_judge_namespaced.py reads its inputs from a hardcoded
    RESULTS_BASE = results/experiments/<pattern>/ (verified: it has --judge-out for
    the WRITE side but NO flag for the READ side).  The refined reports live OUTSIDE
    the corpus in results/experiments_e11_vtr/.  Two corpus-safe ways to bridge that,
    both writing verdicts ONLY to a NEW judge root (never results/judge_gpt52):

      A) RESULTS_BASE override (no copy): launch the namespaced runner with one env-style
         edit so it reads results/experiments_e11_vtr; then --patterns-raw <arm names>.
      B) the e11 arm dirs use e11_* names that CANNOT collide with the base_p* corpus
         dirs, so they may be symlinked into results/experiments and judged in place.

    Either way: --judge-out results/judge_gpt52_e11 (a NEW root), judge = GPT-5.2 only.
    """
    return {
        "writes_refined_reports_to": str(OUT_REPORTS.relative_to(_REPO_ROOT)),
        "arm_patterns": arm_patterns,
        "judge_results_base_note": (
            "run_gpt52_judge_namespaced.py hardcodes RESULTS_BASE=results/experiments and "
            "has NO --results-base flag. To judge the e11 reports corpus-safely, EITHER "
            "(A) point that runner's RESULTS_BASE at results/experiments_e11_vtr (one line), "
            "OR (B) symlink results/experiments_e11_vtr/<arm> -> results/experiments/<arm> "
            "(the e11_* names cannot collide with base_p* corpus dirs)."
        ),
        "turn0_baseline_verdicts": {
            "p0": str(turn0_verdict_dir("p0").relative_to(_REPO_ROOT)),
            "p4": str(turn0_verdict_dir("p4").relative_to(_REPO_ROOT)),
            "note": "Existing GPT-5.2 verdicts for the UN-refined drafts; the VTR net "
                    "effect = refined_overall - turn0_overall, per (base,qid).",
        },
        "judge_command_template": (
            "python scripts/run_gpt52_judge_namespaced.py "
            "--judge-out results/judge_gpt52_e11 "
            "--patterns-raw " + ",".join(arm_patterns) + "  "
            "# after wiring RESULTS_BASE->results/experiments_e11_vtr per the note above"
        ),
        "judge_command_note": (
            "GPT-5.2 is the ONLY authoritative judge. --judge-out is a NEW root (never "
            "results/judge_gpt52). Generation stays gpt-4o/PTU throughout."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def _run(args) -> int:
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for b in bases:
        if b not in ("p0", "p4"):
            raise SystemExit(f"--bases must be p0 and/or p4; got {b!r}")
    for a in arms:
        if a not in ("vtr_gpt4o", "control", "vtr_drjudge"):
            raise SystemExit(
                f"--arms must be vtr_gpt4o, control, and/or vtr_drjudge; got {a!r}."
            )

    # The DR-Judge verifier arm runs a local 7B 4-bit model on the single 16GB GPU.
    # That arm is ON: weights are on disk under models/DR-Judge-7B-LoRA/ (adapter +
    # checkpoint-7617), which contradicts the script's earlier 'weights not on disk' note.
    use_drjudge = "vtr_drjudge" in arms
    if use_drjudge and not (_REPO_ROOT / "models" / "DR-Judge-7B-LoRA" /
                            "adapter_model.safetensors").exists():
        raise SystemExit(
            "REFUSING: --arms includes vtr_drjudge but models/DR-Judge-7B-LoRA/"
            "adapter_model.safetensors is NOT on disk. Download the adapter first."
        )

    # Guard every write root BEFORE anything else.
    out_reports = assert_safe_out(OUT_REPORTS)
    out_meta = assert_safe_out(OUT_META)
    for base in bases:
        for arm in arms:
            assert_safe_arm_name(arm_pattern(base, arm))

    qids = load_variance_qids()
    queries = load_queries()
    criteria = load_general_criteria()
    if len(criteria) != 38:
        print(f"  WARNING: expected 38 general criteria, got {len(criteria)}")

    work = build_worklist(bases, arms, qids, queries)
    if args.limit and args.limit > 0:
        # Deterministic: keep the first N (qids are in the fixed variance order).
        work = work[: args.limit]

    arm_patterns = sorted({w["arm_pattern"] for w in work})
    est = estimate(work, args.rounds)

    # Load the FIXED DR-Judge operating point (canonical) whenever the arm is requested,
    # so it is recorded in the manifest even on --dry-run (ZERO API/GPU).
    drjudge_op = None
    if use_drjudge:
        try:
            drjudge_op = load_drjudge_youden_op()
        except Exception as e:  # canonical missing/malformed -> fail loud, never silent
            raise SystemExit(
                f"REFUSING: vtr_drjudge needs canonical['drjudge_youden_j'] at "
                f"{CANON_PATH} but loading it failed: {type(e).__name__}: {e}"
            )

    verifier_model = ("gpt-4o (PTU) for vtr_gpt4o; control arm uses NO verifier"
                      if not use_drjudge else
                      "gpt-4o (PTU) for vtr_gpt4o; DR-Judge-7B-LoRA (QLoRA Qwen2.5-7B-"
                      "Instruct adapter; local GPU) for vtr_drjudge; control = NO verifier")
    if use_drjudge:
        drjudge_hook = {
            "status": "ON — adapter on disk (models/DR-Judge-7B-LoRA/)",
            "verifier_model": "DR-Judge-7B-LoRA (QLoRA Qwen2.5-7B-Instruct adapter; local GPU)",
            "adapter_dir": "models/DR-Judge-7B-LoRA",
            "adapter_checkpoint": DRJUDGE_CHECKPOINT,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4-bit nf4 + double-quant (peak ~14.6 GiB on RTX 5080)",
            "role": "VERIFIER ONLY — never the authoritative judge (GPT-5.2 remains the "
                    "sole judge); refiner stays gpt-4o on PTU (DEFAULT_MODEL).",
            "operating_point": "canonical['drjudge_youden_j'] signed-Youden-J per dimension; "
                               "NOT_SATISFIED flags acted on only where signed J > "
                               "j_zero_epsilon (chance-level dimensions dropped).",
            "youden_j_per_dimension": (drjudge_op or {}).get("per_dimension", {}),
            "j_zero_epsilon": (drjudge_op or {}).get("_j_zero_epsilon"),
            "concurrency_forced": 1,
            "concurrency_reason": "single 16GB card; DR-Judge 7B 4-bit verify is the serial "
                                  "GPU bottleneck. One shared DRJudgeCaller, _ensure_loaded() "
                                  "once before gather; PTU refiner stays async behind it.",
        }
    else:
        drjudge_hook = {
            "status": "OFF — arm not selected (pass --arms vtr_drjudge to enable; "
                      "adapter IS on disk at models/DR-Judge-7B-LoRA/)",
            "model": "DR-Judge-7B-LoRA (QLoRA Qwen2.5-7B-Instruct adapter)",
        }

    manifest = {
        "_what": "E11 P14-VTR — verify-then-refine at matched budget (PTU generation).",
        "experiment": "E11_P14_VTR",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_model": "gpt-4o (DEFAULT_MODEL, PTU deployment sthree-ptu-02) — "
                            "NEVER overridden; same backbone as the 248k corpus",
        "verifier_model": verifier_model,
        "judge_model": "gpt-5.2 (authoritative; run SEPARATELY via "
                       "run_gpt52_judge_namespaced.py)",
        "search_model": "gpt-4o-mini (corpus default; no new search performed — VTR reuses "
                        "each draft's already-retrieved evidence)",
        "bases": bases,
        "arms": arms,
        "rounds": args.rounds,
        "n_variance_queries": len(qids),
        "quarantined_query": QUARANTINE_QID,
        "matched_budget": {
            "gen_max_tokens": GEN_MAX_TOKENS,
            "verify_max_tokens": VERIFY_MAX_TOKENS,
            "rationale": "GEN_MAX_TOKENS matches the corpus P0/P4 single-shot generation cap "
                         "(8192), so per-round refinement compute is on the same scale; the "
                         "VTR net effect is measured against the un-refined draft at "
                         "matched per-pass budget.",
        },
        "writes_to": [
            str(out_reports.relative_to(_REPO_ROOT)),
            str(out_meta.relative_to(_REPO_ROOT)),
        ],
        "protected_read_only": [str(p.relative_to(_REPO_ROOT)) for p in PROTECTED_PATHS],
        "drjudge_verifier_hook": drjudge_hook,
        "estimate": est,
        "judge_wiring": judge_wiring(arm_patterns),
        "n_work_items": len(work),
    }

    if args.dry_run:
        # ZERO API calls; no LLMCaller constructed. Write nothing outside scratch.
        print(json.dumps({
            "DRY_RUN": True,
            "bases": bases, "arms": arms, "rounds": args.rounds,
            "arm_patterns": arm_patterns,
            "n_work_items": len(work),
            "first_items": [
                {"arm_pattern": w["arm_pattern"], "qid": w["qid"],
                 "src": str(w["src_md"].relative_to(_REPO_ROOT)),
                 "quarantined": w["quarantined"]}
                for w in work[: min(6, len(work))]
            ],
            "estimate": est,
            "writes_to": manifest["writes_to"],
            "judge_command_template": manifest["judge_wiring"]["judge_command_template"],
        }, indent=2))
        # Sanity: confirm the source drafts are readable (no write, no API).
        missing = [str(w["src_md"]) for w in work if not w["src_md"].exists()]
        if missing:
            print(f"\n  WARNING: {len(missing)} source drafts missing (e.g. {missing[:2]})")
        if args.write_manifest:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            write_json(SCRATCH / "manifest_dryrun.json", manifest)
            print(f"\n[dry-run] wrote scratch manifest to "
                  f"{(SCRATCH / 'manifest_dryrun.json').relative_to(_REPO_ROOT)}")
        print("\n[dry-run] ZERO API calls; no LLMCaller built; corpus untouched.",
              file=sys.stderr)
        return 0

    # ── PAID RUN ──────────────────────────────────────────────────────────────
    if not work:
        print("  Nothing to do.")
        return 0

    out_reports.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    write_json(out_meta / "manifest.json", manifest)

    # Construct the PTU caller ONLY now (paid path).
    from deep_research.tools import CostTracker, LLMCaller
    tracker = CostTracker(budget_usd=args.budget_usd)
    llm = LLMCaller(cost_tracker=tracker)

    # GPU SERIALISATION: when the DR-Judge verifier arm is selected, the single 16GB card
    # is the serial bottleneck (DR-Judge 7B 4-bit peak ~14.6 GiB). Force concurrency=1 and
    # construct ONE shared DRJudgeCaller, loading the adapter exactly once before gather.
    # (The PTU gpt-4o refiner calls remain async, but the GPU verify step gates throughput.)
    drjudge = None
    eff_concurrency = args.concurrency
    if use_drjudge:
        if eff_concurrency != 1:
            print(f"  [vtr_drjudge] forcing concurrency 1 (single 16GB GPU); "
                  f"requested {eff_concurrency} ignored for GPU serialisation.")
        eff_concurrency = 1
        from deep_research.tools.dr_judge_caller import DRJudgeCaller
        drjudge = DRJudgeCaller()  # default adapter dir models/DR-Judge-7B-LoRA
        print("  [vtr_drjudge] loading DR-Judge-7B-LoRA adapter onto GPU "
              "(_ensure_loaded; one-time)...")
        await asyncio.get_event_loop().run_in_executor(None, drjudge._ensure_loaded)
        print("  [vtr_drjudge] adapter loaded; verify step is the serial bottleneck.")

    sem = asyncio.Semaphore(eff_concurrency)
    done = {"ok": 0, "fail": 0}
    t0 = time.time()

    async def process(w: dict) -> None:
        out_md = out_reports / w["arm_pattern"] / f"{w['qid']}.md"
        side = out_meta / "sidecars" / w["arm_pattern"] / f"{w['qid']}.json"
        if args.resume and out_md.exists() and side.exists():
            return
        draft0 = w["src_md"].read_text()
        query = queries[w["qid"]]["query"]
        async with sem:
            try:
                refined, telem = await vtr_one(
                    llm, query, draft0, criteria, w["arm"], args.rounds,
                    drjudge=drjudge, drjudge_op=drjudge_op,
                )
                out_md.parent.mkdir(parents=True, exist_ok=True)
                out_md.write_text(refined)
                telem.update({
                    "base": w["base"], "arm_pattern": w["arm_pattern"],
                    "qid": w["qid"], "quarantined": w["quarantined"],
                    "src_md": str(w["src_md"].relative_to(_REPO_ROOT)),
                })
                write_json(side, telem)
                done["ok"] += 1
                print(f"  [{done['ok']+done['fail']}/{len(work)}] "
                      f"{w['arm_pattern']}/{w['qid']}: "
                      f"{telem['rounds_run']} rounds, "
                      f"{telem['len_draft0']}->{telem['len_final']} chars")
            except Exception as e:
                done["fail"] += 1
                print(f"  [{done['ok']+done['fail']}/{len(work)}] "
                      f"{w['arm_pattern']}/{w['qid']}: FAILED — {type(e).__name__}: "
                      f"{str(e)[:160]}")

    await asyncio.gather(*(process(w) for w in work))

    # Cost ledger (PTU = $0, but record token usage for accounting).
    ledger = {
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "completed": done["ok"], "failed": done["fail"],
        "ptu_total_tokens": getattr(tracker, "total_tokens", None),
        "ptu_total_cost_usd": getattr(tracker, "total_cost", 0.0),
        "effective_concurrency": eff_concurrency,
        "drjudge_verifier_used": use_drjudge,
        "drjudge_adapter_checkpoint": DRJUDGE_CHECKPOINT if use_drjudge else None,
        "estimate": est,
    }
    write_json(out_meta / "cost_ledger.json", ledger)
    print(f"\n  DONE: {done['ok']} ok, {done['fail']} failed, "
          f"{ledger['elapsed_min']} min. PTU tokens={ledger['ptu_total_tokens']}.")
    print(f"  Refined reports -> {out_reports.relative_to(_REPO_ROOT)}")
    print(f"  NEXT (separate, paid): {manifest['judge_wiring']['judge_command_template']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="ZERO API calls: print plan + budget estimate; build no LLMCaller.")
    mode.add_argument("--run", action="store_true",
                      help="Execute the PAID PTU generation run (launched by the human).")
    ap.add_argument("--bases", type=str, default="p0,p4",
                    help="Comma-separated base drafts: p0 and/or p4 (default p0,p4).")
    ap.add_argument("--arms", type=str, default="vtr_gpt4o,control",
                    help="Comma-separated arms: vtr_gpt4o, control, vtr_drjudge "
                         "(default vtr_gpt4o,control). vtr_drjudge uses the local "
                         "DR-Judge-7B-LoRA GPU verifier (forces concurrency=1).")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                    help=f"VTR refine rounds (default {DEFAULT_ROUNDS}).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap work items (first N, deterministic) — smoke-test aid. 0 = all.")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Max concurrent VTR loops (default 6).")
    ap.add_argument("--budget-usd", type=float, default=50.0,
                    help="CostTracker ceiling for the PTU run (PTU is $0; safety stop only).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (qid,arm) pairs whose refined .md + sidecar already exist.")
    ap.add_argument("--write-manifest", action="store_true",
                    help="Under --dry-run, also write a scratch manifest (still ZERO API).")
    args = ap.parse_args()

    if not args.dry_run and not args.run:
        # Default to the SAFE mode: dry-run. Never silently launch a paid run.
        args.dry_run = True
        print("[no mode given] defaulting to --dry-run (safe). Use --run for the paid run.\n",
              file=sys.stderr)

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
