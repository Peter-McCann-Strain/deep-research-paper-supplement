#!/usr/bin/env python3
"""DRB-RACE staging + judging harness — CORPUS-SAFE.

Highest-value Phase-1 deliverable for the judge-science paper: run OUR own
9-dimension GPT-5.2 judge panel over the 400 DeepResearch-Bench (DRB-RACE)
released system reports, so a *separate* correlation step can compare our
per-dimension scores against the 150 expert human RACE records — the in-genre
human-validity number for the judge.

Data on disk (read-only inputs)
-------------------------------
* ``data/benchmarks/drb1/drb1_system_reports.json`` — dict keyed by the 4 DRA
  systems {gemini-2.5-pro-deepresearch, grok-deeper-search, openai-deepresearch,
  perplexity-Research}; each maps to a LIST of 100 items
  ``{"id": int, "prompt": str, "article": str}`` (100 tasks each = 400 reports).
* ``data/benchmarks/drb1/drb1_queries.json`` — list of 100 task prompts, each
  ``{"id": "drb1_<n>", "query": str, ..., "metadata": {"original_id": int}}``.
* ``data/benchmarks/drb1/drb1_human_annotations.json`` — 150 expert RACE records
  over 50 unique task ids (consumed by the downstream correlation, NOT here).

What this script does (mirrors scripts/run_e12_extval.py)
--------------------------------------------------------
``--stage stage``
    Stage the 400 reports as a read-only tree
    ``results/drbrace/_judge_stage/drbrace_<system>/<task_id>.md`` and write a
    synthetic eval manifest mapping each staged
    ``(pattern=drbrace_<system>, query_id=<task_id>)`` to the DRB task prompt +
    the 9-dimension V2 rubric (RACE dimensions are a subset that the downstream
    correlation selects). No API calls.

``--stage judge``
    Delegate to the corpus-safe namespaced GPT-5.2 runner
    (``scripts/run_gpt52_judge_namespaced.py``) with ``--judge-out
    results/drbrace/judge_gpt52``, ``JUDGE_RESULTS_BASE`` pointed at the staging
    tree, and a PRIVATE cwd whose ``data/eval_queries_v2.json`` symlinks to our
    DRB manifest (so the real corpus manifest is never touched). ``--resume``
    skips already-judged reports. ``--dry-run`` makes ZERO API calls.

Corpus safety (enforced)
------------------------
Writes ONLY under ``results/drbrace/``. A guard refuses to start if any output
path resolves into a protected corpus path (``results/judge_gpt52``,
``results/experiments``, ``data/analysis``, ``reports/eval_v2/verdicts``). The
namespaced judge applies the SAME guard independently.

Usage
-----
    # Stage the 400 reports + manifest (no API):
    python scripts/stage_and_judge_drbrace.py --stage stage

    # See exactly what the judge would grade (ZERO API spend):
    python scripts/stage_and_judge_drbrace.py --stage judge --dry-run

    # Real paid GPT-5.2 judging (human-launched), corpus-safe + resumable:
    python scripts/stage_and_judge_drbrace.py --stage judge --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# REQUIRED so `python scripts/stage_and_judge_drbrace.py` does not crash with
# ModuleNotFoundError (the failure mode that broke the detector panel).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deep_research.config import JUDGE_MODEL, MODELS  # noqa: E402

# ── DRB-RACE inputs (READ-ONLY) ───────────────────────────────────────────────
DRB_DIR = _REPO_ROOT / "data" / "benchmarks" / "drb1"
DRB_REPORTS = DRB_DIR / "drb1_system_reports.json"
DRB_QUERIES = DRB_DIR / "drb1_queries.json"
DRB_ANNOTATIONS = DRB_DIR / "drb1_human_annotations.json"

# The 4 commercial DRA systems whose released reports we stage + judge.
DRB_SYSTEMS = [
    "gemini-2.5-pro-deepresearch",
    "grok-deeper-search",
    "openai-deepresearch",
    "perplexity-Research",
]

# Source-type tag handed to the rubric builder.  DRB-RACE's native RACE axes
# (Comprehensiveness / depth / instruction-following / readability) map onto our
# dimensions per gold_loaders.DRB_DIMENSION_MAP; "drbench" picks the closest
# source-weight profile.  The 9 dims are scored identically regardless, so the
# weight profile only affects the (auxiliary) overall_score; the downstream
# correlation works on per-dimension scores.
DRB_SOURCE_TYPE = "drbench"

# ── Output roots — ALL brand-new, outside every protected path ────────────────
DRBRACE_ROOT = _REPO_ROOT / "results" / "drbrace"
STAGE_ROOT = DRBRACE_ROOT / "_judge_stage"          # drbrace_<system>/<task_id>.md
JUDGE_OUT = DRBRACE_ROOT / "judge_gpt52"            # verdicts (NEW dir, NOT corpus)
# Synthetic eval manifest the namespaced judge keys on (shape mirrors
# data/eval_queries_v2.json: {"queries": [...]}).
DRB_EVAL_QUERIES = DRBRACE_ROOT / "eval_queries_drbrace.json"
# Provenance: the staged (system, task_id) selection manifest.
DRB_STAGE_MANIFEST = DRBRACE_ROOT / "drbrace_stage_manifest.json"
# Private cwd whose data/eval_queries_v2.json -> our DRB manifest (judge reads it).
JUDGE_CWD = DRBRACE_ROOT / "_judge_cwd"

NAMESPACED_JUDGE = _REPO_ROOT / "scripts" / "run_gpt52_judge_namespaced.py"

# ── Protected (READ-ONLY, never-write) corpus paths ───────────────────────────
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

# GPT-5.2 cost estimate per judged report (matches the namespaced runner's
# heuristic: ~7k tokens/report; gpt-5.2 averaged in/out cost).
_GPT52_SPEC = MODELS.get("gpt-5.2")
_EST_TOKENS_PER_JUDGE = 7000


def pattern_for_system(system: str) -> str:
    """Staged judge pattern name for a DRB system (the staging subdir name)."""
    return f"drbrace_{system}"


# ── Safety guards ─────────────────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_output_paths_safe() -> None:
    """Refuse to start if any DRB-RACE output path lives in / contains a protected dir."""
    out_paths = [
        DRBRACE_ROOT, STAGE_ROOT, JUDGE_OUT, DRB_EVAL_QUERIES,
        DRB_STAGE_MANIFEST, JUDGE_CWD,
    ]
    for out in out_paths:
        out = out.resolve()
        for prot in PROTECTED_PATHS:
            prot = prot.resolve()
            if out == prot:
                raise SystemExit(f"REFUSING: DRB-RACE output {out} IS protected corpus path {prot}.")
            if _is_relative_to(out, prot):
                raise SystemExit(f"REFUSING: DRB-RACE output {out} is INSIDE protected path {prot}.")
            if _is_relative_to(prot, out):
                raise SystemExit(
                    f"REFUSING: DRB-RACE output {out} is a PARENT of protected path {prot} "
                    f"(a run rooted there could traverse into the corpus)."
                )


def assert_judge_is_gpt52() -> None:
    """Assert the authoritative judge is GPT-5.2 and the namespaced runner exists."""
    if JUDGE_MODEL != "gpt-5.2":
        raise SystemExit(
            f"REFUSING: JUDGE_MODEL={JUDGE_MODEL!r}, expected 'gpt-5.2'. "
            f"Never use gpt-4o/gpt-4.1/mini/local as a judge here."
        )
    if not NAMESPACED_JUDGE.exists():
        raise SystemExit(f"REFUSING: namespaced GPT-5.2 judge not found at {NAMESPACED_JUDGE}.")


def assert_inputs_present() -> None:
    for p in (DRB_REPORTS, DRB_QUERIES):
        if not p.exists():
            raise SystemExit(f"REFUSING: required DRB-RACE input missing: {p}")


# ── Load DRB-RACE inputs ──────────────────────────────────────────────────────

def load_query_prompts() -> dict[int, str]:
    """Map the integer task id (metadata.original_id) -> query text.

    Falls back to the report's own inline ``prompt`` when a query row is missing.
    """
    queries = json.loads(DRB_QUERIES.read_text(encoding="utf-8"))
    by_id: dict[int, str] = {}
    for q in queries:
        oid = (q.get("metadata") or {}).get("original_id")
        if oid is None:
            continue
        qtext = (q.get("query") or "").strip()
        if qtext:
            by_id[int(oid)] = qtext
    return by_id


def iter_reports() -> list[dict]:
    """Flatten the system-report tree into staged-report records.

    Yields one dict per (system, task) with keys:
      system, task_id (int), query_id (str e.g. 'drb1_51'), prompt, article.
    """
    reports = json.loads(DRB_REPORTS.read_text(encoding="utf-8"))
    query_prompts = load_query_prompts()
    out: list[dict] = []
    for system in DRB_SYSTEMS:
        items = reports.get(system)
        if items is None:
            raise SystemExit(f"REFUSING: system {system!r} not present in {DRB_REPORTS}.")
        for it in items:
            task_id = int(it["id"])
            article = it.get("article") or ""
            # Prefer the canonical query text; fall back to the report's inline prompt.
            prompt = query_prompts.get(task_id) or (it.get("prompt") or "").strip()
            out.append({
                "system": system,
                "task_id": task_id,
                # query_id is the per-report stem the judge keys on; pattern
                # already namespaces by system, so task id alone is unique within
                # a pattern subdir and never collides with corpus query ids.
                "query_id": f"drb1_{task_id}",
                "prompt": prompt,
                "article": article,
            })
    return out


# ── Staging ───────────────────────────────────────────────────────────────────

def write_eval_manifest(records: list[dict]) -> int:
    """Stage a DRB-specific eval_queries manifest the namespaced judge reads.

    Mirrors data/eval_queries_v2.json shape: ``{"queries": [...]}`` with ``id`` =
    the per-report query_id (``drb1_<task>``), ``query`` = the DRB task prompt,
    and ``source`` = DRB_SOURCE_TYPE so the judge applies the full 9-dim rubric.
    One manifest entry per UNIQUE task (the 4 systems share the same prompt), so
    the judge resolves each staged report by its task-id stem.

    Written to a NEW file under results/drbrace/ — the real manifest is untouched.
    """
    seen: dict[str, dict] = {}
    for rec in records:
        qid = rec["query_id"]
        if qid in seen:
            continue
        seen[qid] = {
            "id": qid,
            "query": rec["prompt"],
            "source": DRB_SOURCE_TYPE,
            # No expected_elements: DRB-RACE has no per-task coverage anchors in
            # the released assets; the 9 general-criteria dims are what we
            # correlate against the human RACE axes.
            "expected_elements": [],
            "metadata": {"drb_task_id": rec["task_id"], "benchmark": "drb_race"},
        }
    DRBRACE_ROOT.mkdir(parents=True, exist_ok=True)
    DRB_EVAL_QUERIES.write_text(
        json.dumps({"queries": list(seen.values())}, indent=2),
        encoding="utf-8",
    )
    return len(seen)


def stage_reports(records: list[dict]) -> tuple[int, int]:
    """Write the 400 reports as ``STAGE_ROOT/drbrace_<system>/<task_id>.md``.

    Returns (n_written, n_existing). The report article text is written verbatim
    (no symlink: the source is one big JSON, not per-file). Idempotent — existing
    staged files with identical content are left untouched.
    """
    written = 0
    existing = 0
    for rec in records:
        pat = pattern_for_system(rec["system"])
        dst_dir = STAGE_ROOT / pat
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{rec['query_id']}.md"
        article = rec["article"]
        if dst.exists() and dst.read_text(encoding="utf-8") == article:
            existing += 1
            continue
        dst.write_text(article, encoding="utf-8")
        written += 1
    return written, existing


def write_stage_provenance(records: list[dict]) -> None:
    """Persist the staged (system, task_id, query_id) selection for reproducibility."""
    manifest = {
        "benchmark": "drb_race",
        "systems": DRB_SYSTEMS,
        "source_type": DRB_SOURCE_TYPE,
        "n_reports": len(records),
        "n_tasks": len({r["task_id"] for r in records}),
        "judge_model": JUDGE_MODEL,
        "patterns": sorted({pattern_for_system(r["system"]) for r in records}),
        "reports": [
            {
                "pattern": pattern_for_system(r["system"]),
                "system": r["system"],
                "task_id": r["task_id"],
                "query_id": r["query_id"],
                "article_chars": len(r["article"]),
            }
            for r in records
        ],
    }
    DRB_STAGE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def do_stage() -> list[dict]:
    """Stage stage: write the 400-report tree + the synthetic manifest."""
    assert_output_paths_safe()
    assert_inputs_present()

    records = iter_reports()
    n_manifest = write_eval_manifest(records)
    n_written, n_existing = stage_reports(records)
    write_stage_provenance(records)

    patterns = sorted({pattern_for_system(r["system"]) for r in records})
    empty = [r for r in records if not r["article"].strip()]
    no_prompt = [r for r in records if not r["prompt"].strip()]

    print("=" * 72)
    print("DRB-RACE staging (CORPUS-SAFE)")
    print("=" * 72)
    print(f"  Reports staged    : {len(records)} ({n_written} written, {n_existing} unchanged)")
    print(f"  Systems / patterns: {len(patterns)}")
    for pat in patterns:
        n = sum(1 for r in records if pattern_for_system(r["system"]) == pat)
        print(f"      {pat:<36}: {n} reports")
    print(f"  Manifest entries  : {n_manifest} unique tasks  ->  {DRB_EVAL_QUERIES.name}")
    print(f"  Stage tree (NEW)  : {STAGE_ROOT}")
    print(f"  Provenance        : {DRB_STAGE_MANIFEST.name}")
    if empty:
        print(f"  WARNING: {len(empty)} staged reports have EMPTY article text "
              f"(e.g. {empty[0]['system']}/{empty[0]['query_id']}).")
    if no_prompt:
        print(f"  WARNING: {len(no_prompt)} reports have NO resolvable prompt "
              f"(e.g. {no_prompt[0]['system']}/{no_prompt[0]['query_id']}).")
    print("=" * 72)
    return records


# ── Judging (delegate to the namespaced GPT-5.2 runner) ───────────────────────

def _prepare_judge_cwd() -> Path:
    """Private cwd whose data/eval_queries_v2.json -> our DRB manifest.

    The namespaced judge hardcodes ``EVAL_QUERIES=Path("data/eval_queries_v2.json")``
    relative to its cwd.  We give it a cwd where that path resolves (via symlink)
    to results/drbrace/eval_queries_drbrace.json, while the runner still imports
    the real package (it inserts the repo root absolutely on sys.path).  The real
    data/eval_queries_v2.json is never modified.
    """
    (JUDGE_CWD / "data").mkdir(parents=True, exist_ok=True)
    link = JUDGE_CWD / "data" / "eval_queries_v2.json"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(DRB_EVAL_QUERIES)
    return JUDGE_CWD


def build_judge_cmd(records: list[dict], dry_run: bool, resume: bool,
                    concurrency: int, limit: int = 0) -> tuple[list[str], dict, list[str]]:
    """Build the namespaced GPT-5.2 judge command + env (no execution here)."""
    patterns = sorted({pattern_for_system(r["system"]) for r in records})
    cmd = [
        # ``-u`` forces the CHILD to run with unbuffered stdout/stderr.  Without
        # it, when this harness's stdout is a pipe/redirect (e.g. the agent
        # harness, or `> log 2>&1`), the child block-buffers its per-report
        # progress and NOTHING is flushed until exit — and the buffer is lost
        # outright if a foreground timeout kills the run, which looks exactly
        # like a silent "zero verdicts, empty log" failure even while verdicts
        # are being written to disk.
        sys.executable, "-u", str(NAMESPACED_JUDGE),
        "--judge-out", str(JUDGE_OUT),
        "--patterns-raw", ",".join(patterns),
        "--concurrency", str(concurrency),
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd.append("--resume")
    if dry_run:
        cmd.append("--dry-run")
    env = dict(os.environ)
    env["JUDGE_RESULTS_BASE"] = str(STAGE_ROOT)
    # Belt-and-braces: also ask Python (in the child) to keep stdout unbuffered.
    env["PYTHONUNBUFFERED"] = "1"
    return cmd, env, patterns


def count_pending(records: list[dict], resume: bool) -> int:
    """How many staged reports the judge would grade (honouring --resume)."""
    pending = 0
    for rec in records:
        pat = pattern_for_system(rec["system"])
        staged = STAGE_ROOT / pat / f"{rec['query_id']}.md"
        if not staged.exists():
            continue
        if resume and (JUDGE_OUT / pat / f"{rec['query_id']}.json").exists():
            continue
        pending += 1
    return pending


def estimate_cost(n_judge: int) -> float:
    spec = _GPT52_SPEC
    if spec is not None:
        avg_cost_per_1k = (spec.cost_per_1k_input + spec.cost_per_1k_output) / 2
    else:
        avg_cost_per_1k = (0.003 + 0.012) / 2
    return round(n_judge * (_EST_TOKENS_PER_JUDGE / 1000.0) * avg_cost_per_1k, 2)


def do_judge(args) -> None:
    """Judge stage: delegate to the corpus-safe namespaced GPT-5.2 runner."""
    # Line-buffer THIS process's banner/status prints too, so they stream live
    # under a pipe/redirect instead of being block-buffered (and lost on a
    # timeout-kill).  See build_judge_cmd for the child-side ``-u`` rationale.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    assert_output_paths_safe()
    assert_judge_is_gpt52()
    assert_inputs_present()

    # Reconstruct the staged record set; require staging first.
    records = iter_reports()
    staged_present = sum(
        1 for r in records
        if (STAGE_ROOT / pattern_for_system(r["system"]) / f"{r['query_id']}.md").exists()
    )
    if staged_present == 0:
        raise SystemExit(
            "REFUSING: no staged reports found. Run "
            "`python scripts/stage_and_judge_drbrace.py --stage stage` first."
        )
    if not DRB_EVAL_QUERIES.exists():
        raise SystemExit(
            f"REFUSING: staged manifest missing ({DRB_EVAL_QUERIES}). Run --stage stage first."
        )

    limit = getattr(args, "limit", 0) or 0
    cmd, env, patterns = build_judge_cmd(
        records, dry_run=args.dry_run, resume=args.resume,
        concurrency=args.concurrency, limit=limit,
    )
    pending = count_pending(records, resume=args.resume)
    if limit > 0:
        pending = min(pending, limit)
    est_cost = estimate_cost(pending)

    print("=" * 72)
    print("DRB-RACE judging — namespaced GPT-5.2 (CORPUS-SAFE)")
    print("=" * 72)
    print(f"  Judge model       : {JUDGE_MODEL} (namespaced runner, GPT-5.2 ONLY)")
    print(f"  Staged reports     : {staged_present} present under {STAGE_ROOT}")
    print(f"  Patterns           : {len(patterns)} ({', '.join(patterns)})")
    print(f"  Pending to judge   : {pending}  (resume={args.resume}"
          + (f", limit={limit}" if limit > 0 else "") + ")")
    print(f"  Est GPT-5.2 cost   : ${est_cost:.2f}  (~{_EST_TOKENS_PER_JUDGE} tok/report)")
    print(f"  Judge out (NEW)    : {JUDGE_OUT}")
    print(f"  JUDGE_RESULTS_BASE : {STAGE_ROOT}")
    print("=" * 72)

    judge_cwd = _prepare_judge_cwd()

    if args.dry_run:
        print("\n[DRY RUN] Would judge with the namespaced runner:")
        print("   " + " ".join(cmd))
        print(f"   cwd={judge_cwd}")
        print(f"   JUDGE_RESULTS_BASE={env['JUDGE_RESULTS_BASE']}")
        print("\n[DRY RUN] Delegating to the namespaced runner in --dry-run "
              "(it makes ZERO API calls and reports its own per-pattern counts):\n")
        # The namespaced runner's own --dry-run resolves the work list and prints
        # per-pattern report counts + a cost estimate, all with ZERO API spend.
        subprocess.run(cmd, env=env, cwd=str(judge_cwd), check=True)
        print(f"\nDRY_RUN_SUMMARY " + json.dumps({
            "n_staged": staged_present,
            "n_pending_gpt52_judge": pending,
            "est_cost_usd": est_cost,
            "judge_model": JUDGE_MODEL,
            "judge_out": str(JUDGE_OUT),
        }))
        return

    # ---- PAID PATH (human-launched only) ----
    print("\n[judge] delegating to namespaced GPT-5.2 runner (real API calls) ...\n")
    subprocess.run(cmd, env=env, cwd=str(judge_cwd), check=True)
    print(f"\n[judge] verdicts written under {JUDGE_OUT}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="DRB-RACE staging + GPT-5.2 judging harness (CORPUS-SAFE)"
    )
    ap.add_argument("--stage", choices=["stage", "judge"], required=True,
                    help="'stage' = write the 400-report tree + manifest (no API); "
                         "'judge' = run the namespaced GPT-5.2 judge.")
    ap.add_argument("--resume", action="store_true",
                    help="(judge) Skip reports already judged under the DRB judge-out.")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="(judge) Max concurrent GPT-5.2 calls (default: 5).")
    ap.add_argument("--limit", type=int, default=0,
                    help="(judge) If >0, judge at most this many still-pending "
                         "reports — for a small proof-of-fix sample. 0 = all.")
    ap.add_argument("--dry-run", action="store_true",
                    help="(judge) ZERO API calls: print what would be judged and exit.")
    args = ap.parse_args()

    if args.stage == "stage":
        do_stage()
    else:
        do_judge(args)


if __name__ == "__main__":
    main()
