#!/usr/bin/env python3
"""Local-7B-benchmark staging + GPT-5.2 judging harness — CORPUS-SAFE.

Stages the 150 local 7B benchmark reports (P9 Qwen2.5-7B-Instruct, P10
GAIR/DeepResearcher-7b across {draco, litqa2, research_qa, deepsearch_qa,
freshwiki}, 15 each) into a read-only tree + a synthetic eval manifest, then
delegates to the corpus-safe namespaced GPT-5.2 judge so the head-to-head
local-model leaderboard cell gets the SAME 9-dimension GPT-5.2 panel as every
other arm of the study.

Structure mirrors ``scripts/stage_and_judge_drbrace.py`` exactly:
  * ``--stage stage`` writes a read-only staging tree + a synthetic manifest;
  * ``--stage judge`` delegates to ``scripts/run_gpt52_judge_namespaced.py``
    with ``JUDGE_RESULTS_BASE`` pointed at the staging tree and a PRIVATE cwd
    whose ``data/eval_queries_v2.json`` symlinks to our manifest;
  * ``--dry-run`` makes ZERO API calls.

Data on disk (read-only inputs)
-------------------------------
* ``results/local_benchmark/<pattern>_<benchmark>/<query_id>.md`` — 150 reports,
  pattern in {p9, p10}, benchmark in {draco, litqa2, research_qa,
  deepsearch_qa, freshwiki} (15 each). The ``<query_id>`` stem is the benchmark
  task's ``id``. ~54 of the 150 are thin (<2KB, source-starved) — they are
  staged and judged anyway: a low score is a valid leaderboard cell.
* ``data/benchmarks/<benchmark>/<benchmark>_queries.json`` — list of task rows
  ``{id, query, domain, difficulty, rubric, reference_answer,
  expected_citations, metadata}``; the report stem keys into ``id``.

Staged judge patterns
----------------------
Each (pattern, benchmark) is namespaced as ``lb_<pattern>_<benchmark>`` (e.g.
``lb_p9_draco``), so the staging subdir + verdict subdir are unique and never
collide with corpus query ids or pattern names. 2 patterns x 5 benchmarks = 10
staged patterns, 150 reports total.

source_type / source-aware weights
-----------------------------------
The synthetic manifest tags each query with the rubric_v2 source-weight key for
its benchmark (``draco`` -> draco, ``litqa2`` -> litqa2, ``research_qa`` ->
researchqa, ``deepsearch_qa`` -> deepsearchqa, ``freshwiki`` -> default) so the
source-aware dimension weights in ``DIMENSION_WEIGHTS_BY_SOURCE`` genuinely
apply, matching how run_e12_extval / drbrace tag their queries. (The literal
benchmark names ``research_qa`` / ``deepsearch_qa`` are NOT registry keys and
silently fall back to ``default``; we map to the real keys so the source profile
is actually used.) Per-benchmark coverage anchors are pulled from each task's
native rubric/reference answer into V2 ``coverage`` criteria, while the 9
dimensions stay identical to the main panel (apples-to-apples).

Corpus safety (enforced)
------------------------
Writes ONLY under ``results/local_benchmark/_judge_stage`` and
``results/local_benchmark/judge_gpt52``. A guard refuses to start if any output
path resolves into a protected corpus path (``results/judge_gpt52``,
``results/experiments``, ``data/analysis``, ``reports/eval_v2/verdicts``). The
namespaced judge applies the SAME guard independently.

Usage
-----
    # Stage the 150 reports + manifest (no API):
    python scripts/stage_and_judge_local_benchmark.py --stage stage

    # See exactly what the judge would grade (ZERO API spend):
    python scripts/stage_and_judge_local_benchmark.py --stage judge --dry-run

    # Real paid GPT-5.2 judging (human-launched), corpus-safe + resumable:
    python scripts/stage_and_judge_local_benchmark.py --stage judge --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# REQUIRED so `python scripts/stage_and_judge_local_benchmark.py` does not crash
# with ModuleNotFoundError (the failure mode that broke the detector panel).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deep_research.config import JUDGE_MODEL, MODELS  # noqa: E402

# ── Local-benchmark inputs (READ-ONLY) ────────────────────────────────────────
LB_REPORTS_ROOT = _REPO_ROOT / "results" / "local_benchmark"
BENCH_DIR = _REPO_ROOT / "data" / "benchmarks"

# Local 7B arms whose reports we stage + judge.
LB_PATTERNS = ["p9", "p10"]

# Benchmarks, in fixed order. 15 reports per (pattern, benchmark) on disk.
LB_BENCHMARKS = ["draco", "litqa2", "research_qa", "deepsearch_qa", "freshwiki"]

# Per-benchmark query file (READ-ONLY); the report stem keys into each row's id.
def bench_query_path(bench: str) -> Path:
    return BENCH_DIR / bench / f"{bench}_queries.json"

# Source-type tag handed to the rubric builder — MUST be a registry key in
# rubric_v2.DIMENSION_WEIGHTS_BY_SOURCE for the source-aware weights to apply.
# The literal benchmark names research_qa / deepsearch_qa are NOT registry keys
# (they fall back to 'default'); we map to the real keys so the source profile
# is genuinely used. freshwiki has no profile -> 'default' (same as run_e12).
BENCH_SOURCE_TYPE = {
    "draco": "draco",
    "litqa2": "litqa2",
    "research_qa": "researchqa",
    "deepsearch_qa": "deepsearchqa",
    "freshwiki": "default",
}

# ── Output roots — ALL brand-new, outside every protected path ────────────────
LB_ROOT = LB_REPORTS_ROOT                                      # results/local_benchmark
STAGE_ROOT = LB_ROOT / "_judge_stage"                          # lb_<pat>_<bench>/<qid>.md
JUDGE_OUT = LB_ROOT / "judge_gpt52"                            # verdicts (NEW dir, NOT corpus)
# Synthetic eval manifest the namespaced judge keys on (shape mirrors
# data/eval_queries_v2.json: {"queries": [...]}).
LB_EVAL_QUERIES = LB_ROOT / "eval_queries_local_benchmark.json"
# Provenance: the staged (pattern, benchmark, query_id) selection manifest.
LB_STAGE_MANIFEST = LB_ROOT / "local_benchmark_stage_manifest.json"
# Private cwd whose data/eval_queries_v2.json -> our LB manifest (judge reads it).
JUDGE_CWD = LB_ROOT / "_judge_cwd"

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


def pattern_for(pat: str, bench: str) -> str:
    """Staged judge pattern name for a (local pattern, benchmark) pair.

    Namespaced ``lb_<pattern>_<benchmark>`` so the staging/verdict subdir is
    unique and never collides with corpus pattern names or query ids.
    """
    return f"lb_{pat}_{bench}"


# ── Safety guards ─────────────────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_output_paths_safe() -> None:
    """Refuse to start if any output path lives in / contains a protected dir."""
    out_paths = [
        LB_ROOT, STAGE_ROOT, JUDGE_OUT, LB_EVAL_QUERIES,
        LB_STAGE_MANIFEST, JUDGE_CWD,
    ]
    for out in out_paths:
        out = out.resolve()
        for prot in PROTECTED_PATHS:
            prot = prot.resolve()
            if out == prot:
                raise SystemExit(f"REFUSING: local-benchmark output {out} IS protected corpus path {prot}.")
            if _is_relative_to(out, prot):
                raise SystemExit(f"REFUSING: local-benchmark output {out} is INSIDE protected path {prot}.")
            if _is_relative_to(prot, out):
                raise SystemExit(
                    f"REFUSING: local-benchmark output {out} is a PARENT of protected path {prot} "
                    f"(a run rooted there could traverse into the corpus)."
                )


def assert_judge_is_gpt52() -> None:
    """Assert the authoritative judge is GPT-5.2 and the namespaced runner exists."""
    if JUDGE_MODEL != "gpt-5.2":
        raise SystemExit(
            f"REFUSING: JUDGE_MODEL={JUDGE_MODEL!r}, expected 'gpt-5.2'. "
            f"Never use gpt-4o/gpt-4.1/mini/local as a judge here "
            f"(a local 7B is P9's own base model — judge independence)."
        )
    if not NAMESPACED_JUDGE.exists():
        raise SystemExit(f"REFUSING: namespaced GPT-5.2 judge not found at {NAMESPACED_JUDGE}.")


def assert_inputs_present() -> None:
    if not LB_REPORTS_ROOT.exists():
        raise SystemExit(f"REFUSING: local-benchmark report root missing: {LB_REPORTS_ROOT}")
    for bench in LB_BENCHMARKS:
        p = bench_query_path(bench)
        if not p.exists():
            raise SystemExit(f"REFUSING: required benchmark query file missing: {p}")


# ── Load local-benchmark inputs ───────────────────────────────────────────────

def load_bench_tasks(bench: str) -> dict[str, dict]:
    """Map a benchmark's task id -> the full task row (query + rubric + ref)."""
    rows = json.loads(bench_query_path(bench).read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows if "id" in r}


def iter_reports() -> list[dict]:
    """Flatten the local-benchmark report tree into staged-report records.

    Yields one dict per (pattern, benchmark, report) with keys:
      pattern, benchmark, judge_pattern (lb_<pat>_<bench>), query_id,
      query (task prompt), task (full benchmark task row), src (report path),
      report_chars.
    """
    out: list[dict] = []
    for bench in LB_BENCHMARKS:
        tasks = load_bench_tasks(bench)
        for pat in LB_PATTERNS:
            src_dir = LB_REPORTS_ROOT / f"{pat}_{bench}"
            if not src_dir.exists():
                raise SystemExit(f"REFUSING: report dir not present: {src_dir}")
            for src in sorted(src_dir.glob("*.md")):
                qid = src.stem
                task = tasks.get(qid)
                if task is None:
                    # Stem must resolve to a benchmark task; otherwise the judge
                    # has no prompt/rubric for it. Surface, do not silently drop.
                    raise SystemExit(
                        f"REFUSING: report {src} has no matching task id {qid!r} "
                        f"in {bench_query_path(bench)}."
                    )
                out.append({
                    "pattern": pat,
                    "benchmark": bench,
                    "judge_pattern": pattern_for(pat, bench),
                    "query_id": qid,
                    "query": (task.get("query") or "").strip(),
                    "task": task,
                    "src": src,
                    "report_chars": src.stat().st_size,
                })
    return out


# ── Per-benchmark coverage anchors (mirrors run_e12_extval) ────────────────────

def bench_coverage_elements(bench: str, task: dict) -> list[str]:
    """Extract per-benchmark coverage anchors from the native rubric/answer.

    These become ``coverage`` criteria in the V2 rubric so the GPT-5.2 panel
    scores the report against the benchmark's own human-authored expectations,
    while the 9 dimensions stay identical to the main panel.
    """
    elements: list[str] = []
    rubric = task.get("rubric") or {}
    ref = (task.get("reference_answer") or "").strip()

    if bench == "research_qa":
        for c in (rubric.get("criteria") or [])[:12]:
            q = (c.get("question") or "").strip()
            if q:
                elements.append(q)
    elif bench == "draco":
        for section, crits in rubric.items():
            if not isinstance(crits, list):
                continue
            for c in crits[:8]:
                if not isinstance(c, dict):
                    continue
                desc = (c.get("description") or c.get("text") or "").strip()
                if desc:
                    elements.append(desc)
    elif bench == "freshwiki":
        for h in (rubric.get("reference_headings") or []):
            if h and h.lower() not in {"references", "external links"}:
                elements.append(f"the section/topic: {h}")
    elif bench in {"deepsearch_qa", "litqa2"}:
        # Objective-answer benchmarks: the verified answer is the anchor.
        exp = rubric.get("expected_answer") or rubric.get("ideal") or ref
        if exp:
            elements.append(f"the verified answer: {exp}")
    if not elements and ref:
        elements.append(f"the reference answer: {ref[:200]}")
    return elements


# ── Staging ───────────────────────────────────────────────────────────────────

def write_eval_manifest(records: list[dict]) -> int:
    """Stage a local-benchmark eval_queries manifest the namespaced judge reads.

    Mirrors data/eval_queries_v2.json shape: ``{"queries": [...]}``. The judge
    keys each report on its ``query_id`` stem, so there is one manifest entry per
    UNIQUE query_id. The same benchmark task id can appear for both P9 and P10
    (identical prompt/rubric), so we de-dup by query_id. Written to a NEW file
    under results/local_benchmark/ — the real manifest is untouched.
    """
    seen: dict[str, dict] = {}
    for rec in records:
        qid = rec["query_id"]
        if qid in seen:
            continue
        bench = rec["benchmark"]
        seen[qid] = {
            "id": qid,
            "query": rec["query"],
            # Registry key so source-aware dimension weights actually apply.
            "source": BENCH_SOURCE_TYPE[bench],
            "expected_elements": bench_coverage_elements(bench, rec["task"]),
            "reference_answer": (rec["task"].get("reference_answer") or ""),
            "metadata": {"benchmark": bench, "orig_id": qid},
        }
    LB_ROOT.mkdir(parents=True, exist_ok=True)
    LB_EVAL_QUERIES.write_text(
        json.dumps({"queries": list(seen.values())}, indent=2),
        encoding="utf-8",
    )
    return len(seen)


def stage_reports(records: list[dict]) -> tuple[int, int]:
    """Symlink each report to ``STAGE_ROOT/lb_<pat>_<bench>/<query_id>.md``.

    Returns (n_linked, n_existing). Symlinks (not copies) so no report is
    duplicated; the source tree stays read-only. Idempotent.
    """
    linked = 0
    existing = 0
    for rec in records:
        dst_dir = STAGE_ROOT / rec["judge_pattern"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{rec['query_id']}.md"
        if dst.exists() or dst.is_symlink():
            existing += 1
            continue
        dst.symlink_to(rec["src"].resolve())
        linked += 1
    return linked, existing


def write_stage_provenance(records: list[dict]) -> None:
    """Persist the staged selection for reproducibility."""
    manifest = {
        "benchmark_family": "local_7b_benchmark",
        "patterns": LB_PATTERNS,
        "benchmarks": LB_BENCHMARKS,
        "source_types": BENCH_SOURCE_TYPE,
        "n_reports": len(records),
        "judge_model": JUDGE_MODEL,
        "judge_patterns": sorted({r["judge_pattern"] for r in records}),
        "reports": [
            {
                "judge_pattern": r["judge_pattern"],
                "pattern": r["pattern"],
                "benchmark": r["benchmark"],
                "query_id": r["query_id"],
                "report_bytes": r["report_chars"],
                "thin": r["report_chars"] < 2048,
            }
            for r in records
        ],
    }
    LB_STAGE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def do_stage() -> list[dict]:
    """Stage stage: write the 150-report tree + the synthetic manifest."""
    assert_output_paths_safe()
    assert_inputs_present()

    records = iter_reports()
    n_manifest = write_eval_manifest(records)
    n_linked, n_existing = stage_reports(records)
    write_stage_provenance(records)

    patterns = sorted({r["judge_pattern"] for r in records})
    no_prompt = [r for r in records if not r["query"].strip()]
    thin = [r for r in records if r["report_chars"] < 2048]

    print("=" * 72)
    print("Local-7B-benchmark staging (CORPUS-SAFE)")
    print("=" * 72)
    print(f"  Reports staged    : {len(records)} ({n_linked} linked, {n_existing} unchanged)")
    print(f"  Judge patterns    : {len(patterns)}")
    for pat in patterns:
        n = sum(1 for r in records if r["judge_pattern"] == pat)
        print(f"      {pat:<24}: {n} reports")
    print(f"  Manifest entries  : {n_manifest} unique query ids  ->  {LB_EVAL_QUERIES.name}")
    print(f"  Thin (<2KB)       : {len(thin)} reports (staged + judged anyway; low score is valid)")
    print(f"  Stage tree (NEW)  : {STAGE_ROOT}")
    print(f"  Provenance        : {LB_STAGE_MANIFEST.name}")
    if no_prompt:
        print(f"  WARNING: {len(no_prompt)} reports have NO resolvable prompt "
              f"(e.g. {no_prompt[0]['judge_pattern']}/{no_prompt[0]['query_id']}).")
    print("=" * 72)
    return records


# ── Judging (delegate to the namespaced GPT-5.2 runner) ───────────────────────

def _prepare_judge_cwd() -> Path:
    """Private cwd whose data/eval_queries_v2.json -> our LB manifest.

    The namespaced judge hardcodes ``EVAL_QUERIES=Path("data/eval_queries_v2.json")``
    relative to its cwd. We give it a cwd where that path resolves (via symlink)
    to results/local_benchmark/eval_queries_local_benchmark.json, while the
    runner still imports the real package (it inserts the repo root absolutely on
    sys.path). The real data/eval_queries_v2.json is never modified.
    """
    (JUDGE_CWD / "data").mkdir(parents=True, exist_ok=True)
    link = JUDGE_CWD / "data" / "eval_queries_v2.json"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(LB_EVAL_QUERIES)
    return JUDGE_CWD


def build_judge_cmd(records: list[dict], dry_run: bool, resume: bool,
                    concurrency: int, limit: int = 0) -> tuple[list[str], dict, list[str]]:
    """Build the namespaced GPT-5.2 judge command + env (no execution here)."""
    patterns = sorted({r["judge_pattern"] for r in records})
    cmd = [
        # ``-u`` forces the CHILD to run with unbuffered stdout/stderr. Without
        # it, when this harness's stdout is a pipe/redirect (the agent harness,
        # or `> log 2>&1`), the child block-buffers its per-report progress and
        # NOTHING is flushed until exit — and the buffer is lost outright if a
        # foreground timeout kills the run, which looks exactly like a silent
        # "zero verdicts, empty log" failure even while verdicts are written.
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
        pat = rec["judge_pattern"]
        staged = STAGE_ROOT / pat / f"{rec['query_id']}.md"
        if not (staged.exists() or staged.is_symlink()):
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
    # timeout-kill). See build_judge_cmd for the child-side ``-u`` rationale.
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
        if (STAGE_ROOT / r["judge_pattern"] / f"{r['query_id']}.md").exists()
    )
    if staged_present == 0:
        raise SystemExit(
            "REFUSING: no staged reports found. Run "
            "`python scripts/stage_and_judge_local_benchmark.py --stage stage` first."
        )
    if not LB_EVAL_QUERIES.exists():
        raise SystemExit(
            f"REFUSING: staged manifest missing ({LB_EVAL_QUERIES}). Run --stage stage first."
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
    print("Local-7B-benchmark judging — namespaced GPT-5.2 (CORPUS-SAFE)")
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
        print("\nDRY_RUN_SUMMARY " + json.dumps({
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
        description="Local-7B-benchmark staging + GPT-5.2 judging harness (CORPUS-SAFE)"
    )
    ap.add_argument("--stage", choices=["stage", "judge"], required=True,
                    help="'stage' = write the 150-report tree + manifest (no API); "
                         "'judge' = run the namespaced GPT-5.2 judge.")
    ap.add_argument("--resume", action="store_true",
                    help="(judge) Skip reports already judged under the LB judge-out.")
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
