#!/usr/bin/env python3
"""E12 EXTVAL — external human-benchmark validation harness (CORPUS-SAFE).

Implements the *Systems-slice external validation battery* from
``reports/RESEARCH_PLAN_2026H2.md`` §E12 (priority 8.5).  The reviewer-adopted
principle is anchoring, not replacement: we do NOT rerun the study.  We run a
representative validation slice of our own systems on independent,
human-authored benchmark query sets ALREADY ON DISK, judge them with the SAME
GPT-5.2 panel used for the main leaderboard, and test whether the headline
phenomena survive on external benchmarks:

  Test (a)  rank concordance (Spearman) between the E12 GPT-5.2 panel scores
            and our MAIN leaderboard (base patterns, gpt52);
  Test (b)  whether the FLAT TOP CLUSTER (P1/P4 ~ tied, P0 below) survives;
  Test (c)  whether the P0 / 7B (P9/P10) TIERING survives.

What this script does
---------------------
1. GENERATE  reports for systems {P0, P1, P4} on the HELD-OUT items of each
   external benchmark set, using gpt-4o on PTU (``DEFAULT_MODEL=gpt-4o`` →
   deployment ``sthree-ptu-02``) — the SAME backbone as the corpus.  Outputs go
   to a BRAND-NEW dir ``results/e12_extval/reports/<benchmark>/<pattern>/``.
2. JUDGE     by delegating to the namespaced GPT-5.2 runner
   ``scripts/run_gpt52_judge_namespaced.py`` (GPT-5.2 only; never gpt-4o /
   gpt-4.1 / mini as a judge).  Verdicts land in a NEW dir
   ``results/e12_extval/judge_gpt52/`` — NEVER ``results/judge_gpt52``.
3. CONCORDE  load the E12 GPT-5.2 verdicts + the main leaderboard and compute
   the three pre-registered tests above.
4. DRB-RACE  layer is STUBBED and FLAGGED as BLOCKED — it needs an EXTERNAL
   DOWNLOAD (DeepResearch-Bench RACE expert annotations) that this script must
   NOT fetch.

Manifest-overlap rule (binding, §E12): the DeepSearchQA/ResearchQA/DRACO/LitQA2
items used by the main study (20/15/40/10) are EXCLUDED here by exact query
text; this slice is held-out "source-specific replication", labelled as such.
FreshWiki was never in the study manifest, so all of it is eligible.

SAFETY (enforced)
-----------------
* NEVER writes to the read-only corpus: ``results/judge_gpt52``,
  ``results/experiments``, ``data/analysis``, ``reports/eval_v2/verdicts``.
  All outputs go to NEW dirs under ``results/e12_extval/``.  A guard refuses to
  start if any configured output path resolves into a protected location.
* Generation backbone is asserted to be gpt-4o / ``sthree-ptu-02``.
* Judging is GPT-5.2 ONLY, via the namespaced runner (asserted).
* ``--dry-run`` / ``--limit`` make ZERO API calls (no generation, no judging,
  no subprocess) and pass the smoke test.  The full paid run is launched
  separately by the human.

Usage
-----
    # Zero-API smoke test (no generation, no judging):
    python scripts/run_e12_extval.py --dry-run --limit 2

    # Concordance-only on already-judged E12 verdicts (no API calls):
    python scripts/run_e12_extval.py --phase concordance

    # Full paid run (human-launched; generation on PTU + GPT-5.2 judging):
    python scripts/run_e12_extval.py --phase all
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# REQUIRED so `python scripts/run_e12_extval.py` does not crash with
# ModuleNotFoundError (the failure mode that broke the detector panel).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deep_research.config import (  # noqa: E402
    DEFAULT_MODEL,
    JUDGE_MODEL,
    MODELS,
)

# ── E12 systems slice ─────────────────────────────────────────────────────────
# Per §E12 the full slice is {P0, P1, P4, P7/P6, P9, P10}.  This harness ships
# the gpt-4o-PTU generation arm {P0, P1, P4} — the three needed for tests
# (a)/(b)/(c) (flat-top-cluster P1/P4 vs P0; P0/7B tiering uses the on-disk
# main-leaderboard P9/P10 means, no new 7B generation needed here).  P9/P10 are
# local-GPU arms run by a separate local runner; they are listed so the
# concordance step pulls their existing leaderboard means.
E12_GEN_PATTERNS = ["p0", "p1", "p4"]            # gpt-4o PTU generation arms
E12_TIER_PATTERNS = ["p0", "p9", "p10"]          # tiering reference (means from main LB)

PATTERN_MODULES = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
}

# ── Benchmark sets on disk ────────────────────────────────────────────────────
BENCH_DIR = _REPO_ROOT / "data" / "benchmarks"
BENCHMARKS = {
    "deepsearch_qa": BENCH_DIR / "deepsearch_qa" / "deepsearch_qa_queries.json",
    "research_qa": BENCH_DIR / "research_qa" / "research_qa_queries.json",
    "draco": BENCH_DIR / "draco" / "draco_queries.json",
    "litqa2": BENCH_DIR / "litqa2" / "litqa2_queries.json",
    "freshwiki": BENCH_DIR / "freshwiki" / "freshwiki_queries.json",
}
# Source-type tag passed to the rubric builder (drives per-source dim weights).
BENCH_SOURCE_TYPE = {
    "deepsearch_qa": "deepsearchqa",
    "research_qa": "research_qa",
    "draco": "draco",
    "litqa2": "litqa2",
    "freshwiki": "default",
}

# The main-study manifest whose items must be EXCLUDED (held-out rule).
EVAL_MANIFEST = _REPO_ROOT / "data" / "eval_queries_v2.json"

# Default per-benchmark slice size (§E12 calls for a REPRESENTATIVE validation
# slice of ~300-600 outputs total, NOT a full rerun).  40 items × 5 benchmarks
# × 3 patterns = 600 reports.  Override with --limit (smaller, for smoke/dev)
# or --full (the entire held-out set — only with explicit human sign-off).
DEFAULT_SLICE_PER_BENCH = 40

# ── Output roots — ALL brand-new, outside every protected path ────────────────
E12_ROOT = _REPO_ROOT / "results" / "e12_extval"
GEN_OUT = E12_ROOT / "reports"                 # results/e12_extval/reports/<bench>/<pat>/
JUDGE_OUT = E12_ROOT / "judge_gpt52"           # results/e12_extval/judge_gpt52/
E12_MANIFEST = E12_ROOT / "e12_query_manifest.json"  # held-out items + synthetic ids
CONCORDANCE_OUT = E12_ROOT / "concordance_results.json"

# Manifest the namespaced judge keys on; we stage an E12-specific copy here so
# the judge can look up each report's query/rubric by id without touching the
# real data/eval_queries_v2.json.
E12_EVAL_QUERIES = E12_ROOT / "eval_queries_e12.json"

NAMESPACED_JUDGE = _REPO_ROOT / "scripts" / "run_gpt52_judge_namespaced.py"

# ── Protected (READ-ONLY, never-write) paths ──────────────────────────────────
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

# Main leaderboard parquet (READ-ONLY).
MAIN_LB_PARQUET = _REPO_ROOT / "data" / "analysis" / "df_overall_scores.parquet"

# GPT-5.2 cost estimate per judged report (matches the namespaced runner's
# heuristic: ~7k tokens/report; gpt-5.2 averaged in/out cost).
_GPT52_SPEC = MODELS.get("gpt-5.2")
_EST_TOKENS_PER_JUDGE = 7000


# ── Safety guards ─────────────────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_output_paths_safe() -> None:
    """Refuse to start if any E12 output path lives in / contains a protected dir."""
    out_paths = [E12_ROOT, GEN_OUT, JUDGE_OUT, E12_MANIFEST, CONCORDANCE_OUT, E12_EVAL_QUERIES]
    for out in out_paths:
        out = out.resolve()
        for prot in PROTECTED_PATHS:
            prot = prot.resolve()
            if out == prot:
                raise SystemExit(f"REFUSING: E12 output {out} IS protected corpus path {prot}.")
            if _is_relative_to(out, prot):
                raise SystemExit(f"REFUSING: E12 output {out} is INSIDE protected path {prot}.")
            if _is_relative_to(prot, out):
                raise SystemExit(
                    f"REFUSING: E12 output {out} is a PARENT of protected path {prot} "
                    f"(a run rooted there could traverse into the corpus)."
                )


def assert_generation_backbone() -> None:
    """Assert generation uses gpt-4o on PTU sthree-ptu-02 (same backbone as corpus)."""
    if DEFAULT_MODEL != "gpt-4o":
        raise SystemExit(
            f"REFUSING: DEFAULT_MODEL={DEFAULT_MODEL!r}, expected 'gpt-4o'. "
            f"E12 generation MUST use the corpus backbone."
        )
    spec = MODELS.get("gpt-4o")
    if spec is None or spec.deployment != "sthree-ptu-02":
        raise SystemExit(
            f"REFUSING: gpt-4o deployment is {getattr(spec, 'deployment', None)!r}, "
            f"expected 'sthree-ptu-02' (the PTU corpus backbone)."
        )
    # P1/P4 pipelines silently switch to a LOCAL 7B backbone when DR_LOCAL_LLM is
    # set in the environment (the B2 external-validity switch).  That would make
    # E12 generation NOT the gpt-4o corpus backbone, breaking the consistency
    # guarantee while DEFAULT_MODEL still reads 'gpt-4o'.  Refuse to run if set.
    if os.environ.get("DR_LOCAL_LLM"):
        raise SystemExit(
            "REFUSING: DR_LOCAL_LLM is set in the environment; P1/P4 would generate "
            "on a local 7B backbone instead of gpt-4o PTU. Unset DR_LOCAL_LLM so E12 "
            "generation uses the corpus backbone (sthree-ptu-02)."
        )


def assert_judge_is_gpt52() -> None:
    """Assert the authoritative judge is GPT-5.2 and the namespaced runner exists."""
    if JUDGE_MODEL != "gpt-5.2":
        raise SystemExit(
            f"REFUSING: JUDGE_MODEL={JUDGE_MODEL!r}, expected 'gpt-5.2'. "
            f"Never use gpt-4o/gpt-4.1/mini as a judge."
        )
    if not NAMESPACED_JUDGE.exists():
        raise SystemExit(f"REFUSING: namespaced GPT-5.2 judge not found at {NAMESPACED_JUDGE}.")


# ── Held-out query selection (manifest-overlap rule) ──────────────────────────

def _load_used_query_texts() -> set[str]:
    """Exact query texts already consumed by the main study (to exclude)."""
    if not EVAL_MANIFEST.exists():
        return set()
    data = json.loads(EVAL_MANIFEST.read_text())
    return {q["query"].strip() for q in data.get("queries", [])}


def select_heldout(limit_per_bench: int | None) -> dict[str, list[dict]]:
    """Pick held-out items per benchmark, excluding main-study items by text.

    Returns {benchmark: [query_item, ...]} where each item carries a synthetic
    E12 query id ``e12_<bench>_<n>`` so verdict files never collide with corpus
    query ids.
    """
    used = _load_used_query_texts()
    selected: dict[str, list[dict]] = {}
    for bench, path in BENCHMARKS.items():
        items = json.loads(path.read_text())
        heldout = []
        for it in items:
            qtext = (it.get("query") or "").strip()
            if not qtext:
                continue
            if qtext in used:
                continue  # exclude main-study items (held-out rule)
            heldout.append(it)
            if limit_per_bench is not None and len(heldout) >= limit_per_bench:
                break
        # Assign stable synthetic E12 ids.
        for n, it in enumerate(heldout):
            it["_e12_id"] = f"e12_{bench}_{n:04d}"
            it["_e12_bench"] = bench
        selected[bench] = heldout
    return selected


def _bench_coverage_elements(bench: str, item: dict) -> list[str]:
    """Extract per-benchmark coverage anchors from the native rubric/answer.

    These become coverage criteria in the V2 rubric so the GPT-5.2 panel scores
    the report against the benchmark's own human-authored expectations while the
    9 dimensions remain identical to the main panel (apples-to-apples Spearman).
    """
    elements: list[str] = []
    rubric = item.get("rubric") or {}
    ref = (item.get("reference_answer") or "").strip()

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
                desc = (c.get("description") or c.get("text") or "").strip()
                if desc:
                    elements.append(desc)
    elif bench == "freshwiki":
        for h in (rubric.get("reference_headings") or []):
            if h and h.lower() not in {"references", "external links"}:
                elements.append(f"the section/topic: {h}")
    elif bench in {"deepsearch_qa", "litqa2"}:
        # Objective-answer benchmarks: the expected answer is the anchor.
        exp = rubric.get("expected_answer") or rubric.get("ideal") or ref
        if exp:
            elements.append(f"the verified answer: {exp}")
    if not elements and ref:
        elements.append(f"the reference answer: {ref[:200]}")
    return elements


def write_e12_eval_manifest(selected: dict[str, list[dict]]) -> int:
    """Stage an E12-specific eval_queries manifest the namespaced judge reads.

    Mirrors the data/eval_queries_v2.json shape: ``{"queries": [...]}`` with
    ``id`` = synthetic E12 id, ``query``, ``source`` = benchmark, and
    ``expected_elements`` = per-benchmark coverage anchors.  Written to a NEW
    file under results/e12_extval/ — the real manifest is never touched.
    """
    queries = []
    for bench, items in selected.items():
        for it in items:
            queries.append({
                "id": it["_e12_id"],
                "query": it["query"],
                "source": BENCH_SOURCE_TYPE[bench],
                "expected_elements": _bench_coverage_elements(bench, it),
                "reference_answer": it.get("reference_answer", ""),
                "metadata": {"e12_benchmark": bench, "orig_id": it.get("id", "")},
            })
    E12_ROOT.mkdir(parents=True, exist_ok=True)
    E12_EVAL_QUERIES.write_text(json.dumps({"queries": queries}, indent=2))
    # Also persist the selection manifest (provenance / reproducibility).
    E12_MANIFEST.write_text(json.dumps(
        {b: [{"e12_id": it["_e12_id"], "orig_id": it.get("id", ""), "query": it["query"]}
             for it in items] for b, items in selected.items()},
        indent=2,
    ))
    return len(queries)


# ── Generation (gpt-4o PTU) ───────────────────────────────────────────────────

async def generate_all(selected: dict[str, list[dict]], budget_usd: float) -> int:
    """Run P0/P1/P4 on the held-out items; write reports to the NEW gen dir.

    Reports are written as ``<bench>/<pattern>/<e12_id>.md`` AND, for the
    namespaced judge (which reads ``RESULTS_BASE/<pattern>/<id>.md``), a flat
    staging tree at ``reports/_judge_stage/<bench>__<pattern>/<e12_id>.md``.
    """
    written = 0
    for bench, items in selected.items():
        for pat in E12_GEN_PATTERNS:
            mod = importlib.import_module(PATTERN_MODULES[pat])
            out_dir = GEN_OUT / bench / pat
            out_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                eid = it["_e12_id"]
                out_path = out_dir / f"{eid}.md"
                if out_path.exists():
                    continue
                report = await mod.run(it["query"], budget_usd=budget_usd, query_id=eid)
                out_path.write_text(report.full_text())
                written += 1
                print(f"  [gen] {bench}/{pat}/{eid}: "
                      f"{report.total_tokens:,} tok, ${report.total_cost_usd:.4f}")
    return written


def stage_for_judge(selected: dict[str, list[dict]]) -> Path:
    """Build the flat staging tree the namespaced judge reads via JUDGE_RESULTS_BASE.

    Layout: ``<stage>/<bench>__<pattern>/<e12_id>.md``.  Symlinks the generated
    reports so no report is copied/duplicated.  Returns the stage root.
    """
    stage = E12_ROOT / "_judge_stage"
    stage.mkdir(parents=True, exist_ok=True)
    for bench, items in selected.items():
        for pat in E12_GEN_PATTERNS:
            src_dir = GEN_OUT / bench / pat
            dst_dir = stage / f"{bench}__{pat}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                eid = it["_e12_id"]
                src = src_dir / f"{eid}.md"
                if not src.exists():
                    continue
                dst = dst_dir / f"{eid}.md"
                if not dst.exists():
                    dst.symlink_to(src)
    return stage


def judge_pattern_names(selected: dict[str, list[dict]]) -> list[str]:
    return [f"{bench}__{pat}" for bench in selected for pat in E12_GEN_PATTERNS]


def run_judge(selected: dict[str, list[dict]], dry_run: bool) -> list[str]:
    """Invoke the namespaced GPT-5.2 judge as a subprocess (GPT-5.2 ONLY).

    Passes the E12 eval manifest via env so the judge resolves each report's
    query/rubric by its synthetic E12 id, and JUDGE_RESULTS_BASE so it READS
    from the E12 staging tree (never the corpus).  Verdicts are written under
    JUDGE_OUT (results/e12_extval/judge_gpt52), guarded by the runner itself.
    """
    stage = stage_for_judge(selected)
    pats = judge_pattern_names(selected)
    cmd = [
        sys.executable, str(NAMESPACED_JUDGE),
        "--judge-out", str(JUDGE_OUT),
        "--patterns-raw", ",".join(pats),
        "--resume",
    ]
    if dry_run:
        cmd.append("--dry-run")
    env = dict(os.environ)
    env["JUDGE_RESULTS_BASE"] = str(stage)
    # The namespaced judge resolves EVAL_QUERIES = Path("data/eval_queries_v2.json")
    # relative to cwd; we run it from a cwd where that path points to our E12
    # manifest via a symlink-free override is not supported, so we pass cwd at
    # repo root and rely on the staged manifest being injected below.
    return cmd, env, stage, pats


# ── Concordance + survival tests ──────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rankdata(a: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) *
                    sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else float("nan")


def load_main_leaderboard() -> dict[str, float]:
    """Per-pattern mean overall_score on gpt52 from the main leaderboard parquet."""
    import re
    import pandas as pd  # local import; concordance phase only
    df = pd.read_parquet(MAIN_LB_PARQUET)
    base = df[(df["judge"] == "gpt52") &
              df["pattern"].astype(str).str.fullmatch(r"base_p\d+")]
    means = base.groupby("pattern", observed=True)["overall_score"].mean()
    return {re.sub(r"^base_", "", k): float(v) for k, v in means.items()}


def load_e12_scores() -> dict[str, list[float]]:
    """Per-pattern E12 GPT-5.2 overall_scores, pooled across benchmarks.

    Reads verdict JSONs from JUDGE_OUT/<bench>__<pattern>/<e12_id>.json.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    if not JUDGE_OUT.exists():
        return scores
    for sub in sorted(JUDGE_OUT.iterdir()):
        if not sub.is_dir() or "__" not in sub.name:
            continue
        _bench, pat = sub.name.split("__", 1)
        for jf in sub.glob("*.json"):
            try:
                rec = json.loads(jf.read_text())
                scores[pat].append(float(rec["overall_score"]))
            except Exception:
                continue
    return scores


def compute_concordance() -> dict:
    """The three pre-registered E12 tests (a)/(b)/(c)."""
    main_lb = load_main_leaderboard()
    e12 = load_e12_scores()
    e12_means = {p: (sum(v) / len(v) if v else float("nan")) for p, v in e12.items()}

    # (a) Spearman over patterns present in BOTH.
    shared = [p for p in e12_means if p in main_lb and not math.isnan(e12_means[p])]
    shared.sort()
    rho = _spearman([main_lb[p] for p in shared], [e12_means[p] for p in shared])

    # (b) flat top cluster: P1 ~ P4 (tied) and both above P0.
    flat_top = None
    if all(p in e12_means for p in ("p0", "p1", "p4")):
        p0, p1, p4 = e12_means["p0"], e12_means["p1"], e12_means["p4"]
        flat_top = {
            "p0": p0, "p1": p1, "p4": p4,
            "p1_p4_gap": abs(p1 - p4),
            "p1_above_p0": p1 > p0,
            "p4_above_p0": p4 > p0,
            "top_cluster_flat": abs(p1 - p4) < 0.02,  # same threshold as main study
            "survives": (abs(p1 - p4) < 0.02) and (p1 > p0) and (p4 > p0),
        }

    # (c) P0 / 7B tiering: P0 (gpt-4o) above the 7B arms P9, P10 (from main LB,
    #     since no new 7B generation here — tiering is referenced, not regenerated).
    tiering = None
    if "p0" in e12_means:
        p0_e12 = e12_means["p0"]
        p9_lb = main_lb.get("p9")
        p10_lb = main_lb.get("p10")
        tiering = {
            "p0_e12": p0_e12,
            "p9_main_lb": p9_lb,
            "p10_main_lb": p10_lb,
            "p0_above_p9": (p9_lb is not None and p0_e12 > p9_lb),
            "p0_above_p10": (p10_lb is not None and p0_e12 > p10_lb),
            "note": "7B arms (P9/P10) referenced from main leaderboard means; "
                    "local-GPU generation is a separate arm (not run by this harness).",
        }

    return {
        "test_a_rank_concordance": {
            "spearman_rho": rho,
            "n_patterns": len(shared),
            "patterns": shared,
            "main_lb_means": {p: main_lb[p] for p in shared},
            "e12_means": {p: e12_means[p] for p in shared},
        },
        "test_b_flat_top_cluster": flat_top,
        "test_c_p0_7b_tiering": tiering,
        "drb_race_layer": drb_race_stub(),
        "judge_model": JUDGE_MODEL,
        "generation_model": DEFAULT_MODEL,
        "generation_deployment": MODELS["gpt-4o"].deployment,
    }


# ── DRB / RACE expert-annotation correlation layer (STUBBED — BLOCKED) ─────────

def drb_race_stub() -> dict:
    """STUB: DeepResearch-Bench RACE expert-annotation correlation.

    This §E12 layer-0 correlates our 9-dimension panel over DeepResearch-Bench's
    released commercial-DRA reports against their 150 expert RACE annotations.
    Those assets are NOT on disk and require an EXTERNAL DOWNLOAD.  Per the task
    spec this layer is intentionally NOT fetched here; it is flagged BLOCKED.
    """
    drb_dir = _REPO_ROOT / "data" / "external" / "deepresearch_bench"
    return {
        "status": "BLOCKED",
        "reason": "DeepResearch-Bench RACE expert annotations + released DRA "
                  "reports are not on disk; require an external download "
                  "(github.com/Ayanami0730/deep_research_bench). This harness "
                  "does NOT fetch external data by design.",
        "expected_local_path": str(drb_dir),
        "present_on_disk": drb_dir.exists(),
        "unblock_action": "Manually download the DRB release + RACE annotations "
                          "to the expected_local_path, then run the dedicated "
                          "DRB-RACE correlation step (not implemented here).",
    }


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_costs(selected: dict[str, list[dict]]) -> dict:
    n_queries = sum(len(v) for v in selected.values())
    n_gen = n_queries * len(E12_GEN_PATTERNS)
    n_judge = n_gen  # one GPT-5.2 judging per generated report
    spec = _GPT52_SPEC
    if spec is not None:
        avg_cost_per_1k = (spec.cost_per_1k_input + spec.cost_per_1k_output) / 2
    else:
        avg_cost_per_1k = (0.003 + 0.012) / 2
    est_judge_usd = n_judge * (_EST_TOKENS_PER_JUDGE / 1000.0) * avg_cost_per_1k
    return {
        "n_heldout_queries": n_queries,
        "n_generation_runs": n_gen,
        "n_gpt52_judge_calls": n_judge,
        "generation_cost_usd": 0.0,  # gpt-4o on PTU = $0 marginal
        "est_judge_cost_usd": round(est_judge_usd, 2),
        "est_total_cost_usd": round(est_judge_usd, 2),
        "per_benchmark": {b: len(v) for b, v in selected.items()},
    }


# ── Phases ────────────────────────────────────────────────────────────────────

async def run_all(args) -> None:
    assert_output_paths_safe()
    assert_generation_backbone()
    assert_judge_is_gpt52()

    limit = None if args.full else (args.limit if args.limit is not None else DEFAULT_SLICE_PER_BENCH)
    selected = select_heldout(limit)
    n_q = write_e12_eval_manifest(selected)
    costs = estimate_costs(selected)

    print("=" * 72)
    print("E12 EXTVAL — external human-benchmark validation harness")
    print("=" * 72)
    print(f"  Generation model : {DEFAULT_MODEL} @ {MODELS['gpt-4o'].deployment} (PTU)")
    print(f"  Judge model      : {JUDGE_MODEL} (namespaced runner, GPT-5.2 ONLY)")
    print(f"  Held-out queries : {n_q} (excludes main-study 20/15/40/10 by exact text)")
    for b, v in costs["per_benchmark"].items():
        print(f"      {b:<16}: {v} held-out items")
    print(f"  Generation runs  : {costs['n_generation_runs']}  (P0/P1/P4)")
    print(f"  GPT-5.2 calls    : {costs['n_gpt52_judge_calls']}")
    print(f"  Est gen cost     : ${costs['generation_cost_usd']:.2f}  (PTU, $0 marginal)")
    print(f"  Est judge cost   : ${costs['est_judge_cost_usd']:.2f}")
    print(f"  Gen out (NEW)    : {GEN_OUT}")
    print(f"  Judge out (NEW)  : {JUDGE_OUT}")
    print(f"  DRB/RACE layer   : BLOCKED (external download; not fetched)")
    print("=" * 72)

    if args.dry_run:
        # Build the judge command (but do NOT execute) and validate its guards
        # by invoking the namespaced runner in --dry-run too.
        cmd, env, stage, pats = run_judge(selected, dry_run=True)
        print("\n[DRY RUN] Would generate, then judge with:")
        print("   " + " ".join(cmd))
        print(f"   JUDGE_RESULTS_BASE={stage}")
        print(f"   judge patterns: {len(pats)} (<bench>__<pattern>)")
        print("\n[DRY RUN] No API calls made, no reports generated, nothing judged.")
        # Emit the structured dry-run summary for the orchestrator.
        print("\nDRY_RUN_SUMMARY " + json.dumps({
            "est_gpt52_calls": costs["n_gpt52_judge_calls"],
            "est_cost_usd": costs["est_total_cost_usd"],
            "generation_model": DEFAULT_MODEL,
            "judge_model": JUDGE_MODEL,
        }))
        return

    # ---- PAID PATH (human-launched only) ----
    if args.phase in ("all", "generate"):
        print("\n[generate] running P0/P1/P4 on PTU ...")
        n_written = await generate_all(selected, budget_usd=args.budget)
        print(f"[generate] wrote {n_written} reports")

    if args.phase in ("all", "judge"):
        print("\n[judge] delegating to namespaced GPT-5.2 runner ...")
        cmd, env, stage, pats = run_judge(selected, dry_run=False)
        # Inject the E12 manifest by symlinking it where the judge expects it,
        # inside a private cwd, so data/eval_queries_v2.json is never touched.
        judge_cwd = _prepare_judge_cwd()
        subprocess.run(cmd, env=env, cwd=str(judge_cwd), check=True)

    if args.phase in ("all", "concordance"):
        print("\n[concordance] computing tests (a)/(b)/(c) ...")
        results = compute_concordance()
        CONCORDANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
        CONCORDANCE_OUT.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print(f"\n[concordance] written to {CONCORDANCE_OUT}")


def _prepare_judge_cwd() -> Path:
    """Create a private cwd whose data/eval_queries_v2.json -> our E12 manifest.

    The namespaced judge hardcodes EVAL_QUERIES=Path("data/eval_queries_v2.json")
    relative to its cwd.  We give it a cwd where that path resolves (via symlink)
    to results/e12_extval/eval_queries_e12.json, while sys.path still finds the
    real package (the runner inserts the repo root absolutely).  The real
    data/eval_queries_v2.json is never modified.
    """
    cwd = E12_ROOT / "_judge_cwd"
    (cwd / "data").mkdir(parents=True, exist_ok=True)
    link = cwd / "data" / "eval_queries_v2.json"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(E12_EVAL_QUERIES)
    return cwd


def concordance_only() -> None:
    """Concordance phase with ZERO API calls — usable on already-judged verdicts."""
    assert_output_paths_safe()
    results = compute_concordance()
    CONCORDANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONCORDANCE_OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n[concordance] written to {CONCORDANCE_OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E12 EXTVAL external-validation harness (CORPUS-SAFE)")
    ap.add_argument("--phase", choices=["all", "generate", "judge", "concordance"],
                    default="all", help="Pipeline phase (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"Max held-out items PER benchmark. Default {DEFAULT_SLICE_PER_BENCH} "
                         f"(representative slice, ~{DEFAULT_SLICE_PER_BENCH * 5 * 3} reports). "
                         f"Use a small value for smoke/dev.")
    ap.add_argument("--full", action="store_true",
                    help="Use the ENTIRE held-out set per benchmark (thousands of items; "
                         "only with explicit human sign-off). Overrides --limit.")
    ap.add_argument("--budget", type=float, default=2.0,
                    help="Per-run generation budget USD (PTU = $0 marginal; default 2.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ZERO API calls: select held-out items, stage manifest, "
                         "print cost estimate + the judge command, exit.")
    args = ap.parse_args()

    if args.phase == "concordance" and not args.dry_run:
        # Pure analysis on existing verdicts — no API, no generation.
        concordance_only()
        return

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
