#!/usr/bin/env python3
"""E9 SCALE-CURVE — orchestration returns vs backbone capability (generation harness).

The first empirical orchestration scaling law (RESEARCH_PLAN_2026H2 §TIER 3, E9).
Holds the *architecture* fixed and sweeps the *backbone* — the ONE allowed
exception to the otherwise-fixed gpt-4o generation backbone of the 248k corpus.

What this script is
-------------------
The GENERATION harness only. It produces fresh research reports across a
capability ladder of backbones x architectures x queries and writes them to a
BRAND-NEW results tree. It does NOT judge — authoritative judging is GPT-5.2 via
the corpus-safe namespaced runner ``scripts/run_gpt52_judge_namespaced.py``,
which this script prints the exact invocation for (and never imports/runs).

Independent variable: BACKBONE
------------------------------
Tiers (capability ladder, all PTU/Azure here):

    gpt-4o-mini  ->  gpt-4.1  ->  gpt-4o

``gpt-4o`` is included as the CORPUS ANCHOR: it is the same backbone
(``DEFAULT_MODEL=gpt-4o`` on PTU deployment ``sthree-ptu-02``) as the existing
248k-report corpus, so its tier is directly comparable to the published numbers
(re-generated here under identical scaffold for a clean within-harness curve).

The local 14B tier (Qwen2.5-14B) specified in the plan WAS skipped in the
original .pyc as GPU-blocked (4-bit transformers OOMs on the 16 GB RTX 5080).
That is now STALE: the llama.cpp GGUF Q4_K_M path
(``scripts/run_e8_vintage_14b_gen.py``, pattern ``p17_scale_qwen25_14b``,
~76 tok/s strict-greedy) runs the 14B on this exact card. The 14B local
capability point is therefore RECOVERED, not blocked — but it is generated on
the GPU QUEUE via that existing E8b runner (one model in 16 GB VRAM at a time),
NOT by this PTU/Azure harness. This harness still logs the local-tier gap
loudly and points at the GPU-queue step that fills it, so the full capability
ladder (gpt-4o-mini -> gpt-4.1 -> gpt-4o + 7B + E8 9B + 14B local) is restored.

How the backbone becomes the IV
-------------------------------
Every pattern binds ``DEFAULT_MODEL`` at import time
(``from deep_research.config import DEFAULT_MODEL``) and threads it into every
``llm.complete(..., model=DEFAULT_MODEL)`` call. So switching the backbone is
done by setting ``os.environ["DEFAULT_MODEL"]`` BEFORE importing config or any
pattern module, then PURGING ``deep_research.config`` / ``deep_research.patterns.*``
from ``sys.modules`` so the backbone re-binds, then lazily
``importlib.import_module``-ing the pattern. We then HARD-ASSERT the imported
module's bound ``DEFAULT_MODEL == tier`` (a "BACKBONE MISMATCH" abort) to protect
comparability. ``SEARCH_MODEL`` is left pinned at its corpus value
(gpt-4o-mini) and is NEVER touched.

CONSISTENCY GUARANTEES (hard requirements)
------------------------------------------
* Generation backbone is gpt-4o EXCEPT in this E9 sweep, where backbone is the
  IV; the gpt-4o tier is always available as the corpus-matched anchor.
* SEARCH_MODEL stays gpt-4o-mini (the corpus value) on every tier.
* Authoritative judging is ALWAYS GPT-5.2 via the namespaced corpus-safe runner;
  this script never wires gpt-4o/gpt-4.1/any other model as the authoritative
  judge. (gpt-4.1 may serve only as a non-authoritative cross-check downstream,
  and never on its own backbone arm — but that is the judge runner's concern,
  not this generation harness.)

SAFETY (never touches the irreplaceable corpus)
-----------------------------------------------
* Writes ONLY to NEW top-level dirs (default ``results/experiments_e9_scale``
  and ``checkpoints/e9_scale``). ``resolve_safe_out`` REFUSES any output root
  that resolves to / inside / above the protected paths:
      results/judge_gpt52, results/experiments, data/analysis,
      reports/eval_v2/verdicts
* ``--link-for-judging`` creates READ-ONLY symlink aliases of each E9 cell under
  ``results/experiments/<cell>`` so the namespaced judge can read them; the
  symlinks point OUT to the E9 root and the judge writes ONLY to --judge-out.
  (You can also just judge in place with
  ``JUDGE_RESULTS_BASE=results/experiments_e9_scale``.)
* ``--dry-run`` (and ``--limit`` with no work) make ZERO API calls and write
  nothing — the smoke test runs this.
* This script BUILDS and VERIFIES generation only; the paid run is launched
  separately by the human.

Usage
-----
    # smoke test — zero API spend, nothing written
    python scripts/run_e9_scale_curve.py --dry-run

    # tiny live slice (3 queries) — paid; launched by human only
    python scripts/run_e9_scale_curve.py --limit 3 --tiers gpt-4o-mini --arches p0

    # full curve (anchor + ladder), best-of-N=4, all 90 queries — paid
    python scripts/run_e9_scale_curve.py --best-of-n 4

Then judge (GPT-5.2, corpus-safe) — printed by --dry-run:
    JUDGE_RESULTS_BASE=results/experiments_e9_scale \\
      python scripts/run_gpt52_judge_namespaced.py \\
        --judge-out results/judge_gpt52_e9_scale \\
        --patterns-raw <comma-separated e9 dir names> --resume
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo root on sys.path so `deep_research...` imports resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── IV: the capability ladder (recovered defaults) ───────────────────────────
TIERS_DEFAULT = ["gpt-4o-mini", "gpt-4.1", "gpt-4o"]   # low -> high; all PTU/Azure
ALLOWED_BACKBONES = {"gpt-4o-mini", "gpt-4.1", "gpt-4o"}
ANCHOR_TIER = "gpt-4o"  # == DEFAULT_MODEL on PTU sthree-ptu-02 == the 248k corpus backbone

# The local 14B tier: in the ORIGINAL .pyc this was SKIPPED (GPU-blocked). It is
# now RECOVERED via the GGUF/llama.cpp path on the GPU queue (run_e8_vintage_14b_gen.py,
# pattern p17_scale_qwen25_14b). This harness does NOT generate it (it is a GPU
# job, one model in VRAM at a time); it only logs the tier + where it is filled.
SKIPPED_LOCAL_TIER = "Qwen2.5-14B-4bit-local"
SKIPPED_LOCAL_REASON = (
    "Original .pyc skipped 4-bit transformers (OOM on RTX 5080 16GB at corpus "
    "context). NOW RECOVERED via GGUF Q4_K_M / llama.cpp on the GPU queue: run "
    "`python scripts/run_e8_vintage_14b_gen.py` (pattern p17_scale_qwen25_14b, "
    "strict-greedy, n=90). This PTU/Azure harness does NOT generate the local "
    "14B point — it is a sequential GPU job."
)

# ── Arches (held FIXED across the sweep) ─────────────────────────────────────
PATTERNS = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3": "deep_research.patterns.p3_meridian.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6": "deep_research.patterns.p6_reactive_interleaved.pipeline",
    "p7": "deep_research.patterns.p7_graph_decomposition.pipeline",
    "p8": "deep_research.patterns.p8_beam_search.pipeline",
}
# Plan-swept arches: P0, P1, P4 (+ best-of-N realised by --best-of-n over P0/cluster).
ARCHES_DEFAULT = ["p0", "p1", "p4"]

# ── Corpus-safe output roots (NEW dirs only) ─────────────────────────────────
DEFAULT_RESULTS_ROOT = "results/experiments_e9_scale"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/e9_scale"
JUDGE_OUT_ROOT = "results/judge_gpt52_e9_scale"

# Paths this harness must NEVER write to / inside / above.
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"

# ── Cost constants (recovered floats) ────────────────────────────────────────
TIER_COST_PER_REPORT_USD = {
    "gpt-4o-mini": 0.01,
    "gpt-4.1": 0.08,
    "gpt-4o": 0.0,        # PTU — $0 marginal
}
GPT52_COST_PER_REPORT_USD = 0.2
BUDGET_USD_DEFAULT = 2.0  # per-report token-budget ceiling

# Per-query timeout (catches a hung web-search/fetch so one bad query can't
# stall the whole multi-hour curve). Recovered: 120s budget cap notion; 500 ->
# unused token const; we use a generous wall timeout matching the 14B runner.
PER_QUERY_TIMEOUT_S = 600


# ── Path safety ──────────────────────────────────────────────────────────────
def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_relative_to_lex(child: Path, parent: Path) -> bool:
    """Lexical (no symlink-following) containment test."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_out(raw: str, label: str) -> Path:
    """Resolve an output root and HARD-REFUSE any path endangering the corpus.

    A protected path (e.g. results/experiments) may itself be a SYMLINK into the
    real canonical store (the workspace-move pointed results/experiments ->
    artifacts/experiments/canonical). So we test parenthood/containment against
    BOTH the lexical (un-resolved) protected path AND its resolved target, and
    we compare the candidate's resolved form against the resolved protected form.
    This way a literal parent like ``results`` (lexical parent of
    results/experiments) is refused even though the resolved target lives
    elsewhere.
    """
    candidate_lex = Path(raw)
    if not candidate_lex.is_absolute():
        candidate_lex = _REPO_ROOT / candidate_lex
    candidate = candidate_lex.resolve()
    for prot in PROTECTED_PATHS:
        # Each protected path contributes two reference points: its lexical
        # (un-resolved) location and its resolved (symlink-followed) target.
        for protected in {prot, prot.resolve()}:
            if candidate == protected or candidate_lex == protected:
                raise SystemExit(
                    f"REFUSING: {label} resolves to protected corpus path "
                    f"{protected}. This harness must NEVER write there. Choose a NEW dir."
                )
            if _is_relative_to(candidate, protected) or _is_relative_to_lex(candidate_lex, protected):
                raise SystemExit(
                    f"REFUSING: {label} ({candidate}) is INSIDE protected path "
                    f"{protected}. Choose a top-level dir outside all protected paths."
                )
            if _is_relative_to(protected, candidate) or _is_relative_to_lex(protected, candidate_lex):
                raise SystemExit(
                    f"REFUSING: {label} ({candidate}) is a PARENT of protected path "
                    f"{protected}; a run rooted there could traverse into the corpus. "
                    f"Choose a sibling dir."
                )
    return candidate


# ── Queries ──────────────────────────────────────────────────────────────────
def load_queries(limit: int, query_ids_file: str | None) -> list[dict]:
    """All eval_queries_v2 queries, sorted by id; optionally restricted/capped."""
    data = json.loads(EVAL_QUERIES.read_text())
    items = data["queries"] if isinstance(data, dict) else data
    if query_ids_file:
        try:
            raw = json.loads(Path(query_ids_file).read_text())
            ids = set(raw["query_ids"] if isinstance(raw, dict) else raw)
        except (json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"REFUSING: bad --query-ids-file {query_ids_file}: {e}")
        items = [q for q in items if str(q["id"]) in ids]
    items = sorted(items, key=lambda q: str(q["id"]))
    if limit and limit > 0:
        items = items[:limit]
    return items


# ── Cell naming / paths ──────────────────────────────────────────────────────
def cell_dir_name(tier: str, arch: str, sample_idx: int, best_of_n: int) -> str:
    """Write-subdir name for one (tier, arch, sample) cell.

    e.g. e9_gpt-4o-mini_p1            (single sample)
         e9_gpt-4o-mini_p1__s2        (best-of-N sample 2)

    The backbone is encoded in the name so judge ingestion and analysis can read
    the IV straight off the directory.
    """
    base = f"e9_{tier}_{arch}"
    if best_of_n > 1:
        base = f"{base}__s{sample_idx}"
    return base


def _result_path(results_root: Path, cell: str, query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return results_root / cell / f"{safe_id}.md"


def _checkpoint_path(ckpt_root: Path, cell: str, query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return ckpt_root / cell / f"{safe_id}.json"


def is_completed(ckpt_root: Path, cell: str, query_id: str) -> bool:
    cp = _checkpoint_path(ckpt_root, cell, query_id)
    if not cp.exists():
        return False
    try:
        data = json.loads(cp.read_text())
        return data.get("status") == "success"
    except (json.JSONDecodeError, OSError):
        return False


# ── Backbone-pinned import (the IV mechanism) ────────────────────────────────
def _import_pattern_for_tier(tier: str, arch: str):
    """Pin DEFAULT_MODEL=tier, purge config/pattern modules, re-import, HARD-verify.

    Returns the freshly-bound pattern module whose `DEFAULT_MODEL` == tier.
    """
    if arch not in PATTERNS:
        raise SystemExit(f"Unknown architecture {arch!r}. Known: {sorted(PATTERNS)}")
    if tier not in ALLOWED_BACKBONES:
        raise SystemExit(
            f"REFUSING tier {tier!r}: not a permitted PTU/Azure backbone. "
            f"Allowed: {sorted(ALLOWED_BACKBONES)}. Local 14B is GPU-blocked here "
            f"and generated on the GPU queue (run_e8_vintage_14b_gen.py)."
        )
    # Set the IV + pin SEARCH_MODEL to the corpus value BEFORE any import.
    os.environ["DEFAULT_MODEL"] = tier
    os.environ["SEARCH_MODEL"] = "gpt-4o-mini"  # corpus value, NEVER changed
    # Purge so the backbone re-binds at import time.
    for name in list(sys.modules):
        if name == "deep_research.config" or name.startswith("deep_research.patterns."):
            del sys.modules[name]
    mod = importlib.import_module(PATTERNS[arch])
    # Verify the backbone at the CONFIG layer — the authoritative source every pattern
    # reads from (config.py: DEFAULT_MODEL = _env("DEFAULT_MODEL", ...)). NOT every
    # pipeline re-exports DEFAULT_MODEL at its top level (p1/p2/p3 import only
    # MAX_COST_PER_RUN), so checking the pattern module's namespace gives a false
    # mismatch. The purge above forces config to re-bind from the env var on re-import.
    import deep_research.config as _cfg
    bound = getattr(_cfg, "DEFAULT_MODEL", None)
    if bound != tier:
        raise SystemExit(
            f"BACKBONE MISMATCH: cell e9_{tier}_{arch} — deep_research.config.DEFAULT_MODEL="
            f"{bound!r} but tier is {tier!r}. The env var must be set before the first "
            f"config/pattern import. Aborting to protect comparability."
        )
    return mod


# ── Per-cell generation ──────────────────────────────────────────────────────
async def generate_cell(
    tier: str,
    arch: str,
    sample_idx: int,
    best_of_n: int,
    queries: list[dict],
    results_root: Path,
    ckpt_root: Path,
    budget: float,
    resume: bool,
) -> dict:
    """Generate all reports for one (tier, arch, sample) cell.

    The backbone (DEFAULT_MODEL) is pinned in the environment for this tier
    BEFORE any pattern import (done inside _import_pattern_for_tier). We import
    the pattern lazily here so it binds the correct backbone.
    """
    cell = cell_dir_name(tier, arch, sample_idx, best_of_n)
    mod = _import_pattern_for_tier(tier, arch)

    n_ok = n_fail = n_skip = 0
    label = f"[CORPUS ANCHOR] " if tier == ANCHOR_TIER else ""
    print(f"\n### TIER backbone={tier} {label}arch={arch} sample={sample_idx}/{best_of_n} "
          f"(DEFAULT_MODEL set; corpus anchor={tier == ANCHOR_TIER}) ###")

    for q in queries:
        query_id = str(q["id"])
        if resume and is_completed(ckpt_root, cell, query_id):
            n_skip += 1
            continue
        rp = _result_path(results_root, cell, query_id)
        cp = _checkpoint_path(ckpt_root, cell, query_id)
        rp.parent.mkdir(parents=True, exist_ok=True)
        cp.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            report = await asyncio.wait_for(
                mod.run(q["query"], budget_usd=budget, query_id=query_id),
                timeout=PER_QUERY_TIMEOUT_S,
            )
            text = report.full_text()
            if not text.strip():
                text = f"# (empty report)\n\nQuery: {q['query']}\n"
            rp.write_text(text)
            elapsed = time.time() - t0
            meta = {
                "experiment": "E9_SCALE_CURVE",
                "backbone": tier,
                "architecture": arch,
                "sample_idx": sample_idx,
                "best_of_n": best_of_n,
                "cell": cell,
                "query": query_id,
                "status": "success",
                "elapsed_seconds": round(elapsed, 1),
                "chars": len(text),
                "total_tokens": getattr(report, "total_tokens", 0),
                "total_cost_usd": getattr(report, "total_cost_usd", 0.0),
                "sections": len(getattr(report, "sections", [])),
                "citations": len(getattr(report, "citations", [])),
                "default_model_bound": getattr(mod, "DEFAULT_MODEL", None),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cp.write_text(json.dumps(meta, indent=2, default=str))
            n_ok += 1
            print(f"    OK   {query_id}: {meta['elapsed_seconds']}s "
                  f"{meta['chars']} chars {meta['citations']} cites "
                  f"{meta['total_tokens']} tok", flush=True)
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 — log + resume
            elapsed = time.time() - t0
            status = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
            meta = {
                "experiment": "E9_SCALE_CURVE",
                "backbone": tier,
                "architecture": arch,
                "sample_idx": sample_idx,
                "best_of_n": best_of_n,
                "cell": cell,
                "query": query_id,
                "status": status,
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e)[:300] or "per-query timeout",
                "default_model_bound": getattr(mod, "DEFAULT_MODEL", None),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cp.write_text(json.dumps(meta, indent=2, default=str))
            n_fail += 1
            print(f"    FAIL {query_id}: {status} {meta['elapsed_seconds']}s "
                  f"— {meta['error'][:120]}", flush=True)

    return {"cell": cell, "tier": tier, "arch": arch, "sample_idx": sample_idx,
            "ok": n_ok, "fail": n_fail, "skip": n_skip}


# ── Planning / costing ───────────────────────────────────────────────────────
def build_plan(tiers: list[str], arches: list[str], best_of_n: int, n_queries: int) -> list[dict]:
    """List of cell descriptors (no API calls). Sorted tiers x arches x samples."""
    plan: list[dict] = []
    for tier in sorted(tiers):
        for arch in sorted(arches):
            for sample_idx in range(1, max(1, best_of_n) + 1):
                plan.append({
                    "tier": tier,
                    "arch": arch,
                    "sample_idx": sample_idx,
                    "best_of_n": best_of_n,
                    "cell": cell_dir_name(tier, arch, sample_idx, best_of_n),
                    "n_reports": n_queries,
                })
    return plan


def estimate_costs(plan: list[dict]) -> dict:
    total_reports = sum(c["n_reports"] for c in plan)
    gen_usd = sum(
        c["n_reports"] * TIER_COST_PER_REPORT_USD.get(c["tier"], 0.0) for c in plan
    )
    gpt52_calls = total_reports
    gpt52_usd = gpt52_calls * GPT52_COST_PER_REPORT_USD
    return {
        "total_reports": total_reports,
        "cells": len(plan),
        "gen_usd": gen_usd,
        "gpt52_calls": gpt52_calls,
        "gpt52_usd": gpt52_usd,
    }


# ── Symlink staging for the namespaced judge ─────────────────────────────────
def link_for_judging(results_root: Path, plan: list[dict]) -> None:
    """Create READ-ONLY symlink aliases of each E9 cell under results/experiments/<cell>.

    Makes ZERO API calls. Symlinks only point OUT to the E9 results root; the
    judge never writes to results/experiments. (You may instead judge in place
    with JUDGE_RESULTS_BASE=<results_root> — no symlinks needed.)
    """
    corpus_exp = _REPO_ROOT / "results" / "experiments"
    n_linked = n_existing = n_missing = 0
    for c in plan:
        src = (results_root / c["cell"]).resolve()
        if not src.exists():
            print(f"  (skip) {c['cell']}: no generated dir yet at {src}")
            n_missing += 1
            continue
        dst = corpus_exp / c["cell"]
        if dst.is_symlink() or dst.exists():
            n_existing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        n_linked += 1
    print(f"\n  Symlink aliases: {n_linked} new, {n_existing} existing, "
          f"{n_missing} not-yet-generated.")
    print(f"    linked results/experiments/<cell> -> {results_root}/<cell> (READ-ONLY)")
    print("    The namespaced runner reads its inputs from results/experiments/<cell>")
    print("    (READ-ONLY) and writes verdicts ONLY under --judge-out. E9 reports")
    print(f"    live under a NEW root, so the aliases only expose them; the real")
    print(f"    files stay under {results_root}.")


# ── Judge-command printer ────────────────────────────────────────────────────
def _print_judge_cmd(results_root: Path, plan: list[dict]) -> None:
    cell_names = ",".join(sorted({c["cell"] for c in plan}))
    print("\n  Now judge (GPT-5.2, corpus-safe):")
    print(f"    JUDGE_RESULTS_BASE={results_root} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} \\")
    print(f"      --patterns-raw {cell_names} --resume")
    print("\n  Authoritative judge stays GPT-5.2 (JUDGE_MODEL=gpt-5.2); the")
    print("  namespaced runner NEVER writes to the corpus.")
    print("\n  (alt: --link-for-judging makes symlink aliases inside results/experiments")
    print("   first, then judge with the default JUDGE_RESULTS_BASE=results/experiments.)")


# ── Main ─────────────────────────────────────────────────────────────────────
async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="E9 SCALE-CURVE generation harness (backbone is the IV; gpt-4o "
                    "is the corpus anchor). Corpus-safe: writes to NEW dirs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tiers", type=str, default=",".join(TIERS_DEFAULT),
                        help=f"Comma-separated backbone tiers (the IV). Default: "
                             f"{','.join(TIERS_DEFAULT)}")
    parser.add_argument("--arches", type=str, default=",".join(ARCHES_DEFAULT),
                        help=f"Comma-separated architectures held fixed across the "
                             f"sweep. Default: {','.join(ARCHES_DEFAULT)}")
    parser.add_argument("--best-of-n", type=int, default=1,
                        help="Independent samples per (tier,arch) cell. N>1 realizes "
                             "the best-of-N arm: best-of-N / selector / oracle are "
                             "computed downstream from these samples. Plan headline N=4.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only the first N queries (smoke/slice mode).")
    parser.add_argument("--query-ids-file", type=str, default="",
                        help="JSON file {'query_ids': [...]} restricting the run.")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT,
                        help=f"NEW results root (default: {DEFAULT_RESULTS_ROOT}).")
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT,
                        help=f"NEW checkpoint root (default: {DEFAULT_CHECKPOINT_ROOT}).")
    parser.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT,
                        help="Per-report budget cap (USD).")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                        help="Re-run completed cells (default: resume).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan + cost estimate. ZERO API calls, nothing written.")
    parser.add_argument("--link-for-judging", action="store_true",
                        help="Create READ-ONLY symlink aliases of each generated E9 cell "
                             "dir under results/experiments/<cell> so the corpus-safe "
                             "GPT-5.2 namespaced judge runner can read them. Makes ZERO "
                             "API calls. Symlinks only point OUT to the E9 results root.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run in-process unit checks (no API, no model). Exit 0/1.")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    # Validate arches/tiers up front (no work yet).
    for a in arches:
        if a not in PATTERNS:
            raise SystemExit(f"Unknown architecture {a!r}. Known: {sorted(PATTERNS)}")
    for t in tiers:
        if t not in ALLOWED_BACKBONES:
            raise SystemExit(
                f"REFUSING tier {t!r}: not a permitted PTU/Azure backbone. "
                f"Allowed: {sorted(ALLOWED_BACKBONES)}. The local 14B tier "
                f"({SKIPPED_LOCAL_TIER}) is GPU-blocked here and generated on the "
                f"GPU queue. {SKIPPED_LOCAL_REASON}"
            )

    results_root = resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = resolve_safe_out(args.checkpoint_root, "--checkpoint-root")

    queries = load_queries(args.limit, args.query_ids_file or None)
    n_queries = len(queries)
    plan = build_plan(tiers, arches, args.best_of_n, n_queries)
    costs = estimate_costs(plan)

    print("=" * 74)
    print("E9 SCALE-CURVE — orchestration returns vs backbone capability")
    print("=" * 74)
    print(f"  IV (backbone) tiers (low->high): {', '.join(tiers)}")
    print(f"  Corpus anchor tier present: {ANCHOR_TIER in tiers} "
          f"(gpt-4o = DEFAULT_MODEL on PTU sthree-ptu-02, matches 248k corpus)")
    print(f"  SKIPPED (GPU-blocked here) tier: {SKIPPED_LOCAL_TIER}")
    print(f"     reason: {SKIPPED_LOCAL_REASON}")
    print(f"  Architectures (fixed across sweep): {', '.join(arches)}")
    print(f"  best-of-N: {args.best_of_n} sample(s) per (tier,arch) cell")
    print(f"  Queries: {n_queries}")
    print(f"  SEARCH_MODEL pinned to corpus value: gpt-4o-mini (untouched)")
    print(f"  Generation WRITE root (NEW): {results_root}")
    print(f"  Checkpoint WRITE root (NEW): {ckpt_root}")
    print(f"  Corpus protected (never written): "
          f"{', '.join(str(p) for p in PROTECTED_PATHS)}")
    print(f"  ->  cells: {costs['cells']}")
    print(f"  ->  total reports: {costs['total_reports']}")
    print(f"  Est. generation cost (standard endpoints; PTU gpt-4o=$0): "
          f"${costs['gen_usd']:.2f}")
    print(f"  Est. GPT-5.2 judge calls: {costs['gpt52_calls']} "
          f"(~${costs['gpt52_usd']:.2f})")
    print("  Per-cell plan:")
    for c in plan:
        anchor = " [CORPUS ANCHOR]" if c["tier"] == ANCHOR_TIER else ""
        print(f"    [{c['cell']}]{anchor} reports={c['n_reports']}")

    if args.dry_run:
        print("\n  [DRY RUN] No API calls made, nothing written.")
        _print_judge_cmd(results_root, plan)
        return

    if args.link_for_judging:
        link_for_judging(results_root, plan)
        _print_judge_cmd(results_root, plan)
        return

    # ── Real generation (sorted cells; idempotent --resume) ──────────────────
    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for c in plan:
        res = await generate_cell(
            tier=c["tier"], arch=c["arch"], sample_idx=c["sample_idx"],
            best_of_n=c["best_of_n"], queries=queries, results_root=results_root,
            ckpt_root=ckpt_root, budget=args.budget, resume=args.resume,
        )
        summary.append(res)

    print("\n" + "=" * 74)
    print("  E9 GENERATION COMPLETE")
    print("=" * 74)
    for r in summary:
        print(f"  [{r['cell']}] ok={r['ok']} fail={r['fail']} skip={r['skip']}")
    print(f"  After generation, judge with the corpus-safe namespaced runner.")
    _print_judge_cmd(results_root, plan)


def _self_test() -> int:
    """Cheap in-process checks: no API, no model, no writes to real roots."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    # cell naming
    check("cell name single-sample",
          cell_dir_name("gpt-4o-mini", "p1", 1, 1) == "e9_gpt-4o-mini_p1")
    check("cell name best-of-N",
          cell_dir_name("gpt-4o-mini", "p1", 2, 4) == "e9_gpt-4o-mini_p1__s2")
    # plan + cost
    plan = build_plan(["gpt-4o-mini", "gpt-4.1", "gpt-4o"], ["p0", "p1", "p4"], 4, 90)
    check("full curve cell count == 3*3*4 == 36", len(plan) == 36)
    costs = estimate_costs(plan)
    check("full curve reports == 36*90 == 3240", costs["total_reports"] == 3240)
    # gen cost: only mini/4.1 cost; gpt-4o is $0. per tier: 3 arches*4 samples*90 = 1080.
    expected_gen = 1080 * 0.01 + 1080 * 0.08 + 1080 * 0.0
    check("gen cost only mini+4.1", abs(costs["gen_usd"] - expected_gen) < 1e-6)
    check("gpt52 calls == total reports", costs["gpt52_calls"] == 3240)
    # safety: protected roots are refused
    for bad in ["results/experiments", "results/judge_gpt52", "data/analysis",
                "results/experiments/foo", "results"]:
        try:
            resolve_safe_out(bad, "--results-root")
            check(f"refuse protected {bad}", False)
        except SystemExit:
            check(f"refuse protected {bad}", True)
    # safe roots accepted
    try:
        p = resolve_safe_out("results/experiments_e9_scale", "--results-root")
        check("accept NEW e9 root", p.name == "experiments_e9_scale")
    except SystemExit:
        check("accept NEW e9 root", False)
    # tier guard
    check("anchor tier in allowed", ANCHOR_TIER in ALLOWED_BACKBONES)
    check("14B not in allowed (GPU job)", SKIPPED_LOCAL_TIER not in ALLOWED_BACKBONES)
    # queries load + sort
    qs = load_queries(3, None)
    check("query load limit 3", len(qs) == 3 and qs == sorted(qs, key=lambda q: str(q["id"])))
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
