#!/usr/bin/env python3
"""Generate our two LOCAL 7B research systems' reports on the 5 external benchmark
query sets, for a head-to-head leaderboard comparison (our 7B systems vs published
systems).

The two local systems:
  - p9  : Qwen2.5-7B-Instruct baseline       (deep_research.patterns.p9_local_baseline)
  - p10 : GAIR/DeepResearcher-7b RL agent     (deep_research.patterns.p10_deep_researcher)

Design constraints (deliberate, do not "optimise" away):
  * ONE model in VRAM at a time. Each pattern loads its 7B model ONCE (via the
    LocalLLMCaller singleton cache), runs over EVERY sliced query across ALL
    benchmarks, then the model is unloaded and VRAM is freed (gc + empty_cache)
    before the next pattern loads. We never hold two 7B models at once.
  * 4-bit nf4 / bf16 quantization is handled inside LocalLLMCaller; we rely on
    PYTORCH_ALLOC_CONF=expandable_segments:True (set in local_llm_caller on import)
    to keep fragmentation down on the 16GB RTX 5080.
  * Corpus-safe: this script writes ONLY under results/local_benchmark/. It never
    touches results/judge_gpt52, results/experiments, data/analysis, or
    reports/eval_v2/verdicts.
  * Resumable: an existing non-empty <query_id>.md is skipped.

Benchmark slice:
  For each of the 5 benchmarks we take a representative slice of N queries
  (default 15), chosen DETERMINISTICALLY as the first N by sorted id. Configurable
  via --limit.

Usage:
    python scripts/run_local_benchmark_gen.py --dry-run            # plan only, no model load
    python scripts/run_local_benchmark_gen.py --limit 15          # full run, 2x5x15 = 150 reports
    python scripts/run_local_benchmark_gen.py --smoke             # load each model in 4-bit,
                                                                  # print VRAM GB, run 1 query each
    python scripts/run_local_benchmark_gen.py --patterns p9       # restrict to one pattern
    python scripts/run_local_benchmark_gen.py --benchmarks draco litqa2
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib
import json
import sys
import time
from pathlib import Path

# Repo root on sys.path so `deep_research...` imports resolve regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Configuration ─────────────────────────────────────────────────────────────

# The two local 7B systems. Order matters: each loads once, runs, unloads.
PATTERN_PIPELINES = {
    "p9": "deep_research.patterns.p9_local_baseline.pipeline",
    "p10": "deep_research.patterns.p10_deep_researcher.pipeline",
}
PATTERN_ORDER = ["p9", "p10"]

# The 5 external benchmark query sets. Each file is a JSON list of objects with the
# shared schema {id, query, domain, difficulty, rubric, reference_answer,
# expected_citations, metadata}.
BENCHMARKS = ["draco", "litqa2", "research_qa", "deepsearch_qa", "freshwiki"]


def _benchmark_path(name: str) -> Path:
    return REPO_ROOT / "data" / "benchmarks" / name / f"{name}_queries.json"


# Corpus-safe output root. The script is hard-pinned to this directory.
OUTPUT_ROOT = REPO_ROOT / "results" / "local_benchmark"

# Directories this script must NEVER write into (sanity guard).
FORBIDDEN_PREFIXES = [
    REPO_ROOT / "results" / "judge_gpt52",
    REPO_ROOT / "results" / "experiments",
    REPO_ROOT / "data" / "analysis",
    REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

DEFAULT_LIMIT = 15
DEFAULT_BUDGET_USD = 2.0


# ── Query loading / slicing ───────────────────────────────────────────────────

def load_slice(benchmark: str, limit: int) -> list[dict]:
    """Deterministic representative slice: first `limit` queries by sorted id."""
    path = _benchmark_path(benchmark)
    if not path.exists():
        raise FileNotFoundError(f"benchmark query file not found: {path}")
    items = json.loads(path.read_text())
    if not isinstance(items, list):  # tolerate {"queries": [...]} just in case
        items = items.get("queries", [])
    items_sorted = sorted(items, key=lambda q: str(q["id"]))
    return items_sorted[:limit]


def build_plan(patterns: list[str], benchmarks: list[str], limit: int) -> dict:
    """Return {(pattern, benchmark): [query dicts]} for the requested slice."""
    plan: dict = {}
    for benchmark in benchmarks:
        sliced = load_slice(benchmark, limit)
        for pattern in patterns:
            plan[(pattern, benchmark)] = sliced
    return plan


# ── Output path / safety ──────────────────────────────────────────────────────

def _report_path(pattern: str, benchmark: str, query_id: str) -> Path:
    # Sanitize query_id for filesystem safety (ids are already filename-safe, but
    # guard against any stray slashes).
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return OUTPUT_ROOT / f"{pattern}_{benchmark}" / f"{safe_id}.md"


def _assert_corpus_safe(path: Path) -> None:
    """Refuse to write anywhere except under OUTPUT_ROOT."""
    resolved = path.resolve()
    if OUTPUT_ROOT.resolve() not in resolved.parents and resolved != OUTPUT_ROOT.resolve():
        raise RuntimeError(f"CORPUS-SAFETY VIOLATION: refusing to write outside "
                           f"{OUTPUT_ROOT}: {resolved}")
    for forbidden in FORBIDDEN_PREFIXES:
        fr = forbidden.resolve()
        if fr == resolved or fr in resolved.parents:
            raise RuntimeError(f"CORPUS-SAFETY VIOLATION: path under forbidden "
                               f"prefix {forbidden}: {resolved}")


def is_done(pattern: str, benchmark: str, query_id: str) -> bool:
    p = _report_path(pattern, benchmark, query_id)
    return p.exists() and p.stat().st_size > 0


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def vram_allocated_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def vram_reserved_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e9
    except Exception:
        pass
    return 0.0


def free_vram() -> None:
    """Unload the cached local model and free VRAM (gc + empty_cache)."""
    try:
        from deep_research.tools.local_llm_caller import unload_model
        unload_model()
    except Exception as e:
        print(f"  [warn] unload_model failed: {e}")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── Single report ─────────────────────────────────────────────────────────────

async def run_one(mod, pattern: str, benchmark: str, query: dict, budget: float) -> dict:
    """Run one pattern pipeline on one query and write its report .md."""
    query_id = query["id"]
    out_path = _report_path(pattern, benchmark, query_id)
    _assert_corpus_safe(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        report = await mod.run(query["query"], budget_usd=budget, query_id=query_id)
        text = report.full_text()
        if not text.strip():
            # Still write something so the failure is visible, but flag it.
            text = f"# (empty report)\n\nQuery: {query['query']}\n"
        out_path.write_text(text)
        elapsed = time.time() - t0
        return {
            "status": "success",
            "pattern": pattern,
            "benchmark": benchmark,
            "query_id": query_id,
            "elapsed_seconds": round(elapsed, 1),
            "chars": len(text),
            "tokens": getattr(report, "total_tokens", 0),
            "sections": len(getattr(report, "sections", [])),
            "citations": len(getattr(report, "citations", [])),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": "error",
            "pattern": pattern,
            "benchmark": benchmark,
            "query_id": query_id,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e)[:300],
        }


# ── Per-pattern run (model loaded once) ───────────────────────────────────────

async def run_pattern(pattern: str, benchmarks: list[str], limit: int,
                      budget: float, resume: bool) -> list[dict]:
    """Run one pattern over all sliced queries across all benchmarks.

    The model is loaded lazily on the first query (LocalLLMCaller singleton cache)
    and stays resident for every subsequent query in this pattern. The caller is
    responsible for unloading it afterwards.
    """
    mod = importlib.import_module(PATTERN_PIPELINES[pattern])
    results: list[dict] = []

    # Build the (benchmark -> queries) work list for this pattern.
    work: list[tuple[str, dict]] = []
    for benchmark in benchmarks:
        for q in load_slice(benchmark, limit):
            if resume and is_done(pattern, benchmark, q["id"]):
                continue
            work.append((benchmark, q))

    total = len(work)
    print(f"\n{'='*64}")
    print(f"PATTERN {pattern}: {total} reports to generate "
          f"(across {len(benchmarks)} benchmarks)")
    print(f"{'='*64}")

    if total == 0:
        print(f"  all {pattern} reports already present — nothing to do")
        return results

    model_loaded = False
    for i, (benchmark, q) in enumerate(work, 1):
        print(f"  [{pattern}] [{i}/{total}] {benchmark} :: {q['id']} ...", flush=True)
        res = await run_one(mod, pattern, benchmark, q, budget)
        results.append(res)
        if not model_loaded:
            # After the first run the model is resident; report VRAM once.
            print(f"    VRAM after first {pattern} load: "
                  f"allocated={vram_allocated_gb():.2f} GB, "
                  f"reserved={vram_reserved_gb():.2f} GB")
            model_loaded = True
        if res["status"] == "success":
            print(f"    OK {res['elapsed_seconds']}s, {res['chars']} chars, "
                  f"{res['sections']} sections, {res['citations']} cites")
        else:
            print(f"    FAILED {res['elapsed_seconds']}s — {res.get('error','')[:120]}")

    return results


# ── Smoke test (verify fit + one query per model) ─────────────────────────────

async def smoke(benchmarks: list[str], limit: int, budget: float) -> None:
    """Load each model in 4-bit, print VRAM GB, run ONE query end-to-end, unload."""
    print(f"\n{'#'*64}\nSMOKE TEST — verify 4-bit fit + one report per model\n{'#'*64}")
    # Pick the first query of the first available benchmark (sorted id).
    first_benchmark = benchmarks[0]
    sample_q = load_slice(first_benchmark, limit)[0]
    print(f"Sample query: {first_benchmark} :: {sample_q['id']}")
    print(f"  {sample_q['query'][:160]}...")

    for pattern in PATTERN_ORDER:
        print(f"\n--- SMOKE {pattern} ---")
        before = vram_allocated_gb()
        print(f"  VRAM before load: allocated={before:.2f} GB")
        mod = importlib.import_module(PATTERN_PIPELINES[pattern])
        t0 = time.time()
        res = await run_one(mod, pattern, first_benchmark, sample_q, budget)
        alloc = vram_allocated_gb()
        reserved = vram_reserved_gb()
        print(f"  VRAM after load+gen: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB "
              f"(must be well under 16 GB)")
        if res["status"] == "success":
            out = _report_path(pattern, first_benchmark, sample_q["id"])
            print(f"  report written: {out}")
            print(f"  {res['chars']} chars, {res['sections']} sections, "
                  f"{res['elapsed_seconds']}s")
        else:
            print(f"  SMOKE FAILED — {res.get('error','')}")
        free_vram()
        print(f"  VRAM after unload: allocated={vram_allocated_gb():.2f} GB "
              f"(elapsed {time.time()-t0:.0f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Queries per benchmark (sorted-id slice). Default {DEFAULT_LIMIT}.")
    parser.add_argument("--patterns", nargs="+", default=PATTERN_ORDER,
                        choices=PATTERN_ORDER, help="Patterns to run (default both).")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS,
                        choices=BENCHMARKS, help="Benchmarks to run (default all 5).")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help="Budget per run (USD). Local inference is $0 but the "
                             "CostTracker enforces a token ceiling.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                        help="Re-generate reports even if a .md already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan + per-(pattern,benchmark) counts. No model load.")
    parser.add_argument("--smoke", action="store_true",
                        help="Load each model in 4-bit, print VRAM, run ONE query each, unload.")
    args = parser.parse_args()

    patterns = [p for p in PATTERN_ORDER if p in args.patterns]
    benchmarks = [b for b in BENCHMARKS if b in args.benchmarks]

    # ── Plan ──────────────────────────────────────────────────────────────────
    plan = build_plan(patterns, benchmarks, args.limit)
    total_planned = sum(len(v) for v in plan.values())

    print(f"{'='*64}")
    print("LOCAL BENCHMARK GENERATION PLAN")
    print(f"{'='*64}")
    print(f"Patterns:    {patterns}")
    print(f"Benchmarks:  {benchmarks}")
    print(f"Limit:       {args.limit} queries/benchmark (deterministic, sorted id)")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Resume:      {args.resume}")
    print()
    print(f"Per-(pattern, benchmark) counts:")
    for pattern in patterns:
        for benchmark in benchmarks:
            qs = plan[(pattern, benchmark)]
            done = sum(1 for q in qs if is_done(pattern, benchmark, q["id"]))
            todo = len(qs) - done
            print(f"  {pattern:<4} x {benchmark:<14} : {len(qs):>3} planned "
                  f"({done} done, {todo} to do)")
    print()
    print(f"TOTAL reports planned: {total_planned} "
          f"({len(patterns)} patterns x {len(benchmarks)} benchmarks x {args.limit})")
    print()

    if args.dry_run:
        print("DRY RUN — no model loaded, nothing written.")
        return

    if args.smoke:
        await smoke(benchmarks, args.limit, args.budget)
        return

    # ── Execute: one pattern at a time, one model in VRAM at a time ────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    all_results: list[dict] = []

    for pattern in patterns:
        results = await run_pattern(pattern, benchmarks, args.limit,
                                    args.budget, args.resume)
        all_results.extend(results)
        # Free VRAM before the next pattern's model loads. One model at a time.
        print(f"\n  [{pattern}] done — unloading model and freeing VRAM ...")
        free_vram()
        print(f"  [{pattern}] VRAM after unload: allocated={vram_allocated_gb():.2f} GB")

    # ── Summary + manifest (written under OUTPUT_ROOT only) ────────────────────
    elapsed = time.time() - overall_start
    successes = sum(1 for r in all_results if r["status"] == "success")
    errors = sum(1 for r in all_results if r["status"] == "error")

    print(f"\n{'='*64}")
    print("LOCAL BENCHMARK GENERATION COMPLETE")
    print(f"{'='*64}")
    print(f"Generated this run: {len(all_results)}  (success {successes}, error {errors})")
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")

    manifest = {
        "patterns": patterns,
        "benchmarks": benchmarks,
        "limit": args.limit,
        "total_planned": total_planned,
        "generated_this_run": len(all_results),
        "successes": successes,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
    }
    manifest_path = OUTPUT_ROOT / "run_manifest.json"
    _assert_corpus_safe(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
