#!/usr/bin/env python3
"""E8 / E9 — generate the Qwen2.5-14B vintage/capacity arm on the FROZEN P9 scaffold.

This is the GENERATION runner for the 14B arm ONLY (pattern p17_scale_qwen25_14b),
the larger-capacity local backbone that EXTENDS the E8 vintage curve. It runs the
EXACT frozen P9 scaffold (search -> extract -> single report-gen) over the SAME query
slice the 7B vintage arms used (all 90 queries of data/eval_queries_v2.json, sorted by
id; n=90 to match the base_p9 / base_p14 7B arms), changing ONLY the backbone: the 14B
runs through llama.cpp / GGUF (LlamaCppLLMCaller) because the transformers/bnb path OOMs
at weights-materialisation on the 16 GB RTX 5080.

Determinism: the GGUF backend decodes strict-greedy (temperature=0, top_k=1, top_p=1.0,
repeat_penalty=1.0, fixed seed=42); queries are processed in sorted-id order.

Corpus safety (HARD): writes ONLY under results/experiments_e8_14b/p17_scale_qwen25_14b/.
It NEVER touches results/experiments, results/judge_gpt52, data/analysis, or
reports/eval_v2/verdicts. Resumable: an existing non-empty <query_id>.md is skipped.

This script does GENERATION only ($0, local). JUDGING is a SEPARATE, human-launched,
paid step:

    JUDGE_RESULTS_BASE=results/experiments_e8_14b \\
      python scripts/run_gpt52_judge_namespaced.py \\
        --judge-out results/judge_gpt52_e8_14b \\
        --patterns-raw p17_scale_qwen25_14b

Usage:
    python scripts/run_e8_vintage_14b_gen.py --dry-run     # plan only, no model load
    python scripts/run_e8_vintage_14b_gen.py --smoke       # load GGUF, print VRAM, run 1 query
    python scripts/run_e8_vintage_14b_gen.py               # full run, 90 queries (resumable)
    python scripts/run_e8_vintage_14b_gen.py --limit 5     # first 5 queries (sorted id)
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path so `deep_research...` imports resolve regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── CUDA library path for the sm_120 llama.cpp build (set BEFORE native import) ──
def _ensure_cuda_ld_library_path() -> None:
    parts: list[str] = []
    cudatk = REPO_ROOT / ".cudatk" / "lib"
    if cudatk.is_dir():
        parts.append(str(cudatk))
    nvidia_root = REPO_ROOT / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        for lib in sorted(nvidia_root.glob("*/lib")):
            if lib.is_dir():
                parts.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = existing.split(":") if existing else []
    new_parts = [p for p in parts if p not in existing_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


_ensure_cuda_ld_library_path()

# ── Configuration ─────────────────────────────────────────────────────────────

PATTERN = "p17_scale_qwen25_14b"
PATTERN_MODULE = "deep_research.patterns.p17_scale_qwen25_14b.pipeline"
GGUF_PATH = REPO_ROOT / "models" / "gguf" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"

EVAL_QUERIES = REPO_ROOT / "data" / "eval_queries_v2.json"

# Corpus-safe output root — hard-pinned. The 14B arm gets its OWN root, distinct
# from the protected corpus and from the 7B arms' dirs.
OUTPUT_ROOT = REPO_ROOT / "results" / "experiments_e8_14b"

# Directories this script must NEVER write into (sanity guard).
FORBIDDEN_PREFIXES = [
    REPO_ROOT / "results" / "judge_gpt52",
    REPO_ROOT / "results" / "experiments",
    REPO_ROOT / "data" / "analysis",
    REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

DEFAULT_BUDGET_USD = 2.0
# n=90 to match the base_p9 / base_p14 7B arms exactly. The quarantined query
# 82de3e92 is judge-specific (Claude-Code AUP false-positive) and is NOT dropped
# from the study, so it IS generated here; the judge step handles its panel.
N_QUERIES = 90


# ── Query loading / slicing ─────────────────────────────────────────────────--

def load_queries(limit: int) -> list[dict]:
    """All eval_queries_v2 queries, sorted by id, optionally capped to `limit`."""
    data = json.loads(EVAL_QUERIES.read_text())
    items = data["queries"] if isinstance(data, dict) else data
    items_sorted = sorted(items, key=lambda q: str(q["id"]))
    if limit and limit > 0:
        items_sorted = items_sorted[:limit]
    return items_sorted


# ── Output path / safety ──────────────────────────────────────────────────────

def _report_path(query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return OUTPUT_ROOT / PATTERN / f"{safe_id}.md"


def _assert_corpus_safe(path: Path) -> None:
    """Refuse to write anywhere except under OUTPUT_ROOT."""
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(
            f"CORPUS-SAFETY VIOLATION: refusing to write outside {OUTPUT_ROOT}: {resolved}"
        )
    for forbidden in FORBIDDEN_PREFIXES:
        fr = forbidden.resolve()
        if fr == resolved or fr in resolved.parents:
            raise RuntimeError(
                f"CORPUS-SAFETY VIOLATION: path under forbidden prefix {forbidden}: {resolved}"
            )


def is_done(query_id: str) -> bool:
    p = _report_path(query_id)
    return p.exists() and p.stat().st_size > 0


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def _nvidia_smi_used_mb() -> float | None:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def print_vram(tag: str) -> None:
    used = _nvidia_smi_used_mb()
    if used is not None:
        print(f"  VRAM ({tag}): {used:.0f} MiB used  (must stay well under 16 GiB)")
    else:
        print(f"  VRAM ({tag}): nvidia-smi unavailable")


def free_vram() -> None:
    try:
        from deep_research.tools.llamacpp_llm_caller import unload_model

        unload_model()
    except Exception as e:
        print(f"  [warn] unload_model failed: {e}")
    gc.collect()


# ── Single report ─────────────────────────────────────────────────────────────

async def run_one(mod, query: dict, budget: float) -> dict:
    """Run the P17 (frozen P9 scaffold, 14B GGUF backbone) on one query; write its .md."""
    query_id = query["id"]
    out_path = _report_path(query_id)
    _assert_corpus_safe(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        report = await mod.run(query["query"], budget_usd=budget, query_id=query_id)
        text = report.full_text()
        if not text.strip():
            text = f"# (empty report)\n\nQuery: {query['query']}\n"
        out_path.write_text(text)
        elapsed = time.time() - t0
        return {
            "status": "success",
            "pattern": PATTERN,
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
            "pattern": PATTERN,
            "query_id": query_id,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e)[:300],
        }


# ── Smoke (verify GGUF load + one query) ───────────────────────────────────────

async def smoke(queries: list[dict], budget: float) -> None:
    print(f"\n{'#'*64}\nSMOKE TEST — GGUF load + ONE report (14B, frozen P9 scaffold)\n{'#'*64}")
    if not GGUF_PATH.exists():
        print(f"  GGUF model MISSING at {GGUF_PATH}; cannot smoke.")
        return
    import importlib

    mod = importlib.import_module(PATTERN_MODULE)
    sample_q = queries[0]
    print(f"Sample query: {sample_q['id']}")
    print(f"  {sample_q['query'][:160]}...")
    print_vram("before load")
    t0 = time.time()
    res = await run_one(mod, sample_q, budget)
    print_vram("after load+gen")
    if res["status"] == "success":
        print(f"  report written: {_report_path(sample_q['id'])}")
        print(f"  {res['chars']} chars, {res['sections']} sections, "
              f"{res['citations']} cites, {res['elapsed_seconds']}s")
    else:
        print(f"  SMOKE FAILED — {res.get('error','')}")
    free_vram()
    print(f"  (smoke elapsed {time.time()-t0:.0f}s)")


# ── Full run (model loaded once, sequential) ──────────────────────────────────

async def full_run(queries: list[dict], budget: float, resume: bool) -> list[dict]:
    import importlib

    if not GGUF_PATH.exists():
        raise FileNotFoundError(f"GGUF model not found at {GGUF_PATH}")

    mod = importlib.import_module(PATTERN_MODULE)

    work = [q for q in queries if not (resume and is_done(q["id"]))]
    total = len(work)
    print(f"\n{'='*64}")
    print(f"PATTERN {PATTERN}: {total} reports to generate (of {len(queries)} planned)")
    print(f"{'='*64}")
    if total == 0:
        print("  all reports already present — nothing to do")
        return []

    results: list[dict] = []
    model_reported = False
    for i, q in enumerate(work, 1):
        print(f"  [{i}/{total}] {q['id']} ...", flush=True)
        try:
            # Per-query timeout: a normal report (search + 14B synthesis) takes ~3-4 min; 10 min
            # catches a hung web-search/fetch so one bad query can't stall the whole run.
            res = await asyncio.wait_for(run_one(mod, q, budget), timeout=600)
        except asyncio.TimeoutError:
            print(f"  [{i}/{total}] {q['id']} TIMEOUT after 600s — skipping (no .md written; resumable)", flush=True)
            res = {"query_id": q["id"], "status": "timeout", "elapsed_seconds": 600.0,
                   "error": "per-query timeout (600s) — flaky search/fetch hang, query left unwritten for resume"}
        results.append(res)
        if not model_reported:
            print_vram("after first 14B load")
            model_reported = True
        if res["status"] == "success":
            print(f"    OK {res['elapsed_seconds']}s, {res['chars']} chars, "
                  f"{res['sections']} sections, {res['citations']} cites")
        else:
            print(f"    FAILED {res['elapsed_seconds']}s — {res.get('error','')[:120]}")

    print("\n  done — unloading GGUF model and freeing VRAM ...")
    free_vram()
    print_vram("after unload")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=N_QUERIES,
                    help=f"Queries (sorted-id slice). Default {N_QUERIES} (matches the 7B arms).")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                    help="Per-run token-budget ceiling (local inference is $0).")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="Re-generate reports even if a .md already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan + counts. No model load, nothing written.")
    ap.add_argument("--smoke", action="store_true",
                    help="Load the GGUF model, print VRAM, run ONE query, unload.")
    args = ap.parse_args()

    queries = load_queries(args.limit)
    done = sum(1 for q in queries if is_done(q["id"]))
    todo = len(queries) - done

    print(f"{'='*64}")
    print("E8/E9 14B VINTAGE/CAPACITY GENERATION PLAN")
    print(f"{'='*64}")
    print(f"Pattern:      {PATTERN}")
    print(f"Backbone:     Qwen2.5-14B-Instruct (GGUF Q4_K_M via llama.cpp, strict-greedy)")
    print(f"GGUF path:    {GGUF_PATH}  ({'present' if GGUF_PATH.exists() else 'MISSING'})")
    print(f"Queries:      {len(queries)} (sorted id; matches the n=90 7B arms)")
    print(f"Output root:  {OUTPUT_ROOT}")
    print(f"Resume:       {args.resume}  ({done} done, {todo} to do)")
    print(f"LD_LIBRARY_PATH set: {bool(os.environ.get('LD_LIBRARY_PATH'))}")
    print()

    if args.dry_run:
        print("DRY RUN — no model loaded, nothing written.")
        print("\nJudging (separate, paid, human-launched):")
        print("  JUDGE_RESULTS_BASE=results/experiments_e8_14b \\")
        print("    python scripts/run_gpt52_judge_namespaced.py \\")
        print("      --judge-out results/judge_gpt52_e8_14b \\")
        print(f"      --patterns-raw {PATTERN}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        await smoke(queries, args.budget)
        return

    overall_start = time.time()
    results = await full_run(queries, args.budget, args.resume)
    elapsed = time.time() - overall_start

    successes = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'='*64}")
    print("14B VINTAGE/CAPACITY GENERATION COMPLETE")
    print(f"{'='*64}")
    print(f"Generated this run: {len(results)} (success {successes}, error {errors})")
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")

    manifest = {
        "pattern": PATTERN,
        "backbone": "Qwen2.5-14B-Instruct GGUF Q4_K_M (llama.cpp, strict-greedy seed=42)",
        "n_planned": len(queries),
        "generated_this_run": len(results),
        "successes": successes,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    manifest_path = OUTPUT_ROOT / "run_manifest_p17_14b.json"
    _assert_corpus_safe(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
