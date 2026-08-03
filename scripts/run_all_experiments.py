#!/usr/bin/env python3
"""Run all research experiments sequentially: base patterns + ablation sweeps.

This script runs every experiment one at a time to avoid rate-limit conflicts.
It uses checkpointing so it can be safely interrupted and resumed.

Usage:
    python scripts/run_all_experiments.py                  # Run everything
    python scripts/run_all_experiments.py --base-only       # Only base patterns
    python scripts/run_all_experiments.py --ablations-only   # Only ablation sweeps
    python scripts/run_all_experiments.py --query q1_bert_vs_gpt  # Single query
    python scripts/run_all_experiments.py --pattern p6       # Single base pattern
    python scripts/run_all_experiments.py --resume            # Explicitly resume from checkpoint (default)
    python scripts/run_all_experiments.py --no-resume         # Re-run completed cells
    python scripts/run_all_experiments.py --dry-run           # Show what would run
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.ablation.framework import ABLATION_REGISTRY, AblationRunner


# ── Pattern registry ──────────────────────────────────────────────────────────

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
    "p9": "deep_research.patterns.p9_local_baseline.pipeline",
    "p10": "deep_research.patterns.p10_deep_researcher.pipeline",
    "p11": "deep_research.patterns.p11_react.pipeline",
    "p12": "deep_research.patterns.p12_rl_trained.pipeline",
}

# Order: GPT-4o patterns first (free on PTU), then local models
PATTERN_ORDER = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p11", "p9", "p10", "p12"]

BUDGET_USD = 2.0

# ── Checkpoint management ─────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints/experiments")
RESULTS_DIR = Path("results/experiments")


def _checkpoint_path(experiment_id: str, query_id: str) -> Path:
    return CHECKPOINT_DIR / experiment_id / f"{query_id}.json"


def _result_path(experiment_id: str, query_id: str) -> Path:
    return RESULTS_DIR / experiment_id / f"{query_id}.md"


def is_completed(experiment_id: str, query_id: str) -> bool:
    cp = _checkpoint_path(experiment_id, query_id)
    if not cp.exists():
        return False
    try:
        data = json.loads(cp.read_text())
        return data.get("status") == "success"
    except (json.JSONDecodeError, KeyError):
        return False


def save_checkpoint(experiment_id: str, query_id: str, result: dict):
    cp = _checkpoint_path(experiment_id, query_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(result, indent=2, default=str))


def save_result(experiment_id: str, query_id: str, report_text: str):
    rp = _result_path(experiment_id, query_id)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report_text)


# ── Load queries ──────────────────────────────────────────────────────────────

# 5 representative queries (one per source) for ablation sweeps
REPRESENTATIVE_QUERY_IDS = [
    "q1_bert_vs_gpt",                                    # custom / NLP
    "ce335c0c-f136-4408-a216-6a891cae861f",              # draco / Academic
    "dsqa_0868",                                          # deepsearch_qa
    "174539434801411914-s20",                             # research_qa
    "a45c277e-55d9-4e7f-b1de-37fc2e19daf6",              # litqa2
]


def load_queries(
    query_filter: str | None = None,
    representative_only: bool = False,
) -> list[dict]:
    with open("data/eval_queries_v2.json") as f:
        data = json.load(f)
    queries = data["queries"]
    if query_filter:
        queries = [q for q in queries if q["id"] == query_filter]
        if not queries:
            print(f"ERROR: Query '{query_filter}' not found")
            sys.exit(1)
    elif representative_only:
        queries = [q for q in queries if q["id"] in REPRESENTATIVE_QUERY_IDS]
    return queries


# ── Reproducibility seed ───────────────────────────────────────────────────────

DEFAULT_SEED = 42


def set_reproducibility_seed(seed: int = DEFAULT_SEED):
    """Set random seeds for reproducibility across runs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Run a single base pattern ─────────────────────────────────────────────────

async def run_base_pattern(
    pattern_key: str,
    query: dict,
    budget: float = BUDGET_USD,
    retriever: str | None = None,
    run_tag: str = "",
) -> dict:
    """Run a base pattern on a single query. Returns result dict.

    When `retriever` is set, the experiment_id becomes `protocol_a_{retriever}_{pattern_key}`
    so Protocol A backend-comparison runs do not clobber canonical baseline checkpoints.
    When `run_tag` is set, it is appended to the experiment_id (e.g. variance runs).
    """
    import importlib
    mod = importlib.import_module(PATTERNS[pattern_key])

    if retriever == "oracle":
        experiment_id = f"oracle_{run_tag or 't1'}_{pattern_key}"
    elif retriever:
        experiment_id = f"protocol_a_{retriever}_{pattern_key}"
        if run_tag:
            experiment_id = f"{experiment_id}_{run_tag}"
    else:
        experiment_id = f"base_{pattern_key}"
        if run_tag:
            experiment_id = f"{experiment_id}_{run_tag}"
    query_id = query["id"]

    if retriever == "oracle":
        # Tell the OracleSearcher which query's frozen corpus to serve this run.
        import os as _os
        _os.environ["ORACLE_QUERY_ID"] = query_id

    print(f"  [{pattern_key}] Running on '{query_id}'...")
    t0 = time.time()

    try:
        report = await mod.run(query["query"], budget_usd=budget, query_id=query_id)
        elapsed = time.time() - t0

        result = {
            "status": "success",
            "experiment_id": experiment_id,
            "pattern": pattern_key,
            "query_id": query_id,
            "elapsed_seconds": elapsed,
            "total_tokens": report.total_tokens,
            "total_cost_usd": report.total_cost_usd,
            "sections": len(report.sections),
            "citations": len(report.citations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        save_checkpoint(experiment_id, query_id, result)
        save_result(experiment_id, query_id, report.full_text())

        print(f"  [{pattern_key}] OK — {elapsed:.0f}s, {report.total_tokens:,} tokens, "
              f"{len(report.sections)} sections, {len(report.citations)} citations")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "status": "error",
            "experiment_id": experiment_id,
            "pattern": pattern_key,
            "query_id": query_id,
            "elapsed_seconds": elapsed,
            "error": str(e)[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(experiment_id, query_id, result)
        print(f"  [{pattern_key}] FAILED — {elapsed:.0f}s — {str(e)[:100]}")
        return result


# ── Run a single ablation config ──────────────────────────────────────────────

async def run_ablation_experiment(
    config,
    query: dict,
    budget: float = BUDGET_USD,
) -> dict:
    """Run an ablation config on a single query. Returns result dict."""
    runner = AblationRunner(
        checkpoint_dir=CHECKPOINT_DIR / "ablations",
        budget_per_run=budget,
    )

    experiment_id = f"ablation_{config.id}"
    query_id = query["id"]

    print(f"  [{config.id}] Running on '{query_id}'...")
    t0 = time.time()

    try:
        abl_result = await runner.run_ablation(config, query["query"], query_id)
        elapsed = time.time() - t0

        result = {
            "status": abl_result.status,
            "experiment_id": experiment_id,
            "ablation_id": config.id,
            "base_pattern": config.base_pattern,
            "query_id": query_id,
            "elapsed_seconds": elapsed,
            "total_tokens": abl_result.total_tokens,
            "error": abl_result.error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save the report text if successful
        if abl_result.status == "success" and abl_result.report_text:
            save_result(experiment_id, query_id, abl_result.report_text)

        save_checkpoint(experiment_id, query_id, result)

        if abl_result.status == "success":
            print(f"  [{config.id}] OK — {elapsed:.0f}s, {abl_result.total_tokens:,} tokens")
        else:
            print(f"  [{config.id}] FAILED — {elapsed:.0f}s — {abl_result.error_message[:100]}")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "status": "error",
            "experiment_id": experiment_id,
            "ablation_id": config.id,
            "base_pattern": config.base_pattern,
            "query_id": query_id,
            "elapsed_seconds": elapsed,
            "error": str(e)[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(experiment_id, query_id, result)
        print(f"  [{config.id}] FAILED — {elapsed:.0f}s — {str(e)[:100]}")
        return result


# ── Main orchestration ────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Run all research experiments")
    parser.add_argument("--base-only", action="store_true", help="Only run base patterns")
    parser.add_argument("--ablations-only", action="store_true", help="Only run ablation sweeps")
    parser.add_argument("--query", type=str, help="Run only this query ID")
    parser.add_argument("--query-ids-file", type=str, default="",
                        help="Path to JSON file containing {'query_ids': [...]} — restricts run to that subset")
    parser.add_argument("--pattern", type=str, help="Run only this base pattern (e.g. p6)")
    parser.add_argument("--ablation", type=str, help="Run only this ablation ID")
    parser.add_argument("--ablation-queries", type=int, default=0,
                        help="Use N representative queries for ablations (default: 0=all 90 for full statistical power, 5=representative subset)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Skip completed runs (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-run completed cells instead of using checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--budget", type=float, default=BUDGET_USD, help="Budget per run (USD)")
    parser.add_argument("--retriever", choices=["bing", "tavily", "oracle"], default=None,
                        help="Override SEARCH_BACKEND for this run. Outputs land in "
                             "experiment_id 'protocol_a_{retriever}_p{N}' instead of 'base_p{N}' "
                             "to avoid clobbering the canonical Bing baseline.")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Suffix appended to experiment_id, used by E5 variance runs "
                             "(e.g. --run-tag v1 → 'base_p4_v1'). Combinable with --retriever.")
    parser.add_argument("--p11-max-turns", type=int, default=0,
                        help="Override the P11 ReAct turn budget for budget-sensitivity runs. "
                             "Use with --pattern p11 --run-tag 16turn to write base_p11_16turn.")
    parser.add_argument("--p12-lora-adapter", type=str, default="",
                        help="Override the P12 LoRA adapter path. Use with --pattern p12 --run-tag v3 "
                             "to evaluate a retry without overwriting base_p12.")
    args = parser.parse_args()

    # Apply retriever override: must mutate config BEFORE any pattern import
    if args.retriever:
        import os as _os
        _os.environ["SEARCH_BACKEND"] = args.retriever
        import deep_research.config as _cfg
        _cfg.SEARCH_BACKEND = args.retriever
        print(f"  retriever override: SEARCH_BACKEND={args.retriever}")
    if args.run_tag:
        print(f"  run-tag suffix: {args.run_tag}")
    if args.p11_max_turns:
        import os as _os
        _os.environ["P11_MAX_TURNS"] = str(args.p11_max_turns)
        print(f"  P11 turn override: P11_MAX_TURNS={args.p11_max_turns}")
    if args.p12_lora_adapter:
        import os as _os
        _os.environ["P12_LORA_ADAPTER_PATH"] = args.p12_lora_adapter
        print(f"  P12 adapter override: P12_LORA_ADAPTER_PATH={args.p12_lora_adapter}")

    resume = args.resume
    if args.query_ids_file:
        ids = json.loads(Path(args.query_ids_file).read_text())["query_ids"]
        with open("data/eval_queries_v2.json") as f:
            all_qs = json.load(f)["queries"]
        by_id = {q["id"]: q for q in all_qs}
        queries = [by_id[qid] for qid in ids if qid in by_id]
        print(f"  query subset: {len(queries)} from {args.query_ids_file}")
    else:
        queries = load_queries(args.query)
    ablation_queries = load_queries(
        args.query,
        representative_only=(args.ablation_queries == 5),
    ) if not args.base_only else []
    # If ablation_queries not specified, use all queries for ablations too
    if not args.ablation_queries and not args.base_only:
        ablation_queries = queries

    print(f"{'='*60}")
    print(f"DEEP RESEARCH EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Queries:     {len(queries)}")
    print(f"Budget/run:  ${args.budget:.2f}")
    print(f"Resume:      {resume}")
    print(f"Run tag:     {args.run_tag or '(none)'}")
    print(f"Retriever:   {args.retriever or '(default config)'}")
    print(f"P11 turns:   {args.p11_max_turns or '(pattern default)'}")
    print(f"P12 adapter: {args.p12_lora_adapter or '(pattern default)'}")
    print()

    # ── Build experiment plan ──────────────────────────────────────────────
    plan = []

    # Base patterns
    if not args.ablations_only:
        patterns_to_run = [args.pattern] if args.pattern else PATTERN_ORDER
        for pattern_key in patterns_to_run:
            if pattern_key not in PATTERNS:
                print(f"ERROR: Unknown pattern '{pattern_key}'")
                sys.exit(1)
            for query in queries:
                # Match the experiment_id naming used by run_base_pattern when
                # --retriever / --run-tag are set, so resume skip behaves correctly
                # for Protocol A and variance runs.
                if args.retriever:
                    experiment_id = f"protocol_a_{args.retriever}_{pattern_key}"
                else:
                    experiment_id = f"base_{pattern_key}"
                if args.run_tag:
                    experiment_id = f"{experiment_id}_{args.run_tag}"
                if resume and is_completed(experiment_id, query["id"]):
                    continue
                plan.append(("base", pattern_key, query, None))

    # Ablation sweeps
    if not args.base_only:
        configs = ABLATION_REGISTRY
        if args.ablation:
            configs = [c for c in configs if c.id == args.ablation]
            if not configs:
                print(f"ERROR: Unknown ablation '{args.ablation}'")
                sys.exit(1)
        for config in configs:
            for query in ablation_queries:
                experiment_id = f"ablation_{config.id}"
                if resume and is_completed(experiment_id, query["id"]):
                    continue
                plan.append(("ablation", config.id, query, config))

    # Summary
    base_count = sum(1 for p in plan if p[0] == "base")
    ablation_count = sum(1 for p in plan if p[0] == "ablation")
    total = len(plan)

    print(f"Experiment plan:")
    print(f"  Base pattern runs: {base_count}")
    print(f"  Ablation sweep runs: {ablation_count}")
    print(f"  Total experiments: {total}")
    print()

    if args.dry_run:
        print("DRY RUN — experiments that would be executed:")
        for exp_type, exp_id, query, config in plan:
            print(f"  {exp_type}: {exp_id} × {query['id']}")
        return

    if total == 0:
        print("All experiments already completed! Use --no-resume to re-run.")
        return

    # ── Execute sequentially ───────────────────────────────────────────────
    overall_start = time.time()
    completed = 0
    errors = 0
    results = []

    for exp_type, exp_id, query, config in plan:
        completed += 1
        progress = f"[{completed}/{total}]"

        # Reset random seeds before each run for reproducibility
        set_reproducibility_seed(DEFAULT_SEED)

        if exp_type == "base":
            print(f"\n{progress} BASE PATTERN: {exp_id} × {query['id']}")
            result = await run_base_pattern(
                exp_id, query, budget=args.budget,
                retriever=args.retriever, run_tag=args.run_tag,
            )
        else:
            print(f"\n{progress} ABLATION: {exp_id} × {query['id']}")
            result = await run_ablation_experiment(config, query, budget=args.budget)

        results.append(result)
        if result["status"] != "success":
            errors += 1

        # Progress summary every 10 runs
        if completed % 10 == 0:
            elapsed = time.time() - overall_start
            avg = elapsed / completed
            remaining = avg * (total - completed)
            print(f"\n--- Progress: {completed}/{total} done, {errors} errors, "
                  f"avg {avg:.0f}s/run, ~{remaining/60:.0f}min remaining ---")

    # ── Final summary ──────────────────────────────────────────────────────
    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"EXPERIMENT RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Total:    {total} experiments")
    print(f"Success:  {total - errors}")
    print(f"Errors:   {errors}")
    print(f"Elapsed:  {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.1f} hours)")
    print()

    # Save run manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "total_experiments": total,
        "successes": total - errors,
        "errors": errors,
        "elapsed_seconds": overall_elapsed,
        "budget_per_run": args.budget,
        "query_count": len(queries),
        "query_ids": [q["id"] for q in queries],
        "resume": resume,
        "retriever": args.retriever,
        "run_tag": args.run_tag,
        "p11_max_turns": args.p11_max_turns or None,
        "p12_lora_adapter": args.p12_lora_adapter or None,
        "base_pattern_order": PATTERN_ORDER,
        "experiment_plan": [
            {
                "type": exp_type,
                "id": exp_id,
                "query_id": query["id"],
            }
            for exp_type, exp_id, query, _ in plan
        ],
        "results": results,
    }
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = RESULTS_DIR / f"run_manifest_{run_stamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    latest_path = RESULTS_DIR / "run_manifest.json"
    latest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Manifest saved to {manifest_path}")
    print(f"Latest manifest saved to {latest_path}")


if __name__ == "__main__":
    asyncio.run(main())
