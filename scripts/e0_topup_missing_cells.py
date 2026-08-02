#!/usr/bin/env python3
"""E0: Per-pattern completion top-up.

Re-runs the 7 missing (pattern × query) cells from the original 990-cell matrix.
All 7 failed for known infrastructure reasons (Azure content-filter or JSON
validation), not model-quality reasons. Re-running can recover ~30-50% on
re-attempt; persistent failures are transparently reported.

Outputs:
  - checkpoints/experiments/{pattern}/{query_id}.json (overwritten if recovered)
  - results/experiments/{pattern}/{query_id}.md (if recovered)
  - reports/phase8_topup/recovery_log.json (recovery status per cell)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PATTERNS = {
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p3": "deep_research.patterns.p3_meridian.pipeline",
    "p5": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6": "deep_research.patterns.p6_reactive_interleaved.pipeline",
}

MISSING_CELLS = [
    ("p1", "8e99d8d2-f6b9-4800-83a9-6f56829898fe"),
    ("p3", "82508e50-497c-445a-b1dd-fd9d7e6dafda"),
    ("p3", "b3c576e7-dfc6-403f-90e7-53c011884d5c"),
    ("p5", "82508e50-497c-445a-b1dd-fd9d7e6dafda"),
    ("p6", "0a652d00-5c22-4621-8ec4-dd92b1f1450b"),
    ("p6", "82508e50-497c-445a-b1dd-fd9d7e6dafda"),
    ("p6", "dsqa_0683"),
]

BUDGET_USD = 2.0
CHECKPOINT_DIR = Path("checkpoints/experiments")
RESULTS_DIR = Path("results/experiments")
LOG_DIR = Path("reports/phase8_topup")


def load_queries() -> dict[str, dict]:
    with open("data/eval_queries_v2.json") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["queries"]}


def cp_path(pattern: str, qid: str) -> Path:
    return CHECKPOINT_DIR / f"base_{pattern}" / f"{qid}.json"


def res_path(pattern: str, qid: str) -> Path:
    return RESULTS_DIR / f"base_{pattern}" / f"{qid}.md"


async def retry_cell(pattern: str, query: dict) -> dict:
    """Retry a single cell. Returns recovery status dict."""
    import importlib
    mod = importlib.import_module(PATTERNS[pattern])
    qid = query["id"]
    print(f"  [base_{pattern}] retrying {qid} ({query.get('source')}, {query.get('difficulty')})", flush=True)
    t0 = time.time()
    try:
        report = await mod.run(query["query"], budget_usd=BUDGET_USD)
        elapsed = time.time() - t0
        # Overwrite checkpoint with success
        cp = cp_path(pattern, qid)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({
            "status": "success",
            "experiment_id": f"base_{pattern}",
            "pattern": pattern,
            "query_id": qid,
            "elapsed_seconds": elapsed,
            "total_tokens": report.total_tokens,
            "total_cost_usd": report.total_cost_usd,
            "sections": len(report.sections),
            "citations": len(report.citations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recovered_in_e0": True,
        }, indent=2, default=str))
        # Save full result
        rp = res_path(pattern, qid)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report.full_text())
        print(f"  [base_{pattern}/{qid}] RECOVERED in {elapsed:.0f}s  ({len(report.citations)} citations)", flush=True)
        return {"status": "recovered", "pattern": pattern, "query_id": qid, "elapsed_seconds": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        err_str = str(e)[:300]
        print(f"  [base_{pattern}/{qid}] STILL FAILED in {elapsed:.0f}s: {err_str[:100]}", flush=True)
        return {"status": "still_failed", "pattern": pattern, "query_id": qid, "error": err_str, "elapsed_seconds": elapsed}


async def main():
    queries = load_queries()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {"started": datetime.now(timezone.utc).isoformat(), "cells": []}

    print(f"E0: attempting recovery on {len(MISSING_CELLS)} missing cells")
    print()

    # Run sequentially through PTU rate gate (each pattern call serializes naturally)
    for pattern, qid in MISSING_CELLS:
        if qid not in queries:
            print(f"  [base_{pattern}/{qid}] QUERY NOT IN MANIFEST — skipping")
            log["cells"].append({"status": "missing_from_manifest", "pattern": pattern, "query_id": qid})
            continue
        try:
            result = await retry_cell(pattern, queries[qid])
        except Exception as e:
            result = {"status": "exception", "pattern": pattern, "query_id": qid, "error": str(e)[:300]}
            print(f"  [base_{pattern}/{qid}] OUTER EXCEPTION: {str(e)[:120]}", flush=True)
        log["cells"].append(result)

    log["finished"] = datetime.now(timezone.utc).isoformat()
    log["summary"] = {
        "total": len(MISSING_CELLS),
        "recovered": sum(1 for c in log["cells"] if c["status"] == "recovered"),
        "still_failed": sum(1 for c in log["cells"] if c["status"] == "still_failed"),
        "missing_from_manifest": sum(1 for c in log["cells"] if c["status"] == "missing_from_manifest"),
    }
    (LOG_DIR / "recovery_log.json").write_text(json.dumps(log, indent=2, default=str))
    print()
    print("=" * 60)
    print(f"E0 complete: {log['summary']['recovered']}/{log['summary']['total']} recovered")
    print(f"Log: {LOG_DIR / 'recovery_log.json'}")


if __name__ == "__main__":
    asyncio.run(main())
