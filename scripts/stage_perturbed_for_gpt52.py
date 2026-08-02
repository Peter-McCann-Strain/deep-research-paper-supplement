#!/usr/bin/env python3
"""Stage the E13' perturbed reports into the per-pattern tree the namespaced GPT-5.2
judge reads via JUDGE_RESULTS_BASE.

The perturbed reports are FLAT:  reports/perturbation_set/perturbed/<pattern>__<qid>.md
The namespaced judge reads:      <JUDGE_RESULTS_BASE>/<pattern>/<qid>.md   (stem == qid,
which must be present in data/eval_queries_v2.json so the same rubric is reused).

This builds, by SYMLINK (no copy, no mutation of the perturbed corpus), the tree
    reports/perturbation_set/_judge_stage/<pattern>/<qid>.md
so the perturbed report is judged with the SAME query/rubric as its clean original.
Idempotent and clobber-safe: only creates symlinks under the NEW _judge_stage dir,
never writes to results/judge_gpt52, results/experiments, or the perturbed corpus.

    python scripts/stage_perturbed_for_gpt52.py --dry-run   # plan only, no writes
    python scripts/stage_perturbed_for_gpt52.py             # build the symlink stage
The script prints the patterns present and the exact JUDGE_RESULTS_BASE to export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(".")
PERTURBED = ROOT / "reports" / "perturbation_set" / "perturbed"
STAGE = ROOT / "reports" / "perturbation_set" / "_judge_stage"
EVAL_QUERIES = ROOT / "data" / "eval_queries_v2.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="plan only, no symlinks")
    args = ap.parse_args()

    qids = {q["id"] for q in json.loads(EVAL_QUERIES.read_text())["queries"]}
    perturbed = sorted(p for p in PERTURBED.glob("*.md"))
    if not perturbed:
        print(f"no perturbed reports under {PERTURBED}", file=sys.stderr)
        return 2

    patterns = set()
    plan = []
    skipped = []
    for src in perturbed:
        stem = src.stem  # <pattern>__<qid>
        if "__" not in stem:
            skipped.append((src.name, "no '__' separator"))
            continue
        pattern, qid = stem.split("__", 1)
        if qid not in qids:
            skipped.append((src.name, f"qid {qid} not in manifest"))
            continue
        patterns.add(pattern)
        plan.append((pattern, qid, src))

    print(f"perturbed reports : {len(perturbed)}")
    print(f"stageable (qid in manifest): {len(plan)}")
    print(f"patterns          : {','.join(sorted(patterns))}")
    if skipped:
        print(f"skipped           : {len(skipped)} -> {skipped[:5]}")
    print(f"stage root        : {STAGE}")
    print(f"export for judge  : JUDGE_RESULTS_BASE={STAGE}")
    print(f"--patterns-raw    : {','.join(sorted(patterns))}")

    if args.dry_run:
        print("[dry-run] no symlinks created.")
        return 0

    n = 0
    for pattern, qid, src in plan:
        dst_dir = STAGE / pattern
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{qid}.md"
        if dst.is_symlink() or dst.exists():
            continue  # idempotent: never clobber an existing link
        dst.symlink_to(src.resolve())
        n += 1
    print(f"[done] created {n} new symlinks under {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
