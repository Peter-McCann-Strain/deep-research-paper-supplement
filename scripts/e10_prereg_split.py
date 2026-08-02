#!/usr/bin/env python3
"""E10 noise-RL — PRE-REGISTER the train/eval split (run BEFORE any GRPO run).

Plan §43 / the prereg require the train/held-out split to be committed BEFORE
the first run so the held-out judge + judge-free objective are not chosen after
seeing results. This script:

  * deterministically partitions the 90 eval_queries_v2 by STRATIFIED difficulty
    (5 simple / 25 moderate / 60 complex) with a FIXED seed into
        train  (rollout prompts for GRPO)
        eval   (held-out: judge-free objective endpoint + final GPT-5.2 judge),
  * FORCES the answer-checkable slice (every query with a non-empty
    reference_answer OR expected_elements) into the EVAL split, so the
    judge-free metric is only ever computed on held-out items,
  * writes data/e10_split.json (sorted, seeded, with a content hash),
  * prints the content hash to paste into docs/publication/prereg/prereg_E10.md.

CPU-only, no model, no paid API, no canonical write. Idempotent: re-running with
the same seed reproduces byte-identical output (atomic tmp+replace write).

    [ -f venv/bin/activate ] && source venv/bin/activate && python scripts/e10_prereg_split.py
    python scripts/e10_prereg_split.py --dry-run     # print, write nothing
    python scripts/e10_prereg_split.py --eval-frac 0.40 --seed 20260623
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _stable_offset(s: str) -> int:
    """Process-stable integer offset from a string (Python's hash() is salted
    per-process via PYTHONHASHSEED, so we use sha256 for reproducibility)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16) % 9973

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "data/eval_queries_v2.json"
SPLIT_OUT = REPO_ROOT / "data/e10_split.json"

# Fixed prereg seed. Distinct from E7's 20260613 to avoid accidental coupling.
DEFAULT_SEED = 20260623
DEFAULT_EVAL_FRAC = 0.40  # held-out fraction WITHIN each difficulty stratum


def _is_answer_checkable(q: dict) -> bool:
    ref = q.get("reference_answer")
    exp = q.get("expected_elements")
    return bool((isinstance(ref, str) and ref.strip())
                or (isinstance(exp, list) and exp))


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_split(queries: list[dict], seed: int, eval_frac: float) -> dict:
    import numpy as np

    # Stratify by difficulty; sort within stratum for determinism.
    strata: dict[str, list[dict]] = {}
    for q in sorted(queries, key=lambda r: r["id"]):
        strata.setdefault(q.get("difficulty", "moderate"), []).append(q)

    train_ids: list[str] = []
    eval_ids: list[str] = []
    forced_eval: list[str] = []

    for diff in sorted(strata):
        items = strata[diff]
        qids = [q["id"] for q in items]
        ac = {q["id"] for q in items if _is_answer_checkable(q)}
        forced_eval.extend(sorted(ac))

        # remaining (non-answer-checkable) items in this stratum get split
        rest = [qid for qid in qids if qid not in ac]
        rng = np.random.default_rng(seed + _stable_offset(diff))
        # how many EVAL items total we want from this stratum
        n_eval_target = int(round(eval_frac * len(qids)))
        # answer-checkable already in eval; top up from `rest` if needed
        need = max(0, n_eval_target - len(ac))
        order = sorted(rest)  # deterministic base order
        perm = rng.permutation(len(order))
        ordered = [order[i] for i in perm]
        extra_eval = ordered[:need]
        stratum_train = ordered[need:]

        eval_ids.extend(sorted(ac) + sorted(extra_eval))
        train_ids.extend(sorted(stratum_train))

    train_ids = sorted(set(train_ids))
    eval_ids = sorted(set(eval_ids))
    assert not (set(train_ids) & set(eval_ids)), "train/eval overlap"
    assert set(train_ids) | set(eval_ids) == {q["id"] for q in queries}, \
        "split does not cover all queries"

    payload_core = {
        "_what": "E10 noise-RL pre-registered train/eval split (committed before first run).",
        "seed": seed,
        "eval_frac": eval_frac,
        "n_total": len(queries),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "n_forced_eval_answer_checkable": len(sorted(set(forced_eval))),
        "stratification": "by difficulty; answer-checkable queries FORCED into eval",
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "forced_eval_answer_checkable_ids": sorted(set(forced_eval)),
    }
    payload_core["content_hash"] = _content_hash(payload_core)
    return payload_core


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".e10split.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--out", type=Path, default=SPLIT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    queries = json.loads(args.queries.read_text())["queries"]
    payload = build_split(queries, args.seed, args.eval_frac)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    print(f"E10 prereg split  seed={args.seed}  eval_frac={args.eval_frac}")
    print(f"  n_total={payload['n_total']}  n_train={payload['n_train']}  "
          f"n_eval={payload['n_eval']}  "
          f"forced_eval(answer-checkable)={payload['n_forced_eval_answer_checkable']}")
    print(f"  content_hash = {payload['content_hash']}")
    print(f"  -> paste this hash into docs/publication/prereg/prereg_E10.md")

    if args.dry_run:
        print("  [dry-run] nothing written.")
        return 0
    _atomic_write(args.out, text)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
