#!/usr/bin/env python3
# =============================================================================
# wire_healthbench_into_judge_vs_human.py  (item A4_healthbench, Track A)
# -----------------------------------------------------------------------------
# Populate the HealthBench cells of canonical_numbers.json['judge_vs_human']
# ['by_family'] from the standalone GPT-5.2 HealthBench-vs-physician result that
# was produced TODAY by scripts/run_healthbench_judge.py.
#
# WHY THIS EXISTS
#   build_judge_vs_human.py rebuilds the WHOLE judge_vs_human block by scanning
#   per-family verdict directories (load_family_verdicts). The HealthBench panel
#   run does NOT write into those per-family dirs; its output is the standalone
#   results/healthbench_judge/{healthbench_judge_vs_physician.json,
#   healthbench_judge_verdicts.jsonl}. So every by_family.*.healthbench cell
#   still reads {"status":"awaiting_panel"} even though the gpt52 verdicts are
#   done. This script wires the standalone result into the canonical store as a
#   SURGICAL, IDEMPOTENT patch:
#       - by_family.gpt52.healthbench   -> {"status":"scored", "agreement":{...}, ...}
#       - by_family.local.healthbench   -> {"status":"pending_judge", ...}   (HONEST: not run)
#       - by_family.claude_sonnet.health-> {"status":"pending_judge", ...}   (HONEST: not run)
#       - by_family.claude_opus.health  -> left UNTOUCHED (HARD RULE: no new Opus judging)
#
#   It does NOT touch any other set/family cell, does NOT call any API, does NOT
#   regenerate the rest of judge_vs_human, and never clobbers an existing scored
#   block unless --force is given (resume-safe by default).
#
# AGREEMENT BLOCK SCHEMA (identical to the existing scored cells, e.g.
# by_family.agent.deepfactbench), so downstream readers need no code change:
#     agreement = {
#       n, n_clusters, human_pos_rate,
#       macro_f1        : {point, ci95, n_clusters},
#       point_biserial_r: {point, ci95, n_clusters},
#       auc             : {point, ci95, n_clusters},
#     }
#   Built from the on-disk per-pair verdicts (human = physician_consensus,
#   verdict = judge_label, cluster = prompt_id) with the SAME seeded cluster
#   bootstrap used by build_judge_vs_human.py (SEED=20260615, n_boot=2000), so
#   the point estimates reconcile with the standalone harness metrics and the
#   CIs are reproducible. The standalone harness summary (accuracy / cohen_kappa
#   / confusion / cost / endpoint / timestamp) is carried verbatim under
#   agreement_block['provenance'] so the cell is fully auditable.
#
# CANONICAL KEY EMITTED (the only keys this script writes):
#     judge_vs_human.by_family.gpt52.healthbench
#     judge_vs_human.by_family.local.healthbench
#     judge_vs_human.by_family.claude_sonnet.healthbench
#
# SAFETY / IDEMPOTENCY
#   - Atomic tmp-then-os.replace write, writing the WHOLE canonical back unchanged
#     except the three cells above.
#   - --resume (default): if gpt52.healthbench is already {"status":"scored"} with
#     the same n verdicts, exits 0 without rewriting (no-op). --force overrides.
#   - Deterministic: seeded numpy generator over SORTED clusters; running twice
#     yields byte-identical numbers.
#   - Pure read of the on-disk standalone result; no network, no model.
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# --- fixed paths (absolute; canonical lives at its MOVED location) -----------
REPO_ROOT = Path(".")
CANON_PATH = REPO_ROOT / "papers/paper_a_bounded_returns/analysis/canonical_numbers.json"
STANDALONE_SUMMARY = REPO_ROOT / "results/healthbench_judge/healthbench_judge_vs_physician.json"
STANDALONE_VERDICTS = REPO_ROOT / "results/healthbench_judge/healthbench_judge_verdicts.jsonl"

# Match build_judge_vs_human.py exactly so the bootstrap CIs are reproducible and
# consistent with the rest of the judge_vs_human block.
SEED = 20260615
N_BOOT = 2000
SET_NAME = "healthbench"
GPT52_FAMILY = "gpt52"
# Families we mark pending HONESTLY (judge not run; NOT awaiting an Opus panel).
# Opus is deliberately excluded per the HARD RULE (no new Opus judging anywhere).
PENDING_FAMILIES = ["local", "claude_sonnet"]


# =============================================================================
# stats kernels — copied verbatim from build_judge_vs_human.py so the numbers
# are identical in form (seeded cluster bootstrap over SORTED unique clusters).
# =============================================================================
def auc(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    vals = np.concatenate([pos, neg])
    s = np.argsort(vals, kind="mergesort")
    sv = vals[s]
    ranks = np.empty(len(vals), dtype=float)
    ranks[s] = np.arange(1, len(vals) + 1, dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[s[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return round(float(u / (len(pos) * len(neg))), 4)


def point_biserial(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(np.unique(y)) < 2 or np.std(x) == 0:
        return None
    return round(float(np.corrcoef(y, x)[0, 1]), 4)


def macro_f1(y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    if len(y) == 0:
        return None
    f1s = []
    for cls in (0, 1):
        tp = int(np.sum((yhat == cls) & (y == cls)))
        fp = int(np.sum((yhat == cls) & (y != cls)))
        fn = int(np.sum((yhat != cls) & (y == cls)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return round(float(np.mean(f1s)), 4)


def cluster_bootstrap(
    rows: list[dict], stat: Callable[[np.ndarray, np.ndarray], Optional[float]],
    y_key: str, x_key: str, cluster_ids: list[str], rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Optional[dict]:
    """Seeded cluster bootstrap CI for stat(y, x). Resamples whole clusters
    (HealthBench prompts) so within-prompt dependence is not under-counted.
    Deterministic: iterates over SORTED unique clusters."""
    y = np.array([r[y_key] for r in rows], dtype=float)
    x = np.array([r[x_key] for r in rows], dtype=float)
    obs = stat(y, x)
    if obs is None:
        return None
    uniq = sorted(set(cluster_ids))
    if len(uniq) < 2:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    by_c: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        by_c[c].append(i)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx: list[int] = []
        for k in pick:
            idx.extend(by_c[uniq[k]])
        v = stat(y[idx], x[idx])
        if v is not None:
            boots.append(v)
    if not boots:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": round(float(obs), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_clusters": len(uniq)}


# =============================================================================
# load the standalone on-disk result (pure read, no API)
# =============================================================================
def load_verdict_rows() -> list[dict]:
    """One row per (completion, criterion) consensus pair:
       human  = physician_consensus  (binary; clustered MET/NOT_MET majority)
       verdict= judge_label          (GPT-5.2 binary verdict)
       cluster= prompt_id            (the HealthBench prompt = meta.pair_key anchor)
    """
    rows: list[dict] = []
    with open(STANDALONE_VERDICTS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            human = int(r["physician_consensus"])
            verdict = int(r["judge_label"])
            rows.append({
                "item_id": r["pair_id"],
                "human_bin": human,
                "score": float(verdict),       # binary verdict used as the score
                "score_bin": verdict,
                "cluster": str(r["prompt_id"]),
            })
    return rows


def build_agreement_block(rows: list[dict], rng: np.random.Generator) -> dict:
    clusters = [r["cluster"] for r in rows]
    return {
        "n": len(rows),
        "n_clusters": len(set(clusters)),
        "human_pos_rate": round(float(np.mean([r["human_bin"] for r in rows])), 4),
        "macro_f1": cluster_bootstrap(rows, macro_f1, "human_bin", "score_bin", clusters, rng),
        "point_biserial_r": cluster_bootstrap(rows, point_biserial, "human_bin", "score", clusters, rng),
        "auc": cluster_bootstrap(rows, auc, "human_bin", "score", clusters, rng),
    }


def build_provenance(summary: dict) -> dict:
    """Carry the standalone harness summary through verbatim for auditability."""
    m = summary.get("metrics_judge_vs_physician", {})
    return {
        "source_result": str(STANDALONE_SUMMARY.relative_to(REPO_ROOT)),
        "source_verdicts": str(STANDALONE_VERDICTS.relative_to(REPO_ROOT)),
        "harness": summary.get("harness"),
        "judge_model": summary.get("judge_model"),
        "judge_endpoint": summary.get("judge_endpoint"),
        "timestamp_utc": summary.get("timestamp_utc"),
        "consensus_policy": summary.get("consensus_policy"),
        "sampling_plan": summary.get("sampling_plan"),
        "harness_accuracy": m.get("accuracy"),
        "harness_macro_f1": m.get("macro_f1"),
        "harness_auc": m.get("auc"),
        "harness_cohen_kappa": m.get("cohen_kappa"),
        "harness_confusion": m.get("confusion"),
        "harness_ci95": {
            "macro_f1": m.get("macro_f1_ci95"),
            "accuracy": m.get("accuracy_ci95"),
            "auc": m.get("auc_ci95"),
        },
        "cost_estimate_usd": (summary.get("cost_estimate") or {}).get("total_cost_usd"),
    }


# =============================================================================
# atomic, surgical canonical patch
# =============================================================================
def patch_canonical(write: bool, force: bool) -> dict:
    if not CANON_PATH.exists():
        sys.exit(f"FATAL: canonical not found at {CANON_PATH}")
    if not STANDALONE_SUMMARY.exists() or not STANDALONE_VERDICTS.exists():
        sys.exit("FATAL: standalone HealthBench result not on disk "
                 f"({STANDALONE_SUMMARY} / {STANDALONE_VERDICTS})")

    cn = json.load(open(CANON_PATH))
    jvh = cn.get("judge_vs_human")
    if not isinstance(jvh, dict) or "by_family" not in jvh:
        sys.exit("FATAL: judge_vs_human.by_family missing — run build_judge_vs_human.py first")
    by_family = jvh["by_family"]

    rows = load_verdict_rows()
    summary = json.load(open(STANDALONE_SUMMARY))
    n_now = len(rows)

    # --- resume guard: already wired with the same n verdicts -> no-op --------
    existing = by_family.get(GPT52_FAMILY, {}).get(SET_NAME, {})
    if (not force and existing.get("status") == "scored"
            and (existing.get("agreement") or {}).get("n") == n_now):
        report = {
            "action": "noop_already_wired",
            "n_verdicts": n_now,
            "gpt52_macro_f1": (existing.get("agreement") or {}).get("macro_f1"),
        }
        return report

    rng = np.random.default_rng(SEED)
    agreement = build_agreement_block(rows, rng)
    provenance = build_provenance(summary)

    gpt52_cell = {
        "status": "scored",
        "agreement": agreement,
        "provenance": provenance,
        "note": "GPT-5.2 vs balanced HealthBench physician-consensus pairs "
                "(human=physician_consensus, verdict=judge_label, cluster=prompt_id). "
                "Wired from the standalone run_healthbench_judge.py result; CIs via the "
                "shared seeded cluster bootstrap (SEED=20260615, n_boot=2000).",
    }

    # honest pending markers: judge simply was not run on these families for
    # HealthBench. Explicitly NOT awaiting an Opus panel (Opus excluded by rule).
    pending_cell_template = {
        "status": "pending_judge",
        "reason": "HealthBench panel not yet run for this judge family; standalone "
                  "GPT-5.2 result is wired, the other families remain to be judged on "
                  "the same balanced consensus-pair sample.",
        "awaiting_opus": False,
    }

    by_family.setdefault(GPT52_FAMILY, {})[SET_NAME] = gpt52_cell
    for fam in PENDING_FAMILIES:
        cur = by_family.get(fam, {}).get(SET_NAME, {})
        # never clobber a real scored cell for a pending family (resume-safe)
        if cur.get("status") == "scored" and not force:
            continue
        by_family.setdefault(fam, {})[SET_NAME] = dict(pending_cell_template)

    # claude_opus.healthbench is intentionally left exactly as found (HARD RULE).

    if write:
        cn["judge_vs_human"]["by_family"] = by_family
        txt = json.dumps(cn, indent=1)
        tmp = str(CANON_PATH) + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(txt)
        os.replace(tmp, CANON_PATH)

    return {
        "action": "wired",
        "n_verdicts": n_now,
        "gpt52_cell": gpt52_cell,
        "pending_families": PENDING_FAMILIES,
        "opus_untouched": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rewrite even if the gpt52 healthbench cell is already scored")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the cell but do NOT write the canonical")
    args = ap.parse_args()

    report = patch_canonical(write=not args.dry_run, force=args.force)

    print("=" * 78)
    print("wire_healthbench_into_judge_vs_human")
    print("=" * 78)
    print(f"  canonical : {CANON_PATH}")
    print(f"  action    : {report['action']}")
    print(f"  n verdicts: {report.get('n_verdicts')}")
    if report["action"] == "wired":
        ag = report["gpt52_cell"]["agreement"]
        print(f"  gpt52.healthbench : status=scored  n={ag['n']}  clusters={ag['n_clusters']}  "
              f"human_pos_rate={ag['human_pos_rate']}")
        print(f"      macro_f1         = {ag['macro_f1']}")
        print(f"      point_biserial_r = {ag['point_biserial_r']}")
        print(f"      auc              = {ag['auc']}")
        print(f"  pending (honest, NOT awaiting Opus): {report['pending_families']}")
        print("  claude_opus.healthbench : UNTOUCHED (no new Opus judging)")
        if args.dry_run:
            print("  [dry-run] canonical NOT written")
    elif report["action"] == "noop_already_wired":
        print("  already wired with the same n verdicts; nothing to do (use --force to rewrite)")
    print("=" * 78)


if __name__ == "__main__":
    main()
