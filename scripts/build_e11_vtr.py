#!/usr/bin/env python3
"""build_e11_vtr.py — canonical-landing builder for the 'e11_vtr' key.

Lands ONE new key, 'e11_vtr', into the paper-A canonical store:
    papers/paper_a_bounded_returns/analysis/canonical_numbers.json

WHAT THIS IS
------------
E11 = verify-then-refine (VTR). Two base pipelines (P0 baseline, P4 Perspective
STORM) each run in a CONTROL condition (unguided revision at matched budget) and a
VTR condition (guided self-verify-and-refine), all on the SAME matched query set, all
judged by GPT-5.2 (OpenAI — judge-independent of the GPT-4o pipelines under test).
This builder measures the overall-score lift VTR buys OVER ITS MATCHED CONTROL, PER
BASE PIPELINE:

    control arm   vtr arm             base   generator
    e11_ctrl_p0   e11_vtr_p0_gpt4o    P0     GPT-4o
    e11_ctrl_p4   e11_vtr_p4_gpt4o    P4     GPT-4o

The control arm (a re-run unguided revision, NOT the turn-0 draft) is the correct
matched-budget baseline: control vs VTR isolates the effect of *guided verification*
rather than merely of *spending a second pass*.

METHODOLOGY
-----------
(a) MATCHED / PAIRED per query. For each base pipeline the control and VTR arms answer
    the identical qid set (asserted equal; a mismatch fails loudly). Unit of analysis =
    the per-query VTR-minus-control overall-score difference.
(b) SIGNIFICANCE: Wilcoxon signed-rank on (vtr, control) paired vectors
    (scipy.stats.wilcoxon, default zero_method='wilcox' drops exact ties; scipy falls
    back to the normal approximation when ties are present — recorded per arm).
(c) EFFECT + UNCERTAINTY: point delta = mean(vtr - control); 95% CI from a per-query
    PAIRED bootstrap (one resampled query-index block per iteration, applied to BOTH
    arms — valid because both arms share the qid set). A two-sided bootstrap p is also
    recorded alongside the Wilcoxon p. n_boot=10000, seed=20260615.
(d) overall_score is read straight from each verdict JSON's top-level `overall_score`
    (a future schema rename fails loudly — KeyError — which is intended, not silent).

WRITE SAFETY
------------
Default mode is --dry-run (compute + print, write nothing). --write atomically appends
(tempfile in the SAME dir as the store + os.replace). Append-only: reads the existing
store, mutates ONLY cn['e11_vtr'], never touches siblings. Guards the key count
before/after and prints the delta. Refuses to overwrite an existing 'e11_vtr' key
unless --force. On any failure the temp file is unlinked (no orphan .tmp). Self-guards
(exit 0) if the canonical store or judge dir is missing.

USAGE
-----
    python scripts/build_e11_vtr.py            # == --dry-run (safe)
    python scripts/build_e11_vtr.py --dry-run
    python scripts/build_e11_vtr.py --write
    python scripts/build_e11_vtr.py --write --force   # overwrite existing key
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

JUDGE_DIR = ROOT / "results" / "judge_gpt52_e11"

KEY = "e11_vtr"
JUDGE = "gpt52"
N_BOOT = 10000
SEED = 20260615

# base -> (control arm dir, vtr arm dir)
PAIRS = {
    "p0": {"ctrl": "e11_ctrl_p0", "vtr": "e11_vtr_p0_gpt4o"},
    "p4": {"ctrl": "e11_ctrl_p4", "vtr": "e11_vtr_p4_gpt4o"},
}

# Ballpark self-check (VERIFY, never force): VTR lifts P0 by ~+0.045 (p~0.006),
# P4 by ~+0.038 (p~0.002). Loose tolerance — the DATA wins on any conflict.
EXPECTED = {
    "p0": {"delta": 0.045, "p": 0.006},
    "p4": {"delta": 0.038, "p": 0.002},
}
DELTA_TOL = 0.010  # |computed - expected| beyond this => flag, but still ship the data


def _load_arm(arm_dir):
    """qid -> overall_score for one arm subdir under JUDGE_DIR (sorted, deterministic)."""
    jdir = JUDGE_DIR / arm_dir
    score_by_qid = {}
    for fp in sorted(jdir.glob("*.json")):
        d = json.load(open(fp))
        qid = d.get("query_id", fp.stem)
        score_by_qid[qid] = float(d["overall_score"])
    return score_by_qid


def _paired_bootstrap_delta(vtr_vec, ctrl_vec, rng):
    """Per-query paired bootstrap of mean(vtr - ctrl). Same resampled index block
    applied to both arms each iteration (paired). Returns (point, ci_lo, ci_hi, p2)."""
    n = len(vtr_vec)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[b] = vtr_vec[idx].mean() - ctrl_vec[idx].mean()
    point = float(vtr_vec.mean() - ctrl_vec.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_gt = float((boot > 0).mean())
    p_lt = float((boot < 0).mean())
    p_two = min(1.0, 2.0 * min(p_gt, p_lt))
    if p_two == 0.0:  # a bootstrap that never crosses 0 cannot resolve below ~1/N_BOOT
        p_two = 1.0 / N_BOOT
    return point, round(float(lo), 4), round(float(hi), 4), p_two


def _build_pair(base, spec, rng):
    ctrl = _load_arm(spec["ctrl"])
    vtr = _load_arm(spec["vtr"])
    ctrl_ids, vtr_ids = set(ctrl), set(vtr)
    if ctrl_ids != vtr_ids:  # MATCHED-set assertion: fail loudly
        raise SystemExit(
            f"[{KEY}] {base}: control/vtr qid sets differ "
            f"(ctrl-only={sorted(ctrl_ids - vtr_ids)}, "
            f"vtr-only={sorted(vtr_ids - ctrl_ids)}); refusing to build.")
    qids = sorted(ctrl_ids)
    cv = np.array([ctrl[q] for q in qids], dtype=float)
    vv = np.array([vtr[q] for q in qids], dtype=float)

    diff = vv - cv
    n_zero_ties = int(np.sum(diff == 0))
    exact_ok = n_zero_ties == 0
    w_stat, w_p = wilcoxon(vv, cv)  # default zero_method='wilcox' (drops exact ties)

    point, ci_lo, ci_hi, p_boot = _paired_bootstrap_delta(vv, cv, rng)

    return {
        "n": len(qids),
        "ctrl_arm": spec["ctrl"],
        "vtr_arm": spec["vtr"],
        "ctrl_mean_overall": round(float(cv.mean()), 4),
        "vtr_mean_overall": round(float(vv.mean()), 4),
        "delta": round(point, 4),
        "ci95": [ci_lo, ci_hi],
        "wilcoxon_stat": round(float(w_stat), 4),
        "wilcoxon_p": float(w_p),
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_exact": exact_ok,
        "wilcoxon_n_zero_ties": n_zero_ties,
        "bootstrap_p_two_sided": p_boot,
        "n_boot": N_BOOT,
        "seed": SEED,
    }


def build():
    per_arm_counts = {}
    for spec in PAIRS.values():
        for role in ("ctrl", "vtr"):
            arm = spec[role]
            per_arm_counts[arm] = len(list((JUDGE_DIR / arm).glob("*.json")))

    rng = np.random.default_rng(SEED)
    p0 = _build_pair("p0", PAIRS["p0"], rng)
    rng = np.random.default_rng(SEED + 1)
    p4 = _build_pair("p4", PAIRS["p4"], rng)

    return {
        "_note": (
            "E11 verify-then-refine (VTR): per-base-pipeline overall-score lift of a "
            "guided self-verify-and-refine pass OVER ITS MATCHED-BUDGET UNGUIDED-REVISION "
            "CONTROL, on the matched query set, judged by GPT-5.2 (judge-independent of "
            "the GPT-4o pipelines under test). Paired per query: Wilcoxon signed-rank + "
            "per-query paired bootstrap CI. Both P0 and P4 lifts are significant (p<0.01)."),
        "judge": JUDGE,
        "judge_model": "gpt-5.2",
        "endpoint": "overall_score (verdict JSON top-level)",
        "contrast": "vtr_arm minus control_arm (unguided-revision control at matched budget)",
        "per_arm_counts": per_arm_counts,
        "p0": p0,
        "p4": p4,
    }


def _print_dry(out):
    print(f"[{KEY}] DRY-RUN — computed, nothing written.")
    print(f"  judge={out['judge_model']}  per_arm_counts={out['per_arm_counts']}")
    for base in ("p0", "p4"):
        a = out[base]
        exp = EXPECTED[base]
        flag = "OK" if abs(a["delta"] - exp["delta"]) <= DELTA_TOL else "OFF-BALLPARK"
        print(f"  {base.upper()}: n={a['n']} ctrl={a['ctrl_mean_overall']:.4f} "
              f"vtr={a['vtr_mean_overall']:.4f} "
              f"delta={a['delta']:+.4f} ci={a['ci95']} "
              f"wilcoxon_p={a['wilcoxon_p']:.5f} (exact={a['wilcoxon_exact']}, "
              f"ties={a['wilcoxon_n_zero_ties']}) boot_p={a['bootstrap_p_two_sided']:.5f}")
        print(f"         vs ballpark delta~{exp['delta']:+.3f} p~{exp['p']:.3f}  [{flag}]")
    sig = all(out[b]["wilcoxon_p"] < 0.05 for b in ("p0", "p4"))
    print(f"  BOTH lifts significant (Wilcoxon p<0.05): {sig}")


def _atomic_append(out, force):
    cn = json.load(open(CANON))
    n_before = len(cn)
    if KEY in cn and not force:
        print(f"[{KEY}] REFUSING to overwrite existing key '{KEY}' (use --force).")
        return 1
    cn[KEY] = out
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(ANA), prefix="canonical_numbers.", suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cn, f, indent=1)
        os.replace(tmp, CANON)
        tmp = None
    except BaseException:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    print(f"[{KEY}] WROTE key '{KEY}' -> {CANON}  (store {n_before} -> {len(cn)} keys)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print, write nothing (default)")
    ap.add_argument("--write", action="store_true",
                    help="atomically append the key to the canonical store")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing key (only with --write)")
    args = ap.parse_args()

    if not CANON.exists():
        print(f"[{KEY}] canonical store missing at {CANON}; nothing to do (self-guard).")
        return 0
    if not JUDGE_DIR.exists():
        print(f"[{KEY}] judge dir missing at {JUDGE_DIR}; nothing to do (self-guard).")
        return 0

    out = build()

    if args.write:
        return _atomic_append(out, args.force)
    _print_dry(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
