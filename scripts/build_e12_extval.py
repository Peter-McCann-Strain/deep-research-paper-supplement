#!/usr/bin/env python3
"""build_e12_extval.py — canonical-landing builder for the 'e12_extval' key.

Lands ONE new key, 'e12_extval', into the paper-A canonical store:
    papers/paper_a_bounded_returns/analysis/canonical_numbers.json

WHAT THIS IS
------------
E12 EXTERNAL VALIDATION: does the frontier-vs-cluster finding (P1/P4 above P0)
replicate on a FRESH 600-report battery? Generation = GPT-4o (deployment
sthree-ptu-02), judge = GPT-5.2 (independent). The battery is 5 query families x
{p0,p1,p4} x 40 reports = 600 verdicts under results/e12_extval/judge_gpt52/.

This builder:
  - folds the pre-computed concordance test outcomes from
    results/e12_extval/concordance_results.json
    (test_a_rank_concordance, test_b_flat_top_cluster, test_c_p0_7b_tiering,
     drb_race_layer),
  - INDEPENDENTLY recomputes the 600-verdict per-pattern + per-subdir means from the
    raw verdict `overall_score` values (sorted dir walk, np.mean) as a self-consistency
    cross-check against the concordance file (match within 5e-4), and
  - carries the DRB-RACE 4th layer through verbatim as status=BLOCKED, so the
    replication claim is recorded as PARTIAL (3 of 4 layers computed).

WRITE SAFETY
------------
Default mode is --dry-run (compute + print, write nothing). --write atomically appends
(tempfile in the SAME dir as the store + os.replace). Append-only: reads the existing
store, mutates ONLY cn['e12_extval'], never touches siblings. Refuses to overwrite an
existing 'e12_extval' key unless --force. On any failure the temp file is unlinked (no
orphan .tmp). Self-guards (exit 0) if the canonical store OR the concordance file is
missing.

USAGE
-----
    python scripts/build_e12_extval.py            # == --dry-run (safe)
    python scripts/build_e12_extval.py --dry-run
    python scripts/build_e12_extval.py --write
    python scripts/build_e12_extval.py --write --force   # overwrite existing key
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(".")
# NEW canonical location (post-0a80ba6). Resolved from repo root, not the stale path.
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

E12_DIR = ROOT / "results" / "e12_extval"
CONCORDANCE = E12_DIR / "concordance_results.json"
JUDGE_DIR = E12_DIR / "judge_gpt52"

KEY = "e12_extval"
PATTERNS = ["p0", "p1", "p4"]
MATCH_TOL = 5e-4


def _recompute_means():
    """Walk the 600 verdicts (sorted) and recompute per-pattern + per-subdir means.

    Reads the verdict JSON top-level `overall_score` directly; a future schema rename
    fails loudly (KeyError), which is intended, not silent.
    """
    per_pattern = {p: [] for p in PATTERNS}
    per_subdir = {}
    for sub in sorted(JUDGE_DIR.glob("*/")):
        name = sub.name
        # subdir naming: <family>__<pattern>
        pat = name.split("__")[1]
        vals = []
        for fp in sorted(sub.glob("*.json")):
            d = json.load(open(fp))
            vals.append(float(d["overall_score"]))
        per_subdir[name] = {"n": len(vals), "mean": float(np.mean(vals))}
        per_pattern[pat].extend(vals)
    pat_means = {
        p: {"n": len(per_pattern[p]), "mean": float(np.mean(per_pattern[p]))}
        for p in PATTERNS
    }
    n_total = sum(per_pattern[p].__len__() for p in PATTERNS)
    return pat_means, per_subdir, n_total


def build():
    conc = json.load(open(CONCORDANCE))

    pat_means, per_subdir, n_total = _recompute_means()

    # --- self-consistency cross-check vs concordance file declared e12 means ---
    declared = conc["test_a_rank_concordance"]["e12_means"]
    recomputed = {p: pat_means[p]["mean"] for p in PATTERNS}
    max_abs_diff = max(abs(recomputed[p] - declared[p]) for p in PATTERNS)
    consistent = bool(max_abs_diff <= MATCH_TOL)

    ta = conc["test_a_rank_concordance"]
    tb = conc["test_b_flat_top_cluster"]
    tc = conc["test_c_p0_7b_tiering"]
    drb = conc["drb_race_layer"]

    out = {
        "_note": (
            "E12 EXTERNAL VALIDATION of the frontier-vs-cluster finding (P1/P4 above P0) "
            "on a FRESH 600-report battery (5 query families x {p0,p1,p4} x 40), "
            "generation GPT-4o (sthree-ptu-02), judge GPT-5.2 (independent). Folds the "
            "concordance test outcomes from concordance_results.json AND independently "
            "recomputes the 600-verdict per-pattern means from raw overall_score values "
            "(sorted dir walk) as a self-consistency cross-check. PARTIAL replication: 3 "
            "of 4 layers computed; the 4th (DeepResearch-Bench RACE) is carried through as "
            "status=BLOCKED (external annotations not on disk). rho=0.5 reflects a "
            "within-top-cluster P1/P4 order swap (n=3, ordinal only), not a tier flip; "
            "both P1 and P4 reproduce above P0."),
        "key_version": "1.0",
        "judge_model": conc["judge_model"],
        "generation_model": conc["generation_model"],
        "generation_deployment": conc["generation_deployment"],
        "n_reports_judged": n_total,
        "battery_design": {
            "n_query_families": 5,
            "patterns": PATTERNS,
            "per_family_per_pattern": 40,
            "subdirs": sorted(per_subdir.keys()),
        },
        "e12_means_recomputed_from_verdicts": {
            p: {"n": pat_means[p]["n"], "mean": pat_means[p]["mean"]} for p in PATTERNS
        },
        "e12_means_per_subdir": {
            k: per_subdir[k] for k in sorted(per_subdir.keys())
        },
        "consistency_with_concordance_file": {
            "declared_e12_means": declared,
            "recomputed_e12_means": recomputed,
            "max_abs_diff": max_abs_diff,
            "match_within_5e-4": consistent,
        },
        "test_a_rank_concordance": {
            "spearman_rho": ta["spearman_rho"],
            "n_patterns": ta["n_patterns"],
            "patterns": ta["patterns"],
            "main_lb_means": ta["main_lb_means"],
            "e12_means": ta["e12_means"],
            "p_value": ta.get("p_value"),
            "note": (
                "n=3 not powered; ordinal agreement only. rho=0.5 reflects a within-top-"
                "cluster P1/P4 order swap (main_lb has P4>=P1, e12 has P1>P4), not a tier "
                "flip."),
        },
        "test_b_flat_top_cluster": {
            "means": {p: tb[p] for p in PATTERNS},
            "p1_p4_gap": tb["p1_p4_gap"],
            "p1_above_p0": tb["p1_above_p0"],
            "p4_above_p0": tb["p4_above_p0"],
            "top_cluster_flat": tb["top_cluster_flat"],
            "survives": tb["survives"],
        },
        "test_c_p0_7b_tiering": {
            "p0_e12": round(float(tc["p0_e12"]), 4),
            "p9_main_lb": round(float(tc["p9_main_lb"]), 4),
            "p10_main_lb": round(float(tc["p10_main_lb"]), 4),
            "p0_above_p9": tc["p0_above_p9"],
            "p0_above_p10": tc["p0_above_p10"],
            "note": tc.get("note"),
        },
        "drb_race_layer": {
            "status": drb["status"],
            "reason": drb["reason"],
            "expected_local_path": drb["expected_local_path"],
            "present_on_disk": drb["present_on_disk"],
            "unblock_action": drb["unblock_action"],
        },
        "interpretation": (
            "PARTIAL external replication. Both P1 (0.486) and P4 (0.453) reproduce ABOVE "
            "P0 (0.183) on the fresh GPT-4o/GPT-5.2 battery, confirming the frontier-vs-"
            "cluster gap. test_a rho=0.5 is driven by a within-cluster P1/P4 swap (not a "
            "tier flip) and is underpowered at n=3. test_b: top cluster is NOT flat "
            "(p1-p4 gap=0.034) but both clear P0. test_c: P0 (GPT-4o) sits above the 7B "
            "P9 but below the RL-trained 7B P10. The 4th layer (DRB-RACE) is BLOCKED on "
            "external annotations, so the replication is 3-of-4 layers."),
    }
    return out, consistent


def _print_dry(out, consistent):
    print(f"[{KEY}] DRY-RUN — computed, nothing written.")
    print(f"  n_reports_judged={out['n_reports_judged']}  "
          f"judge={out['judge_model']} gen={out['generation_model']}")
    print("  recomputed per-pattern means:")
    for p in PATTERNS:
        m = out["e12_means_recomputed_from_verdicts"][p]
        print(f"    {p}: n={m['n']} mean={m['mean']:.7f}")
    cc = out["consistency_with_concordance_file"]
    print(f"  consistency vs concordance file: max_abs_diff={cc['max_abs_diff']:.2e} "
          f"match_within_5e-4={cc['match_within_5e-4']}")
    print(f"  test_a rho={out['test_a_rank_concordance']['spearman_rho']} "
          f"(n={out['test_a_rank_concordance']['n_patterns']})")
    tb = out["test_b_flat_top_cluster"]
    print(f"  test_b p1_p4_gap={tb['p1_p4_gap']:.4f} survives={tb['survives']} "
          f"p1>p0={tb['p1_above_p0']} p4>p0={tb['p4_above_p0']}")
    tc = out["test_c_p0_7b_tiering"]
    print(f"  test_c p0>p9={tc['p0_above_p9']} p0>p10={tc['p0_above_p10']}")
    print(f"  drb_race_layer status={out['drb_race_layer']['status']}")
    print(f"  CONSISTENCY {'OK' if consistent else 'FAILED'}")


def _atomic_append(out, force):
    cn = json.load(open(CANON))
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
    print(f"[{KEY}] WROTE key '{KEY}' -> {CANON}  (store now {len(cn)} keys)")
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
    if not CONCORDANCE.exists():
        print(f"[{KEY}] concordance file missing at {CONCORDANCE}; nothing to do "
              f"(self-guard).")
        return 0
    if not JUDGE_DIR.exists():
        print(f"[{KEY}] judge dir missing at {JUDGE_DIR}; nothing to do (self-guard).")
        return 0

    out, consistent = build()

    if args.write:
        return _atomic_append(out, args.force)
    _print_dry(out, consistent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
