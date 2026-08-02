#!/usr/bin/env python
"""A2 / E14 — Land P9 & P10 in the oracle arm, and scaffold the claim-level
retrieval-vs-utilisation factual decomposition  P(correct) = P(R) x P(U|R).

WHY THIS SCRIPT EXISTS
----------------------
1. P9 (Qwen2.5-7B, P0 architecture) and P10 (DeepResearcher-7b, RL) were judged in
   the oracle / perfect-retrieval arm (gpt52):
       results/judge_gpt52/oracle_t1_p9   (n=30, mean 0.2609)
       results/judge_gpt52/oracle_t1_p10  (n=30, mean 0.2723)
   and their matched baselines are already in the released parquet
       results/judge_gpt52/base_p9, base_p10  ->  df_overall_scores.parquet / df_scores.parquet
   BUT canonical['oracle']['per_pattern'] only has p0..p8, because build_numbers.py:oracle()
   iterates `for i in range(9)` AND the oracle_t1_p9 / oracle_t1_p10 verdicts were never
   folded into df_overall_scores.parquet (they live only as raw judge JSON).  This script
   reads those raw verdicts and appends p9, p10 to per_pattern using the *identical*
   paired-bootstrap idiom as build_numbers.py:oracle() (rng seed 7, 2000 reps, sorted-common
   pairing, round-4), so the numbers are bit-for-bit consistent with p0..p8.

2. It scaffolds a NEW canonical key  oracle['rxu_decomposition']  that splits the factual
   channel into a RETRIEVAL probability and a UTILISATION-given-retrieval probability:
       P(R)    = (n_claims - n_no_source) / n_claims           # a source was retrieved
       P(U|R)  = n_supports / (n_claims - n_no_source)         # retrieved source used correctly
       P(corr) = n_supports / n_claims = P(R) x P(U|R) = verified_factual_accuracy   (identity)
   The BASE arm is computed NOW ($0, pure recompute from data/analysis/df_c0_per_report.parquet,
   which already covers base_p0..base_p11 incl. base_p9/base_p10).  The ORACLE arm requires the
   claim-extraction + entailment pass over the oracle_t1_* reports, which is the EXISTING PTU
   script scripts/run_e14_oracle_entail.py (DEFAULT_MODEL gpt-4o, PTU deployment sthree-ptu-02,
   $0 marginal but a real API pass — NOT run here).  If its output parquet
   data/analysis/df_e14_oracle_per_report.parquet exists, the oracle arm and the
   retrieval-vs-utilisation delta are filled in; otherwise the oracle arm is marked
   status="pending_ptu_entailment" with the exact command to produce it.

ENDPOINT POLICY: gpt-5.2 judge verdicts are READ from disk only (already on disk, never
regenerated).  No Opus anywhere.  The only generative dependency (the oracle entailment pass)
is PTU gpt-4o and is NOT executed by this script.

PATH-BUG FIX: reads AND writes the canonical store at its CURRENT home
    papers/paper_a_bounded_returns/analysis/canonical_numbers.json
(it was moved here by commit 0a80ba6 from papers/paper_a_bounded_returns/analysis/).  This script
NEVER touches the dead old path.

IDEMPOTENT / CLOBBER-SAFE:
  * Re-runnable: recomputes p9/p10 + decomposition deterministically and overwrites only the
    keys it owns (oracle.per_pattern.p9, oracle.per_pattern.p10, oracle.rxu_decomposition).
  * Never deletes or rewrites p0..p8 or any other oracle sub-key.
  * --check  : compute and print, write NOTHING (dry run / CI guard).

Run:   ./venv/bin/python scripts/build_a2_e14_oracle_p9p10_and_rxu.py
Check: ./venv/bin/python scripts/build_a2_e14_oracle_p9p10_and_rxu.py --check
Out:   appends/refreshes papers/paper_a_bounded_returns/analysis/canonical_numbers.json
         oracle.per_pattern.p9, oracle.per_pattern.p10
         oracle.rxu_decomposition
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "."
ANA = f"{ROOT}/papers/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
JUDGE_DIR = f"{ROOT}/results/judge_gpt52"
VARQ_FILE = f"{ROOT}/data/variance_stratified.json"
C0_PER_REPORT = f"{ROOT}/data/analysis/df_c0_per_report.parquet"
E14_PER_REPORT = f"{ROOT}/data/analysis/df_e14_oracle_per_report.parquet"

# Same 9-dim ordering build_numbers.py uses for oracle per_pattern dims.
DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization",
        "instruction_following", "attribution_quality"]

# The non-frontier 7B patterns we are landing in the oracle arm. Neither is a cluster
# pattern (the cluster set is p1,p4,p5,p6,p7,p8), so is_cluster=False for both.
NEW_PATTERNS = ["p9", "p10"]
CLUSTER = ("p1", "p4", "p5", "p6", "p7", "p8")

# Patterns to populate in the BASE arm of the R x U decomposition. We restrict the
# decomposition's headline pairing to p9/p10 (the A2 focus) but also carry every base
# pattern C0 already covers, for context. Oracle-arm pairing fills in only when the PTU
# entailment parquet exists.
DECOMP_PATTERNS = ["p0", "p1", "p4", "p5", "p7", "p8", "p9", "p10", "p11"]


# ----------------------------------------------------------------------------------
# raw gpt-5.2 judge verdict readers (oracle_t1_p9 / p10 are NOT in the parquet)
# ----------------------------------------------------------------------------------
def _load_overall(pat: str, varq: set[str]) -> dict[str, float]:
    """query_id -> overall_score for one judged cell, VARQ-filtered (mirrors overall_map)."""
    out: dict[str, float] = {}
    for f in glob.glob(f"{JUDGE_DIR}/{pat}/*.json"):
        v = json.load(open(f))
        q = v["query_id"]
        s = v.get("overall_score")
        if q in varq and s is not None:
            out[q] = float(s)
    return out


def _load_dim(pat: str, dim: str, varq: set[str]) -> dict[str, float]:
    """query_id -> dimension score for one judged cell, VARQ-filtered (mirrors dim_map)."""
    out: dict[str, float] = {}
    for f in glob.glob(f"{JUDGE_DIR}/{pat}/*.json"):
        v = json.load(open(f))
        q = v["query_id"]
        dd = v.get("dimensions", {}).get(dim)
        if q in varq and dd is not None and dd.get("score") is not None:
            out[q] = float(dd["score"])
    return out


def _paired_delta(omap: dict[str, float], bmap: dict[str, float]) -> dict | None:
    """EXACT replica of build_numbers.py:oracle().paired_delta (seed 7, 2000 reps, sorted)."""
    common = sorted(q for q in omap if q in bmap)  # sorted: hash order varies per process
    if not common:
        return None
    d = np.array([omap[q] - bmap[q] for q in common])
    rng_pd = np.random.default_rng(7)
    boot = [rng_pd.choice(d, len(d), replace=True).mean() for _ in range(2000)]
    return {
        "n": int(len(d)),
        "delta": round(float(d.mean()), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                 round(float(np.percentile(boot, 97.5)), 4)],
        "oracle_mean": round(float(np.mean([omap[q] for q in common])), 4),
        "base_mean": round(float(np.mean([bmap[q] for q in common])), 4),
    }


def build_per_pattern_new(varq: set[str]) -> dict:
    """Compute the p9/p10 per_pattern records from raw judge JSON."""
    recs = {}
    for p in NEW_PATTERNS:
        opat, bpat = f"oracle_t1_{p}", f"base_{p}"
        ov_o, ov_b = _load_overall(opat, varq), _load_overall(bpat, varq)
        if not ov_o:
            raise FileNotFoundError(
                f"No oracle verdicts found for {opat} under {JUDGE_DIR} — cannot land {p}.")
        rec = {
            "is_cluster": p in CLUSTER,
            "overall": _paired_delta(ov_o, ov_b),
            "dims": {d: _paired_delta(_load_dim(opat, d, varq), _load_dim(bpat, d, varq))
                     for d in DIMS},
            "_source": "raw_gpt52_judge_json (not in df_overall_scores.parquet)",
        }
        recs[p] = rec
    return recs


# ----------------------------------------------------------------------------------
# R x U factual decomposition  (P(correct) = P(R) x P(U|R))
# ----------------------------------------------------------------------------------
def _rxu_from_c0(df: pd.DataFrame, pat_col_value: str) -> dict | None:
    """Pool per-report claim counts for one pattern into P(R), P(U|R), P(correct).

    df rows: n_claims, n_supports, n_neutral, n_contradicts, n_no_source per (pattern, qid).
    Claim-pooled (sum over reports) so empty reports don't bias the rates:
       retrieved = total_claims - total_no_source
       P(R)      = retrieved / total_claims
       P(U|R)    = total_supports / retrieved
       P(correct)= total_supports / total_claims   (== P(R)*P(U|R), == verified_factual_accuracy)
    """
    d = df[df["pattern"].eq(pat_col_value)]
    if d.empty:
        return None
    n_claims = int(d["n_claims"].sum())
    if n_claims == 0:
        return None
    n_supports = int(d["n_supports"].sum())
    n_no_source = int(d["n_no_source"].sum())
    retrieved = n_claims - n_no_source
    p_r = retrieved / n_claims
    p_u_given_r = (n_supports / retrieved) if retrieved > 0 else 0.0
    p_correct = n_supports / n_claims
    return {
        "n_reports": int(d["query_id"].nunique()),
        "n_claims": n_claims,
        "n_supports": n_supports,
        "n_no_source": n_no_source,
        "P_R": round(float(p_r), 4),
        "P_U_given_R": round(float(p_u_given_r), 4),
        "P_correct": round(float(p_correct), 4),
        "_identity_check": round(float(p_r * p_u_given_r - p_correct), 6),  # ~0
    }


def build_rxu_decomposition() -> dict:
    """Base arm now ($0); oracle arm if the PTU entailment parquet exists, else pending stub."""
    out = {
        "_note": ("Claim-level factual decomposition P(correct)=P(R)xP(U|R). R = a source was "
                  "retrieved for the claim (claim is not no_source); U|R = the retrieved source "
                  "actually supports the claim. P(correct) equals the C0/E14 verified factual "
                  "accuracy by construction. Claim-pooled over reports. gpt52 judge feeds the "
                  "headline scores; the claim/entailment labels are the C0 pipeline (PTU gpt-4o, "
                  "deployment sthree-ptu-02, $0 marginal, NO Opus, NO 7B)."),
        "definition": {
            "P_R": "(n_claims - n_no_source) / n_claims",
            "P_U_given_R": "n_supports / (n_claims - n_no_source)",
            "P_correct": "n_supports / n_claims = P_R * P_U_given_R = verified_factual_accuracy",
        },
        "base": {},
        "oracle": {},
        "retrieval_vs_utilisation_delta": {},
    }

    # --- BASE arm: from the already-built C0 per-report parquet ($0) ---
    if os.path.exists(C0_PER_REPORT):
        c0 = pd.read_parquet(C0_PER_REPORT)
        out["base"]["_source"] = "data/analysis/df_c0_per_report.parquet"
        for p in DECOMP_PATTERNS:
            r = _rxu_from_c0(c0, f"base_{p}")
            if r is not None:
                out["base"][p] = r
    else:
        out["base"]["status"] = "missing_df_c0_per_report"

    # --- ORACLE arm: from the E14 oracle entailment parquet (PTU pass), if present ---
    if os.path.exists(E14_PER_REPORT):
        e14 = pd.read_parquet(E14_PER_REPORT)
        out["oracle"]["_source"] = "data/analysis/df_e14_oracle_per_report.parquet"
        out["oracle"]["status"] = "ready"
        for p in DECOMP_PATTERNS:
            # E14 parquet stores oracle reports under the oracle_t1_* pattern name.
            r = _rxu_from_c0(e14, f"oracle_t1_{p}")
            if r is not None:
                out["oracle"][p] = r
        # Retrieval-vs-utilisation delta: oracle gives perfect-retrieval sources, so any
        # residual factual gap is utilisation-bound. Decompose the oracle-minus-base factual
        # lift into a retrieval channel (Delta P_R) and a utilisation channel (Delta P_U|R).
        for p in DECOMP_PATTERNS:
            b = out["base"].get(p)
            o = out["oracle"].get(p)
            if isinstance(b, dict) and isinstance(o, dict):
                out["retrieval_vs_utilisation_delta"][p] = {
                    "delta_P_R": round(o["P_R"] - b["P_R"], 4),
                    "delta_P_U_given_R": round(o["P_U_given_R"] - b["P_U_given_R"], 4),
                    "delta_P_correct": round(o["P_correct"] - b["P_correct"], 4),
                }
    else:
        out["oracle"]["status"] = "pending_ptu_entailment"
        out["oracle"]["_blocked_on"] = (
            "data/analysis/df_e14_oracle_per_report.parquet does not exist yet")
        out["oracle"]["_endpoint"] = "PTU gpt-4o (sthree-ptu-02), $0 marginal — NOT $0-local"
        out["oracle"]["_command_to_produce_it"] = (
            "./venv/bin/python scripts/run_e14_oracle_entail.py "
            "--patterns all --max-claims 20 --concurrency 3")
        out["oracle"]["_dry_run_first"] = (
            "./venv/bin/python scripts/run_e14_oracle_entail.py --dry-run")
        out["retrieval_vs_utilisation_delta"]["status"] = (
            "pending: needs oracle arm (run_e14_oracle_entail.py) before delta is defined")
    return out


# ----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Compute and print only; write NOTHING to canonical_numbers.json.")
    args = ap.parse_args()

    if not os.path.exists(CANON):
        sys.exit(f"[FATAL] canonical store not found at {CANON} (path bug? check commit 0a80ba6).")

    varq = set(json.load(open(VARQ_FILE))["query_ids"])

    new_pp = build_per_pattern_new(varq)
    rxu = build_rxu_decomposition()

    # Report
    for p in NEW_PATTERNS:
        ov = new_pp[p]["overall"]
        print(f"oracle.per_pattern.{p:3s}  n={ov['n']}  base {ov['base_mean']:.4f} -> "
              f"oracle {ov['oracle_mean']:.4f}  delta {ov['delta']:+.4f}  ci95 {ov['ci95']}")
    print(f"rxu_decomposition base arm: {len([k for k in rxu['base'] if not k.startswith('_')])} "
          f"patterns | oracle arm: {rxu['oracle'].get('status')}")
    if rxu["oracle"].get("status") == "pending_ptu_entailment":
        print(f"  -> to fill oracle arm: {rxu['oracle']['_command_to_produce_it']}")

    if args.check:
        print("[--check] no write performed.")
        return

    # Idempotent write: refresh only the keys we own.
    cn = json.load(open(CANON))
    if "oracle" not in cn:
        sys.exit("[FATAL] canonical['oracle'] missing — run build_numbers.py first.")
    cn["oracle"].setdefault("per_pattern", {})
    for p in NEW_PATTERNS:
        cn["oracle"]["per_pattern"][p] = new_pp[p]
    cn["oracle"]["rxu_decomposition"] = rxu

    json.dump(cn, open(CANON, "w"), indent=1)
    print(f"WROTE {CANON}  (oracle.per_pattern += {NEW_PATTERNS}; oracle.rxu_decomposition)")


if __name__ == "__main__":
    main()
