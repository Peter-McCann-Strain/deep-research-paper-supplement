#!/usr/bin/env python
"""E14: retrieval-vs-utilisation decomposition from claim-level entailment (A2 key).

Pairs the ORACLE verified-factual-accuracy (perfect-retrieval, every cited source is the
cached real page) against the BASE verified-factual-accuracy (the model's own retrieval),
both produced by the same FActScore/SAFE C0 entailment pipeline on PTU gpt-4o:

  base vfa     := df_c0_per_report.parquet           (base_pN, existing)
  oracle vfa   := df_e14_oracle_per_report.parquet   (oracle_t1_pN, scripts/run_e14_oracle_entail.py)

For each cluster pattern (p1,p4,p5,p6,p7,p8 — the orchestration cluster used throughout the
oracle arm) we report, paired on the 30 variance queries:

  * utilisation_ceiling   = mean oracle vfa            (how well claims are grounded GIVEN perfect
                            sources — the share of the factual gap that is NOT retrieval-bound)
  * retrieval_component   = mean(oracle vfa - base vfa) (the gap closed purely by perfect retrieval)
  * base_vfa              = mean base vfa

This is the compute backing A2's decomposition: it splits the factual-accuracy shortfall
into a retrieval-bound part (recoverable with better sources) and a utilisation ceiling
(a grounding limit the architecture cannot pass even with oracle sources).

Appends canonical_numbers.json['e14_oracle_entail']. Self-guards: if the E14 oracle parquet
is absent (entailment run not yet executed) it writes a 'pending' block and exits 0, so the
rebuild chain stays green pre-E14.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
A = ROOT / "data" / "analysis"
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"   # CORRECTED path (post-0a80ba6)
CANON = ANA / "canonical_numbers.json"

CLUSTER = ("p1", "p4", "p5", "p6", "p7", "p8")  # ORDERED tuple (deterministic; Rule-5)
BASE_PR = A / "df_c0_per_report.parquet"
ORACLE_PR = A / "df_e14_oracle_per_report.parquet"


def _vfa_map(df: pd.DataFrame, pattern: str, varq: set) -> dict:
    d = df[df.pattern.astype(str) == pattern]
    return {q: float(v) for q, v in zip(d.query_id.astype(str), d.verified_factual_accuracy)
            if q in varq and pd.notna(v)}


def main():
    canon = json.loads(CANON.read_text())

    if not ORACLE_PR.exists():
        canon["e14_oracle_entail"] = {
            "status": "pending",
            "note": ("E14 oracle entailment parquet absent; run "
                     "scripts/run_e14_oracle_entail.py (PTU gpt-4o) first."),
            "expected_input": str(ORACLE_PR),
        }
        CANON.write_text(json.dumps(canon, indent=1))
        print("[E14] oracle parquet absent -> wrote 'pending' block, exiting 0.")
        return

    varq = set(json.loads((ROOT / "data" / "variance_stratified.json").read_text())["query_ids"])
    base = pd.read_parquet(BASE_PR) if BASE_PR.exists() else pd.DataFrame(
        columns=["pattern", "query_id", "verified_factual_accuracy"])
    orac = pd.read_parquet(ORACLE_PR)
    rng = np.random.default_rng(20260622)

    per_pattern = {}
    retr_means, util_means = [], []
    for p in CLUSTER:
        om = _vfa_map(orac, f"oracle_t1_{p}", varq)
        bm = _vfa_map(base, f"base_{p}", varq)
        paired_q = [q for q in om if q in bm]
        if not paired_q:
            per_pattern[p] = {"status": "no_paired_queries",
                              "n_oracle": len(om), "n_base": len(bm)}
            continue
        util = float(np.mean([om[q] for q in paired_q]))
        bvfa = float(np.mean([bm[q] for q in paired_q]))
        retr = float(np.mean([om[q] - bm[q] for q in paired_q]))
        per_pattern[p] = {
            "n_paired": len(paired_q),
            "utilisation_ceiling_oracle_vfa": round(util, 4),
            "base_vfa": round(bvfa, 4),
            "retrieval_component_oracle_minus_base": round(retr, 4),
        }
        util_means.append(util)
        retr_means.append(retr)

    # Cluster-level decomposition with a paired bootstrap over the per-pattern retrieval means
    # (6 cluster patterns = the correct inferential unit; patterns share the same 30 queries).
    def boot_ci(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 2:
            return [None, None]
        bs = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(10000)])
        return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    block = {
        "_note": ("Retrieval-vs-utilisation decomposition of the factual channel from claim-level "
                  "entailment (PTU gpt-4o). oracle vfa = grounding under perfect retrieval; "
                  "(oracle-base) = the part of the factual gap that perfect retrieval closes."),
        "judge_endpoint": "PTU gpt-4o (entailment verifier; no LLM-as-judge, no Opus)",
        "cluster": list(CLUSTER),
        "paired_on": "30 variance-stratified queries",
        "per_pattern": per_pattern,
        "cluster_utilisation_ceiling": round(float(np.mean(util_means)), 4) if util_means else None,
        "cluster_utilisation_ceiling_ci95": boot_ci(util_means),
        "cluster_retrieval_component": round(float(np.mean(retr_means)), 4) if retr_means else None,
        "cluster_retrieval_component_ci95": boot_ci(retr_means),
        "interpretation": ("If the retrieval_component CI excludes a practically-large value while "
                           "the utilisation ceiling stays well below 1.0, the factual shortfall is "
                           "utilisation-bound, not retrieval-bound — the A2 decomposition key."),
    }

    canon["e14_oracle_entail"] = block
    CANON.write_text(json.dumps(canon, indent=1))
    print(json.dumps({k: block[k] for k in
                      ["cluster_utilisation_ceiling", "cluster_utilisation_ceiling_ci95",
                       "cluster_retrieval_component", "cluster_retrieval_component_ci95"]}, indent=1))
    print("per-pattern (oracle vfa | base vfa | retrieval Δ):")
    for p, v in per_pattern.items():
        if "utilisation_ceiling_oracle_vfa" in v:
            print(f"  {p}: {v['utilisation_ceiling_oracle_vfa']:.3f} | "
                  f"{v['base_vfa']:.3f} | {v['retrieval_component_oracle_minus_base']:+.3f}")


if __name__ == "__main__":
    main()
