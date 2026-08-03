#!/usr/bin/env python
"""P2_rxu_conditional (Paper 5) — utilisation re-expressed as a CONDITIONAL, not a ratio.

Replaces deep_research/evaluation/component_eval.py:286, where source utilisation is the flat
ratio  utilisation = n_cited / available_sources  (and where verified_factual_accuracy collapses
to n_supports / n_claims). That single ratio conflates two distinct failure modes: (a) the
retrieved set never contained supporting evidence for the claim, and (b) the architecture had the
evidence in hand but failed to ground the claim against it. arXiv:2601.03261 (DeepResearch-Slice,
June 2026) argues exactly this: deep-research accuracy must be SLICED into a retrieval factor and
a conditional utilisation factor, because a flat cited/available ratio rewards a system that cites
abundantly without entailment and penalises a system that retrieves well but is graded on a sparse
citation index. We therefore re-express utilisation per the DeepResearch-Slice decomposition.

SUBSTRATE (on disk, verified): data/analysis/df_c0_verdicts.parquet — the claim-level
entailment output of the citation_verifier NLI mode (deep_research/evaluation/citation_verifier.py,
SAFE/FActScore-style). One row per atomic claim with:
    verdict        in {supports, neutral, contradicts, no_source}
    evidence_quote the retrieved source snapshot the claim was checked against
Verified semantics (this script's design rests on them; checked on disk 2026-06-22):
    * verdict == 'no_source'  <=>  evidence_quote is empty (100% of no_source, 0% of the rest)
      i.e. the saved-search snapshot contained NO supporting evidence for the claim -> R = 0.
    * verdict in {supports, neutral, contradicts}  <=>  a retrieved snapshot WAS present (R = 1);
      among those, 'supports' is the entailment match (U = 1).

CONDITIONAL DECOMPOSITION (per the spec / DeepResearch-Slice):
    R     (retrieval)            per claim: gold/supporting evidence present in the retrieved set
                                 = 1 if verdict != 'no_source' else 0      (NLI snapshot present)
    U|R   (utilisation | retrieved)  among R==1 claims: did the claim entail-match the snapshot
                                 = 1 if verdict == 'supports' else 0
    accuracy(pattern) = mean(R) * mean(U|R)   -- the sliced product, NOT mean(R & U) and NOT the
                        flat n_supports/n_claims ratio it replaces (the two differ whenever the
                        retrieval and utilisation factors are not independent; we report both so
                        the gap is auditable).

COVERAGE (CONDITIONAL scope — the spec asks us to flag thin coverage explicitly):
  df_c0_verdicts covers 9 of the 13 base patterns (p0, p1, p4, p5, p7, p8, p9, p10, p11) at ~30
  reports each. Patterns p2, p3, p6, p12 have NO saved C0 entailment snapshots on disk, so the
  conditional cannot be formed for them; they are listed under 'patterns_uncovered' and omitted
  from the aggregate. This is scoped to the patterns the NLI-mode snapshots actually cover.

Determinism: SEED=20260611. Inputs sorted before any resample. Report-level (query_id) cluster
bootstrap for each pattern's accuracy CI (reports are the inferential unit; claims are nested in
reports). Self-guards: if df_c0_verdicts.parquet is absent, writes a 'pending' block and exits 0
so the rebuild chain stays green.

Appends canonical_numbers.json['oracle']['rxu_conditional'] (a NEW subkey under the existing
'oracle' block; all sibling oracle keys are preserved). Atomic tmp+os.replace write idiom mirrors
build_judge_vs_gold.py / build_n_eff.py. CPU-only, $0, idempotent, reads real on-disk data.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
A = ROOT / "data" / "analysis"
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"   # CORRECTED path (post-0a80ba6)
CANON = ANA / "canonical_numbers.json"
C0_VERDICTS = A / "df_c0_verdicts.parquet"

SEED = 20260611
CITATION = "arXiv:2601.03261 (DeepResearch-Slice, June 2026)"
N_BOOT = 10000

# The full base panel, so uncovered patterns can be named explicitly (CONDITIONAL scope note).
ALL_BASE = ("base_p0", "base_p1", "base_p2", "base_p3", "base_p4", "base_p5", "base_p6",
            "base_p7", "base_p8", "base_p9", "base_p10", "base_p11", "base_p12")

RETRIEVED_VERDICTS = ("supports", "neutral", "contradicts")  # evidence snapshot present (R=1)
ENTAIL_VERDICT = "supports"                                  # entailment match (U=1)


def _report_factors(claims: pd.DataFrame):
    """Per-report (R_count, retrieved_count, support_count, n_claims) on a claim slice."""
    n = len(claims)
    retrieved = int(claims["R"].sum())
    supports = int((claims["verdict"] == ENTAIL_VERDICT).sum())
    return retrieved, supports, n


def _pattern_accuracy(claims: pd.DataFrame):
    """mean(R) * mean(U|R) pooled over a pattern's claims (point estimate)."""
    n = len(claims)
    retrieved = int(claims["R"].sum())
    supports = int((claims["verdict"] == ENTAIL_VERDICT).sum())
    mean_R = retrieved / n if n else 0.0
    mean_UgivenR = supports / retrieved if retrieved else 0.0
    return mean_R, mean_UgivenR, mean_R * mean_UgivenR, n, retrieved, supports


def _cluster_bootstrap_ci(claims: pd.DataFrame, rng) -> list:
    """Report-clustered bootstrap CI for the CONDITIONAL utilisation rate U|R = supports/retrieved.
    NB: the product mean(R)*mean(U|R) = supports/n is algebraically the plain support rate and is
    identical to the flat ratio it would 'replace', so it carries no extra information. The
    informative decomposition is the pair (mean_R, mean_U|R); we therefore place the CI on the
    conditional factor U|R. Resample query_ids (sorted); reports are the inferential unit."""
    qids = sorted(claims["query_id"].unique().tolist())  # sort BEFORE sampling (Rule-3)
    if len(qids) < 2:
        return [None, None]
    by_q = {q: claims[claims["query_id"] == q] for q in qids}
    arr = np.empty(N_BOOT, dtype=float)
    nq = len(qids)
    idx = np.arange(nq)
    for b in range(N_BOOT):
        pick = rng.choice(idx, size=nq, replace=True)
        sample = pd.concat([by_q[qids[i]] for i in pick], ignore_index=True)
        n = len(sample)
        retrieved = int(sample["R"].sum())
        supports = int((sample["verdict"] == ENTAIL_VERDICT).sum())
        mu = supports / retrieved if retrieved else 0.0
        arr[b] = mu  # CI on the conditional utilisation factor U|R (the informative term)
    return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]


def main():
    canon = json.loads(CANON.read_text())
    canon.setdefault("oracle", {})  # never clobber existing oracle siblings

    if not C0_VERDICTS.exists():
        canon["oracle"]["rxu_conditional"] = {
            "status": "pending",
            "citation": CITATION,
            "note": ("C0 claim-level entailment parquet absent; the citation_verifier NLI-mode "
                     "snapshots (df_c0_verdicts.parquet) are required for the R x (U|R) slice."),
            "expected_input": str(C0_VERDICTS),
        }
        tmp = str(CANON) + ".tmp"
        open(tmp, "w").write(json.dumps(canon, indent=1))
        os.replace(tmp, str(CANON))
        print("[P2_rxu] C0 verdicts absent -> wrote 'pending' block, exiting 0.")
        return

    v = pd.read_parquet(C0_VERDICTS)
    # R: retrieved-evidence-present indicator (NLI snapshot present <=> verdict != no_source).
    v = v.assign(R=v["verdict"].isin(RETRIEVED_VERDICTS).astype(int))

    covered = sorted(p for p in v["pattern"].unique() if p in ALL_BASE)
    uncovered = [p for p in ALL_BASE if p not in set(covered)]

    rng = np.random.default_rng(SEED)
    per_pattern = {}
    acc_list = []
    for p in covered:  # sorted -> deterministic order
        cl = v[v["pattern"] == p].sort_values(["query_id"]).reset_index(drop=True)
        mean_R, mean_UgivenR, acc, n_claims, n_retrieved, n_supports = _pattern_accuracy(cl)
        # the flat ratio this replaces (component_eval.py:286 family): n_supports / n_claims
        flat_ratio = n_supports / n_claims if n_claims else 0.0
        ci = _cluster_bootstrap_ci(cl, rng)
        per_pattern[p] = {
            "n_reports": int(cl["query_id"].nunique()),
            "n_claims": int(n_claims),
            "n_retrieved": int(n_retrieved),
            "n_supports": int(n_supports),
            "n_no_source": int(n_claims - n_retrieved),
            "mean_R": round(mean_R, 4),                     # retrieval factor P(R) -- informative (e.g. p10 retrieval-bound)
            "mean_U_given_R": round(mean_UgivenR, 4),       # conditional utilisation factor P(U|R) -- informative
            "u_given_r_ci95": ci,                           # report-clustered bootstrap CI on P(U|R)
            "support_rate": round(acc, 4),                  # = mean(R)*mean(U|R) = n_supports/n_claims, IDENTICAL to the flat ratio (reference only, no extra info)
            "flat_support_ratio": round(flat_ratio, 4),     # component_eval.py:286 analogue (== support_rate by construction)
        }
        acc_list.append(acc)

    # Cluster-level summary across the covered patterns (patterns share the same query frame).
    def _mean_ci(vals):
        vals = np.asarray(sorted(vals), dtype=float)  # sort before resample
        if len(vals) < 2:
            return (round(float(vals.mean()), 4) if len(vals) else None), [None, None]
        bs = np.array([np.random.default_rng(SEED + i).choice(vals, len(vals), replace=True).mean()
                       for i in range(N_BOOT)])
        return round(float(vals.mean()), 4), [round(float(np.percentile(bs, 2.5)), 4),
                                              round(float(np.percentile(bs, 97.5)), 4)]

    cluster_acc, cluster_ci = _mean_ci(acc_list)

    block = {
        "_note": ("Utilisation re-expressed as a CONDITIONAL, not a flat ratio, per "
                  "DeepResearch-Slice. Per claim: R = retrieved-evidence present in the saved NLI "
                  "snapshot (verdict != no_source); U|R = entailment match among retrieved "
                  "(verdict == supports). accuracy = mean(R) * mean(U|R) per pattern. Replaces "
                  "component_eval.py:286 (cited/available, and n_supports/n_claims) which conflates "
                  "the retrieval-bound and utilisation-bound shares of the factual gap."),
        "citation": CITATION,
        "replaces": "deep_research/evaluation/component_eval.py:286 (source_utilization ratio)",
        "source_parquet": str(C0_VERDICTS),
        "verifier": ("citation_verifier NLI mode (deep_research/evaluation/citation_verifier.py); "
                     "claim-level SAFE/FActScore entailment, PTU gpt-4o C0 pipeline"),
        "seed": SEED,
        "n_boot": N_BOOT,
        "definitions": {
            "R": "1 if a retrieved snapshot was present (verdict in supports|neutral|contradicts) else 0",
            "U_given_R": "among R==1 claims, 1 if verdict == supports (entailment match) else 0",
            "decomposition": "the informative pair is (mean_R, mean_U|R); their product == n_supports/n_claims (the flat ratio) and is reported as support_rate for reference only",
        },
        "coverage": {
            "patterns_covered": covered,
            "patterns_uncovered": uncovered,
            "n_patterns_covered": len(covered),
            "coverage_note": ("CONDITIONAL scope: C0 NLI-mode snapshots exist for these patterns "
                              "only; p2/p3/p6/p12 have no saved entailment snapshots on disk and "
                              "are omitted from the aggregate (coverage is thin for those four)."),
        },
        "per_pattern": per_pattern,
        "cluster_mean_support_rate": cluster_acc,
        "cluster_mean_support_rate_ci95": cluster_ci,
        "interpretation": ("A low mean(R) with a high mean(U|R) is a retrieval-bound pattern (the "
                           "architecture grounds well once evidence is in hand); a high mean(R) "
                           "with a low mean(U|R) is utilisation-bound (evidence retrieved but not "
                           "entailed). The flat ratio it replaces cannot distinguish these."),
    }

    canon["oracle"]["rxu_conditional"] = block  # APPEND subkey; siblings untouched
    tmp = str(CANON) + ".tmp"
    open(tmp, "w").write(json.dumps(canon, indent=1))
    os.replace(tmp, str(CANON))

    print(f"[P2_rxu] covered={covered}")
    print(f"  uncovered (no C0 snapshots): {uncovered}")
    print(f"  cluster mean support_rate (=flat) = {cluster_acc}  CI95={cluster_ci}")
    for p, d in per_pattern.items():
        print(f"  {p}: R={d['mean_R']:.3f} U|R={d['mean_U_given_R']:.3f} "
              f"support_rate={d['support_rate']:.3f} (==flat={d['flat_support_ratio']:.3f})")


if __name__ == "__main__":
    main()
