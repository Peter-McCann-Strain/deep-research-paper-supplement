#!/usr/bin/env python
"""P2 FAITHFULNESS (Paper 5/2) — citation faithfulness via leave-one-source-out replay.

Canonical key: `citation_faithfulness`.

Motivation (arXiv:2412.18004, "Towards Verifiable Text Generation with Evolving
Knowledge", Dec-2024): up to ~57% of citations in LLM long-form answers are post-hoc
RATIONALISATIONS — the cited source is attached after the claim is written and the claim
would stand (remain entailed) even WITHOUT that particular source in context. A faithful
citation is one whose specific cited chunk is load-bearing: drop it and the claim is no
longer entailed by the remaining retrieved context. We operationalise that as a
leave-one-source-out (LOSO) entailment replay over saved trajectories:

  for each sampled CITED claim c with cited chunk s_c and retrieved context C(c):
      drop s_c from C(c); re-run citation_verifier.nli_verify_batch(claim=c, premise=C(c)\\{s_c})
      if the claim is STILL entailed without its own cited chunk
          -> post-rationalisation-suspect (the citation was not load-bearing)
  per-pattern faithfulness rate = 1 - (suspect / cited-claims-tested)

This is a PURE REPLAY on saved trajectories: no new model rollouts. It nonetheless needs
SAVED, REPLAYABLE inputs:
  (i)   per CITED claim, the set of MULTIPLE retrieved source chunks that were in context
        (so one can be dropped and the rest re-tested), and
  (ii)  the saved NLI entailment verdict of the claim against the leave-one-out remainder
        context (so no fresh LLM/NLI call is required — Rule: CPU-only, $0, no paid API).

CONDITIONAL GUARD (mirrors build_e14_oracle_entail.py's pending-block idiom):
We verify the on-disk citation_verifier coverage BEFORE attempting the replay. If the
required LOSO replay artifact is absent, we DO NOT fabricate a number and DO NOT call any
API. We write a documented `data_insufficient` block recording exactly which input is
missing, plus a deterministic, fully-on-disk PROXY (see below) reported as an exploratory
lower-bound only — never as the headline faithfulness rate.

VERIFIED on-disk substrate (2026-06-22):
  * data/analysis/df_c0_verdicts.parquet — the FActScore/SAFE C0 claim-level entailment
    table (3096 claims; cols pattern,query_id,claim,citation_idx,verdict,evidence_quote).
    It stores ONE evidence_quote per claim and a single citation_idx (non-null for only
    176/3096 claims); there is ~1 row per (pattern,query,claim) (share>1 ≈ 0.001). It does
    NOT store the multi-chunk per-claim context, and it has NO leave-one-out re-entailment
    verdict. -> requirement (i)/(ii) NOT satisfied for the full LOSO replay.
  * artifacts/.../citation_verify.json snapshots are aggregate stubs
    ({accuracy_rate, checked, flagged}) — no per-claim chunk sets, no NLI premises.
    -> NOT a replay substrate.
Conclusion: the strict LOSO faithfulness replay is NOT computable at $0 from current
on-disk data; this builder emits `status:"data_insufficient"` for the strict metric and
fills the PROXY block.

PROXY (deterministic, on-disk, labelled exploratory): among CITED claims (citation_idx
non-null) in df_c0_verdicts, the share whose OWN cited source does NOT entail the claim
(verdict in {neutral, no_source, contradicts}). A citation whose own cited chunk already
fails to entail the claim is, a fortiori, a post-rationalisation footprint (the citation is
not load-bearing for entailment). This UNDER-counts true post-rationalisation (it cannot
catch the LOSO case where the cited chunk entails but so does the remainder), hence it is a
LOWER BOUND on the post-rationalisation rate and an UPPER BOUND on faithfulness. Reported
per-pattern, clearly distinguished from the strict (pending) LOSO metric.

Determinism: SEED=20260611; inputs sorted before any sampling/bootstrap; closed-form rates
otherwise. Atomic tmp+os.replace write; appends ONLY the `citation_faithfulness` key.

Run flag: writes canonical only when invoked as __main__ (no module-import side effects).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260611
CITATION = "arXiv:2412.18004"  # ~57% of LLM citations are post-hoc rationalisations (Dec-2024)

ROOT = Path(".")
A = ROOT / "data" / "analysis"
# CORRECTED canonical path (post-0a80ba6); the dead reports/paper_world_class path is gone.
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

C0_VERDICTS = A / "df_c0_verdicts.parquet"
# The strict-LOSO replay substrate, IF it is ever materialised by a dedicated replay run.
# Expected schema (one row per cited claim): pattern, query_id, claim, citation_idx,
#   own_chunk_verdict, loso_remainder_verdict  (verdicts in {supports,neutral,contradicts,no_source}).
LOSO_REPLAY = A / "df_citation_loso_replay.parquet"

# verdict labels that mean "the (premise) text does NOT entail the claim"
NON_ENTAIL = {"neutral", "no_source", "contradicts"}


def _short(pat: str) -> str:
    # 'base_p4' -> 'p4'
    return str(pat).split("_", 1)[1] if str(pat).startswith("base_") else str(pat)


def _boot_ci(rng, vals, n=10000):
    vals = np.asarray(sorted(float(v) for v in vals), dtype=float)  # sort -> deterministic
    if len(vals) < 2:
        return [None, None]
    bs = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def _strict_loso_block(rng) -> dict:
    """Strict leave-one-source-out faithfulness, computed ONLY from saved replay verdicts.

    Returns a populated block if df_citation_loso_replay.parquet exists, else a documented
    `data_insufficient` block. NEVER calls a model (pure replay)."""
    if not LOSO_REPLAY.exists():
        return {
            "status": "data_insufficient",
            "metric": "leave_one_source_out_faithfulness",
            "missing_input": str(LOSO_REPLAY.relative_to(ROOT)),
            "reason": (
                "Strict LOSO faithfulness requires, per cited claim, (i) the multi-chunk "
                "retrieved context and (ii) a SAVED NLI verdict of the claim against the "
                "leave-one-out remainder. df_c0_verdicts.parquet stores one evidence_quote "
                "per claim with ~1 row per (pattern,query,claim) and no remainder re-test; "
                "citation_verify.json snapshots are aggregate stubs. Neither is a replay "
                "substrate, so the strict metric is NOT computable at $0 from on-disk data."),
            "to_materialise": (
                "Run a dedicated LOSO replay (citation_verifier.nli_verify_batch over "
                "claim x remainder-context) and persist per-cited-claim "
                "{own_chunk_verdict, loso_remainder_verdict} to the missing parquet; this "
                "builder then computes per-pattern faithfulness with no further calls."),
        }
    df = pd.read_parquet(LOSO_REPLAY)
    df = df.sort_values(["pattern", "query_id", "citation_idx", "claim"]).reset_index(drop=True)
    per_pattern, rates = {}, []
    for pat in sorted(df.pattern.unique(), key=str):
        sub = df[df.pattern == pat]
        # tested = cited claims whose OWN chunk entailed (so LOSO drop is meaningful)
        tested = sub[sub.own_chunk_verdict == "supports"]
        if len(tested) == 0:
            per_pattern[_short(pat)] = {"status": "no_cited_entailed_claims", "n_cited": int(len(sub))}
            continue
        # suspect = still entailed by the remainder WITHOUT its own cited chunk
        suspect = int((tested.loso_remainder_verdict == "supports").sum())
        faith = 1.0 - suspect / len(tested)
        per_pattern[_short(pat)] = {
            "n_cited_tested": int(len(tested)),
            "n_post_rationalisation_suspect": suspect,
            "post_rationalisation_rate": round(suspect / len(tested), 4),
            "faithfulness_rate": round(faith, 4),
        }
        rates.append(faith)
    return {
        "status": "ok",
        "metric": "leave_one_source_out_faithfulness",
        "definition": ("faithfulness = 1 - share of own-chunk-entailed cited claims that remain "
                       "entailed by the remainder context after dropping their cited chunk."),
        "per_pattern": per_pattern,
        "cluster_mean_faithfulness": round(float(np.mean(rates)), 4) if rates else None,
        "cluster_mean_faithfulness_ci95": _boot_ci(rng, rates),
    }


def _proxy_block(rng) -> dict:
    """Deterministic, fully-on-disk lower-bound proxy from df_c0_verdicts (cited claims)."""
    if not C0_VERDICTS.exists():
        return {"status": "data_insufficient", "missing_input": str(C0_VERDICTS.relative_to(ROOT))}
    d = pd.read_parquet(C0_VERDICTS)
    cited = d[d.citation_idx.notna()].copy()
    cited = cited.sort_values(["pattern", "query_id", "citation_idx", "claim"]).reset_index(drop=True)
    per_pattern, rates = {}, []
    for pat in sorted(cited.pattern.unique(), key=str):
        sub = cited[cited.pattern == pat]
        n = int(len(sub))
        suspect = int(sub.verdict.isin(NON_ENTAIL).sum())  # own cited chunk fails to entail
        faith_ub = 1.0 - suspect / n if n else None
        per_pattern[_short(pat)] = {
            "n_cited": n,
            "n_own_chunk_non_entail": suspect,
            "post_rationalisation_rate_lowerbound": round(suspect / n, 4) if n else None,
            "faithfulness_rate_upperbound": round(faith_ub, 4) if n else None,
        }
        if n:
            rates.append(faith_ub)
    tot = int(len(cited))
    tot_suspect = int(cited.verdict.isin(NON_ENTAIL).sum())
    return {
        "status": "exploratory_lowerbound",
        "metric": "own_chunk_non_entailment_proxy",
        "definition": ("Among CITED claims (citation_idx non-null), the share whose own cited "
                       "source does NOT entail the claim (verdict in {neutral,no_source,"
                       "contradicts}). This is a LOWER BOUND on post-rationalisation (misses the "
                       "LOSO case where the cited chunk entails but the remainder also does), "
                       "hence an UPPER BOUND on faithfulness. NOT the headline metric."),
        "source": str(C0_VERDICTS.relative_to(ROOT)),
        "n_cited_total": tot,
        "post_rationalisation_rate_lowerbound_overall": round(tot_suspect / tot, 4) if tot else None,
        "per_pattern": per_pattern,
        "cluster_mean_faithfulness_upperbound": round(float(np.mean(rates)), 4) if rates else None,
        "cluster_mean_faithfulness_upperbound_ci95": _boot_ci(rng, rates),
    }


def build() -> dict:
    rng = np.random.default_rng(SEED)
    strict = _strict_loso_block(rng)
    proxy = _proxy_block(rng)
    block = {
        "_note": ("Citation FAITHFULNESS via leave-one-source-out (LOSO) entailment replay: a "
                  "faithful citation's own cited chunk is load-bearing — drop it and the claim is "
                  "no longer entailed by the remaining retrieved context; claims still entailed "
                  "without their own citation are post-rationalisation-suspect. Per-pattern "
                  "faithfulness = 1 - suspect/tested. Pure replay on saved trajectories; no model "
                  "calls. CONDITIONAL: strict LOSO needs saved multi-chunk context + saved "
                  "remainder NLI verdicts; if absent, strict block is `data_insufficient` and a "
                  "labelled on-disk lower-bound proxy is reported instead."),
        "citation": CITATION,
        "citation_finding": "~57% of LLM long-form citations are post-hoc rationalisations.",
        "seed": SEED,
        "method": "leave_one_source_out_nli_replay (citation_verifier.nli_verify_batch)",
        "strict_loso": strict,
        "proxy_lowerbound": proxy,
        "data_sufficient_for_strict_metric": strict.get("status") == "ok",
        "interpretation": ("If/when the LOSO replay parquet is materialised, a per-pattern "
                           "faithfulness rate well below 1 (≈0.43 if the arXiv:2412.18004 ~57% "
                           "post-rationalisation rate replicates here) would show citations are "
                           "largely decorative rather than load-bearing — sharpening the paper's "
                           "'bounded returns to orchestration' claim on the citation channel. "
                           "Until then the on-disk PROXY gives a labelled bound only."),
    }
    return block


def main():
    block = build()
    canon = json.loads(CANON.read_text())
    canon["citation_faithfulness"] = block
    # atomic write: serialise fully to a tmp string first, then os.replace, so a
    # serialisation failure cannot truncate the live canonical file. Append-only.
    txt = json.dumps(canon, indent=1)
    tmp = f"{CANON}.tmp"
    with open(tmp, "w") as fh:
        fh.write(txt)
    os.replace(tmp, CANON)

    s = block["strict_loso"]["status"]
    p = block["proxy_lowerbound"]
    print(f"citation_faithfulness: strict_loso={s}  (data_sufficient_for_strict={block['data_sufficient_for_strict_metric']})")
    if s == "ok":
        print(f"  cluster mean faithfulness={block['strict_loso']['cluster_mean_faithfulness']}")
    else:
        print(f"  -> strict LOSO pending; missing {block['strict_loso'].get('missing_input')}")
    print(f"  PROXY (lower-bound): n_cited={p.get('n_cited_total')} "
          f"post-rationalisation_LB={p.get('post_rationalisation_rate_lowerbound_overall')} "
          f"faithfulness_UB={p.get('cluster_mean_faithfulness_upperbound')}")
    print(f"  cite: {CITATION}")


if __name__ == "__main__":
    main()
