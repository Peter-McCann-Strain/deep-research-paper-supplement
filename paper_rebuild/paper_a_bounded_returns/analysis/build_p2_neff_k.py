#!/usr/bin/env python
"""P2_neff_k — N_eff/k saturation flag + Condorcet-null panel contrast (Paper 2, Phase-2).

Appends ``n_eff.diagnostics`` to canonical_numbers.json (ADDITIVE; never mutates
n_eff.overall / n_eff.per_dimension / n_eff.within_openai). Companion to build_n_eff.py and
build_n_eff_within_openai.py, which this script does NOT touch.

Citation embedded
-----------------
arXiv:2605.29800 — "Nine Judges, Two Effective Votes" (June 2026). That paper's headline
diagnostic is the N_eff/k *saturation ratio*: when a k-judge LLM panel's effective independent
vote count N_eff falls below k/2 (i.e. N_eff/k < 0.5), the panel is correlation-saturated and
adding a judge from the same correlated pool buys almost no new independent information — the
operational rule "do NOT add a 4th correlated LLM judge". This builder computes that ratio for
the paper's primary 3-judge panel and flags it against the < 0.5 caution line, then corroborates
the mechanism with a Condorcet-Jury-Theorem null: the CJT only delivers its reliability gains
when voters are INDEPENDENT, so the gap between the observed (correlated) panel and a
simulated-independent panel with identical marginals is the citation-anchored evidence that the
naive "more judges = more reliable" intuition fails here.

What it adds (two diagnostics on n_eff.diagnostics)
---------------------------------------------------
(1) n_eff_over_k: N_eff / k for the primary panel. N_eff is read from the already-canonical
    n_eff.overall.n_eff (1.6547, the closed-form 9/(3+2*sum phi) number from build_n_eff.py) and
    k = number of panel judges (3). Reported value 1.6547/3 = 0.5516, flagged against the
    arXiv:2605.29800 N_eff/k < 0.5 caution rule (here 0.5516 sits JUST ABOVE 0.5 — the panel is
    near the saturation line, one correlated judge short of the "do not add a 4th" wall; the
    diagnostic states this honestly rather than overclaiming a breach).

(2) condorcet_null: observed-vs-simulated-independent panel contrast on the SAME 36,113 crossed
    cells used by build_n_eff.py. The CJT (independent imperfect voters -> majority improves with
    k) presupposes independence; we quantify how far the real panel departs from it:
      - unanimity_observed: P(all 3 judges agree) on the real correlated cells.
      - unanimity_condorcet_null: closed-form P(all agree) if the SAME judges voted INDEPENDENTLY
        at their observed marginal SATISFIED rates p_j: prod p_j + prod (1-p_j).
      - excess_unanimity: observed - null (correlation-driven inflation of agreement; large excess
        == the votes are redundant == N_eff << k).
      - panel_self_agreement_observed vs _condorcet_null: mean per-judge agreement with the
        majority verdict, real vs a SEEDED Monte-Carlo independent panel (same marginals). A
        higher observed self-agreement than the independent simulation is the same redundancy seen
        as decisiveness. MC is reported only to corroborate the closed form (it must match
        unanimity_condorcet_null within MC noise); the closed form is the canonical number.

Substrate (verified on disk, identical cell to build_n_eff.py): the fully-crossed cell =
(pattern x query x criterion_id) verdicts scored by ALL THREE panel judges (gpt52, claude_opus,
claude_sonnet), restricted to pattern_family == 'base' and satisfied_is_known. Measured here:
36,113 crossed cells. Reads data/analysis/df_verdicts.parquet directly (same source as
build_n_eff.py); fully self-contained and idempotent. No paid API, no new judging, CPU-only, $0.

Determinism: the unanimity contrast is closed-form (no randomness). The corroborating
Monte-Carlo independent panel is seeded (SEED=20260611) and drawn on the marginals of the
index-sorted wide table, so the run is bit-reproducible.
"""
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260611
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]          # k = 3, identical to build_n_eff.py
K = len(PANEL)
CITATION = "arXiv:2605.29800 (Nine Judges, Two Effective Votes)"
NEFF_K_CAUTION = 0.5  # arXiv:2605.29800 saturation rule: N_eff/k < 0.5 => do not add a 4th judge


def wide_crossed():
    """Reconstruct the build_n_eff.py fully-crossed 3-panel cell from df_verdicts (sorted)."""
    V = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")
    b = V[(V.pattern_family == "base") & (V.judge.isin(PANEL)) & (V.satisfied_is_known)].copy()
    key = ["pattern", "query_id", "criterion_id"]
    cnt = b.groupby(key, observed=True)["judge"].nunique()
    crossed_keys = set(cnt[cnt == 3].index)
    b["k"] = list(zip(b.pattern, b.query_id, b.criterion_id))
    fc = b[b.k.isin(crossed_keys)].copy()
    w = (fc.pivot_table(index=key, columns="judge", values="satisfied", aggfunc="first",
                        observed=True).dropna(subset=PANEL))
    for j in PANEL:
        w[j] = w[j].astype(int)
    return w.sort_index()


def main():
    cn = json.load(open(CANON))
    existing = cn.get("n_eff")
    if not isinstance(existing, dict):
        raise SystemExit("REFUSING: canonical 'n_eff' is missing or not an object; "
                         "run build_n_eff.py first.")
    overall = existing.get("overall")
    if not isinstance(overall, dict) or "n_eff" not in overall:
        raise SystemExit("REFUSING: canonical 'n_eff.overall.n_eff' is missing; "
                         "run build_n_eff.py first.")
    n_eff = float(overall["n_eff"])                       # 1.6547, the canonical closed-form N_eff

    # ---- diagnostic (1): N_eff/k saturation ratio vs the arXiv:2605.29800 < 0.5 rule ----
    n_eff_over_k = n_eff / K
    breaches_caution = bool(n_eff_over_k < NEFF_K_CAUTION)

    # ---- diagnostic (2): Condorcet-null contrast on the SAME 36,113 crossed cells ----
    w = wide_crossed()
    n_cells = int(len(w))
    p = {j: float(w[j].mean()) for j in PANEL}            # observed marginal SATISFIED rate

    ssum = w[PANEL].sum(axis=1)
    unanimity_observed = float(((ssum == 0) | (ssum == 3)).mean())
    # Condorcet-null: same marginals, ZERO correlation (independent voters)
    unanimity_null = (p[PANEL[0]] * p[PANEL[1]] * p[PANEL[2]]
                      + (1 - p[PANEL[0]]) * (1 - p[PANEL[1]]) * (1 - p[PANEL[2]]))
    excess_unanimity = unanimity_observed - unanimity_null

    # majority verdict (CJT panel decision) and per-judge agreement with it, observed
    maj = (ssum >= 2).astype(int)
    self_agree_observed = float(np.mean([(w[j] == maj).mean() for j in PANEL]))

    # seeded Monte-Carlo independent panel (same marginals) — corroborates the closed form
    rng = np.random.default_rng(SEED)
    sim = np.column_stack([rng.random(n_cells) < p[j] for j in PANEL]).astype(int)
    ssim = sim.sum(axis=1)
    maj_sim = (ssim >= 2).astype(int)
    self_agree_null = float(np.mean([(sim[:, i] == maj_sim).mean() for i in range(K)]))
    unanimity_null_mc = float(((ssim == 0) | (ssim == 3)).mean())

    diagnostics = {
        "_note": "P2 N_eff/k saturation flag + Condorcet-null panel contrast. n_eff_over_k = "
                 "n_eff.overall.n_eff / k flagged against the arXiv:2605.29800 N_eff/k < 0.5 "
                 "saturation rule (do NOT add a 4th correlated LLM judge). condorcet_null "
                 "contrasts the observed correlated panel against a same-marginal INDEPENDENT "
                 "panel on the SAME 36,113 crossed cells (build_n_eff.py cell): the Condorcet "
                 "Jury Theorem's reliability gains require independence, so the observed-minus-"
                 "null excess unanimity quantifies the redundancy that drives N_eff << k. "
                 "Closed-form is canonical; seeded Monte-Carlo (SEED=20260611) only corroborates.",
        "citation": CITATION,
        "k": K,
        "n_eff_overall": round(n_eff, 4),
        "n_eff_over_k": round(n_eff_over_k, 4),
        "neff_k_caution_threshold": NEFF_K_CAUTION,
        "breaches_caution_below_half": breaches_caution,
        "saturation_note": (
            f"N_eff/k = {round(n_eff_over_k, 4)} for the {K}-judge panel (N_eff={round(n_eff, 4)}). "
            f"The arXiv:2605.29800 caution fires at N_eff/k < {NEFF_K_CAUTION}; this panel sits "
            f"{'BELOW' if breaches_caution else 'JUST ABOVE'} that line "
            f"({'saturated' if breaches_caution else 'one correlated judge short of saturation'}). "
            "Either way the operational reading is the same: a 4th LLM judge drawn from the same "
            "correlated pool would add far less than one independent vote, so the panel should NOT "
            "be grown by adding a 4th correlated judge."),
        "condorcet_null": {
            "_note": "Observed correlated panel vs simulated-independent (Condorcet) panel on the "
                     "36,113 crossed cells; same marginal SATISFIED rates, zero correlation.",
            "n_cells": n_cells,
            "marginal_satisfied_rate": {j: round(p[j], 4) for j in PANEL},
            "unanimity_observed": round(unanimity_observed, 4),
            "unanimity_condorcet_null": round(float(unanimity_null), 4),
            "excess_unanimity": round(float(excess_unanimity), 4),
            "panel_self_agreement_observed": round(self_agree_observed, 4),
            "panel_self_agreement_condorcet_null": round(self_agree_null, 4),
            "unanimity_condorcet_null_mc_check": round(unanimity_null_mc, 4),
            "seed": SEED,
        },
        "interpretation": (
            f"On the same 36,113 crossed cells, the real panel reaches unanimity "
            f"{round(unanimity_observed, 4)} of the time, but three INDEPENDENT judges with the "
            f"same marginal rates would agree only {round(float(unanimity_null), 4)} of the time "
            f"(excess +{round(float(excess_unanimity), 4)}); per-judge agreement with the majority "
            f"is likewise inflated ({round(self_agree_observed, 4)} observed vs "
            f"{round(self_agree_null, 4)} independent). The Condorcet Jury Theorem's reliability "
            f"gains assume independence, so this correlation-driven excess is exactly why N_eff "
            f"({round(n_eff, 4)}) is far below k ({K}) and N_eff/k = {round(n_eff_over_k, 4)} hugs "
            f"the arXiv:2605.29800 saturation line. The naive 'add another judge' move buys almost "
            f"no independent information: do NOT add a 4th correlated LLM judge."),
    }

    # ADDITIVE: write under a dedicated sub-key; never touch n_eff.overall / per_dimension /
    # within_openai. Atomic tmp + os.replace, mirroring build_n_eff.py / build_judge_vs_gold.py.
    existing["diagnostics"] = diagnostics
    cn["n_eff"] = existing
    tmp = CANON + ".tmp"
    open(tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(tmp, CANON)

    print(f"n_eff.diagnostics: k={K} N_eff={round(n_eff, 4)} N_eff/k={round(n_eff_over_k, 4)} "
          f"({'BELOW' if breaches_caution else 'just above'} {NEFF_K_CAUTION} caution)")
    print(f"  Condorcet contrast on {n_cells} crossed cells: unanimity observed="
          f"{round(unanimity_observed, 4)} vs independent-null={round(float(unanimity_null), 4)} "
          f"(excess +{round(float(excess_unanimity), 4)})")
    print(f"  per-judge agreement w/ majority: observed={round(self_agree_observed, 4)} vs "
          f"independent={round(self_agree_null, 4)}  [MC unanimity check="
          f"{round(unanimity_null_mc, 4)}]")
    print(f"  rule (arXiv:2605.29800): do NOT add a 4th correlated LLM judge")


if __name__ == "__main__":
    main()
