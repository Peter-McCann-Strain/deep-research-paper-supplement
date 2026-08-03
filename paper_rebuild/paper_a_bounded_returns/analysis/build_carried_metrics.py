#!/usr/bin/env python
"""Carried / cross-phase metrics, made regenerable with explicit provenance (audit m5/CG-5).

Several load-bearing statistics were "carried" from earlier eval phases and not regenerable by
the paper build chain; one (Cramer's V) traced to a file that does not actually contain it.
This script recomputes what is cheaply recomputable from documented inputs, records the rest
with a one-line provenance pointer to a file that DOES contain the number, and emits
canonical_numbers.json['carried_metrics'] so every carried figure is auditable.
"""
import json
import numpy as np

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

# --- Cramer's V for family x failure-mode (fixes broken provenance) -----------------------
# Contingency counts are the documented input in reports/phase4_failures/chi2_family_test.md
# (GPT-4o vs Local7B over 10 failure modes). We recompute chi2 + Cramer's V here so the V is
# derived in-code from the source counts, and assert chi2 matches the documented 992.201.
CONTINGENCY = {  # provenance: reports/phase4_failures/chi2_family_test.md
    "GPT-4o":  [797, 190, 109, 724, 1, 147, 1310, 40, 47, 260],
    "Local7B": [38, 516, 30, 216, 1, 99, 413, 24, 15, 197],
}
M = np.array([CONTINGENCY["GPT-4o"], CONTINGENCY["Local7B"]], dtype=float)
N = M.sum()
row = M.sum(1, keepdims=True); col = M.sum(0, keepdims=True)
exp = row @ col / N
chi2 = float(((M - exp) ** 2 / exp).sum())
r, c = M.shape
cramers_v = float(np.sqrt(chi2 / (N * (min(r, c) - 1))))
assert abs(chi2 - 992.201) < 0.5, f"family chi2 drifted from documented 992.201: {chi2}"

# --- DR-Judge training set size (regenerable by line count) -------------------------------
import os
train = f"{ROOT}/data/dr_judge_training/train.jsonl"
drjudge_n = sum(1 for _ in open(train)) if os.path.exists(train) else None

# --- MDE80 (now computed in build_oracle_factual_tost -> canonical.oracle.factual_tost) ----
cn = json.load(open(f"{ANA}/canonical_numbers.json"))
mde80 = cn.get("oracle", {}).get("factual_tost", {}).get("mde80_6cluster")

out = {
    "_note": "Carried cross-phase metrics with provenance; recomputed where cheap, else pinned to a "
             "source file that contains the number. Audit m5/CG-5.",
    "cramers_v_family_failuremode": {
        "value": round(cramers_v, 4), "chi2": round(chi2, 3), "df": int((r - 1) * (c - 1)), "N": int(N),
        "provenance": "recomputed in build_carried_metrics.py from the contingency counts documented in "
                      "reports/phase4_failures/chi2_family_test.md",
        "note": "large effect (V~0.44); the cited md reports chi2 but not V, so V is regenerated here.",
    },
    "lr_pattern_x_source": {
        "value": 202.25, "df": 44, "p": 0.0,
        "provenance": "reports/phase2_statistics/04_stratification.md (regenerable by the phase2 stratification script)",
        "note": "load-bearing for the source-conditional / competence-conditional premium.",
    },
    "lr_pattern_x_difficulty": {
        "value": 35.67, "df": 22, "p": 0.03296,
        "provenance": "reports/phase2_statistics/04_stratification.md",
    },
    "mde80_oracle_factual_6cluster": {
        "value": mde80, "provenance": "canonical_numbers.json oracle.factual_tost (build_oracle_factual_tost.py)",
    },
    "drjudge_train_n": {
        "value": drjudge_n, "provenance": "wc -l data/dr_judge_training/train.jsonl",
    },
}
cn["carried_metrics"] = out
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
print(f"Cramer's V = {cramers_v:.4f} (chi2={chi2:.1f}, N={int(N)}); LR_src=202.25/44; "
      f"MDE80={mde80}; DR-Judge n={drjudge_n}")
