#!/usr/bin/env python
"""Bracketing sensitivity for the E14 retrieval-vs-utilisation decomposition.

The strict rule counts only 'supports'. Any partial-support rubric can promote claims
only from 'neutral' (contradicts/no_source stay unsupported), so scoring
(supports + neutral)/n_claims gives an UPPER envelope on the verified-support rate under
any partial-support rule. Recomputing the oracle-minus-live retrieval component under
that envelope bounds how much the strict rule's floor compression could be hiding.

Lands canonical key: e14_oracle_entail.neutral_bracket
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
A = ROOT / "data/analysis"
ANA = ROOT / "paper_rebuild/paper_a_bounded_returns/analysis"
CLUSTER = ["p1", "p4", "p5", "p7", "p8"]  # paired-decomposition five (P6 has no base claims)
SEED = 20260702
N_BOOT = 5000

base = pd.read_parquet(A / "df_c0_per_report.parquet")
orac = pd.read_parquet(A / "df_e14_oracle_per_report.parquet")
for df in (base, orac):
    df["vfa_upper"] = (df.n_supports + df.n_neutral) / df.n_claims.clip(lower=1)

def component(rule_col):
    comps, pairs = [], []
    for p in CLUSTER:
        b = base[base.pattern == f"base_{p}"].set_index("query_id")[rule_col]
        o = orac[orac.pattern == f"oracle_t1_{p}"].set_index("query_id")[rule_col]
        common = sorted(set(b.index) & set(o.index))
        for q in common:
            pairs.append(float(o[q] - b[q]))
        comps.append(float(o[common].mean() - b[common].mean()))
    return np.array(comps), np.array(pairs)

comps_strict, pairs_strict = component("verified_factual_accuracy")
comps_upper, pairs_upper = component("vfa_upper")
rng = np.random.default_rng(SEED)
boots = [np.mean(rng.choice(pairs_upper, size=len(pairs_upper), replace=True))
         for _ in range(N_BOOT)]
lo, hi = np.percentile(boots, [2.5, 97.5])
out = {
    "rule": "supports+neutral over n_claims (upper envelope on any partial-support rubric)",
    "cluster_retrieval_component_strict_check": round(float(np.mean(comps_strict)), 4),
    "cluster_retrieval_component_upper": round(float(np.mean(comps_upper)), 4),
    "cluster_retrieval_component_upper_ci95": [round(float(lo), 4), round(float(hi), 4)],
    "n_pairs": int(len(pairs_upper)), "cluster": CLUSTER, "seed": SEED, "n_boot": N_BOOT,
}
store = json.load(open(ANA / "canonical_numbers.json"))
if "neutral_bracket" in store["e14_oracle_entail"]:
    assert store["e14_oracle_entail"]["neutral_bracket"] == out, "value drift vs landed key"
    print("already landed, verified identical"); raise SystemExit(0)
store["e14_oracle_entail"]["neutral_bracket"] = out
json.dump(store, open(ANA / "canonical_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
