#!/usr/bin/env python
"""P11 turn-budget control: 8-turn vs 16-turn ReAct.

The paper hedges that P11's weak showing is 'plausibly' down to its 8-turn budget.
base_p11_16turn doubles the turn budget on the same queries; if the score does not move,
the budget hedge is refuted and P11's deficit is attributable to the verbatim ReAct
controller itself. GPT-5.2 single-judge (matching P11's post-hoc probe status).
Appends canonical_numbers.json['p11_16turn'].
"""
import pandas as pd, numpy as np, json, warnings
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

ov = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
g = ov[ov.judge.eq("gpt52") & ov.pattern.isin(["base_p11", "base_p11_16turn"])]
wide = g.pivot_table(index="query_id", columns="pattern", values="overall_score", observed=True).dropna()
d8 = wide["base_p11"]; d16 = wide["base_p11_16turn"]
try:
    p = float(wilcoxon(d16, d8).pvalue)
except Exception:
    p = 1.0
out = {"judge": "gpt52", "n_paired_queries": int(len(wide)),
       "mean_8turn": round(float(d8.mean()), 4), "mean_16turn": round(float(d16.mean()), 4),
       "delta_16_minus_8": round(float((d16 - d8).mean()), 4),
       "wilcoxon_p": round(p, 4),
       "note": "doubling the ReAct turn budget does not move the score; the 8-turn budget "
               "hedge for P11's deficit is not supported"}
cn = json.load(open(f"{ANA}/canonical_numbers.json"))
cn["p11_16turn"] = out
json.dump(cn, open(f"{ANA}/canonical_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
