#!/usr/bin/env python
"""Capability gap on the contamination-clean residual (E6 extension).

The E6 detector flags 73/90 queries (results/contamination_e6/contaminated_queries.json,
citation basis); the existing 17-query clean-residual check (canonical e6_decontamination)
protects only the ORCHESTRATION contrast (cluster vs P0). The paper concedes contamination
is not symmetric for the frontier-vs-7B contrast (finding ii) yet runs no control there.
This script computes, on the 17 clean queries, the two capability gaps:

  P0 - P9        (same architecture, GPT-4o vs Qwen2.5-7B; full set ~ 0.23)
  cluster - P9   (six-pattern top cluster mean vs P9;      full set ~ 0.38)

Three-judge panel mean per query (gpt52 + claude_opus + claude_sonnet, sonnet corrected via
overall_score_recomputed, per DATA_DICTIONARY), paired by query; cluster_q = mean over the
cluster patterns with a report on q. Seeded query-bootstrap percentile CIs. A gpt52-only
sensitivity is included because the same-lab-judge concern (see judge_scale_standardized_gaps)
applies to level gaps too. Full-90 references recomputed on the identical basis.

Writes NEW canonical subkey contamination.clean_residual_capability (atomic tmp+os.replace;
the top-level `contamination` rate-regression key from build_contamination.py --finalize does
not exist yet, so the parent dict is created holding only this subkey). Deterministic: seeded
generator on SORTED query lists.
"""
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANONICAL = f"{ANA}/canonical_numbers.json"
CONTAM = f"{ROOT}/results/contamination_e6/contaminated_queries.json"
SEED = 20260702
N_BOOT = 5000

PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
CLUSTER = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]

ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
ovc = ov["overall_score"].copy()
msk = ov["judge"].eq("claude_sonnet")
ovc = ovc.where(~msk, ov["overall_score_recomputed"])
ov["ovc"] = ovc

contam = set(json.load(open(CONTAM))["contaminated_query_set"])
all_q = sorted(ov[ov.pattern.str.match(r"^base_p\d+$")].query_id.unique())
clean_q = sorted(set(all_q) - contam)

base = ov[ov.pattern.isin(CLUSTER + ["base_p0", "base_p9"]) & ov.judge.isin(PANEL)].copy()

def per_query_scores(judges):
    """query -> {pattern: mean score over the given judges} (panel mean per cell first)."""
    d = base[base.judge.isin(judges)]
    cell = d.groupby(["pattern", "query_id"], observed=True)["ovc"].mean()
    w = cell.unstack("pattern").sort_index()
    out = pd.DataFrame(index=w.index)
    out["p0"] = w["base_p0"]
    out["p9"] = w["base_p9"]
    out["p4"] = w["base_p4"]
    out["cluster"] = w[[c for c in CLUSTER if c in w.columns]].mean(axis=1)
    return out

def gaps_on(qids, judges, rng):
    tbl = per_query_scores(judges).loc[[q for q in qids]].dropna(subset=["p9"])
    res = {}
    for name, col in [("p0_minus_p9", "p0"), ("p4_minus_p9", "p4"), ("cluster_minus_p9", "cluster")]:
        sub = tbl[[col, "p9"]].dropna()
        diffs = (sub[col] - sub["p9"]).to_numpy()
        n = len(diffs)
        boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[name] = {"gap": round(float(diffs.mean()), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "excludes_0": bool(lo > 0 or hi < 0), "n_queries_paired": int(n),
                     "sd_paired_diffs": round(float(diffs.std(ddof=1)), 4)}
    res["means"] = {k: round(float(tbl[k].mean()), 4) for k in ["p0", "p9", "p4", "cluster"]}
    return res

# coverage note on the clean slice
cov = {p: int(base[base.pattern.eq(p) & base.query_id.isin(clean_q)]
              .query_id.nunique()) for p in ["base_p0", "base_p9"] + CLUSTER}

rng = np.random.default_rng(SEED)
clean_panel = gaps_on(clean_q, PANEL, rng)
clean_gpt52 = gaps_on(clean_q, ["gpt52"], rng)
full_panel = gaps_on(all_q, PANEL, rng)

survives = {
    "p0_minus_p9": bool(clean_panel["p0_minus_p9"]["excludes_0"]
                        and clean_panel["p0_minus_p9"]["gap"] > 0),
    "p4_minus_p9": bool(clean_panel["p4_minus_p9"]["excludes_0"]
                        and clean_panel["p4_minus_p9"]["gap"] > 0),
    "cluster_minus_p9": bool(clean_panel["cluster_minus_p9"]["excludes_0"]
                             and clean_panel["cluster_minus_p9"]["gap"] > 0),
}

block = {
    "_note": (
        "Frontier-vs-7B capability gaps recomputed on the E6 contamination-clean residual "
        "(the 17 queries the citation-basis detector does NOT flag), extending the "
        "e6_decontamination control from the orchestration contrast to finding (ii). "
        "Three-judge panel mean per query (sonnet corrected), paired by query with P9; "
        "cluster = six-pattern top cluster {p1,p4,p5,p6,p7,p8}. Full-90 references "
        "recomputed on the identical basis. gpt52_only = family-clean single-judge "
        "sensitivity. Seeded query bootstrap, percentile CIs."),
    "detector_source": "results/contamination_e6/contaminated_queries.json (basis=citation)",
    "seed": SEED, "n_boot": N_BOOT,
    "n_queries_flagged": len(contam & set(all_q)),
    "n_queries_clean": len(clean_q),
    "clean_query_coverage": cov,
    "clean_panel3": clean_panel,
    "clean_gpt52_only": clean_gpt52,
    "full90_panel3_reference": full_panel,
    "survives_on_clean_slice": survives,
}

cn = json.load(open(CANONICAL))
parent = cn.setdefault("contamination", {})
parent["clean_residual_capability"] = block  # this script owns and fully regenerates this key
tmp = CANONICAL + ".tmp"
with open(tmp, "w") as fh:
    fh.write(json.dumps(cn, indent=1))
os.replace(tmp, CANONICAL)
print(json.dumps({k: v for k, v in block.items() if k != "_note"}, indent=1))
