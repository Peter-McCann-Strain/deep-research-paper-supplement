#!/usr/bin/env python
"""Standardized per-judge orchestration gaps (reviewer objection: judge SCALE vs judge BIAS).

The paper reports the P0-to-cluster gain as 0.146 (three-judge panel) vs 0.065 (GPT-5.2
alone) and glosses the difference as same-lab inflation by the two Claude judges. The
objection: Claude judges use a wider score range, so a RAW-gap comparison conflates each
judge's scale with its bias, contradicting the paper's own "judges certify orderings, not
levels" doctrine. This script computes, per judge (gpt52, claude_opus, claude_sonnet):

  raw_gap            mean over queries of (cluster_q - p0_q), cluster_q = mean of the six
                     top-cluster patterns' scores for that judge on query q (paired by query)
  cohen_d_paired     raw_gap / SD(cluster_q - p0_q): the paired effect size, scale-free
  z_gap              raw_gap / SD(all base-pattern cells for that judge): the gap after
                     z-scoring the judge's scores within-judge (z-scoring is linear, so the
                     within-judge z-scored gap is exactly raw_gap / within-judge SD)

plus seeded query-bootstrap CIs on each standardized quantity AND on the paired per-query
DIFFERENCE of each Claude judge's standardized gap vs GPT-5.2's (the direct test of
"disproportionately larger after standardization"). Sonnet uses overall_score_recomputed
(corrupted overall_score, per DATA_DICTIONARY), same as build_numbers.py.

Writes NEW canonical key `judge_scale_standardized_gaps` (atomic tmp+os.replace; never
overwrites existing keys). Deterministic: seeded generator on SORTED query lists.
"""
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANONICAL = f"{ANA}/canonical_numbers.json"
SEED = 20260702
N_BOOT = 5000

PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
CLUSTER = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]

ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
# sonnet correction (identical to build_numbers.py corrected_overall)
ovc = ov["overall_score"].copy()
m = ov["judge"].eq("claude_sonnet")
ovc = ovc.where(~m, ov["overall_score_recomputed"])
ov["ovc"] = ovc

# eleven canonical base patterns only; excludes base_p11/base_p12, the post-hoc
# single-judge-by-design probes (adversarial review 2026-07-28, round 13: sd_all below
# ("within-judge SD over ALL base cells") was silently pooling them in, contaminating the
# z-scored-gap denominator even though the cluster/p0 columns it's paired against were
# already safely restricted by explicit column name).
base = ov[ov.pattern.str.match(r"^base_p([0-9]|10)$") & ov.judge.isin(PANEL)].copy()

# ---- per-judge per-query p0 and cluster scores ----
per_judge = {}
for j in PANEL:
    d = base[base.judge.eq(j)]
    w = d.pivot_table(index="query_id", columns="pattern", values="ovc", observed=True)
    w = w.sort_index()
    p0 = w["base_p0"]
    cl = w[[c for c in CLUSTER if c in w.columns]].mean(axis=1)  # mean over available cluster patterns
    ok = p0.notna() & cl.notna()
    diffs = (cl - p0)[ok]
    sd_all = float(d["ovc"].std(ddof=1))          # within-judge SD over ALL base cells (z basis)
    per_judge[j] = {"diffs": diffs, "p0": p0[ok], "cl": cl[ok], "sd_all": sd_all}

common_q = sorted(set.intersection(*[set(per_judge[j]["diffs"].index) for j in PANEL]))
rng = np.random.default_rng(SEED)
nq = len(common_q)
boot_idx = rng.integers(0, nq, size=(N_BOOT, nq))   # ONE shared set of query draws -> paired boots

def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

stats, boots = {}, {}
for j in PANEL:
    dif = per_judge[j]["diffs"].loc[common_q].to_numpy()
    sd_all = per_judge[j]["sd_all"]
    raw = float(dif.mean()); sd_d = float(dif.std(ddof=1))
    d_boot = np.array([dif[ix].mean() / dif[ix].std(ddof=1) for ix in boot_idx])
    z_boot = np.array([dif[ix].mean() / sd_all for ix in boot_idx])
    raw_boot = np.array([dif[ix].mean() for ix in boot_idx])
    boots[j] = {"d": d_boot, "z": z_boot, "raw": raw_boot}
    stats[j] = {
        "n_queries_paired": nq,
        "p0_mean": round(float(per_judge[j]["p0"].loc[common_q].mean()), 4),
        "cluster_mean": round(float(per_judge[j]["cl"].loc[common_q].mean()), 4),
        "raw_gap": round(raw, 4),
        "raw_gap_ci95": ci(raw_boot),
        "sd_paired_diffs": round(sd_d, 4),
        "sd_within_judge_all_base_cells": round(sd_all, 4),
        "cohen_d_paired": round(raw / sd_d, 4),
        "cohen_d_paired_ci95": ci(d_boot),
        "z_gap_within_judge_sd_units": round(raw / sd_all, 4),
        "z_gap_ci95": ci(z_boot),
    }

# ---- direct paired tests: Claude judge standardized gap minus GPT-5.2's ----
deltas = {}
for j in ["claude_opus", "claude_sonnet"]:
    for metric in ["d", "z", "raw"]:
        db = boots[j][metric] - boots["gpt52"][metric]
        key = {"d": "cohen_d", "z": "z_gap", "raw": "raw_gap"}[metric]
        obs = {"cohen_d": stats[j]["cohen_d_paired"] - stats["gpt52"]["cohen_d_paired"],
               "z_gap": stats[j]["z_gap_within_judge_sd_units"] - stats["gpt52"]["z_gap_within_judge_sd_units"],
               "raw_gap": stats[j]["raw_gap"] - stats["gpt52"]["raw_gap"]}[key]
        lo, hi = np.percentile(db, [2.5, 97.5])
        deltas.setdefault(j, {})[f"delta_{key}_vs_gpt52"] = {
            "delta": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_0": bool(lo > 0 or hi < 0)}

panel_raw = float(np.mean([stats[j]["raw_gap"] for j in PANEL]))
claude_share_raw = float((stats["claude_opus"]["raw_gap"] + stats["claude_sonnet"]["raw_gap"])
                         / (3 * panel_raw))
sum_d = sum(stats[j]["cohen_d_paired"] for j in PANEL)
claude_share_d = float((stats["claude_opus"]["cohen_d_paired"]
                        + stats["claude_sonnet"]["cohen_d_paired"]) / sum_d)
sum_z = sum(stats[j]["z_gap_within_judge_sd_units"] for j in PANEL)
claude_share_z = float((stats["claude_opus"]["z_gap_within_judge_sd_units"]
                        + stats["claude_sonnet"]["z_gap_within_judge_sd_units"]) / sum_z)

any_excl = {j: {k: v["excludes_0"] for k, v in deltas[j].items()} for j in deltas}
equalizes = not any(deltas[j][k]["excludes_0"] for j in deltas for k in deltas[j]
                    if k != "delta_raw_gap_vs_gpt52")

block = {
    "_note": (
        "Reviewer-objection control for the 0.146-panel vs 0.065-GPT-5.2 P0-to-cluster gap "
        "claim: raw per-judge gaps conflate judge scale with judge bias. Per judge we report "
        "the paired Cohen's d over queries (gap / SD of paired per-query cluster-P0 diffs) and "
        "the within-judge z-scored gap (gap / SD of that judge's scores over all base-pattern "
        "cells; z-scoring is linear so this equals the gap recomputed on z-scored scores). "
        "Sonnet corrected via overall_score_recomputed. Cluster = six-pattern top cluster "
        "{p1,p4,p5,p6,p7,p8}, cluster_q = per-judge mean over available cluster patterns, "
        "paired by query with P0. delta_* vs gpt52 use a SHARED seeded query-bootstrap "
        "(same draws for all judges -> paired CIs on the difference)."),
    "seed": SEED, "n_boot": N_BOOT, "n_queries_common": nq,
    "cluster_patterns": CLUSTER,
    "per_judge": stats,
    "delta_vs_gpt52": deltas,
    "panel_raw_gap": round(panel_raw, 4),
    "claude_share_of_panel_raw_gap": round(claude_share_raw, 4),
    "claude_share_of_summed_cohen_d": round(claude_share_d, 4),
    "claude_share_of_summed_z_gap": round(claude_share_z, 4),
    "standardized_gaps_equalize": bool(equalizes),
    "delta_excludes_0_map": any_excl,
}

cn = json.load(open(CANONICAL))
cn["judge_scale_standardized_gaps"] = block  # this script owns and fully regenerates this key
tmp = CANONICAL + ".tmp"
with open(tmp, "w") as fh:
    fh.write(json.dumps(cn, indent=1))
os.replace(tmp, CANONICAL)
print(json.dumps({k: v for k, v in block.items() if k != "_note"}, indent=1))
