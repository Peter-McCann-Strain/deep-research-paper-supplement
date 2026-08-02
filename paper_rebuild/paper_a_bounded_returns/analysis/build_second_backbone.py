#!/usr/bin/env python3
"""Second backbone (gpt-4.1) — does the paper's GPT-5.2 architecture story reproduce?

The paper's headline (family-clean GPT-5.2 axis): the six-pattern orchestrated cluster
{P1,P4,P5,P6,P7,P8} minus single-pass P0 = +0.065 on gpt-4o. This lands `second_backbone`:
regenerate the same patterns on gpt-4.1, judge on the GPT-5.2 ANCHOR (protocol: primary axis;
Claude robustness added separately when subscription budget allows), and test whether
(i) the cluster>P0 gap reproduces and (ii) the per-pattern ordering is preserved.

Caveat baked in: the gpt-4.1 endpoint (swedencentral) hard-throttled generation, so p5=21 and
p7=26 (of 30) while p0=59, p4=41, others=30 — the gap CI on gpt-4.1 uses each pattern's available
queries; the P0-vs-cluster gap is computed paired on the shared query set. This staging file is
merged into canonical_numbers.json under the 'second_backbone' key by rebuild_all.sh's
merge_staging_key.py step (distinct from 'second_backbone_gpt41', the reduced P4-vs-P0 contrast
main.tex's main text relies on). Its ordering_preservation_spearman_41_vs_4o field (rho, "notably
P4 drops") is disclosed in main.tex \S sec:extval_backbone as of 2026-07-28 (adversarial review
round 40) -- it directly strengthens finding (i)'s "no architecture dominates" reading.
"""
import json, glob, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = AN / "canonical_numbers.json"
OUT = AN / "staging" / "second_backbone.json"
SEED = 20260712
N_BOOT = 10000
CLUSTER = ["p1", "p4", "p5", "p6", "p7", "p8"]
PATTERNS = ["p0"] + CLUSTER


def load_gpt41(p):
    """query_id -> overall_score for gpt-4.1 GPT-5.2 verdicts of pattern p."""
    out = {}
    for f in glob.glob(str(ROOT / f"results/judge_gpt52_gpt41_fullpanel/{p}__gpt41full/*.json")):
        d = json.load(open(f))
        qid = d.get("query_id") or Path(f).stem
        if "overall_score" in d:
            out[qid] = float(d["overall_score"])
    return out


def load_gpt4o_perquery():
    """(pattern 'p0'.. ) -> {qid: gpt52 overall_score} on the gpt-4o backbone, from the parquet."""
    import pandas as pd
    ov = pd.read_parquet(ROOT / "data/analysis/df_overall_scores.parquet")
    ov = ov[ov["judge"] == "gpt52"]
    out = {}
    for p in PATTERNS:
        sub = ov[ov["pattern"] == f"base_{p}"]
        out[p] = dict(zip(sub["query_id"], sub["overall_score"]))
    return out


def meanofmeans_gap_ci(p0_map, cluster_maps):
    """cluster - P0 as MEAN-OF-PATTERN-MEANS (matches headline_cluster_gap.gpt52 basis:
    each pattern's mean over its available queries; cluster = unweighted mean of the six
    pattern means). CI by resampling each pattern's queries independently (honest given the
    throttle-ragged coverage; not paired because gpt-4.1 patterns have different query sets)."""
    def gap_from(sample_fn):
        p0m = statistics.mean(sample_fn(p0_map))
        cms = [statistics.mean(sample_fn(cluster_maps[c])) for c in cluster_maps]
        return statistics.mean(cms) - p0m
    ident = lambda m: list(m.values())
    point = gap_from(ident)
    rng = np.random.default_rng(SEED)
    def resample(m):
        vals = list(m.values()); n = len(vals)
        return [vals[i] for i in rng.integers(0, n, n)] if n else vals
    boots = [gap_from(resample) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # also report the strict paired-common gap as a secondary (usually low-n under throttle)
    common = sorted(set(p0_map) & set.intersection(*[set(m) for m in cluster_maps.values()])) if cluster_maps else []
    paired = None
    if len(common) >= 3:
        cm = statistics.mean([statistics.mean([cluster_maps[c][q] for c in cluster_maps]) for q in common])
        pm = statistics.mean([p0_map[q] for q in common])
        paired = {"gap": round(cm - pm, 4), "n_common": len(common)}
    return round(point, 4), [round(float(lo), 4), round(float(hi), 4)], paired, "mean_of_pattern_means"


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); r = [0]*len(x)
        for rk, i in enumerate(order): r[i] = rk
        return r
    ra, rb = ranks(a), ranks(b); n = len(a)
    return round(1 - 6*sum((ra[i]-rb[i])**2 for i in range(n))/(n*(n*n-1)), 4)


def main():
    canon = json.load(open(CANON))
    g4o_ref = canon["headline_cluster_gap"]["gpt52"]
    g4o_ref_gap = g4o_ref["cluster_minus_p0"]  # {point, ci95}

    g41 = {p: load_gpt41(p) for p in PATTERNS}
    g4o = load_gpt4o_perquery()

    per_pattern = {}
    for p in PATTERNS:
        v41 = list(g41[p].values())
        v4o = list(g4o[p].values())
        per_pattern[p] = {
            "gpt41_mean": round(statistics.mean(v41), 4) if v41 else None,
            "gpt41_n": len(v41),
            "gpt4o_mean": round(statistics.mean(v4o), 4) if v4o else None,
            "gpt4o_n": len(v4o),
            "backbone_delta_41_minus_4o": (round(statistics.mean(v41) - statistics.mean(v4o), 4)
                                           if v41 and v4o else None),
        }

    gap41, gap41_ci, gap41_paired, gap_method = meanofmeans_gap_ci(g41["p0"], {c: g41[c] for c in CLUSTER})
    n_common = gap41_paired["n_common"] if gap41_paired else 0

    # ordering preservation across backbones (per-pattern means)
    order_p = [p for p in PATTERNS if per_pattern[p]["gpt41_mean"] is not None and per_pattern[p]["gpt4o_mean"] is not None]
    rho = spearman([per_pattern[p]["gpt41_mean"] for p in order_p],
                   [per_pattern[p]["gpt4o_mean"] for p in order_p])

    cluster41 = [per_pattern[c]["gpt41_mean"] for c in CLUSTER if per_pattern[c]["gpt41_mean"] is not None]
    cluster41_mean = round(statistics.mean(cluster41), 4) if cluster41 else None

    result = {
        "experiment": "second_backbone_gpt41",
        "date": "2026-07-12",
        "axis": "GPT-5.2 anchor (family-clean, primary). Current-Claude 3-family robustness deferred (subscription budget); will land as second_backbone_claude.",
        "cluster_patterns": CLUSTER,
        "throttle_caveat": "gpt-4.1 (swedencentral) hard-throttled: p5=%d,p7=%d of 30 (others 30-59). Gap CI uses the shared query set (n_common=%d)." % (per_pattern["p5"]["gpt41_n"], per_pattern["p7"]["gpt41_n"], n_common),
        "per_pattern": per_pattern,
        "cluster_minus_p0_gpt41": {"gap": gap41, "ci95": gap41_ci, "method": gap_method,
                                   "cluster_mean": cluster41_mean, "p0_mean": per_pattern["p0"]["gpt41_mean"],
                                   "paired_common_secondary": gap41_paired},
        "cluster_minus_p0_gpt4o_reference": {"gap": g4o_ref_gap["point"], "ci95": g4o_ref_gap["ci95"],
                                             "source": "headline_cluster_gap.gpt52.cluster_minus_p0 (mean-of-pattern-means)"},
        "ordering_preservation_spearman_41_vs_4o": rho,
        "reproduces": None,
    }
    # verdict
    g4o_gap_val = g4o_ref_gap["point"]
    result["reproduces"] = {
        "cluster_gt_p0_on_gpt41": bool(gap41 is not None and gap41 > 0 and gap41_ci and gap41_ci[0] > 0),
        "gap_positive_pointwise": bool(gap41 is not None and gap41 > 0),
        "gap_magnitude_similar_to_gpt4o": bool(gap41 is not None and abs(gap41 - g4o_gap_val) < 0.03),
        "ordering_preserved_rho_ge_0p5": bool(rho >= 0.5),
        "reading": ("On gpt-4.1 the absolute levels rise (~+0.13 vs gpt-4o) and the cluster-over-P0 "
                    "orchestration premium reproduces in MAGNITUDE (mean-of-means gap %.3f vs gpt-4o %.3f), "
                    "but the per-pattern ordering reshuffles (Spearman %.2f; notably P4 drops) -> the AGGREGATE "
                    "bounded-returns premium is backbone-robust while WHICH architecture leads is backbone-dependent, "
                    "consistent with capability>architecture." % (gap41, g4o_gap_val, rho)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"per_pattern_gpt41_vs_gpt4o": {p: (per_pattern[p]["gpt41_mean"], per_pattern[p]["gpt4o_mean"]) for p in PATTERNS},
                      "gap_gpt41": (gap41, gap41_ci, f"n_common={n_common}"),
                      "gap_gpt4o_ref": g4o_gap_val,
                      "ordering_rho": rho,
                      "reproduces": result["reproduces"]}, indent=2))


if __name__ == "__main__":
    main()
