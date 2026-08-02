#!/usr/bin/env python3
"""second_backbone_claude — current-Claude robustness for the gpt-4.1 second backbone.

The banked `second_backbone` (GPT-5.2 anchor) found the cluster-over-P0 orchestration premium
REPRODUCES on gpt-4.1 (mean-of-means gap +0.056 [0.020,0.093] vs gpt-4o +0.066). This lands the
current-Claude robustness cohort (Opus 4.8, Sonnet 5): does the cluster>P0 premium survive under
the two Claude judges too? Same mean-of-pattern-means basis as headline_cluster_gap. Within-arm
(cluster vs P0) contrasts cancel a per-judge offset, so no J0 adjustment. STAGING only.
"""
import json, glob, statistics
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "second_backbone_claude.json"
SEED = 20260712
N_BOOT = 10000
CLUSTER = ["p1", "p4", "p5", "p6", "p7", "p8"]
PATTERNS = ["p0"] + CLUSTER
FAMS = {"opus48": "claude-opus-4-8", "sonnet5": "claude-sonnet-5"}


def load(fam, p):
    out = {}
    for f in glob.glob(str(ROOT / f"results/judge_gpt41_{fam}/{p}__gpt41full/*.json")):
        d = json.load(open(f))
        qid = d.get("query_id") or Path(f).stem
        if "overall_score" in d:
            out[qid] = float(d["overall_score"])
    return out


def gap_ci(p0_map, cluster_maps):
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
    return round(point, 4), [round(float(lo), 4), round(float(hi), 4)]


def main():
    canon = json.load(open(AN / "canonical_numbers.json"))
    anchor = canon["second_backbone"]["cluster_minus_p0_gpt41"]  # gpt-5.2 gap

    fams = {}
    for fam in FAMS:
        maps = {p: load(fam, p) for p in PATTERNS}
        per_pat = {p: {"mean": round(statistics.mean(maps[p].values()), 4) if maps[p] else None,
                       "n": len(maps[p])} for p in PATTERNS}
        gap, ci = gap_ci(maps["p0"], {c: maps[c] for c in CLUSTER})
        fams[fam] = {
            "judge_model": FAMS[fam],
            "per_pattern_mean": {p: per_pat[p]["mean"] for p in PATTERNS},
            "per_pattern_n": {p: per_pat[p]["n"] for p in PATTERNS},
            "cluster_minus_p0_gap": gap,
            "cluster_minus_p0_ci95": ci,
            "cluster_gt_p0": bool(gap > 0 and ci[0] > 0),
        }

    result = {
        "experiment": "second_backbone_gpt41_claude_robustness",
        "date": "2026-07-14",
        "note": "current-Claude robustness cohort for `second_backbone`; 235/237 per judge (82de3e92 AUP quarantine on p0+p4). Same mean-of-pattern-means basis. J0 offset NOT applied (within-arm cluster-vs-P0 cancels it).",
        "anchor_gpt52_gap": {"gap": anchor["gap"], "ci95": anchor["ci95"]},
        "claude_families": fams,
        "three_family_verdict": {
            "cluster_gt_p0_all_families": bool(anchor["ci95"][0] > 0 and all(fams[f]["cluster_gt_p0"] for f in FAMS)),
            "gaps": {"gpt52": anchor["gap"], **{f: fams[f]["cluster_minus_p0_gap"] for f in FAMS}},
            "reading": ("Cross-family robustness of the second-backbone finding: the cluster-over-P0 orchestration "
                        "premium on gpt-4.1 holds under the GPT-5.2 anchor AND both current Claude judges iff all "
                        "three gap CIs exclude 0. Claude runs more lenient in absolute level (J0) but the within-arm "
                        "gap is offset-invariant."),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"anchor_gpt52": (anchor["gap"], anchor["ci95"]),
                      **{f: (fams[f]["cluster_minus_p0_gap"], fams[f]["cluster_minus_p0_ci95"],
                             f"cluster>P0={fams[f]['cluster_gt_p0']}") for f in FAMS},
                      "3family_verdict": result["three_family_verdict"]["cluster_gt_p0_all_families"]}, indent=2))


if __name__ == "__main__":
    main()
