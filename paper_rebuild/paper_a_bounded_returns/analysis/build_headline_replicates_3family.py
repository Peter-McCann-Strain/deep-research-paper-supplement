#!/usr/bin/env python3
"""headline_replicates_3family — current-Claude robustness for the replicate-run ranking.

The banked `headline_replicated` (GPT-5.2, E2 30q subset) shows the single-run P0->cluster
ranking is not a run-noise artefact. This lands the two current Claude judges on the SAME
replicate reports and asks: is the replicate-based pattern ORDERING judge-robust?

CAVEAT (honest): the replicate cells are ragged in N (p0=7, p4=1, p6=32, p8=36, p11=94), so
this is a rank-agreement robustness check, NOT a powered per-pattern re-derivation. p4 (n=1) is
descriptive-only. Per-pattern = replicate cells pooled. J0 offset NOT applied (within-substrate
rank contrasts cancel a per-judge level offset). $0 CPU. STAGING only.

NB base_p6__rep1 Claude verdicts were re-parsed 2026-07-25 after a salvage-script verdict-format
bug (raw used verdict:"True"/"False", salvage matched "SATISFIED") flipped them to all-False;
corrected before this build.
"""
import json, glob, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "headline_replicates_3family.json"

JUDGE_DIRS = {
    "gpt52": ROOT / "results" / "judge_gpt52_headline_replicates",
    "opus48": ROOT / "results" / "judge_headline_replicates_opus48",
    "sonnet5": ROOT / "results" / "judge_headline_replicates_sonnet5",
}
CELLS = ["base_p0__rep1", "base_p4__rep1", "base_p6__rep1", "base_p6__rep2",
         "base_p8__rep1", "base_p8__rep2", "base_p11__rep1", "base_p11__rep2"]
QUAR = "82de3e92"


def pattern_of(cell):
    return cell.split("__")[0]  # base_p6__rep1 -> base_p6


def load(judge_dir, cell):
    out = {}
    for f in glob.glob(str(judge_dir / cell / "*.json")):
        qid = Path(f).stem
        if qid.startswith(QUAR):
            continue
        d = json.load(open(f))
        if "overall_score" in d:
            out[qid] = float(d["overall_score"])
    return out


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    if n < 2: return None
    return round(1 - 6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1)), 4)


def main():
    per_judge = {}
    for jn, jd in JUDGE_DIRS.items():
        by_cell = {c: load(jd, c) for c in CELLS}
        by_pattern = defaultdict(list)
        for c in CELLS:
            by_pattern[pattern_of(c)].extend(by_cell[c].values())
        per_judge[jn] = {
            "per_cell_mean": {c: (round(statistics.mean(by_cell[c].values()), 4) if by_cell[c] else None) for c in CELLS},
            "per_cell_n": {c: len(by_cell[c]) for c in CELLS},
            "per_pattern_mean": {p: round(statistics.mean(v), 4) for p, v in by_pattern.items() if v},
            "per_pattern_n": {p: len(v) for p, v in by_pattern.items()},
        }

    # cross-judge rank agreement on per-pattern means (exclude p4, n=1 descriptive-only)
    patterns = ["base_p0", "base_p6", "base_p8", "base_p11"]  # powered enough (n>=7)
    def vec(jn): return [per_judge[jn]["per_pattern_mean"].get(p) for p in patterns]
    rank_agree = {
        "patterns_ranked": patterns,
        "gpt52_vs_opus48": spearman(vec("gpt52"), vec("opus48")),
        "gpt52_vs_sonnet5": spearman(vec("gpt52"), vec("sonnet5")),
        "opus48_vs_sonnet5": spearman(vec("opus48"), vec("sonnet5")),
    }
    # ordering per judge
    order = {jn: sorted(patterns, key=lambda p: -(per_judge[jn]["per_pattern_mean"].get(p) or -1)) for jn in JUDGE_DIRS}
    orders_identical = len({tuple(order[jn]) for jn in JUDGE_DIRS}) == 1

    result = {
        "experiment": "headline_replicates_3family",
        "date": "2026-07-25",
        "substrate": "replicate runs of base_p0/p4/p6/p8/p11 (166 reports/judge post-quarantine); ragged N per cell",
        "caveat": "ragged cell sizes (p0=7,p4=1,p6=32,p8=36,p11=94); rank-agreement robustness check, NOT a powered per-pattern re-derivation. p4 (n=1) excluded from rank agreement. base_p6__rep1 Claude verdicts re-parsed after a salvage verdict-format bug.",
        "per_judge": per_judge,
        "cross_judge_rank_agreement_spearman": rank_agree,
        "per_judge_pattern_order": order,
        "verdict": {
            "pattern_ordering_identical_across_judges": orders_identical,
            "min_pairwise_rank_spearman": min(v for v in [rank_agree["gpt52_vs_opus48"], rank_agree["gpt52_vs_sonnet5"], rank_agree["opus48_vs_sonnet5"]] if v is not None),
            "reading": ("Robustness of the replicate-run pattern ranking across judge families. High cross-judge "
                        "rank agreement => the replicate-based ordering is not a single-judge artefact (complements "
                        "the banked run-noise-robustness result). Levels differ by judge leniency (J0) but the "
                        "ordering is the load-bearing quantity here."),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"per_pattern_mean": {jn: per_judge[jn]["per_pattern_mean"] for jn in JUDGE_DIRS},
                      "rank_agreement": rank_agree,
                      "orders": order,
                      "orders_identical": orders_identical}, indent=2))


if __name__ == "__main__":
    main()
