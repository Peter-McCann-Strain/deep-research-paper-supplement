#!/usr/bin/env python3
"""Bake-off 3-family leaderboard + backend sub-study -> staging/bakeoff.json.

7 off-the-shelf DR frameworks, fixed gpt-4o-mini backbone, 30 frozen queries,
scored by the 3-family panel (GPT-5.2 anchor + current Opus 4.8 + Sonnet 5) on
the identical rubric_v2. Per JUDGE-VERSION PROTOCOL, GPT-5.2 is the primary axis;
current-Claude is a labelled robustness cohort (raw here; J0 offset applies only
if level-comparing to the banked corpus, which this self-contained bake-off does not).

Metrics: per-arm weighted overall (from each verdict file's overall_score) + per-dim
satisfied-rate (recomputed uniformly from verdicts), query-clustered bootstrap CIs,
cross-family Spearman rank agreement, completion rate, and the ODR Azure-vs-Tavily
backend contrast per family (why the families disagree on the backend). Deterministic,
seed=20260712. STAGING only.
"""
import json, glob, os, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "bakeoff.json"
CONC = AN / "staging" / "bakeoff_concentration.json"
SEED = 20260712
N_BOOT = 10000
ARMS = ["gpt_researcher", "open_deep_research", "open_deep_research_tavily",
        "storm", "ii_researcher", "owl", "deerflow"]
FAMS = {"gpt52": "gpt-5.2", "opus48": "claude-opus-4-8", "sonnet5": "claude-sonnet-5"}
CANON_N = 30


def load(fam, arm):
    """query_id -> {overall, dims:{dim:rate}}"""
    out = {}
    for f in glob.glob(str(ROOT / f"results/judge_bakeoff_{fam}/{arm}__bakeoff/*.json")):
        d = json.load(open(f))
        qid = d.get("query_id") or Path(f).stem
        by = defaultdict(list)
        for v in d.get("verdicts", []):
            by[v["dimension"]].append(1 if v.get("satisfied") else 0)
        dims = {k: statistics.mean(v) for k, v in by.items()}
        out[qid] = {"overall": d.get("overall_score"), "dims": dims}
    return out


def cluster_boot_ci(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    rng = np.random.default_rng(SEED)
    arr = np.array(vals)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(N_BOOT)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def spearman(a, b):
    # rank correlation without scipy
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4)


def main():
    data = {fam: {arm: load(fam, arm) for arm in ARMS} for fam in FAMS}

    # per-arm per-family overall + CI
    per_family = {}
    for fam in FAMS:
        rows = {}
        for arm in ARMS:
            ov = [r["overall"] for r in data[fam][arm].values() if r["overall"] is not None]
            rows[arm] = {"n": len(ov), "overall": round(statistics.mean(ov), 4) if ov else None,
                         "ci95": cluster_boot_ci(ov)}
        per_family[fam] = rows

    # GPT-5.2 anchor leaderboard (primary), ranked
    anchor = sorted(ARMS, key=lambda a: -(per_family["gpt52"][a]["overall"] or 0))
    anchor_board = [{"framework": a, "overall": per_family["gpt52"][a]["overall"],
                     "ci95": per_family["gpt52"][a]["ci95"], "n": per_family["gpt52"][a]["n"]}
                    for a in anchor]

    # top-4 tightness (bounded-returns echo)
    top4 = [per_family["gpt52"][a]["overall"] for a in anchor[:4]]
    tightness = {"top4_frameworks": anchor[:4], "spread": round(max(top4) - min(top4), 4),
                 "note": "spread across the top-4 frameworks on the GPT-5.2 anchor; small spread = bounded returns to orchestration reproduced across independent codebases"}

    # panel mean (simple mean of 3 family means; ranking is the point, levels differ by leniency)
    panel = {}
    for arm in ARMS:
        ms = [per_family[f][arm]["overall"] for f in FAMS if per_family[f][arm]["overall"] is not None]
        panel[arm] = round(statistics.mean(ms), 4) if ms else None

    # cross-family Spearman rank agreement on arm-level overall
    fam_scores = {f: [per_family[f][a]["overall"] or 0 for a in ARMS] for f in FAMS}
    rank_agree = {
        "gpt52_vs_opus48": spearman(fam_scores["gpt52"], fam_scores["opus48"]),
        "gpt52_vs_sonnet5": spearman(fam_scores["gpt52"], fam_scores["sonnet5"]),
        "opus48_vs_sonnet5": spearman(fam_scores["opus48"], fam_scores["sonnet5"]),
        "note": "Spearman of arm rankings across judge families; low = families disagree on the leaderboard",
    }

    # leniency levels (mean over all arms)
    leniency = {f: round(statistics.mean([per_family[f][a]["overall"] for a in ARMS
                                          if per_family[f][a]["overall"] is not None]), 4) for f in FAMS}

    # backend sub-study: ODR azure vs tavily, per family, paired by query, on overall + citation/attribution
    backend = {}
    for fam in FAMS:
        az, tv = data[fam]["open_deep_research"], data[fam]["open_deep_research_tavily"]
        common = sorted(set(az) & set(tv))
        d_over = [tv[q]["overall"] - az[q]["overall"] for q in common
                  if az[q]["overall"] is not None and tv[q]["overall"] is not None]
        def dim_delta(dim, tv=tv, az=az, common=common):
            ds = [tv[q]["dims"].get(dim, 0) - az[q]["dims"].get(dim, 0) for q in common]
            return round(statistics.mean(ds), 4) if ds else None
        backend[fam] = {
            "d_overall_tavily_minus_azure": round(statistics.mean(d_over), 4) if d_over else None,
            "d_overall_ci95": cluster_boot_ci(d_over),
            "d_citation_quality": dim_delta("citation_quality"),
            "d_attribution_quality": dim_delta("attribution_quality"),
            "n_pairs": len(common),
        }

    # completion rates
    completion = {arm: {"n_judged": per_family["gpt52"][arm]["n"],
                        "completion_rate": round(per_family["gpt52"][arm]["n"] / CANON_N, 3)}
                  for arm in ARMS}

    concentration = json.load(open(CONC)) if CONC.exists() else None

    result = {
        "experiment": "bakeoff_3family_leaderboard",
        "date": "2026-07-12",
        "design": "7 frameworks x fixed gpt-4o-mini backbone x 30 frozen queries x 3-family panel (GPT-5.2 anchor + Opus 4.8 + Sonnet 5), identical rubric_v2",
        "judge_version_note": "GPT-5.2 = primary anchor (comparable to banked corpus). Opus 4.8/Sonnet 5 = labelled current-Claude robustness cohort; raw here (J0 offset judge_version_bridge applies only for level-comparison to banked numbers).",
        "anchor_leaderboard_gpt52": anchor_board,
        "top_cluster_tightness": tightness,
        "per_family_overall": {f: {a: per_family[f][a] for a in ARMS} for f in FAMS},
        "panel_mean_overall": panel,
        "cross_family_rank_agreement_spearman": rank_agree,
        "judge_leniency_mean_overall": leniency,
        "backend_substudy_quality": backend,
        "backend_substudy_concentration": concentration,
        "completion_rates": completion,
        "headline": None,  # filled below
    }
    # headline synthesis
    result["headline"] = {
        "quality_bounded": f"top-4 frameworks within {tightness['spread']} on GPT-5.2 anchor",
        "backend_on_quality_judge_dependent": {
            "gpt52_d_overall": backend["gpt52"]["d_overall_tavily_minus_azure"],
            "opus48_d_overall": backend["opus48"]["d_overall_tavily_minus_azure"],
            "sonnet5_d_overall": backend["sonnet5"]["d_overall_tavily_minus_azure"],
            "reading": "GPT-5.2 sees ~0 backend effect on quality; Claude families reward the citation-richer Tavily arm -> the backend's quality effect is judge-dependent",
        },
        "backend_on_concentration": (concentration or {}).get("backend_contrast", {}).get("hhi_azure_vs_tavily") if concentration else None,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("\n=== GPT-5.2 ANCHOR LEADERBOARD ===")
    for r in anchor_board:
        print(f"  {r['framework']:<28} {r['overall']:.3f}  CI{r['ci95']}  n={r['n']}")
    print(f"\ntop-4 tightness: {tightness['spread']}")
    print(f"leniency (mean overall): {leniency}")
    print(f"cross-family rank agreement: {rank_agree}")
    print("\nbackend Δoverall (tavily-azure) by family:")
    for f in FAMS:
        print(f"  {f}: {backend[f]['d_overall_tavily_minus_azure']} CI{backend[f]['d_overall_ci95']} "
              f"(Δcit={backend[f]['d_citation_quality']}, Δattr={backend[f]['d_attribution_quality']})")


if __name__ == "__main__":
    main()
