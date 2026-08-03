#!/usr/bin/env python3
"""Matched-compute test for the frozen-defence draft_revise finding (Phase 1.5).

frozen_defence found draft_revise (draft->critique->revise, 3 LLM calls) beats single_pass
on factual accuracy by ~+0.08. Reviewer risk: is that just MORE COMPUTE? The 5 synth-ablation
scaffolds sit at a natural compute ladder (llm_calls): single_pass=1, draft_revise=3, beam=7,
map_reduce=7, verifier_select=45 — all on the SAME frozen evidence, same gpt-4o-mini. So we test
the matched-compute question WITHOUT new generation: if draft_revise (only 3 calls) is NOT beaten
by the 7- and 45-call scaffolds, the gain is ARCHITECTURAL (the revise loop), not compute.

$0 CPU: reads existing GPT-5.2 verdicts + checkpoint llm_calls. Deterministic, seed=20260724.
Lands `synth_matched_compute` (companion to frozen_defence). STAGING only.
"""
import json, glob, statistics
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "synth_matched_compute.json"
SEED = 20260724
N_BOOT = 10000
SCAFFOLDS = ["single_pass", "draft_revise", "map_reduce", "beam", "verifier_select"]


def compute_per_scaffold():
    """median llm_calls per scaffold from checkpoints (compute proxy)."""
    out = {}
    for s in SCAFFOLDS:
        calls = []
        for f in glob.glob(str(ROOT / f"checkpoints/synthablation/{s}/*.json")):
            d = json.load(open(f))
            m = d.get("metrics", d)
            c = m.get("llm_calls", d.get("llm_calls"))
            if c is not None:
                calls.append(int(c))
        out[s] = int(statistics.median(calls)) if calls else None
    return out


def factual_and_overall(s):
    """{qid: (factual_rate, overall_rate)} for one scaffold from GPT-5.2 verdicts."""
    out = {}
    for f in glob.glob(str(ROOT / f"results/judge_gpt52_synthablation/{s}/*.json")):
        d = json.load(open(f))
        qid = d.get("query_id") or Path(f).stem
        fac = [1 if v.get("satisfied") else 0 for v in d.get("verdicts", []) if v.get("dimension") == "factual_accuracy"]
        allc = [1 if v.get("satisfied") else 0 for v in d.get("verdicts", [])]
        if fac:
            out[qid] = (statistics.mean(fac), statistics.mean(allc) if allc else None)
    return out


def paired_delta_ci(a_map, b_map, idx):
    """mean(b - a) on shared qids for metric index idx (0=factual,1=overall), bootstrap CI."""
    common = sorted(set(a_map) & set(b_map))
    diffs = [b_map[q][idx] - a_map[q][idx] for q in common
             if a_map[q][idx] is not None and b_map[q][idx] is not None]
    if len(diffs) < 3:
        return None
    rng = np.random.default_rng(SEED)
    arr = np.array(diffs)
    point = float(arr.mean())
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(diffs)}


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    return round(1 - 6*sum((rx[i]-ry[i])**2 for i in range(n))/(n*(n*n-1)), 4)


def main():
    comp = compute_per_scaffold()
    data = {s: factual_and_overall(s) for s in SCAFFOLDS}

    per_scaffold = {}
    for s in SCAFFOLDS:
        fac = [v[0] for v in data[s].values()]
        ovr = [v[1] for v in data[s].values() if v[1] is not None]
        per_scaffold[s] = {
            "llm_calls": comp[s],
            "factual_mean": round(statistics.mean(fac), 4) if fac else None,
            "overall_mean": round(statistics.mean(ovr), 4) if ovr else None,
            "n": len(fac),
        }

    # draft_revise vs each HIGHER-compute scaffold (paired factual): does more compute beat it?
    dr = data["draft_revise"]
    higher = [s for s in SCAFFOLDS if s != "draft_revise" and (comp[s] or 0) >= (comp["draft_revise"] or 0) and s != "draft_revise"]
    vs_draft_revise = {}
    for s in SCAFFOLDS:
        if s == "draft_revise":
            continue
        d_fac = paired_delta_ci(dr, data[s], 0)  # (s) - draft_revise
        vs_draft_revise[s] = {
            "llm_calls": comp[s],
            "factual_delta_vs_draft_revise": d_fac,  # positive => s beats draft_revise
            "draft_revise_still_wins": bool(d_fac and d_fac["delta"] < 0),
        }

    # compute -> factual trend across the 5 arm means
    arms = [s for s in SCAFFOLDS if per_scaffold[s]["llm_calls"] is not None and per_scaffold[s]["factual_mean"] is not None]
    rho_compute_factual = spearman([per_scaffold[s]["llm_calls"] for s in arms],
                                   [per_scaffold[s]["factual_mean"] for s in arms])

    best_factual = max(arms, key=lambda s: per_scaffold[s]["factual_mean"])
    higher_compute_than_dr = [s for s in SCAFFOLDS if (comp[s] or 0) > (comp["draft_revise"] or 0)]
    none_higher_beats_dr = all(vs_draft_revise[s]["draft_revise_still_wins"] for s in higher_compute_than_dr)

    result = {
        "experiment": "synth_matched_compute",
        "date": "2026-07-24",
        "question": "Is draft_revise's +0.08 factual gain over single_pass just more compute? Test across the natural compute ladder of the 5 frozen-evidence scaffolds (no new generation).",
        "compute_ladder_llm_calls": {s: comp[s] for s in SCAFFOLDS},
        "per_scaffold": per_scaffold,
        "draft_revise_vs_others_factual": vs_draft_revise,
        "compute_factual_spearman_across_arms": rho_compute_factual,
        "best_factual_scaffold": best_factual,
        "verdict": {
            "best_factual_is_draft_revise": best_factual == "draft_revise",
            "no_higher_compute_scaffold_beats_draft_revise": bool(none_higher_beats_dr),
            "compute_does_not_predict_factual": bool(rho_compute_factual <= 0.3),
            "reading": ("If draft_revise (3 calls) holds the top factual score and neither the 7-call "
                        "(beam/map_reduce) nor 45-call (verifier_select) scaffold beats it, and factual "
                        "does not rise with compute across arms, then the +0.08 gain is ARCHITECTURAL "
                        "(the draft->critique->revise loop) rather than a compute artefact. This is the "
                        "matched-compute defence of the frozen_defence finding, at $0."),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"compute_ladder": result["compute_ladder_llm_calls"],
                      "factual_by_scaffold": {s: per_scaffold[s]["factual_mean"] for s in SCAFFOLDS},
                      "compute_factual_rho": rho_compute_factual,
                      "best_factual": best_factual,
                      "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
