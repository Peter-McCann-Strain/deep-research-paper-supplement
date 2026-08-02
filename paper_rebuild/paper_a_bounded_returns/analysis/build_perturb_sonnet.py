#!/usr/bin/env python3
"""perturb_sonnet_crosscheck — current-Claude (Sonnet 5) score comparison on the E13'
perturbed-report set (`results/judge_gpt52_perturb`), judged independently.

SCOPE NOTE (read before citing): `build_e13_detector_roc.py`'s `gpt52_injection_roc`
computes a matched-pair SCORE-DROP ROC-AUC (perturbed vs its clean, un-perturbed
original from `results/judge_gpt52`, on the defect-targeted dimension) -- that needs
BOTH the perturbed set AND its clean-original matches judged by the SAME model. Only
the 34 perturbed reports (13 base_p* patterns, 82de3e92 quarantined) were queued for
Sonnet-5 judging (J11) -- the clean originals were NOT re-judged with Sonnet here, so
this does NOT reproduce the injection-ROC/AUC. What this DOES check, honestly: do
GPT-5.2 and Sonnet-5 AGREE on the (low) quality of the same perturbed reports, judged
independently? A correlated, low-scoring read from both is corroborating evidence that
the perturbed set is genuinely degraded (not a GPT-5.2-specific severity artifact);
it is NOT a replication of the detector AUC itself. Sonnet-only per 2026-07-27
decision. $0 (subscription judging, already banked to disk). STAGING.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "perturb_sonnet.json"

GPT_DIR = ROOT / "results" / "judge_gpt52_perturb"
SONNET_DIR = ROOT / "results" / "judge_perturb_sonnet5"
QUARANTINE = "82de3e92-abe2-46ac-ad17-23417b9c4da7"

SEED = 20260705
N_BOOT = 10000


def _dim(v, name):
    d = v.get("dimensions", {}).get(name)
    return float(d["score"]) if isinstance(d, dict) and d.get("total", 0) else None


def _load_pairs(root):
    out = {}
    for pat_dir in sorted(root.glob("*")):
        if not pat_dir.is_dir():
            continue
        pattern = pat_dir.name
        for f in sorted(pat_dir.glob("*.json")):
            qid = f.stem
            if qid == QUARANTINE:
                continue
            v = json.loads(f.read_text())
            out[(pattern, qid)] = {
                "overall": float(v["overall_score"]) if v.get("overall_score") is not None else None,
                "factual_accuracy": _dim(v, "factual_accuracy"),
            }
    return out


def _mean_ci(vals, rng):
    a = np.asarray([x for x in vals if x is not None], float)
    n = len(a)
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    boots = np.array([a[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": round(float(a.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(n)}


def main():
    rng = np.random.default_rng(SEED)
    gpt = _load_pairs(GPT_DIR)
    sonnet = _load_pairs(SONNET_DIR)
    common = sorted(set(gpt) & set(sonnet))

    gpt_overall = [gpt[k]["overall"] for k in common]
    sonnet_overall = [sonnet[k]["overall"] for k in common]
    gpt_fact = [gpt[k]["factual_accuracy"] for k in common]
    sonnet_fact = [sonnet[k]["factual_accuracy"] for k in common]

    valid = [i for i in range(len(common)) if gpt_overall[i] is not None and sonnet_overall[i] is not None]
    if len(valid) >= 3:
        g = np.array([gpt_overall[i] for i in valid])
        s = np.array([sonnet_overall[i] for i in valid])
        pearson_r = float(np.corrcoef(g, s)[0, 1])
        g_rank = np.argsort(np.argsort(g))
        s_rank = np.argsort(np.argsort(s))
        spearman_rho = float(np.corrcoef(g_rank, s_rank)[0, 1])
    else:
        pearson_r = spearman_rho = None

    result = {
        "experiment": "perturb_sonnet_crosscheck",
        "date": "2026-07-27",
        "judge_model": "claude-sonnet-5",
        "judge_source": "perturb_j11",
        "scope_note": (
            "Does NOT reproduce build_e13_detector_roc.py's gpt52_injection_roc "
            "(matched-pair score-DROP vs clean originals, which needs the clean "
            "originals re-judged with Sonnet too -- not done here). This is an "
            "independent-judge score-agreement check on the 34 perturbed reports "
            "alone (13 base_p* patterns, 82de3e92 quarantined)."
        ),
        "coverage": {"gpt52": len(gpt), "sonnet5": len(sonnet), "common_pairs": len(common)},
        "gpt52_overall": _mean_ci(gpt_overall, rng),
        "sonnet_overall": _mean_ci(sonnet_overall, rng),
        "gpt52_factual_accuracy": _mean_ci(gpt_fact, rng),
        "sonnet_factual_accuracy": _mean_ci(sonnet_fact, rng),
        "cross_family_agreement": {
            "pearson_r_overall_score": round(pearson_r, 4) if pearson_r is not None else None,
            "spearman_rho_overall_score": round(spearman_rho, 4) if spearman_rho is not None else None,
            "n_pairs": len(valid),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({
        "coverage": result["coverage"],
        "gpt52_overall": result["gpt52_overall"],
        "sonnet_overall": result["sonnet_overall"],
        "cross_family_agreement": result["cross_family_agreement"],
    }, indent=2))


if __name__ == "__main__":
    main()
