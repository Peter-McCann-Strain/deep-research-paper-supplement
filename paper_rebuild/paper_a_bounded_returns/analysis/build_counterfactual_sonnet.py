#!/usr/bin/env python3
"""counterfactual_sonnet_crosscheck — current-Claude (Sonnet 5) robustness for the
counterfactual evidence-faithfulness finding in `frozen_defence` (GPT-5.2-only, banked).

Banked finding (build_frozen_defence.py, judge=gpt-5.2): frozen evidence carries an
attribute that contradicts the model's likely prior; a low rule-based override-rate +
high GPT-5.2 factual/attribution faithfulness on the same 30 reports supports "the
writer follows the frozen evidence rather than its prior." The override-rate itself is
RULE-BASED (deterministic, judge-independent) so it is NOT recomputed here -- only the
GPT-5.2 evidence-faithfulness judge scores are replicated under Sonnet 5 (a distinct
judge family, per JUDGE-VERSION PROTOCOL). Sonnet-only (not full 3-family) per
2026-07-27 decision. $0 (subscription judging, already banked to disk). STAGING.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "counterfactual_sonnet.json"
CANONICAL = AN / "canonical_numbers.json"

SONNET_JUDGE = ROOT / "results" / "judge_counterfactual_sonnet5"
QUARANTINE = "82de3e92-abe2-46ac-ad17-23417b9c4da7"

SEED = 20260705  # matches build_frozen_defence.py
N_BOOT = 10000


def _dim(v, name):
    d = v.get("dimensions", {}).get(name)
    return float(d["score"]) if isinstance(d, dict) and d.get("total", 0) else None


def _load_verdict(path):
    v = json.loads(path.read_text())
    return {
        "query_id": v.get("query_id", path.stem),
        "overall": float(v["overall_score"]) if v.get("overall_score") is not None else None,
        "factual_accuracy": _dim(v, "factual_accuracy"),
        "attribution_quality": _dim(v, "attribution_quality"),
    }


def _load_dir(dirpath):
    out = {}
    if not dirpath.exists():
        return out
    for p in sorted(dirpath.glob("*.json")):
        if p.stem == QUARANTINE:
            continue
        rec = _load_verdict(p)
        out[rec["query_id"]] = rec
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
    verdicts = _load_dir(SONNET_JUDGE)
    faithfulness = {
        "factual_accuracy": _mean_ci([r["factual_accuracy"] for r in verdicts.values()], rng),
        "attribution_quality": _mean_ci([r["attribution_quality"] for r in verdicts.values()], rng),
        "overall": _mean_ci([r["overall"] for r in verdicts.values()], rng),
    }
    coverage = {"got": len(verdicts), "expected": 29}

    canon = json.loads(CANONICAL.read_text())
    gpt52_cf = canon.get("frozen_defence", {}).get("counterfactual", {})
    gpt52_faith = gpt52_cf.get("gpt52_faithfulness", {})

    result = {
        "experiment": "counterfactual_sonnet_crosscheck",
        "date": "2026-07-27",
        "judge_model": "claude-sonnet-5",
        "judge_source": "counterfactual_j10",
        "note": ("Current-Claude (Sonnet 5 only) cross-family replication of the banked "
                 "GPT-5.2 `frozen_defence.counterfactual` evidence-faithfulness scores. "
                 "The rule-based override-rate is judge-independent and not recomputed "
                 "here -- see `frozen_defence.counterfactual.override_rate_probe_class` "
                 "(unchanged). 29 reports (82de3e92 quarantined)."),
        "coverage": coverage,
        "sonnet_faithfulness": faithfulness,
        "cross_family_verdict": {
            "gpt52_factual_accuracy_mean": gpt52_faith.get("factual_accuracy", {}).get("mean"),
            "sonnet_factual_accuracy_mean": faithfulness["factual_accuracy"]["mean"],
            "gpt52_attribution_quality_mean": gpt52_faith.get("attribution_quality", {}).get("mean"),
            "sonnet_attribution_quality_mean": faithfulness["attribution_quality"]["mean"],
            "both_judges_high_faithfulness": bool(
                (faithfulness["factual_accuracy"]["mean"] or 0) >= 0.5
                and (gpt52_faith.get("factual_accuracy", {}).get("mean") or 0) >= 0.5
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"coverage": coverage, "cross_family_verdict": result["cross_family_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
