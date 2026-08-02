#!/usr/bin/env python3
"""distractor_sonnet_crosscheck — current-Claude (Sonnet 5) robustness for the
distractor-dose finding in `frozen_defence` (GPT-5.2-only, banked).

Banked finding (build_frozen_defence.py, judge=gpt-5.2): as distractor noise rises
(0/20/40/70% doped passages), the orchestrated cluster (p1/p4) loses LESS quality per
unit of noise than p0 -- a POSITIVE cluster-minus-p0 gap_slope, meaning orchestration's
value reappears precisely when evidence is unreliable. This replicates the exact same
per-pattern dose-response + cluster-vs-p0 gap-slope contrasts under Sonnet 5 (a distinct
judge family, per JUDGE-VERSION PROTOCOL). Sonnet-only (not full 3-family) per
2026-07-27 decision. Same seed/bootstrap machinery as build_frozen_defence.py's
build_distractor() for exact methodological parity. $0 (subscription judging, already
banked to disk). STAGING.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
OUT = AN / "staging" / "distractor_sonnet.json"
CANONICAL = AN / "canonical_numbers.json"

SONNET_JUDGE = ROOT / "results" / "judge_distractor_sonnet5"
DIST_PATTERNS = ["p0", "p1", "p4"]
DOSES = [("d000", 0.0), ("d020", 0.20), ("d040", 0.40), ("d070", 0.70)]
CLUSTER_PATTERNS = ["p1", "p4"]
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


def _paired_diff(a_vals, b_vals, rng):
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    d = a - b
    n = len(d)
    if n == 0:
        return {"point": None, "ci95": [None, None], "p_two_sided_boot": None, "n_pairs": 0}
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt, p_lt = float((boots > 0).mean()), float((boots < 0).mean())
    p_two = min(1.0, 2.0 * min(p_gt, p_lt))
    p_two_r = round(p_two, 4)
    p_report = ("<%.0e" % (1.0 / N_BOOT)) if p_two_r == 0.0 else p_two_r
    return {"point": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided_boot": p_report, "n_pairs": int(n)}


def _slope_ci(dose_x, per_dose_arrays, rng):
    x = np.asarray(dose_x, float)
    n = len(per_dose_arrays[0]) if per_dose_arrays else 0
    if n == 0:
        return {"slope": None, "ci95": [None, None], "p_two_sided_boot": None, "n_queries": 0}

    def slope_of(idx):
        means = np.array([arr[idx].mean() for arr in per_dose_arrays])
        return float(np.polyfit(x, means, 1)[0])

    point = slope_of(np.arange(n))
    boots = np.array([slope_of(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt, p_lt = float((boots > 0).mean()), float((boots < 0).mean())
    p_two = min(1.0, 2.0 * min(p_gt, p_lt))
    return {"slope": round(point, 5),
            "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "p_two_sided_boot": round(p_two, 4), "n_queries": int(n)}


def _gap_slope_ci(dose_x, cluster_by_dose, p0_by_dose, rng):
    x = np.asarray(dose_x, float)
    n = len(cluster_by_dose[0]) if cluster_by_dose else 0
    if n == 0:
        return {"slope": None, "ci95": [None, None], "p_two_sided_boot": None, "n_queries": 0}

    def slope_of(idx):
        gaps = np.array([cluster_by_dose[i][idx].mean() - p0_by_dose[i][idx].mean()
                         for i in range(len(x))])
        return float(np.polyfit(x, gaps, 1)[0])

    point = slope_of(np.arange(n))
    boots = np.array([slope_of(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt, p_lt = float((boots > 0).mean()), float((boots < 0).mean())
    p_two = min(1.0, 2.0 * min(p_gt, p_lt))
    return {"slope": round(point, 5),
            "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "p_two_sided_boot": round(p_two, 4), "n_queries": int(n)}


def main():
    rng = np.random.default_rng(SEED)
    arms = {}
    for p in DIST_PATTERNS:
        for tag, _ in DOSES:
            arm = f"{p}_{tag}"
            arms[arm] = _load_dir(SONNET_JUDGE / arm)
    coverage = {arm: len(arms[arm]) for arm in arms}

    dose_x = [d for _, d in DOSES]
    per_pattern = {}
    for p in DIST_PATTERNS:
        per_dose = {}
        for tag, d in DOSES:
            arm = f"{p}_{tag}"
            per_dose[tag] = {
                "dose": d,
                "overall": _mean_ci([r["overall"] for r in arms[arm].values()], rng),
                "factual_accuracy": _mean_ci([r["factual_accuracy"] for r in arms[arm].values()], rng),
            }
        common = set.intersection(*[set(arms[f"{p}_{tag}"]) for tag, _ in DOSES]) \
            if all(arms[f"{p}_{tag}"] for tag, _ in DOSES) else set()
        common = sorted(common)
        if common:
            ov_arrays = [np.array([arms[f"{p}_{tag}"][q]["overall"] for q in common], float)
                         for tag, _ in DOSES]
            fa_arrays = [np.array([arms[f"{p}_{tag}"][q]["factual_accuracy"] for q in common], float)
                         for tag, _ in DOSES]
            slope_ov = _slope_ci(dose_x, ov_arrays, rng)
            slope_fa = _slope_ci(dose_x, fa_arrays, rng)
        else:
            slope_ov = slope_fa = {"slope": None, "ci95": [None, None],
                                   "p_two_sided_boot": None, "n_queries": 0}
        per_pattern[p] = {
            "per_dose": per_dose,
            "dose_slope_overall": slope_ov,
            "dose_slope_factual": slope_fa,
            "n_common_queries": len(common),
        }

    cluster_vs_p0 = {}
    for c in CLUSTER_PATTERNS:
        gap_by_dose = {}
        for tag, d in DOSES:
            a_arm, b_arm = f"{c}_{tag}", f"p0_{tag}"
            common = sorted(set(arms[a_arm]) & set(arms[b_arm]))
            a_ov = [arms[a_arm][q]["overall"] for q in common]
            b_ov = [arms[b_arm][q]["overall"] for q in common]
            a_fa = [arms[a_arm][q]["factual_accuracy"] for q in common]
            b_fa = [arms[b_arm][q]["factual_accuracy"] for q in common]
            gap_by_dose[tag] = {
                "dose": d,
                "overall_gap": _paired_diff(a_ov, b_ov, rng),
                "factual_gap": _paired_diff(a_fa, b_fa, rng),
            }
        all_arms = [f"{c}_{tag}" for tag, _ in DOSES] + [f"p0_{tag}" for tag, _ in DOSES]
        common = set.intersection(*[set(arms[a]) for a in all_arms]) \
            if all(arms[a] for a in all_arms) else set()
        common = sorted(common)
        if common:
            cl_ov = [np.array([arms[f"{c}_{tag}"][q]["overall"] for q in common], float)
                     for tag, _ in DOSES]
            p0_ov = [np.array([arms[f"p0_{tag}"][q]["overall"] for q in common], float)
                     for tag, _ in DOSES]
            cl_fa = [np.array([arms[f"{c}_{tag}"][q]["factual_accuracy"] for q in common], float)
                     for tag, _ in DOSES]
            p0_fa = [np.array([arms[f"p0_{tag}"][q]["factual_accuracy"] for q in common], float)
                     for tag, _ in DOSES]
            gap_slope_ov = _gap_slope_ci(dose_x, cl_ov, p0_ov, rng)
            gap_slope_fa = _gap_slope_ci(dose_x, cl_fa, p0_fa, rng)
        else:
            gap_slope_ov = gap_slope_fa = {"slope": None, "ci95": [None, None],
                                           "p_two_sided_boot": None, "n_queries": 0}
        cluster_vs_p0[f"{c}_minus_p0"] = {
            "gap_by_dose": gap_by_dose,
            "gap_slope_overall": gap_slope_ov,
            "gap_slope_factual": gap_slope_fa,
            "n_common_queries": len(common),
            "orchestration_value_grows_with_noise_overall":
                bool(gap_slope_ov["slope"] is not None and gap_slope_ov["slope"] > 0),
        }

    # pull the ACTUAL banked GPT-5.2 result for comparison -- do not hardcode
    canon = json.loads(CANONICAL.read_text())
    gpt52_db = canon.get("frozen_defence", {}).get("distractor", {})
    gpt52_cluster = gpt52_db.get("cluster_vs_p0", {})
    cross_family = {}
    for k in cluster_vs_p0:
        g = gpt52_cluster.get(k, {})
        g_slope = g.get("gap_slope_overall", {}) if g else {}
        s_slope = cluster_vs_p0[k]["gap_slope_overall"]
        g_ci = g_slope.get("ci95", [None, None])
        s_ci = s_slope.get("ci95", [None, None])
        g_sig = bool(g_ci[0] is not None and (g_ci[0] > 0 or g_ci[1] < 0))
        s_sig = bool(s_ci[0] is not None and (s_ci[0] > 0 or s_ci[1] < 0))
        cross_family[k] = {
            "gpt52_gap_slope_overall": g_slope.get("slope"),
            "gpt52_gap_slope_ci95": g_ci,
            "gpt52_significant": g_sig,
            "gpt52_grows_with_noise_point_estimate": g.get("orchestration_value_grows_with_noise_overall"),
            "sonnet_gap_slope_overall": s_slope["slope"],
            "sonnet_gap_slope_ci95": s_slope["ci95"],
            "sonnet_significant": s_sig,
            "sonnet_grows_with_noise_point_estimate": cluster_vs_p0[k]["orchestration_value_grows_with_noise_overall"],
            "either_judge_significant": bool(g_sig or s_sig),
            "reading": (
                "Neither judge's gap-slope CI excludes zero -- point-estimate sign "
                "differences between judges are noise around a null, not a genuine "
                "cross-family disagreement." if not (g_sig or s_sig) else
                "At least one judge's CI excludes zero -- treat point-estimate sign "
                "agreement/disagreement as meaningful."
            ),
        }

    result = {
        "experiment": "distractor_sonnet_crosscheck",
        "date": "2026-07-27",
        "judge_model": "claude-sonnet-5",
        "judge_source": "distractor_j9",
        "note": ("Current-Claude (Sonnet 5 only, Opus dropped per 2026-07-27 efficiency "
                 "decision) cross-family replication of the banked GPT-5.2 `frozen_defence."
                 "distractor` finding. Same 12 arms (3 patterns x 4 doses) x ~29 "
                 "(82de3e92 quarantined; p4_d020 has 28), same seed/paired-bootstrap "
                 "machinery as build_frozen_defence.py."),
        "coverage": coverage,
        "per_pattern": per_pattern,
        "cluster_vs_p0": cluster_vs_p0,
        "cross_family_verdict": cross_family,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({"coverage": coverage, "cross_family_verdict": cross_family}, indent=2))


if __name__ == "__main__":
    main()
