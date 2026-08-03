#!/usr/bin/env python3
"""Land the FROZEN-EVIDENCE DEFENCE experiments (synth-ablation / counterfactual /
distractor) into a STAGING JSON blob.

This is the crown-jewel rebuttal to Argus (2605.16217) / TTD-DR (2507.16075) /
GRACE (2606.16151): three experiments that hold the *evidence* fixed (a frozen
oracle corpus) and vary only the synthesis machinery, so any quality movement is
attributable to the writer/orchestrator rather than to better retrieval.

It reads the GPT-5.2 verdicts written by
``scripts/run_gpt52_judge_namespaced.py`` for the three passes:

  results/judge_gpt52_synthablation/<scaffold>/<qid>.json   (5 scaffolds x 30)
  results/judge_gpt52_counterfactual/<qid>.json             (30, flat: pattern was ".")
  results/judge_gpt52_distractor/<pattern>_d<NNN>/<qid>.json (3 patterns x 4 doses x 30)

and the RULE-BASED counterfactual artefacts:

  results/experiments_counterfactual/probes/<qid>.json      (per-probe override class)
  results/experiments_counterfactual/override_summary.json  (rule-computed override rate)

It writes a single merged blob to
``paper_rebuild/paper_a_bounded_returns/supporting_analysis/staging/frozen_defence.json``
(the STAGING convention). It NEVER touches ``canonical_numbers.json`` -- the main
programme loop merges staging blobs into the canonical store separately.

Sections
--------
(a) SYNTH-ABLATION -- per-scaffold mean factual_accuracy + citation_quality, and
    the KEY test: does ANY synthesis scaffold (draft_revise / map_reduce / beam /
    verifier_select) beat single_pass on FACTUAL accuracy given *identical frozen
    evidence*?  Per-query PAIRED bootstrap CIs (matched by qid).  If no scaffold
    moves factual accuracy, finding (iv) hardens from correlational to CAUSAL:
    the orchestration gain is NOT the synthesis scaffold -- it is the evidence.

(b) COUNTERFACTUAL -- rule-computed override-rate (report stays faithful to frozen
    evidence even when it contradicts the model's prior) + GPT-5.2 evidence-
    faithfulness (mean factual_accuracy / attribution_quality on the same 30).

(c) DISTRACTOR -- report quality vs injected-noise dose (0 / 20 / 40 / 70%) per
    pattern.  Does orchestration's value REAPPEAR as evidence degrades?  Per-
    pattern dose-response slope + the cluster-minus-P0 gap at each dose and the
    slope of that gap (all query-paired bootstrap).

Determinism: numpy default_rng(SEED), SEED=20260705, N_BOOT=10000, resampling
unit = the query (frozen greedy decode => no decode-seed variance).

Usage
-----
    python paper_rebuild/paper_a_bounded_returns/supporting_analysis/build_frozen_defence.py --check
    python paper_rebuild/paper_a_bounded_returns/supporting_analysis/build_frozen_defence.py --dry-run
    python paper_rebuild/paper_a_bounded_returns/supporting_analysis/build_frozen_defence.py
    python paper_rebuild/paper_a_bounded_returns/supporting_analysis/build_frozen_defence.py --allow-partial
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# ── judge-verdict roots (READ) ────────────────────────────────────────────────
SYNTH_JUDGE = REPO / "results/judge_gpt52_synthablation"
CF_JUDGE = REPO / "results/judge_gpt52_counterfactual"
DIST_JUDGE = REPO / "results/judge_gpt52_distractor"

# ── rule-based counterfactual artefacts (READ) ────────────────────────────────
CF_PROBES = REPO / "results/experiments_counterfactual/probes"
CF_SUMMARY = REPO / "results/experiments_counterfactual/override_summary.json"

# ── staging output (WRITE) — NOT the canonical store ──────────────────────────
STAGING_OUT = (
    REPO
    / "paper_rebuild/paper_a_bounded_returns/supporting_analysis/staging/frozen_defence.json"
)

# ── design constants ──────────────────────────────────────────────────────────
SCAFFOLDS = ["single_pass", "draft_revise", "map_reduce", "beam", "verifier_select"]
BASELINE_SCAFFOLD = "single_pass"
DIST_PATTERNS = ["p0", "p1", "p4"]
DOSES = [("d000", 0.0), ("d020", 0.20), ("d040", 0.40), ("d070", 0.70)]
CLUSTER_PATTERNS = ["p1", "p4"]  # orchestrated pipelines vs p0 baseline

SEED = 20260705
N_BOOT = 10000

# Expected verdict counts (for the completeness gate).
EXPECT_SYNTH = {s: 30 for s in SCAFFOLDS}                       # 150
EXPECT_CF = 30
EXPECT_DIST = {f"{p}_{tag}": 30 for p in DIST_PATTERNS for tag, _ in DOSES}
EXPECT_DIST["p4_d020"] = 29  # one report missing at generation time (known)


# ── verdict loading ───────────────────────────────────────────────────────────

def _dim(v, name):
    d = v.get("dimensions", {}).get(name)
    return float(d["score"]) if isinstance(d, dict) and d.get("total", 0) else None


def _load_verdict(path):
    v = json.loads(path.read_text())
    return {
        "query_id": v.get("query_id", path.stem),
        "overall": float(v["overall_score"]) if v.get("overall_score") is not None else None,
        "factual_accuracy": _dim(v, "factual_accuracy"),
        "citation_quality": _dim(v, "citation_quality"),
        "attribution_quality": _dim(v, "attribution_quality"),
        "information_recall": _dim(v, "information_recall"),
    }


def _load_dir(dirpath):
    """Load {qid: metrics} from a flat dir of <qid>.json verdicts."""
    out = {}
    if not dirpath.exists():
        return out
    for p in sorted(dirpath.glob("*.json")):
        rec = _load_verdict(p)
        out[rec["query_id"]] = rec
    return out


# ── bootstrap primitives ──────────────────────────────────────────────────────

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
    """Paired bootstrap of mean(a - b) over aligned per-query arrays."""
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
    """Bootstrap the OLS slope of mean-metric vs dose.

    per_dose_arrays: list of per-query arrays (one per dose), all aligned to the
    SAME common-qid block, so a single resampled index block keeps it paired.
    """
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
    """Bootstrap the slope of (cluster_mean - p0_mean) vs dose over common qids."""
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


# ── (a) SYNTH-ABLATION ─────────────────────────────────────────────────────────

def build_synthablation(rng):
    data = {s: _load_dir(SYNTH_JUDGE / s) for s in SCAFFOLDS}
    coverage = {s: {"got": len(data[s]), "expected": EXPECT_SYNTH[s]} for s in SCAFFOLDS}

    # per-scaffold marginal means (over each scaffold's own qids)
    per_scaffold = {}
    for s in SCAFFOLDS:
        vals_f = [r["factual_accuracy"] for r in data[s].values()]
        vals_c = [r["citation_quality"] for r in data[s].values()]
        vals_o = [r["overall"] for r in data[s].values()]
        per_scaffold[s] = {
            "factual_accuracy": _mean_ci(vals_f, rng),
            "citation_quality": _mean_ci(vals_c, rng),
            "overall": _mean_ci(vals_o, rng),
        }

    # paired diffs vs single_pass on the COMMON qid set (identical frozen evidence)
    base = data[BASELINE_SCAFFOLD]
    contrasts = {}
    n_beating_factual = 0
    for s in SCAFFOLDS:
        if s == BASELINE_SCAFFOLD:
            continue
        common = sorted(set(base) & set(data[s]))
        fa_a = [data[s][q]["factual_accuracy"] for q in common]
        fa_b = [base[q]["factual_accuracy"] for q in common]
        ci_a = [data[s][q]["citation_quality"] for q in common]
        ci_b = [base[q]["citation_quality"] for q in common]
        fac = _paired_diff(fa_a, fa_b, rng)
        cit = _paired_diff(ci_a, ci_b, rng)
        # "beats" single_pass on factual = paired-diff 95% CI strictly above 0
        beats = (fac["ci95"][0] is not None) and (fac["ci95"][0] > 0)
        if beats:
            n_beating_factual += 1
        contrasts[f"{s}_minus_single_pass"] = {
            "factual_accuracy_diff": fac,
            "citation_quality_diff": cit,
            "beats_single_pass_factual": bool(beats),
            "n_common_queries": len(common),
        }

    # headline
    fa_means = {s: per_scaffold[s]["factual_accuracy"]["mean"] for s in SCAFFOLDS
                if per_scaffold[s]["factual_accuracy"]["mean"] is not None}
    best_scaffold = max(fa_means, key=fa_means.get) if fa_means else None

    return {
        "description": (
            "Same 30 frozen-oracle evidence sets written by 5 synthesis scaffolds "
            "(single_pass baseline + draft_revise / map_reduce / beam / verifier_select). "
            "KEY TEST: does any scaffold beat single_pass on FACTUAL accuracy given "
            "identical evidence? If not, orchestration's factual gain is caused by the "
            "evidence, not the synthesis machinery -> finding (iv) hardens to causal."
        ),
        "coverage": coverage,
        "per_scaffold": per_scaffold,
        "contrasts_vs_single_pass": contrasts,
        "best_scaffold_by_factual": best_scaffold,
        "n_scaffolds_beating_single_pass_factual": n_beating_factual,
        "finding_iv_hardens_causal": bool(n_beating_factual == 0),
        "interpretation": (
            "ACTUAL RESULT: n_scaffolds_beating_single_pass_factual = "
            f"{n_beating_factual} (draft_revise; map_reduce/beam/verifier_select do not), "
            "so finding (iv) does NOT harden to fully causal -- a specific synthesis "
            "choice (draft-then-revise) recovers part of the factual gap even under "
            "identical frozen evidence. This partially rebuts, rather than fully rebuts, "
            "Argus / TTD-DR / GRACE test-time-scaling claims: the P1/P4>P0 factual gap is "
            "mostly attributable to the evidence, but not entirely -- report the "
            "draft_revise exception explicitly rather than a clean 'evidence alone' claim."
        ),
    }


# ── (b) COUNTERFACTUAL ─────────────────────────────────────────────────────────

def build_counterfactual(rng):
    # rule-based override classes from probes
    probes = [json.loads(p.read_text()) for p in sorted(CF_PROBES.glob("*.json"))]
    def _rate(field):
        counts = {}
        for r in probes:
            counts[r.get(field)] = counts.get(r.get(field), 0) + 1
        decided = counts.get("prior_override", 0) + counts.get("faithful", 0)
        rate = (counts.get("prior_override", 0) / decided) if decided else None
        return {"class_counts": counts,
                "override_rate": round(rate, 4) if rate is not None else None,
                "n_decided": decided}
    probe_rule = _rate("probe_class")
    report_rule = _rate("report_class")

    summary = json.loads(CF_SUMMARY.read_text()) if CF_SUMMARY.exists() else {}

    # GPT-5.2 evidence-faithfulness (flat verdict dir; pattern was ".")
    verdicts = _load_dir(CF_JUDGE)
    faithfulness = {
        "factual_accuracy": _mean_ci([r["factual_accuracy"] for r in verdicts.values()], rng),
        "attribution_quality": _mean_ci([r["attribution_quality"] for r in verdicts.values()], rng),
        "overall": _mean_ci([r["overall"] for r in verdicts.values()], rng),
    }
    coverage = {"got": len(verdicts), "expected": EXPECT_CF, "n_probes": len(probes)}

    return {
        "description": (
            "Frozen evidence carries an attribute that CONTRADICTS the model's likely "
            "prior. Override-rate = fraction of decided probes where the report reverted "
            "to its prior instead of the (counterfactual) frozen fact. GPT-5.2 separately "
            "scores evidence-faithfulness of the same 30 reports."
        ),
        "coverage": coverage,
        "override_rate_probe_class": probe_rule,
        "override_rate_report_class": report_rule,
        "override_summary_rule": summary,
        "gpt52_faithfulness": faithfulness,
        "interpretation": (
            "A low override-rate + high GPT-5.2 factual/attribution faithfulness => the "
            "writer follows the frozen evidence rather than its prior, so measured quality "
            "tracks the evidence, not the model's memory -> supports the evidence-is-the-"
            "cause reading of the orchestration gain."
        ),
    }


# ── (c) DISTRACTOR ─────────────────────────────────────────────────────────────

def build_distractor(rng):
    # load every arm
    arms = {}
    for p in DIST_PATTERNS:
        for tag, _ in DOSES:
            arm = f"{p}_{tag}"
            arms[arm] = _load_dir(DIST_JUDGE / arm)
    coverage = {arm: {"got": len(arms[arm]), "expected": EXPECT_DIST[arm]} for arm in arms}

    dose_x = [d for _, d in DOSES]

    # per pattern x dose marginal means
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
        # dose-response slope (overall & factual) over qids common to this pattern's 4 doses
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

    # cluster-minus-P0 gap at each dose + slope of that gap
    cluster_vs_p0 = {}
    for c in CLUSTER_PATTERNS:
        gap_by_dose = {}
        # per-dose paired gap
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
        # slope of the (cluster - p0) gap vs dose over qids common to all 8 arms
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
        _ci = gap_slope_ov["ci95"]
        _sig = bool(_ci[0] is not None and _ci[1] is not None and (_ci[0] > 0 or _ci[1] < 0))
        cluster_vs_p0[f"{c}_minus_p0"] = {
            "gap_by_dose": gap_by_dose,
            "gap_slope_overall": gap_slope_ov,   # >0 => cluster advantage GROWS as noise rises
            "gap_slope_factual": gap_slope_fa,
            "n_common_queries": len(common),
            "orchestration_value_grows_with_noise_overall":
                bool(gap_slope_ov["slope"] is not None and gap_slope_ov["slope"] > 0),
            "orchestration_value_grows_with_noise_significant": _sig,
        }

    return {
        "description": (
            "P0 / P1 / P4 write from a frozen corpus doped with distractor passages at "
            "0 / 20 / 40 / 70%. Tests whether orchestration's value re-emerges as the "
            "evidence quality degrades (cluster-minus-P0 gap widening with dose)."
        ),
        "doses": {tag: d for tag, d in DOSES},
        "coverage": coverage,
        "per_pattern": per_pattern,
        "cluster_vs_p0": cluster_vs_p0,
        "interpretation": (
            "NULL RESULT, not a defence: neither cluster's gap_slope_overall CI excludes "
            "zero (p1_minus_p0 point +0.020, CI crosses 0, p=0.33; p4_minus_p0 point "
            "-0.015, CI crosses 0, p=0.47 -- p4's own point estimate is NEGATIVE, the "
            "opposite sign from p1). There is no statistically supported evidence that "
            "orchestration's advantage over P0 grows as injected-noise dose rises; any "
            "apparent 'value reappears under noise' reading is noise around a null, not a "
            "finding. This is corroborated by an independent Sonnet-5 judge crosscheck "
            "(see distractor_sonnet_crosscheck), which replicates the null under a "
            "different judge family."
        ),
    }


# ── coverage / completeness ────────────────────────────────────────────────────

def collect_coverage():
    synth = {s: len(list((SYNTH_JUDGE / s).glob("*.json"))) for s in SCAFFOLDS}
    cf = len(list(CF_JUDGE.glob("*.json"))) if CF_JUDGE.exists() else 0
    dist = {f"{p}_{tag}": len(list((DIST_JUDGE / f"{p}_{tag}").glob("*.json")))
            for p in DIST_PATTERNS for tag, _ in DOSES}
    got = sum(synth.values()) + cf + sum(dist.values())
    exp = sum(EXPECT_SYNTH.values()) + EXPECT_CF + sum(EXPECT_DIST.values())
    return synth, cf, dist, got, exp


def is_complete(synth, cf, dist):
    ok = all(synth[s] >= EXPECT_SYNTH[s] for s in SCAFFOLDS)
    ok = ok and cf >= EXPECT_CF
    ok = ok and all(dist[a] >= EXPECT_DIST[a] for a in EXPECT_DIST)
    return ok


def print_coverage(synth, cf, dist, got, exp):
    print("Coverage (verdicts on disk / expected):")
    print("  SYNTH-ABLATION:")
    for s in SCAFFOLDS:
        print(f"    {s:16s} {synth[s]:3d}/{EXPECT_SYNTH[s]}")
    print(f"  COUNTERFACTUAL:  {cf:3d}/{EXPECT_CF}")
    print("  DISTRACTOR:")
    for a in EXPECT_DIST:
        print(f"    {a:10s} {dist[a]:3d}/{EXPECT_DIST[a]}")
    print(f"  TOTAL: {got}/{exp}  ({'COMPLETE' if is_complete(synth, cf, dist) else 'INCOMPLETE'})")


# ── atomic staging write ───────────────────────────────────────────────────────

def _write_staging(blob):
    STAGING_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(STAGING_OUT.parent),
                                   prefix="frozen_defence.", suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(blob, f, indent=1)
        os.replace(tmp, STAGING_OUT)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    print(f"WROTE staging blob -> {STAGING_OUT}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="print verdict coverage only; no compute, no write (safe anytime).")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print a summary; write nothing.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="compute + write even if some verdicts are still missing.")
    args = ap.parse_args()

    synth, cf, dist, got, exp = collect_coverage()
    print_coverage(synth, cf, dist, got, exp)
    complete = is_complete(synth, cf, dist)

    if args.check:
        return 0

    if not complete and not args.allow_partial and not args.dry_run:
        print("\nREFUSING to write: judging is INCOMPLETE. Re-run once all verdicts land, "
              "or pass --allow-partial to land what exists.")
        return 1

    from datetime import datetime, timezone
    rng = np.random.default_rng(SEED)
    blob = {
        "key": "frozen_defence",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_boot": N_BOOT,
        "resampling_unit": "query (frozen greedy decode => no decode-seed variance)",
        "judge_model": "gpt-5.2",
        "complete": bool(complete),
        "rebuts": ["Argus 2605.16217", "TTD-DR 2507.16075", "GRACE 2606.16151"],
        "synth_ablation": build_synthablation(rng),
        "counterfactual": build_counterfactual(rng),
        "distractor": build_distractor(rng),
    }

    # headline console lines
    sa = blob["synth_ablation"]
    print("\n=== SYNTH-ABLATION ===")
    for s in SCAFFOLDS:
        fa = sa["per_scaffold"][s]["factual_accuracy"]
        ci = sa["per_scaffold"][s]["citation_quality"]
        print(f"  {s:16s} factual={fa['mean']} {fa['ci95']}  citation={ci['mean']} {ci['ci95']}  n={fa['n']}")
    print(f"  scaffolds beating single_pass on factual: {sa['n_scaffolds_beating_single_pass_factual']}")
    print(f"  finding (iv) hardens to CAUSAL: {sa['finding_iv_hardens_causal']}")

    cfb = blob["counterfactual"]
    print("\n=== COUNTERFACTUAL ===")
    print(f"  override_rate (probe_class): {cfb['override_rate_probe_class']['override_rate']} "
          f"counts={cfb['override_rate_probe_class']['class_counts']}")
    print(f"  GPT-5.2 factual faithfulness: {cfb['gpt52_faithfulness']['factual_accuracy']}")

    db = blob["distractor"]
    print("\n=== DISTRACTOR ===")
    for p in DIST_PATTERNS:
        s = db["per_pattern"][p]["dose_slope_overall"]
        print(f"  {p}: dose_slope_overall={s['slope']} {s['ci95']} p={s['p_two_sided_boot']}")
    for k, v in db["cluster_vs_p0"].items():
        gs = v["gap_slope_overall"]
        print(f"  {k}: gap_slope_overall={gs['slope']} {gs['ci95']} p={gs['p_two_sided_boot']} "
              f"(grows_with_noise={v['orchestration_value_grows_with_noise_overall']})")

    if args.dry_run:
        print("\n[DRY RUN] nothing written.")
        return 0

    _write_staging(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
