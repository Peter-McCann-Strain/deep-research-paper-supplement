#!/usr/bin/env python
"""A3_localbench: ingest the P9/P10 local 7B-tier benchmark judge verdicts into canonical.

150 GPT-5.2 single-judge verdicts produced today live under
  results/local_benchmark/judge_gpt52/lb_{p9,p10}_{deepsearch_qa,draco,freshwiki,litqa2,research_qa}/*.json
(10 cells x 15 queries). Each verdict file carries an `overall_score` in [0,1].

This script reads those on-disk verdicts and appends canonical_numbers.json['local_benchmark'],
containing:
  - per_cell:   per-pattern x per-benchmark mean overall_score + n (the headline grid)
  - per_pattern: pooled-over-benchmarks mean (cell-mean macro-average AND query-pooled micro-mean)
  - per_benchmark: pooled-over-patterns mean
  - tier_7b_external_validation: the 7B-tier external-validation summary (P9 vs P10 pooled,
    the RL-training delta P10 - P9, and a query-paired Wilcoxon + seeded-bootstrap CI on that delta)

Single judge = GPT-5.2 (cloud Azure JUDGE endpoint); no Opus anywhere. P9 = Qwen2.5-7B-Instruct
on the P0 single-pass architecture; P10 = GAIR/DeepResearcher-7b RL-trained agent. The P10 - P9
contrast isolates the RL-training effect at fixed 7B scale on held-out external benchmarks.

Deterministic: sorted file inputs, fixed bootstrap seed (0). Idempotent: pure function of the
on-disk verdicts; rerunning overwrites only canonical_numbers.json['local_benchmark'] and never
mutates any verdict/report. Atomic write via os.replace.

Appends canonical_numbers.json['local_benchmark'].
"""
import json, glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "."
# Canonical store was MOVED here by commit 0a80ba6 from papers/paper_a_bounded_returns/analysis/.
# Use the real on-disk location (the old path no longer exists).
ANA = f"{ROOT}/papers/paper_a_bounded_returns/analysis"
JUDGE_DIR = f"{ROOT}/results/local_benchmark/judge_gpt52"

PATTERNS = ["p9", "p10"]
BENCHMARKS = ["deepsearch_qa", "draco", "freshwiki", "litqa2", "research_qa"]
JUDGE = "gpt52"
BOOT_SEED = 0
N_BOOT = 10000


def _cell_dir(pat, bench):
    return f"{JUDGE_DIR}/lb_{pat}_{bench}"


def load_cell(pat, bench):
    """Return {query_id: overall_score} for one pattern x benchmark cell, sorted-deterministic."""
    out = {}
    for f in sorted(glob.glob(f"{_cell_dir(pat, bench)}/*.json")):
        d = json.load(open(f))
        s = d.get("overall_score")
        if s is None:
            continue
        # Prefer the explicit query_id field; fall back to filename stem.
        qid = d.get("query_id") or os.path.basename(f)[:-5]
        out[qid] = float(s)
    return out


def r4(x):
    return round(float(x), 4) if x is not None else None


# ---- load every cell ----------------------------------------------------------
cells = {(p, b): load_cell(p, b) for p in PATTERNS for b in BENCHMARKS}

# ---- per-cell grid (per-pattern x per-benchmark means) ------------------------
per_cell = {}
for p in PATTERNS:
    per_cell[p] = {}
    for b in BENCHMARKS:
        vals = list(cells[(p, b)].values())
        per_cell[p][b] = {
            "mean": r4(np.mean(vals)) if vals else None,
            "n": len(vals),
        }

# ---- per-pattern pooled (macro = mean of cell-means; micro = mean over all queries) ----
per_pattern = {}
for p in PATTERNS:
    cell_means = [per_cell[p][b]["mean"] for b in BENCHMARKS if per_cell[p][b]["mean"] is not None]
    all_vals = [v for b in BENCHMARKS for v in cells[(p, b)].values()]
    per_pattern[p] = {
        "mean_macro": r4(np.mean(cell_means)) if cell_means else None,  # equal weight per benchmark
        "mean_micro": r4(np.mean(all_vals)) if all_vals else None,      # equal weight per query
        "std_micro": r4(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else None,
        "n_queries": len(all_vals),
        "n_benchmarks": len(cell_means),
    }

# ---- per-benchmark pooled (over patterns) -------------------------------------
per_benchmark = {}
for b in BENCHMARKS:
    bvals = [v for p in PATTERNS for v in cells[(p, b)].values()]
    per_benchmark[b] = {
        "mean": r4(np.mean(bvals)) if bvals else None,
        "n": len(bvals),
        "by_pattern": {p: per_cell[p][b]["mean"] for p in PATTERNS},
    }

# ---- 7B-tier external-validation summary: RL-training delta P10 - P9 ----------
# Paired by (benchmark, query_id) across the two patterns on the same external items.
#
# DEFECT FIXED (heterogeneity audit): the prior code pooled all 75 paired deltas across
# five benchmarks of DIFFERENT scales via a single flat Wilcoxon + flat i.i.d. bootstrap.
# That treats the benchmark as fixed and ignores between-benchmark heterogeneity, so the
# effective n is 5 benchmark clusters, not 75 independent queries -> anticonservative CI.
#
# FIX: fit the benchmark as a RANDOM EFFECT. We (a) report each benchmark's own n + delta,
# (b) give a query-pooled (micro / fixed-effect) estimate for reference, and (c) give the
# headline RANDOM-EFFECTS pooled estimate = the macro-mean over the five per-benchmark deltas,
# with a benchmark-CLUSTERED CI from a two-stage block bootstrap (resample the 5 benchmarks,
# then queries within each chosen benchmark) -- the same cluster-aware scheme as
# build_oracle_robust_ci.py:49 block_boot, with the benchmark as the cluster.

# Per-benchmark paired deltas (each a list of P10-P9 over shared queries in that benchmark).
per_bench_deltas = {}
for b in BENCHMARKS:
    c9, c10 = cells[("p9", b)], cells[("p10", b)]
    per_bench_deltas[b] = np.array(
        [c10[q] - c9[q] for q in sorted(set(c9) & set(c10))], dtype=float
    )
# Flat-pooled deltas (kept only for the fixed-effect / micro reference point).
d = np.concatenate([per_bench_deltas[b] for b in BENCHMARKS]) if per_bench_deltas else np.array([])

# Per-benchmark summary: n + delta (so heterogeneity is visible).
per_benchmark_delta = {}
for b in BENCHMARKS:
    db = per_bench_deltas[b]
    per_benchmark_delta[b] = {"n": int(db.size), "delta_p10_minus_p9": r4(np.mean(db)) if db.size else None}


def _block_boot_benchmark(bench_arrays, rng, reps=N_BOOT):
    """Two-stage block bootstrap with the benchmark as cluster (see build_oracle_robust_ci.py:49).

    Resample the benchmarks (clusters) with replacement, then queries within each chosen
    benchmark. The per-iteration statistic is the random-effects (macro) estimate: the mean of
    the chosen benchmarks' resampled within-benchmark mean deltas (equal weight per benchmark).
    """
    benches = [b for b in BENCHMARKS if bench_arrays[b].size > 0]
    nben = len(benches)
    out = np.empty(reps)
    for i in range(reps):
        chosen = rng.integers(0, nben, nben)
        macro = np.empty(nben)
        for j, c in enumerate(chosen):
            arr = bench_arrays[benches[c]]
            macro[j] = arr[rng.integers(0, len(arr), len(arr))].mean()
        out[i] = macro.mean()
    return out


delta_micro = delta_macro = test = None
if d.size > 0:
    from scipy.stats import wilcoxon
    try:
        pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        pv = 1.0
    # Fixed-effect / micro reference: flat query-pooled mean (NOT used for the headline CI).
    delta_micro = r4(np.mean(d))
    # Headline random-effects (macro) estimate = equal-weight mean over the 5 per-benchmark deltas.
    bench_means = np.array([np.mean(per_bench_deltas[b]) for b in BENCHMARKS if per_bench_deltas[b].size])
    delta_macro = r4(np.mean(bench_means))
    # Benchmark-clustered CI for the random-effects estimate.
    rng = np.random.default_rng(BOOT_SEED)
    boots = _block_boot_benchmark(per_bench_deltas, rng)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    test = {
        "n_paired_total": int(d.size),
        "n_benchmarks": int(sum(per_bench_deltas[b].size > 0 for b in BENCHMARKS)),
        "per_benchmark": per_benchmark_delta,
        "delta_p10_minus_p9_macro": delta_macro,   # headline: random-effects (benchmark) pooled
        "delta_p10_minus_p9_micro": delta_micro,   # reference: flat query-pooled (fixed effect)
        "wilcoxon_p": round(pv, 4),
        "ci95_macro_benchmark_clustered": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci_method": "two-stage block bootstrap, benchmark as random-effect cluster "
                     "(resample 5 benchmarks, then queries within); seed=%d, reps=%d" % (BOOT_SEED, N_BOOT),
        "significant": bool(ci_lo > 0.0),  # CI-based: excludes 0 under the clustered interval
        "wilcoxon_note": "Wilcoxon p shown for reference only; it pools across benchmarks of "
                         "different scales and ignores between-benchmark heterogeneity, so the "
                         "benchmark-clustered macro CI governs the significance call.",
    }

tier = {
    "p9_mean_micro": per_pattern["p9"]["mean_micro"],
    "p10_mean_micro": per_pattern["p10"]["mean_micro"],
    "p9_mean_macro": per_pattern["p9"]["mean_macro"],
    "p10_mean_macro": per_pattern["p10"]["mean_macro"],
    "rl_training_delta_test": test,
    "interpretation": "P10 (GAIR/DeepResearcher-7b, RL-trained) minus P9 (Qwen2.5-7B-Instruct, "
                      "P0 single-pass arch) at fixed 7B scale on five external benchmarks; "
                      "isolates the RL-training effect out-of-distribution. Headline pooled "
                      "delta is the random-effects (benchmark-clustered) macro estimate; the "
                      "CI accounts for between-benchmark heterogeneity (effective n = 5 "
                      "benchmark clusters, not 75 queries).",
}

out = {
    "judge": JUDGE,
    "judge_model": "gpt-5.2",
    "scale": "local_7b",
    "patterns": {"p9": "Qwen2.5-7B-Instruct (P0 single-pass arch)",
                 "p10": "GAIR/DeepResearcher-7b (RL-trained agent)"},
    "benchmarks": BENCHMARKS,
    "n_verdicts": int(sum(len(v) for v in cells.values())),
    "per_cell": per_cell,
    "per_pattern": per_pattern,
    "per_benchmark": per_benchmark,
    "tier_7b_external_validation": tier,
    "source": "results/local_benchmark/judge_gpt52/lb_{p9,p10}_{benchmark}/*.json",
    "note": "GPT-5.2 single-judge external validation of the 7B tier; no Opus judge. "
            "Deterministic (sorted inputs, bootstrap seed 0); idempotent rebuild.",
}

_WRITE = ("--write" in __import__("sys").argv) and ("--dry-run" not in __import__("sys").argv)
if _WRITE:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    cn["local_benchmark"] = out
    _tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(_tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(_tmp, f"{ANA}/canonical_numbers.json")
else:
    print("[dry-run] canonical_numbers.json NOT written (pass --write to persist).")

print(json.dumps({
    "n_verdicts": out["n_verdicts"],
    "per_pattern": {p: {k: per_pattern[p][k] for k in ("mean_micro", "mean_macro", "n_queries")}
                    for p in PATTERNS},
    "tier_7b_external_validation": tier,
}, indent=1))
