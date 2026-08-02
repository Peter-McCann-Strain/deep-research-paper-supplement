#!/usr/bin/env python3
"""build_frozen_vintage.py — canonical-landing builder for the 'frozen_vintage' key.

Lands ONE new key, 'frozen_vintage', into the paper-A canonical store:
    papers/paper_a_bounded_returns/analysis/canonical_numbers.json

WHAT THIS IS
------------
Four local-model arms run under the FROZEN P9 local-baseline scaffold (identical
tools / prompts / retrieval) over the SAME hash-pinned 89-query frozen source set
(data/frozen_corpus_vintage), judged by GPT-5.2 (OpenAI — judge-independent of the
Qwen-family arms). Two contrast axes:

    vintage_axis (release DATE, capacity ~constant 7-8B):
        p9  Qwen2.5-7B-Instruct                (2024-09)  transformers argmax
        p14 DeepSeek-R1-Distill-Qwen-7B        (2025-01)  transformers argmax
        p13 Qwen3-8B                           (2025-04)  llama.cpp GGUF greedy

    capacity_axis (parameters, vintage held at 2024-09):
        p9  Qwen2.5-7B-Instruct                (2024-09)  transformers argmax
        p17 Qwen2.5-14B-Instruct               (2024-09)  llama.cpp GGUF greedy

P17 (14B) shares P9's 2024-09 vintage (x=0 years on the date axis) and is therefore
recorded on a SEPARATE capacity_axis sub-result, NOT as a third vintage-date point
(two points at x=0 would be an axis error) — consistent with build_e8_vintage.py.

METHODOLOGY (June-2026 best practice)
-------------------------------------
(a) Per-arm mean + per-QUERY PAIRED bootstrap 95% CIs. Resampling unit = the 89
    frozen queries; the SAME resampled query-index block is applied to every arm in
    each iteration (valid because all arms answer the identical qid set). Greedy
    decode -> no decode-seed variance. n_boot=10000, seed=20260611.
(b) LENGTH-CONTROLLED scoring. Pooled OLS (np.linalg.lstsq, no statsmodels) over all
    4x89 reports: score ~ arm_dummies + beta*(words/1000 - grand_mean). Length is
    mean-centred over ALL reports, so each arm dummy coefficient IS that arm's
    length-adjusted mean = counterfactual score at grand-mean length. Length is the
    whitespace word count of each report .md — tokenizer-independent, hence comparable
    across the GGUF (p13/p17) and transformers (p9/p14) backends. The manifest `tokens`
    field is deliberately NOT used (per-backbone tokenizer-specific, not comparable).
(c) DECODE-BACKEND caveat recorded: p13+p17 llama.cpp GGUF greedy; p9+p14 transformers
    argmax. Token choice comparable, kernels differ — do NOT over-claim 'only weights
    differ'.
(d) The model-independent GPT-4o extractor (frozen retrieval) is the held-constant
    retrieval control, recorded under retrieval_control.

WRITE SAFETY
------------
Default mode is --dry-run (compute + print, write nothing). --write atomically appends
(tempfile in the SAME dir as the store + os.replace). Append-only: reads the existing
store, mutates ONLY cn['frozen_vintage'], never touches siblings. Refuses to overwrite
an existing 'frozen_vintage' key unless --force. On any failure the temp file is
unlinked (no orphan .tmp). Self-guards (exit 0) if the canonical store is missing.

USAGE
-----
    python scripts/build_frozen_vintage.py            # == --dry-run (safe)
    python scripts/build_frozen_vintage.py --dry-run
    python scripts/build_frozen_vintage.py --write
    python scripts/build_frozen_vintage.py --write --force   # overwrite existing key
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(".")
# NEW canonical location (post-0a80ba6). Resolved from repo root, not the stale path.
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

JUDGE_DIR = ROOT / "results" / "judge_gpt52_frozen_vintage"
REPORT_DIR = ROOT / "results" / "experiments_frozen_vintage"
FROZEN_CORPUS = "data/frozen_corpus_vintage"

KEY = "frozen_vintage"
JUDGE_MODEL = "gpt-5.2"
N_BOOT = 10000
SEED = 20260611

DIMENSIONS = [
    "information_recall", "factual_accuracy", "coverage", "analytical_depth",
    "citation_quality", "logical_coherence", "organization",
    "instruction_following", "attribution_quality",
]

# Arm registry. `dir` = subdir basename under BOTH JUDGE_DIR and REPORT_DIR.
ARMS = [
    {"pattern": "p9", "dir": "base_p9", "label": "P9",
     "model": "Qwen/Qwen2.5-7B-Instruct", "release_date": "2024-09",
     "decode_backend": "llama.cpp GGUF greedy", "axis": "vintage"},
    {"pattern": "p14", "dir": "base_p14_vintage_deepseek_qwen7b", "label": "P14",
     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "release_date": "2025-01",
     "decode_backend": "llama.cpp GGUF greedy", "axis": "vintage"},
    {"pattern": "p13", "dir": "base_p13_vintage_qwen3_8b", "label": "P13",
     "model": "Qwen/Qwen3-8B", "release_date": "2025-04",
     "decode_backend": "llama.cpp GGUF greedy", "axis": "vintage"},
    {"pattern": "p17", "dir": "base_p17_scale_qwen25_14b", "label": "P17",
     "model": "Qwen/Qwen2.5-14B-Instruct", "release_date": "2024-09",
     "decode_backend": "llama.cpp GGUF greedy", "axis": "capacity"},
]

# GGUF-era raw overall means (de-confounded 2026-06-30, all four arms llama.cpp GGUF
# greedy). Rounded to 3dp of the canonical frozen_vintage arms[*].raw_overall_mean
# (p9=0.2784, p14=0.2233, p13=0.3557, p17=0.3037). Self-check constant only.
VERIFIED_MEANS = {"p9": 0.278, "p14": 0.223, "p13": 0.356, "p17": 0.304}


def _load_arm(arm):
    """Return (qids_sorted, score_by_qid, dim_by_qid) for one arm.

    Reads the verdict JSON top-level `overall_score` directly; a future schema rename
    fails loudly (KeyError), which is intended, not silent.
    """
    jdir = JUDGE_DIR / arm["dir"]
    files = sorted(jdir.glob("*.json"))
    score_by_qid = {}
    dim_by_qid = {}
    for fp in files:
        d = json.load(open(fp))
        qid = fp.stem
        score_by_qid[qid] = float(d["overall_score"])
        dim_by_qid[qid] = {
            dim: float(d["dimensions"][dim]["score"]) for dim in DIMENSIONS
        }
    qids = sorted(score_by_qid)
    return qids, score_by_qid, dim_by_qid


def _report_length(arm, qid):
    """Whitespace word count + char count of the report .md (tokenizer-independent)."""
    fp = REPORT_DIR / arm["dir"] / f"{qid}.md"
    text = fp.read_text(encoding="utf-8", errors="replace")
    return len(text.split()), len(text)


def _paired_bootstrap(qids, score_mats, rng):
    """Per-query paired bootstrap.

    score_mats: dict pattern -> np.array aligned to `qids`.
    Returns: per-arm mean-CIs and per-pair diff stats, all from the SAME resampled
    query-index block each iteration (paired).
    """
    n = len(qids)
    patterns = list(score_mats)
    boot_means = {p: np.empty(N_BOOT) for p in patterns}
    pair_specs = {
        "p14_minus_p9": ("p14", "p9"),
        "p13_minus_p14": ("p13", "p14"),
        "p13_minus_p9": ("p13", "p9"),
        "p17_minus_p9": ("p17", "p9"),
    }
    boot_diffs = {k: np.empty(N_BOOT) for k in pair_specs}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)  # same block for every arm => paired
        for p in patterns:
            boot_means[p][b] = score_mats[p][idx].mean()
        for k, (a, c) in pair_specs.items():
            boot_diffs[k][b] = score_mats[a][idx].mean() - score_mats[c][idx].mean()

    mean_ci = {}
    for p in patterns:
        lo, hi = np.percentile(boot_means[p], [2.5, 97.5])
        mean_ci[p] = [round(float(lo), 4), round(float(hi), 4)]

    diff_stats = {}
    for k, (a, c) in pair_specs.items():
        d = boot_diffs[k]
        point = float(score_mats[a].mean() - score_mats[c].mean())
        lo, hi = np.percentile(d, [2.5, 97.5])
        # Two-sided bootstrap p: 2*min(P(d>0), P(d<0)), clamped to [0,1].
        p_gt = float((d > 0).mean())
        p_lt = float((d < 0).mean())
        p_two = min(1.0, 2.0 * min(p_gt, p_lt))
        # A bootstrap whose resamples NEVER cross 0 cannot prove p=0; the smallest
        # resolvable two-sided p with N_BOOT iterations is ~1/N_BOOT. Floor an exact
        # 0.0 to a reported upper bound '<1e-4' (= '<%.0e' % (1/N_BOOT)) rather than
        # claiming an impossible exact zero.
        p_two_rounded = round(p_two, 4)
        if p_two_rounded == 0.0:
            p_two_report = "<%.0e" % (1.0 / N_BOOT)
        else:
            p_two_report = p_two_rounded
        diff_stats[k] = {
            "point": round(point, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided_boot": p_two_report,
        }
    return mean_ci, diff_stats


def _length_control(qids, score_mats, words_mats):
    """Pooled OLS: score ~ arm_dummies + beta*(words/1000 - grand_mean_kwords).

    Length mean-centred over ALL reports => each arm dummy coefficient IS that arm's
    length-adjusted mean (counterfactual score at grand-mean length). One dummy per arm
    (no global intercept) so coefficients are directly the adjusted means.
    """
    patterns = list(score_mats)
    rows_y = []
    rows_X = []
    all_words = np.concatenate([words_mats[p] for p in patterns])
    grand_mean_kwords = float(all_words.mean()) / 1000.0

    n_arms = len(patterns)
    for j, p in enumerate(patterns):
        for i in range(len(qids)):
            y = score_mats[p][i]
            dummies = [0.0] * n_arms
            dummies[j] = 1.0
            length_term = words_mats[p][i] / 1000.0 - grand_mean_kwords
            rows_y.append(y)
            rows_X.append(dummies + [length_term])
    Y = np.asarray(rows_y, dtype=float)
    X = np.asarray(rows_X, dtype=float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    adj_means = {p: round(float(coef[j]), 4) for j, p in enumerate(patterns)}
    beta_per_kwords = float(coef[-1])
    return {
        "adj_means": adj_means,
        "length_coef_per_1000_words": round(beta_per_kwords, 4),
        "grand_mean_words": round(grand_mean_kwords * 1000.0, 1),
    }


def build():
    # --- load all arms, verify qid alignment ---
    loaded = {}
    qid_ref = None
    for arm in ARMS:
        qids, sbq, dbq = _load_arm(arm)
        loaded[arm["pattern"]] = (qids, sbq, dbq)
        if qid_ref is None:
            qid_ref = qids
        elif qids != qid_ref:
            raise SystemExit(
                f"[frozen_vintage] qid mismatch on arm {arm['pattern']}: "
                f"{len(qids)} vs {len(qid_ref)} — arms not aligned, refusing.")
    qids = qid_ref
    n = len(qids)
    qid_equal = all(loaded[a["pattern"]][0] == qids for a in ARMS)

    # --- aligned score/dim/length matrices ---
    score_mats = {}
    words_mats = {}
    chars_mats = {}
    dim_means = {}
    for arm in ARMS:
        p = arm["pattern"]
        _, sbq, dbq = loaded[p]
        score_mats[p] = np.array([sbq[q] for q in qids], dtype=float)
        w = np.empty(n)
        c = np.empty(n)
        for i, q in enumerate(qids):
            w[i], c[i] = _report_length(arm, q)
        words_mats[p] = w
        chars_mats[p] = c
        dim_means[p] = {
            dim: round(float(np.mean([dbq[q][dim] for q in qids])), 4)
            for dim in DIMENSIONS
        }

    # --- bootstrap (fixed seed, deterministic) ---
    rng = np.random.default_rng(SEED)
    mean_ci, diff_stats = _paired_bootstrap(qids, score_mats, rng)

    # --- length control ---
    lc = _length_control(qids, score_mats, words_mats)

    # --- assemble arms block ---
    arms_out = {}
    for arm in ARMS:
        p = arm["pattern"]
        arms_out[arm["dir"]] = {
            "pattern": p,
            "label": arm["label"],
            "model": arm["model"],
            "release_date": arm["release_date"],
            "decode_backend": arm["decode_backend"],
            "axis": arm["axis"],
            "n_queries": n,
            "raw_overall_mean": round(float(score_mats[p].mean()), 4),
            "overall_ci95_paired_bootstrap": mean_ci[p],
            "length_adjusted_overall_mean": lc["adj_means"][p],
            "mean_output_words": round(float(words_mats[p].mean()), 1),
            "mean_output_chars": round(float(chars_mats[p].mean()), 1),
            "per_dimension_mean": dim_means[p],
        }

    out = {
        "_note": (
            "Frozen-vintage local-model comparison. Four arms under the FROZEN P9 "
            "local-baseline scaffold (identical tools/prompts/retrieval), same "
            "hash-pinned 89-query frozen source set, judged by GPT-5.2 (judge-independent "
            "of the Qwen-family arms). vintage_axis = release DATE at ~constant 7-8B "
            "capacity (p9->p14->p13); capacity_axis = parameters at fixed 2024-09 vintage "
            "(p9->p17). P17 (14B) shares P9's vintage so it is a SEPARATE capacity point, "
            "not a third date point (avoids two-points-at-x=0). Per-query PAIRED bootstrap "
            "CIs (queries are the resampling unit; greedy decode => no seed variance) and a "
            "length-debiased score (pooled OLS, length mean-centred so each arm dummy IS its "
            "counterfactual score at grand-mean length) control the verbosity confound. "
            "DE-CONFOUNDED (2026-06-30): all four arms now decode under llama.cpp GGUF greedy "
            "(p9+p14 re-generated via GGUF), so the earlier transformers-vs-GGUF backend split "
            "is removed and the axes isolate the generation model alone."),
        "key_version": "1.0",
        "judge_model": JUDGE_MODEL,
        "scaffold": "P9 local-baseline (frozen)",
        "retrieval_control": {
            "extractor_model": "gpt-4o",
            "note": (
                "The model-independent GPT-4o extractor is held constant across all four "
                "arms (frozen retrieval / source set), so cross-arm differences isolate the "
                "generation model, not retrieval."),
        },
        "n_queries": n,
        "frozen_corpus": FROZEN_CORPUS,
        "dropped_queries": {
            "n_dropped": 1,
            "n_full_set": 90,
            "n_used": n,
            "reason": (
                "query #90 dropped at corpus-build for a benign flaky-fetch; excluded not "
                "re-fetched, to keep the hash-pinned frozen source set stable. The run "
                "manifest declares n_queries=90 but the corpus + all four judged arms "
                "contain exactly 89 query files. Reason recorded per task spec, not parsed "
                "from disk; edit if the true reason differs."),
        },
        "qid_alignment_all_arms_equal": bool(qid_equal),
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "resampling_unit": "query",
            "paired": True,
            "note": (
                "np.random.default_rng(seed); the SAME resampled query-index block is "
                "applied to every arm each iteration (paired). Two-sided p = "
                "2*min(P(d>0),P(d<0))."),
        },
        "length_control": {
            "method": (
                "pooled OLS via np.linalg.lstsq (no statsmodels): "
                "score ~ arm_dummies + beta*(words/1000 - grand_mean_kwords); length "
                "mean-centred over ALL 4x89 reports so each arm dummy coefficient is its "
                "length-adjusted mean = counterfactual score at grand-mean length; shared "
                "length slope beta."),
            "length_unit": (
                "output words (whitespace split of .md; tokenizer-independent so comparable "
                "across GGUF vs transformers backends — manifest `tokens` deliberately NOT "
                "used, it is per-backbone tokenizer-specific)"),
            "length_coef_per_1000_words": lc["length_coef_per_1000_words"],
            "grand_mean_words": lc["grand_mean_words"],
        },
        "decode_backend_caveat": (
            "all four arms llama.cpp GGUF greedy (de-confounded 2026-06-30); no backend split"),
        "decode_backend_confound": {
            "transformers_argmax": [],
            "llama_cpp_gguf_greedy": ["p9", "p14", "p13", "p17"],
            "vintage_axis_confounded": False,
            "capacity_axis_confounded": False,
            "de_confounded": True,
            "de_confound_date": "2026-06-30",
            "note": (
                "DE-CONFOUNDED (2026-06-30). p9 + p14 were RE-GENERATED via llama.cpp "
                "GGUF greedy on the SAME 89 frozen sources, so ALL FOUR arms now share "
                "one decode backend (llama.cpp GGUF greedy). The earlier transformers-vs-"
                "GGUF backend confound on the vintage axis is REMOVED: the release-date "
                "axis (p9 -> p14 -> p13) and the capacity axis (p9 -> p17) now isolate "
                "the generation model alone, with backend held constant. The capacity "
                "result survives the de-confound (p17 - p9 = +0.025, p=0.005). Earlier "
                "transformers-backend reports are archived under "
                "archive/frozen_vintage_transformers_p9p14_*."),
        },
        "arms": arms_out,
        "vintage_axis": {
            "ordered_patterns": ["p9", "p14", "p13"],
            "description": (
                "Release DATE axis at ~constant 7-8B capacity: Qwen2.5-7B (2024-09) -> "
                "DeepSeek-R1-Distill-Qwen-7B (2025-01) -> Qwen3-8B (2025-04)."),
            "paired_diffs": {
                "p14_minus_p9": diff_stats["p14_minus_p9"],
                "p13_minus_p14": diff_stats["p13_minus_p14"],
                "p13_minus_p9": diff_stats["p13_minus_p9"],
            },
        },
        "capacity_axis": {
            "ordered_patterns": ["p9", "p17"],
            "description": (
                "Parameter/capacity axis at FIXED 2024-09 vintage: Qwen2.5-7B -> "
                "Qwen2.5-14B (same release vintage, ~2x params). Separate from the date "
                "axis to avoid two points at x=0 years."),
            "paired_diffs": {
                "p17_minus_p9": diff_stats["p17_minus_p9"],
            },
        },
        "verified_means_expected": VERIFIED_MEANS,
    }
    return out


def _print_dry(out):
    print(f"[{KEY}] DRY-RUN — computed, nothing written.")
    print(f"  n_queries={out['n_queries']}  qid_aligned={out['qid_alignment_all_arms_equal']}")
    print(f"  bootstrap: n_boot={out['bootstrap']['n_boot']} seed={out['bootstrap']['seed']}")
    for d, a in out["arms"].items():
        print(f"  {a['pattern']:>4} {a['label']:>4} raw={a['raw_overall_mean']:.4f} "
              f"ci={a['overall_ci95_paired_bootstrap']} "
              f"len_adj={a['length_adjusted_overall_mean']:.4f} "
              f"words={a['mean_output_words']:.0f}  ({a['axis']})")
    print("  vintage paired diffs:")
    for k, v in out["vintage_axis"]["paired_diffs"].items():
        print(f"    {k}: {v['point']:+.4f} {v['ci95']} p={v['p_two_sided_boot']}")
    print("  capacity paired diffs:")
    for k, v in out["capacity_axis"]["paired_diffs"].items():
        print(f"    {k}: {v['point']:+.4f} {v['ci95']} p={v['p_two_sided_boot']}")
    print(f"  length_coef_per_1000_words={out['length_control']['length_coef_per_1000_words']} "
          f"grand_mean_words={out['length_control']['grand_mean_words']}")
    ok = True
    for arm in ARMS:
        p = arm["pattern"]
        got = out["arms"][arm["dir"]]["raw_overall_mean"]
        exp = VERIFIED_MEANS[p]
        match = abs(round(got, 3) - exp) < 1e-9
        ok = ok and match
        print(f"  verify {p}: got={got:.4f} expected~{exp} {'OK' if match else 'MISMATCH'}")
    print(f"  VERIFIED-MEANS {'ALL OK' if ok else 'FAILED'}")


def _atomic_append(out, force):
    cn = json.load(open(CANON))
    if KEY in cn and not force:
        print(f"[{KEY}] REFUSING to overwrite existing key '{KEY}' (use --force).")
        return 1
    cn[KEY] = out
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(ANA), prefix="canonical_numbers.", suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cn, f, indent=1)
        os.replace(tmp, CANON)
        tmp = None
    except BaseException:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    print(f"[{KEY}] WROTE key '{KEY}' -> {CANON}  (store now {len(cn)} keys)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print, write nothing (default)")
    ap.add_argument("--write", action="store_true",
                    help="atomically append the key to the canonical store")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing key (only with --write)")
    args = ap.parse_args()

    if not CANON.exists():
        print(f"[{KEY}] canonical store missing at {CANON}; nothing to do (self-guard).")
        return 0
    if not JUDGE_DIR.exists():
        print(f"[{KEY}] judge dir missing at {JUDGE_DIR}; nothing to do (self-guard).")
        return 0

    out = build()

    if args.write:
        return _atomic_append(out, args.force)
    _print_dry(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
