#!/usr/bin/env python3
"""E7 SELECTOR-KAPPA — canonicalise the realizable best-of-N selector contrast at matched
kappa, as the inference-time de-risk metric for E10's Gate G2.

WHAT THIS IS (and is NOT)
-------------------------
This is a PURE-CPU, READ-ONLY distiller. The heavy lifting (parsing the GPT-5.2 variance
replicate corpus, bootstrapping every selector arm) was already done by
``scripts/run_e7_selector.py`` and frozen to disk at::

    reports/e7_selector_results.json            (the ~5.5 kB standalone E7 artefact)

That generator ALSO tries to merge its result into the canonical store, but it hardcodes the
PRE-MOVE path ``papers/paper_a_bounded_returns/analysis/canonical_numbers.json`` (deleted by commit
0a80ba6 when the store moved to ``papers/paper_a_bounded_returns/analysis/``), so its canonical
merge crashes on the final write and the E7 numbers never reach the paper. This script closes
that gap WITHOUT rerunning anything: it reads the frozen standalone JSON and emits the canonical
key ``e7_selector_kappa`` against the CORRECT new path. No GPU, no paid API, no judge call, no
regeneration — only arithmetic over an existing file.

THE CANONICAL KEY — ``e7_selector_kappa``
-----------------------------------------
A compact, paper-facing block holding:
  * ``selector_ladder``  — realized overall-score gain of each realizable selector over the
    single-run mean, framed as the ORACLE upper bound -> {gpt52_noise, gpt4o_noise} realizable
    arms -> RANDOM lower bound. (the "what can a deployed selector actually buy you" curve.)
  * ``matched_kappa``    — for each target kappa in the sweep, the STRUCTURED-noise selector gain
    vs the RANDOM-noise selector gain at the SAME marginal flip rate, and their difference
    ``structured_minus_random``. This is the selector contrast at matched kappa.
  * ``gate_g2``          — the de-risk verdict for E10 Gate G2. G2 asks whether STRUCTURED judge
    error (criteria flip together) costs more SELECTION skill than the same amount of RANDOM
    (i.i.d.) error at matched kappa. The plan's gate is "(b) ~= (e) at matched kappa" -> if the
    structured-vs-random selector gap is ~0 across the kappa sweep, E10 should be REFRAMED from
    "structured vs random" to "magnitude + rescue" BEFORE any GPU is committed. We compute the
    largest |structured - random| over the sweep and a small-margin verdict against a documented
    threshold, so the paper can cite a single de-risk number.
  * ``drjudge_selector``— carried through verbatim from the standalone: the (omitted) small-model
    selector arm and the exact hook to add it once GAIR/DeepResearcher-7b is on disk. NO small
    model is used as an authoritative judge anywhere here.

DETERMINISM / IDEMPOTENCE / SAFETY
----------------------------------
  * Deterministic: a single arithmetic pass over a fixed on-disk JSON. No RNG, no sampling.
  * Idempotent: re-running overwrites only canonical_numbers.json['e7_selector_kappa'] with the
    identical block; every other key is preserved by a load-merge-write.
  * Self-guarding: if reports/e7_selector_results.json is absent, prints a notice and exits 0
    (so rebuild_all.sh stays green before E7 has been run), writing nothing.
  * Atomic write via a temp file + os.replace (never a partial canonical).
  * NEVER clobbers verdicts/reports — it only touches the analysis canonical JSON.

Usage::
    python scripts/build_e7_selector_kappa.py            # merge into canonical (new path)
    python scripts/build_e7_selector_kappa.py --dry-run  # print block, write nothing
    python scripts/build_e7_selector_kappa.py --no-canonical  # print + standalone only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

# Repo-rooted, move-proof paths. parents[1] == repo root (this file lives in scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
E7_STANDALONE = _REPO_ROOT / "reports" / "e7_selector_results.json"
CANON = (_REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
         / "canonical_numbers.json")

CANONICAL_KEY = "e7_selector_kappa"

# G2 de-risk threshold: the structured-minus-random selector gap (in overall-score points) below
# which we treat structured and random selector noise as EQUIVALENT in selection skill -> the
# gate fires "(b) ~= (e)" and E10 reframes. Anchored at one bootstrap step (~0.005) of the arms'
# CI half-widths in the standalone; documented, not fitted on any outcome.
G2_EQUIVALENCE_THRESHOLD = 0.005

# Seeded determinism for the paired bootstrap on the structured-minus-random contrast.
# Anchored to the E7 generator's master seed so the de-risk CI reproduces bit-for-bit.
G2_BOOT_SEED = 20260613
G2_N_BOOT = 10000


def _round(x, n=4):
    return round(float(x), n) if x is not None else None


# ── Paired one-sample TOST on a vector of per-kappa diffs ──────────────────────
# Copied verbatim from scripts/build_e5_equivalence.py:127 (no shared module yet).
def tost_one_sample(diffs: np.ndarray, bound: float) -> dict:
    """TOST that the mean of `diffs` lies within +/-bound (paired one-sample t)."""
    n = len(diffs)
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df_ = n - 1
    # H0_lower: mu <= -bound (reject when mean sufficiently ABOVE -bound)
    t_lower = (m - (-bound)) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df_))
    # H0_upper: mu >= +bound (reject when mean sufficiently BELOW +bound)
    t_upper = (m - bound) / se
    p_upper = float(stats.t.cdf(t_upper, df_))
    p_tost = max(p_lower, p_upper)
    # 90% CI (TOST <-> 90% CI inside +/-bound at alpha=0.05)
    tcrit90 = float(stats.t.ppf(0.95, df_))
    ci90 = [round(m - tcrit90 * se, 4), round(m + tcrit90 * se, 4)]
    return {
        "n": n, "mean_diff": round(m, 4), "sd": round(sd, 4), "se": round(se, 5),
        "bound": bound, "p_lower": round(p_lower, 4), "p_upper": round(p_upper, 4),
        "p_tost": round(p_tost, 4), "equivalent_at_05_alpha": bool(p_tost < 0.05),
        "ci90_inside_bound": ci90,
        "ci90_within_bound": bool(ci90[0] > -bound and ci90[1] < bound),
    }


def mde80_one_sample(diffs: np.ndarray) -> float:
    """Two-sided alpha=0.05 80%-power MDE for a one-sample paired t at this n/SD.

    Copied from scripts/build_e5_equivalence.py:153.
    """
    n = len(diffs)
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df_ = n - 1
    tcrit = float(stats.t.ppf(0.975, df_))
    tpow = float(stats.t.ppf(0.80, df_))
    return float((tcrit + tpow) * se)


def paired_bootstrap_ci90(diffs: np.ndarray, bound: float,
                          seed: int = G2_BOOT_SEED, b: int = G2_N_BOOT) -> dict:
    """Seeded resample of the paired per-kappa diffs; 90% CI on the mean diff.

    Adapted from scripts/build_e5_equivalence.py:164 (paired_bootstrap_ci), switched to a
    90% interval to match the TOST/Gate-G2 +/-bound equivalence convention. Resamples the
    paired (structured - random) contrast across the kappa grid with a fixed seed and sorted
    input, so the de-risk CI is deterministic.
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = np.arange(n)
    boot = np.array([diffs[rng.choice(idx, n, replace=True)].mean() for _ in range(b)])
    ci90 = [round(float(np.percentile(boot, 5)), 4),
            round(float(np.percentile(boot, 95)), 4)]
    return {
        "mean": round(float(diffs.mean()), 4),
        "ci90": ci90,
        "ci90_within_bound": bool(ci90[0] > -bound and ci90[1] < bound),
        "n_boot": b, "seed": seed, "n_kappa": n,
    }


def build_block(e7: dict) -> dict:
    """Distil the frozen E7 standalone into the compact canonical block. Pure arithmetic."""
    arms = e7.get("arms", {})
    flip = e7.get("flip_arms", {})
    smr = e7.get("structured_minus_random_gain", {})
    calib = e7.get("_calibration", {})

    def arm_gain(name):
        a = arms.get(name, {})
        return {
            "gain": _round(a.get("gain")),
            "gain_ci95": [_round(c) for c in a.get("gain_ci95", [None, None])],
            "selected_mean": _round(a.get("selected_mean")),
            "n_cells": a.get("n_cells"),
        }

    # (a) ORACLE upper bound -> realizable {gpt52,gpt4o}_noise arms -> (b) RANDOM lower bound.
    selector_ladder = {
        "single_run_mean": _round(arms.get("oracle", {}).get("single_run_mean")),
        "oracle_upper_bound": arm_gain("oracle"),
        "gpt52_noise": arm_gain("gpt52_noise"),
        "gpt4o_noise": arm_gain("gpt4o_noise"),
        "random_lower_bound": arm_gain("random"),
        "_note": ("Realized overall-score GAIN over the single-run mean. ORACLE = argmax of TRUE "
                  "GPT-5.2 score (upper bound); RANDOM = uniform pick (lower bound); the noise "
                  "arms are realizable second-pass selectors at matched spend."),
    }

    # Selector contrast at matched kappa: structured vs random flip-noise at the SAME marginal
    # flip rate, per target kappa. The de-risk signal is structured_minus_random.
    matched_kappa = {}
    for kappa in calib.get("kappa_targets", []):
        ktag = f"kappa{kappa:.2f}"
        s = flip.get(f"{ktag}_structured", {})
        r = flip.get(f"{ktag}_random", {})
        diff = smr.get(ktag)
        if diff is None and s.get("gain") is not None and r.get("gain") is not None:
            diff = s["gain"] - r["gain"]
        matched_kappa[ktag] = {
            "target_kappa": kappa,
            "marginal_flip_p": s.get("marginal_flip_p", r.get("marginal_flip_p")),
            "structured_gain": _round(s.get("gain")),
            "structured_gain_ci95": [_round(c) for c in s.get("gain_ci95", [None, None])],
            "random_gain": _round(r.get("gain")),
            "random_gain_ci95": [_round(c) for c in r.get("gain_ci95", [None, None])],
            "structured_minus_random": _round(diff),
        }

    # Gate G2 verdict: is structured ~= random across the whole sweep?
    #
    # FIX (real equivalence test, not a bare point): the old gate fired on a BARE POINT
    # estimate max_abs_structured_minus_random <= 0.005. That ignores sampling error and can
    # declare equivalence on noise. We now require a REAL equivalence test on the paired
    # (structured - random) contrast across the kappa grid: a paired one-sample TOST plus a
    # seeded paired bootstrap 90% CI, and declare equivalence ONLY if the 90% CI lies inside
    # +/-0.005. The point estimate is retained for continuity, and an underpowered flag is
    # raised when the 80%-power MDE exceeds the equivalence bound (too few kappa points to
    # resolve a 0.005 effect either way).
    diffs_map = {k: v["structured_minus_random"] for k, v in matched_kappa.items()
                 if v["structured_minus_random"] is not None}
    # Sorted by kappa tag for deterministic paired vector.
    diffs = np.array([diffs_map[k] for k in sorted(diffs_map)], dtype=float)
    max_abs = max((abs(d) for d in diffs_map.values()), default=None)
    worst_kappa = (max(diffs_map, key=lambda k: abs(diffs_map[k])) if diffs_map else None)

    tost = paired_ci = None
    mde80 = underpowered = None
    tost_ci90_within = boot_ci90_within = False
    if len(diffs) >= 2:
        tost = tost_one_sample(diffs, G2_EQUIVALENCE_THRESHOLD)
        paired_ci = paired_bootstrap_ci90(diffs, G2_EQUIVALENCE_THRESHOLD)
        mde80 = _round(mde80_one_sample(diffs))
        underpowered = bool(mde80 is not None and mde80 > G2_EQUIVALENCE_THRESHOLD)
        # Parametric TOST 90% CI (small-n honest) and the seeded bootstrap 90% CI.
        tost_ci90_within = bool(tost["ci90_within_bound"])
        boot_ci90_within = bool(paired_ci["ci90_within_bound"])

    point_within = (max_abs is not None and max_abs <= G2_EQUIVALENCE_THRESHOLD)
    # Equivalence is DECLARED only when (a) the parametric TOST 90% CI clears +/-bound AND
    # (b) the test is NOT underpowered. The bare point and the small-n bootstrap CI are
    # reported but do NOT gate: at n=3 the percentile bootstrap under-covers and would fire
    # on noise, which is exactly the failure mode this fix removes.
    ci90_within = bool(tost_ci90_within)
    gate_fires = bool(tost_ci90_within and not underpowered)
    gate_g2 = {
        "question": ("Does STRUCTURED judge error (criteria flip together) cost more SELECTION "
                     "skill than the same amount of RANDOM (i.i.d.) error at matched kappa?"),
        "metric": "paired_tost_structured_minus_random",
        # Point estimate retained (continuity); no longer the gate criterion on its own.
        "max_abs_structured_minus_random": _round(max_abs),
        "point_within_bound": bool(point_within),
        "worst_kappa": worst_kappa,
        "equivalence_threshold": G2_EQUIVALENCE_THRESHOLD,
        "n_kappa": int(len(diffs)),
        # Real equivalence test on the paired contrast across the kappa grid.
        "tost": tost,
        "p_tost": (tost["p_tost"] if tost else None),
        "paired_bootstrap_ci90": paired_ci,
        "ci90_within_bound": ci90_within,
        "mde80_paired": mde80,
        "underpowered": underpowered,
        "gate_fires_b_approx_e": gate_fires,
        "implication": (
            "(b) ~= (e): structured and random selector noise are statistically EQUIVALENT in "
            "selection skill at matched kappa (paired 90% CI within +/-{thr}) -> E10 should "
            "REFRAME from 'structured vs random' to 'magnitude + rescue' BEFORE committing GPU."
            .format(thr=G2_EQUIVALENCE_THRESHOLD)
            if gate_fires else
            "NOT DECLARED equivalent: although the point gap is small "
            "(max|structured-random|={pt}), the paired 90% CI on the structured-vs-random "
            "contrast is NOT contained within +/-{thr} (p_tost={p}{up}), so the equivalence "
            "gate does not fire on the available kappa grid; do not reframe E10 on this evidence "
            "alone.".format(
                pt=_round(max_abs), thr=G2_EQUIVALENCE_THRESHOLD,
                p=(tost["p_tost"] if tost else None),
                up="; UNDERPOWERED" if underpowered else "")
        ),
        "_plan_ref": "RESEARCH_PLAN_2026H2.md Gate G2: '(b) ~= (e) at matched kappa'",
    }

    return {
        "_what": ("E7 realizable best-of-N selector contrast at matched kappa; the inference-time "
                  "de-risk metric for E10 Gate G2."),
        "source_artifact": str(E7_STANDALONE.relative_to(_REPO_ROOT)),
        "judge": e7.get("judge"),
        "corpus": e7.get("corpus"),
        "seed": e7.get("seed"),
        "n_boot": e7.get("n_boot"),
        "n_cells_total": e7.get("n_cells_total"),
        "n_architectures": e7.get("n_architectures"),
        "architectures": e7.get("architectures"),
        "selector_ladder": selector_ladder,
        "matched_kappa": matched_kappa,
        "gate_g2": gate_g2,
        "calibration": {
            "sigma_gpt52_run_sd": calib.get("sigma_gpt52_run_sd"),
            "sigma_gpt4o": calib.get("sigma_gpt4o"),
            "gpt4o_noise_multiplier": calib.get("gpt4o_noise_multiplier"),
            "gpt4o_multiplier_source": calib.get("gpt4o_multiplier_source"),
            "gpt4o_multiplier_band": calib.get("gpt4o_multiplier_band"),
            "kappa_targets": calib.get("kappa_targets"),
            "kappa_to_flip_p": calib.get("kappa_to_flip_p"),
        },
        "drjudge_selector": e7.get("drjudge_selector"),
    }


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the would-be canonical block; write nothing.")
    ap.add_argument("--no-canonical", action="store_true",
                    help="print the block; do NOT merge into canonical_numbers.json.")
    args = ap.parse_args()

    if not E7_STANDALONE.exists():
        print(f"[e7_selector_kappa] {E7_STANDALONE} not present yet; nothing to canonicalise. "
              f"Run scripts/run_e7_selector.py first. (exit 0, wrote nothing)", file=sys.stderr)
        return 0

    e7 = json.loads(E7_STANDALONE.read_text())
    block = build_block(e7)

    summary = {
        "selector_ladder": {
            "oracle": block["selector_ladder"]["oracle_upper_bound"]["gain"],
            "gpt52_noise": block["selector_ladder"]["gpt52_noise"]["gain"],
            "gpt4o_noise": block["selector_ladder"]["gpt4o_noise"]["gain"],
            "random": block["selector_ladder"]["random_lower_bound"]["gain"],
        },
        "matched_kappa_structured_minus_random": {
            k: v["structured_minus_random"] for k, v in block["matched_kappa"].items()},
        "gate_g2": {
            "max_abs_structured_minus_random":
                block["gate_g2"]["max_abs_structured_minus_random"],
            "p_tost": block["gate_g2"]["p_tost"],
            "paired_bootstrap_ci90": (block["gate_g2"]["paired_bootstrap_ci90"] or {}).get("ci90"),
            "ci90_within_bound": block["gate_g2"]["ci90_within_bound"],
            "underpowered": block["gate_g2"]["underpowered"],
            "gate_fires_b_approx_e": block["gate_g2"]["gate_fires_b_approx_e"],
        },
    }
    print(json.dumps({CANONICAL_KEY: summary}, indent=1))

    if args.dry_run:
        print(f"\n[dry-run] canonical_numbers.json['{CANONICAL_KEY}'] NOT written.",
              file=sys.stderr)
        return 0
    if args.no_canonical:
        print(f"\n[no-canonical] standalone block computed; canonical NOT touched.",
              file=sys.stderr)
        return 0

    if not CANON.exists():
        print(f"[e7_selector_kappa] canonical store missing at {CANON}; refusing to create it "
              f"from scratch (run build_numbers.py first). (exit 0, wrote nothing)",
              file=sys.stderr)
        return 0

    canon = json.loads(CANON.read_text())
    canon[CANONICAL_KEY] = block
    atomic_write_json(CANON, canon)
    print(f"\n[full] merged canonical_numbers.json['{CANONICAL_KEY}'] -> {CANON}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
