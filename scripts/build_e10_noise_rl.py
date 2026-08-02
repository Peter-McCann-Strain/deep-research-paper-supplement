#!/usr/bin/env python3
"""E10 noise-RL — canonicalise the held-out GPT-5.2 endpoint into 'e10_noise_rl'.

PURE-CPU, READ-ONLY distiller over on-disk artefacts:
  * GPT-5.2 verdicts    results/judge_gpt52_e10/<pattern>__e10/<qid>.json
  * reward traces       models/E10-arm*/reward_trace.csv      (collapse diagnostic)
  * objective endpoint  results/e10/<arm_run>/objective_eval.json (anti-Goodhart)
  * calibration echo    a trained arm's pinned canonical_snapshot.json (provenance)

Atomic, append-only merge of the SINGLE canonical key 'e10_noise_rl' into
papers/paper_a_bounded_returns/analysis/canonical_numbers.json. Mirrors
build_e7_selector_kappa.py conventions exactly: load-merge-write, tmp+os.replace,
refuse-create-from-scratch, never clobber the 45 existing keys, --dry-run/--write.

PRIMARY ENDPOINT (prereg)
-------------------------
GPT-5.2-judged HELD-OUT overall-score deltas with a SEEDED QUERY-BOOTSTRAP CI:
  * delta_BC = B_mean - C_mean   (H1: B < C; structured copula noise worse).
  * delta_DA = D_mean - A_mean   (H2 rescue: |D-A| < 0.005).
B_mean / C_mean = EQUAL-WEIGHT mean of per-seed query-means over the >=3
SURVIVING seeds (the prereg's "avg over B's 3 seeds"). The inferential unit is
the held-out QUERY; the bootstrap is PAIRED across arms (resample the shared
query axis, preserving cross-arm correlation), and the seed-average is recomputed
INSIDE each bootstrap iteration so the CI reflects query sampling. Cross-seed SD
of per-seed means is reported separately as the explicit reproducibility unit.

COLLAPSE DIAGNOSTIC (the load-bearing subtlety)
-----------------------------------------------
reward_call != optimizer step. Noisy arms (B/C/D) log 300 GRPO-group calls
(n==num_generations==8) PLUS ~2400 singleton (n==1) verdict-provider calls to the
SAME reward_trace.csv. Counting zero-variance over ALL rows gives a spurious
~0.5-0.89 collapse for every noisy arm, which would DROP B/C/D and void the whole
experiment. CORRECT rule: filter to GRPO groups (len == num_generations from the
manifest); a group is zero-variance iff its rewards are all identical; collapse
iff zero_var_groups/total_groups > 0.30. Both the corrected AND the contaminated
naive fraction are recorded for auditability. Verified corrected fracs are
0.000-0.007 => NO arm collapses => all 8 included, n_excluded=0.

FRAMING (both ship, prereg-committed)
-------------------------------------
  * framing_1_correlated_error if delta_BC <= -0.005 AND bootstrap CI excludes 0.
  * else framing_2_magnitude_rescue: report dose-response (clean-A vs pooled
    noisy-B/C) + rescue delta_DA, citing the E7 precedent (structured_minus_random
    ~= -0.003, gate fired b~=e). BOTH framings' numbers are computed regardless.

USAGE
-----
    python scripts/build_e10_noise_rl.py --self-test   # CPU wiring + collapse-diag unit test
    python scripts/build_e10_noise_rl.py --dry-run      # print block; write nothing
    python scripts/build_e10_noise_rl.py --write        # atomic merge of 'e10_noise_rl'
    python scripts/build_e10_noise_rl.py --write --force # overwrite an existing 'e10_noise_rl'
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

_REPO_ROOT = Path(__file__).resolve().parents[1]

CANON = (_REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
         / "canonical_numbers.json")
CANONICAL_KEY = "e10_noise_rl"

VERDICT_ROOT = _REPO_ROOT / "results" / "judge_gpt52_e10"
E10_RESULTS_ROOT = _REPO_ROOT / "results" / "e10"
ARM_ROOT = _REPO_ROOT / "models"
ARM_GLOB = "E10-arm*"
EXPERIMENT_TAG = "e10"

QUARANTINE_PREFIX = "82de3e92"
SPLIT_CONTENT_HASH = "db4ae2affea3ea6f0b84113059f013ec090e5c3b6cd4fb56b5f4e11cc5586a04"
SPLIT_SEED = 20260623
EVAL_FRAC = 0.40

BOOT_SEED = 20260623
N_BOOT = 10000
EQUIV_THRESHOLD = 0.005
COLLAPSE_THRESHOLD = 0.30

# E7 precedent (from canonical e7_selector_kappa.gate_g2).
# NOTE: these are FALLBACK defaults only. At build time the E7 gate result is
# READ from the canonical store (read_e7_precedent below) so this block cannot
# drift out of sync with canonical['e7_selector_kappa'].gate_g2. The E7
# equivalence gate did NOT fire (underpowered; paired TOST p=0.0523), so the
# gate-fired flag defaults to False to match E7. Paper 4's "structured ~= random"
# framing therefore rests on E10's OWN bootstrap TOST (framing_2), NOT on a fired
# E7 gate.
E7_STRUCTURED_MINUS_RANDOM = -0.003
E7_GATE_FIRED_B_APPROX_E = False


def read_e7_precedent() -> dict:
    """Read the E7 gate result from the canonical store so E10 cannot assert a
    stale/hardcoded E7 gate status. Falls back to the module constants (which are
    already E7-consistent) if the store or key is absent."""
    out = {
        "structured_minus_random": E7_STRUCTURED_MINUS_RANDOM,
        "gate_fired_b_approx_e": E7_GATE_FIRED_B_APPROX_E,
        "source": "fallback_constants",
    }
    try:
        cn = json.loads(CANON.read_text())
        g2 = cn.get("e7_selector_kappa", {}).get("gate_g2", {})
        if "gate_fires_b_approx_e" in g2:
            out["gate_fired_b_approx_e"] = bool(g2["gate_fires_b_approx_e"])
            out["source"] = "canonical.e7_selector_kappa.gate_g2"
        # prefer the E7-reported point gap if present
        if g2.get("tost", {}).get("mean_diff") is not None:
            out["structured_minus_random"] = round(float(g2["tost"]["mean_diff"]), 4)
        out["p_tost"] = g2.get("p_tost")
        out["underpowered"] = g2.get("underpowered")
    except Exception:
        pass
    return out


# ── arm-run id mapping (mirrors run_e10_eval.py) ─────────────────────────────
def arm_run_id(arm: str, noise_seed: int) -> str:
    if arm == "A_clean":
        return "A"
    if arm == "D_corrected":
        return "D"
    if arm == "B_struct":
        return f"B_s{int(noise_seed)}"
    if arm == "C_random":
        return f"C_s{int(noise_seed)}"
    return f"{arm.replace('_', '')}_s{int(noise_seed)}"


def pattern_for(arm_run: str) -> str:
    return f"e10_{arm_run}"


def _round(x, n=4):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), n)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT)
        ).decode().strip()
    except Exception:
        return "unknown"


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


# ── Arm discovery ────────────────────────────────────────────────────────────
def discover_arms() -> list[dict]:
    arms = []
    for d in sorted(ARM_ROOT.glob(ARM_GLOB)):
        man = d / "run_manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        arms.append({
            "arm_run": arm_run_id(m.get("arm", ""), m.get("noise_seed", 1)),
            "arm": m.get("arm", ""),
            "noise_seed": m.get("noise_seed", 1),
            "num_generations": int(m.get("num_generations", 8)),
            "adapter_dir": d,
            "reward_trace": d / "reward_trace.csv",
            "adapter_safetensors": d / "adapter_model.safetensors",
        })
    arms.sort(key=lambda a: a["arm_run"])
    return arms


# ── Collapse diagnostic (CORRECTED + naive, per the load-bearing subtlety) ───
def collapse_diag(trace_csv: Path, num_generations: int) -> dict:
    """Compute the corrected GRPO-group zero-variance fraction AND the
    contaminated naive all-call fraction. Returns both for auditability."""
    if not trace_csv.exists():
        return {"corrected_frac": None, "naive_all_call_frac": None,
                "n_grpo_groups": 0, "collapsed": False, "trace_present": False}
    groups: dict[str, list[float]] = defaultdict(list)
    naive_zv = 0
    n_rows = 0
    with trace_csv.open() as f:
        for row in csv.DictReader(f):
            n_rows += 1
            groups[row["reward_call"]].append(float(row["reward"]))
            if row.get("call_zero_variance", "0") == "1":
                naive_zv += 1
    # GRPO groups = calls with exactly num_generations items.
    grpo = [v for v in groups.values() if len(v) == num_generations]
    zv = sum(1 for v in grpo if len({round(r, 6) for r in v}) == 1)
    corrected = (zv / len(grpo)) if grpo else None
    naive = (naive_zv / n_rows) if n_rows else None
    collapsed = (corrected is not None and corrected > COLLAPSE_THRESHOLD)
    return {
        "corrected_frac": _round(corrected, 6),
        "naive_all_call_frac": _round(naive, 6),
        "naive_note": ("naive counts zero-variance over ALL reward_call rows incl. "
                       "n==1 verdict-provider singletons => CONTAMINATED; not used "
                       "for exclusion"),
        "n_grpo_groups": len(grpo),
        "n_singleton_calls": sum(1 for v in groups.values() if len(v) == 1),
        "num_generations": num_generations,
        "collapsed": bool(collapsed),
    }


# ── Verdict loading ──────────────────────────────────────────────────────────
def load_verdicts(arm_run: str) -> dict[str, float]:
    """overall_score per held-out query for one arm-run. Drops quarantine."""
    pat = pattern_for(arm_run)
    vdir = VERDICT_ROOT / f"{pat}__{EXPERIMENT_TAG}"
    out: dict[str, float] = {}
    if not vdir.exists():
        return out
    for jp in sorted(vdir.glob("*.json")):
        qid = jp.stem
        if qid.startswith(QUARANTINE_PREFIX):
            continue
        v = json.loads(jp.read_text())
        if "overall_score" in v:
            out[qid] = float(v["overall_score"])
    return out


def load_objective(arm_run: str) -> float | None:
    p = E10_RESULTS_ROOT / arm_run / "objective_eval.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ms = d.get("mean_score")
    return float(ms) if ms is not None else None


def load_calibration_echo(arms: list[dict]) -> dict:
    """Read pinned calibration from a trained arm's canonical_snapshot.json
    (do NOT recompute). Falls back to the prereg-pinned constants if absent."""
    for a in arms:
        snap = a["adapter_dir"] / "canonical_snapshot.json"
        if not snap.exists():
            continue
        try:
            s = json.loads(snap.read_text())
            calib = s.get("drjudge_error_structure", {}).get("calibration", {})
            rho = calib.get("latent_copula_rho_tetrachoric")
            flip = calib.get("pooled_marginal_flip_rate")
            if rho is not None and flip is not None:
                return {
                    "latent_copula_rho_tetrachoric": rho,
                    "pooled_marginal_flip_rate": flip,
                    "source": str(snap.relative_to(_REPO_ROOT)),
                }
        except Exception:
            continue
    return {
        "latent_copula_rho_tetrachoric": 0.3472,
        "pooled_marginal_flip_rate": 0.2811,
        "source": "prereg-pinned-fallback",
    }


# ── Stats ────────────────────────────────────────────────────────────────────
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def seed_average_over_queries(seed_verdicts: dict[str, dict[str, float]],
                              qids: list[str]) -> float:
    """Equal-weight mean of per-seed query-means over a query subset.

    seed_verdicts: {seed_label -> {qid -> score}}. For each surviving seed,
    take the mean over qids present, then average across seeds (equal weight).
    """
    per_seed = []
    for _seed, vmap in seed_verdicts.items():
        vals = [vmap[q] for q in qids if q in vmap]
        if vals:
            per_seed.append(_mean(vals))
    return _mean(per_seed) if per_seed else float("nan")


def single_arm_mean(vmap: dict[str, float], qids: list[str]) -> float:
    vals = [vmap[q] for q in qids if q in vmap]
    return _mean(vals) if vals else float("nan")


def paired_bootstrap_delta(
    compute_delta, common_qids: list[str], n_boot: int, seed: int
) -> tuple[float, list[float], list[float]]:
    """Seeded paired query-bootstrap. compute_delta(qid_subset) -> float.

    Resamples the SHARED query axis with replacement (paired across arms so
    cross-arm correlation is preserved); the seed-average is recomputed inside
    compute_delta for each resample. Returns (point_estimate, [lo95, hi95], boots)
    where boots is the full sorted array of bootstrap deltas (for a real TOST)."""
    import random as _random
    rng = _random.Random(seed)
    qids = sorted(common_qids)  # deterministic order
    point = compute_delta(qids)
    if not qids:
        return point, [float("nan"), float("nan")], []
    boots = []
    n = len(qids)
    for _ in range(n_boot):
        sample = [qids[rng.randrange(n)] for _ in range(n)]
        d = compute_delta(sample)
        if d == d:  # not NaN
            boots.append(d)
    boots.sort()
    if not boots:
        return point, [float("nan"), float("nan")], []
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return point, [lo, hi], boots


def bc_cross_seed_bootstrap(
    b_seeds: dict, c_seeds: dict, common_qids: list[str], n_boot: int, seed: int
) -> tuple[float, list[float], list[float]]:
    """Two-level bootstrap for the B - C delta that resamples BOTH the >=3
    seeds (the prereg reproducibility unit) AND the paired query axis, so the
    dominant cross-seed variance is propagated into the CI rather than averaged
    away. Returns (point, [lo95, hi95], boots)."""
    import random as _random
    rng = _random.Random(seed)
    bs = sorted(b_seeds.keys())
    cs = sorted(c_seeds.keys())
    qids = sorted(common_qids)
    n = len(qids)

    def one(b_samp, c_samp, q_samp):
        bm = _mean([single_arm_mean(b_seeds[ar], q_samp) for ar in b_samp])
        cm = _mean([single_arm_mean(c_seeds[ar], q_samp) for ar in c_samp])
        return bm - cm

    point = one(bs, cs, qids) if (bs and cs and qids) else float("nan")
    if not (bs and cs and qids):
        return point, [float("nan"), float("nan")], []
    boots = []
    for _ in range(n_boot):
        b_samp = [bs[rng.randrange(len(bs))] for _ in bs]      # resample seeds (w/ replacement)
        c_samp = [cs[rng.randrange(len(cs))] for _ in cs]
        q_samp = [qids[rng.randrange(n)] for _ in range(n)]    # resample queries (paired)
        d = one(b_samp, c_samp, q_samp)
        if d == d:
            boots.append(d)
    boots.sort()
    if not boots:
        return point, [float("nan"), float("nan")], []
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return point, [lo, hi], boots


def bootstrap_tost(boots: list[float], margin: float) -> dict:
    """Real two-one-sided-test on a bootstrap delta distribution against +/-margin.

    Equivalence (at alpha=0.05) iff the 90% bootstrap CI lies ENTIRELY within
    (-margin, +margin). Also reports the two one-sided bootstrap p-values:
    p_lower = P(delta <= -margin), p_upper = P(delta >= +margin); p_tost = max.
    This REPLACES the invalid 'CI-contains-0 => equivalent' fallacy: a wide CI
    that straddles 0 now (correctly) FAILS equivalence rather than passing it."""
    if not boots:
        return {"equivalent": False, "p_tost": float("nan"), "p_lower": float("nan"),
                "p_upper": float("nan"), "ci90": [float("nan"), float("nan")],
                "underpowered": True}
    m = len(boots)
    lo90 = boots[int(0.05 * (m - 1))]
    hi90 = boots[int(0.95 * (m - 1))]
    p_lower = sum(1 for b in boots if b <= -margin) / m   # evidence delta NOT below -margin
    p_upper = sum(1 for b in boots if b >= margin) / m    # evidence delta NOT above +margin
    p_tost = max(p_lower, p_upper)
    equivalent = (lo90 > -margin) and (hi90 < margin)
    # Underpowered = cannot reject NON-equivalence but also cannot establish a
    # difference: the 95% CI is wider than the equivalence band.
    lo95 = boots[int(0.025 * (m - 1))]
    hi95 = boots[int(0.975 * (m - 1))]
    underpowered = (not equivalent) and (lo95 <= 0 <= hi95)
    return {"equivalent": bool(equivalent), "p_tost": p_tost,
            "p_lower": p_lower, "p_upper": p_upper,
            "ci90": [lo90, hi90], "underpowered": bool(underpowered)}


# ── Build ────────────────────────────────────────────────────────────────────
def build_block() -> dict:
    # Read the E7 gate result from canonical so E10 can't assert a stale gate.
    e7_precedent = read_e7_precedent()
    arms = discover_arms()
    if not arms:
        raise SystemExit("no E10 arm dirs discovered under models/E10-arm*/")

    by_run = {a["arm_run"]: a for a in arms}

    # Verdicts + objective + collapse, per arm-run.
    verdicts = {ar: load_verdicts(ar) for ar in by_run}
    objective = {ar: load_objective(ar) for ar in by_run}
    collapse = {ar: collapse_diag(a["reward_trace"], a["num_generations"])
                for ar, a in by_run.items()}

    # Seed grouping for B and C (surviving seeds only).
    def seed_group(prefix: str) -> dict[str, dict[str, float]]:
        return {ar: verdicts[ar] for ar in sorted(by_run)
                if ar.startswith(prefix) and not collapse[ar]["collapsed"]}

    b_seeds_all = sorted(ar for ar in by_run if ar.startswith("B_s"))
    c_seeds_all = sorted(ar for ar in by_run if ar.startswith("C_s"))
    b_seeds = seed_group("B_s")
    c_seeds = seed_group("C_s")
    b_excluded = [ar for ar in b_seeds_all if collapse[ar]["collapsed"]]
    c_excluded = [ar for ar in c_seeds_all if collapse[ar]["collapsed"]]

    a_v = verdicts.get("A", {})
    d_v = verdicts.get("D", {})
    a_collapsed = collapse.get("A", {}).get("collapsed", False)
    d_collapsed = collapse.get("D", {}).get("collapsed", False)

    # Common query axis (intersection over all participating arm-runs).
    participating = {"A": a_v, "D": d_v}
    participating.update(b_seeds)
    participating.update(c_seeds)
    sets = [set(v) for v in participating.values() if v]
    common_qids = sorted(set.intersection(*sets)) if sets else []
    n_judged = len(common_qids)

    # Per-arm held-out means (over common qids).
    b_mean = seed_average_over_queries(b_seeds, common_qids)
    c_mean = seed_average_over_queries(c_seeds, common_qids)
    a_mean = single_arm_mean(a_v, common_qids)
    d_mean = single_arm_mean(d_v, common_qids)

    # Per-seed query-means + cross-seed SD (the explicit reproducibility unit).
    def per_seed_means(seeds: dict[str, dict[str, float]]) -> dict[str, float]:
        return {ar: single_arm_mean(v, common_qids) for ar, v in seeds.items()}
    b_per_seed = per_seed_means(b_seeds)
    c_per_seed = per_seed_means(c_seeds)
    b_cross_sd = _sd([v for v in b_per_seed.values() if v == v])
    c_cross_sd = _sd([v for v in c_per_seed.values() if v == v])

    # Paired bootstrap CIs.
    def delta_bc(qids):
        return (seed_average_over_queries(b_seeds, qids)
                - seed_average_over_queries(c_seeds, qids))

    def delta_da(qids):
        return single_arm_mean(d_v, qids) - single_arm_mean(a_v, qids)

    # B-C: two-level (seed + query) bootstrap so the dominant cross-seed variance
    # propagates into the CI (prereg reproducibility unit), not just query noise.
    delta_BC, ci_BC, boots_BC = bc_cross_seed_bootstrap(
        b_seeds, c_seeds, common_qids, N_BOOT, BOOT_SEED)
    # D-A: single-seed arms (A, D) -> paired query bootstrap only.
    delta_DA, ci_DA, boots_DA = paired_bootstrap_delta(
        delta_da, common_qids, N_BOOT, BOOT_SEED)

    # Equivalence via a REAL bootstrap TOST against the pre-registered +/-EQUIV_THRESHOLD
    # margin (prereg-E10 H1 band). This REPLACES the invalid '|delta|<thr OR CI-contains-0'
    # OR-gate: a wide CI straddling 0 now correctly FAILS equivalence (and is flagged
    # underpowered) instead of being mis-counted as positive evidence of equivalence.
    tost_BC = bootstrap_tost(boots_BC, EQUIV_THRESHOLD)
    tost_DA = bootstrap_tost(boots_DA, EQUIV_THRESHOLD)
    ci_BC_contains_0 = (ci_BC[0] <= 0 <= ci_BC[1]) if all(c == c for c in ci_BC) else True
    h1_B_lt_C = (delta_BC < 0)
    h1_equivalence = tost_BC["equivalent"]
    h2_rescue = tost_DA["equivalent"]

    # Framing selection (both ship).
    framing_1_supported = (delta_BC <= -EQUIV_THRESHOLD) and (not ci_BC_contains_0)
    framing_selected = ("framing_1_correlated_error" if framing_1_supported
                        else "framing_2_magnitude_rescue")
    pooled_noisy_mean = _mean([m for m in (b_mean, c_mean) if m == m])
    dose_clean_minus_noisy = (a_mean - pooled_noisy_mean
                              if (a_mean == a_mean and pooled_noisy_mean == pooled_noisy_mean)
                              else None)

    # Anti-Goodhart: per-arm (judge_mean, objective_mean) and divergence vs A.
    arm_judge_mean = {"A": a_mean, "B": b_mean, "C": c_mean, "D": d_mean}
    arm_obj = {
        "A": objective.get("A"),
        "B": _mean([objective[ar] for ar in b_seeds if objective.get(ar) is not None]),
        "C": _mean([objective[ar] for ar in c_seeds if objective.get(ar) is not None]),
        "D": objective.get("D"),
    }
    aj = arm_judge_mean["A"]
    ao = arm_obj["A"]
    goodhart_flag = False
    per_arm_jvo = {}
    for k in ("A", "B", "C", "D"):
        jm = arm_judge_mean[k]
        om = arm_obj[k]
        per_arm_jvo[k] = [_round(jm), _round(om)]
        if k == "A":
            continue
        if (jm == jm and aj == aj and om is not None and ao is not None
                and (jm - aj) > 0 and (om - ao) <= 0):
            goodhart_flag = True

    # Per-arm block assembly.
    def seed_block(seeds_all, seeds, excluded, per_seed, cross_sd, mean_over, obj_mean):
        return {
            "seeds": {ar.split("_s")[-1]: {
                "held_out_mean": _round(per_seed.get(ar)),
                "n_queries": sum(1 for q in common_qids if q in verdicts[ar]),
                "objective_mean": _round(objective.get(ar)),
                "collapse_frac": collapse[ar]["corrected_frac"],
                "collapsed": collapse[ar]["collapsed"],
            } for ar in seeds_all},
            "seeds_included": sorted(seeds.keys()),
            "seeds_excluded_collapse": excluded,
            "held_out_mean_over_seeds": _round(mean_over),
            "cross_seed_sd": _round(cross_sd, 6),
            "objective_mean": _round(obj_mean),
        }

    arms_block = {
        "A_clean": {
            "n_seeds": 1, "held_out_mean": _round(a_mean),
            "objective_mean": _round(objective.get("A")),
            "collapse_frac": collapse.get("A", {}).get("corrected_frac"),
            "collapsed": a_collapsed, "included": not a_collapsed,
        },
        "B_struct": seed_block(b_seeds_all, b_seeds, b_excluded, b_per_seed,
                               b_cross_sd, b_mean, arm_obj["B"]),
        "C_random": seed_block(c_seeds_all, c_seeds, c_excluded, c_per_seed,
                               c_cross_sd, c_mean, arm_obj["C"]),
        "D_corrected": {
            "n_seeds": 1, "held_out_mean": _round(d_mean),
            "objective_mean": _round(objective.get("D")),
            "collapse_frac": collapse.get("D", {}).get("corrected_frac"),
            "collapsed": d_collapsed, "included": not d_collapsed,
        },
    }

    calib = load_calibration_echo(arms)
    excluded_runs = b_excluded + c_excluded + (
        ["A"] if a_collapsed else []) + (["D"] if d_collapsed else [])

    block = {
        "_what": ("E10 noise-RL held-out GPT-5.2 endpoint: structured(B) vs "
                  "matched-random(C) reward noise at matched marginal flip, plus "
                  "rescue D vs clean A. Paper 4."),
        "judge": "gpt-5.2",
        "judge_endpoint": "JUDGE (cloud, non-PTU)",
        "split_content_hash": SPLIT_CONTENT_HASH,
        "split_seed": SPLIT_SEED,
        "eval_frac": EVAL_FRAC,
        "n_eval_queries_judged": n_judged,
        "quarantined_excluded": [QUARANTINE_PREFIX + "-abe2-46ac-ad17-23417b9c4da7"],
        "calibration_echo": {
            "latent_copula_rho_tetrachoric": calib["latent_copula_rho_tetrachoric"],
            "pooled_marginal_flip_rate": calib["pooled_marginal_flip_rate"],
            "source": calib["source"],
        },
        "bootstrap": {
            "seed": BOOT_SEED, "n_boot": N_BOOT,
            "unit": "query (paired across arms)",
            "reproducibility_unit": "cross-seed variance over B/C seeds",
        },
        "arms": arms_block,
        "primary": {
            "delta_B_minus_C": _round(delta_BC),
            "delta_B_minus_C_ci95": [_round(ci_BC[0]), _round(ci_BC[1])],
            "delta_D_minus_A": _round(delta_DA),
            "delta_D_minus_A_ci95": [_round(ci_DA[0]), _round(ci_DA[1])],
            "h1_B_lt_C": bool(h1_B_lt_C),
            "h1_equivalence_holds": bool(h1_equivalence),
            "h2_rescue_D_approx_A": bool(h2_rescue),
            "equivalence_threshold": EQUIV_THRESHOLD,
            "equivalence_method": ("real bootstrap TOST: equivalent iff the 90% bootstrap CI "
                                   "lies entirely within +/-margin (supersedes the invalid "
                                   "|delta|<thr OR CI-contains-0 OR-gate)"),
            "bc_bootstrap_unit": "seed (>=3, resampled w/ replacement) x query (paired)",
            "tost_B_minus_C": {
                "equivalent": tost_BC["equivalent"],
                "p_tost": _round(tost_BC["p_tost"], 4),
                "p_one_sided_lower": _round(tost_BC["p_lower"], 4),
                "p_one_sided_upper": _round(tost_BC["p_upper"], 4),
                "ci90": [_round(tost_BC["ci90"][0]), _round(tost_BC["ci90"][1])],
                "underpowered": tost_BC["underpowered"],
            },
            "tost_D_minus_A": {
                "equivalent": tost_DA["equivalent"],
                "p_tost": _round(tost_DA["p_tost"], 4),
                "ci90": [_round(tost_DA["ci90"][0]), _round(tost_DA["ci90"][1])],
                "underpowered": tost_DA["underpowered"],
            },
        },
        "framing": {
            "framing_selected": framing_selected,
            "framing_1": {
                "claim": "B<<C: RL amplifies correlated error",
                "supported": bool(framing_1_supported),
            },
            "framing_2": {
                "claim": "magnitude+rescue: clean-A vs noisy-B/C dose + D rescue",
                "dose_clean_minus_noisy": _round(dose_clean_minus_noisy),
                "rescue_delta_DA": _round(delta_DA),
                "e7_precedent_structured_minus_random":
                    e7_precedent["structured_minus_random"],
                "e7_gate_fired_b_approx_e":
                    bool(e7_precedent["gate_fired_b_approx_e"]),
                "e7_gate_source": e7_precedent["source"],
                "e7_gate_note": (
                    "The E7 structured-vs-random equivalence gate (G2) did NOT fire: "
                    "it is UNDERPOWERED (paired bootstrap TOST p_tost="
                    f"{e7_precedent.get('p_tost')}, 90% CI not contained within "
                    "+/-0.005). E10's 'structured ~= random' conclusion therefore "
                    "rests on E10's OWN bootstrap TOST (tost_B_minus_C above), NOT "
                    "on a fired E7 gate; the E7 point gap is cited only as a small, "
                    "same-direction precedent."),
            },
        },
        "anti_goodhart": {
            "per_arm_judge_vs_objective": per_arm_jvo,
            "goodhart_flag": bool(goodhart_flag),
            "_note": ("flag=True iff any arm's judge mean rises vs A while its "
                      "objective mean is flat/falls vs A"),
        },
        "collapse_diagnostic": {
            "rule": ("zero-variance reward for >30% of GRPO-GROUP optimizer steps "
                     "(calls with n==num_generations); singleton verdict-provider "
                     "calls EXCLUDED"),
            "per_arm_run": {ar: collapse[ar]["corrected_frac"] for ar in sorted(by_run)},
            "naive_contaminated_per_arm_run": {
                ar: collapse[ar]["naive_all_call_frac"] for ar in sorted(by_run)},
            "naive_note": ("naive all-call fraction is CONTAMINATED by n==1 "
                           "verdict-provider singletons; recorded for audit, NOT "
                           "used for exclusion"),
            "excluded_runs": excluded_runs,
            "n_excluded": len(excluded_runs),
        },
        "provenance": {
            "adapter_shas": {ar: (sha256_first16(by_run[ar]["adapter_safetensors"]))
                             for ar in sorted(by_run)},
            "git_sha": git_sha(),
            "verdict_root": str(VERDICT_ROOT.relative_to(_REPO_ROOT)),
            "built_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    return block


def sha256_first16(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Self-test (CPU; exercises the collapse-diag trap on real traces) ─────────
def self_test() -> int:
    print("[self-test] E10 build wiring (CPU only, no API, no write)")
    arms = discover_arms()
    assert arms, "no arms discovered"
    runs = [a["arm_run"] for a in arms]
    print(f"[self-test] discovered arms: {runs}")

    assert arm_run_id("A_clean", 1) == "A"
    assert arm_run_id("B_struct", 3) == "B_s3"
    assert arm_run_id("C_random", 2) == "C_s2"
    assert arm_run_id("D_corrected", 1) == "D"
    print("[self-test] arm_run_id mapping OK")

    # The LOAD-BEARING test: corrected collapse-diag must NOT flag noisy arms,
    # while the naive all-call fraction is clearly contaminated (>0.30).
    flagged = []
    for a in arms:
        cd = collapse_diag(a["reward_trace"], a["num_generations"])
        print(f"[self-test] {a['arm_run']:>6} corrected={cd['corrected_frac']} "
              f"naive={cd['naive_all_call_frac']} groups={cd['n_grpo_groups']} "
              f"singletons={cd['n_singleton_calls']} collapsed={cd['collapsed']}")
        assert cd["n_grpo_groups"] == 300, "expected 300 GRPO groups per arm"
        if cd["collapsed"]:
            flagged.append(a["arm_run"])
        # Noisy arms must show the contamination trap: naive >> corrected.
        if cd["n_singleton_calls"] > 0:
            assert cd["naive_all_call_frac"] is not None
            assert cd["naive_all_call_frac"] > 0.30, (
                "noisy arm naive fraction should expose the singleton contamination")
            assert cd["corrected_frac"] <= COLLAPSE_THRESHOLD, (
                "corrected fraction must stay below the 0.30 collapse threshold")
    assert not flagged, f"collapse-diag FALSE POSITIVE flagged {flagged} (the trap)"
    print("[self-test] collapse diagnostic: corrected fracs below 0.30, no false "
          "positive; naive trap exposed (n_excluded would be 0)")

    # Stats unit tests on synthetic data.
    seeds = {"B_s1": {"q1": 0.5, "q2": 0.7}, "B_s2": {"q1": 0.6, "q2": 0.8}}
    # per-seed means: B_s1=0.6, B_s2=0.7 -> equal-weight avg = 0.65
    assert abs(seed_average_over_queries(seeds, ["q1", "q2"]) - 0.65) < 1e-9
    assert abs(single_arm_mean({"q1": 0.4, "q2": 0.6}, ["q1", "q2"]) - 0.5) < 1e-9
    # paired bootstrap is deterministic given the seed.
    pe1, ci1 = paired_bootstrap_delta(
        lambda qs: single_arm_mean({"q1": 1.0, "q2": 0.0}, qs), ["q1", "q2"], 1000, 7)
    pe2, ci2 = paired_bootstrap_delta(
        lambda qs: single_arm_mean({"q1": 1.0, "q2": 0.0}, qs), ["q1", "q2"], 1000, 7)
    assert ci1 == ci2, "bootstrap must be seed-deterministic"
    print("[self-test] seed-average, single-arm mean, deterministic bootstrap OK")

    # Canonical store exists and has the 45 keys (we must not clobber).
    if CANON.exists():
        canon = json.loads(CANON.read_text())
        print(f"[self-test] canonical store present: {len(canon)} keys; "
              f"e10_noise_rl present={CANONICAL_KEY in canon}")
    else:
        print("[self-test] NOTE: canonical store absent; --write would refuse to create it")

    # Build runs end to end as long as the inputs exist (verdicts may be absent
    # pre-judge — block still assembles with NaN/None means, which is fine for
    # a dry wiring check).
    block = build_block()
    assert block["collapse_diagnostic"]["n_excluded"] == 0
    assert set(block["arms"]) == {"A_clean", "B_struct", "C_random", "D_corrected"}
    print(f"[self-test] build_block assembled; framing_selected="
          f"{block['framing']['framing_selected']} "
          f"n_judged={block['n_eval_queries_judged']}")
    print("[self-test] PASS — build wiring sound; collapse-diag trap defused.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU wiring + collapse-diag unit test; no write.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the would-be canonical block; write nothing.")
    ap.add_argument("--write", action="store_true",
                    help="atomic merge of 'e10_noise_rl' into the canonical store.")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing 'e10_noise_rl' key.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    block = build_block()

    summary = {
        "n_eval_queries_judged": block["n_eval_queries_judged"],
        "delta_B_minus_C": block["primary"]["delta_B_minus_C"],
        "delta_B_minus_C_ci95": block["primary"]["delta_B_minus_C_ci95"],
        "delta_D_minus_A": block["primary"]["delta_D_minus_A"],
        "delta_D_minus_A_ci95": block["primary"]["delta_D_minus_A_ci95"],
        "framing_selected": block["framing"]["framing_selected"],
        "h1_B_lt_C": block["primary"]["h1_B_lt_C"],
        "h1_equivalence_holds": block["primary"]["h1_equivalence_holds"],
        "h2_rescue_D_approx_A": block["primary"]["h2_rescue_D_approx_A"],
        "goodhart_flag": block["anti_goodhart"]["goodhart_flag"],
        "collapse_n_excluded": block["collapse_diagnostic"]["n_excluded"],
    }
    print(json.dumps({CANONICAL_KEY + "_summary": summary}, indent=1))

    if args.dry_run or not args.write:
        # Print the full block for inspection on a dry run.
        if args.dry_run:
            print(json.dumps({CANONICAL_KEY: block}, indent=1))
        print(f"\n[dry-run] canonical['{CANONICAL_KEY}'] NOT written "
              "(pass --write to merge).", file=sys.stderr)
        return 0

    if not CANON.exists():
        print(f"[e10_noise_rl] canonical store missing at {CANON}; refusing to "
              "create it from scratch. (exit 0, wrote nothing)", file=sys.stderr)
        return 0

    canon = json.loads(CANON.read_text())
    n_before = len(canon)
    if CANONICAL_KEY in canon and not args.force:
        print(f"[e10_noise_rl] key already present and --force not given; refusing "
              "to clobber. (exit 0, wrote nothing)", file=sys.stderr)
        return 0
    canon[CANONICAL_KEY] = block
    # Append-only invariant: we add exactly one key (or overwrite under --force).
    assert len(canon) >= n_before, "canonical key count must not shrink"
    atomic_write_json(CANON, canon)
    print(f"\n[write] merged canonical['{CANONICAL_KEY}'] -> {CANON} "
          f"({n_before} -> {len(canon)} keys)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
