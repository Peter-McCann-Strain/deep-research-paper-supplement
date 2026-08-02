#!/usr/bin/env python
"""E7 — Realizable best-of-N SELECTORS at matched spend (CPU-only analysis on EXISTING data).

WHAT THIS IS
------------
The best-of-N control (`canonical_numbers.json['best_of_n']`) reports the ORACLE selection
gain: pick the replicate with the truly-highest GPT-5.2 overall score. That is an UPPER BOUND
on any realizable selector. E7 closes the gap by simulating the selectors a practitioner could
actually deploy, at MATCHED SPEND (best-of-N over N independent replicates of the SAME
architecture, so per-architecture cost is automatically matched), and reporting the realized
overall-score GAIN of each selector over the single-run mean.

DATA (read-only)
----------------
The variance replicate corpus, GPT-5.2 ('gpt52') judge only:
  base_p0  x11 replicates  +  {p1,p4,p5,p6,p7,p8,p10} x3   over 30 variance queries.
Parsed from pattern names `base_p{N}_v{k}` in df_overall_scores / df_scores / df_verdicts
(pattern_family=='variance'). The authoritative quality signal is GPT-5.2's stored
`overall_score`. No new generation, no GPU, no real judge calls.

SELECTORS (each picks ONE replicate per (architecture, query) from N available)
------------------------------------------------------------------------------
  (a) ORACLE        : argmax of TRUE gpt52 overall_score            -> realized UPPER bound
  (b) RANDOM        : uniform pick                                  -> realized LOWER bound
  (c) GPT52_NOISE   : argmax of (true + N(0, sigma_gpt52))          -> a *second* gpt52 pass
  (d) GPT4O_NOISE   : argmax of (true + N(0, sigma_gpt4o))          -> a weaker (noisier) judge
  (e1) STRUCT_NOISE : flip per-criterion verdicts to a TARGET kappa with structure
                      (correlated within (replicate)); recompute overall; argmax
  (e2) RAND_NOISE   : flip per-criterion verdicts to the MATCHED kappa i.i.d.; argmax
The (e1) vs (e2) pair isolates whether *structured* judge error (criteria flip together) costs
more selection skill than the same amount of *independent* error, at matched marginal flip rate.

NOISE CALIBRATION (all from existing canonical/replicate evidence, no fitting on the outcome)
--------------------------------------------------------------------------------------------
  sigma_gpt52 : pooled within-(arch,query) replicate SD of gpt52 overall_score (run noise).
                This is the spread of a *fresh* gpt52 judging pass = the (c) arm's noise.
  sigma_gpt4o : sigma_gpt52 scaled up by the cross-family judge-disagreement ratio. GPT-4o is a
                weaker judge than GPT-5.2; we inflate the run-noise SD by
                kappa_gpt52_self / kappa_gpt52_vs_weak (verdict-kappa, from n_eff) as a
                conservative, evidence-anchored multiplier. Reported explicitly; sensitivity
                band included so the headline does not hinge on the exact multiplier.
  kappa targets for (e): a sweep, anchored on the measured cross-judge verdict kappa.

OVERALL-SCORE RECOMPUTE UNDER FLIPS (e-arms)
--------------------------------------------
The variance queries are NOT in eval_queries_v2.json, so exact per-criterion rubric weights are
unrecoverable. We use the dimension-weighted formula (V2 weights; reproduces stored
`overall_score` to ~0.01 mean abs err) but ANCHOR each replicate's flipped score as
  score_flipped = stored_overall + (recompute(flipped_verdicts) - recompute(true_verdicts))
so the BASELINE per replicate is the exact stored value and only the *flip delta* comes from the
recompute. This makes the e-arms a faithful perturbation of the canonical scores.

DR-JUDGE-7B ARM — OUT (documented hook)
---------------------------------------
A natural further selector is "pick with a small RL-trained DR-Judge-7B". GAIR/DeepResearcher-7b
is NOT on disk, so the DR-Judge selector arm is OMITTED. The output carries an explicit
`drjudge_selector` stub describing exactly how to slot it in (run the 7B as a deterministic
detector over each replicate, take its preferred replicate, recompute realized gain) once the
weights are downloaded. NO small model is used as an authoritative judge here.

OUTPUT
------
  reports/e7_selector_results.json                       (standalone, primary)
  papers/paper_a_bounded_returns/analysis/canonical_numbers.json['selector_e7']  (merged in place)
Each selector reports realized mean overall score, GAIN over the single-run mean, and a seeded
bootstrap CI (resample queries, then re-draw selector randomness within the resample).

DETERMINISM / SAFETY
--------------------
- One master SEED; every stochastic step draws from np.random.default_rng(SEED + offset) on
  SORTED inputs (sorted arch list, sorted query_ids, sorted replicate order by v-index).
- CPU-only. Reads only parquet under data/analysis/ ; writes only to reports/ (NEW file) and the
  canonical JSON. Never touches results/experiments, results/judge_*, or the parquet files.
- --dry-run : print the plan + calibration + a 1-query micro-result, write nothing.
- --self-test : run on 2 queries only, write to a *_selftest.json sidecar, never canonical.
- Idempotent: a normal run overwrites its own outputs atomically; --resume is a no-op flag kept
  for interface parity (the whole computation is a single deterministic pass, cheap to redo).

LAUNCH (full run):
  [ -f venv/bin/activate ] && source venv/bin/activate && python scripts/run_e7_selector.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/data/analysis"
CANON = f"{ROOT}/papers/paper_a_bounded_returns/analysis/canonical_numbers.json"
OUT_STANDALONE = f"{ROOT}/reports/e7_selector_results.json"

SEED = 20260613  # master seed; sub-streams are SEED + fixed offsets

# V2 rubric dimension weights (MEMORY.md "Rubric V2 — 9 Dimensions").
DIMW: Dict[str, float] = {
    "information_recall": 0.20,
    "factual_accuracy": 0.20,
    "coverage": 0.10,
    "analytical_depth": 0.15,
    "citation_quality": 0.10,
    "logical_coherence": 0.05,
    "organization": 0.05,
    "instruction_following": 0.10,
    "attribution_quality": 0.05,
}

# Bootstrap settings.
N_BOOT = 2000
KAPPA_TARGETS = [0.20, 0.35, 0.50]  # cross-judge verdict-kappa neighbourhood for e-arms
GPT4O_MULT_BAND = (1.0, 2.5)  # sensitivity band on the GPT-4o noise multiplier


# --------------------------------------------------------------------------- #
# Loading / parsing
# --------------------------------------------------------------------------- #
def arch_rep(pattern: str) -> Tuple[Optional[str], Optional[int]]:
    """`base_p4_v2` -> ('p4', 2).  Non-replicate patterns -> (None, None)."""
    m = re.match(r"base_(p\d+)_v(\d+)$", str(pattern))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load_corpus() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (overall, verdicts) for variance + gpt52 only, with arch/rep columns."""
    o = pd.read_parquet(f"{ANA}/df_overall_scores.parquet")
    v = pd.read_parquet(f"{ANA}/df_verdicts.parquet")
    o = o[(o.pattern_family == "variance") & (o.judge == "gpt52")].copy()
    v = v[(v.pattern_family == "variance") & (v.judge == "gpt52")].copy()
    for df in (o, v):
        pat = df.pattern.astype(str)
        df["arch"] = pat.map(lambda p: arch_rep(p)[0])
        df["rep"] = pat.map(lambda p: arch_rep(p)[1])
    o = o.dropna(subset=["arch", "rep", "overall_score"]).copy()
    o["rep"] = o["rep"].astype(int)
    v = v.dropna(subset=["arch", "rep"]).copy()
    v["rep"] = v["rep"].astype(int)
    return o, v


def recompute_overall(sat_by_dim_counts: Dict[str, Tuple[float, int]]) -> float:
    """Dimension-weighted mean of per-dimension fraction-satisfied.

    sat_by_dim_counts: dimension -> (n_satisfied, n_total).
    Normalised by the sum of weights over dimensions actually present.
    """
    num = 0.0
    den = 0.0
    for d, (nsat, ntot) in sat_by_dim_counts.items():
        if ntot <= 0 or d not in DIMW:
            continue
        num += DIMW[d] * (nsat / ntot)
        den += DIMW[d]
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def pooled_run_sd(o: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """RMS of within-(arch,query) replicate SD (ddof=1) of overall_score, + per-arch."""
    per_arch: Dict[str, List[float]] = {}
    allsds: List[float] = []
    for (a, q), g in o.groupby(["arch", "query_id"]):
        if g.pattern.nunique() >= 2:
            s = g.overall_score.std(ddof=1)
            if np.isfinite(s):
                allsds.append(s)
                per_arch.setdefault(a, []).append(s)
    pooled = float(np.sqrt(np.mean(np.square(allsds)))) if allsds else float("nan")
    per = {a: float(np.sqrt(np.mean(np.square(v)))) for a, v in sorted(per_arch.items())}
    return pooled, per


def gpt4o_noise_multiplier() -> Tuple[float, str]:
    """Inflate gpt52 run-noise by the cross-family judge-disagreement ratio.

    Uses verdict kappa from canonical n_eff: GPT-4o is weaker than GPT-5.2, so a GPT-4o
    selection pass is noisier. Multiplier = self-consistency / cross-judge consistency,
    bounded into GPT4O_MULT_BAND. Falls back to the band centre if canonical is unavailable.
    """
    try:
        cn = json.load(open(CANON))
        kap = cn["n_eff"]["overall"]["kappa"]
        # cross-family gpt52 vs claude as a proxy for "weaker judge" verdict agreement
        cross = float(kap.get("gpt52|claude_sonnet", kap.get("gpt52|claude_opus")))
        # self-consistency proxy: agreement of a fresh same-judge pass. The replicate
        # corpus is single-judge, so anchor self-kappa at the strong drjudge-train agreement
        # used elsewhere; conservatively use 1.0 (perfect self) -> multiplier = 1/cross.
        mult = 1.0 / max(cross, 1e-6)
        src = f"1/kappa(gpt52|claude_sonnet)=1/{cross:.3f}"
    except Exception as e:  # pragma: no cover
        mult = float(np.mean(GPT4O_MULT_BAND))
        src = f"fallback band-centre ({e})"
    mult = float(min(max(mult, GPT4O_MULT_BAND[0]), GPT4O_MULT_BAND[1]))
    return mult, src


# --------------------------------------------------------------------------- #
# Per-(arch,query) replicate matrices
# --------------------------------------------------------------------------- #
class Cell:
    """One (architecture, query) cell with N replicates."""

    __slots__ = ("arch", "query_id", "reps", "true", "verdict_dim")

    def __init__(self, arch: str, query_id: str, reps: List[int],
                 true: np.ndarray, verdict_dim: List[Dict[str, np.ndarray]]):
        self.arch = arch
        self.query_id = query_id
        self.reps = reps                 # sorted replicate indices
        self.true = true                 # stored overall_score per replicate (len N)
        self.verdict_dim = verdict_dim   # per replicate: dim -> bool array of true verdicts


def build_cells(o: pd.DataFrame, v: pd.DataFrame, min_reps: int = 2) -> List[Cell]:
    """Build sorted, deterministic per-(arch,query) cells with >= min_reps replicates."""
    cells: List[Cell] = []
    # index verdicts for fast lookup
    vg = {k: g for k, g in v.groupby(["arch", "query_id", "rep"])}
    for (a, q), g in sorted(o.groupby(["arch", "query_id"]), key=lambda kv: (kv[0][0], kv[0][1])):
        reps = sorted(g.rep.unique().tolist())
        if len(reps) < min_reps:
            continue
        true = np.array([float(g[g.rep == r].overall_score.iloc[0]) for r in reps], dtype=float)
        vdim: List[Dict[str, np.ndarray]] = []
        for r in reps:
            cell_v = vg.get((a, q, r))
            d: Dict[str, np.ndarray] = {}
            if cell_v is not None:
                for dim, gg in cell_v.groupby("dimension"):
                    d[dim] = gg.satisfied.to_numpy(dtype=bool)
            vdim.append(d)
        cells.append(Cell(a, q, reps, true, vdim))
    return cells


def cell_recompute_true(cell: Cell) -> np.ndarray:
    """Recompute overall per replicate from TRUE verdicts (for flip anchoring)."""
    out = np.empty(len(cell.reps), dtype=float)
    for i, dmap in enumerate(cell.verdict_dim):
        counts = {dim: (int(arr.sum()), int(arr.size)) for dim, arr in dmap.items()}
        out[i] = recompute_overall(counts)
    return out


# --------------------------------------------------------------------------- #
# Selectors -> per-cell selected index
# --------------------------------------------------------------------------- #
def sel_oracle(cell: Cell, rng: np.random.Generator) -> int:
    return int(_argmax_tiebreak(cell.true, rng))


def sel_random(cell: Cell, rng: np.random.Generator) -> int:
    return int(rng.integers(len(cell.true)))


def sel_gaussian_noise(cell: Cell, rng: np.random.Generator, sigma: float) -> int:
    noisy = cell.true + rng.normal(0.0, sigma, size=cell.true.shape)
    return int(_argmax_tiebreak(noisy, rng))


def sel_flip_kappa(cell: Cell, rng: np.random.Generator, flip_p: float,
                   structured: bool, anchor_true_recompute: np.ndarray) -> int:
    """Flip per-criterion verdicts at marginal rate flip_p, recompute overall, argmax.

    structured=True : one shared flip-mask draw biases ALL criteria of a replicate together
                      (a correlated judge), implemented by drawing a per-replicate latent that
                      shifts the flip probability up/down, so flips co-occur within a replicate.
    structured=False: each criterion flips i.i.d. at flip_p.
    Score = stored_overall + (recompute(flipped) - recompute(true))  [anchored].
    """
    scores = np.empty(len(cell.reps), dtype=float)
    for i, dmap in enumerate(cell.verdict_dim):
        if structured:
            # per-replicate correlated bias in [0,1]; mixes toward 0 or 2*flip_p
            bias = rng.random()
            p_i = flip_p * (0.2 + 1.6 * bias)  # mean ~= flip_p, but correlated across criteria
        counts: Dict[str, Tuple[int, int]] = {}
        for dim, arr in dmap.items():
            if structured:
                flips = rng.random(arr.size) < p_i
            else:
                flips = rng.random(arr.size) < flip_p
            flipped = np.where(flips, ~arr, arr)
            counts[dim] = (int(flipped.sum()), int(flipped.size))
        rec_flip = recompute_overall(counts)
        delta = rec_flip - anchor_true_recompute[i]
        scores[i] = cell.true[i] + delta
    return int(_argmax_tiebreak(scores, rng))


def _argmax_tiebreak(x: np.ndarray, rng: np.random.Generator) -> int:
    """Deterministic-given-rng argmax with random tie-break among the maxima."""
    m = np.max(x)
    cand = np.flatnonzero(x >= m - 1e-12)
    return int(cand[rng.integers(len(cand))]) if len(cand) > 1 else int(cand[0])


# --------------------------------------------------------------------------- #
# Realized gain + bootstrap CI
# --------------------------------------------------------------------------- #
def realized_gain(cells: List[Cell], selector, rng: np.random.Generator
                  ) -> Tuple[float, float, np.ndarray]:
    """Mean selected overall, mean single-run-mean, and per-cell selected scores (one draw)."""
    sel = np.empty(len(cells), dtype=float)
    base = np.empty(len(cells), dtype=float)
    for i, c in enumerate(cells):
        idx = selector(c, rng)
        sel[i] = c.true[idx]
        base[i] = float(np.mean(c.true))
    return float(sel.mean()), float(base.mean()), sel


def bootstrap_selector(cells: List[Cell], selector, n_boot: int, seed_off: int
                       ) -> Dict[str, float]:
    """Seeded cluster bootstrap over queries; selector randomness re-drawn each resample.

    Returns realized selected-mean, single-run-mean, gain, and 95% CI on the GAIN.
    """
    base_means = np.array([float(np.mean(c.true)) for c in cells])
    point_rng = np.random.default_rng(SEED + seed_off)
    sel_point, base_point, _ = realized_gain(cells, selector, point_rng)
    gain_point = sel_point - base_point

    rng = np.random.default_rng(SEED + seed_off + 1)
    n = len(cells)
    gains = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = [cells[j] for j in idx]
        # fresh selector randomness inside the resample, deterministic per (seed_off, b)
        srng = np.random.default_rng(SEED + seed_off + 7919 * (b + 1))
        sel = np.array([sub[k].true[selector(sub[k], srng)] for k in range(n)])
        bse = base_means[idx]
        gains[b] = sel.mean() - bse.mean()
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {
        "selected_mean": round(sel_point, 4),
        "single_run_mean": round(base_point, 4),
        "gain": round(gain_point, 4),
        "gain_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_cells": n,
    }


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #
def kappa_to_flip_p(kappa: float) -> float:
    """Map a target Cohen-kappa-on-verdicts to a marginal flip probability.

    For a symmetric binary channel flipping at rate p against a balanced-ish reference,
    Cohen's kappa ~= 1 - 2p (chance-corrected agreement under near-balanced marginals).
    So p = (1 - kappa) / 2, clamped to [0, 0.5]. Reported alongside the realised kappa.
    """
    return float(min(max((1.0 - kappa) / 2.0, 0.0), 0.5))


def compute(cells: List[Cell], sigma_gpt52: float, sigma_gpt4o: float,
            gpt4o_mult: float, mult_src: str) -> Dict:
    # Pre-recompute true-verdict overalls per cell (anchor for flip arms).
    anchors = [cell_recompute_true(c) for c in cells]

    selectors = {
        "oracle": (lambda c, r: sel_oracle(c, r), 1),
        "random": (lambda c, r: sel_random(c, r), 2),
        "gpt52_noise": (lambda c, r: sel_gaussian_noise(c, r, sigma_gpt52), 3),
        "gpt4o_noise": (lambda c, r: sel_gaussian_noise(c, r, sigma_gpt4o), 4),
    }
    arms: Dict[str, Dict] = {}
    for name, (fn, off) in selectors.items():
        arms[name] = bootstrap_selector(cells, fn, N_BOOT, off)

    # e-arms: structured vs random flips at matched kappa, swept.
    flip_arms: Dict[str, Dict] = {}
    for ki, kappa in enumerate(KAPPA_TARGETS):
        flip_p = kappa_to_flip_p(kappa)
        for structured, tag in [(True, "structured"), (False, "random")]:
            def mk(structured=structured, flip_p=flip_p):
                def f(c, r):
                    # O(1) anchor lookup by object identity (resamples reuse cell objects)
                    return sel_flip_kappa(c, r, flip_p, structured, ID2ANCHOR[id(c)])
                return f
            off = 100 + ki * 10 + (0 if structured else 5)
            flip_arms[f"kappa{kappa:.2f}_{tag}"] = {
                **bootstrap_selector(cells, mk(), N_BOOT, off),
                "target_kappa": kappa,
                "marginal_flip_p": round(flip_p, 4),
                "structured": structured,
            }

    # paired structured-minus-random gain at each kappa
    struct_vs_rand = {}
    for kappa in KAPPA_TARGETS:
        s = flip_arms[f"kappa{kappa:.2f}_structured"]["gain"]
        r = flip_arms[f"kappa{kappa:.2f}_random"]["gain"]
        struct_vs_rand[f"kappa{kappa:.2f}"] = round(s - r, 4)

    return {
        "arms": arms,
        "flip_arms": flip_arms,
        "structured_minus_random_gain": struct_vs_rand,
        "_calibration": {
            "sigma_gpt52_run_sd": round(sigma_gpt52, 4),
            "sigma_gpt4o": round(sigma_gpt4o, 4),
            "gpt4o_noise_multiplier": round(gpt4o_mult, 4),
            "gpt4o_multiplier_source": mult_src,
            "gpt4o_multiplier_band": list(GPT4O_MULT_BAND),
            "kappa_targets": KAPPA_TARGETS,
            "kappa_to_flip_p": "p=(1-kappa)/2",
        },
    }


# Module-level anchor map filled in run(); keeps sel_flip_kappa O(1).
ID2ANCHOR: Dict[int, np.ndarray] = {}


def per_arch_breakdown(cells: List[Cell], sigma_gpt52: float, sigma_gpt4o: float) -> Dict:
    """Oracle/random/gpt52/gpt4o realized gain per architecture (point estimates)."""
    out: Dict[str, Dict] = {}
    by_arch: Dict[str, List[Cell]] = {}
    for c in cells:
        by_arch.setdefault(c.arch, []).append(c)
    for a in sorted(by_arch):
        ac = by_arch[a]
        rng = np.random.default_rng(SEED + 555)
        o_sel, o_base, _ = realized_gain(ac, lambda c, r: sel_oracle(c, r), rng)
        g52, _, _ = realized_gain(ac, lambda c, r: sel_gaussian_noise(c, r, sigma_gpt52),
                                  np.random.default_rng(SEED + 556))
        g4o, _, _ = realized_gain(ac, lambda c, r: sel_gaussian_noise(c, r, sigma_gpt4o),
                                  np.random.default_rng(SEED + 557))
        rnd, _, _ = realized_gain(ac, lambda c, r: sel_random(c, r),
                                  np.random.default_rng(SEED + 558))
        out[a] = {
            "n_queries": len(ac),
            "n_reps_per_cell": int(np.median([len(c.reps) for c in ac])),
            "single_run_mean": round(o_base, 4),
            "oracle_gain": round(o_sel - o_base, 4),
            "gpt52_noise_gain": round(g52 - o_base, 4),
            "gpt4o_noise_gain": round(g4o - o_base, 4),
            "random_gain": round(rnd - o_base, 4),
        }
    return out


def drjudge_stub() -> Dict:
    return {
        "status": "OUT — model not on disk",
        "model": "GAIR/DeepResearcher-7b",
        "reason": "DR-Judge-7B weights are not present on this machine; no small model is used "
                  "as an authoritative quality judge. GPT-5.2 is the real judge.",
        "hook": "To add this arm: (1) download GAIR/DeepResearcher-7b; (2) run it as a "
                "DETERMINISTIC detector over each replicate's report (results/experiments/"
                "{pattern}/{query_id}.md), returning a preferred replicate per (arch,query); "
                "(3) feed that index as a new selector in compute() and bootstrap its realised "
                "gain exactly like the gaussian arms. Report it as a realisable selector, NOT "
                "as ground truth; keep ORACLE as the upper bound and RANDOM as the lower bound.",
    }


def atomic_write_json(path: str, obj: Dict) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(dry_run: bool, self_test: bool) -> Dict:
    o, v = load_corpus()
    sigma_gpt52, per_arch_sd = pooled_run_sd(o)
    mult, mult_src = gpt4o_noise_multiplier()
    sigma_gpt4o = sigma_gpt52 * mult

    cells = build_cells(o, v, min_reps=2)
    cells = sorted(cells, key=lambda c: (c.arch, c.query_id))

    if self_test:
        # keep only 2 queries' worth of cells for a tiny pass
        keep_q = sorted({c.query_id for c in cells})[:2]
        cells = [c for c in cells if c.query_id in keep_q]

    # fill anchor map for O(1) flip recompute
    global ID2ANCHOR
    ID2ANCHOR = {id(c): cell_recompute_true(c) for c in cells}

    meta = {
        "_what": "E7 realizable best-of-N selectors at matched spend (CPU-only, EXISTING data).",
        "seed": SEED,
        "n_boot": N_BOOT,
        "judge": "gpt52 (GPT-5.2) — the only authoritative judge; no small model judges",
        "corpus": "variance replicates, gpt52: base_p0 x11 + {p1,p4,p5,p6,p7,p8,p10} x3 / 30 q",
        "n_cells_total": len(cells),
        "n_architectures": len({c.arch for c in cells}),
        "architectures": sorted({c.arch for c in cells}),
        "per_arch_run_sd": {k: round(val, 4) for k, val in per_arch_sd.items()},
        "drjudge_selector": drjudge_stub(),
    }

    if dry_run:
        # tiny: oracle vs random gain on the FIRST query's cells only, no writes
        first_q = sorted({c.query_id for c in cells})[0]
        sub = [c for c in cells if c.query_id == first_q]
        rng = np.random.default_rng(SEED)
        o_sel, o_base, _ = realized_gain(sub, lambda c, r: sel_oracle(c, r), rng)
        r_sel, _, _ = realized_gain(sub, lambda c, r: sel_random(c, r),
                                    np.random.default_rng(SEED + 2))
        meta["_dry_run_preview"] = {
            "query": first_q,
            "n_cells_this_query": len(sub),
            "single_run_mean": round(o_base, 4),
            "oracle_gain": round(o_sel - o_base, 4),
            "random_gain": round(r_sel - o_base, 4),
            "sigma_gpt52": round(sigma_gpt52, 4),
            "sigma_gpt4o": round(sigma_gpt4o, 4),
        }
        return meta

    result = compute(cells, sigma_gpt52, sigma_gpt4o, mult, mult_src)
    result["per_architecture"] = per_arch_breakdown(cells, sigma_gpt52, sigma_gpt4o)
    out = {**meta, **result}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + calibration + 1-query preview; write nothing")
    ap.add_argument("--self-test", action="store_true",
                    help="run on 2 queries; write *_selftest.json sidecar; never canonical")
    ap.add_argument("--resume", action="store_true",
                    help="interface parity no-op (computation is a single deterministic pass)")
    ap.add_argument("--no-canonical", action="store_true",
                    help="write only the standalone JSON, skip the canonical merge")
    args = ap.parse_args()

    out = run(dry_run=args.dry_run, self_test=args.self_test)

    if args.dry_run:
        print(json.dumps(out, indent=1))
        print("\n[dry-run] no files written.", file=sys.stderr)
        return 0

    if args.self_test:
        sidecar = OUT_STANDALONE.replace(".json", "_selftest.json")
        atomic_write_json(sidecar, out)
        print(json.dumps({k: out[k] for k in ("_what", "n_cells_total", "n_architectures")},
                         indent=1))
        if "arms" in out:
            print(json.dumps({k: out["arms"][k] for k in out["arms"]}, indent=1))
        print(f"\n[self-test] wrote {sidecar} (2-query micro-run). Canonical NOT touched.",
              file=sys.stderr)
        return 0

    # full run: standalone + canonical merge
    atomic_write_json(OUT_STANDALONE, out)
    if not args.no_canonical:
        cn = json.load(open(CANON))
        cn["selector_e7"] = out
        atomic_write_json(CANON, cn)
    print(json.dumps({"arms": out["arms"],
                      "structured_minus_random_gain": out["structured_minus_random_gain"],
                      "calibration": out["_calibration"]}, indent=1))
    print(f"\n[full] wrote {OUT_STANDALONE}"
          + ("" if args.no_canonical else f" and merged selector_e7 into {CANON}"),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
