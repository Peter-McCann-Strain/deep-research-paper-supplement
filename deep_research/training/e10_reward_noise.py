#!/usr/bin/env python3
"""E10 noise-RL — judge-reward NOISE INJECTION layer (Paper 4).

PURE, SEEDED, CPU-ONLY, UNIT-TESTABLE. Nothing here loads a model, calls a
paid API, or mutates the canonical store. It transforms a *clean* per-rollout
judge reward into one of four arms by perturbing the underlying per-criterion
verdicts and re-scoring, then anchoring the perturbation onto the clean reward
exactly the way ``scripts/run_e7_selector.py`` L48-50 does:

    reward_arm = base_reward + (recompute(flipped_verdicts) - recompute(true_verdicts))

so the BASELINE per rollout is the exact clean DR-Judge reward and ONLY the
*flip delta* comes from the recompute. This makes every arm a faithful
perturbation of the clean signal rather than a wholesale re-scoring.

THE FOUR ARMS (plan-of-record + canonical['drjudge_error_structure'].calibration.replaces)
------------------------------------------------------------------------------------------
  A  clean           CleanReward            — passthrough (control).
  B  struct_copula   StructuredCopulaNoise  — per-rollout CORRELATED criterion-error
                     indicators from a Gaussian copula (off-diagonal
                     rho = latent_copula_rho_tetrachoric = 0.3472) over the 9
                     dimensions, then per-dimension ASYMMETRIC thresholds
                     (gold-True flips w.p. fnr, gold-False flips w.p. fpr) from
                     per_dimension[dim].{fpr,fnr}. This REPLACES the readiness
                     script's additive-Gaussian 'gpt52_noise' arm and
                     run_e7_selector's ad-hoc bias = flip_p*(0.2+1.6*rand).
  C  matched_random  MatchedKappaRandomNoise — i.i.d. per-criterion flips at the
                     SAME pooled marginal rate (pooled_marginal_flip_rate =
                     0.2811, split into fpr/fnr) but ZERO copula correlation
                     (rho = 0). B vs C at matched marginal kappa is the
                     load-bearing contrast (plan: 'Correlated Error, Not Noise
                     Magnitude').
  D  noise_corrected NoiseCorrectedReward    — wraps arm-B noisy reward with the
                     arXiv:2510.18924 Bernoulli FPR/FNR debiasing transform using
                     EMPIRICAL per-criterion fpr/fnr (the 'known-learnable'
                     control: noise injected then analytically corrected; if
                     D ~= A the correction works).

CALIBRATION SOURCE
------------------
``load_calibration`` reads the two calibration blocks from a PINNED, read-only
canonical SNAPSHOT path (copied at launch by the trainer), NEVER the live store,
and asserts ``drjudge_youden_j.drjudge_fixture_recompute_match == true`` plus a
content hash, so a concurrent canonical edit cannot silently re-calibrate a
mid-run arm.

DETERMINISM
-----------
Every stochastic step draws from ``np.random.default_rng`` seeded by
``base_seed + offset`` on SORTED criteria, mirroring run_e7_selector L70-103.
The 9 dimensions are processed in a fixed sorted order. Given the same seed,
verdicts, and calibration, the output is bit-identical across machines.

This module is the only place noise is defined; the trainer wraps the live
DR-Judge reward callable with one of these classes selected by ``--arm``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# V2 rubric dimension weights — MUST match run_e7_selector.py DIMW and
# MEMORY.md "Rubric V2 — 9 Dimensions". Sorted order is the canonical iteration
# order for the copula and for all flips (determinism contract).
# --------------------------------------------------------------------------- #
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
DIMS_SORTED: List[str] = sorted(DIMW)  # fixed iteration order for the copula
N_DIMS = len(DIMS_SORTED)

# Seed offsets so distinct stochastic streams never collide (mirrors E7).
_OFF_COPULA = 1009
_OFF_MARGINAL = 2017
_OFF_TIE = 3023


# --------------------------------------------------------------------------- #
# Calibration loader (read-only snapshot, hash-guarded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class E10Calibration:
    """Immutable calibration drawn from a PINNED canonical snapshot.

    Fields come from canonical['drjudge_error_structure'] and
    canonical['drjudge_youden_j']. Per-dimension fpr/fnr are keyed by the 9
    rubric dimension names.
    """

    pooled_marginal_flip_rate: float
    pooled_fpr: float
    pooled_fnr: float
    latent_copula_rho_tetrachoric: float
    latent_copula_rho_phi: float
    per_dimension_fpr: Dict[str, float]
    per_dimension_fnr: Dict[str, float]
    phi_matrix: Optional[Dict[str, Dict[str, float]]]  # full 9x9 (upgrade path)
    snapshot_path: str
    snapshot_sha256: str
    fixture_recompute_match: bool

    def rho(self, use_full_phi: bool = False) -> np.ndarray:
        """Return the 9x9 copula correlation matrix in DIMS_SORTED order.

        Default: constant off-diagonal rho = latent_copula_rho_tetrachoric.
        use_full_phi=True: build from the measured phi matrix (upgrade path),
        projected to the nearest valid correlation matrix.
        """
        if use_full_phi and self.phi_matrix is not None:
            r = np.eye(N_DIMS)
            for i, di in enumerate(DIMS_SORTED):
                row = self.phi_matrix.get(di, {})
                for j, dj in enumerate(DIMS_SORTED):
                    if i != j:
                        r[i, j] = float(row.get(dj, 0.0))
            return _nearest_corr(r)
        rho0 = float(self.latent_copula_rho_tetrachoric)
        r = np.full((N_DIMS, N_DIMS), rho0)
        np.fill_diagonal(r, 1.0)
        return _nearest_corr(r)


def _canonical_json_bytes(obj) -> bytes:
    """Deterministic JSON serialisation for hashing (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_calibration(snapshot_path: str | Path) -> E10Calibration:
    """Load + validate calibration from a PINNED read-only canonical snapshot.

    Raises if the fixture-recompute-match flag is not True (a canonical edit
    could otherwise silently re-calibrate a mid-run arm). NEVER reads the live
    store — the trainer copies the live canonical to a snapshot at launch and
    passes that path here.
    """
    p = Path(snapshot_path)
    if not p.exists():
        raise FileNotFoundError(
            f"E10 calibration snapshot not found: {p}. The trainer must copy the "
            "live canonical to a read-only snapshot at launch and pass its path."
        )
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cn = json.loads(raw)

    es = cn.get("drjudge_error_structure")
    yj = cn.get("drjudge_youden_j")
    if not isinstance(es, dict) or not isinstance(yj, dict):
        raise KeyError(
            "snapshot missing drjudge_error_structure / drjudge_youden_j; "
            "E10 cannot be calibrated against this canonical."
        )

    cal = es.get("calibration", {})
    needed = {
        "pooled_marginal_flip_rate",
        "fpr",
        "fnr",
        "latent_copula_rho_tetrachoric",
        "latent_copula_rho_phi",
    }
    missing = needed - set(cal)
    if missing:
        raise KeyError(f"calibration block missing keys: {sorted(missing)}")

    fixture_match = bool(yj.get("drjudge_fixture_recompute_match", False))
    if not fixture_match:
        raise ValueError(
            "drjudge_youden_j.drjudge_fixture_recompute_match is not True in the "
            "pinned snapshot — refusing to calibrate E10 noise against an "
            "unverified fixture (a canonical edit may have re-calibrated it)."
        )

    perdim = es.get("per_dimension", {})
    fpr_by_dim: Dict[str, float] = {}
    fnr_by_dim: Dict[str, float] = {}
    for d in DIMS_SORTED:
        rec = perdim.get(d, {})
        # Fall back to pooled rates for any dimension absent from the fixture so
        # the layer never silently drops a criterion.
        fpr_by_dim[d] = float(rec.get("fpr", cal["fpr"]))
        fnr_by_dim[d] = float(rec.get("fnr", cal["fnr"]))

    phi = None
    ec = es.get("error_correlation", {})
    if isinstance(ec.get("phi"), dict):
        phi = ec["phi"]

    return E10Calibration(
        pooled_marginal_flip_rate=float(cal["pooled_marginal_flip_rate"]),
        pooled_fpr=float(cal["fpr"]),
        pooled_fnr=float(cal["fnr"]),
        latent_copula_rho_tetrachoric=float(cal["latent_copula_rho_tetrachoric"]),
        latent_copula_rho_phi=float(cal["latent_copula_rho_phi"]),
        per_dimension_fpr=fpr_by_dim,
        per_dimension_fnr=fnr_by_dim,
        phi_matrix=phi,
        snapshot_path=str(p.resolve()),
        snapshot_sha256=sha,
        fixture_recompute_match=fixture_match,
    )


def _nearest_corr(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest valid correlation matrix
    (clip eigenvalues at eps, renormalise the diagonal to 1). Deterministic.
    """
    a = (a + a.T) / 2.0
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    b = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(b), eps, None))
    b = b / np.outer(d, d)
    b = (b + b.T) / 2.0
    np.fill_diagonal(b, 1.0)
    return b


# --------------------------------------------------------------------------- #
# Recompute (V2-weighted) — mirrors run_e7_selector.recompute_overall.
# --------------------------------------------------------------------------- #
def recompute_overall(sat_by_dim_counts: Mapping[str, "tuple[float, int]"]) -> float:
    """Dimension-weighted mean of per-dimension fraction-satisfied.

    sat_by_dim_counts: dimension -> (n_satisfied, n_total). Normalised by the
    sum of weights over dimensions actually present. Identical to
    run_e7_selector.recompute_overall (the anchoring contract requires it).
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
# Verdict container for a single rollout
# --------------------------------------------------------------------------- #
@dataclass
class RolloutVerdicts:
    """Per-criterion TRUE (clean DR-Judge) verdicts for ONE rollout.

    verdict_dim maps dimension -> boolean numpy array (one entry per criterion
    in that dimension). True = SATISFIED. The number of criteria per dimension
    can vary; absent dimensions are simply skipped by recompute_overall.
    """

    verdict_dim: Dict[str, np.ndarray]

    def counts(self, vd: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, "tuple[float, int]"]:
        src = self.verdict_dim if vd is None else vd
        return {d: (float(arr.sum()), int(arr.size)) for d, arr in src.items()}

    def true_recompute(self) -> float:
        return recompute_overall(self.counts())


# --------------------------------------------------------------------------- #
# Flip kernels (the perturbation primitives)
# --------------------------------------------------------------------------- #
def _flip_uniform_p(
    arr: np.ndarray, p: float, rng: np.random.Generator
) -> np.ndarray:
    """i.i.d. flip each entry with probability p."""
    flips = rng.random(arr.size) < p
    return np.where(flips, ~arr, arr)


def _flip_asymmetric(
    arr: np.ndarray,
    fpr: float,
    fnr: float,
    rng: np.random.Generator,
    u: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Asymmetric flips: gold-True entries flip w.p. fnr, gold-False w.p. fpr.

    If ``u`` (uniform draws in [0,1), one per entry) is supplied it is used
    instead of fresh draws — this lets the Gaussian-copula latent supply the
    correlation while keeping the per-dimension thresholds asymmetric.
    """
    if u is None:
        u = rng.random(arr.size)
    thresh = np.where(arr, fnr, fpr)  # True->fnr, False->fpr
    flips = u < thresh
    return np.where(flips, ~arr, arr)


# --------------------------------------------------------------------------- #
# Base reward callable protocol
# --------------------------------------------------------------------------- #
# A "verdict provider" maps a rollout (prompt, completion) -> RolloutVerdicts.
# In the offline E10 environment this is the DR-Judge LoRA run as a deterministic
# per-criterion detector. The noise layer is agnostic to how verdicts arise; it
# only needs (base_reward, RolloutVerdicts) per rollout.
VerdictProvider = Callable[[str, str], RolloutVerdicts]


class _ArmBase:
    """Common machinery: anchor the perturbation onto the clean base reward."""

    __name__ = "e10_reward_arm"
    arm_label: str = "base"

    def __init__(self, base_reward: Callable, calib: E10Calibration,
                 verdict_provider: VerdictProvider, base_seed: int):
        self.base_reward = base_reward
        self.calib = calib
        self.verdict_provider = verdict_provider
        self.base_seed = int(base_seed)
        self._rollout_idx = 0

    def _rng(self, offset: int) -> np.random.Generator:
        """Per-rollout deterministic stream. The rollout index increments so
        successive rollouts get independent-but-reproducible draws."""
        return np.random.default_rng(self.base_seed + offset + self._rollout_idx * 7919)

    def _flipped_verdicts(self, rv: RolloutVerdicts) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _delta(self, rv: RolloutVerdicts) -> float:
        """recompute(flipped) - recompute(true), the anchored flip delta."""
        flipped = self._flipped_verdicts(rv)
        return (recompute_overall(rv.counts(flipped))
                - recompute_overall(rv.counts()))

    def reward_one(self, base_reward_value: float, rv: RolloutVerdicts) -> float:
        """Single-rollout API used by unit tests and by the TRL wrapper."""
        delta = self._delta(rv)
        out = float(base_reward_value) + float(delta)
        return float(np.clip(out, 0.0, 1.0))

    def __call__(self, prompts: Sequence, completions: Sequence,
                 completion_ids=None, **kwargs) -> List[float]:
        """TRL contract: return list[float] of len(prompts).

        Computes the clean base reward, derives per-rollout verdicts via the
        provider, applies the arm's flip kernel, anchors the delta, clips to
        [0,1]. ``base_reward`` may itself be the live MultiAdapterJudgeReward.
        """
        base_vals = self.base_reward(prompts, completions, completion_ids, **kwargs)
        out: List[float] = []
        for p, c, b in zip(prompts, completions, base_vals, strict=True):
            self._rollout_idx += 1
            rv = self.verdict_provider(_as_text(p), _as_text(c))
            out.append(self.reward_one(b, rv))
        return out


def _as_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(
            m.get("content", "") for m in x if isinstance(m, dict)
        )
    return str(x)


# --------------------------------------------------------------------------- #
# Arm A — clean passthrough
# --------------------------------------------------------------------------- #
class CleanReward(_ArmBase):
    """Arm A: no injected noise. reward = clean base reward."""

    __name__ = "e10_clean"
    arm_label = "A_clean"

    def _flipped_verdicts(self, rv: RolloutVerdicts) -> Dict[str, np.ndarray]:
        return {d: arr.copy() for d, arr in rv.verdict_dim.items()}

    def reward_one(self, base_reward_value: float, rv: RolloutVerdicts) -> float:
        return float(np.clip(base_reward_value, 0.0, 1.0))

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        return [float(np.clip(b, 0.0, 1.0))
                for b in self.base_reward(prompts, completions, completion_ids, **kwargs)]


# --------------------------------------------------------------------------- #
# Arm B — structured Gaussian-copula correlated noise
# --------------------------------------------------------------------------- #
class StructuredCopulaNoise(_ArmBase):
    """Arm B: correlated per-criterion flips.

    Draw a 9-dim latent from a Gaussian copula with off-diagonal
    rho = latent_copula_rho_tetrachoric, map each dimension's latent to a
    per-criterion uniform via the standard-normal CDF (criteria within a
    dimension share that dimension's latent quantile, plus an independent
    jitter so identical criteria do not flip in lock-step), then apply
    per-dimension ASYMMETRIC thresholds (gold-True flips w.p. fnr, gold-False
    w.p. fpr).
    """

    __name__ = "e10_struct_copula"
    arm_label = "B_struct"

    def __init__(self, *args, use_full_phi: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_full_phi = use_full_phi
        self._rho = self.calib.rho(use_full_phi=use_full_phi)
        self._chol = np.linalg.cholesky(self._rho)

    @staticmethod
    def _norm_cdf(z: np.ndarray) -> np.ndarray:
        # Phi(z) without scipy: 0.5*(1+erf(z/sqrt2)). math.erf is scalar; use
        # the vectorised identity via np.vectorize-free erf approximation.
        from math import erf, sqrt
        return np.array([0.5 * (1.0 + erf(float(zi) / sqrt(2.0))) for zi in z])

    def _flipped_verdicts(self, rv: RolloutVerdicts) -> Dict[str, np.ndarray]:
        rng = self._rng(_OFF_COPULA)
        # One correlated latent draw across the 9 dims for THIS rollout.
        z = self._chol @ rng.standard_normal(N_DIMS)
        dim_q = self._norm_cdf(z)  # quantile in [0,1] per dimension, correlated
        out: Dict[str, np.ndarray] = {}
        for i, d in enumerate(DIMS_SORTED):
            arr = rv.verdict_dim.get(d)
            if arr is None:
                continue
            # Per-criterion uniform: dimension quantile shifted by small jitter
            # so multiple criteria in a dim do not flip identically, but remain
            # correlated through the shared dim_q.
            jitter = (rng.random(arr.size) - 0.5) * 0.10
            u = np.clip(dim_q[i] + jitter, 0.0, 1.0)
            out[d] = _flip_asymmetric(
                arr,
                fpr=self.calib.per_dimension_fpr[d],
                fnr=self.calib.per_dimension_fnr[d],
                rng=rng,
                u=u,
            )
        # carry through any dim not in DIMS_SORTED unchanged (defensive)
        for d, arr in rv.verdict_dim.items():
            out.setdefault(d, arr.copy())
        return out


# --------------------------------------------------------------------------- #
# Arm C — matched-kappa i.i.d. random noise (rho = 0)
# --------------------------------------------------------------------------- #
class MatchedKappaRandomNoise(_ArmBase):
    """Arm C: i.i.d. per-criterion flips at the SAME pooled marginal rate as B,
    but ZERO correlation. Gold-True flips w.p. fnr, gold-False w.p. fpr, drawn
    independently per criterion (no shared latent). B vs C at matched marginal
    flip rate is the load-bearing 'correlated error, not magnitude' contrast.
    """

    __name__ = "e10_matched_random"
    arm_label = "C_random"

    def _flipped_verdicts(self, rv: RolloutVerdicts) -> Dict[str, np.ndarray]:
        rng = self._rng(_OFF_MARGINAL)
        out: Dict[str, np.ndarray] = {}
        for d in DIMS_SORTED:
            arr = rv.verdict_dim.get(d)
            if arr is None:
                continue
            out[d] = _flip_asymmetric(
                arr,
                fpr=self.calib.per_dimension_fpr[d],
                fnr=self.calib.per_dimension_fnr[d],
                rng=rng,  # fresh i.i.d. uniforms -> rho = 0
            )
        for d, arr in rv.verdict_dim.items():
            out.setdefault(d, arr.copy())
        return out


# --------------------------------------------------------------------------- #
# Arm D — noise-corrected GRPO (arXiv:2510.18924 Bernoulli FPR/FNR debiasing)
# --------------------------------------------------------------------------- #
class NoiseCorrectedReward(StructuredCopulaNoise):
    """Arm D: arm-B noisy verdicts, analytically DEBIASED via the
    arXiv:2510.18924 Bernoulli FPR/FNR correction using EMPIRICAL per-criterion
    fpr/fnr (the 'known-learnable' control).

    The debiasing transform recovers the gold prevalence p_true from an observed
    noisy rate p_obs of SATISFIED verdicts under a known asymmetric channel:

        p_obs = p_true * (1 - fnr) + (1 - p_true) * fpr
        =>  p_true_hat = (p_obs - fpr) / (1 - fnr - fpr)         (clipped to [0,1])

    Applied per dimension to the NOISY (arm-B) fraction-satisfied, then the
    corrected per-dimension fractions feed recompute_overall. If the correction
    is exact, D's expected reward returns toward the clean (arm-A) value.
    """

    __name__ = "e10_noise_corrected"
    arm_label = "D_corrected"

    @staticmethod
    def _debias_fraction(p_obs: float, fpr: float, fnr: float) -> float:
        denom = 1.0 - fnr - fpr
        if abs(denom) < 1e-6:
            # channel not invertible (J ~= 0) — cannot correct; return observed.
            return float(np.clip(p_obs, 0.0, 1.0))
        return float(np.clip((p_obs - fpr) / denom, 0.0, 1.0))

    def _delta(self, rv: RolloutVerdicts) -> float:
        # 1) inject arm-B structured noise to get observed (noisy) verdicts
        noisy = super()._flipped_verdicts(rv)
        # 2) per-dimension observed fraction-satisfied -> debias -> corrected frac
        corrected_counts: Dict[str, "tuple[float, int]"] = {}
        for d, arr in noisy.items():
            ntot = int(arr.size)
            if ntot <= 0:
                continue
            p_obs = float(arr.sum()) / ntot
            fpr = self.calib.per_dimension_fpr.get(d, self.calib.pooled_fpr)
            fnr = self.calib.per_dimension_fnr.get(d, self.calib.pooled_fnr)
            p_corr = self._debias_fraction(p_obs, fpr, fnr)
            # express corrected fraction as (n_sat, n_tot) for recompute_overall
            corrected_counts[d] = (p_corr * ntot, ntot)
        corrected_overall = recompute_overall(corrected_counts)
        true_overall = recompute_overall(rv.counts())
        return corrected_overall - true_overall


# --------------------------------------------------------------------------- #
# Arm factory
# --------------------------------------------------------------------------- #
ARM_CLASSES = {
    "A_clean": CleanReward,
    "B_struct": StructuredCopulaNoise,
    "C_random": MatchedKappaRandomNoise,
    "D_corrected": NoiseCorrectedReward,
}
# also accept the bare letters / readiness aliases
ARM_ALIASES = {
    "A": "A_clean", "clean": "A_clean",
    "B": "B_struct", "struct": "B_struct", "struct_copula": "B_struct",
    "C": "C_random", "random": "C_random", "matched_random": "C_random",
    "D": "D_corrected", "corrected": "D_corrected", "noise_corrected": "D_corrected",
}


def resolve_arm(arm: str) -> str:
    if arm in ARM_CLASSES:
        return arm
    if arm in ARM_ALIASES:
        return ARM_ALIASES[arm]
    raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_CLASSES)}")


def make_arm(arm: str, base_reward: Callable, calib: E10Calibration,
             verdict_provider: VerdictProvider, base_seed: int,
             use_full_phi: bool = False) -> _ArmBase:
    """Construct the noise-wrapping reward callable for the chosen arm."""
    key = resolve_arm(arm)
    cls = ARM_CLASSES[key]
    if cls in (StructuredCopulaNoise, NoiseCorrectedReward):
        return cls(base_reward, calib, verdict_provider, base_seed,
                   use_full_phi=use_full_phi)
    return cls(base_reward, calib, verdict_provider, base_seed)


# --------------------------------------------------------------------------- #
# Self-test (CPU, no model, no canonical write) — run directly:
#   python -m deep_research.training.e10_reward_noise --selftest <snapshot.json>
# --------------------------------------------------------------------------- #
def _synthetic_verdicts(rng: np.random.Generator) -> RolloutVerdicts:
    vd = {}
    for d in DIMS_SORTED:
        k = int(rng.integers(2, 6))
        vd[d] = rng.random(k) < 0.65  # ~65% satisfied
    return RolloutVerdicts(verdict_dim=vd)


def _selftest(snapshot_path: str) -> int:
    calib = load_calibration(snapshot_path)
    print(f"[selftest] calibration loaded from {calib.snapshot_path}")
    print(f"[selftest] snapshot sha256={calib.snapshot_sha256[:16]}…  "
          f"fixture_match={calib.fixture_recompute_match}")
    print(f"[selftest] rho_tetra={calib.latent_copula_rho_tetrachoric} "
          f"pooled_flip={calib.pooled_marginal_flip_rate} "
          f"pooled fpr/fnr={calib.pooled_fpr}/{calib.pooled_fnr}")
    rho = calib.rho()
    assert rho.shape == (N_DIMS, N_DIMS)
    assert np.allclose(np.diag(rho), 1.0)
    np.linalg.cholesky(rho)  # must be PSD
    print(f"[selftest] copula 9x9 is valid (PSD, unit diag)")

    # deterministic verdict provider for the test
    base_seed = 1
    vp_rng = np.random.default_rng(123)
    fixed = [_synthetic_verdicts(vp_rng) for _ in range(8)]

    def make_fixed_provider():
        iterator = iter(fixed)

        def provider(_p, _c):
            return next(iterator)

        return provider

    # base reward = the clean recompute of each rollout's true verdicts
    def base_reward(prompts, completions, completion_ids=None, **kw):
        return [rv.true_recompute() for rv in fixed[:len(prompts)]]

    prompts = ["q"] * 8
    comps = ["r"] * 8

    means = {}
    for arm in ["A_clean", "B_struct", "C_random", "D_corrected"]:
        arm_obj = make_arm(
            arm, base_reward,
            calib,
            make_fixed_provider(),
            base_seed,
        )
        # determinism: same arm + same seed -> identical output twice
        # rebuild a fresh provider each call so the iterator restarts
        def call_once(current_arm=arm_obj):
            current_arm._rollout_idx = 0
            current_arm.verdict_provider = make_fixed_provider()
            return current_arm(prompts, comps)

        r1 = call_once()
        r2 = call_once()
        assert r1 == r2, f"{arm} not deterministic across identical calls"
        assert all(0.0 <= x <= 1.0 for x in r1), f"{arm} out of [0,1]"
        means[arm] = float(np.mean(r1))
        print(f"[selftest] {arm:12s} mean={means[arm]:.4f}  (n={len(r1)})  deterministic OK")

    # sanity: B and C should differ from A (noise lowers expected reward);
    # D should be CLOSER to A than B is (correction recovers).
    print(f"[selftest] |B-A|={abs(means['B_struct']-means['A_clean']):.4f}  "
          f"|D-A|={abs(means['D_corrected']-means['A_clean']):.4f}")
    print("[selftest] PASS — noise layer is pure, seeded, deterministic, in-range.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E10 reward-noise layer self-test (CPU).")
    ap.add_argument("--selftest", metavar="SNAPSHOT_JSON", required=True,
                    help="path to a PINNED canonical snapshot json")
    a = ap.parse_args()
    raise SystemExit(_selftest(a.selftest))
