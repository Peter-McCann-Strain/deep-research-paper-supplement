#!/usr/bin/env python3
"""DR-Judge-7B REAL error-correlation structure -> canonical key ``drjudge_error_structure``.

WHAT THIS IS (and is NOT)
-------------------------
Tier-1 blocker T1_drjudge_errmatrix. A PURE-CPU, READ-ONLY distiller that computes the
REAL per-criterion confusion / cross-dimension error-correlation structure of the
DR-Judge-7B judge versus the GPT-5.2 ground-truth labels, from the 3,824 paired verdicts
already frozen behind canonical['drjudge'].

It REPLACES the SYNTHETIC verdict-flip model used by E7/E10. Today E7's
``sel_flip_kappa`` (scripts/run_e7_selector.py) injects judge error as a SINGLE symmetric
marginal rate ``flip_p`` plus an AD-HOC per-replicate latent ``bias = 0.2 + 1.6*rand`` to
fake cross-criterion correlation. That synthetic model is wrong in two measurable ways the
real data fixes:
  (1) DR-Judge error is strongly ASYMMETRIC (false-negative rate >> false-positive rate),
      not a single symmetric flip probability; and
  (2) DR-Judge errors are POSITIVELY CORRELATED across dimensions WITHIN a report
      (a report DR-Judge mis-scores on one dimension it tends to mis-score on others) ---
      a real structure the ``0.2 + 1.6*rand`` latent only approximates by accident.
This block supplies the measured numbers (per-dimension FPR/FNR, the 9x9 phi and
tetrachoric error-correlation matrices, the mean off-diagonal correlation, and the global
2x2 confusion) so E7's structured-noise arm and E10's STRUCT_NOISE arm (Arm D) can be
calibrated to the OBSERVED structure instead of a synthetic one. It GATES Paper-4 GPU spend.

THE INPUT (real, on disk)
-------------------------
    reports/phase12_drjudge/eval_predictions_full.parquet   (3,824 rows)
columns: pattern, query_id, criterion_id, dimension, is_disputed, n_judges,
         target (GPT-5.2 ground-truth label, bool), predicted (DR-Judge-7B label, bool).
This is the EXACT artefact that canonical['drjudge'] (built by build_numbers.py:drjudge())
is computed from --- same 3,824 paired verdicts, same 99 report units (pattern x query),
same 9 dimensions. error := (predicted != target).

THE CANONICAL KEY --- ``drjudge_error_structure``
-------------------------------------------------
  * ``confusion``        --- global 2x2 {gold False/True} x {pred False/True}, plus the
                             marginal FPR (pred T | gold F), FNR (pred F | gold T), and the
                             overall error rate. These REPLACE the single symmetric flip_p.
  * ``per_dimension``    --- for each of the 9 dimensions: n, n_gold_pos/neg, FPR, FNR,
                             marginal error rate. (asymmetric, per-dimension flip targets.)
  * ``error_correlation``--- the 9x9 cross-dimension correlation of the per-report
                             dimension-level error indicator, computed two ways:
                               - ``phi``        : Pearson phi (point) correlation of the
                                                  binary report-level error indicator;
                               - ``tetrachoric``: latent-normal (tetrachoric) correlation,
                                                  the right object for E7/E10's latent-bias
                                                  Gaussian-copula noise model.
                             plus ``mean_offdiag_*`` summaries. Report-level error indicator
                             is "DR-Judge made >=1 criterion error on this dimension in this
                             report" (matches the per-replicate flip-mask granularity of
                             sel_flip_kappa, which flips per criterion but correlates per
                             replicate). A second ``rate`` variant correlates the continuous
                             per-(report,dim) error RATE for robustness.
  * ``calibration``      --- the E7/E10 hand-off: pooled marginal flip rate, the implied
                             cohen-kappa of the real DR-Judge vs gold (carried from
                             canonical['drjudge'] for cross-check), and the mean off-diagonal
                             tetrachoric correlation == the latent-bias copula rho that the
                             synthetic ``0.2 + 1.6*rand`` was standing in for.

DETERMINISM / IDEMPOTENCE / SAFETY
----------------------------------
  * Deterministic: closed-form arithmetic + a fixed-grid numerical tetrachoric MLE. No RNG,
    no sampling, no judge call, no paid API, no GPU.
  * Idempotent: re-running overwrites ONLY canonical['drjudge_error_structure'] via a
    load-merge-write; every other key is preserved byte-for-byte.
  * Self-guarding: if the parquet is absent, prints a notice and exits 0 (rebuild_all.sh
    stays green); writes nothing.
  * Refuses to CREATE the canonical store from scratch (run build_numbers.py first).
  * Atomic write via tempfile + os.replace (never a partial canonical).
  * NEVER clobbers verdicts/reports/parquets --- only the analysis canonical JSON.

Usage::
    python scripts/build_drjudge_error_structure.py            # merge into canonical
    python scripts/build_drjudge_error_structure.py --dry-run  # print summary, write nothing
    python scripts/build_drjudge_error_structure.py --no-canonical  # print full block only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

# --------------------------------------------------------------------------- #
# Paths (hardcoded, correct post-move absolute canonical path)
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(".")
EVAL_PARQUET = _REPO_ROOT / "reports" / "phase12_drjudge" / "eval_predictions_full.parquet"
CANON = (_REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
         / "canonical_numbers.json")
CANONICAL_KEY = "drjudge_error_structure"

# Fixed dimension order (matches build_numbers.py DIMS) so the matrix is stable.
DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization",
        "instruction_following", "attribution_quality"]


def _r(x, n=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, n)


# --------------------------------------------------------------------------- #
# Tetrachoric correlation: latent-normal MLE on a 2x2 table.
# Deterministic golden-section search over rho in (-0.999, 0.999) maximising the
# bivariate-normal log-likelihood of the four cell counts given the observed
# marginal thresholds. Standard estimator; the right object for a Gaussian-copula
# latent-bias noise model (E7/E10).
# --------------------------------------------------------------------------- #
def _bvn_cdf(h, k, rho):
    """P(X<=h, Y<=k) for standard bivariate normal with correlation rho."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    return float(multivariate_normal.cdf([h, k], mean=[0.0, 0.0],
                                         cov=[[1.0, rho], [rho, 1.0]]))


def _tetrachoric_from_table(n00, n01, n10, n11):
    """MLE tetrachoric correlation from a 2x2 table.

    Rows = variable X (0/1), cols = variable Y (0/1). n_ij = count(X=i, Y=j).
    Thresholds from observed marginals; rho found by deterministic golden search.
    Returns None for degenerate tables (an empty margin/cell).
    """
    n = n00 + n01 + n10 + n11
    if n == 0:
        return None
    # marginal proportions of the "0" class -> thresholds h, k
    p_x0 = (n00 + n01) / n
    p_y0 = (n00 + n10) / n
    # degenerate margins (a constant variable) -> correlation undefined
    if p_x0 in (0.0, 1.0) or p_y0 in (0.0, 1.0):
        return None
    # an empty cell -> boundary correlation; report the signed boundary, not None,
    # so the matrix has no holes. (Yule-style: zero cell => |rho|->1.)
    if min(n00, n01, n10, n11) == 0:
        # sign from the diagonal vs off-diagonal mass
        return 1.0 if (n00 * n11) >= (n01 * n10) else -1.0
    h = norm.ppf(p_x0)
    k = norm.ppf(p_y0)
    obs = n00 / n  # observed P(X=0, Y=0)

    def negll_gap(rho):
        # match P(X<=h, Y<=k) to observed joint-0 proportion (single moment is
        # exact for the 2x2 tetrachoric: the other three cells are determined by
        # the two marginals). Minimise squared gap -> deterministic, convex-ish.
        return (_bvn_cdf(h, k, rho) - obs) ** 2

    # golden-section minimisation over [-0.999, 0.999]
    a, b = -0.999, 0.999
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = negll_gap(c), negll_gap(d)
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = negll_gap(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = negll_gap(d)
        if abs(b - a) < 1e-10:
            break
    return float((a + b) / 2.0)


def _phi(x, y):
    """Pearson phi (point) correlation of two binary vectors. None if a constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
# Block builder
# --------------------------------------------------------------------------- #
def build_block(e: pd.DataFrame) -> dict:
    e = e.copy()
    e["err"] = (e["target"].astype(bool) != e["predicted"].astype(bool)).astype(int)

    # ---- global confusion (criterion level) ----
    gold = e["target"].astype(bool).values
    pred = e["predicted"].astype(bool).values
    n11 = int(np.sum(gold & pred))    # gold T, pred T
    n10 = int(np.sum(gold & ~pred))   # gold T, pred F (false negative)
    n01 = int(np.sum(~gold & pred))   # gold F, pred T (false positive)
    n00 = int(np.sum(~gold & ~pred))  # gold F, pred F
    n_pos = n11 + n10
    n_neg = n00 + n01
    confusion = {
        "n": int(len(e)),
        "gold_true_pred_true": n11,
        "gold_true_pred_false": n10,
        "gold_false_pred_true": n01,
        "gold_false_pred_false": n00,
        "n_gold_pos": n_pos,
        "n_gold_neg": n_neg,
        "fpr": _r(n01 / n_neg) if n_neg else None,   # pred T | gold F
        "fnr": _r(n10 / n_pos) if n_pos else None,    # pred F | gold T
        "error_rate": _r(e["err"].mean()),
        "_note": ("Replaces the single symmetric flip_p of sel_flip_kappa: real DR-Judge "
                  "error is asymmetric (FNR >> FPR)."),
    }

    # ---- per-dimension asymmetric error rates ----
    per_dimension = {}
    for d in DIMS:
        g = e[e["dimension"] == d]
        if len(g) == 0:
            continue
        gd = g["target"].astype(bool).values
        pd_ = g["predicted"].astype(bool).values
        npos = int(np.sum(gd))
        nneg = int(np.sum(~gd))
        fn = int(np.sum(gd & ~pd_))
        fp = int(np.sum(~gd & pd_))
        per_dimension[d] = {
            "n": int(len(g)),
            "n_gold_pos": npos,
            "n_gold_neg": nneg,
            "fpr": _r(fp / nneg) if nneg else None,
            "fnr": _r(fn / npos) if npos else None,
            "error_rate": _r(g["err"].mean()),
        }

    # ---- report-level dimension error indicator: 99 reports x 9 dims ----
    # "DR-Judge made >=1 criterion error on this dimension in this report" (any-error
    # indicator, matches the per-replicate correlated flip granularity of E7) and the
    # continuous per-(report,dim) error RATE (robustness variant).
    ind = (e.groupby(["pattern", "query_id", "dimension"])["err"].max()
           .unstack("dimension").reindex(columns=DIMS))
    rate = (e.groupby(["pattern", "query_id", "dimension"])["err"].mean()
            .unstack("dimension").reindex(columns=DIMS))
    n_reports = int(ind.shape[0])

    # phi correlation matrix of the binary any-error indicator
    phi_mat = {}
    tet_mat = {}
    for di in DIMS:
        phi_mat[di] = {}
        tet_mat[di] = {}
        xi = ind[di].values
        for dj in DIMS:
            xj = ind[dj].values
            if di == dj:
                phi_mat[di][dj] = 1.0
                tet_mat[di][dj] = 1.0
                continue
            phi_mat[di][dj] = _r(_phi(xi, xj))
            # tetrachoric on the 2x2 table of (xi, xj)
            t00 = int(np.sum((xi == 0) & (xj == 0)))
            t01 = int(np.sum((xi == 0) & (xj == 1)))
            t10 = int(np.sum((xi == 1) & (xj == 0)))
            t11 = int(np.sum((xi == 1) & (xj == 1)))
            tet_mat[di][dj] = _r(_tetrachoric_from_table(t00, t01, t10, t11))

    # continuous error-RATE Pearson correlation (robustness)
    rate_corr = rate.corr(method="pearson")
    rate_mat = {di: {dj: _r(rate_corr.loc[di, dj]) for dj in DIMS} for di in DIMS}

    def _mean_offdiag(mat):
        vals = [mat[di][dj] for di in DIMS for dj in DIMS
                if di != dj and mat[di][dj] is not None]
        return _r(np.mean(vals)) if vals else None

    mean_phi = _mean_offdiag(phi_mat)
    mean_tet = _mean_offdiag(tet_mat)
    mean_rate = _mean_offdiag(rate_mat)

    error_correlation = {
        "dims": DIMS,
        "level": ("report-level dimension any-error indicator across "
                  f"{n_reports} reports (pattern x query)"),
        "n_reports": n_reports,
        "phi": phi_mat,
        "tetrachoric": tet_mat,
        "rate_pearson": rate_mat,
        "mean_offdiag_phi": mean_phi,
        "mean_offdiag_tetrachoric": mean_tet,
        "mean_offdiag_rate_pearson": mean_rate,
        "_note": ("Cross-dimension error correlation REPLACES the synthetic per-replicate "
                  "latent 'bias = 0.2 + 1.6*rand' in sel_flip_kappa. mean_offdiag_tetrachoric "
                  "is the latent-normal copula rho a structured-noise sampler should use."),
    }

    # ---- E7/E10 calibration hand-off ----
    calibration = {
        "pooled_marginal_flip_rate": confusion["error_rate"],
        "fpr": confusion["fpr"],
        "fnr": confusion["fnr"],
        "latent_copula_rho_tetrachoric": mean_tet,
        "latent_copula_rho_phi": mean_phi,
        "replaces": {
            "e7_sel_flip_kappa_symmetric_flip_p": "scripts/run_e7_selector.py::sel_flip_kappa",
            "e7_synthetic_latent_bias": "bias = flip_p * (0.2 + 1.6 * rand)  (ad-hoc)",
            "e10_struct_noise_arm": "scripts/e10_noise_rl_readiness.py Arm D (--noise-mode struct_noise)",
        },
        "_note": ("E7/E10 should draw per-report correlated flips from a Gaussian copula with "
                  "off-diagonal rho = latent_copula_rho_tetrachoric and per-dimension "
                  "asymmetric thresholds (fpr/fnr from per_dimension), instead of a single "
                  "symmetric flip_p with a hand-tuned latent."),
    }

    return {
        "_what": ("REAL DR-Judge-7B vs GPT-5.2 per-criterion error structure (confusion + "
                  "9x9 cross-dimension error correlation) that replaces the synthetic "
                  "verdict-flip model and calibrates E7/E10. Tier-1 blocker "
                  "T1_drjudge_errmatrix; gates Paper-4 GPU."),
        "source_artifact": "reports/phase12_drjudge/eval_predictions_full.parquet",
        "n_paired_verdicts": int(len(e)),
        "n_reports": n_reports,
        "n_dimensions": len(DIMS),
        "judge_under_test": "DR-Judge-7B (GAIR/DeepResearcher-7b finetune); predicted column",
        "ground_truth": "GPT-5.2 label; target column (NO Opus; no new judging)",
        "deterministic": True,
        "confusion": confusion,
        "per_dimension": per_dimension,
        "error_correlation": error_correlation,
        "calibration": calibration,
    }


# --------------------------------------------------------------------------- #
# Atomic canonical write (load-merge-write)
# --------------------------------------------------------------------------- #
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
                    help="print the summary; write nothing.")
    ap.add_argument("--no-canonical", action="store_true",
                    help="print the full block; do NOT merge into canonical_numbers.json.")
    args = ap.parse_args()

    if not EVAL_PARQUET.exists():
        print(f"[{CANONICAL_KEY}] {EVAL_PARQUET} not present; nothing to compute. "
              f"(exit 0, wrote nothing)", file=sys.stderr)
        return 0

    e = pd.read_parquet(EVAL_PARQUET)
    block = build_block(e)

    summary = {
        "n_paired_verdicts": block["n_paired_verdicts"],
        "n_reports": block["n_reports"],
        "confusion": {
            "fpr": block["confusion"]["fpr"],
            "fnr": block["confusion"]["fnr"],
            "error_rate": block["confusion"]["error_rate"],
        },
        "mean_offdiag_tetrachoric": block["error_correlation"]["mean_offdiag_tetrachoric"],
        "mean_offdiag_phi": block["error_correlation"]["mean_offdiag_phi"],
        "latent_copula_rho_tetrachoric": block["calibration"]["latent_copula_rho_tetrachoric"],
    }
    print(json.dumps({CANONICAL_KEY: summary}, indent=1))

    if args.dry_run:
        print(f"\n[dry-run] canonical_numbers.json['{CANONICAL_KEY}'] NOT written.",
              file=sys.stderr)
        return 0
    if args.no_canonical:
        print(json.dumps({CANONICAL_KEY: block}, indent=1))
        print(f"\n[no-canonical] full block printed; canonical NOT touched.", file=sys.stderr)
        return 0

    if not CANON.exists():
        print(f"[{CANONICAL_KEY}] canonical store missing at {CANON}; refusing to create it "
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
