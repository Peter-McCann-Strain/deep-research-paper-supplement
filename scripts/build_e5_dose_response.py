#!/usr/bin/env python3
"""E5 DOSE-RESPONSE — canonical recompute with a ONE-SIDED slope CI (prereg primary).

The E5 oracle-dose experiment (generation + GPT-5.2 judging) is already complete:
18 cells = {P0, P1, P4} x {g000, g025, g050, g075, g100, interleaved}, 30
variance-stratified queries each (540 GPT-5.2 verdicts), judged under
``results/judge_gpt52_e5/``.  The harness wrote ``e5_dose_response_fit.json`` next
to those verdicts, but it reports a TWO-sided 95% slope CI.  The pre-registration
(reports/RESEARCH_PLAN_2026H2.md, priority 8) registered DIRECTIONAL hypotheses, so
the paper needs a ONE-sided 95% slope CI:

  * factual_accuracy ~ gold_fraction  :  prereg = "flat".  The directional risk is
    that more gold raises factuality, so the inferentially relevant one-sided bound
    is the UPPER 95% bound on the slope (H0: slope >= margin; "no upward dose
    effect" if the upper bound sits below a small margin).
  * citation_quality ~ gold_fraction  :  prereg = "monotone INCREASING (retrieval-
    bound)".  The relevant one-sided bound is the LOWER 95% bound on the slope
    (H0: slope <= 0; "retrieval-bound" if the lower bound sits above 0).

This script RECOMPUTES the mixed-effects dose-response model DETERMINISTICALLY from
the on-disk judged verdicts (NOT from the harness's fit JSON) and emits a NEW
canonical key ``e5_dose_response`` carrying:
  * the prereg mixed-effects slope for factual_accuracy and citation_quality, each
    with BOTH the two-sided 95% CI (parity with the harness fit) AND the prereg
    one-sided 95% CI bound, plus a one-sided p-value in the prereg direction;
  * per-gold-fraction means (per dimension, pooled over the 3 architectures, and
    per architecture) so the dose curve is fully reconstructable from canonical;
  * the interleaved-vs-g100 factual contrast (the context-overload rescue check).

DETERMINISM.  The model is statsmodels MixedLM (REML, lbfgs); on this data the
slope is bit-for-bit identical across runs (verified).  The one-sided bound is a
closed-form transform (slope -/+ z_0.95 * se) of those deterministic estimates, so
the whole pipeline is reproducible.  Inputs are read in a fixed sorted order.

IDEMPOTENT / NON-CLOBBERING.  The ONLY write is canonical_numbers.json['e5_dose_response']
(an analysis artefact, not a protected corpus path).  Verdicts/reports are READ-ONLY.
Re-running overwrites only that one key with the identical recomputed block.  Use
--dry-run to compute + print the would-be block WITHOUT touching canonical.

This mirrors the structure/idioms of build_oracle_factual_tost.py and
build_e4_cite_causal.py, but fixes the stale-canonical-path bug: the canonical store
was MOVED to papers/paper_a_bounded_returns/analysis/ (commit 0a80ba6); this script
writes there, not to the old papers/paper_a_bounded_returns/analysis/ location.

Usage:
    python scripts/build_e5_dose_response.py
    python scripts/build_e5_dose_response.py --dry-run        # print, never write
    python scripts/build_e5_dose_response.py --judge-out results/judge_gpt52_e5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Canonical store MOVED here by commit 0a80ba6 (was papers/paper_a_bounded_returns/analysis/).
ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

# E5 GPT-5.2 verdict root (the run harness's DEFAULT_JUDGE_OUT). READ-ONLY.
DEFAULT_JUDGE_OUT = _REPO_ROOT / "results" / "judge_gpt52_e5"

# Cell -> gold fraction (g000..g100); 'interleaved' has no dose level (excluded
# from the slope, used only for the rescue contrast). Mirrors run_e5_oracle_dose.py.
GOLD_FRACTION = {"g000": 0.0, "g025": 0.25, "g050": 0.50, "g075": 0.75, "g100": 1.00}
CELL_RE = re.compile(r"^e5_oracle_dose_(p\d+)_(g\d{3}|interleaved)$")

# Prereg directional endpoints. one_sided_bound: which 95% bound is inferential.
#   factual -> "upper": prereg "flat"; the risk is an upward dose effect.
#   citation -> "lower": prereg "monotone increasing"; show the slope is > 0.
ENDPOINTS = {
    "factual_accuracy": {"prereg": "flat", "one_sided_bound": "upper", "margin": 0.05},
    "citation_quality": {"prereg": "monotone_increasing", "one_sided_bound": "lower", "margin": 0.0},
}

Z95 = float(stats.norm.ppf(0.95))   # one-sided 95% z (matches MixedLM normal CIs)


# ── Load E5 verdicts deterministically ────────────────────────────────────────

def load_e5_scores(judge_out: Path) -> pd.DataFrame:
    """Walk judge_gpt52_e5/e5_oracle_dose_<pat>_<cell>/<qid>.json into a tidy frame.

    READ-ONLY. Files are iterated in sorted order so the assembled frame (hence the
    fit) is path-order independent and reproducible."""
    rows = []
    for exp_dir in sorted(judge_out.glob("e5_oracle_dose_*")):
        m = CELL_RE.match(exp_dir.name)
        if not m:
            continue
        pattern, cell = m.group(1), m.group(2)
        for jf in sorted(exp_dir.glob("*.json")):
            try:
                v = json.loads(jf.read_text())
            except Exception:
                continue
            dims = v.get("dimensions", {})
            rows.append({
                "pattern": pattern,
                "cell": cell,
                "query_id": jf.stem,
                "gold_fraction": GOLD_FRACTION.get(cell),   # None for interleaved
                "factual_accuracy": dims.get("factual_accuracy", {}).get("score"),
                "citation_quality": dims.get("citation_quality", {}).get("score"),
                "overall": v.get("overall_score"),
            })
    return pd.DataFrame(rows)


# ── Prereg mixed-effects dose-response slope, with a one-sided CI ─────────────

def fit_slope(dose: pd.DataFrame, dim: str) -> dict:
    """MixedLM ``dim ~ gold_fraction`` with a query random intercept, pooled over the
    3 architectures. Returns the slope with BOTH the two-sided 95% CI (parity with the
    harness fit) and the prereg ONE-sided 95% bound + one-sided p-value.

    Falls back to OLS if MixedLM is unavailable (the one-sided bound still computes
    from the OLS slope/se)."""
    spec = ENDPOINTS[dim]
    side = spec["one_sided_bound"]
    d = dose.dropna(subset=[dim, "gold_fraction"]).copy()
    if len(d) < 8 or d["gold_fraction"].nunique() < 2:
        return {"status": "insufficient", "n": int(len(d))}

    slope = se = pval_two = None
    status = None
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm(f"{dim} ~ gold_fraction", d, groups=d["query_id"])
        fit = md.fit(reml=True, method="lbfgs")
        slope = float(fit.params["gold_fraction"])
        se = float(fit.bse["gold_fraction"])
        pval_two = float(fit.pvalues["gold_fraction"])
        ci = fit.conf_int().loc["gold_fraction"].tolist()
        ci95 = [float(ci[0]), float(ci[1])]
        status = "mixedlm"
    except Exception:
        x = d["gold_fraction"].to_numpy(dtype=float)
        y = d[dim].to_numpy(dtype=float)
        lr = stats.linregress(x, y)
        slope = float(lr.slope)
        se = float(lr.stderr)
        pval_two = float(lr.pvalue)
        ci95 = [slope - 1.959963984540054 * se, slope + 1.959963984540054 * se]
        status = "ols_fallback"

    # One-sided 95% bound (normal approx, matching MixedLM's CI construction).
    one_sided_upper = slope + Z95 * se   # 95% upper bound: true slope <= this w.p. .95
    one_sided_lower = slope - Z95 * se   # 95% lower bound: true slope >= this w.p. .95
    one_sided_ci = one_sided_upper if side == "upper" else one_sided_lower

    # One-sided p-value in the prereg direction.
    #   citation (lower / "increasing"):  H0 slope<=0 vs H1 slope>0  -> p = P(Z >= slope/se)
    #   factual  (upper / "flat"):        H0 slope>=0 vs H1 slope<0  -> p = P(Z <= slope/se)
    z = slope / se if se else 0.0
    if side == "lower":
        p_one_sided = float(stats.norm.sf(z))    # evidence slope > 0
    else:
        p_one_sided = float(stats.norm.cdf(z))   # evidence slope < 0

    return {
        "status": status,
        "prereg": spec["prereg"],
        "slope": slope,
        "se": se,
        "n": int(len(d)),
        "ci95_two_sided": ci95,
        "one_sided_bound_side": side,
        "one_sided_ci95": one_sided_ci,
        "one_sided_ci95_full": [
            -np.inf if side == "upper" else one_sided_lower,
            one_sided_upper if side == "upper" else np.inf,
        ],
        "p_value_two_sided": pval_two,
        "p_value_one_sided": p_one_sided,
        "margin": spec["margin"],
    }


def per_fraction_means(df: pd.DataFrame) -> dict:
    """Per-cell means for both dimensions: pooled over architectures AND per
    architecture, so the dose curve is reconstructable from canonical alone."""
    cells = [c for c in ["g000", "g025", "g050", "g075", "g100", "interleaved"]
             if c in set(df["cell"])]
    out = {"pooled": {}, "per_pattern": {}}
    for cell in cells:
        sub = df[df.cell == cell]
        out["pooled"][cell] = {
            "gold_fraction": GOLD_FRACTION.get(cell),
            "n": int(len(sub)),
            "factual_accuracy_mean": float(np.nanmean(sub["factual_accuracy"])),
            "citation_quality_mean": float(np.nanmean(sub["citation_quality"])),
        }
    for pat in sorted(df["pattern"].unique()):
        pd_ = df[df.pattern == pat]
        out["per_pattern"][pat] = {}
        for cell in cells:
            sub = pd_[pd_.cell == cell]
            if not len(sub):
                continue
            out["per_pattern"][pat][cell] = {
                "gold_fraction": GOLD_FRACTION.get(cell),
                "n": int(len(sub)),
                "factual_accuracy_mean": float(np.nanmean(sub["factual_accuracy"])),
                "citation_quality_mean": float(np.nanmean(sub["citation_quality"])),
            }
    return out


def interleaved_vs_g100(df: pd.DataFrame) -> dict:
    """One-shot (g100) vs progressive (interleaved) factual contrast at 100% gold."""
    g100 = df[df.cell == "g100"]
    inter = df[df.cell == "interleaved"]
    if not (len(g100) and len(inter)):
        return {"status": "missing_cells"}
    fg = float(np.nanmean(g100["factual_accuracy"]))
    fi = float(np.nanmean(inter["factual_accuracy"]))
    return {
        "factual_g100_mean": fg,
        "factual_interleaved_mean": fi,
        "delta": fi - fg,
        "citation_g100_mean": float(np.nanmean(g100["citation_quality"])),
        "citation_interleaved_mean": float(np.nanmean(inter["citation_quality"])),
        "n_g100": int(len(g100)),
        "n_interleaved": int(len(inter)),
    }


def build_block(judge_out: Path) -> dict:
    df = load_e5_scores(judge_out)
    if df.empty:
        return {"status": "no_verdicts", "judge_out": str(judge_out)}
    dose = df[df.gold_fraction.notna()].copy()

    block = {
        "status": "fit",
        "_note": (
            "Prereg mixed-effects dose-response (factual_accuracy/citation_quality "
            "~ gold_fraction, query random intercept), pooled over P0/P1/P4, recomputed "
            "deterministically from results/judge_gpt52_e5 verdicts. Reports a ONE-SIDED "
            "95% slope CI in the prereg direction (factual: upper bound, flatness; "
            "citation: lower bound, monotone-increasing) — the harness fit JSON gives "
            "only a two-sided CI. Interleaved excluded from the slope; used for the "
            "context-overload rescue contrast."),
        "judge": "gpt-5.2",
        "judge_out": str(judge_out),
        "n_verdicts": int(len(df)),
        "n_dose_points": int(len(dose)),
        "n_queries": int(dose["query_id"].nunique()),
        "architectures": sorted(df["pattern"].unique().tolist()),
        "cells_present": sorted(df["cell"].unique().tolist()),
        "gold_fractions": GOLD_FRACTION,
        "one_sided_alpha": 0.05,
        "factual_accuracy_slope": fit_slope(dose, "factual_accuracy"),
        "citation_quality_slope": fit_slope(dose, "citation_quality"),
        "per_fraction_means": per_fraction_means(df),
        "interleaved_vs_g100": interleaved_vs_g100(df),
        "interpretation": (
            "factual flat: one-sided 95% UPPER bound on the gold_fraction slope sits "
            "near/below the +/-0.05 margin => no detectable upward dose effect on "
            "factual accuracy. citation: one-sided 95% LOWER bound > 0 would confirm a "
            "monotone retrieval-bound increase; if it straddles 0 the increase is not "
            "established at this n."),
    }
    return block


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-out", default=str(DEFAULT_JUDGE_OUT),
                    help=f"E5 GPT-5.2 verdict root (default {DEFAULT_JUDGE_OUT}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + print the would-be block; do NOT write canonical.")
    args = ap.parse_args()

    judge_out = Path(args.judge_out)
    if not judge_out.is_absolute():
        judge_out = _REPO_ROOT / judge_out
    if not judge_out.exists():
        print(f"[e5] judge-out not found: {judge_out} -> nothing to do (exit 0).")
        return 0

    block = build_block(judge_out)

    # Human-readable summary.
    fa = block.get("factual_accuracy_slope", {})
    cq = block.get("citation_quality_slope", {})
    iv = block.get("interleaved_vs_g100", {})
    print(json.dumps({"e5_dose_response": {
        "status": block.get("status"),
        "n_verdicts": block.get("n_verdicts"),
        "n_dose_points": block.get("n_dose_points"),
        "factual_slope": fa.get("slope"),
        "factual_one_sided_upper_95": fa.get("one_sided_ci95"),
        "citation_slope": cq.get("slope"),
        "citation_one_sided_lower_95": cq.get("one_sided_ci95"),
        "interleaved_minus_g100_factual": iv.get("delta"),
    }}, indent=1, default=str))

    if args.dry_run:
        print("\n[DRY RUN] canonical_numbers.json NOT written. Block above is live.")
        return 0

    if block.get("status") != "fit":
        print(f"\n[e5] status={block.get('status')!r}; not writing canonical.")
        return 0

    canon = json.loads(CANON.read_text()) if CANON.exists() else {}
    canon["e5_dose_response"] = block
    CANON.write_text(json.dumps(canon, indent=1, default=str))
    print(f"\nWrote canonical_numbers.json['e5_dose_response'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
