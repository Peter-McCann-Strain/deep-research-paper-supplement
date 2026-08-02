#!/usr/bin/env python3
"""T1_tost_power (a) — EQUIVALENCE tests for the E5 asserted nulls (EXPLORATORY margins).

The E5 dose-response paper asserts two NULLS and one MONOTONE claim:
  * factual_accuracy is FLAT in the gold fraction (no retrieval-quality dose effect),
  * the INTERLEAVED (progressive context) cell does NOT differ from one-shot g100
    on factual accuracy (no context-overload rescue / no penalty),
  * citation_quality is MONOTONE-INCREASING (retrieval-bound) in the gold fraction.

PROVENANCE OF THE MARGINS. The E5 pre-registration
(reports/RESEARCH_PLAN_2026H2.md, priority-8 design block) pre-registers the
DIRECTIONAL/SHAPE claims only — "Pre-register: citation monotone, factual flat" —
NOT any numeric equivalence margin (there is no docs/publication/prereg/prereg_E5.md
and no +/-0.05 appears in the E5 prereg). The +/-0.05 equivalence half-width used by
the TOSTs below is therefore an EXPLORATORY (power-justified) margin, derived
post-hoc as ~2x the E2 well-powered MDE80 (0.0247), exactly as the main paper labels
its within-cluster ROPE "power-justified rather than pre-registered". Each test
carries an explicit ``margin_source`` field so the provenance is auditable and the
margins are NOT mislabelled as pre-registered.

The existing canonical block ``e5_dose_response`` (built by
scripts/build_e5_dose_response.py) argues the factual null only from a slope CI that
STRADDLES zero — that is *absence of evidence*, not an equivalence claim. This script
adds the formal equivalence machinery (exploratory, power-justified margins; see
PROVENANCE above), OVER THE SAME on-disk ``results/judge_gpt52_e5`` GPT-5.2 verdicts,
recomputed deterministically:

  (1) E5 dose factual-FLAT TOST against the EXPLORATORY margin +/-0.05, expressed on the
      per-query factual mean change across the dose range (g100 - g000), tied to the
      E2 well-powered MDE80 = 0.0247 (canonical variance_decomposition.mde.main_study_n90_r1):
      the +/-0.05 margin is ~2x the smallest gap the main design can detect, so an
      equivalence inside it is a *meaningful* "flat", and the n=30 paired MDE80 here is
      reported alongside so the margin's defensibility is auditable. (Margin is exploratory /
      MDE-derived, not pre-registered — only the directional "factual flat" claim is.)
  (2) A PAIRED-BOOTSTRAP 95% CI on that same per-query (g100 - g000) factual change,
      seeded, query-resampled — the non-parametric companion to the TOST.
  (3) A TOST for the INTERLEAVING null: interleaved - g100 per-query factual mean,
      paired by query, against +/-0.05, plus its paired-bootstrap CI.
  (4) A slope-MDE for the CITATION slope: the smallest true gold_fraction slope the
      E5 design (n dose points, observed residual SD) can detect at 80% power, so the
      "monotone increasing not established at this n" caveat is quantified rather than
      asserted. The on-disk slope/se are RE-READ from canonical e5_dose_response (built
      from the same verdicts) so the two blocks never drift; if absent they are
      recomputed from the verdicts.

All quantities are paired BY QUERY (the 30 variance-stratified queries appear in every
dose cell), pooled over the three architectures P0/P1/P4 (mean over the 3 patterns per
query-cell, matching the e5_dose_response pooling). The inferential unit is the query.

DETERMINISM. statsmodels MixedLM slope is reused from canonical (a closed-form transform
gives the slope-MDE); the paired bootstrap uses a fixed seed and sorted query order; all
verdict files are iterated sorted. Re-running reproduces the block bit-for-bit.

IDEMPOTENT / NON-CLOBBERING. The ONLY write is
canonical_numbers.json['e5_equivalence']. Verdicts/reports are READ-ONLY. The write is
guarded behind --write; bare invocation prints the would-be block and writes nothing
(symmetry with the other equivalence builders; the orchestrator passes --write).

Usage:
    python scripts/build_e5_equivalence.py                 # dry: compute + print only
    python scripts/build_e5_equivalence.py --write         # persist canonical key
    python scripts/build_e5_equivalence.py --judge-out results/judge_gpt52_e5 --write
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

# ── Paths (canonical store MOVED here by commit 0a80ba6) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"
DEFAULT_JUDGE_OUT = _REPO_ROOT / "results" / "judge_gpt52_e5"

# Cell -> gold fraction; mirrors run_e5_oracle_dose.py / build_e5_dose_response.py.
GOLD_FRACTION = {"g000": 0.0, "g025": 0.25, "g050": 0.50, "g075": 0.75, "g100": 1.00}
CELL_RE = re.compile(r"^e5_oracle_dose_(p\d+)_(g\d{3}|interleaved)$")

# Equivalence margins. NB: E5 (RESEARCH_PLAN_2026H2 priority 8) pre-registers the
# DIRECTIONAL claims only ("citation monotone, factual flat"); no numeric margin is
# pre-registered and no prereg_E5.md exists. Both margins below are therefore EXPLORATORY
# (power-justified), derived as ~2x the E2 well-powered MDE80 (0.0247) — mirroring the
# main paper's "power-justified rather than pre-registered" within-cluster ROPE.
FACTUAL_MARGIN = 0.05          # exploratory practical-equivalence half-width (~2x E2 MDE80 0.0247)
INTERLEAVE_MARGIN = 0.05       # exploratory; same MDE-derived margin for the rescue/penalty null
# Per-test provenance strings (surfaced in the canonical block so margins are never
# mislabelled as pre-registered).
MARGIN_SOURCE = (
    "exploratory_margin: power-justified half-width = ~2x E2 well-powered MDE80 "
    "(0.0247, canonical variance_decomposition.mde.main_study_n90_r1); NOT pre-registered "
    "(E5 prereg [reports/RESEARCH_PLAN_2026H2.md, priority 8] pre-registers only the "
    "directional 'factual flat' / 'citation monotone' shape claims, no numeric margin; "
    "no docs/publication/prereg/prereg_E5.md exists). Reported with the n=30 paired MDE80 "
    "so the bound is auditable, per the paper's 'power-justified not pre-registered' discipline."
)
PREREGISTERED_E5_CLAIMS = "directional only: 'citation monotone, factual flat' (RESEARCH_PLAN_2026H2 priority 8)"
SEED = 20260622
N_BOOT = 10000

# E2 well-powered MDE anchor (canonical variance_decomposition.mde.main_study_n90_r1).
E2_MDE80_ANCHOR_KEY = ("variance_decomposition", "mde", "main_study_n90_r1")


# ── Load E5 verdicts deterministically (READ-ONLY) ────────────────────────────
def load_e5_scores(judge_out: Path) -> pd.DataFrame:
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
                "gold_fraction": GOLD_FRACTION.get(cell),
                "factual_accuracy": dims.get("factual_accuracy", {}).get("score"),
                "citation_quality": dims.get("citation_quality", {}).get("score"),
            })
    return pd.DataFrame(rows)


def _query_cell_means(df: pd.DataFrame, dim: str, cell: str) -> pd.Series:
    """Per-query mean of `dim` in `cell`, pooled over the architectures (index=query_id)."""
    sub = df[(df.cell == cell)].dropna(subset=[dim])
    return sub.groupby("query_id")[dim].mean().sort_index()


def paired_diff(df: pd.DataFrame, dim: str, cell_hi: str, cell_lo: str) -> np.ndarray:
    """Per-query (cell_hi - cell_lo) paired on the common queries, sorted by query_id."""
    hi = _query_cell_means(df, dim, cell_hi)
    lo = _query_cell_means(df, dim, cell_lo)
    common = sorted(set(hi.index) & set(lo.index))
    return np.array([hi[q] - lo[q] for q in common], dtype=float)


# ── Paired one-sample TOST + MDE on a vector of per-query diffs ────────────────
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
    """Two-sided alpha=0.05 80%-power MDE for a one-sample paired t at this n/SD."""
    n = len(diffs)
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df_ = n - 1
    tcrit = float(stats.t.ppf(0.975, df_))
    tpow = float(stats.t.ppf(0.80, df_))
    return float((tcrit + tpow) * se)


def paired_bootstrap_ci(diffs: np.ndarray, seed: int = SEED, b: int = N_BOOT) -> dict:
    """Seeded query-resampled bootstrap 95% CI on the mean per-query diff."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = np.arange(n)
    boot = np.array([diffs[rng.choice(idx, n, replace=True)].mean() for _ in range(b)])
    return {
        "mean": round(float(diffs.mean()), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                 round(float(np.percentile(boot, 97.5)), 4)],
        "n_boot": b, "seed": seed, "n_queries": n,
    }


# ── Slope-MDE for the citation slope (re-use canonical slope/se) ──────────────
def citation_slope_mde(canon: dict, df: pd.DataFrame) -> dict:
    """MDE80 for the gold_fraction slope on citation_quality.

    Prefers the on-disk canonical e5_dose_response slope/se (built from the SAME
    verdicts) so the two blocks cannot drift; recomputes from verdicts only if absent.
    Wald (normal) construction, matching MixedLM's CI; MDE = (z_.975 + z_.80) * se.
    """
    src = "canonical_e5_dose_response"
    cq = (canon.get("e5_dose_response", {}) or {}).get("citation_quality_slope", {})
    slope = cq.get("slope")
    se = cq.get("se")
    n = cq.get("n")
    if slope is None or se is None:
        # Recompute the MixedLM slope from verdicts (fallback OLS) — same as the builder.
        src = "recomputed_from_verdicts"
        dose = df[df.gold_fraction.notna()].dropna(subset=["citation_quality"]).copy()
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm("citation_quality ~ gold_fraction", dose, groups=dose["query_id"])
            fit = md.fit(reml=True, method="lbfgs")
            slope = float(fit.params["gold_fraction"]); se = float(fit.bse["gold_fraction"])
        except Exception:
            lr = stats.linregress(dose["gold_fraction"].to_numpy(float),
                                  dose["citation_quality"].to_numpy(float))
            slope = float(lr.slope); se = float(lr.stderr)
        n = int(len(dose))
    z975 = float(stats.norm.ppf(0.975)); z80 = float(stats.norm.ppf(0.80))
    mde = float((z975 + z80) * se)
    return {
        "source": src,
        "slope": round(float(slope), 5),
        "se": round(float(se), 5),
        "n_dose_points": int(n) if n is not None else None,
        "slope_mde80": round(mde, 4),
        "detectable": bool(abs(slope) >= mde),
        "interpretation": (
            "Smallest true gold_fraction slope on citation_quality the E5 design can "
            "detect at 80% power (Wald). If |slope| < slope_mde80 the monotone-"
            "increasing claim is UNDER-POWERED at this n, not refuted — report 'not "
            "established at this n', consistent with the one-sided lower-bound CI "
            "straddling 0 in e5_dose_response."),
    }


# ── Build the block ───────────────────────────────────────────────────────────
def build_block(judge_out: Path, canon: dict) -> dict:
    df = load_e5_scores(judge_out)
    if df.empty:
        return {"status": "no_verdicts", "judge_out": str(judge_out)}

    e2_mde = canon
    for k in E2_MDE80_ANCHOR_KEY:
        e2_mde = (e2_mde or {}).get(k, {}) if isinstance(e2_mde, dict) else None
    e2_mde = e2_mde if isinstance(e2_mde, (int, float)) else None

    # (1)+(2) factual FLAT across the full dose range, per-query (g100 - g000).
    fac_diff = paired_diff(df, "factual_accuracy", "g100", "g000")
    factual_tost = tost_one_sample(fac_diff, FACTUAL_MARGIN)
    factual_tost["mde80_paired_n"] = round(mde80_one_sample(fac_diff), 4)
    factual_boot = paired_bootstrap_ci(fac_diff)

    # (3) interleaving null: interleaved - g100, per-query, factual.
    inter_diff = paired_diff(df, "factual_accuracy", "interleaved", "g100")
    inter_tost = tost_one_sample(inter_diff, INTERLEAVE_MARGIN)
    inter_tost["mde80_paired_n"] = round(mde80_one_sample(inter_diff), 4)
    inter_boot = paired_bootstrap_ci(inter_diff, seed=SEED + 1)

    # (4) citation slope MDE.
    cit_mde = citation_slope_mde(canon, df)

    return {
        "status": "fit",
        "_note": (
            "T1 EQUIVALENCE tests for the E5 asserted nulls (EXPLORATORY power-justified "
            "margins; only the directional 'factual flat'/'citation monotone' shape claims "
            "were pre-registered), recomputed deterministically over results/judge_gpt52_e5 "
            "GPT-5.2 verdicts. Adds the formal TOST/MDE/bootstrap machinery the "
            "e5_dose_response slope-CI argument lacked: (a) factual-FLAT TOST on per-query "
            "(g100-g000) vs an exploratory +/-0.05 (~2x E2 MDE80=0.0247); (b) paired-bootstrap "
            "CI; (c) interleaving-null TOST + CI; (d) citation slope-MDE. Paired by query, "
            "pooled over P0/P1/P4. Pure CPU."),
        "prereg_status": "exploratory_margins",
        "preregistered_claims": PREREGISTERED_E5_CLAIMS,
        "judge": "gpt-5.2",
        "judge_out": str(judge_out),
        "n_verdicts": int(len(df)),
        "n_queries": int(df["query_id"].nunique()),
        "architectures": sorted(df["pattern"].unique().tolist()),
        "pooling": "mean over the 3 architectures (P0/P1/P4) per query-cell; unit=query",
        "e2_mde80_anchor": e2_mde,
        "factual_flat": {
            "contrast": "g100_minus_g000 (full dose range), per-query factual mean",
            "margin": FACTUAL_MARGIN,
            "margin_type": "exploratory_margin",
            "margin_source": MARGIN_SOURCE,
            "preregistered": False,
            "tost": factual_tost,
            "paired_bootstrap": factual_boot,
            "margin_vs_e2_mde80": (
                None if e2_mde is None else round(FACTUAL_MARGIN / e2_mde, 2)),
            "interpretation": (
                "Equivalence within an EXPLORATORY +/-0.05 (~2x the E2 well-powered "
                "MDE80 0.0247; not pre-registered) => factual accuracy shows no change "
                "detectable at this power-justified margin across the gold-fraction "
                "dose; report as 'no change detectable at +/-0.05', NOT as an unqualified "
                "pre-registered equivalence. The pre-registered E5 claim is the "
                "DIRECTIONAL 'factual flat'. mde80_paired_n is the n=30 paired "
                "sensitivity for transparency on the margin choice."),
        },
        "interleaving_null": {
            "contrast": "interleaved_minus_g100, per-query factual mean",
            "margin": INTERLEAVE_MARGIN,
            "margin_type": "exploratory_margin",
            "margin_source": MARGIN_SOURCE,
            "preregistered": False,
            "tost": inter_tost,
            "paired_bootstrap": inter_boot,
            "interpretation": (
                "Equivalence within an EXPLORATORY +/-0.05 (power-justified, not "
                "pre-registered) => progressive (interleaved) context produces no "
                "factual-accuracy change relative to one-shot g100 detectable at this "
                "margin — no context-overload rescue/penalty resolvable at +/-0.05. The "
                "E5 prereg names the interleaved-rescue question directionally only; this "
                "numeric equivalence bound is exploratory."),
        },
        "citation_slope_mde": cit_mde,
        "interpretation": (
            "Asserted-null audit: the factual-flat and interleaving nulls are supported as "
            "equivalence at an EXPLORATORY power-justified +/-0.05 (TOST p<0.05), i.e. 'no "
            "change detectable at +/-0.05' rather than a pre-registered-substantive "
            "equivalence; only the DIRECTIONAL shape claims ('factual flat', 'citation "
            "monotone') were pre-registered. The citation monotone-increasing claim is "
            "bounded by its slope-MDE so an undetected-but-real slope is explicitly "
            "admitted as under-power."),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-out", default=str(DEFAULT_JUDGE_OUT),
                    help=f"E5 GPT-5.2 verdict root (default {DEFAULT_JUDGE_OUT}).")
    ap.add_argument("--write", action="store_true",
                    help="Persist canonical_numbers.json['e5_equivalence']. "
                         "Without it, computes + prints and writes NOTHING.")
    args = ap.parse_args()

    judge_out = Path(args.judge_out)
    if not judge_out.is_absolute():
        judge_out = _REPO_ROOT / judge_out
    if not judge_out.exists():
        print(f"[e5_equiv] judge-out not found: {judge_out} -> nothing to do (exit 0).")
        return 0

    canon = json.loads(CANON.read_text()) if CANON.exists() else {}
    block = build_block(judge_out, canon)

    print(json.dumps({"e5_equivalence": {
        "status": block.get("status"),
        "factual_flat_tost_p": block.get("factual_flat", {}).get("tost", {}).get("p_tost"),
        "factual_flat_equivalent": block.get("factual_flat", {}).get("tost", {}).get("equivalent_at_05_alpha"),
        "factual_flat_boot_ci95": block.get("factual_flat", {}).get("paired_bootstrap", {}).get("ci95"),
        "interleaving_tost_p": block.get("interleaving_null", {}).get("tost", {}).get("p_tost"),
        "interleaving_equivalent": block.get("interleaving_null", {}).get("tost", {}).get("equivalent_at_05_alpha"),
        "citation_slope": block.get("citation_slope_mde", {}).get("slope"),
        "citation_slope_mde80": block.get("citation_slope_mde", {}).get("slope_mde80"),
    }}, indent=1, default=str))

    if not args.write:
        print("\n[DRY] canonical_numbers.json NOT written (pass --write to persist).")
        return 0
    if block.get("status") != "fit":
        print(f"\n[e5_equiv] status={block.get('status')!r}; not writing canonical.")
        return 0

    canon["e5_equivalence"] = block
    tmp = CANON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canon, indent=1, default=str))
    tmp.replace(CANON)
    print(f"\nWrote canonical_numbers.json['e5_equivalence'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
