#!/usr/bin/env python
"""Pattern x source LR df audit + judge-averaged 11-pattern refit (carried-metric fix).

The paper quotes "likelihood-ratio 202.3, df 44" (carried_metrics.lr_pattern_x_source,
pinned to reports/phase2_statistics/04_stratification.md) in a section about ELEVEN
patterns, but df 44 = (12-1)(5-1) implies TWELVE patterns. Provenance audit
(scripts/phase2_statistical_analysis.py): the original fit used
  * agg_overall_base = JUDGE-AVERAGED (pattern x query) rows -- mean of the 'trustworthy'
    overall across every judge present in df_overall_scores at the time (NOT judge-level
    rows, so the pseudo-replication concern does not apply), and
  * BASE_PATTERNS = every pattern matching base_* then present = base_p0..base_p11,
    i.e. the 12th pattern is base_p11 (the verbatim ReAct controller, a post-hoc probe
    the paper treats as single-judge by design; base_p12 was not yet judged). The
    difficulty LR df=22 = (12-1)(3-1) confirms the same basis.
This builder refits the mixed-model LRT (mixedlm, query_id random intercept, ML,
full: overall ~ C(pattern)*C(source) vs null: overall ~ C(pattern)+C(source); identical
machinery to phase2) on:
  1. eleven_pattern  (THE citable refit): base_p0..base_p10, canonical 3-judge panel
     (gpt52, claude_opus, claude_sonnet; sonnet -> overall_score_recomputed, the same
     'ovc' treatment as build_pairwise.py), judge-averaged pattern x query rows.
  2. twelve_pattern_repro: same modern basis plus base_p11 (df should return to 44).
  3. phase2_basis_repro: the original recipe verbatim (trustworthy-overall, ALL judges
     in the parquet averaged, 12 patterns) to confirm 202.25/44 provenance.
Also refits pattern x difficulty on basis (1) since the carried 35.67/df22 has the same
12-pattern origin. Deterministic (ML fits, no randomness).

APPEND-ONLY: lands the NEW TOP-LEVEL key canonical_numbers.json['lr_pattern_x_source_refit']
(a subkey inside carried_metrics would be clobbered by a build_carried_metrics.py rerun,
which rewrites that key wholesale); refuses to overwrite; atomic tempfile+replace.

Usage: python build_lr_pattern_source_refit.py [--write] [--force]
"""
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
KEY = "lr_pattern_x_source_refit"
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
P11SET = [f"base_p{i}" for i in range(11)]            # base_p0..base_p10
P12SET = P11SET + ["base_p11"]

ov = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
qs = pd.read_parquet(f"{ROOT}/data/analysis/df_queries.parquet")
ov["pattern"] = ov["pattern"].astype(str)
ov["judge"] = ov["judge"].astype(str)

# canonical treatment (build_pairwise.py): sonnet -> recomputed
ov["ovc"] = ov["overall_score"].where(~ov.judge.eq("claude_sonnet"),
                                      ov.get("overall_score_recomputed"))
# phase2 treatment: trustworthy ? overall_score : overall_score_recomputed
ov["overall_p2"] = np.where(ov["overall_score_trustworthy"],
                            ov["overall_score"], ov["overall_score_recomputed"])


def agg_rows(patterns, judges, col):
    d = ov[ov.pattern.isin(patterns)]
    if judges is not None:
        d = d[d.judge.isin(judges)]
    d = d.dropna(subset=[col])
    a = d.groupby(["pattern", "query_id"], observed=True)[col].mean().reset_index()
    a = a.merge(qs[["query_id", "source", "difficulty"]], on="query_id", how="left")
    a = a.rename(columns={col: "overall"})
    for c in ("pattern", "source", "difficulty"):
        if str(a[c].dtype) == "category":
            a[c] = a[c].astype(str)
    return a.dropna(subset=["overall", "source"])


def _fit_mixed(formula, df):
    m = smf.mixedlm(formula, df, groups=df["query_id"])
    last = None
    for method in (["lbfgs"], ["bfgs"], ["powell"], ["nm"]):
        try:
            r = m.fit(reml=False, method=method)
            if np.isfinite(r.llf):
                return r
        except Exception as e:
            last = e
    raise RuntimeError(f"mixedlm failed: {last}")


def lrt(df, factor):
    full = _fit_mixed(f"overall ~ C(pattern) * C({factor})", df)
    null = _fit_mixed(f"overall ~ C(pattern) + C({factor})", df)
    lr = 2 * (full.llf - null.llf)
    k = len(full.fe_params) - len(null.fe_params)
    p = float(1 - stats.chi2.cdf(lr, df=max(k, 1)))
    return {"lr": round(float(lr), 2), "df": int(k),
            "p": (round(p, 6) if p > 1e-12 else 0.0),
            "n_rows": int(len(df)),
            "n_patterns": int(df.pattern.nunique()),
            "n_queries": int(df.query_id.nunique())}


# 1. citable refit: 11 patterns, canonical 3-judge ovc, judge-averaged
a11 = agg_rows(P11SET, PANEL, "ovc")
fit11 = lrt(a11, "source")
fit11_diff = lrt(a11, "difficulty")

# 2. 12-pattern repro on the modern canonical basis
a12 = agg_rows(P12SET, PANEL, "ovc")
fit12 = lrt(a12, "source")

# 3. phase2-recipe repro (trustworthy overall, ALL judges averaged, 12 patterns)
a12_p2 = agg_rows(P12SET, None, "overall_p2")
judges_p2 = sorted(ov[ov.pattern.isin(P12SET)].judge.unique())
fit12_p2 = lrt(a12_p2, "source")
fit12_p2_diff = lrt(a12_p2, "difficulty")

out = {
    "_note": ("Audit + refit of carried_metrics.lr_pattern_x_source ('202.3, df 44'). "
              "PROVENANCE: the original phase2 fit (scripts/phase2_statistical_analysis.py "
              "-> reports/phase2_statistics/04_stratification.md) was on JUDGE-AVERAGED "
              "(pattern x query) rows -- NOT judge-level rows, so no pseudo-replication -- "
              "but over TWELVE patterns: base_p0..base_p10 PLUS base_p11 (verbatim ReAct "
              "controller, a post-hoc probe the paper elsewhere treats as single-judge by "
              "design; df 44 = (12-1)(5-1); the difficulty df 22 = (12-1)(3-1) confirms). "
              "The paper's eleven-pattern section should quote 'eleven_pattern' "
              "below. Machinery identical to phase2: mixedlm with query_id random "
              "intercept, ML, LRT full C(pattern)*C(source) vs null C(pattern)+C(source)."),
    "provenance_finding": {
        "original_row_basis": "judge-averaged pattern x query rows (no pseudo-replication)",
        "original_pattern_count": 12,
        "twelfth_pattern": "base_p11 (verbatim ReAct controller; a post-hoc probe the "
                           "paper treats as single-judge by design, a second reason it "
                           "does not belong in the panel-averaged eleven-pattern family)",
        "original_judge_set": "all judges present in df_overall_scores at phase2 time, "
                              "trustworthy-overall treatment",
    },
    "eleven_pattern": {
        **fit11,
        "basis": "base_p0..base_p10; canonical 3-judge panel (gpt52, claude_opus, "
                 "claude_sonnet; sonnet ovc-recomputed); judge-averaged pattern x query rows",
        "expected_df": 40,
    },
    "eleven_pattern_difficulty": {
        **fit11_diff,
        "basis": "same as eleven_pattern; refit of the carried 35.67/df22 companion",
        "expected_df": 20,
    },
    "twelve_pattern_repro": {
        **fit12,
        "basis": "eleven_pattern basis + base_p11 (df returns to 44)",
    },
    "phase2_basis_repro": {
        **fit12_p2,
        "difficulty": fit12_p2_diff,
        "basis": f"original recipe: trustworthy-overall, ALL parquet judges {judges_p2} "
                 "averaged, 12 patterns",
        "carried_value": {"lr": 202.25, "df": 44},
    },
}

print(json.dumps(out, indent=1))

if "--write" in sys.argv:
    cn = json.load(open(CANON))  # fresh read: keep read-modify-write window short
    if KEY in cn and "--force" not in sys.argv:
        print(f"[REFUSING] '{KEY}' already in store (use --force)")
        sys.exit(1)
    cn[KEY] = out
    fd, tmp = tempfile.mkstemp(dir=ANA, prefix="canonical_numbers.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[WROTE canonical_numbers.json['{KEY}'] (store -> {len(cn)} top-level keys)]")
else:
    print("[DRY-RUN: no write; pass --write to land the key]")
