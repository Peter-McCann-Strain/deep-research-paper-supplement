#!/usr/bin/env python3
"""E4 CITE-CAUSAL — Step 6: causal analysis of citation density on factual-accuracy
verdicts, per judge family, with a Claude-vs-OpenAI interaction test, plus CoT mining of
the baseline 248k verdict corpus to localise where the (hypothesised) bias enters.

Clones the modelling pattern of ``build_citation_regression.py`` but on the E4
content-FIXED citation-perturbed re-judgements (so a density effect here is CAUSAL: the
prose is identical across conditions, only citation tokens move).

Design (per condition C0..C4, per judge):
  density_index:  C1=-2 (strip), C2=-1 (halve), C0=0 (orig), C3=+1 (double), C4=0 (shuffle;
                  density unchanged, mapping scrambled -> separates "more markers" from
                  "more CORRECT markers").
  Mixed-effects:  dim_score ~ density_index + is_shuffle + C(judge_family) + judge_family:density_index
                  + (1 | query_id)   [query random intercept]
  -> The judge_family:density_index interaction tests whether the density->score slope is
     Claude-specific (H1) vs shared with OpenAI.

Endpoints: factual_accuracy (primary), citation_quality, attribution_quality.

CoT mining (READ-ONLY over data/analysis/df_verdicts.parquet, the 248k baseline corpus):
  - For factual_accuracy verdicts, count how often the judge's `reasoning`/`evidence`
    explicitly references citations/sources ("cite", "source", "reference", "[", ...).
  - Compare that citation-mention rate across judge families: if Claude mentions citations
    when justifying factual verdicts far more than GPT-5.2, that localises the leakage.

Outputs:
  - canonical_numbers.json['e4_cite_causal']   (also writes if E4 judge data absent: the
    CoT-mining sub-block always populates from the baseline corpus; the causal sub-block
    is marked pending until the E4 re-judge runs land).
  - prints a human summary.

This script is READ-ONLY on every protected corpus path. The only write is the canonical
JSON (an analysis artefact, not a protected path).

Usage:
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e4_cite_causal.py
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e4_cite_causal.py --dry-run   # CoT-mining
        only; never writes canonical (prints would-be block).
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

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

A = _REPO_ROOT / "data" / "analysis"                       # READ-ONLY parquets
ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

# E4 re-judge outputs (produced later by the human-launched paid runs).
E4_JUDGE_DIRS = {
    "gpt52":         _REPO_ROOT / "results" / "judge_gpt52_e4",
    "gpt-4.1":       _REPO_ROOT / "results" / "judge_gpt41_e4",
    "gpt-4o":        _REPO_ROOT / "results" / "judge_gpt4o_e4",
    "claude_opus":   _REPO_ROOT / "results" / "judge_claude_opus_e4",
    "claude_sonnet": _REPO_ROOT / "results" / "judge_claude_sonnet_e4",
}
JUDGE_FAMILY = {
    "gpt52": "openai", "gpt-4.1": "openai", "gpt-4o": "openai",
    "claude_opus": "claude", "claude_sonnet": "claude",
}
DENSITY_INDEX = {"C1": -2, "C2": -1, "C0": 0, "C3": 1, "C4": 0}
IS_SHUFFLE = {"C1": 0, "C2": 0, "C0": 0, "C3": 0, "C4": 1}
DIMS = ["factual_accuracy", "citation_quality", "attribution_quality"]
CITE_WORDS = re.compile(r"\b(cit\w*|sources?|references?|provenance|attribut\w*|"
                        r"\[\d+\]|grounded|unsourced|uncited)\b", re.I)


# ── Load E4 re-judge verdicts (if present) ────────────────────────────────────

def load_e4_scores() -> pd.DataFrame:
    """Walk judge_*_e4/{condition}/{pattern}/{qid}.json into a tidy frame."""
    rows = []
    for judge, root in E4_JUDGE_DIRS.items():
        if not root.exists():
            continue
        for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cond = cond_dir.name
            for pat_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                pattern = pat_dir.name
                for jf in sorted(pat_dir.glob("*.json")):
                    try:
                        d = json.loads(jf.read_text())
                    except Exception:
                        continue
                    dims = d.get("dimensions", {})
                    for dim in DIMS:
                        if dim in dims:
                            rows.append({
                                "judge": judge, "judge_family": JUDGE_FAMILY[judge],
                                "condition": cond, "pattern": pattern,
                                "query_id": jf.stem, "dimension": dim,
                                "score": dims[dim]["score"],
                                "density_index": DENSITY_INDEX.get(cond, 0),
                                "is_shuffle": IS_SHUFFLE.get(cond, 0),
                            })
    return pd.DataFrame(rows)


def fit_causal(df: pd.DataFrame, dim: str) -> dict:
    import statsmodels.formula.api as smf
    d = df[(df.dimension == dim)].copy()
    # Drop near-null p10 reports from the density contrast (their transforms are no-ops).
    d = d[d.pattern != "base_p10__nearnull"]
    if d.empty or d.judge_family.nunique() < 1:
        return {"status": "pending", "note": "E4 re-judge data not yet present"}
    d["judge_family"] = d["judge_family"].astype("category")
    out = {"status": "fit", "n_obs": int(len(d)),
           "judges": sorted(d.judge.unique().tolist()),
           "conditions": sorted(d.condition.unique().tolist())}
    # Per-judge-family density slope (separate OLS for interpretability).
    per_family = {}
    for fam in sorted(d.judge_family.unique()):
        sub = d[d.judge_family == fam]
        if sub.query_id.nunique() < 3:
            continue
        try:
            m = smf.ols("score ~ density_index + is_shuffle + C(pattern)", data=sub).fit()
            per_family[fam] = {
                "beta_density": round(float(m.params.get("density_index", np.nan)), 4),
                "p_density": float(m.pvalues.get("density_index", np.nan)),
                "beta_shuffle": round(float(m.params.get("is_shuffle", np.nan)), 4),
                "p_shuffle": float(m.pvalues.get("is_shuffle", np.nan)),
                "n": int(len(sub))}
        except Exception as e:
            per_family[fam] = {"error": str(e)[:120]}
    out["per_family_density_slope"] = per_family
    # Interaction: is the density slope Claude-specific? Mixed model w/ query RE.
    if d.judge_family.nunique() >= 2 and d.query_id.nunique() >= 3:
        try:
            md = smf.mixedlm(
                "score ~ density_index * C(judge_family) + is_shuffle + C(pattern)",
                data=d, groups=d["query_id"])
            mf = md.fit(reml=False, method="lbfgs")
            inter = [k for k in mf.params.index if "density_index:" in k]
            out["interaction"] = {
                k: {"beta": round(float(mf.params[k]), 4), "p": float(mf.pvalues[k])}
                for k in inter}
            out["interaction_note"] = (
                "density_index:C(judge_family) coefficient = how much the density->score "
                "slope DIFFERS from the reference family. Significant + positive for claude "
                "=> the density effect is Claude-specific (H1).")
        except Exception as e:
            out["interaction"] = {"error": str(e)[:150]}
    return out


# ── CoT mining over the baseline 248k corpus (READ-ONLY) ──────────────────────

def cot_mining() -> dict:
    """How often do judges invoke citations/sources when justifying factual verdicts?

    A Claude-specific elevation localises where citation density leaks into the
    factual-accuracy channel. READ-ONLY over df_verdicts.parquet."""
    vp = A / "df_verdicts.parquet"
    if not vp.exists():
        return {"status": "no_verdicts_parquet"}
    v = pd.read_parquet(vp, columns=["pattern", "judge", "dimension", "satisfied",
                                     "evidence", "reasoning"])
    v = v[v.pattern.astype(str).str.match(r"^base_p([0-9]|10)$")]
    fa = v[v.dimension == "factual_accuracy"].copy()
    txt = (fa["reasoning"].fillna("") + " " + fa["evidence"].fillna(""))
    fa["mentions_citation"] = txt.str.contains(CITE_WORDS)
    by_judge = {}
    for j in sorted(fa.judge.unique()):
        sub = fa[fa.judge == j]
        fam = {"gpt52": "openai", "gpt-4.1": "openai", "gpt-4o": "openai",
               "claude_opus": "claude", "claude_sonnet": "claude",
               "claude_code": "claude"}.get(j, "other")
        # Among UNSATISFIED factual verdicts, how often is a citation/source the stated reason?
        unsat = sub[~sub.satisfied]
        by_judge[j] = {
            "family": fam,
            "n_factual_verdicts": int(len(sub)),
            "citation_mention_rate_all": round(float(sub["mentions_citation"].mean()), 4),
            "citation_mention_rate_unsatisfied": (
                round(float(unsat["mentions_citation"].mean()), 4) if len(unsat) else None),
        }
    # Family-level contrast on the unsatisfied channel (the leakage signal).
    fa["family"] = fa.judge.map({"gpt52": "openai", "gpt-4.1": "openai", "gpt-4o": "openai",
                                 "claude_opus": "claude", "claude_sonnet": "claude",
                                 "claude_code": "claude"}).fillna("other")
    unsat_all = fa[~fa.satisfied]
    fam_rate = (unsat_all.groupby("family")["mentions_citation"].mean().round(4).to_dict()
                if len(unsat_all) else {})
    return {"status": "mined", "by_judge": by_judge,
            "family_unsatisfied_citation_mention_rate": fam_rate,
            "interpretation": (
                "If claude >> openai on citation_mention_rate_unsatisfied, Claude judges cite "
                "missing/weak citations as a reason to FAIL factual_accuracy more often than "
                "OpenAI — the channel through which density could causally move the verdict."),
            "n_factual_verdicts_total": int(len(fa))}


def mitigation_prompts() -> dict:
    """The 2-3 mitigation prompts E4 will A/B (drafted in prereg_E4.md; text lives here so
    the paid run can pull them). Not executed in this build phase."""
    return {
        "status": "drafted_not_run",
        "prompts": {
            "M1_decouple": (
                "When judging FACTUAL ACCURACY, evaluate only whether the stated claims are "
                "internally consistent and correct. Do NOT consider the presence, count, or "
                "formatting of citation markers — those are scored separately under "
                "citation_quality and attribution_quality."),
            "M2_density_blind": (
                "Citation markers such as [3] have been programmatically altered and are NOT "
                "reliable signals of accuracy. Judge factual claims on their substance alone."),
            "M3_count_invariance": (
                "Two reports with identical prose but different numbers of citation markers MUST "
                "receive the same factual_accuracy verdict. Ignore citation density entirely for "
                "this dimension."),
        },
        "note": "A/B each vs the unmodified rubric on C0 vs C3 (double-density); the mitigation "
                "wins if it shrinks the Claude density slope toward the OpenAI slope.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run CoT mining over the baseline corpus and PRINT the would-be "
                         "canonical block; do NOT write canonical_numbers.json.")
    args = ap.parse_args()

    e4 = load_e4_scores()
    causal = {dim: (fit_causal(e4, dim) if not e4.empty else
                    {"status": "pending", "note": "E4 re-judge data not yet present"})
              for dim in DIMS}
    block = {
        "design": {
            "conditions": DENSITY_INDEX, "is_shuffle": IS_SHUFFLE,
            "endpoints": DIMS,
            "model": ("dim_score ~ density_index * C(judge_family) + is_shuffle + C(pattern) "
                      "+ (1 | query_id)"),
            "hypothesis_H1": "density_index:claude interaction > 0 (effect is Claude-specific)",
            "hypothesis_null": ("content is fixed across conditions, so any non-zero density "
                                "slope is a judge artefact, not a quality signal"),
            "n_plan": "100 reports x 5 conditions x 5 judges",
            "near_null_handling": "base_p10 near-null reports down-weighted/excluded from contrast",
        },
        "e4_judge_data_present": {j: E4_JUDGE_DIRS[j].exists() for j in E4_JUDGE_DIRS},
        "causal": causal,
        "cot_mining": cot_mining(),
        "mitigation_prompts": mitigation_prompts(),
    }

    print(json.dumps({"e4_cite_causal": {
        "e4_judge_data_present": block["e4_judge_data_present"],
        "causal_status": {d: causal[d].get("status") for d in DIMS},
        "cot_mining": block["cot_mining"].get("family_unsatisfied_citation_mention_rate"),
    }}, indent=1))

    if args.dry_run:
        print("\n[DRY RUN] canonical_numbers.json NOT written. CoT-mining block above is live.")
        return 0

    canon = json.loads(CANON.read_text()) if CANON.exists() else {}
    canon["e4_cite_causal"] = block
    CANON.write_text(json.dumps(canon, indent=1))
    print(f"\nWrote canonical_numbers.json['e4_cite_causal'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
