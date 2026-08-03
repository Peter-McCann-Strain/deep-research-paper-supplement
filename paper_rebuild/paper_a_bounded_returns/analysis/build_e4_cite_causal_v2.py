#!/usr/bin/env python3
"""E4 CITE-CAUSAL — Step 6 (v2): canonical-path-FIXED build that lands the causal
citation-intervention result, NO-Opus family arm.

Why this v2 file exists
-----------------------
The original ``build_e4_cite_causal.py`` hardcodes the DEAD canonical path
``paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json`` (lines 58-60).  That tree was
MOVED to ``paper_rebuild/paper_a_bounded_returns/analysis/`` by commit 0a80ba6, so the original
crashes on the final ``CANON.write_text(...)`` (the parent dir no longer exists) and the
``e4_cite_causal`` block has therefore NEVER landed in canonical.  In ``rebuild_all.sh`` the
step is wrapped ``2>&1 || true``, so the crash is SILENTLY swallowed.

This v2 reuses ALL of the original modelling/CoT-mining logic by importing it (no copy/drift)
and only OVERRIDES the canonical output path to the real, current store.  It is additive
(read-modify-write): it loads existing canonical, sets exactly one key
``canon['e4_cite_causal']``, and rewrites — every other key is preserved untouched.

NO-Opus scope (hard rule, 2026-06-22)
-------------------------------------
The Opus arm is NOT regenerated.  The causal model reads whatever judge dirs are PRESENT:
  * gpt52  (judge_gpt52_e4)         — DONE  (authoritative OpenAI judge)
  * gpt-4.1 (judge_gpt41_e4)        — panel comparator, finish remaining cells
  * claude_sonnet (judge_claude_sonnet_e4) — the NO-Opus Claude FULL-N family arm
  * claude_opus (judge_claude_opus_e4)     — READ-ONLY IF it already exists from a prior
        run; this script never *generates* Opus verdicts.  Per the rule, prefer running
        the build AFTER sonnet lands so the claude family slope is sonnet-driven.
The block records which judges contributed (``e4_judge_data_present`` + ``causal.*.judges``)
so the Opus-exclusion scope effect is explicit in canonical.

READ-ONLY everywhere except the single canonical JSON (an analysis artefact, not protected).

Usage:
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e4_cite_causal_v2.py --dry-run
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e4_cite_causal_v2.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.stats as _st
import statsmodels.formula.api as _smf

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent

# Import the original module by file path (it lives next to this file) so we reuse its
# load_e4_scores / fit_causal / cot_mining / mitigation_prompts / DIMS / E4_JUDGE_DIRS
# verbatim — single source of truth for the statistics.
_ORIG = _THIS_DIR / "build_e4_cite_causal.py"
_spec = importlib.util.spec_from_file_location("_e4_orig", _ORIG)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# The ONE fix: canonical store now lives in THIS dir (or wherever this file lives), never
# the dead paper_rebuild/paper_a_bounded_returns path the original hardcodes.
ANA = _THIS_DIR
if (_REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis").exists():
    ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"


# ── Small-sample-correct denominator dof for the judge_family x density interaction ──
#
# The original ``_mod.fit_causal`` reports an ASYMPTOTIC Wald p for the
# ``density_index:C(judge_family)`` interaction (mixedlm z-test, infinite dof).  For a
# single-random-intercept LMM with finite clusters that p is anti-conservative.  We refer
# the same Wald t-statistic to a Satterthwaite / Kenward-Roger-style small-sample
# denominator dof instead, and report the corrected p ALONGSIDE the Wald p so the
# inference is defensible.
#
# pymer4/lme4 (true Kenward-Roger) is not installed here, so we use the between-within
# (containment) Satterthwaite denominator dof, which is EXACT for balanced designs and a
# well-grounded approximation otherwise:
#       ddf = n_obs - rank(fixed effects) - (n_groups - 1)
# ``density_index`` varies WITHIN ``query_id``, so the interaction is a within-cluster
# term and takes the within-subject (residual) dof.  The whole computation is
# deterministic (REML fit, fixed formula, sorted inputs from ``load_e4_scores``).

def _satterthwaite_interaction(dim: str, e4) -> dict | None:
    """Refit the canonical interaction LMM and return KR/Satterthwaite-corrected p's
    keyed by interaction coefficient, each with the matching asymptotic Wald p."""
    d = e4[e4.dimension == dim].copy()
    d = d[d.pattern != "base_p10__nearnull"]
    if d.empty or d.judge_family.nunique() < 2 or d.query_id.nunique() < 3:
        return None
    d["judge_family"] = d["judge_family"].astype("category")
    formula = "score ~ density_index * C(judge_family) + is_shuffle + C(pattern)"
    md = _smf.mixedlm(formula, data=d, groups=d["query_id"])
    mf = md.fit(reml=True, method="lbfgs")
    n_obs = int(len(d))
    n_groups = int(d.query_id.nunique())
    p_fixed = int(md.exog.shape[1])                       # rank of fixed-effects design
    ddf = n_obs - p_fixed - (n_groups - 1)                # between-within (containment) dof
    out = {}
    for k in mf.params.index:
        if "density_index:" not in k:
            continue
        beta = float(mf.params[k])
        se = float(mf.bse[k])
        t = beta / se if se else float("nan")
        p_kr = float(2 * _st.t.sf(abs(t), ddf)) if np.isfinite(t) and ddf > 0 else float("nan")
        out[k] = {
            "beta": round(beta, 4),
            "p_wald": float(mf.pvalues[k]),               # asymptotic z-test (original)
            "p_satterthwaite": p_kr,                      # small-sample-correct denominator dof
            "t": round(float(t), 4),
            "ddf": int(ddf),
        }
    return {
        "method": ("Satterthwaite/Kenward-Roger between-within (containment) denominator dof; "
                   "ddf = n_obs - rank(fixed) - (n_groups-1); REML fit. lme4/pymer4 absent so "
                   "exact KR unavailable, BW is exact for balanced designs."),
        "n_obs": n_obs, "n_groups": n_groups, "ddf": int(ddf),
        "interaction": out,
    }


def build_block() -> dict:
    e4 = _mod.load_e4_scores()
    causal = {
        dim: (_mod.fit_causal(e4, dim) if not e4.empty
              else {"status": "pending", "note": "E4 re-judge data not yet present"})
        for dim in _mod.DIMS
    }
    # Augment each fitted dim's interaction with the small-sample-correct (KR/Satterthwaite)
    # p alongside the Wald p. Additive: never removes the original Wald result.
    if not e4.empty:
        for dim in _mod.DIMS:
            blk = causal.get(dim, {})
            if blk.get("status") != "fit":
                continue
            try:
                kr = _satterthwaite_interaction(dim, e4)
            except Exception as exc:  # pragma: no cover - defensive
                kr = {"error": str(exc)[:150]}
            if kr is not None:
                blk["interaction_small_sample"] = kr
                # Mirror the corrected p back onto the original interaction dict for easy reads.
                if isinstance(blk.get("interaction"), dict) and "interaction" in kr:
                    for ck, cv in kr["interaction"].items():
                        if ck in blk["interaction"] and isinstance(blk["interaction"][ck], dict):
                            blk["interaction"][ck]["p_satterthwaite"] = cv["p_satterthwaite"]
                            blk["interaction"][ck]["ddf"] = cv["ddf"]
    return {
        "design": {
            "conditions": _mod.DENSITY_INDEX, "is_shuffle": _mod.IS_SHUFFLE,
            "endpoints": _mod.DIMS,
            "model": ("dim_score ~ density_index * C(judge_family) + is_shuffle + C(pattern) "
                      "+ (1 | query_id)"),
            "hypothesis_H1": "density_index:claude interaction > 0 (effect is Claude-specific)",
            "hypothesis_null": ("content is fixed across conditions, so any non-zero density "
                                "slope is a judge artefact, not a quality signal"),
            "n_plan": "100 reports x 5 conditions x 5 judges",
            "near_null_handling": "base_p10 near-null reports down-weighted/excluded from contrast",
        },
        "no_opus_scope": {
            "rule": "NO Opus judging in NEW runs (programme owner 2026-06-22)",
            "claude_family_arm": "claude_sonnet (full-n) is the Claude family arm; "
                                 "claude_opus is read-only-if-present, never regenerated",
            "effect": ("If judge_claude_opus_e4 is absent, the 'claude' judge_family slope is "
                       "estimated from Sonnet alone; the Claude-vs-OpenAI interaction therefore "
                       "reflects Sonnet, not an Opus+Sonnet pool. Interpret H1 accordingly."),
        },
        "e4_judge_data_present": {j: _mod.E4_JUDGE_DIRS[j].exists() for j in _mod.E4_JUDGE_DIRS},
        "causal": causal,
        "cot_mining": _mod.cot_mining(),
        "mitigation_prompts": _mod.mitigation_prompts(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute CoT-mining + causal block and PRINT a summary; do NOT write "
                         "canonical_numbers.json.")
    args = ap.parse_args()

    block = build_block()

    def _interaction_ps(dim: str) -> dict:
        ss = block["causal"].get(dim, {}).get("interaction_small_sample")
        if not ss or "interaction" not in ss:
            return {}
        return {k: {"beta": v["beta"], "p_wald": v["p_wald"],
                    "p_satterthwaite": v["p_satterthwaite"], "ddf": v["ddf"]}
                for k, v in ss["interaction"].items()}

    print(json.dumps({"e4_cite_causal": {
        "canonical_target": str(CANON),
        "e4_judge_data_present": block["e4_judge_data_present"],
        "causal_status": {d: block["causal"][d].get("status") for d in _mod.DIMS},
        "claude_judges_contributing": sorted(
            j for j in ("claude_sonnet", "claude_opus")
            if block["e4_judge_data_present"].get(j)),
        "interaction_wald_vs_satterthwaite": {d: _interaction_ps(d) for d in _mod.DIMS},
        "cot_mining": block["cot_mining"].get("family_unsatisfied_citation_mention_rate"),
    }}, indent=1))

    if args.dry_run:
        print("\n[DRY RUN] canonical_numbers.json NOT written. Block computed above is live.")
        return 0

    if not CANON.exists():
        print(f"REFUSING: canonical store not found at {CANON}. "
              f"Will not create a stray canonical file at the wrong path.")
        return 1

    canon = json.loads(CANON.read_text())  # read-modify-write: preserves all other keys
    canon["e4_cite_causal"] = block
    CANON.write_text(json.dumps(canon, indent=1))
    print(f"\nWrote canonical_numbers.json['e4_cite_causal'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
