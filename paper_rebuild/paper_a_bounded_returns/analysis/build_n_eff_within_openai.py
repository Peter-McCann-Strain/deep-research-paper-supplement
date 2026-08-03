#!/usr/bin/env python
"""T1_within_openai_neff — recompute N_eff on a 2-family x >=2-judge grid and append
``n_eff.within_openai`` to canonical_numbers.json (ADDITIVE; never mutates n_eff.overall).

Pre-registration: docs/publication/prereg/prereg_E3.md (Phase-2 "symmetric within-OpenAI
cell", line 41). This is the companion to build_n_eff.py, which it does NOT touch.

What it adds
------------
build_n_eff.py reports the 3-judge panel (gpt52, claude_opus, claude_sonnet) where the
within-family pair is Anthropic (opus, sonnet) and the cross is OpenAI->Anthropic. That
design cannot separate *same-lab* redundancy from *family-level* redundancy on the OpenAI
side, because OpenAI has only one judge (gpt52) in the base panel. This script adds the two
new OpenAI secondary judges (gpt-4.1, gpt-4o; verdicts produced by
scripts/run_openai_panel_judge_fullcorpus.py) and computes the SYMMETRIC grid:

  * within-OpenAI pairwise phi over {gpt52, gpt-4.1, gpt-4o}  (same-lab redundancy)
  * within-Anthropic pairwise phi over {claude_opus, claude_sonnet}  (same-lab redundancy)
  * cross-family phi (OpenAI judge vs Anthropic judge)
  * N_eff for the full OpenAI-pair-vs-Anthropic-pair grid, and the within-family-only N_eff
    for each lab, so same-lab vs family-level collapse is read off directly.

Substrate: the fully-crossed cell = (pattern x query x criterion_id) verdicts scored by ALL
judges in the requested grid, restricted to pattern_family == 'base' and satisfied_is_known.
Alignment is by criterion_id, which is md5[:12] of the whitespace/case-normalised criterion
text — identical to scripts/build_analysis_dataframes.py, because every judge here scores the
SAME rubric_v2 criteria text. The script reads the on-disk judge JSON dirs DIRECTLY (it does
NOT depend on df_verdicts being rebuilt), so it is fully self-contained and idempotent.

Method (same as build_n_eff.py): N_eff = N^2 / (1' R 1) = N^2 / (N + 2*sum_{i<j} rho_ij),
rho = phi (Pearson on 0/1 verdicts). Determinism: closed-form, inputs sorted, no randomness.

NO Opus is judged or invoked by this script. It only READS verdict JSONs already on disk
(including any pre-existing Opus verdicts, which is permitted re-use, not new judging).
"""
import hashlib
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"
RESULTS = ROOT / "results"
EVAL_QUERIES = ROOT / "data" / "eval_queries_v2.json"

# judge_name -> (verdict dir, family). judge_name strings match build_analysis_dataframes.py.
JUDGE_DIRS = {
    "gpt52":         (RESULTS / "judge_gpt52",          "openai"),
    "gpt41":         (RESULTS / "judge_gpt41",          "openai"),
    "gpt4o":         (RESULTS / "judge_gpt4o",          "openai"),
    "claude_opus":   (RESULTS / "judge_claude_opus",    "anthropic"),
    "claude_sonnet": (RESULTS / "judge_claude_sonnet",  "anthropic"),
}

# Core base patterns the N_eff base cell uses (base_p0..base_p12). Matches the default
# corpus set; base_pN_vK (variance family) and 7b/16turn arms are excluded for the
# symmetric core-base grid, exactly as the primary n_eff cell restricts to family 'base'
# numeric patterns shared by the panel.
BASE_PATTERNS = [f"base_p{i}" for i in range(13)]

OPENAI = ["gpt52", "gpt41", "gpt4o"]
ANTHROPIC = ["claude_opus", "claude_sonnet"]


def _normalize_criterion(text: str) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _crit_id(text: str) -> str:
    return hashlib.md5(_normalize_criterion(text).encode("utf-8")).hexdigest()[:12]


def _extract_verdict_satisfied(v: dict):
    """Mirror build_analysis_dataframes._extract_verdict_fields for the satisfied bit."""
    sat_raw = v.get("satisfied")
    if sat_raw is None:
        verdict_str = str(v.get("verdict", "")).strip().upper()
        if verdict_str == "SATISFIED":
            return True
        if verdict_str in ("NOT_SATISFIED", "NOT SATISFIED", "UNSATISFIED", "FAILED"):
            return False
        return None
    return bool(sat_raw)


def _crit_text(v: dict) -> str:
    return (v.get("criterion") or v.get("description") or v.get("text")
            or v.get("criterion_text") or "")


def load_verdicts() -> pd.DataFrame:
    """Read every available judge's base-pattern verdict JSONs directly from disk."""
    rows = []
    for judge, (jdir, family) in JUDGE_DIRS.items():
        if not jdir.exists():
            continue
        for pattern in BASE_PATTERNS:
            pdir = jdir / pattern
            if not pdir.exists():
                continue
            for jpath in sorted(pdir.glob("*.json")):
                qid = jpath.stem
                try:
                    data = json.loads(jpath.read_text())
                except Exception:
                    continue
                for v in (data.get("verdicts") or []):
                    if not isinstance(v, dict):
                        continue
                    sat = _extract_verdict_satisfied(v)
                    if sat is None:
                        continue
                    ctext = _crit_text(v)
                    rows.append({
                        "judge": judge, "family": family, "pattern": pattern,
                        "query_id": qid, "criterion_id": _crit_id(ctext),
                        "satisfied": int(bool(sat)),
                    })
    return pd.DataFrame(rows)


def phi(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def wide(df, judges):
    key = ["pattern", "query_id", "criterion_id"]
    w = (df[df.judge.isin(judges)]
         .pivot_table(index=key, columns="judge", values="satisfied", aggfunc="first")
         .dropna(subset=judges))
    for j in judges:
        w[j] = w[j].astype(int)
    return w.sort_index()


def n_eff_from_rhos(rhos, n):
    s = sum(r for r in rhos if np.isfinite(r))
    denom = n + 2 * s
    return float((n * n) / denom) if denom > 0 else float("nan")


def grid_neff(df, judges, label):
    """Fully-crossed N_eff over the given judge set + the pairwise phi matrix."""
    w = wide(df, judges)
    pairs = list(itertools.combinations(judges, 2))
    rho = {f"{a}|{b}": round(phi(w[a], w[b]), 4) for a, b in pairs}
    rhos = [phi(w[a], w[b]) for a, b in pairs]
    return {
        "label": label,
        "judges": list(judges),
        "n_cells": int(len(w)),
        "phi": rho,
        "n_eff": round(n_eff_from_rhos(rhos, len(judges)), 4),
    }


def mean_cross_family_phi(df, openai_js, anth_js):
    """Mean cross-family phi over the fully-crossed cell of all listed judges."""
    all_js = list(openai_js) + list(anth_js)
    w = wide(df, all_js)
    vals = []
    for a in openai_js:
        for b in anth_js:
            vals.append(phi(w[a], w[b]))
    return round(float(np.nanmean(vals)), 4) if vals else float("nan"), int(len(w))


def main():
    df = load_verdicts()
    present = sorted(df.judge.unique().tolist()) if len(df) else []
    openai_present = [j for j in OPENAI if j in present]
    anth_present = [j for j in ANTHROPIC if j in present]

    new_openai = [j for j in ("gpt41", "gpt4o") if j in present]
    ready = len(openai_present) >= 2 and len(anth_present) >= 2 and len(new_openai) >= 1

    out = {
        "_note": "T1 within-OpenAI N_eff control cell. Adds gpt-4.1 + gpt-4o as full-corpus "
                 "secondary judges (verdicts from run_openai_panel_judge_fullcorpus.py, JUDGE "
                 "endpoint only) and computes the SYMMETRIC 2-family grid: within-OpenAI pair "
                 "vs within-Anthropic pair. Separates same-lab judge redundancy from "
                 "family-level redundancy. Method matches build_n_eff.py "
                 "(N_eff = N^2/(N+2*sum phi), phi on 0/1 verdicts). Self-contained: reads judge "
                 "JSON dirs directly; criterion_id == md5[:12] of normalised criterion text, "
                 "identical to build_analysis_dataframes.py. NO Opus judged here (pre-existing "
                 "Opus verdicts on disk are re-used, not re-generated).",
        "prereg": "docs/publication/prereg/prereg_E3.md (Phase-2 within-OpenAI cell)",
        "base_patterns": BASE_PATTERNS,
        "judges_present": present,
        "openai_judges_present": openai_present,
        "anthropic_judges_present": anth_present,
        "new_openai_judges_present": new_openai,
        "ready": ready,
    }

    if not ready:
        out["status"] = ("PENDING_VERDICTS: need >=2 OpenAI judges (incl. >=1 of gpt41/gpt4o) "
                         "AND >=2 Anthropic judges on the base cell. Run "
                         "run_openai_panel_judge_fullcorpus.py --judge gpt-4.1 / --judge gpt-4o "
                         "first, then re-run this builder.")
    else:
        # within-OpenAI grid (the new control cell)
        out["within_openai"] = grid_neff(df, openai_present, "within_openai")
        # within-Anthropic grid (the comparator same-lab pair)
        out["within_anthropic"] = grid_neff(df, anth_present, "within_anthropic")

        wo_phis = [v for v in out["within_openai"]["phi"].values() if np.isfinite(v)]
        wa_phis = [v for v in out["within_anthropic"]["phi"].values() if np.isfinite(v)]
        out["within_openai"]["mean_within_phi"] = round(float(np.mean(wo_phis)), 4) if wo_phis else float("nan")
        out["within_anthropic"]["mean_within_phi"] = round(float(np.mean(wa_phis)), 4) if wa_phis else float("nan")

        cross_phi, cross_n = mean_cross_family_phi(df, openai_present, anth_present)
        out["cross_family_phi_mean"] = cross_phi
        out["cross_family_n_cells"] = cross_n

        # full 2-family grid N_eff (all present OpenAI + Anthropic judges crossed)
        out["full_grid"] = grid_neff(df, openai_present + anth_present, "openai_pair_x_anthropic_pair")

        out["interpretation"] = (
            f"Same-lab redundancy: within-OpenAI mean phi = {out['within_openai']['mean_within_phi']} "
            f"(N_eff_openai = {out['within_openai']['n_eff']} over {len(openai_present)} judges); "
            f"within-Anthropic mean phi = {out['within_anthropic']['mean_within_phi']} "
            f"(N_eff_anthropic = {out['within_anthropic']['n_eff']}). Cross-family mean phi = "
            f"{out['cross_family_phi_mean']}. If within-family phi >> cross-family phi for BOTH "
            "labs, judge redundancy is a same-lab effect symmetric across families, not a Claude-"
            "specific artefact; if only Anthropic collapses, the headline within-Claude redundancy "
            "is family-specific. Compare against n_eff.overall (3-judge panel N_eff)."
        )

    cn = json.load(open(CANON))
    existing = cn.get("n_eff")
    if not isinstance(existing, dict):
        raise SystemExit("REFUSING: canonical 'n_eff' is missing or not an object; run build_n_eff.py first.")
    if "within_openai" in existing and existing.get("within_openai", {}).get("ready"):
        # idempotent: only overwrite if the new computation is at least as complete.
        pass
    # ADDITIVE: write under a dedicated sub-key; never touch n_eff.overall / per_dimension.
    existing["within_openai"] = out
    cn["n_eff"] = existing
    tmp = str(CANON) + ".tmp"
    open(tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(tmp, CANON)

    if ready:
        print(f"n_eff.within_openai: openai={openai_present} anthropic={anth_present}")
        print(f"  within-OpenAI N_eff={out['within_openai']['n_eff']} "
              f"(mean phi={out['within_openai']['mean_within_phi']}, n_cells={out['within_openai']['n_cells']})")
        print(f"  within-Anthropic N_eff={out['within_anthropic']['n_eff']} "
              f"(mean phi={out['within_anthropic']['mean_within_phi']})")
        print(f"  cross-family mean phi={out['cross_family_phi_mean']}")
        print(f"  full 2-family grid N_eff={out['full_grid']['n_eff']} ({out['full_grid']['n_cells']} cells)")
    else:
        print(f"n_eff.within_openai: PENDING — judges present={present}. {out['status']}")


if __name__ == "__main__":
    main()
