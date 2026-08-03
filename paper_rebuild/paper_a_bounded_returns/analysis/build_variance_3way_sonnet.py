#!/usr/bin/env python
"""T1_sonnet_variance_3way — ingest the Sonnet variance-replicate cell and fit the
pre-registered 3-way run x query x judge REML variance decomposition.

Context
-------
`build_variance_decomposition.py` estimates the RUN facet from the gpt52-ONLY replicate
corpus (`base_{arch}_v{n}`), and explicitly states (single_judge_caveat) that run and judge
variance live on NON-CROSSED substrates: run from the gpt52 replicates, judge from the
multi-judge PANEL in canonical key `variance_components`. The genuine 3-way crossed
run x query x judge cell needs the SAME replicate reports scored by a SECOND judge.

Those second-judge verdicts EXIST on disk: `artifacts/judges/judge_claude_sonnet48/
base_p{arch}_v{n}/<qid>.json` (Sonnet-4.6, claude_code_manual source). They were never
ingested into the analysis parquet because `build_analysis_dataframes.py::JUDGE_DIRS` maps
`claude_sonnet -> results/judge_claude_sonnet` (the single-run canonical dir), NOT the
`judge_claude_sonnet48` replicate dir. This script ingests them WITHOUT touching the global
parquet build (no clobber of df_*.parquet) and fits the crossed model.

NO NEW JUDGING is performed: this re-uses Sonnet verdicts already on disk (read-only).
NO PAID API CALL. CPU-only. Deterministic. Idempotent. Reads real on-disk inputs.

What it emits
-------------
Appends `variance_decomposition.three_way` to canonical_numbers.json. The pre-existing keys
of `variance_decomposition` (run_noise, components, pooled, flip_rates, citation_stability,
mde, leaderboard_flip, single_judge_caveat, ...) are PRESERVED untouched; only the new
sub-key is added/overwritten. The legacy `variance_components` (panel) key is NOT touched.

Faithful score recompute
-------------------------
Sonnet's STORED `overall_score` is corrupted upstream (documented in build_analysis_dataframes
and TRUSTWORTHY_OVERALL_SCORE_JUDGES). We therefore recompute `overall_score_recomputed` for
BOTH judges using the EXACT builder weighting logic (source-type weights from
DIMENSION_WEIGHTS_BY_SOURCE), imported from build_analysis_dataframes so there is a single
source of truth. A self-check asserts our recompute reproduces the gpt52 parquet value to 1e-6
before the Sonnet recompute is trusted.

Honest coverage caveat
-----------------------
Sonnet replicate coverage is THINNER and UNBALANCED vs gpt52 (gpt52: P0 x11 over 30 queries,
7 arch x3; Sonnet: fewer queries per cell, e.g. P0 covers 27 not 30). The crossed REML is fit
on the BALANCED INTERSECTION (cells/queries scored by BOTH judges) so the judge facet is
identified on a fully-crossed substrate; the dropped (gpt52-only) rows are reported as a
caveat field. The 3-way fit is the EXTENSION promised by single_judge_caveat, with its limits
stated, NOT a replacement for the gpt52 main result.
"""
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(".")
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"
SONNET_DIR = ROOT / "artifacts" / "judges" / "judge_claude_sonnet48"
OVERALL_PARQUET = ROOT / "data" / "analysis" / "df_overall_scores.parquet"
MANIFEST = ROOT / "data" / "eval_queries_v2.json"
SEED = 20260611  # match build_variance_decomposition for any resampling determinism

# Single source of truth for the recompute: import the builder's own helpers/constants.
sys.path.insert(0, str(ROOT))
from scripts.build_analysis_dataframes import (  # noqa: E402
    DIMENSION_WEIGHTS_V2,
    DIMENSION_WEIGHTS_BY_SOURCE,
    SOURCE_TO_WEIGHT_KEY,
    _extract_dim_met_total,
    _safe_load_json,
)

CELL_RE = re.compile(r"^(base_p\d+)_v(\d+)$")


def arch_rep_of(cell):
    """'base_p0_v11' -> ('base_p0', 11); non-replicate cells -> (None, None)."""
    m = CELL_RE.match(str(cell))
    return (m.group(1), int(m.group(2))) if m else (None, None)


# ---------- recompute: byte-identical to build_analysis_dataframes ----------
_man = json.load(open(MANIFEST))["queries"]
QID_TO_SOURCE = {q["id"]: str(q.get("source", "")) for q in _man}


def recompute_overall(dims, weights):
    """Reproduce build_judge_frames._recompute exactly (source-type weights)."""
    nd = {}
    for dn, de in (dims or {}).items():
        if isinstance(de, dict):
            nd[dn] = _extract_dim_met_total(de)
        elif isinstance(de, (int, float)):
            nd[dn] = (float(de), None, None)
    s_sum = w_sum = 0.0
    for dn, wt in weights.items():
        tup = nd.get(dn)
        if tup is None:
            continue
        sv = tup[0]
        if sv is None:
            m, t = tup[1], tup[2]
            if m is not None and t and t > 0:
                sv = m / t
        if sv is None:
            continue
        s_sum += float(sv) * float(wt)
        w_sum += float(wt)
    if w_sum == 0:
        return None
    return s_sum / w_sum if w_sum < 0.999 else s_sum


def weights_for(qid):
    src_key = SOURCE_TO_WEIGHT_KEY.get(QID_TO_SOURCE.get(qid, ""), "default")
    return DIMENSION_WEIGHTS_BY_SOURCE.get(src_key, DIMENSION_WEIGHTS_V2)


def score_cell_file(path):
    data = _safe_load_json(Path(path))
    if data is None:
        return None
    qid = Path(path).stem
    return recompute_overall(data.get("dimensions", {}) or {}, weights_for(qid))


# ---------- 1. gpt52 replicate frame (from the already-built parquet) ----------
O = pd.read_parquet(OVERALL_PARQUET)
g = O[(O.pattern_family == "variance") & (O.judge == "gpt52")].copy()
_ar = [arch_rep_of(p) for p in g.pattern.astype(str)]
g["arch"] = [a for a, _ in _ar]
g["replicate"] = [r for _, r in _ar]
g = g.dropna(subset=["arch", "overall_score_recomputed"])
gpt = g[["arch", "replicate", "query_id", "overall_score_recomputed"]].copy()
gpt["judge"] = "gpt52"
gpt = gpt.rename(columns={"overall_score_recomputed": "score"})

# ---------- self-check: our recompute reproduces the parquet for gpt52 ----------
# (validates the recompute path before we trust the Sonnet recompute)
SELF_CHECK = {"checked": 0, "max_abs_err": 0.0}
_gpt_dir = ROOT / "results" / "judge_gpt52"
for _, r in gpt.sample(min(40, len(gpt)), random_state=SEED).iterrows():
    cell = f"{r.arch}_v{int(r.replicate)}"
    fp = _gpt_dir / cell / f"{r.query_id}.json"
    if fp.exists():
        mine = score_cell_file(fp)
        if mine is not None:
            SELF_CHECK["checked"] += 1
            SELF_CHECK["max_abs_err"] = max(SELF_CHECK["max_abs_err"], abs(mine - float(r.score)))
assert SELF_CHECK["checked"] > 0, "self-check found no gpt52 files to validate recompute"
assert SELF_CHECK["max_abs_err"] < 1e-6, (
    f"recompute drift vs parquet: max_abs_err={SELF_CHECK['max_abs_err']:.2e}")

# ---------- 2. ingest Sonnet replicate verdicts off disk ----------
son_rows = []
son_cells = sorted(d for d in os.listdir(SONNET_DIR) if CELL_RE.match(d))
for cell in son_cells:
    arch, rep = arch_rep_of(cell)
    cdir = SONNET_DIR / cell
    for jf in sorted(cdir.glob("*.json")):
        sc = score_cell_file(jf)
        if sc is None:
            continue
        son_rows.append({"arch": arch, "replicate": rep, "query_id": jf.stem,
                         "judge": "claude_sonnet48", "score": float(sc)})
son = pd.DataFrame(son_rows)
assert len(son) > 0, "no Sonnet replicate verdicts ingested"

# ---------- 3. honest coverage accounting (Sonnet thinner/unbalanced vs gpt52) ----------
def cov(df):
    out = {}
    for a, sub in df.groupby("arch"):
        out[a] = {"n_replicates": int(sub.replicate.nunique()),
                  "n_queries": int(sub.query_id.nunique()),
                  "n_rows": int(len(sub))}
    return out


cov_gpt, cov_son = cov(gpt), cov(son)
coverage = {
    "gpt52": {"arches": sorted(cov_gpt), "per_arch": cov_gpt,
              "n_rows": int(len(gpt)), "n_queries": int(gpt.query_id.nunique())},
    "claude_sonnet48": {"arches": sorted(cov_son), "per_arch": cov_son,
                        "n_rows": int(len(son)), "n_queries": int(son.query_id.nunique())},
    "caveat": ("Sonnet replicate coverage is THINNER and UNBALANCED vs gpt52: gpt52 scores "
               "P0 x11 reps over 30 variance queries (7 further arch x3); Sonnet covers fewer "
               "queries per cell (e.g. P0: 27 of 30). Sonnet's STORED overall_score is "
               "corrupted upstream, so both judges use overall_score_recomputed (source-type "
               "weights, identical to build_analysis_dataframes; self-checked to <1e-6 vs the "
               "gpt52 parquet). The crossed REML below is fit on the BALANCED INTERSECTION "
               "scored by BOTH judges; gpt52-only rows are dropped from the fit and reported "
               "here."),
}

# ---------- 4. balanced intersection (cells x queries scored by BOTH judges) ----------
key = ["arch", "replicate", "query_id"]
both = (set(map(tuple, gpt[key].values)) & set(map(tuple, son[key].values)))
both_idx = pd.MultiIndex.from_tuples(sorted(both), names=key)
gpt_i = gpt.set_index(key).loc[both_idx].reset_index()
son_i = son.set_index(key).loc[both_idx].reset_index()
long = pd.concat([gpt_i, son_i], ignore_index=True)
long = long.sort_values(["judge", "arch", "replicate", "query_id"]).reset_index(drop=True)
# stable string factor levels for crossed REML
long["run_id"] = long.arch + "_v" + long.replicate.astype(int).astype(str)
long["qid"] = long.query_id.astype(str)
long["jid"] = long.judge.astype(str)

intersection = {
    "n_cells_run_query": int(len(both)),
    "n_rows_per_judge": int(len(both)),
    "n_total_rows": int(len(long)),
    "n_runs": int(long.run_id.nunique()),
    "n_queries": int(long.qid.nunique()),
    "n_judges": int(long.jid.nunique()),
    "arches": sorted(long.arch.unique()),
    "design": "fully crossed run x query within the two-level judge factor (gpt52, sonnet48)",
}

# ---------- 5. pre-registered 3-way crossed REML (run x query x judge) ----------
# statsmodels MixedLM: a single trivial group + variance-components formula gives crossed,
# independent variance components for run, query and judge; residual = scale. We mean-center
# nothing (intercept absorbs the grand mean). This is the standard crossed-RE idiom.
from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: E402

three_way = {"_status": "not_fit"}
try:
    long["_grp"] = 0  # single group so VCs are crossed (not nested)
    # CORRECTED (world-class review B3): the 32 run_id levels span 8 ARCHITECTURES,
    # so the earlier 3-way (run x query x judge) absorbed systematic between-architecture
    # mean differences into sigma2_run, inflating the "run noise" share. We add an
    # ARCHITECTURE variance component (run_id is nested within arch), so sigma2_run is
    # now pure within-architecture replicate noise and between-architecture signal is
    # separated into sigma2_arch. This is a 4-way crossed/nested decomposition.
    vc = {"arch": "0 + C(arch)", "run": "0 + C(run_id)",
          "query": "0 + C(qid)", "judge": "0 + C(jid)"}
    md = MixedLM.from_formula("score ~ 1", groups="_grp", vc_formula=vc, data=long)
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        mf = md.fit(reml=True, method="lbfgs", maxiter=300)
    _fit_warnings = sorted({str(wi.message)[:120] for wi in _w})
    # statsmodels stores VC estimates in mf.vcomp aligned to the vc_formula key order
    names = list(vc.keys())
    s2 = {names[i]: float(mf.vcomp[i]) for i in range(len(names))}
    s2_resid = float(mf.scale)
    tot = sum(s2.values()) + s2_resid
    three_way = {
        "_status": "fit",
        "model": ("MixedLM REML, score ~ 1 + (1|arch) + (1|run) + (1|query) + (1|judge), "
                  "crossed (run nested in arch)"),
        "estimator": "statsmodels.MixedLM vc_formula, reml=True, lbfgs",
        "correction_note": ("B3 review fix: added (1|arch) so sigma2_run is within-"
                            "architecture replicate noise, not arch-confounded run share."),
        "converged": bool(mf.converged),
        "fit_warnings": _fit_warnings,
        "all_vc_strictly_positive": bool(
            all(mf.vcomp[i] > 0 for i in range(len(names))) and mf.scale > 0),
        "grand_mean": round(float(mf.fe_params["Intercept"]), 5),
        "sigma2_arch": round(s2["arch"], 6),
        "sigma2_run": round(s2["run"], 6),
        "sigma2_query": round(s2["query"], 6),
        "sigma2_judge": round(s2["judge"], 6),
        "sigma2_resid": round(s2_resid, 6),
        "var_fraction": {
            "arch": round(s2["arch"] / tot, 4) if tot > 0 else None,
            "run": round(s2["run"] / tot, 4) if tot > 0 else None,
            "query": round(s2["query"] / tot, 4) if tot > 0 else None,
            "judge": round(s2["judge"] / tot, 4) if tot > 0 else None,
            "resid": round(s2_resid / tot, 4) if tot > 0 else None,
        },
        "icc_arch": round(s2["arch"] / tot, 4) if tot > 0 else None,
        "icc_query": round(s2["query"] / tot, 4) if tot > 0 else None,
        "icc_judge": round(s2["judge"] / tot, 4) if tot > 0 else None,
        "icc_run": round(s2["run"] / tot, 4) if tot > 0 else None,
        "n_obs": int(len(long)),
    }
except Exception as e:  # noqa: BLE001
    three_way = {"_status": "error", "error": str(e)[:200]}

# ---------- 6. cross-judge anchors (descriptive, robust to REML fit) ----------
# Mean per-judge level + raw between-judge SD on the matched cells (identification-free check
# that the judge facet is real even if the REML judge VC is small with only 2 levels).
piv = long.pivot_table(index=["run_id", "qid"], columns="jid", values="score")
piv = piv.dropna()
judge_means = {j: round(float(long[long.jid == j].score.mean()), 5) for j in sorted(long.jid.unique())}
cross_judge = {
    "per_judge_mean_overall": judge_means,
    "mean_gpt52_minus_sonnet48": round(float(piv["gpt52"].mean() - piv["claude_sonnet48"].mean()), 5)
    if {"gpt52", "claude_sonnet48"} <= set(piv.columns) else None,
    "paired_cell_corr_pearson": round(float(piv.corr().iloc[0, 1]), 4) if piv.shape[1] == 2 else None,
    "n_paired_cells": int(len(piv)),
    "note": ("Judge facet has only TWO levels (gpt52, sonnet48), so the REML judge variance "
             "component is a single between-judge contrast and is estimated with low precision; "
             "the per-judge means and paired-cell correlation are reported as a robust, "
             "identification-free companion to sigma2_judge."),
}

out = {
    "_note": ("3-way crossed run x query x judge variance decomposition. EXTENDS "
              "variance_decomposition (gpt52-only run facet) by ingesting the on-disk Sonnet-4.6 "
              "replicate verdicts (judge_claude_sonnet48/base_p*_v*) and fitting the crossed REML "
              "on the balanced gpt52-and-sonnet intersection. No new judging (Sonnet verdicts read "
              "from disk); no paid API. Addresses single_judge_caveat: run and judge variance are "
              "now on a CROSSED substrate (limited to the 2-judge intersection)."),
    "prereg": "docs/publication/prereg/prereg_E2.md",
    "judges": ["gpt52", "claude_sonnet48"],
    "recompute": ("overall_score_recomputed (source-type weights), identical to "
                  "build_analysis_dataframes; Sonnet stored overall_score is corrupted and NOT used."),
    "self_check_recompute_vs_parquet": {"n_checked": SELF_CHECK["checked"],
                                        "max_abs_err": SELF_CHECK["max_abs_err"]},
    "coverage": coverage,
    "intersection": intersection,
    "reml_3way": three_way,
    "cross_judge_anchor": cross_judge,
    "limitations": ("(1) judge facet = 2 levels only -> sigma2_judge is a single contrast, low "
                    "precision; (2) fit on the balanced intersection, so power is set by the "
                    "Sonnet (thinner) coverage; (3) Sonnet source is claude_code_manual; "
                    "(4) this is the crossed EXTENSION, not a replacement for the gpt52 run-facet "
                    "estimate in variance_decomposition.components."),
}

# ---------- 7. persist: append three_way; preserve all existing keys ----------
cn = json.load(open(CANON))
vd = cn.get("variance_decomposition")
if not isinstance(vd, dict):
    raise SystemExit("ERROR: variance_decomposition missing/not a dict; run "
                     "build_variance_decomposition.py first (this script EXTENDS it).")
vd["three_way"] = out  # add/overwrite ONLY this sub-key; siblings untouched
cn["variance_decomposition"] = vd
_tmp = str(CANON) + ".tmp"
open(_tmp, "w").write(json.dumps(cn, indent=1))
os.replace(_tmp, str(CANON))

print(f"variance_decomposition.three_way: intersection cells={intersection['n_cells_run_query']} "
      f"runs={intersection['n_runs']} queries={intersection['n_queries']} judges={intersection['n_judges']}")
if three_way.get("_status") == "fit":
    print(f"  REML var: run={three_way['sigma2_run']} query={three_way['sigma2_query']} "
          f"judge={three_way['sigma2_judge']} resid={three_way['sigma2_resid']} "
          f"(converged={three_way['converged']})")
    print(f"  frac: {three_way['var_fraction']}")
else:
    print(f"  REML status={three_way.get('_status')} {three_way.get('error','')}")
print(f"  cross-judge: means={cross_judge['per_judge_mean_overall']} "
      f"paired_r={cross_judge['paired_cell_corr_pearson']} n_pairs={cross_judge['n_paired_cells']}")
print(f"  self-check recompute max_abs_err={SELF_CHECK['max_abs_err']:.2e} (n={SELF_CHECK['checked']})")
