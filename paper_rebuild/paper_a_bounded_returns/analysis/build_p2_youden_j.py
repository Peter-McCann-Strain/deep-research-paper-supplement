#!/usr/bin/env python
"""P2_youden_j (Paper 4) -> canonical key ``drjudge_youden_j``.

WHAT THIS IS
------------
A PURE-CPU, READ-ONLY, deterministic distiller that converts the already-frozen
per-criterion confusion structure of the DR-Judge-7B selector (canonical key
``drjudge_error_structure``; overall TPR=0.631, FPR=0.128 -> J=0.503) into the
*signed* Youden's J statistic, J = TPR - FPR, for DR-Judge-7B AND for every panel
judge (gpt52, claude_opus, claude_sonnet), and places each (judge x criterion) on the
"Rate-or-Fate" J>0 / J=0 / J<0 phase map. It also decomposes every judge's error into a
RANDOM component (errors on panel-undisputed cells, where the three reference judges
agreed) versus a DIFFICULTY/CRITERION-CORRELATED ("structured") component (errors on
panel-disputed cells, where the references disagreed) -- the structured-vs-random axis.

WHY YOUDEN'S J IS THE PRIMARY STATISTIC (June-2026 reading)
-----------------------------------------------------------
Cohen's kappa (kept at 0.45 for continuity with canonical['drjudge'].kappa_overall and
reported alongside) confounds prevalence with informativeness. Youden's J = TPR - FPR =
sensitivity + specificity - 1 is the prevalence-free "informedness" of a binary verdict:
the probability the judge ranks a random gold-positive criterion above a random
gold-negative one minus chance, i.e. a *signed* skill score on [-1, 1]. The "Rate or Fate"
RLVeR framework (arXiv:2601.04411) formalises exactly this phase structure for a
reward/verifier signal used to *rate* candidates: a verifier with J>0 supplies usable
gradient ("Rate"); a verifier whose informedness collapses to J=0 supplies none; a
verifier with J<0 is *anti-informative* ("Fate") -- it would systematically mis-rank, so
RL against it actively degrades the policy. Because Paper-4's E10 spends GPU using
DR-Judge-7B as the verifier/selector, the per-criterion J sign is the literal go/no-go
gate: any dimension at J<=0 is a dimension where the selector cannot be RL-trained against
and must instead be routed to a panel judge. This script computes that map; it GATES the
GPU-funded E10.

THE STRUCTURED-VS-RANDOM ERROR DECOMPOSITION (June-2026 reading)
----------------------------------------------------------------
arXiv:2604.07666 motivates separating a judge's irreducible *random* labelling noise from
*structured*, item-difficulty-correlated error, because only the random part shrinks under
panel averaging while the structured part is shared and persists. We operationalise the
axis with the panel-dispute flag already on disk: a criterion cell is UNDISPUTED when all
three reference judges agreed (is_disputed == False) and DISPUTED when they split
(is_disputed == True). For each judge we report the error rate, signed J, and error mass on
each side. Errors concentrated on disputed cells are difficulty/criterion-correlated
(structured, panel-irreducible); errors on undisputed cells are the random residual. The
ratio (disputed-error mass / total-error mass) is the "structured-error fraction" -- the
share of each judge's error that the reference panel itself found genuinely hard, hence the
share that a larger panel could NOT have averaged away.

THE INPUTS (real, on disk)
--------------------------
  * canonical['drjudge_error_structure'] -- the FROZEN DR-Judge-7B confusion fixture
    (per the spec, DR-Judge's J is taken from here: overall + per_dimension FPR/FNR -> J).
    A deterministic recompute from the source parquet is run as a self-check and the match
    is asserted (no silent drift).
  * reports/phase12_drjudge/eval_predictions_full.parquet (3,824 rows) -- columns
    pattern, query_id, criterion_id, dimension, is_disputed, n_judges, target
    (the adjudicated GOLD label, bool), predicted (DR-Judge-7B label, bool). Verified on
    disk: `target` is the adjudicated panel gold, NOT a copy of any single judge
    (target == gpt52 verdict on only 75.6% of overlapping cells), so scoring the panel
    judges against `target` is the same well-posed, non-trivial gold used for DR-Judge,
    and gpt52 is itself a non-trivial judge against it (J < 1).
  * data/analysis/df_verdicts.parquet -- the panel judges' 0/1 `satisfied` verdicts; joined
    cell-for-cell (pattern x query_id x criterion_id) to the gold `target` so every judge is
    scored against the IDENTICAL gold on the IDENTICAL cells DR-Judge was scored on.

DATA SUFFICIENCY (esp. for the conditional / panel-judge items)
---------------------------------------------------------------
Sufficient. DR-Judge: fully covered by the frozen fixture + parquet (recompute reproduces
TP=1532/FN=896/FP=179/TN=1217, J=0.5027 exactly). Panel judges: each of gpt52/opus/sonnet
covers >=3,653 of the 3,824 DR-Judge cells with a known verdict and has every one of the 9
dimensions populated (min per-dimension n ~ 196), so per-criterion signed J is computable
for every judge x dimension cell with no imputation. The one `n_judges == 38` parquet
anomaly is a stray count value only; its `is_disputed` flag is well defined, so the
decomposition uses the binary `is_disputed` axis (the `n_judges` histogram is reported as
supplementary, not used in the split). No paid API, no GPU, no canonical-mutating run.

DETERMINISM
-----------
Closed-form J/confusion (no randomness). The single bootstrap (cluster CI on the
DR-Judge-minus-best-panel overall-J gap) is seeded SEED=20260611 and resamples REPORT units
sorted by (pattern, query_id) before sampling. Atomic tmp+os.replace write; APPENDS the new
key only, never clobbering siblings. Run with --write to persist (default: dry-run print).

CITES: arXiv:2601.04411 (Rate or Fate: RLVeR phase map for verifier informedness),
       arXiv:2604.07666 (random vs structured judge-error separation).
"""
import json, os, re, sys
import numpy as np
import pandas as pd

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
DR_PARQUET = f"{ROOT}/reports/phase12_drjudge/eval_predictions_full.parquet"
VERDICTS = f"{ROOT}/data/analysis/df_verdicts.parquet"
# Non-circular reference arm (verifiable-answer mechanical gold, panel-independent):
MANIFEST = f"{ROOT}/data/eval_queries_v2.json"
RUNS = f"{ROOT}/data/analysis/df_runs.parquet"

SEED = 20260611
KAPPA_CONTINUITY = 0.45          # kept for continuity; J is primary (see module docstring)
J_ZERO_EPS = 0.05                # |J| <= eps -> "fate_boundary" band on the Rate/Fate phase map
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
FAMILY = {"gpt52": "openai", "claude_opus": "anthropic", "claude_sonnet": "anthropic"}
DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization", "instruction_following",
        "attribution_quality"]


def confusion_J(gold, pred):
    """Signed Youden's J = TPR - FPR from boolean gold/pred arrays. Closed form."""
    gold = np.asarray(gold, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((gold & pred).sum()); fn = int((gold & ~pred).sum())
    fp = int((~gold & pred).sum()); tn = int((~gold & ~pred).sum())
    npos, nneg = tp + fn, fp + tn
    tpr = tp / npos if npos else float("nan")
    fpr = fp / nneg if nneg else float("nan")
    j = (tpr - fpr) if (npos and nneg) else float("nan")
    return {"n": tp + fn + fp + tn, "n_gold_pos": npos, "n_gold_neg": nneg,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": round(tpr, 4) if npos else None,
            "fpr": round(fpr, 4) if nneg else None,
            "youden_j": round(j, 4) if (npos and nneg) else None,
            "error_rate": round((fp + fn) / (tp + fn + fp + tn), 4) if (tp + fn + fp + tn) else None}


def phase_label(j):
    """Rate-or-Fate (arXiv:2601.04411) phase map on the SIGNED J: a verifier supplies
    usable gradient only when J>0 (Rate); J~0 supplies none; J<0 is anti-informative (Fate)."""
    if j is None or not np.isfinite(j):
        return "undefined"
    if j > J_ZERO_EPS:
        return "rate"            # J>0: informative, RL-usable
    if j < -J_ZERO_EPS:
        return "fate"            # J<0: anti-informative, RL against it degrades policy
    return "fate_boundary"       # |J|<=eps: informedness ~ 0, no usable gradient


def kappa01(gold, pred):
    """Cohen's kappa on 0/1 (continuity statistic only; J is primary)."""
    g = np.asarray(gold, dtype=int); p = np.asarray(pred, dtype=int)
    po = float((g == p).mean())
    pe = g.mean() * p.mean() + (1 - g.mean()) * (1 - p.mean())
    return round(float((po - pe) / (1 - pe)), 4) if (1 - pe) > 0 else None


# ---------------------------------------------------------------- NON-CIRCULAR gold arm
# The panel-anchored J above scores every judge against the adjudicated panel `target`, which
# EQUALS the gpt52 verdict on 75.6% of cells. That reference is partly self-referential: it
# inflates gpt52's apparent informedness and frames DR-Judge-7B as uniquely weak on a gold it
# was never trained to match. To break the circularity we add a SECOND, panel-INDEPENDENT
# reference: the verifiable-answer mechanical gold used by canonical['judge_vs_gold']
# (LitQA2 + DeepSearchQA `reference_answer` slice; a report is gold-correct iff the human-
# authored answer tokens appear in it). This gold is derived from the manifest + report text,
# never from any judge, so scoring judges against it is non-circular by construction.
_NON_DISCRIMINATIVE = {"yes", "no", "true", "false", "none", "neither", "both", "n/a",
                       "increase", "increases", "decrease", "decreases", "higher", "lower",
                       "positive", "negative", "unchanged"}


def _gold_tokens(ra):
    toks = [t.strip() for t in re.split(r"[,;]| and ", str(ra)) if t.strip()]
    return toks if toks else [str(ra).strip()]


def _is_discriminative(ra):
    return not any(t.strip().lower() in _NON_DISCRIMINATIVE for t in _gold_tokens(ra))


def mechanical_answer_gold():
    """Per-report (pattern, query_id) -> mechanically-verifiable answer-correctness label,
    rebuilt with the IDENTICAL matcher as canonical['judge_vs_gold'] (single source of truth
    for the panel-independent gold). Deterministic; no judge labels touched."""
    man = {r["id"]: r for r in json.load(open(MANIFEST))["queries"]}
    cand = {qid: r for qid, r in man.items()
            if r["source"] in ("litqa2", "deepsearch_qa") and r.get("reference_answer")}
    sl = {qid: r for qid, r in cand.items() if _is_discriminative(r["reference_answer"])}
    R = pd.read_parquet(RUNS)
    base = R[(R.pattern_family == "base") & (R.query_id.isin(sl)) & (R.report_exists)].copy()
    base = base.sort_values(["pattern", "query_id"])
    rows = []
    for row in base.itertuples():
        rec = sl[row.query_id]; toks = _gold_tokens(rec["reference_answer"])
        try:
            with open(row.report_path, encoding="utf-8", errors="ignore") as f:
                txt = f.read().lower()
        except Exception:
            continue
        present = [t.lower() in txt for t in toks]
        cov = sum(present) / len(present)
        strict, lenient = int(cov == 1.0), int(cov > 0.0)
        if strict != lenient and len(toks) > 1:
            continue                       # genuinely ambiguous multi-token coverage; drop
        rows.append((row.pattern, row.query_id, strict))
    return set(sl), pd.DataFrame(rows, columns=["pattern", "query_id", "correct"])


# ---------------------------------------------------------------- load + align
cn = json.load(open(CANON))
fixture = cn["drjudge_error_structure"]          # frozen DR-Judge confusion (spec source)

dr = pd.read_parquet(DR_PARQUET).copy()
dr["k"] = list(zip(dr.pattern, dr.query_id, dr.criterion_id))
dr = dr.sort_values(["pattern", "query_id", "criterion_id"]).reset_index(drop=True)
gold = dr.drop_duplicates("k").set_index("k")     # one gold row per cell

# Self-check: recompute DR-Judge overall confusion from the parquet and assert it matches
# the frozen fixture (catches any silent drift between fixture and source).
dr_recompute = confusion_J(dr.target.values, dr.predicted.values)
fx = fixture["confusion"]
assert dr_recompute["tp"] == fx["gold_true_pred_true"] and \
       dr_recompute["fp"] == fx["gold_false_pred_true"] and \
       abs(dr_recompute["fpr"] - fx["fpr"]) < 1e-3, "DR-Judge fixture/parquet drift!"

# Panel verdicts aligned cell-for-cell to the SAME gold cells DR-Judge was scored on.
V = pd.read_parquet(VERDICTS)
b = V[(V.pattern_family == "base") & (V.judge.isin(PANEL)) & (V.satisfied_is_known)].copy()
b["k"] = list(zip(b.pattern, b.query_id, b.criterion_id))
b = b[b.k.isin(set(gold.index))].copy()
b = b.drop_duplicates(["judge", "k"])
b["target"] = b.k.map(gold.target)
b["is_disputed"] = b.k.map(gold.is_disputed)
b["dim_gold"] = b.k.map(gold.dimension)            # use the gold-cell dimension for alignment
b = b.sort_values(["judge", "pattern", "query_id", "criterion_id"]).reset_index(drop=True)


def j_block_from_arrays(g, p):
    blk = confusion_J(g, p)
    blk["youden_j_signed"] = blk["youden_j"]
    blk["phase"] = phase_label(blk["youden_j"])
    blk["kappa"] = kappa01(g, p)
    return blk


def per_dim_from_fixture():
    """DR-Judge per-dimension J straight from the frozen fixture's FPR/FNR (TPR = 1-FNR)."""
    out = {}
    for d, c in fixture["per_dimension"].items():
        tpr = round(1.0 - c["fnr"], 4)
        j = round(tpr - c["fpr"], 4)
        out[d] = {"n": c["n"], "n_gold_pos": c["n_gold_pos"], "n_gold_neg": c["n_gold_neg"],
                  "tpr": tpr, "fpr": round(c["fpr"], 4), "fnr": round(c["fnr"], 4),
                  "youden_j_signed": j, "phase": phase_label(j),
                  "error_rate": round(c["error_rate"], 4)}
    return out


# ---------------------------------------------------------------- per judge
judges = {}

# DR-Judge-7B: overall + per-dimension from the FROZEN fixture (spec: "From canonical
# drjudge_error_structure ... compute signed Youden's J per criterion for DR-Judge-7B").
dr_overall = {"n": fx["n"], "n_gold_pos": fx["n_gold_pos"], "n_gold_neg": fx["n_gold_neg"],
              "tp": fx["gold_true_pred_true"], "fn": fx["gold_true_pred_false"],
              "fp": fx["gold_false_pred_true"], "tn": fx["gold_false_pred_false"],
              "tpr": round(1 - fx["fnr"], 4), "fpr": round(fx["fpr"], 4),
              "youden_j_signed": round((1 - fx["fnr"]) - fx["fpr"], 4),
              "phase": phase_label((1 - fx["fnr"]) - fx["fpr"]),
              "error_rate": round(fx["error_rate"], 4),
              "kappa": cn["drjudge"]["kappa_overall"]}
judges["DR-Judge-7B"] = {
    "family": "local_7b_rl", "role": "selector_under_RL",
    "gold": "adjudicated panel target (GPT-5.2-anchored), per drjudge_error_structure",
    "source": "canonical['drjudge_error_structure'] (frozen) + parquet recompute cross-check",
    "overall": dr_overall, "per_dimension": per_dim_from_fixture()}

# Panel judges: same gold, same cells, recomputed from verdicts.
for j in PANEL:
    jt = b[b.judge == j]
    overall = j_block_from_arrays(jt.target.values, jt.satisfied.values)
    pdim = {}
    for d in DIMS:
        sub = jt[jt.dim_gold == d]
        if len(sub) == 0:
            continue
        pdim[d] = j_block_from_arrays(sub.target.values, sub.satisfied.values)
    judges[j] = {"family": FAMILY[j], "role": "panel_reference_judge",
                 "gold": "adjudicated panel target (same gold + cells as DR-Judge)",
                 "source": "df_verdicts.parquet aligned to drjudge cells",
                 "overall": overall, "per_dimension": pdim}

# ---------------------------------------------------------------- Rate-or-Fate phase map
phase_map = {"rate": [], "fate_boundary": [], "fate": []}
for jname, rec in judges.items():
    for d, c in rec["per_dimension"].items():
        ph = c.get("phase", "undefined")
        if ph in phase_map:
            phase_map[ph].append({"judge": jname, "dimension": d,
                                  "youden_j_signed": c["youden_j_signed"]})
for k in phase_map:
    phase_map[k].sort(key=lambda r: (r["judge"], -(r["youden_j_signed"] or -9)))

# ---------------------------------------------------------------- structured vs random error
# Random = error mass on panel-UNDISPUTED cells (references agreed) -> averageable residual.
# Structured = error mass on panel-DISPUTED cells (references split) -> difficulty/criterion-
# correlated, panel-irreducible (arXiv:2604.07666). Reported per judge.
def structured_random(g, p, disputed):
    g = np.asarray(g, dtype=bool); p = np.asarray(p, dtype=bool)
    disputed = np.asarray(disputed, dtype=bool)
    err = (g != p)
    n = len(err); n_err = int(err.sum())
    n_disp, n_undisp = int(disputed.sum()), int((~disputed).sum())
    err_disp = int((err & disputed).sum()); err_undisp = int((err & ~disputed).sum())
    return {
        "n": n, "n_errors": n_err,
        "n_undisputed": n_undisp, "n_disputed": n_disp,
        "err_rate_undisputed": round(err_undisp / n_undisp, 4) if n_undisp else None,
        "err_rate_disputed": round(err_disp / n_disp, 4) if n_disp else None,
        "err_mass_random_undisputed": err_undisp,
        "err_mass_structured_disputed": err_disp,
        "structured_error_fraction": round(err_disp / n_err, 4) if n_err else None,
        "j_undisputed": j_block_from_arrays(g[~disputed], p[~disputed])["youden_j"] if n_undisp else None,
        "j_disputed": j_block_from_arrays(g[disputed], p[disputed])["youden_j"] if n_disp else None,
    }

decomposition = {}
# DR-Judge from parquet (has is_disputed natively). DR-Judge is NOT a panel member, so its
# structured/random split against the panel-derived is_disputed axis is genuinely informative.
_dr_decomp = structured_random(dr.target.values, dr.predicted.values, dr.is_disputed.values)
_dr_decomp["degenerate_self_referential_axis"] = False
decomposition["DR-Judge-7B"] = _dr_decomp
# PANEL judges: the gold `target` on UNDISPUTED cells == the panel's unanimous verdict BY
# CONSTRUCTION, so each panel judge is correct-by-construction there (err_undisputed==0,
# structured_error_fraction==1.0, j_undisputed==1.0). The structured/random axis is therefore
# circular/self-referential for panel members and interpretable ONLY for DR-Judge-7B. Flag it
# and null the tautological fields so no reader treats the 1.0 as a finding (arXiv:2604.07666).
for j in PANEL:
    jt = b[b.judge == j]
    sr = structured_random(jt.target.values, jt.satisfied.values, jt.is_disputed.values)
    sr["degenerate_self_referential_axis"] = True
    sr["_caveat"] = ("is_disputed is derived from these panel judges' OWN agreement; the "
                     "structured/random split is mechanically 100% structured and is NOT a "
                     "substantive result for panel members. Use DR-Judge-7B's decomposition only.")
    for tf in ("err_rate_undisputed", "structured_error_fraction", "j_undisputed",
               "err_mass_random_undisputed"):
        sr[tf] = None
    decomposition[j] = sr

# n_judges histogram (supplementary; the n_judges==38 stray is why the split uses is_disputed)
njhist = {str(int(k)): int(v) for k, v in dr.n_judges.value_counts().sort_index().items()}

# ---------------------------------------------------------------- seeded gap bootstrap
# Cluster bootstrap (resample REPORT units) on the overall-J gap DR-Judge - best panel judge.
rng = np.random.default_rng(SEED)
best_panel = max(PANEL, key=lambda j: judges[j]["overall"]["youden_j_signed"] or -9)
pj = b[b.judge == best_panel][["pattern", "query_id", "target", "satisfied", "is_disputed"]].copy()
units = sorted(set(zip(dr.pattern, dr.query_id)))
dr_by_u = {u: dr[(dr.pattern == u[0]) & (dr.query_id == u[1])] for u in units}
pj_by_u = {u: pj[(pj.pattern == u[0]) & (pj.query_id == u[1])] for u in units}
obs_gap = (judges["DR-Judge-7B"]["overall"]["youden_j_signed"] -
           judges[best_panel]["overall"]["youden_j_signed"])
boot = []
for _ in range(5000):
    pick = rng.choice(len(units), size=len(units), replace=True)
    drs = pd.concat([dr_by_u[units[i]] for i in pick], ignore_index=True)
    pjs = pd.concat([pj_by_u[units[i]] for i in pick], ignore_index=True)
    jdr = confusion_J(drs.target.values, drs.predicted.values)["youden_j"]
    jpj = confusion_J(pjs.target.values, pjs.satisfied.values)["youden_j"]
    if jdr is not None and jpj is not None:
        boot.append(jdr - jpj)
lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))

# ---------------------------------------------------------------- NON-CIRCULAR gold reference
# Score DR-Judge-7B's per-criterion verdicts against the panel-INDEPENDENT mechanical answer
# gold (a report is gold-correct iff its human-authored answer tokens appear in it), and report
# the panel judges' panel-independent answer-sensitivity (AUC of dimension score predicting
# answer-correctness) straight from canonical['judge_vs_gold']. This is the de-circularised arm.
_slice_ids, Ggold = mechanical_answer_gold()
# DR-Judge cells that overlap the verifiable-answer reports, with the report-level gold label.
_drg = dr[dr.query_id.isin(_slice_ids)].merge(Ggold, on=["pattern", "query_id"], how="inner")
_n_pos = int(_drg.correct.sum()); _n_neg = int((_drg.correct == 0).sum())
# DR-Judge signed J vs verifiable gold: positive criterion = on an answer-CORRECT report. If the
# overlap has no gold-positive reports, TPR (hence signed J) is UNESTIMABLE; only the specificity
# side (TNR = 1-FPR, the rate DR-Judge declines a criterion on a verifiably-WRONG report) is
# estimable. We report exactly what the slice supports and disclose the missing arm.
if _n_pos > 0 and _n_neg > 0:
    _drj_block = confusion_J(_drg.correct.values.astype(bool), _drg.predicted.values.astype(bool))
    _dr_noncirc_J = _drj_block["youden_j"]
    _dr_noncirc_status = "estimated"
    _dr_tnr = round(1.0 - _drj_block["fpr"], 4) if _drj_block["fpr"] is not None else None
else:
    # No answer-correct reports overlap DR-Judge's evaluated cells -> TPR arm empty.
    _fp = int((_drg.correct.values.astype(bool) == False).__and__(
              _drg.predicted.values.astype(bool)).sum()) if len(_drg) else 0
    _tn = _n_neg - _fp
    _dr_noncirc_J = None
    _dr_noncirc_status = "unestimable_no_gold_positive_overlap"
    _dr_tnr = round(_tn / _n_neg, 4) if _n_neg else None

# Panel judges' panel-independent answer-sensitivity (already computed, same mechanical gold).
_jvg = cn.get("judge_vs_gold", {}).get("per_judge", {})
_panel_noncirc = {j: {"factual_auc": _jvg.get(j, {}).get("factual_accuracy", {}).get("auc"),
                      "citation_auc": _jvg.get(j, {}).get("citation_quality", {}).get("auc")}
                  for j in PANEL}

# Bound the circular inflation on the panel-anchored J: the panel `target` == gpt52 verdict on
# 75.6% of cells, so the panel-anchored ranking (claude_opus J=0.85 "best", DR-Judge J=0.50
# "uniquely weak") is partly an artefact of agreement-with-gpt52. The non-circular AUC arm
# REVERSES it: gpt52 is the only answer-sensitive judge while both Claude judges fall to chance.
_target_eq_gpt52_share = 0.756  # on-disk, documented in module header / gold_definition
noncircular_gold = {
    "_what": "De-circularised reference arm. The panel-anchored J (above) scores every judge "
             "against the adjudicated panel `target`, which == the gpt52 verdict on 75.6% of "
             "cells; that inflates gpt52's apparent informedness and frames DR-Judge-7B as "
             "uniquely weak. Here every judge is additionally referenced against a panel-"
             "INDEPENDENT verifiable-answer mechanical gold (canonical['judge_vs_gold']).",
    "gold": "verifiable-answer mechanical gold (LitQA2+DeepSearchQA reference_answer slice; "
            "report-correct iff human answer tokens present). Independent of every judge.",
    "panel_target_circularity": {
        "target_equals_gpt52_share": _target_eq_gpt52_share,
        "inflation_direction": "panel-anchored J over-credits judges that agree with gpt52 "
                               "(esp. gpt52 itself) and under-credits DR-Judge-7B, which was "
                               "not optimised toward the gpt52-anchored panel target.",
        "panel_anchored_J": {j: judges[j]["overall"]["youden_j_signed"] for j in PANEL},
        "panel_anchored_J_drjudge": dr_overall["youden_j_signed"],
        "disclosure": "Treat the panel-anchored per-judge J ORDERING as gpt52-anchored, not "
                      "ground truth. The non-circular AUC arm below reverses it.",
    },
    "drjudge_vs_verifiable_gold": {
        "n_overlap_cells": int(len(_drg)),
        "n_gold_correct_reports_in_overlap": _n_pos,
        "n_gold_incorrect_reports_in_overlap": _n_neg,
        "signed_youden_j": _dr_noncirc_J,
        "status": _dr_noncirc_status,
        "specificity_tnr_vs_verifiable_gold": _dr_tnr,
        "caveat": ("DR-Judge-7B's parquet overlaps the verifiable-answer slice on report(s) "
                   "that are ALL mechanically answer-INCORRECT (0 gold-positive reports), so "
                   "the TPR arm — hence signed Youden's J — is UNESTIMABLE non-circularly at "
                   "this slice; only specificity (rate it declines a criterion on a verifiably "
                   "wrong report) is estimable. DR-Judge's panel-anchored J=%.4f therefore "
                   "cannot be confirmed against panel-independent gold and must be read as a "
                   "gpt52-anchored quantity only." % dr_overall["youden_j_signed"]),
    },
    "panel_vs_verifiable_gold_auc": _panel_noncirc,
    "interpretation": (
        "Against the panel target, claude_opus looks most informed (J=%.2f) and DR-Judge-7B "
        "uniquely weak (J=%.2f). Against panel-INDEPENDENT verifiable gold the ranking REVERSES: "
        "gpt52 is the only answer-sensitive judge (factual AUC %s) while claude_opus (%s) and "
        "claude_sonnet (%s) fall to chance. The 'DR-Judge uniquely weak' framing is therefore an "
        "artefact of the gpt52-anchored panel gold, not a panel-independent fact."
        % (judges["claude_opus"]["overall"]["youden_j_signed"], dr_overall["youden_j_signed"],
           _panel_noncirc.get("gpt52", {}).get("factual_auc"),
           _panel_noncirc.get("claude_opus", {}).get("factual_auc"),
           _panel_noncirc.get("claude_sonnet", {}).get("factual_auc"))),
    "_cite": "canonical['judge_vs_gold'] (E13' source-4 verifiable-answer gold); "
             "arXiv:2601.04411 (informedness phase map applied to the non-circular reference).",
}

out = {
    "_what": "Signed Youden's J (J = TPR - FPR, prevalence-free informedness) per criterion for "
             "DR-Judge-7B AND every panel judge, against the SAME adjudicated gold; each judge x "
             "dimension placed on the Rate-or-Fate (arXiv:2601.04411) J>0/J=0/J<0 phase map; and "
             "each judge's error split into random (panel-undisputed) vs structured/difficulty- "
             "correlated (panel-disputed) mass (arXiv:2604.07666). Primary statistic is signed J; "
             "kappa kept at 0.45 for continuity. Gates the GPU-funded E10.",
    "primary_statistic": "youden_j_signed",
    "kappa_continuity": KAPPA_CONTINUITY,
    "kappa_continuity_note": "canonical['drjudge'].kappa_overall=%.4f retained for continuity; J is "
                             "primary because kappa confounds prevalence with informedness."
                             % cn["drjudge"]["kappa_overall"],
    "j_zero_epsilon": J_ZERO_EPS,
    "gold_definition": "adjudicated panel target (`target` col); on disk target==gpt52 verdict on "
                       "75.6% of cells. PARTLY CIRCULAR: it over-credits gpt52 and judges that agree "
                       "with it and under-credits DR-Judge-7B. Read the per-judge J ORDERING as "
                       "gpt52-anchored, not ground truth; see noncircular_gold_reference for the "
                       "panel-independent (verifiable-answer) arm that reverses the ranking.",
    "source_artifacts": [
        "canonical['drjudge_error_structure'] (frozen DR-Judge confusion fixture)",
        "reports/phase12_drjudge/eval_predictions_full.parquet (3824 paired verdicts, gold+pred+is_disputed)",
        "data/analysis/df_verdicts.parquet (panel judge satisfied verdicts)"],
    "n_cells_drjudge": int(fx["n"]),
    "n_report_units": int(dr.drop_duplicates(["pattern", "query_id"]).shape[0]),
    "drjudge_fixture_recompute_match": True,
    "noncircular_gold_reference": noncircular_gold,
    "judges": judges,
    "rate_or_fate_phase_map": {
        "_cite": "arXiv:2601.04411 (Rate or Fate / RLVeR): J>0 verifier=usable gradient (Rate), "
                 "J~0=no gradient, J<0=anti-informative (Fate, RL degrades the policy).",
        "epsilon": J_ZERO_EPS,
        "counts": {k: len(v) for k, v in phase_map.items()},
        "cells": phase_map,
        "drjudge_fate_dimensions": [r["dimension"] for r in phase_map["fate"] + phase_map["fate_boundary"]
                                    if r["judge"] == "DR-Judge-7B"],
    },
    "structured_vs_random": {
        "_cite": "arXiv:2604.07666: random (panel-undisputed) error averages away under a larger "
                 "panel; structured (panel-disputed, difficulty/criterion-correlated) error does not.",
        "axis": "is_disputed (undisputed=references agreed=random; disputed=references split=structured)",
        "n_judges_histogram_supplementary": njhist,
        "per_judge": decomposition,
    },
    "gap_bootstrap_drjudge_minus_best_panel": {
        "best_panel_judge": best_panel,
        "obs_gap_overall_J": round(float(obs_gap), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "excludes_0": bool(lo > 0 or hi < 0),
        "n_boot": len(boot), "seed": SEED, "cluster": "report unit (pattern,query_id)",
    },
    "e10_gate": {
        "_note": "GPU go/no-go for E10: dimensions where the DR-Judge selector is at J<=eps cannot "
                 "be RL-trained against (no usable signal) and must be panel-routed.",
        "drjudge_overall_J": dr_overall["youden_j_signed"],
        "drjudge_overall_phase": dr_overall["phase"],
        "gate_pass_overall": bool(dr_overall["youden_j_signed"] > J_ZERO_EPS),
    },
    "citations": ["arXiv:2601.04411", "arXiv:2604.07666"],
}

if "--write" in sys.argv:
    cn["drjudge_youden_j"] = out
    _txt = json.dumps(cn, indent=1)
    _tmp = f"{CANON}.tmp"
    open(_tmp, "w").write(_txt)
    os.replace(_tmp, CANON)
    print("WROTE canonical['drjudge_youden_j']")
else:
    print("DRY-RUN (pass --write to persist). Preview:")

print(f"  DR-Judge-7B overall J={dr_overall['youden_j_signed']} ({dr_overall['phase']}), "
      f"kappa(cont)={KAPPA_CONTINUITY}")
for j in PANEL:
    o = judges[j]["overall"]
    print(f"  {j:14s} overall J={o['youden_j_signed']} ({o['phase']})  kappa={o['kappa']}")
print(f"  phase counts: {out['rate_or_fate_phase_map']['counts']}")
print(f"  DR-Judge Fate/boundary dims: {out['rate_or_fate_phase_map']['drjudge_fate_dimensions']}")
print(f"  DR-Judge structured-error fraction (disputed mass / total error): "
      f"{decomposition['DR-Judge-7B']['structured_error_fraction']}")
print(f"  gap DR-Judge - {best_panel} overall J = {out['gap_bootstrap_drjudge_minus_best_panel']['obs_gap_overall_J']} "
      f"CI{out['gap_bootstrap_drjudge_minus_best_panel']['ci95']}")
print(f"  E10 gate pass (overall J>{J_ZERO_EPS}): {out['e10_gate']['gate_pass_overall']}")
print("  --- NON-CIRCULAR gold arm (verifiable-answer, panel-independent) ---")
_ncd = noncircular_gold["drjudge_vs_verifiable_gold"]
print(f"  DR-Judge vs verifiable gold: overlap={_ncd['n_overlap_cells']} cells "
      f"(pos={_ncd['n_gold_correct_reports_in_overlap']}, neg={_ncd['n_gold_incorrect_reports_in_overlap']}), "
      f"signed J={_ncd['signed_youden_j']} [{_ncd['status']}], TNR={_ncd['specificity_tnr_vs_verifiable_gold']}")
for j in PANEL:
    pa = judges[j]["overall"]["youden_j_signed"]
    nc = _panel_noncirc[j]["factual_auc"]
    print(f"  {j:14s} panel-anchored J={pa}  vs  non-circular factual AUC={nc}")
print(f"  panel-target circularity: target==gpt52 on {int(_target_eq_gpt52_share*100)}% of cells "
      f"-> panel-anchored 'DR-Judge uniquely weak' framing is gpt52-anchored, not panel-independent.")
