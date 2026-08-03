#!/usr/bin/env python
"""P2_judge_kappa (Paper 2) — chance-corrected calibration of each judge vs the human gold slice.

Companion to build_judge_vs_gold.py. That script reports DISCRIMINATION (AUC / point-biserial /
bootstrap mean-gap): "do answer-correct reports get higher factual scores than incorrect ones?".
Discrimination is rank-only and threshold-free, so it can be high even when the judge and the gold
DISAGREE on the actual label of most reports. This script adds the missing CALIBRATION /
chance-corrected AGREEMENT number:

    weighted Cohen's kappa (ordinal, quadratic weights) and Krippendorff's alpha of each judge's
    (thresholded / binned) factual_accuracy verdict vs the mechanical human gold, on the SAME
    LitQA2+DeepSearchQA verifiable-answer slice that judge_vs_gold already builds.

WHY this is the right second number (citation: arXiv:2510.09738, June-2026):
  2510.09738 ("Beyond the high-AUC illusion ...") shows, for LLM-judge / classifier validity, that
  a high AUC can co-exist with poor chance-corrected agreement: AUC ranks pairs, but kappa/alpha
  ask whether the judge and the reference assign the SAME label net of chance agreement. A judge
  can rank correct>incorrect (AUC>0.6) yet, at any single operating threshold, agree with the gold
  barely above chance (kappa~0). We therefore report kappa/alpha ALONGSIDE the existing AUCs in
  judge_vs_gold.per_judge so a reader cannot read the AUC as if it were agreement.

PRE-STATED BANDS (registered here, before reading the result — Landis-Koch style, with a
high-stakes overlay):
  kappa/alpha  >= 0.80  : near-gold, adequate even for high-stakes / safety auto-grading (~0.85
                          is the conventional high-stakes floor; we flag >=0.85 explicitly)
               0.60-0.80 : SUBSTANTIAL — defensible as a primary validity anchor
               0.40-0.60 : moderate
               0.20-0.40 : fair
               <  0.20   : poor / near-chance
The pre-stated expectation (consistent with the AUC asymmetry already in judge_vs_gold) is that
even the answer-sensitive judge (GPT-5.2) lands BELOW the 0.60 substantial band on this slice,
i.e. high AUC but poor kappa — the 2510.09738 illusion, demonstrated on our own panel.

CONSTRUCTION (two pre-stated discretisations; both reported, no post-hoc pick):
  binary    — judge "calls correct" iff its continuous factual_accuracy score is >= the gold-slice
              MEDIAN for that judge (a fixed, judge-internal operating point; no gold peeking). A
              2x2 table vs binary gold. On a 2-level scale quadratic-weighted kappa == unweighted
              Cohen's kappa, so this is the canonical chance-corrected agreement; Krippendorff alpha
              is computed nominal (== binary).
  ordinal3  — judge factual_accuracy binned into 3 ordinal levels {0,1,2} by its gold-slice
              TERTILES; gold mapped to the ordinal endpoints {incorrect->0, correct->2}. This
              genuinely exercises the ORDINAL quadratic-weighted kappa and ordinal Krippendorff
              alpha (a judge in the MIDDLE band is penalised less than one at the wrong extreme).
  Thresholds (median, tertiles) are JUDGE-internal score quantiles — they do NOT use the gold
  labels, so the operating point is not tuned to inflate agreement.

Honest scope (inherits judge_vs_gold's caveats): the gold is GENERAL mechanical answer-presence on
a small verifiable slice; factual_accuracy aggregates general criteria. This measures whether a
judge's factual scoring, AT AN OPERATING POINT, agrees with verifiable answer-correctness net of
chance — it anchors, it does not certify. Citation_quality is carried as the same non-answer-
determined contrast used in judge_vs_gold.

Reuses the judge_vs_gold fixture join (identical SLICE + mechanical-gold construction, so the n's
match canonical['judge_vs_gold']['n_reports_with_gold']). CPU-only, $0, no API.
Writes canonical_numbers.json['judge_vs_gold']['per_judge'][judge]['calibration'] (per dimension)
and a top-level canonical_numbers.json['judge_vs_gold']['calibration'] summary block. APPENDS only;
never clobbers the existing AUC fields or any other canonical key. Atomic tmp+os.replace write.
Determinism: SEED=20260611, all inputs sorted before any resample; bootstrap CI on kappa seeded.
"""
import json, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260611
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
FAMILY = {"gpt52": "openai", "claude_opus": "anthropic", "claude_sonnet": "anthropic"}
DIMS = ["factual_accuracy", "citation_quality"]
CITATION = "arXiv:2510.09738"  # high AUC can co-exist with poor chance-corrected agreement
# Pre-stated Landis-Koch bands with a high-stakes overlay (registered before reading results).
BANDS = {"poor": "<0.20", "fair": "0.20-0.40", "moderate": "0.40-0.60",
         "substantial": "0.60-0.80", "near_gold": ">=0.80", "high_stakes_floor": ">=0.85"}

# ---------- load (mirror build_judge_vs_gold.py exactly) ----------
man = {r["id"]: r for r in json.load(open(f"{ROOT}/data/eval_queries_v2.json"))["queries"]}
S = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
R = pd.read_parquet(f"{ROOT}/data/analysis/df_runs.parquet")

NON_DISCRIMINATIVE = {"yes", "no", "true", "false", "none", "neither", "both", "n/a",
                      "increase", "increases", "decrease", "decreases", "higher", "lower",
                      "positive", "negative", "unchanged"}

def gold_tokens(ra):
    toks = [t.strip() for t in re.split(r"[,;]| and ", str(ra)) if t.strip()]
    return toks if toks else [str(ra).strip()]

def is_discriminative(ra):
    return not any(t.strip().lower() in NON_DISCRIMINATIVE for t in gold_tokens(ra))

_cand = {qid: r for qid, r in man.items()
         if r["source"] in ("litqa2", "deepsearch_qa") and r.get("reference_answer")}
SLICE = {qid: r for qid, r in _cand.items() if is_discriminative(r["reference_answer"])}

def read_report(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().lower()
    except Exception:
        return None

# ---------- mechanical gold per (pattern, query) base report (identical to judge_vs_gold) ----------
base = R[(R.pattern_family == "base") & (R.query_id.isin(SLICE)) & (R.report_exists)].copy()
gold_rows = []
for row in base.itertuples():
    rec = SLICE[row.query_id]; src = rec["source"]; toks = gold_tokens(rec["reference_answer"])
    txt = read_report(row.report_path)
    if txt is None:
        continue
    present = [t.lower() in txt for t in toks]
    cov = sum(present) / len(present)
    strict = int(cov == 1.0)
    lenient = int(cov > 0.0)
    if strict != lenient and len(toks) > 1:
        continue
    gold_rows.append((row.pattern, row.query_id, src, strict))

G = pd.DataFrame(gold_rows, columns=["pattern", "query_id", "source", "correct"])

def dim_table(dim):
    d = S[(S.pattern_family == "base") & (S.judge.isin(PANEL)) & (S.dimension == dim) &
          (S.query_id.isin(SLICE))][["pattern", "query_id", "judge", "score"]]
    return d.merge(G, on=["pattern", "query_id"], how="inner")

# ---------- agreement metrics ----------
import krippendorff  # project-standard alpha tool (same package build_irr_robust.py uses)

rng = np.random.default_rng(SEED)

def band_of(v):
    if v is None or not np.isfinite(v):
        return None
    if v >= 0.85:
        return "near_gold_high_stakes"
    if v >= 0.80:
        return "near_gold"
    if v >= 0.60:
        return "substantial"
    if v >= 0.40:
        return "moderate"
    if v >= 0.20:
        return "fair"
    return "poor"

def kripp(judge_labels, gold_labels, level):
    rd = np.vstack([np.asarray(judge_labels, float), np.asarray(gold_labels, float)])
    try:
        a = krippendorff.alpha(reliability_data=rd, level_of_measurement=level)
        return round(float(a), 4) if np.isfinite(a) else None
    except Exception:
        return None

def boot_kappa_ci(j_lab, g_lab, weights, n_boot=5000):
    """Seeded cluster-free bootstrap CI on kappa over the gold-slice reports (sorted index)."""
    j = np.asarray(j_lab); g = np.asarray(g_lab); n = len(j)
    if n < 3 or len(set(g)) < 2:
        return None
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(n, size=n, replace=True)
        js, gs = j[pick], g[pick]
        if len(set(gs)) < 2 or len(set(js)) < 2:
            continue
        try:
            vals.append(cohen_kappa_score(js, gs, weights=weights))
        except Exception:
            continue
    if len(vals) < 100:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]

def calibrate(jt):
    """jt: rows for one judge on the gold slice, columns score + correct. Sorted for determinism."""
    jt = jt.sort_values(["query_id", "pattern"]).reset_index(drop=True)
    n = len(jt)
    gold = jt.correct.astype(int).values
    score = jt.score.astype(float).values
    out = {"n": int(n), "n_gold_correct": int(gold.sum()), "n_gold_incorrect": int((gold == 0).sum())}

    # --- binary: judge calls 'correct' iff score >= its own gold-slice median ---
    median = float(np.median(score))
    j_bin = (score >= median).astype(int)
    # On a 2-level scale quadratic-weighted kappa == unweighted Cohen kappa.
    if len(set(gold)) > 1 and len(set(j_bin)) > 1:
        k_bin = round(float(cohen_kappa_score(j_bin, gold)), 4)
    else:
        k_bin = None
    out["binary"] = {
        "operating_point": "judge factual score >= gold-slice median",
        "median_threshold": round(median, 4),
        "cohen_kappa": k_bin,
        "cohen_kappa_band": band_of(k_bin),
        "cohen_kappa_ci95": boot_kappa_ci(j_bin, gold, weights=None),
        "krippendorff_alpha_nominal": kripp(j_bin, gold, "nominal"),
    }

    # --- ordinal3: judge tertile bins {0,1,2} vs gold endpoints {0->incorrect, 2->correct} ---
    q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
    j_ord = np.where(score >= q2, 2, np.where(score >= q1, 1, 0)).astype(int)
    g_ord = np.where(gold == 1, 2, 0).astype(int)
    if len(set(g_ord)) > 1 and len(set(j_ord)) > 1:
        k_ord = round(float(cohen_kappa_score(j_ord, g_ord, weights="quadratic",
                                              labels=[0, 1, 2])), 4)
    else:
        k_ord = None
    out["ordinal3"] = {
        "construction": "judge tertile bins {0,1,2} (gold-slice tertiles) vs gold endpoints {0,2}",
        "tertile_cuts": [round(float(q1), 4), round(float(q2), 4)],
        "weighted_kappa_quadratic": k_ord,
        "weighted_kappa_band": band_of(k_ord),
        "weighted_kappa_ci95": boot_kappa_ci(j_ord, g_ord, weights="quadratic"),
        "krippendorff_alpha_ordinal": kripp(j_ord, g_ord, "ordinal"),
    }
    return out

# ---------- compute per judge per dimension; append into per_judge[*][dim]['calibration'] ----------
cn = json.load(open(CANON))
jvg = cn["judge_vs_gold"]
per_judge = jvg.setdefault("per_judge", {})

summary = {}
for dim in DIMS:
    T = dim_table(dim)
    for j in PANEL:
        jt = T[T.judge == j]
        if len(jt) == 0:
            continue
        cal = calibrate(jt)
        rec = per_judge.setdefault(j, {"family": FAMILY[j]})
        dim_rec = rec.setdefault(dim, {})
        # APPEND calibration next to the existing auc/boot_diff fields; never overwrite them.
        dim_rec["calibration"] = cal
        if dim == "factual_accuracy":
            summary[j] = {
                "family": FAMILY[j],
                "auc": dim_rec.get("auc"),                       # existing discrimination number
                "binary_cohen_kappa": cal["binary"]["cohen_kappa"],
                "binary_kappa_band": cal["binary"]["cohen_kappa_band"],
                "ordinal3_weighted_kappa": cal["ordinal3"]["weighted_kappa_quadratic"],
                "ordinal3_kappa_band": cal["ordinal3"]["weighted_kappa_band"],
                "krippendorff_alpha_ordinal": cal["ordinal3"]["krippendorff_alpha_ordinal"],
            }

# ---------- top-level calibration block ----------
gpt = summary.get("gpt52", {})
auc_high_kappa_low = bool(
    gpt.get("auc") is not None and gpt["auc"] >= 0.60
    and gpt.get("binary_cohen_kappa") is not None and gpt["binary_cohen_kappa"] < 0.60)

jvg["calibration"] = {
    "_note": "P2_judge_kappa: chance-corrected AGREEMENT (weighted Cohen kappa + Krippendorff "
             "alpha) of each judge's thresholded factual_accuracy vs the human gold slice, reported "
             "ALONGSIDE the existing rank-only AUCs in per_judge. AUC measures discrimination "
             "(ranking correct>incorrect); kappa/alpha measure whether the judge assigns the SAME "
             f"label as the gold net of chance, at a fixed judge-internal operating point. Per {CITATION}, "
             "a high AUC can co-exist with poor kappa; this block lets a reader see both. Same "
             "LitQA2+DeepSearchQA slice as judge_vs_gold (n's match n_reports_with_gold). Anchors, "
             "does not certify (general mechanical gold, small slice). Two pre-stated discretisations "
             "(binary at median; ordinal3 at tertiles) reported, no post-hoc selection.",
    "citation": CITATION,
    "metric": "weighted_cohen_kappa_quadratic + krippendorff_alpha (ordinal & nominal)",
    "prestated_bands": BANDS,
    "operating_point_note": "thresholds are JUDGE-internal score quantiles (median / tertiles); they "
                            "do NOT use the gold labels, so agreement is not tuned upward.",
    "per_judge_factual": summary,
    "high_auc_low_kappa_demonstrated": auc_high_kappa_low,
    "headline": (
        f"Even the answer-sensitive judge (GPT-5.2: factual AUC {gpt.get('auc')}) shows "
        f"chance-corrected agreement of only kappa(binary)={gpt.get('binary_cohen_kappa')} "
        f"[{gpt.get('binary_kappa_band')}], weighted-kappa(ordinal)={gpt.get('ordinal3_weighted_kappa')} "
        f"[{gpt.get('ordinal3_kappa_band')}] vs the human gold slice — below the 0.60 substantial "
        f"band and far below the ~0.85 high-stakes floor. This is the {CITATION} high-AUC/low-kappa "
        "illusion on our own panel: ranking is not agreement."),
}

# ---------- atomic append-only write (mirror build_judge_vs_gold / build_n_eff) ----------
_txt = json.dumps(cn, indent=1)
_tmp = f"{CANON}.tmp"
open(_tmp, "w").write(_txt)
os.replace(_tmp, CANON)

print(f"P2_judge_kappa: slice n_reports_gold={int(len(G))} correct_rate={round(float(G.correct.mean()),4)}")
print(f"  pre-stated bands: substantial 0.60-0.80 | high-stakes floor ~0.85 | cite {CITATION}")
for j in PANEL:
    s = summary.get(j)
    if s:
        print(f"  {j:14s} factual: AUC={s['auc']}  kappa(binary)={s['binary_cohen_kappa']} "
              f"[{s['binary_kappa_band']}]  w-kappa(ord)={s['ordinal3_weighted_kappa']} "
              f"[{s['ordinal3_kappa_band']}]  alpha(ord)={s['krippendorff_alpha_ordinal']}")
print(f"  high-AUC/low-kappa illusion (gpt52, {CITATION}) demonstrated: {auc_high_kappa_low}")
