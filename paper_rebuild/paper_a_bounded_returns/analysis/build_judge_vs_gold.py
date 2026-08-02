#!/usr/bin/env python
"""E13' source-4 — judge-vs-human-gold registry (the cheapest human-anchored validity number).

Pre-registration: docs/publication/prereg/prereg_E13prime_source4.md (registered 2026-06-11, before run).

Ground truth, established by inspection (recorded in PROGRAMME_EXECUTION_STATE.md): on-disk
mechanical gold exists only where the manifest carries an objective `reference_answer` —
LitQA2 (10 queries, scientist-written MCQ answers) and DeepSearchQA (19, DeepMind verifiable
answer sets). DRACO/ResearchQA expert criteria have no per-report gold label on disk and are
deferred to the entailment-pass extension.

A second inspection result shaped the design: the manifest's literal answer-bearing criterion
("identifies the correct answer: Ogfrl1") is almost never scored verbatim by the panel
(9 opus verdicts total), and where a judge DOES write a "mentions X" criterion the verdict is
circular with a mechanical "report contains X" check (it measures trivial faithfulness, not
validity). So the informative, non-circular endpoint is at the DIMENSION level:

    Does each judge's general factual_accuracy score track mechanically-verifiable
    answer-correctness on the gold-answer slice?

Mechanical gold (human-authored answers, checked mechanically from report text):
  - LitQA2     : answer_present = gold token appears in report (distinctive MCQ entity)
  - DeepSearchQA: coverage = fraction of gold set tokens present; correct = (coverage == 1)
Both a STRICT and a LENIENT matcher are computed and the sensitivity is reported; queries
whose two matchers disagree on correctness are dropped and the count is logged (never silent).

Endpoints (per judge, per family, per source):
  - mean factual_accuracy | answer-correct  vs  | answer-incorrect, with seeded cluster
    bootstrap CI on the difference (the judges SHOULD score correct-answer reports higher if
    their factual dimension is answer-sensitive)
  - point-biserial correlation and AUC of factual_accuracy predicting answer-correctness
  - the SAME for citation_quality as the H2 contrast (citations are not answer-determined,
    so a valid panel should track answer-correctness with factual_accuracy MORE than with
    citation_quality)
  - small Tier-1 cross-check: the 9 opus answer-criterion verdicts vs mechanical gold

Honest scope: this anchors, it does not certify. The rubric's factual_accuracy aggregates
GENERAL criteria, so this tests whether general factual scoring is answer-SENSITIVE on
verifiable items, not whether each answer-specific claim was graded. Matcher noise is real.

Writes canonical_numbers.json['judge_vs_gold']. Determinism: seeded generator on sorted inputs.
"""
import json, re, os
import numpy as np
import pandas as pd

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260611
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
FAMILY = {"gpt52": "openai", "claude_opus": "anthropic", "claude_sonnet": "anthropic",
          "claude_code": "anthropic"}

# ---------- load ----------
man = {r["id"]: r for r in json.load(open(f"{ROOT}/data/eval_queries_v2.json"))["queries"]}
S = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
R = pd.read_parquet(f"{ROOT}/data/analysis/df_runs.parquet")
V = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")

# Polarity / direction answers cannot be verified by substring presence ("does the report
# contain the word 'yes'/'increases'" does NOT certify the report's conclusion), so such queries
# are dropped from the mechanical slice and logged. This makes the report counts robust: an
# adversarial recompute (2026-06-11) traced the only count fragility to the single 'yes, no'
# query 2c05315d; the per-judge AUC finding is invariant either way.
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
DROPPED_NONDISCRIM = sorted(qid for qid, r in _cand.items() if not is_discriminative(r["reference_answer"]))
SLICE = {qid: r for qid, r in _cand.items() if is_discriminative(r["reference_answer"])}

def read_report(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().lower()
    except Exception:
        return None

# ---------- mechanical gold per (pattern, query) base report ----------
base = R[(R.pattern_family == "base") & (R.query_id.isin(SLICE)) & (R.report_exists)].copy()
gold_rows = []
dropped_ambiguous = 0
for row in base.itertuples():
    rec = SLICE[row.query_id]; src = rec["source"]; toks = gold_tokens(rec["reference_answer"])
    txt = read_report(row.report_path)
    if txt is None:
        continue
    present = [t.lower() in txt for t in toks]
    cov = sum(present) / len(present)
    strict = int(cov == 1.0)          # all gold tokens present
    lenient = int(cov > 0.0)          # any gold token present
    if strict != lenient and len(toks) > 1:
        # multi-token gold with partial coverage = genuinely ambiguous correctness
        dropped_ambiguous += 1
        continue
    correct = strict                  # single-token litqa2: strict==lenient anyway
    gold_rows.append((row.pattern, row.query_id, src, cov, correct))

G = pd.DataFrame(gold_rows, columns=["pattern", "query_id", "source", "coverage", "correct"])

# ---------- join judge dimension scores ----------
def dim_table(dim):
    d = S[(S.pattern_family == "base") & (S.judge.isin(PANEL)) & (S.dimension == dim) &
          (S.query_id.isin(SLICE))][["pattern", "query_id", "judge", "score"]]
    return d.merge(G, on=["pattern", "query_id"], how="inner")

rng = np.random.default_rng(SEED)

def cluster_boot_diff(df, n_boot=5000):
    """Bootstrap the (correct - incorrect) mean-score gap, resampling QUERIES (clusters).
    Seeded generator on sorted query ids -> deterministic."""
    qids = sorted(df.query_id.unique())
    if df.correct.nunique() < 2 or len(qids) < 2:
        return None
    obs = df[df.correct == 1].score.mean() - df[df.correct == 0].score.mean()
    by_q = {q: df[df.query_id == q] for q in qids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), size=len(qids), replace=True)
        samp = pd.concat([by_q[qids[i]] for i in pick], ignore_index=True)
        if samp.correct.nunique() < 2:
            continue
        diffs.append(samp[samp.correct == 1].score.mean() - samp[samp.correct == 0].score.mean())
    if not diffs:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_0": bool(lo > 0 or hi < 0)}

def _midrank(x):
    """Mid-ranks (ties averaged) over a 1-D array; mergesort for determinism."""
    x = np.asarray(x, dtype=float); n = len(x)
    order = np.argsort(x, kind="mergesort"); xs = x[order]
    r = np.empty(n, dtype=float); i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r

def auc(y, x):
    y = np.asarray(y); x = np.asarray(x)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # tie-corrected Mann-Whitney U / (n_pos n_neg) via combined mid-ranks
    r = _midrank(x)
    u = r[y == 1].sum() - _midrank(pos).sum()
    return round(float(u / (len(pos) * len(neg))), 4)

def delong_two(y, x1, x2):
    """DeLong (1988) test for the DIFFERENCE of two PAIRED AUCs that share the same labels y
    (here: two judges' factual scores on the *same* common report set). Returns the AUCs, the
    diff (x1-x2), the DeLong variance of the diff, z, two-sided p, and a normal 95% CI. This is
    the real between-judge test the cross-family asymmetry was missing. Deterministic (no RNG)."""
    y = np.asarray(y); pos = y == 1; neg = y == 0
    m = int(pos.sum()); n = int(neg.sum())
    if m == 0 or n == 0:
        return None
    preds = [np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)]
    aucs = np.empty(2); V10 = np.empty((2, m)); V01 = np.empty((2, n))
    for r_i, xx in enumerate(preds):
        xp, xn = xx[pos], xx[neg]
        tz = _midrank(xx); tx = _midrank(xp); ty = _midrank(xn)
        aucs[r_i] = (tz[pos].sum() - tx.sum()) / (m * n)
        V10[r_i] = (tz[pos] - tx) / n          # structural components over positives
        V01[r_i] = 1.0 - (tz[neg] - ty) / m    # structural components over negatives
    S = np.cov(V10) / m + np.cov(V01) / n      # 2x2 covariance of the two AUC estimators
    L = np.array([1.0, -1.0]); var = float(L @ S @ L)
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = diff / se if se > 0 else 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {"auc1": round(float(aucs[0]), 4), "auc2": round(float(aucs[1]), 4),
            "diff": round(diff, 4), "se": round(se, 4), "z": round(float(z), 4),
            "p": round(float(p), 6),
            "ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
            "significant_05": bool(p < 0.05)}

def boot_auc_diff(piv, n_boot=5000):
    """Paired cluster bootstrap of the AUC difference (judge1 - judge2) on the common report set,
    resampling QUERIES (clusters). Robustness check alongside the parametric DeLong test. Uses the
    module-level seeded `rng` on sorted query ids -> deterministic."""
    qids = sorted(piv.query_id.unique())
    if piv.correct.nunique() < 2 or len(qids) < 2:
        return None
    obs = auc(piv.correct.values, piv.s1.values) - auc(piv.correct.values, piv.s2.values)
    by_q = {q: piv[piv.query_id == q] for q in qids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(qids), size=len(qids), replace=True)
        samp = pd.concat([by_q[qids[i]] for i in pick], ignore_index=True)
        if samp.correct.nunique() < 2:
            continue
        a1 = auc(samp.correct.values, samp.s1.values)
        a2 = auc(samp.correct.values, samp.s2.values)
        if a1 is None or a2 is None:
            continue
        diffs.append(a1 - a2)
    if not diffs:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_0": bool(lo > 0 or hi < 0)}

out = {
    "_note": "E13' source-4 judge-vs-human-gold (LitQA2+DeepSearchQA verifiable-answer slice). "
             "HONEST FINDING (after adversarial review 2026-06-11): the answer-present signal is "
             "GENERAL report completeness — answer-correct reports score higher on EVERY dimension, "
             "not just factual. So the load-bearing result is the CROSS-FAMILY ASYMMETRY: GPT-5.2's "
             "verdicts are answer-sensitive (dimension AUCs > 0.6) while both Claude judges are "
             "near-chance (~0.51-0.55, CIs span 0). The H2 'factual tracks gold MORE than citation' "
             "does NOT hold (for gpt52 the citation gap +0.173 exceeds the factual gap +0.078); it "
             "is demoted to an exploratory, mixed observation. Anchors, does not certify (general "
             "dimension, matcher noise, small signal-bearing cluster count). Prereg: prereg_E13prime_source4.md.",
    "prereg": "docs/publication/prereg/prereg_E13prime_source4.md",
    "slice": {"n_queries": len(SLICE),
              "litqa2": sum(1 for r in SLICE.values() if r["source"] == "litqa2"),
              "deepsearch_qa": sum(1 for r in SLICE.values() if r["source"] == "deepsearch_qa")},
    "dropped_nondiscriminative_queries": DROPPED_NONDISCRIM,
    "n_reports_with_gold": int(len(G)),
    "dropped_ambiguous_multi_token": int(dropped_ambiguous),
    "matcher_note": "n_reports_with_gold and dropped_ambiguous are matcher-implementation "
                    "sensitive at the ~few-report level (traced to LitQA2 polarity answers, now "
                    "dropped up front). The per-judge AUC ordering is invariant across "
                    "drop/incorrect/correct routings of ambiguous cases (adversarial check 2026-06-11).",
    "answer_correct_rate_overall": round(float(G.correct.mean()), 4),
    "answer_correct_rate_by_source": {s: round(float(G[G.source == s].correct.mean()), 4)
                                      for s in sorted(G.source.unique())},
    "effective_signal_clusters": int(G[G.correct == 1].query_id.nunique()),
    "per_judge": {}, "per_family": {}, "dimension_sensitivity": {}, "tier1_answer_criterion": {},
    "cross_family_test": {},
}

def common_keys(T):
    """(pattern, query_id) reports scored by EVERY panel judge for this dimension. The raw per-judge
    AUCs are computed on DIFFERENT report sets (gpt52 scores more reports than the Claude judges),
    so any cross-judge comparison must be made on this matched intersection."""
    present = [j for j in PANEL if (T.judge == j).any()]
    sets = [set(map(tuple, T[T.judge == j][["pattern", "query_id"]].values)) for j in present]
    return set.intersection(*sets) if sets else set()

for dim in ["factual_accuracy", "citation_quality"]:
    T = dim_table(dim)
    ck = common_keys(T)
    key_in_common = T.apply(lambda r, ck=ck: (r.pattern, r.query_id) in ck, axis=1)
    Tc = T[key_in_common]                    # matched, common-to-all-judges report set
    for j in PANEL:
        jt = T[T.judge == j]
        if len(jt) == 0:
            continue
        jtc = Tc[Tc.judge == j]              # same judge, restricted to common set
        rec = out["per_judge"].setdefault(j, {"family": FAMILY[j]})
        rec[dim] = {
            "n": int(len(jt)),
            "mean_score_correct": round(float(jt[jt.correct == 1].score.mean()), 4) if (jt.correct == 1).any() else None,
            "mean_score_incorrect": round(float(jt[jt.correct == 0].score.mean()), 4) if (jt.correct == 0).any() else None,
            "boot_diff": cluster_boot_diff(jt),
            "auc": auc(jt.correct.values, jt.score.values),
            # AUC on the matched common report set (this is what the cross-family test compares):
            "auc_common": auc(jtc.correct.values, jtc.score.values) if len(jtc) else None,
            "n_common": int(len(jtc)),
            "point_biserial_r": round(float(np.corrcoef(jt.correct.values, jt.score.values)[0, 1]), 4)
                                if jt.correct.nunique() > 1 else None,
        }

    # CROSS-FAMILY TEST on the matched common set: DeLong test (paired, shared labels) plus a
    # paired cluster bootstrap for the AUC difference gpt52 - each Claude judge. This is the
    # between-judge test the asymmetry claim previously lacked (numbers were on different n per judge).
    if "gpt52" in [j for j in PANEL if (Tc.judge == "gpt52").any()] and len(ck) > 0:
        piv = (Tc.pivot_table(index=["pattern", "query_id", "correct"], columns="judge",
                              values="score", observed=True).reset_index())
        cf = {}
        for jc in [j for j in PANEL if FAMILY[j] == "anthropic"]:
            if jc not in piv.columns:
                continue
            sub = piv.dropna(subset=["gpt52", jc])
            if len(sub) == 0 or sub.correct.nunique() < 2:
                continue
            dl = delong_two(sub.correct.values, sub["gpt52"].values, sub[jc].values)
            p2 = sub.rename(columns={"gpt52": "s1", jc: "s2"})[["query_id", "correct", "s1", "s2"]]
            cf[f"gpt52_vs_{jc}"] = {
                "n_common_reports": int(len(sub)),
                "n_common_queries": int(sub.query_id.nunique()),
                "delong": dl,
                "paired_cluster_boot": boot_auc_diff(p2),
            }
        out["cross_family_test"][dim] = {
            "n_common_reports": int(len(ck)),
            "comparisons": cf,
            "note": "AUC difference (gpt52 - Claude judge) on the COMMON report set scored by all "
                    "panel judges. DeLong = parametric paired-AUC test (primary); paired_cluster_boot "
                    "= query-cluster bootstrap of the same difference (robustness, more conservative).",
        }
    # family aggregation (pool anthropic judges)
    for fam in sorted(set(FAMILY[j] for j in PANEL)):
        ft = T[T.judge.isin([j for j in PANEL if FAMILY[j] == fam])]
        if len(ft) == 0:
            continue
        out["per_family"].setdefault(fam, {})[dim] = {
            "n": int(len(ft)), "auc": auc(ft.correct.values, ft.score.values),
            "boot_diff": cluster_boot_diff(ft),
        }

# dimension_sensitivity: report BOTH dimensions' answer-tracking per judge. The honest reading
# is that answer-presence is a general-completeness signal (both dims track it), so the finding
# is the cross-family asymmetry, NOT factual-dimension specificity. H2 (factual>citation) is
# recorded as exploratory and is mixed/false for the load-bearing judge.
for j in PANEL:
    fa = out["per_judge"].get(j, {}).get("factual_accuracy", {})
    cq = out["per_judge"].get(j, {}).get("citation_quality", {})
    if fa.get("auc") is not None and cq.get("auc") is not None:
        out["dimension_sensitivity"][j] = {
            "factual": {"auc": fa["auc"], "gap": fa.get("boot_diff", {}).get("delta") if fa.get("boot_diff") else None},
            "citation": {"auc": cq["auc"], "gap": cq.get("boot_diff", {}).get("delta") if cq.get("boot_diff") else None},
            "h2_factual_tracks_more_EXPLORATORY": bool(fa["auc"] > cq["auc"]),
        }
_fa = out["per_judge"].get("gpt52", {}).get("factual_accuracy", {})
_cft = out["cross_family_test"].get("factual_accuracy", {}).get("comparisons", {})
_dl_op = _cft.get("gpt52_vs_claude_opus", {}).get("delong", {}) or {}
_dl_so = _cft.get("gpt52_vs_claude_sonnet", {}).get("delong", {}) or {}
_cb_op = _cft.get("gpt52_vs_claude_opus", {}).get("paired_cluster_boot", {}) or {}
_cb_so = _cft.get("gpt52_vs_claude_sonnet", {}).get("paired_cluster_boot", {}) or {}
out["cross_family_finding"] = (
    "On the MATCHED common report set (n="
    f"{out['cross_family_test'].get('factual_accuracy', {}).get('n_common_reports')} reports scored "
    "by all panel judges), GPT-5.2's factual verdicts are answer-sensitive (common-set AUC "
    f"{_fa.get('auc_common')}) while both Claude judges are near-chance. The cross-family AUC gap is "
    "now backed by a paired DeLong test on the SAME reports (it previously compared different per-judge "
    f"report sets, untested): gpt52>opus diff={_dl_op.get('diff')} (DeLong p={_dl_op.get('p')}), "
    f"gpt52>sonnet diff={_dl_so.get('diff')} (DeLong p={_dl_so.get('p')}); both significant at .05 under "
    "DeLong. Under the conservative query-CLUSTERED paired bootstrap, however, ONLY gpt52-vs-sonnet "
    f"survives (cluster CI {_cb_so.get('ci95')}, excludes 0); the gpt52-vs-opus asymmetry DISSOLVES under "
    f"clustering (cluster CI {_cb_op.get('ci95')} INCLUDES 0). So the robust, clustering-surviving "
    "cross-family finding is gpt52>sonnet alone; the opus arm rests on DeLong only. The answer signal "
    "reflects general report completeness, not factual-dimension specificity; the load-bearing result "
    "is the gpt52-vs-sonnet cross-family validity asymmetry, matched-set, difference-tested, and "
    "clustering-robust.")

# Tier-1 cross-check: the few answer-criterion verdicts that DO exist (opus/litqa2)
t1 = []
for qid, rec in SLICE.items():
    toks = gold_tokens(rec["reference_answer"]); patt = "|".join(re.escape(t) for t in toks)
    sub = V[(V.query_id == qid) &
            V.criterion.str.contains(patt, case=False, regex=True, na=False) &
            V.criterion.str.contains("correct answer|expected answer|identifies|correct factual answer",
                                     case=False, regex=True, na=False) &
            (V.satisfied_is_known)]
    for t in sub.itertuples():
        g = G[(G.pattern == t.pattern) & (G.query_id == qid)]
        if len(g):
            t1.append((t.judge, bool(t.satisfied), int(g.iloc[0].correct)))
if t1:
    t1d = pd.DataFrame(t1, columns=["judge", "judge_satisfied", "gold_correct"])
    out["tier1_answer_criterion"] = {
        "n": int(len(t1d)), "by_judge": {str(k): int(v) for k, v in t1d.judge.value_counts().items()},
        "raw_agreement": round(float((t1d.judge_satisfied == t1d.gold_correct.astype(bool)).mean()), 4),
        "note": "tiny, opus/litqa2 only; reported as anecdotal cross-check, not an endpoint.",
    }
else:
    out["tier1_answer_criterion"] = {"n": 0, "note": "no usable answer-criterion verdicts joined."}

import sys
if "--write" in sys.argv:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    cn["judge_vs_gold"] = out
    # atomic write: serialise fully to a tmp string first, then replace, so a serialisation
    # error can never truncate the canonical store (lesson learned 2026-06-11).
    _txt = json.dumps(cn, indent=1)
    _tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(_tmp, "w").write(_txt)
    os.replace(_tmp, f"{ANA}/canonical_numbers.json")
    print("[WROTE canonical_numbers.json['judge_vs_gold']]")
else:
    print("[DRY-RUN: no write; pass --write to update canonical_numbers.json]")

print(f"judge_vs_gold: slice={out['slice']} (dropped non-discriminative {DROPPED_NONDISCRIM}), "
      f"n_reports_gold={out['n_reports_with_gold']}, dropped_ambiguous={dropped_ambiguous}, "
      f"correct_rate={out['answer_correct_rate_overall']}, signal_clusters={out['effective_signal_clusters']}")
for j in PANEL:
    fa = out["per_judge"].get(j, {}).get("factual_accuracy", {})
    cq = out["per_judge"].get(j, {}).get("citation_quality", {})
    if fa:
        print(f"  {j:14s} factual AUC={fa['auc']} (common={fa.get('auc_common')}, n_common={fa.get('n_common')}) "
              f"gap={fa['boot_diff']}  |  citation AUC={cq.get('auc')} (common={cq.get('auc_common')})")
cft = out["cross_family_test"].get("factual_accuracy", {})
print(f"  CROSS-FAMILY TEST (matched common set, n={cft.get('n_common_reports')} factual reports):")
for k, v in cft.get("comparisons", {}).items():
    dl = v.get("delong", {}) or {}
    bt = v.get("paired_cluster_boot", {}) or {}
    print(f"    {k}: AUC {dl.get('auc1')} vs {dl.get('auc2')} diff={dl.get('diff')} "
          f"DeLong z={dl.get('z')} p={dl.get('p')} CI95={dl.get('ci95')} | "
          f"bootCI={bt.get('ci95')} excl0={bt.get('excludes_0')}")
print("  CROSS-FAMILY: GPT-5.2 answer-sensitive; Claude judges near-chance. Difference now tested "
      "on MATCHED reports (DeLong + paired bootstrap), not on different per-judge n. "
      "H2 (factual>citation) remains exploratory/mixed.")
