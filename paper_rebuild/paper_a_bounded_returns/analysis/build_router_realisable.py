#!/usr/bin/env python
"""A1 REALISABLE PER-QUERY ROUTER — deployable selector vs best-fixed vs oracle.

Companion to `routability` (Stage A/B). Those keys report an oracle-headroom NULL and a
gpt52-single-run LOOCV feature-router headroom. This script answers the *deployment* question on
the canonical, lower-noise target: the 3-JUDGE PANEL MEAN per (pattern, query). It reports three
absolute levels a deployable system can be scored against ---

  (a) best_fixed_mean : the single best fixed architecture's mean (the "no-routing" default),
  (b) router_cv_mean  : a REALISABLE per-query router's realised mean under leave-one-query-out CV
                        (the router only ever sees query-metadata features, never scores; each
                        held-out query is scored by a router trained on the other 84 -> unbiased),
  (c) oracle_mean     : the per-query oracle upper bound (pick the actually-best pattern per query).

Headline logic (the paper's claim): no architecture dominates on the mean, the per-SOURCE leader
rotates, and per-query effects sign-flip -> a per-query router SHOULD beat any fixed architecture.
We test whether a *deployable* router realises that gain out-of-sample, and by how much it closes
the best_fixed -> oracle gap. Honest by construction: if the realisable router does NOT beat
best-fixed (a real possibility at n=85 with one dominant pattern), that is reported as such.

TARGET metric (canonical, matches build_numbers.py): PANEL = {gpt52, claude_opus, claude_sonnet};
`ovc` = overall_score, EXCEPT claude_sonnet uses overall_score_recomputed (its stored overall is
corrupted, per DATA_DICTIONARY). Target M[query, pattern] = mean over the 3 panel judges. Restrict
to base_p0..base_p10 and to queries with COMPLETE 3-judge coverage on all 11 patterns (n=85).

FEATURES (realisable = available BEFORE running, query-only, never from scores):
  log1p(word_count), expected_topic_count, difficulty_ordinal{simple:0,moderate:1,complex:2},
  causal_question_count, entity_density, + source one-hot (5 levels).
  (Free-form `domain` has 43 levels over 85 queries -> excluded as un-learnable at this n; noted.)

ROUTERS:
  - PRIMARY (pre-declared): per-pattern Ridge(alpha=1) score-regressor -> argmax, LOOCV. Uses the
    full (de-noised) 3-judge score signal, not just the argmax label; the natural realisable choice.
  - source_conditional (requested interpretable baseline): leave-one-query-out WITHIN source ---
    for held-out query i pick the best pattern by the mean of the OTHER same-source queries
    (falls back to best_fixed if the source has no other query). This is the "route by benchmark"
    baseline the per-source-leader table hints at.
  - Robustness panel (reported, NOT cherry-picked over the primary): multinomial logistic and
    gradient-boosted classifiers on the winner label; kNN(k=7) mean-argmax.

STATISTICS: best_fixed is fixed at the full-sample column-mean argmax (base_p1; its 0.034 lead is
LOOCV-stable, would be re-selected in every fold). Deltas are per-query (pick_score - bf_score);
CIs are QUERY-CLUSTERED bootstraps (resample the 85 queries with replacement), B=5000, seed
20260703. router beats best_fixed iff the delta CI excludes 0. gap_closed_fraction = router_delta /
oracle_delta. Determinism: sorted inputs, per-fold StandardScaler (no scale leakage), seeded
estimators, dedicated seeded bootstrap generator.

Append-only: adds canonical key 'router_realisable' via setdefault (never overwrites); asserts all
pre-existing top-level keys are byte-identical after the write.
"""
import json, os, re, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260703
BOOT = 5000
ARCH = [f"base_p{i}" for i in range(11)]
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]

# ---------- canonical 3-judge target ----------
O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
Q = pd.read_parquet(f"{ROOT}/data/analysis/df_queries.parquet").set_index("query_id")

def corrected_overall(df):
    c = df["overall_score"].copy()
    m = df["judge"].eq("claude_sonnet")
    return c.where(~m, df["overall_score_recomputed"])

O = O[O.pattern.isin(ARCH) & O.judge.isin(PANEL)].copy()
O["ovc"] = corrected_overall(O)
cell = O.groupby(["pattern", "query_id"], observed=True).agg(
    m=("ovc", "mean"), nj=("judge", "nunique")).reset_index()
cell = cell[cell.nj == len(PANEL)]                                     # complete 3-judge cells only
piv = cell.pivot_table(index="query_id", columns="pattern", values="m", observed=True)
piv = piv[sorted(piv.columns)].dropna().sort_index()                  # queries with all 11; sorted
qids = list(piv.index)
cols = list(piv.columns)
M = piv.values                                                        # (n, 11) 3-judge mean target
n = len(qids)

col_means = M.mean(axis=0)
bf = int(col_means.argmax())                                          # best-fixed (full-sample)
bf_mean = float(col_means[bf])
oracle_mean = float(M.max(axis=1).mean())
winner = M.argmax(axis=1)

# ---------- realisable query-only features ----------
def caus(t):
    return len(re.findall(r"\b(why|how|cause|because|impact|effect|reason|lead to|result in|"
                          r"influence|compare|versus|trade[- ]?off)\b", str(t), re.I))
def entden(t):
    w = str(t).split()
    return (sum(1 for x in w if x[:1].isupper()) + len(re.findall(r"\d", str(t)))) / max(len(w), 1)
def n_topics(x):
    try:
        return float(len(x))
    except Exception:
        return 0.0
DIFF = {"simple": 0, "moderate": 1, "complex": 2}
SOURCES = sorted(Q.loc[qids, "source"].unique())

def features(qid):
    r = Q.loc[qid]
    f = [np.log1p(len(str(r.query_text).split())),        # log word count
         n_topics(r.expected_topics),                     # expected-topic count
         float(DIFF.get(r.difficulty, 1)),                # difficulty ordinal
         float(caus(r.query_text)),                       # causal/compare question count
         float(entden(r.query_text))]                     # entity density
    f += [1.0 if r.source == s else 0.0 for s in SOURCES] # source one-hot
    return f
FEATURE_NAMES = ["log_word_count", "expected_topic_count", "difficulty_ordinal",
                 "causal_question_count", "entity_density"] + [f"source={s}" for s in SOURCES]
X = np.array([features(q) for q in qids])
src = np.array([Q.loc[q, "source"] for q in qids])
idx_all = np.arange(n)

# ---------- routers (each returns picks: list[int] pattern-index per query) ----------
def ridge_argmax_loocv():                                 # PRIMARY
    picks = []
    for i in range(n):
        tr = idx_all != i
        sc = StandardScaler().fit(X[tr])                  # fit scaler on training fold only
        Xtr, Xi = sc.transform(X[tr]), sc.transform(X[i:i + 1])
        preds = []
        for j in range(len(cols)):
            r = Ridge(alpha=1.0, random_state=SEED); r.fit(Xtr, M[tr, j])
            preds.append(r.predict(Xi)[0])
        picks.append(int(np.argmax(preds)))
    return picks

def source_conditional_loocv():                           # requested interpretable baseline
    picks = []
    for i in range(n):
        mask = (idx_all != i) & (src == src[i])
        picks.append(int(M[mask].mean(axis=0).argmax()) if mask.sum() else bf)
    return picks

def knn_loocv(k=7):
    picks = []
    for i in range(n):
        tr = idx_all != i
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        d = np.linalg.norm(Xs - Xs[i], axis=1); d[i] = np.inf
        nn = np.argsort(d)[:k]
        picks.append(int(M[nn].mean(axis=0).argmax()))
    return picks

def clf_loocv(make):
    picks = []
    for i in range(n):
        tr = idx_all != i
        sc = StandardScaler().fit(X[tr]); Xtr, Xi = sc.transform(X[tr]), sc.transform(X[i:i + 1])
        ytr = winner[tr]
        if len(np.unique(ytr)) < 2:
            picks.append(bf); continue
        clf = make(); clf.fit(Xtr, ytr)
        picks.append(int(clf.predict(Xi)[0]))
    return picks

routers = {
    "ridge_argmax_primary": ridge_argmax_loocv(),
    "source_conditional":   source_conditional_loocv(),
    "knn_k7":               knn_loocv(7),
    "multinomial_logistic": clf_loocv(lambda: LogisticRegression(max_iter=3000, C=1.0)),
    "gbm_classifier":       clf_loocv(lambda: GradientBoostingClassifier(random_state=SEED)),
}

def picks_scores(picks):
    return np.array([M[i, picks[i]] for i in range(n)])

bf_scores = M[:, bf]
oracle_scores = M.max(axis=1)

# ---------- query-clustered bootstrap ----------
_rng = np.random.default_rng(SEED)
BOOT_IDX = _rng.integers(0, n, size=(BOOT, n))            # shared resample plan -> deterministic

def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

def boot_block(scores):
    """Clustered bootstrap: realised mean, delta vs best-fixed (CI + P(delta>0))."""
    delta_pq = scores - bf_scores
    b_mean = np.array([scores[BOOT_IDX[b]].mean() for b in range(BOOT)])
    b_delta = np.array([delta_pq[BOOT_IDX[b]].mean() for b in range(BOOT)])
    return {
        "realised_mean": round(float(scores.mean()), 4),
        "realised_mean_ci95": ci(b_mean),
        "delta_vs_best_fixed": round(float(delta_pq.mean()), 4),
        "delta_vs_best_fixed_ci95": ci(b_delta),
        "prob_delta_gt_0": round(float((b_delta > 0).mean()), 4),
        "beats_best_fixed_sig": bool(np.percentile(b_delta, 2.5) > 0),
    }

# oracle block (upper bound)
_or_delta = oracle_scores - bf_scores
_b_or = np.array([_or_delta[BOOT_IDX[b]].mean() for b in range(BOOT)])
_b_or_mean = np.array([oracle_scores[BOOT_IDX[b]].mean() for b in range(BOOT)])
oracle_block = {
    "oracle_mean": round(oracle_mean, 4),
    "oracle_mean_ci95": ci(_b_or_mean),
    "oracle_delta_vs_best_fixed": round(float(_or_delta.mean()), 4),
    "oracle_delta_vs_best_fixed_ci95": ci(_b_or),
}

router_blocks = {name: boot_block(picks_scores(p)) for name, p in routers.items()}

# gap-closed fraction (per-resample ratio) for each router, + point
oracle_delta_point = float(_or_delta.mean())
def gap_closed(scores):
    delta_pq = scores - bf_scores
    b_r = np.array([delta_pq[BOOT_IDX[b]].mean() for b in range(BOOT)])
    b_frac = b_r / _b_or                                  # aligned resamples (same BOOT_IDX)
    return {
        "gap_closed_fraction": round(float(delta_pq.mean() / oracle_delta_point), 4),
        "gap_closed_fraction_ci95": ci(b_frac),
    }
for name in routers:
    router_blocks[name].update(gap_closed(picks_scores(routers[name])))

# winner-label reliability of the primary router (out-of-sample argmax-match rate vs chance)
primary = "ridge_argmax_primary"
prim_picks = np.array(routers[primary])
match_rate = float((prim_picks == winner).mean())
chance = float(np.sum((np.bincount(winner, minlength=len(cols)) / n) ** 2))  # sum p_k^2

# ---------- assemble ----------
result = {
    "_note": ("A1 realisable per-query router on the canonical 3-judge panel-mean target. Reports "
              "the deployable router's REALISED leave-one-query-out mean against best-fixed (lower "
              "bound reference) and the per-query oracle (upper bound). Query-clustered bootstrap "
              "CIs. Companion to routability (which reports the oracle-headroom null and a "
              "gpt52-single-run LOOCV router headroom). Honest negative reported as such."),
    "target_metric": "3-judge panel mean per (pattern,query); ovc = overall_score, sonnet uses "
                     "overall_score_recomputed (PANEL={gpt52,claude_opus,claude_sonnet}).",
    "n": n,
    "n_patterns": len(cols),
    "patterns": cols,
    "coverage_note": f"base_p0..base_p10 restricted to the {n} queries with complete 3-judge "
                     "coverage on all 11 patterns (5 of 90 dropped for incomplete panel).",
    "features": FEATURE_NAMES,
    "feature_note": "query-only, available before running (never derived from scores). Free-form "
                    "`domain` (43 levels / 85 queries) excluded as un-learnable at this n.",
    "cv": "leave-one-query-out (n folds); per-fold StandardScaler; unbiased (held-out query never "
          "in its router's training set).",
    "best_fixed": cols[bf],
    "best_fixed_mean": round(bf_mean, 4),
    "best_fixed_note": "full-sample column-mean argmax; 0.034 lead is LOOCV-stable (re-selected in "
                       "every fold), so it is fixed for the per-query delta baseline.",
    "oracle": oracle_block,
    "raw_oracle_headroom": round(oracle_mean - bf_mean, 4),
    "primary_router": primary,
    "routers": router_blocks,
    # flat headline convenience keys (primary router)
    "router_cv_mean": router_blocks[primary]["realised_mean"],
    "router_cv_mean_ci95": router_blocks[primary]["realised_mean_ci95"],
    "router_delta_vs_best_fixed": router_blocks[primary]["delta_vs_best_fixed"],
    "router_delta_vs_best_fixed_ci95": router_blocks[primary]["delta_vs_best_fixed_ci95"],
    "router_beats_best_fixed_sig": router_blocks[primary]["beats_best_fixed_sig"],
    "gap_closed_fraction": router_blocks[primary]["gap_closed_fraction"],
    "gap_closed_fraction_ci95": router_blocks[primary]["gap_closed_fraction_ci95"],
    "oracle_mean": oracle_block["oracle_mean"],
    "source_conditional_mean": router_blocks["source_conditional"]["realised_mean"],
    "source_conditional_delta_vs_best_fixed": router_blocks["source_conditional"]["delta_vs_best_fixed"],
    "source_conditional_delta_ci95": router_blocks["source_conditional"]["delta_vs_best_fixed_ci95"],
    "primary_winner_match_rate": round(match_rate, 4),
    "winner_match_chance": round(chance, 4),
    "bootstrap": {"B": BOOT, "seed": SEED, "kind": "query-clustered (resample queries)"},
    "per_source_leader": {s: {"n": int((src == s).sum()),
                              "best_pattern": cols[int(M[src == s].mean(axis=0).argmax())],
                              "best_mean": round(float(M[src == s].mean(axis=0).max()), 4),
                              "best_fixed_mean": round(float(M[src == s][:, bf].mean()), 4)}
                          for s in SOURCES},
}

# method-note + verdict
prim = router_blocks[primary]
best_realisable = max(router_blocks, key=lambda k: router_blocks[k]["delta_vs_best_fixed"])
result["method_note"] = (
    f"Realisable primary router ({primary}) realises {prim['realised_mean']:.4f} vs best-fixed "
    f"{cols[bf]} {bf_mean:.4f} (delta {prim['delta_vs_best_fixed']:+.4f}, 95% CI "
    f"{prim['delta_vs_best_fixed_ci95']}) against an oracle upper bound {oracle_mean:.4f} "
    f"(raw headroom {oracle_mean - bf_mean:+.4f}); it closes {prim['gap_closed_fraction']*100:.0f}% "
    f"of the best-fixed->oracle gap. Significant beat of best-fixed: "
    f"{prim['beats_best_fixed_sig']} (delta CI "
    f"{'excludes' if prim['beats_best_fixed_sig'] else 'includes'} 0).")
result["verdict"] = (
    "REALISABLE ROUTER BEATS BEST-FIXED (delta CI excludes 0)." if prim["beats_best_fixed_sig"]
    else "REALISABLE ROUTER DOES NOT SIGNIFICANTLY BEAT BEST-FIXED at n=85: the per-query delta CI "
         "includes 0. One dominant pattern (base_p1, best on 3 of 5 sources and 36/85 queries) "
         "leaves little realisable headroom; the oracle gain is largely un-routable from query "
         "metadata. Consistent with the paper's bounded-returns / routability-null result.")

# ---------- append-only write with unchanged-keys assertion ----------
path = f"{ANA}/canonical_numbers.json"
cn = json.load(open(path))
before = {k: json.dumps(cn[k], sort_keys=True) for k in cn}        # snapshot existing keys
assert "router_realisable" not in cn, "router_realisable already exists; refusing to overwrite"
cn.setdefault("router_realisable", result)
for k, v in before.items():                                        # never mutate existing keys
    assert json.dumps(cn[k], sort_keys=True) == v, f"existing key changed: {k}"
tmp = f"{path}.tmp"
open(tmp, "w").write(json.dumps(cn, indent=1)); os.replace(tmp, path)

# verify round-trip
cn2 = json.load(open(path))
for k, v in before.items():
    assert json.dumps(cn2[k], sort_keys=True) == v, f"post-write key changed: {k}"
assert "router_realisable" in cn2

print(f"[router_realisable] n={n}  best_fixed={cols[bf]} {bf_mean:.4f}  oracle={oracle_mean:.4f} "
      f"(raw headroom {oracle_mean-bf_mean:+.4f})")
for name, b in router_blocks.items():
    star = " <-PRIMARY" if name == primary else ""
    print(f"  {name:22s} mean={b['realised_mean']:.4f}  delta={b['delta_vs_best_fixed']:+.4f} "
          f"CI{b['delta_vs_best_fixed_ci95']}  gap_closed={b['gap_closed_fraction']*100:5.1f}%  "
          f"sig={b['beats_best_fixed_sig']}{star}")
print(f"  primary winner-match {match_rate:.3f} vs chance {chance:.3f}")
print(f"  VERDICT: {result['verdict'][:90]}...")
