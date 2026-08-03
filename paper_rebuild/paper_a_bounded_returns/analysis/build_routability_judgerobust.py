#!/usr/bin/env python
"""T1 ROUTABILITY judge-robustness — re-run the ENTIRE E1 null over {gpt52, opus, sonnet,
panel_mean}, reusing the per-query score matrices ALREADY on disk in df_overall_scores.parquet.

This is a robustness wrapper around build_routability.py (Stage A) and build_routability_stageb.py
(Stage B). It does NOT re-judge anything: it reads the existing Opus (~84q), Sonnet (~85q), gpt52
(~87q) and panel-mean (84q 3-judge intersection) 11-architecture matrices and recomputes the full
battery per judge. It appends ONE new canonical key `routability.judge_robustness` (a dict keyed by
judge); it does NOT touch the existing `routability` Stage-A/Stage-B gpt52 blocks.

Pre-registration: docs/publication/prereg/prereg_E1.md. Decision gate G1 (best realizable router
headroom < 0.02 over best-fixed => null framing) is re-evaluated per judge.

Per-judge battery (mirrors the two existing builders' idioms exactly):
  Stage-A pieces:
    - architecture means, best-fixed identity, raw oracle gain over P1 and over best-fixed
    - per-dimension and per-source raw oracle gain over P1 (from df_scores.parquet)
    - winner_label_reliability (split-half test-retest of per-query argmax on the REPLICATE corpus)
        -> ONLY computable for gpt52 (the only judge with a variance/replicate corpus on disk);
           recorded N/A (with reason) for opus/sonnet/panel_mean.
    - replicate_cv_headroom (real-independent-run noise correction on the replicated subset)
        -> same gpt52-only availability.
    - noise_corrected_headroom (parametric bootstrap: select on T+N(0,sigma2_run), evaluate on T).
        sigma2_run is the gpt52-derived run-noise scalar from canonical variance_decomposition; it is
        the noise MODEL and is applied judge-agnostically (flagged in the output).
  Stage-B pieces (need only the single-run 11-arch matrix, computable for ALL judges):
    - source_router / knn_router_k7 / gbm_router / logreg_router LOOCV realized headroom
    - leave-one-benchmark-out (source + gbm)
    - strongest_realizable_router = candidate-restricted ridge {p1,p4,p5,p6,p7}, with bootstrap CI
    - per-judge Gate G1 keyed on the best realized LOOCV router headroom.

panel_mean: per-(pattern,query) mean across the three judges, over the 84-query intersection where
all 11 archs are present for all three judges. best-fixed is recomputed within each judge.

Determinism: same SEED and sorted inputs as the two source builders. Idempotent: rewrites only the
`routability.judge_robustness` key via atomic tmp+os.replace; never clobbers other keys.
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
CANON = f"{ANA}/canonical_numbers.json"
SEED = 20260611
ARCH = [f"base_p{i}" for i in range(11)]
DIMS = ["information_recall", "factual_accuracy", "coverage", "analytical_depth",
        "citation_quality", "logical_coherence", "organization", "instruction_following",
        "attribution_quality"]
P1 = "base_p1"
# judge label -> df judge value(s). panel_mean is synthesised from the three real judges.
JUDGE_DF = {"gpt52": "gpt52", "opus": "claude_opus", "sonnet": "claude_sonnet"}
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
JUDGES = ["gpt52", "opus", "sonnet", "panel_mean"]

O = pd.read_parquet(f"{ROOT}/data/analysis/df_overall_scores.parquet")
Sd = pd.read_parquet(f"{ROOT}/data/analysis/df_scores.parquet")
Q = pd.read_parquet(f"{ROOT}/data/analysis/df_queries.parquet")
cn = json.load(open(CANON))
sigma_run = float(np.sqrt(cn["variance_decomposition"]["pooled"]["sigma2_run"]))


# ----- per-judge overall 11-arch matrix (sorted columns, queries with all 11) -----
def overall_pivot(judge):
    """Return (piv DataFrame: index=query_id sorted, columns=sorted 11 archs)."""
    if judge == "panel_mean":
        mats = {}
        for dfj in PANEL:
            g = O[(O.judge == dfj) & (O.pattern.isin(ARCH))][["pattern", "query_id", "overall_score_recomputed"]].dropna()
            pv = g.pivot_table(index="query_id", columns="pattern", values="overall_score_recomputed", observed=True)
            mats[dfj] = pv[sorted(pv.columns)].dropna().sort_index()
        common = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
        cols = sorted(set.intersection(*[set(m.columns) for m in mats.values()]))
        stacked = sum(m.loc[common, cols] for m in mats.values()) / float(len(mats))
        return stacked.sort_index()
    dfj = JUDGE_DF[judge]
    g = O[(O.judge == dfj) & (O.pattern.isin(ARCH))][["pattern", "query_id", "overall_score_recomputed"]].dropna()
    piv = g.pivot_table(index="query_id", columns="pattern", values="overall_score_recomputed", observed=True)
    return piv[sorted(piv.columns)].dropna().sort_index()


def dim_pivot(judge, dim):
    """Per-dimension matrix; panel_mean averages the three real judges over the intersection."""
    def one(dfj):
        d = Sd[(Sd.judge == dfj) & (Sd.pattern.isin(ARCH)) & (Sd.dimension == dim)]
        pv = d.pivot_table(index="query_id", columns="pattern", values="score", observed=True)
        return pv[[c for c in sorted(pv.columns)]].dropna()
    if judge == "panel_mean":
        mats = {dfj: one(dfj) for dfj in PANEL}
        common = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
        cols = sorted(set.intersection(*[set(m.columns) for m in mats.values()]))
        if not common or not cols:
            return pd.DataFrame()
        return sum(m.loc[common, cols] for m in mats.values()) / float(len(mats))
    return one(JUDGE_DF[judge])


# ============================ STAGE A (per judge) ============================
def stage_a(judge, rng):
    piv = overall_pivot(judge)
    M = piv.values
    cols = list(piv.columns)
    means = {p: round(float(piv[p].mean()), 4) for p in cols}
    best_fixed = max(means, key=means.get)

    def oracle_gain(mat, base_col):
        return float(mat.max(axis=1).mean() - mat[:, cols.index(base_col)].mean())

    raw = {
        "best_fixed_by_mean": best_fixed, "p1_mean": means.get(P1), "p4_mean": means.get("base_p4"),
        "n_queries_all11": int(M.shape[0]),
        "oracle_mean": round(float(M.max(axis=1).mean()), 4),
        "raw_gain_over_p1": round(oracle_gain(M, P1), 4) if P1 in cols else None,
        "raw_gain_over_best_fixed": round(oracle_gain(M, best_fixed), 4),
    }
    # per-dimension raw oracle gain over P1
    raw["per_dimension_gain_over_p1"] = {}
    for dim in DIMS:
        pv = dim_pivot(judge, dim)
        if P1 in getattr(pv, "columns", []) and len(pv) > 5:
            raw["per_dimension_gain_over_p1"][dim] = round(float(pv.values.max(axis=1).mean() - pv[P1].mean()), 4)
    # per-source raw oracle gain over P1
    src = Q.set_index("query_id")["source"].to_dict()
    raw["per_source_gain_over_p1"] = {}
    if P1 in cols:
        for s in sorted(set(src.get(q) for q in piv.index if src.get(q))):
            qs = [q for q in piv.index if src.get(q) == s]
            if len(qs) >= 4:
                sub = piv.loc[qs].values
                raw["per_source_gain_over_p1"][s] = {"n": len(qs),
                    "gain": round(float(sub.max(axis=1).mean() - sub[:, cols.index(P1)].mean()), 4)}

    # --- winner-label reliability + replicate-CV: only gpt52 has a replicate corpus on disk ---
    NA = ("REPLICATE corpus exists only for gpt52 in df_overall_scores.parquet "
          "(pattern_family=='variance' is empty for opus/sonnet); not recomputable for this judge.")
    if judge == "gpt52":
        winner_reliability, replicate_cv, nc_extra = _replicate_pieces(rng)
    else:
        winner_reliability = {"available": False, "reason": NA}
        replicate_cv = {"available": False, "reason": NA}
        nc_extra = None

    # --- noise-corrected parametric bootstrap (computable for every judge) ---
    B = 4000
    i_p1 = cols.index(P1) if P1 in cols else None
    i_bf = cols.index(best_fixed)
    gains_p1, gains_bf = [], []
    for _ in range(B):
        noisy = M + rng.normal(0, sigma_run, size=M.shape)
        pick = noisy.argmax(axis=1)
        deployed = M[np.arange(M.shape[0]), pick].mean()
        if i_p1 is not None:
            gains_p1.append(deployed - M[:, i_p1].mean())
        gains_bf.append(deployed - M[:, i_bf].mean())
    nc_p1 = float(np.mean(gains_p1)) if gains_p1 else None
    nc_bf = float(np.mean(gains_bf))
    noise_corrected = {
        "sigma_run_used": round(sigma_run, 4),
        "sigma_run_provenance": "gpt52-derived sigma2_run from canonical variance_decomposition.pooled; "
                                "applied judge-agnostically as the run-noise MODEL (no per-judge replicates "
                                "exist for opus/sonnet/panel_mean).",
        "n_boot": B,
        "headroom_over_p1": round(nc_p1, 4) if nc_p1 is not None else None,
        "headroom_over_p1_ci95": ([round(float(np.percentile(gains_p1, 2.5)), 4),
                                   round(float(np.percentile(gains_p1, 97.5)), 4)] if gains_p1 else None),
        "headroom_over_best_fixed": round(nc_bf, 4),
        "fraction_of_raw_surviving_over_p1": (round(nc_p1 / raw["raw_gain_over_p1"], 3)
                                              if nc_p1 is not None and raw["raw_gain_over_p1"] else None),
        "method": "select on T + N(0,sigma2_run), evaluate on observed T (optimistically biased, "
                  "evaluates on the same noisy run; see Stage-B LOOCV for the unbiased estimate).",
    }
    return {
        "architecture_means": means, "raw": raw,
        "winner_label_reliability": winner_reliability,
        "noise_corrected_headroom": noise_corrected,
        "replicate_cv_headroom": replicate_cv,
    }


def _replicate_pieces(rng):
    """gpt52-only: split-half winner-label reliability + replicate-CV headroom (verbatim idiom)."""
    ov = O[(O.judge == "gpt52") & (O.pattern_family == "variance")][["pattern", "query_id", "overall_score_recomputed"]].dropna().copy()
    ov["arch"] = ov.pattern.str.extract(r"(base_p\d+)_v")
    varch = sorted(ov.arch.dropna().unique())
    vqs = sorted(ov.query_id.unique())
    reps = {a: sorted(ov[ov.arch == a].pattern.unique()) for a in varch}

    def half_argmax(rep_assign):
        win = {}
        for q in vqs:
            best_a, best_v = None, -1e9
            for a in varch:
                ps = rep_assign[a]
                vals = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(ps))].overall_score_recomputed
                if len(vals):
                    m = float(vals.mean())
                    if m > best_v:
                        best_v, best_a = m, a
            win[q] = best_a
        return win

    N_SPLIT = 200
    agrees = []
    for _ in range(N_SPLIT):
        h1, h2 = {}, {}
        for a in varch:
            r = reps[a][:]
            if len(r) >= 2:
                rng.shuffle(r); k = len(r) // 2
                h1[a], h2[a] = r[:max(k, 1)], r[max(k, 1):] or r[:1]
            else:
                h1[a] = h2[a] = r
        w1, w2 = half_argmax(h1), half_argmax(h2)
        both = [q for q in vqs if w1[q] and w2[q]]
        if both:
            agrees.append(np.mean([w1[q] == w2[q] for q in both]))
    from collections import Counter
    full_win = half_argmax({a: reps[a] for a in varch})
    freq = Counter(full_win[q] for q in vqs if full_win[q]); tot = sum(freq.values())
    chance = float(sum((c / tot) ** 2 for c in freq.values())) if tot else None
    winner_reliability = {
        "available": True, "n_replicated_archs": len(varch), "archs": varch, "n_queries": len(vqs),
        "test_retest_agreement": round(float(np.mean(agrees)), 4) if agrees else None,
        "chance_agreement": round(chance, 4) if chance else None,
        "above_chance": bool(np.mean(agrees) > chance) if agrees and chance else None,
        "note": "split-half test-retest of the per-query argmax among the replicated architectures "
                "(gpt52 variance corpus); chance = sum of squared winner frequencies.",
    }
    # replicate-CV headroom
    sub_best = max(varch, key=lambda a: ov[ov.arch == a].overall_score_recomputed.mean())
    cv_gains = []
    for _ in range(N_SPLIT):
        h1, h2 = {}, {}
        for a in varch:
            r = reps[a][:]
            if len(r) >= 2:
                rng.shuffle(r); k = max(len(r) // 2, 1); h1[a], h2[a] = r[:k], r[k:] or r[:1]
            else:
                h1[a] = h2[a] = r
        m1, m2 = {}, {}
        for a in varch:
            for q in vqs:
                v1 = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(h1[a]))].overall_score_recomputed
                v2 = ov[(ov.arch == a) & (ov.query_id == q) & (ov.pattern.isin(h2[a]))].overall_score_recomputed
                m1[(a, q)] = float(v1.mean()) if len(v1) else np.nan
                m2[(a, q)] = float(v2.mean()) if len(v2) else np.nan
        dep, bf = [], []
        for q in vqs:
            cand = [(a, m1[(a, q)]) for a in varch if not np.isnan(m1[(a, q)]) and not np.isnan(m2[(a, q)])]
            if not cand:
                continue
            win = max(cand, key=lambda x: x[1])[0]
            if not np.isnan(m2[(win, q)]) and not np.isnan(m2[(sub_best, q)]):
                dep.append(m2[(win, q)]); bf.append(m2[(sub_best, q)])
        if dep:
            cv_gains.append(np.mean(dep) - np.mean(bf))
    replicate_cv = {
        "available": True, "subset_archs": varch, "subset_best_fixed": sub_best,
        "cv_headroom": round(float(np.mean(cv_gains)), 4) if cv_gains else None,
        "cv_headroom_ci95": ([round(float(np.percentile(cv_gains, 2.5)), 4),
                              round(float(np.percentile(cv_gains, 97.5)), 4)] if cv_gains else None),
        "note": "pick winner on replicate-half-1, evaluate on half-2 over the replicated archs; real "
                "independent runs, no simulated noise; Monte-Carlo over seeded half-splits (gpt52 only).",
    }
    return winner_reliability, replicate_cv, None


# ============================ STAGE B (per judge) ============================
def stage_b(judge, rng):
    piv = overall_pivot(judge)
    qids = list(piv.index)
    cols = list(piv.columns)
    M = piv.values
    bf = max(range(len(cols)), key=lambda j: M[:, j].mean())
    bf_mean = M[:, bf].mean()
    oracle_mean = M.max(axis=1).mean()

    def caus(t):
        return len(re.findall(r"\b(why|how|cause|because|impact|effect|reason|lead to|result in|influence)\b", str(t), re.I))

    def entden(t):
        w = str(t).split()
        return (sum(1 for x in w if x[:1].isupper()) + len(re.findall(r"\d", str(t)))) / max(len(w), 1)

    DIFF = {"simple": 0, "moderate": 1, "complex": 2}
    Qi = Q.set_index("query_id")
    SOURCES = sorted(Qi.source.unique())

    def features(qid):
        r = Qi.loc[qid]
        f = [np.log1p(len(str(r.query_text))), caus(r.query_text), entden(r.query_text), DIFF.get(r.difficulty, 1)]
        f += [1.0 if r.source == s else 0.0 for s in SOURCES]
        return f

    X = np.array([features(q) for q in qids])
    src = np.array([Qi.loc[q].source for q in qids])
    winner = M.argmax(axis=1)

    def realized(picks):
        return float(np.mean([M[i, picks[i]] for i in range(len(qids))]) - bf_mean)

    def realized_diffs(picks):
        return np.array([M[i, picks[i]] - M[i, bf] for i in range(len(qids))])

    def boot_ci(diffs, B=5000):
        n = len(diffs); idx = np.arange(n)
        means = [diffs[rng.choice(idx, n, replace=True)].mean() for _ in range(B)]
        return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

    CAND = [cols.index(c) for c in ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7"] if c in cols]

    def ridge_candidate_router_loocv():
        Xs = StandardScaler().fit_transform(X)
        picks = []
        for i in range(len(qids)):
            tr = np.arange(len(qids)) != i
            preds = []
            for j in CAND:
                r = Ridge(alpha=1.0); r.fit(Xs[tr], M[tr, j]); preds.append(r.predict(Xs[i:i + 1])[0])
            picks.append(CAND[int(np.argmax(preds))])
        return picks

    def source_router_loocv():
        picks = []
        for i in range(len(qids)):
            mask = (np.arange(len(qids)) != i) & (src == src[i])
            picks.append(bf if mask.sum() == 0 else int(M[mask].mean(axis=0).argmax()))
        return realized(picks)

    def knn_router_loocv(k=7):
        Xs = StandardScaler().fit_transform(X)
        picks = []
        for i in range(len(qids)):
            d = np.linalg.norm(Xs - Xs[i], axis=1); d[i] = np.inf
            picks.append(int(M[np.argsort(d)[:k]].mean(axis=0).argmax()))
        return realized(picks)

    def clf_router_loocv(make):
        Xs = StandardScaler().fit_transform(X)
        picks = []
        for i in range(len(qids)):
            tr = np.arange(len(qids)) != i
            ytr = winner[tr]
            if len(np.unique(ytr)) < 2:
                picks.append(bf); continue
            clf = make(); clf.fit(Xs[tr], ytr)
            picks.append(int(clf.predict(Xs[i:i + 1])[0]))
        return realized(picks)

    def lobo(router):
        picks = [None] * len(qids)
        for s in SOURCES:
            te = src == s; trm = ~te
            if router == "source":
                pick = int(M[trm].mean(axis=0).argmax())
                for i in np.where(te)[0]: picks[i] = pick
            else:
                Xs = StandardScaler().fit(X[trm]).transform(X)
                ytr = winner[trm]
                if len(np.unique(ytr)) < 2:
                    for i in np.where(te)[0]: picks[i] = bf
                    continue
                clf = GradientBoostingClassifier(random_state=SEED); clf.fit(Xs[trm], ytr)
                for i in np.where(te)[0]: picks[i] = int(clf.predict(Xs[i:i + 1])[0])
        return realized(picks)

    res = {
        "n_queries": len(qids), "best_fixed": cols[bf], "best_fixed_mean": round(float(bf_mean), 4),
        "oracle_mean": round(float(oracle_mean), 4),
        "raw_oracle_headroom": round(float(oracle_mean - bf_mean), 4),
        "loocv_realized_headroom": {
            "source_router": round(source_router_loocv(), 4),
            "knn_router_k7": round(knn_router_loocv(7), 4),
            "gbm_router": round(clf_router_loocv(lambda: GradientBoostingClassifier(random_state=SEED)), 4),
            "logreg_router": round(clf_router_loocv(lambda: LogisticRegression(max_iter=2000)), 4),
        },
        "leave_one_benchmark_out_headroom": {
            "source_router": round(lobo("source"), 4),
            "gbm_router": round(lobo("gbm"), 4),
        },
    }
    best_router_headroom = max(res["loocv_realized_headroom"].values())
    res["best_router_loocv_headroom"] = round(float(best_router_headroom), 4)

    ridge_picks = ridge_candidate_router_loocv()
    res["strongest_realizable_router"] = {
        "name": "candidate_restricted_ridge_{p1,p4,p5,p6,p7}",
        "loocv_headroom": round(realized(ridge_picks), 4),
        "loocv_ci95": boot_ci(realized_diffs(ridge_picks)),
    }

    def src_picks():
        return [int(M[(np.arange(len(qids)) != i) & (src == src[i])].mean(axis=0).argmax())
                if ((np.arange(len(qids)) != i) & (src == src[i])).sum() else bf for i in range(len(qids))]

    def knn_picks():
        Xs = StandardScaler().fit_transform(X); out = []
        for i in range(len(qids)):
            d = np.linalg.norm(Xs - Xs[i], axis=1); d[i] = np.inf
            out.append(int(M[np.argsort(d)[:7]].mean(axis=0).argmax()))
        return out

    res["headroom_ci95"] = {
        "source_router": boot_ci(realized_diffs(src_picks())),
        "knn_router_k7": boot_ci(realized_diffs(knn_picks())),
        "strongest_realizable_router": res["strongest_realizable_router"]["loocv_ci95"],
    }
    G1_THRESHOLD = 0.02
    fires = best_router_headroom < G1_THRESHOLD
    res["gate_g1_stage_b"] = {
        "threshold": G1_THRESHOLD,
        "best_router_loocv_headroom": round(float(best_router_headroom), 4),
        "fires": bool(fires),
        "decision": ("G1 FIRES: no realizable feature router beats best-fixed by >= 0.02 out-of-sample "
                     f"(best {best_router_headroom:+.4f} over {cols[bf]})." if fires else
                     "G1 does NOT fire: a feature router realizes >= 0.02 over best-fixed out-of-sample."),
    }
    return res


def main():
    per_judge = {}
    for judge in JUDGES:
        rng = np.random.default_rng(SEED)  # fresh seeded generator per judge (deterministic)
        a = stage_a(judge, rng)
        b = stage_b(judge, rng)
        per_judge[judge] = {"stage_a": a, "stage_b": b}

    # cross-judge G1 robustness summary
    g1 = {j: per_judge[j]["stage_b"]["gate_g1_stage_b"]["fires"] for j in JUDGES}
    best_routers = {j: per_judge[j]["stage_b"]["best_router_loocv_headroom"] for j in JUDGES}
    raw_oracle = {j: per_judge[j]["stage_b"]["raw_oracle_headroom"] for j in JUDGES}
    best_fixed = {j: per_judge[j]["stage_b"]["best_fixed"] for j in JUDGES}

    out = {
        "_note": "T1 judge-robustness of the E1 routability null. Re-runs the ENTIRE E1 battery (Stage A "
                 "raw oracle / noise-corrected / winner-label + Stage B LOOCV/LOBO + strongest-realizable "
                 "router) over judge in {gpt52, opus, sonnet, panel_mean}, REUSING the on-disk per-query "
                 "score matrices (no new judging). winner_label_reliability and replicate_cv_headroom are "
                 "gpt52-only (the replicate corpus exists only for gpt52); all other pieces are computed for "
                 "every judge. panel_mean = per-(arch,query) mean of the three judges over their 84-query "
                 "all-11 intersection. sigma_run for the parametric bootstrap is the gpt52 variance_decomp "
                 "scalar applied as a judge-agnostic noise model. Prereg: prereg_E1.md.",
        "prereg": "docs/publication/prereg/prereg_E1.md",
        "judges": JUDGES,
        "g1_fires_by_judge": g1,
        "best_router_loocv_headroom_by_judge": best_routers,
        "raw_oracle_headroom_by_judge": raw_oracle,
        "best_fixed_by_judge": best_fixed,
        "robust_conclusion": ("G1 FIRES on ALL judges" if all(g1.values()) else
                              "G1 does NOT fire on all judges: " + ", ".join(f"{j}={g1[j]}" for j in JUDGES)),
        "per_judge": per_judge,
    }

    cn2 = json.load(open(CANON))
    cn2.setdefault("routability", {})["judge_robustness"] = out
    _tmp = f"{CANON}.tmp"
    open(_tmp, "w").write(json.dumps(cn2, indent=1))
    os.replace(_tmp, CANON)

    print("routability.judge_robustness written.")
    for j in JUDGES:
        sb = per_judge[j]["stage_b"]
        print(f"  {j:11s} best_fixed={sb['best_fixed']:8s} raw_oracle={sb['raw_oracle_headroom']:+.4f} "
              f"best_router={sb['best_router_loocv_headroom']:+.4f} G1_fires={g1[j]}")
    print(f"  *** ROBUST: {out['robust_conclusion']}")


if __name__ == "__main__":
    main()
