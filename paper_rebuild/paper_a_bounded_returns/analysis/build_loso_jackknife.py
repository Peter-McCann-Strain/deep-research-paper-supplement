#!/usr/bin/env python
"""
T1_loso_jackknife  ->  canonical_numbers.json['loso_robustness']

Leave-one-source-out (LOSO) jackknife of the three headline gates, plus a
source-stratified block bootstrap of cluster membership. Pure re-analysis of
already-scored base-pattern parquets (df_overall_scores) -- no model calls, CPU only.

For each of the FIVE benchmark sources {custom, deepsearch_qa, draco, litqa2,
research_qa}, we DROP that source's queries and re-run:

  (Gate-1) the crossed random-effects mixed model  ovc ~ C(pattern)  with query
           and judge variance components (exactly build_numbers.variance_components),
           reporting ICC(query), ICC(judge), sigma2_resid.

  (Gate-3) the judge-robust pairwise separation: within each of the three judges
           {gpt52, claude_opus, claude_sonnet} a query-paired Wilcoxon test per
           pattern pair with Holm correction at 0.05; a pair is JUDGE-ROBUST iff it
           is Holm-significant in ALL THREE judges AND the three judges agree on the
           sign of the difference (exactly build_pairwise). We report the count out
           of 55 pairs and the count out of the 10 inner-5 cluster pairs.

  (rank table) the 3-judge sonnet-corrected per-pattern mean ranking (headline.rank_desc).

Then a source-stratified block bootstrap of CLUSTER MEMBERSHIP: resampling whole
queries WITH replacement WITHIN each source stratum (preserving the 5/20/40/10/15
source mix), we recompute, on the full panel, which of the 11 patterns is
judge-robustly separated from EVERY member of the six-pipeline top cluster
C6={p1,p4,p5,p6,p7,p8}. A pattern that is separated from all six is OUTSIDE the
cluster; one separated from none/some is reported by its stability fraction. This
quantifies how stable the "flat six-pipeline cluster" finding is to source mix.

Idioms mirror build_pairwise.py / build_numbers.py exactly (sonnet `ovc`
correction, base_p\\d+ filter, three-judge panel, Holm, crossed-RE mixedlm).
Deterministic (fixed seed); idempotent (load -> set one key -> write).

Run:  ./venv/bin/python paper_rebuild/paper_a_bounded_returns/analysis/build_loso_jackknife.py
"""
import json, warnings, itertools
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANON = f"{ANA}/canonical_numbers.json"

PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
PATS = [f"base_p{i}" for i in range(11)]
PAIRS = list(itertools.combinations(PATS, 2))
SOURCES = ["custom", "deepsearch_qa", "draco", "litqa2", "research_qa"]
# Six-pipeline top cluster and the inner-5 (drop p5, the one that needs an extra
# judge agreement) -- identical to build_pairwise.c6 / i5.
C6 = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]
I5 = ["base_p1", "base_p4", "base_p6", "base_p7", "base_p8"]
I5_PAIRS = list(itertools.combinations(I5, 2))
N_BOOT = 2000
SEED = 20260622

# ---------- load (sonnet-corrected ovc, source-joined base panel) ----------
ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
ov["ovc"] = ov["overall_score"].where(
    ~ov.judge.eq("claude_sonnet"), ov.get("overall_score_recomputed"))
q = pd.read_parquet(f"{A}/df_queries.parquet")
qid2src = dict(zip(q["query_id"], q["source"].astype(str)))
BASE = ov[ov.pattern.astype(str).str.match(r"^base_p([0-9]|10)$")
          & ov.judge.astype(str).isin(PANEL)].copy()
BASE["source"] = BASE["query_id"].map(qid2src)
assert BASE["source"].isna().sum() == 0, "unmapped base query_ids -> source"
assert set(BASE["source"].unique()) == set(SOURCES), \
    f"source set mismatch: {sorted(BASE['source'].unique())}"


def holm(pv):
    idx = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * pv[i]); adj[i] = min(run, 1.0)
    return adj


def judge_pairs(df, j, pairs=PAIRS):
    """Per-judge Holm-corrected paired-Wilcoxon significance + sign, per build_pairwise."""
    d = df[df.judge == j]
    wide = d.pivot_table(index="query_id", columns="pattern",
                         values="ovc", observed=True)
    pv = []; sign = {}
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            pv.append(1.0); sign[(a, b)] = 0.0; continue
        s = wide[[a, b]].dropna()
        try:
            p = wilcoxon(s[a], s[b]).pvalue if len(s) else 1.0
        except Exception:
            p = 1.0
        pv.append(p)
        sign[(a, b)] = float(np.sign((s[a] - s[b]).mean())) if len(s) else 0.0
    adj = holm(np.array(pv))
    sig = {pairs[i]: bool(adj[i] < 0.05) for i in range(len(pairs))}
    return sig, sign


def gate3_robust(df, pairs=PAIRS):
    """Judge-robust separations: Holm-sig in all 3 judges AND unanimous sign."""
    res = {j: judge_pairs(df, j, pairs) for j in PANEL}
    robust = []
    for pr in pairs:
        all_sig = all(res[j][0][pr] for j in PANEL)
        signs = {res[j][1][pr] for j in PANEL}
        if all_sig and len(signs) == 1 and 0.0 not in signs:
            robust.append(pr)
    return robust, res


def gate1_mixed(df):
    """Crossed-RE variance components -- exactly build_numbers.variance_components."""
    import statsmodels.formula.api as smf
    b = df.rename(columns={"query_id": "query"}).copy()
    for c in ("pattern", "judge", "query"):
        b[c] = b[c].astype(str)
    b["grp"] = 1
    md = smf.mixedlm("ovc ~ C(pattern)", b, groups=b["grp"],
                     vc_formula={"query": "0+C(query)", "judge": "0+C(judge)"})
    f = md.fit(reml=True, method="lbfgs")
    vq = float(f.vcomp[0]); vj = float(f.vcomp[1]); ve = float(f.scale)
    tot = vq + vj + ve
    return {"sigma2_query": round(vq, 5), "sigma2_judge": round(vj, 5),
            "sigma2_resid": round(ve, 5),
            "icc_query": round(vq / tot, 4), "icc_judge": round(vj / tot, 4),
            "converged": bool(f.converged), "n": int(len(b))}


def rank_table(df):
    """3-judge sonnet-corrected per-pattern mean, ranked desc -- headline.rank_desc."""
    means = (df.groupby("pattern", observed=True)["ovc"].mean()
               .sort_values(ascending=False))
    return {"rank_desc": [str(p) for p in means.index],
            "means": {str(p): round(float(v), 4) for p, v in means.items()}}


# ---------- LOSO: drop one source at a time ----------
print("[loso] full-panel baseline ...")
full_robust, _ = gate3_robust(BASE)
full_inner5, _ = gate3_robust(BASE, I5_PAIRS)
baseline = {
    "gate1_mixed": gate1_mixed(BASE),
    "gate3_judge_robust_of_55": len(full_robust),
    "gate3_inner5_robust_of_10": len(full_inner5),
    "rank_table": rank_table(BASE),
    "n_queries": int(BASE.query_id.nunique()),
}
full_rank = baseline["rank_table"]["rank_desc"]

per_source = {}
for src in SOURCES:
    sub = BASE[BASE.source != src].copy()
    rob, _ = gate3_robust(sub)
    in5, _ = gate3_robust(sub, I5_PAIRS)
    rt = rank_table(sub)
    # Kendall-tau-style displacement of the rank order vs full panel.
    pos_full = {p: i for i, p in enumerate(full_rank)}
    max_disp = max(abs(pos_full[p] - i) for i, p in enumerate(rt["rank_desc"]))
    per_source[src] = {
        "dropped_source": src,
        "n_dropped_queries": int(BASE[BASE.source == src].query_id.nunique()),
        "n_queries_remaining": int(sub.query_id.nunique()),
        "gate1_mixed": gate1_mixed(sub),
        "gate3_judge_robust_of_55": len(rob),
        "gate3_inner5_robust_of_10": len(in5),
        "rank_table": rt["rank_desc"],
        "rank_max_displacement_vs_full": int(max_disp),
        "top1_pattern": rt["rank_desc"][0],
        "top1_matches_full": bool(rt["rank_desc"][0] == full_rank[0]),
    }
    print(f"[loso] drop {src:14s}: robust55={len(rob):2d} inner5={len(in5)} "
          f"icc_q={per_source[src]['gate1_mixed']['icc_query']} "
          f"top1={rt['rank_desc'][0]} maxdisp={max_disp}")

# ---------- source-stratified block bootstrap of cluster membership ----------
# Resample whole queries WITH replacement WITHIN each source stratum (keeps the
# 5/20/40/10/15 source mix), then on the full three-judge panel recompute which
# patterns are judge-robustly separated from EVERY member of C6. A pattern
# separated from all six C6 members is OUTSIDE the flat cluster on that resample.
print(f"[loso] source-stratified block bootstrap of cluster membership "
      f"({N_BOOT} resamples) ...")
qids_by_src = {s: BASE[BASE.source == s].query_id.unique() for s in SOURCES}
rng = np.random.default_rng(SEED)
# index BASE by query for fast block assembly
by_q = {qid: g for qid, g in BASE.groupby("query_id", observed=True)}
C6_set = set(C6)
non_c6 = [p for p in PATS if p not in C6_set]
# count, per pattern, how often it is "separated from the whole cluster" (i.e.
# judge-robustly distinct from all 6 C6 members) -> outside-cluster fraction.
outside_count = {p: 0 for p in PATS}
# also track inner-6 internal separation count (how many of the 15 within-C6
# pairs are judge-robust on each resample) -> stability of "flat cluster".
c6_pairs = list(itertools.combinations(C6, 2))
inner6_sep_counts = []

for _ in range(N_BOOT):
    drawn = []
    for s in SOURCES:
        pool = qids_by_src[s]
        drawn.append(rng.choice(pool, size=len(pool), replace=True))
    drawn = np.concatenate(drawn)
    # Assemble resampled panel; duplicate query draws get distinct synthetic ids
    # so paired-Wilcoxon treats them as independent paired blocks.
    frames = []
    for k, qid in enumerate(drawn):
        g = by_q[qid].copy()
        g["query_id"] = f"{qid}__b{k}"
        frames.append(g)
    bs = pd.concat(frames, ignore_index=True)
    robust, _ = gate3_robust(bs)
    robust_set = set(robust)

    def sep(a, b, robust_set=robust_set):
        return (a, b) in robust_set or (b, a) in robust_set
    # pattern outside cluster iff separated from ALL six C6 members
    for p in PATS:
        targets = [c for c in C6 if c != p]
        if targets and all(sep(p, c) for c in targets):
            outside_count[p] += 1
    inner6_sep_counts.append(sum(1 for a, b in c6_pairs if sep(a, b)))

inner6 = np.array(inner6_sep_counts)
block_bootstrap = {
    "n_resamples": N_BOOT,
    "seed": SEED,
    "strata": {s: int(len(qids_by_src[s])) for s in SOURCES},
    "cluster_definition": C6,
    "outside_cluster_fraction": {
        p: round(outside_count[p] / N_BOOT, 4) for p in PATS},
    "inner6_pairs_separated_mean": round(float(inner6.mean()), 4),
    "inner6_pairs_separated_ci95": [int(np.percentile(inner6, 2.5)),
                                    int(np.percentile(inner6, 97.5))],
    "prob_cluster_internally_flat": round(float((inner6 == 0).mean()), 4),
    "note": ("Whole queries resampled with replacement within each source "
             "stratum (mix preserved). A pattern is OUTSIDE the cluster on a "
             "resample iff it is judge-robustly separated from all six C6 "
             "members. inner6=# of 15 within-C6 pairs that are judge-robust; "
             "prob_cluster_internally_flat = P(inner6==0)."),
}

out = {
    "spec": ("LOSO source jackknife (drop one of "
             "{custom,deepsearch_qa,draco,litqa2,research_qa}) of Gate-1 crossed-RE "
             "mixed model + Gate-3 judge-robust pairwise separation + 3-judge rank "
             "table, plus a source-stratified block bootstrap of cluster membership."),
    "sources": SOURCES,
    "panel": PANEL,
    "full_panel_baseline": baseline,
    "leave_one_source_out": per_source,
    "block_bootstrap_cluster_membership": block_bootstrap,
    "robustness_summary": {
        "gate3_robust55_range": [
            min(v["gate3_judge_robust_of_55"] for v in per_source.values()),
            max(v["gate3_judge_robust_of_55"] for v in per_source.values())],
        "gate3_inner5_max_over_loso": max(
            v["gate3_inner5_robust_of_10"] for v in per_source.values()),
        "top1_always_matches_full": all(
            v["top1_matches_full"] for v in per_source.values()),
        "max_rank_displacement_over_loso": max(
            v["rank_max_displacement_vs_full"] for v in per_source.values()),
        "icc_query_range": [
            round(min(v["gate1_mixed"]["icc_query"] for v in per_source.values()), 4),
            round(max(v["gate1_mixed"]["icc_query"] for v in per_source.values()), 4)],
        "note": ("Headline survives LOSO iff inner-5 stays 0/10 robustly separated "
                 "and the within-cluster pairs never become judge-robust across all "
                 "five drops; the rank table's leader and the cluster set should be "
                 "stable up to within-cluster micro-order."),
    },
}

# ---------- persist: load -> set one key -> write (build_pairwise idiom) ----------
p = json.load(open(CANON))
p["loso_robustness"] = out
json.dump(p, open(CANON, "w"), indent=1)
print("\n[loso] wrote canonical key 'loso_robustness'")
print(json.dumps(out["robustness_summary"], indent=1))
print(json.dumps(block_bootstrap["outside_cluster_fraction"], indent=1))
