#!/usr/bin/env python
"""A2: REALISABLE best-of-N selectors vs the oracle-decoupled upper bound.

build_bestofn_decoupled.py selects the best of k independent P0 replicates using a
HELD-OUT-CRITERIA ORACLE (split each dimension's criteria into a selection half A and a
disjoint scoring half B; pick the replicate with the best half-A judge score, report its
half-B judge score). That is an UPPER BOUND on any deployable selector, because the selector
peeks at (one half of) the very judge that scores the report. The paper flags exactly this:
"oracle selection on held-out criteria remains an upper bound on any deployable selector."

A2 replaces the oracle with two REALISABLE selectors that pick among the k reports using ONLY
the report text -- they never see any judge criterion:

  1. realisable_selfconsistency: for each query and each k, pick the medoid report -- the one
     with the highest mean pairwise TF-IDF cosine similarity to the other k-1 reports. This is
     a pure self-consistency / representativeness selector (the report most typical of the
     ensemble), computable at deployment time from text alone.

  2. realisable_heldout_scorer: a cheap proxy quality score computed from the report itself
     (unique in-text citations, unique source URLs, section-heading count, and a sane-length
     band membership), combined as an equal-weight sum of within-query z-scores (pre-registered,
     no tuning against the judge). Pick the argmax; several single-signal ablations are also
     reported. The picked report is then scored on the FULL / held-out judge, exactly as a
     deployed selector would be.

Both selectors' picks are scored on TWO independent bases and reported against the SAME
references the oracle uses:
  * half-B judge basis  -> directly overlays best_of_n.decoupled.curve (the oracle upper bound)
    and its cluster_mean_half_B; selection (text) is independent of the judge so half-B is an
    unbiased estimate of the picked report's quality with no selection-on-noise winner's curse.
  * full-criteria basis -> the deployment-realistic realised score, vs the full cluster mean.

For each selector and k=1..N we report the realised curve, its gap to the orchestrated cluster
(seeded query-bootstrap CI), the oracle-decoupled value at the same k, the fraction of the
oracle's realised headroom captured, the fraction of the cluster gap closed, and the smallest k
(if any) at which the realisable curve draws level with the cluster (point estimate reaching the
cluster, plus the CI-overlap k for context).

$0 pure re-analysis: no generation, no API. Reads existing GPT-5.2 verdicts + on-disk report
text. Deterministic (fixed tokenizer, argmax ties -> earliest replicate, seeded bootstrap
20260703). Appends NEW subkeys best_of_n.realisable_selfconsistency and
best_of_n.realisable_heldout_scorer; refuses to overwrite; the forward oracle-decoupled curve is
recomputed and asserted equal to the stored canonical values (drift guard) before any write;
atomic tmp + os.replace.
"""
import pandas as pd, numpy as np, json, warnings, glob, os, re, math
from collections import Counter
warnings.filterwarnings("ignore")

ROOT = "."
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
CANONICAL = f"{ANA}/canonical_numbers.json"
EXP = f"{ROOT}/results/experiments"
SEED = 20260703
NBOOT = 5000

W = {"information_recall": 0.20, "factual_accuracy": 0.20, "coverage": 0.10,
     "analytical_depth": 0.15, "citation_quality": 0.10, "logical_coherence": 0.05,
     "organization": 0.05, "instruction_following": 0.10, "attribution_quality": 0.05}

VARQ = set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
P0 = ["base_p0"] + sorted([os.path.basename(d) for d in glob.glob(f"{ROOT}/results/judge_gpt52/base_p0_v*")
                           if re.match(r"base_p0_v\d+$", os.path.basename(d))],
                          key=lambda s: int(s.split("_v")[1]))
CLUSTER = ["base_p1", "base_p4", "base_p5", "base_p6", "base_p7", "base_p8"]

# ============================================================================
# 1. Reproduce the decoupled loader EXACTLY (drift guard + shared basis/ordering)
# ============================================================================
ov = pd.read_parquet(f"{A}/df_overall_scores.parquet")
g = ov[ov.judge.eq("gpt52")]

def patq(p, q):
    d = g[(g.pattern == p) & (g.query_id == q)]
    return float(d.overall_score.iloc[0]) if len(d) and pd.notna(d.overall_score.iloc[0]) else None

samples = {}
for q in VARQ:
    s = [patq(p, q) for p in P0]
    s = [x for x in s if x is not None]
    if len(s) >= 3:
        samples[q] = s
qs = sorted(samples)
N = min(len(samples[q]) for q in qs)

vd = pd.read_parquet(f"{A}/df_verdicts.parquet")
vd = vd[vd.judge.eq("gpt52") & vd.query_id.isin(qs) & vd.satisfied_is_known
        & vd.pattern.isin(P0 + CLUSTER)].copy()
vd = vd.sort_values("criterion_index")
vd["half"] = vd.groupby(["pattern", "query_id", "dimension"], observed=True).cumcount() % 2  # 0=A,1=B

def half_score(rows):
    dim = rows.groupby("dimension", observed=True)["satisfied"].mean()
    wsum = sum(W[d] for d in dim.index)
    return float(sum(W[d] * dim[d] for d in dim.index) / wsum) if wsum else np.nan

hs = vd.groupby(["pattern", "query_id", "half"], observed=True).apply(half_score).rename("score").reset_index()
hsA = hs[hs.half == 0].set_index(["pattern", "query_id"]).score
hsB = hs[hs.half == 1].set_index(["pattern", "query_id"]).score

def hget(tbl, p, q):
    try:
        v = tbl.loc[(p, q)]
        return float(v) if pd.notna(v) else None
    except KeyError:
        return None

# per-query lists of (replicate index kept) -> half A / half B / full scores, same [:N] convention
kept_idx, halfA, halfB, full = {}, {}, {}, {}
for q in qs:
    a = [hget(hsA, p, q) for p in P0]
    b = [hget(hsB, p, q) for p in P0]
    fu = [patq(p, q) for p in P0]
    keep = [i for i in range(len(a)) if a[i] is not None and b[i] is not None and fu[i] is not None]
    kept_idx[q] = keep[:N]
    halfA[q] = [a[i] for i in kept_idx[q]]
    halfB[q] = [b[i] for i in kept_idx[q]]
    full[q]  = [fu[i] for i in kept_idx[q]]
qs2 = [q for q in qs if len(halfA[q]) >= 3]
N2 = min(len(halfA[q]) for q in qs2)

cluster_B_q = {q: float(np.mean([x for x in (hget(hsB, p, q) for p in CLUSTER) if x is not None])) for q in qs2}
cluster_B = float(np.mean(list(cluster_B_q.values())))
cluster_full_q = {q: float(np.mean([patq(p, q) for p in CLUSTER if patq(p, q) is not None])) for q in qs2}
cluster_full = float(np.mean(list(cluster_full_q.values())))

# --- drift guard: forward oracle-decoupled curve must reproduce the stored canonical ---
cn = json.load(open(CANONICAL))
dec_stored = cn["best_of_n"]["decoupled"]
assert round(cluster_B, 4) == dec_stored["cluster_mean_half_B"], \
    f"cluster_B drift {round(cluster_B,4)} vs {dec_stored['cluster_mean_half_B']}"
oracle_B = {}
for k in range(1, N2 + 1):
    dec = float(np.mean([halfB[q][int(np.argmax(halfA[q][:k]))] for q in qs2]))
    oracle_B[k] = dec
    assert round(dec, 4) == dec_stored["curve"][str(k)]["best_of_k_decoupled"], \
        f"forward oracle-decoupled curve drift at k={k}: {round(dec,4)} vs stored"
baseline_B = oracle_B[1]  # shared k=1 pick (base_p0), == 0.4324
# k at which the oracle-decoupled curve itself draws level with the cluster (paper's k≈7);
# derived with a small tolerance so it matches the stored gap=0.0004 crossing at k=7.
ORACLE_LEVEL_K = next((k for k in range(1, N2 + 1) if oracle_B[k] >= cluster_B - 1e-3), N2)

# ============================================================================
# 2. Report-text features (report text only; never touches the judge)
# ============================================================================
_STOP = set("the a an and or of to in for on with as is are be by from that this it its their "
            "these those at into than then also such can may which while both between".split())
_tok_re = re.compile(r"[a-z][a-z]+")

def read_report(p, q):
    fp = f"{EXP}/{p}/{q}.md"
    with open(fp, encoding="utf-8") as fh:
        return fh.read()

def tokenize(text):
    return [t for t in _tok_re.findall(text.lower()) if t not in _STOP and len(t) > 2]

def report_features(text):
    words = re.findall(r"\S+", text)
    nw = len(words)
    cites = re.findall(r"\[(\d+)\]", text)
    urls = re.findall(r"https?://\S+", text)
    headings = len(re.findall(r"(?m)^#{1,6}\s", text))
    # sane length band [600, 3000] words: 0 inside, negative (scaled) outside
    lo, hi = 600, 3000
    band = -(max(0, lo - nw) + max(0, nw - hi)) / 600.0
    return {"n_words": nw, "uniq_cites": len(set(cites)), "n_cites": len(cites),
            "uniq_urls": len(set(urls)), "n_headings": headings, "length_band": band}

# precompute token counters + features per (replicate, query)
counters, feats = {}, {}
for q in qs2:
    for i in kept_idx[q]:
        p = P0[i]
        txt = read_report(p, q)
        counters[(p, q)] = Counter(tokenize(txt))
        feats[(p, q)] = report_features(txt)

# ============================================================================
# 3. Selector 1 -- self-consistency medoid (max mean pairwise TF-IDF cosine)
# ============================================================================
def tfidf_vectors(docs):
    """docs: list of Counter. Realisable-at-k TF-IDF with smoothed IDF over just these docs."""
    n = len(docs)
    df = Counter()
    for c in docs:
        for t in c:
            df[t] += 1
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
    vecs = []
    for c in docs:
        tot = sum(c.values()) or 1
        v = {t: (cnt / tot) * idf[t] for t, cnt in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs

def cosine(u, v):
    small, big = (u, v) if len(u) <= len(v) else (v, u)
    return sum(val * big.get(t, 0.0) for t, val in small.items())

def selfconsistency_pick(q, k):
    idxs = kept_idx[q][:k]
    if k == 1:
        return idxs[0]
    docs = [counters[(P0[i], q)] for i in idxs]
    vecs = tfidf_vectors(docs)
    best_j, best_score = 0, -1.0
    for j in range(k):
        s = sum(cosine(vecs[j], vecs[m]) for m in range(k) if m != j) / (k - 1)
        if s > best_score + 1e-12:          # strict -> ties resolve to earliest replicate
            best_score, best_j = s, j
    return idxs[best_j]

# ============================================================================
# 4. Selector 2 -- held-out proxy scorer (report-derived quality proxy)
# ============================================================================
PROXY_FEATURES = ["uniq_cites", "uniq_urls", "n_headings", "length_band"]

def zsum_pick(q, k, feature_list):
    """Pick argmax of the equal-weight sum of within-query z-scores of feature_list."""
    idxs = kept_idx[q][:k]
    if k == 1:
        return idxs[0]
    scores = np.zeros(k)
    for f in feature_list:
        vals = np.array([feats[(P0[i], q)][f] for i in idxs], dtype=float)
        sd = vals.std()
        if sd > 1e-12:                        # drop zero-variance features for this query/k
            scores += (vals - vals.mean()) / sd
    j = int(np.argmax(scores))                # ties -> earliest replicate
    return idxs[j]

def single_pick(q, k, feature):
    idxs = kept_idx[q][:k]
    if k == 1:
        return idxs[0]
    vals = np.array([feats[(P0[i], q)][feature] for i in idxs], dtype=float)
    return idxs[int(np.argmax(vals))]

# ============================================================================
# 5. Curve builder: realised score of the picked report vs cluster + oracle
# ============================================================================
def build_curve(pick_fn, score_tbl_q, cluster_ref, cluster_ref_q, oracle_curve):
    """pick_fn(q,k)->replicate index; score via score_tbl_q[(replicate_name,q)];
    returns per-k dict incl seeded query-bootstrap CI on the per-query gap, oracle headroom
    fraction, cluster-gap-closed fraction; and the draws-level k (point + CI)."""
    curve, gaps_pt = {}, {}
    draws_level_point, draws_level_ci = None, None
    for k in range(1, N2 + 1):
        picks = {q: pick_fn(q, k) for q in qs2}
        realised_q = {q: float(score_tbl_q[(P0[picks[q]], q)]) for q in qs2}
        val = float(np.mean(list(realised_q.values())))
        gaps = np.array([realised_q[q] - cluster_ref_q[q] for q in qs2])  # realised - cluster
        rng = np.random.default_rng(SEED + k)
        boot = np.array([gaps[rng.integers(0, len(gaps), len(gaps))].mean() for _ in range(NBOOT)])
        ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
        gap_to_cluster = round(cluster_ref - val, 4)
        row = {"best_of_k": round(val, 4), "gap_to_cluster": gap_to_cluster,
               "gap_signed_realised_minus_cluster": round(float(gaps.mean()), 4), "gap_ci95": ci}
        if oracle_curve is not None:
            orc = oracle_curve[k]
            denom_o = orc - baseline_B
            denom_c = cluster_ref - baseline_B
            row["oracle_decoupled_B"] = round(orc, 4)
            row["frac_oracle_headroom"] = (round(float((val - baseline_B) / denom_o), 4)
                                           if abs(denom_o) > 1e-9 else None)
            row["frac_cluster_gap_closed"] = (round(float((val - baseline_B) / denom_c), 4)
                                              if abs(denom_c) > 1e-9 else None)
        curve[k] = row
        gaps_pt[k] = val
        if draws_level_point is None and val >= cluster_ref - 1e-9:
            draws_level_point = k
        if draws_level_ci is None and ci[0] <= 0.0 <= ci[1]:
            draws_level_ci = k
    return curve, draws_level_point, draws_level_ci

# score tables keyed (replicate_name, q)
scoreB_tbl = {(P0[i], q): halfB[q][kept_idx[q].index(i)] for q in qs2 for i in kept_idx[q]}
scoreFull_tbl = {(P0[i], q): full[q][kept_idx[q].index(i)] for q in qs2 for i in kept_idx[q]}

def make_block(pick_fn, note, ablations=None):
    cB, dlp_B, dlc_B = build_curve(pick_fn, scoreB_tbl, cluster_B, cluster_B_q, oracle_B)
    cF, dlp_F, dlc_F = build_curve(pick_fn, scoreFull_tbl, cluster_full, cluster_full_q, None)
    fracs = [cB[k]["frac_oracle_headroom"] for k in cB if cB[k]["frac_oracle_headroom"] is not None]
    block = {
        "judge": "gpt52", "n_queries": len(qs2), "n_samples": N2, "seed_bootstrap": SEED,
        "cluster_mean_half_B": round(cluster_B, 4), "cluster_mean_full": round(cluster_full, 4),
        "baseline_k1_half_B": round(baseline_B, 4),
        "curve_half_B": {str(k): v for k, v in cB.items()},
        "curve_full": {str(k): v for k, v in cF.items()},
        "draws_level_k_point_half_B": dlp_B, "draws_level_k_ci_overlap_half_B": dlc_B,
        "draws_level_k_point_full": dlp_F, "draws_level_k_ci_overlap_full": dlc_F,
        "oracle_draws_level_k": ORACLE_LEVEL_K,  # oracle-decoupled point-crossing (paper's k≈7)
        "frac_oracle_headroom_at_oracle_level_k": cB[ORACLE_LEVEL_K]["frac_oracle_headroom"],
        "frac_cluster_gap_closed_at_oracle_level_k": cB[ORACLE_LEVEL_K]["frac_cluster_gap_closed"],
        "frac_oracle_headroom_at_kmax": cB[N2]["frac_oracle_headroom"],
        "frac_cluster_gap_closed_at_kmax": cB[N2]["frac_cluster_gap_closed"],
        "max_frac_oracle_headroom": (round(max(fracs), 4) if fracs else None),
        "max_frac_oracle_headroom_note": ("frac_oracle_headroom is unstable at small k where the "
            "oracle's own headroom over baseline is ~0 (ratio blows up); use the anchored "
            "at_oracle_level_k (k=%d) and at_kmax (k=%d) fields for interpretation" % (ORACLE_LEVEL_K, N2)),
        "note": note}
    if ablations:
        block["ablations"] = {}
        for name, fn in ablations.items():
            aB, adlp, adlc = build_curve(fn, scoreB_tbl, cluster_B, cluster_B_q, oracle_B)
            block["ablations"][name] = {
                "curve_half_B_best_of_k": {str(k): aB[k]["best_of_k"] for k in aB},
                "gap_to_cluster_B": {str(k): aB[k]["gap_to_cluster"] for k in aB},
                "frac_oracle_headroom": {str(k): aB[k]["frac_oracle_headroom"] for k in aB},
                "draws_level_k_point_half_B": adlp, "draws_level_k_ci_overlap_half_B": adlc}
    return block

sc_block = make_block(
    selfconsistency_pick,
    ("realisable self-consistency: for each query & k, pick the medoid report (max mean pairwise "
     "TF-IDF cosine to the other k-1). Selection uses report text only, never the judge, so half-B "
     "and full-criteria scores are both unbiased (no selection-on-noise winner's curse). Curve is "
     "overlayable on best_of_n.decoupled.curve (oracle upper bound) on the identical half-B basis; "
     "frac_oracle_headroom = (realised-baseline)/(oracle_decoupled_B-baseline), baseline=k1 pick."))

hs_block = make_block(
    lambda q, k: zsum_pick(q, k, PROXY_FEATURES),
    ("realisable held-out proxy scorer: pick argmax of the equal-weight sum of within-query "
     "z-scores of {uniq_cites, uniq_urls, n_headings, length_band[600-3000w]} -- a pre-registered, "
     "judge-blind report-quality proxy (equal weights, no tuning against the judge). Picked report "
     "scored on the held-out half-B judge and on the full criteria, vs the orchestrated cluster on "
     "the same bases; frac_oracle_headroom relative to best_of_n.decoupled (oracle upper bound)."),
    ablations={
        "citations_only":  lambda q, k: single_pick(q, k, "uniq_cites"),
        "urls_only":       lambda q, k: single_pick(q, k, "uniq_urls"),
        "headings_only":   lambda q, k: single_pick(q, k, "n_headings"),
        "length_band_only": lambda q, k: single_pick(q, k, "length_band"),
        "longest":         lambda q, k: single_pick(q, k, "n_words")})

# ============================================================================
# 6. Write -- never overwrite; assert existing keys intact; atomic replace
# ============================================================================
bo = cn["best_of_n"]
snapshot = {"decoupled.cluster_mean_half_B": bo["decoupled"]["cluster_mean_half_B"],
            "decoupled.curve": json.dumps(bo["decoupled"]["curve"], sort_keys=True),
            "cluster_mean": bo["cluster_mean"], "best_of_N": bo["best_of_N"]}
for key in ["realisable_selfconsistency", "realisable_heldout_scorer"]:
    assert key not in bo, f"refusing to overwrite existing subkey best_of_n.{key}"
bo["realisable_selfconsistency"] = sc_block
bo["realisable_heldout_scorer"] = hs_block
# assert nothing else moved
assert bo["decoupled"]["cluster_mean_half_B"] == snapshot["decoupled.cluster_mean_half_B"]
assert json.dumps(bo["decoupled"]["curve"], sort_keys=True) == snapshot["decoupled.curve"]
assert bo["cluster_mean"] == snapshot["cluster_mean"] and bo["best_of_N"] == snapshot["best_of_N"]

tmp = CANONICAL + ".tmp"
with open(tmp, "w") as fh:
    fh.write(json.dumps(cn, indent=1))
os.replace(tmp, CANONICAL)

# ============================================================================
# 7. Console summary
# ============================================================================
def summarise(name, blk):
    cB = blk["curve_half_B"]
    print(f"\n=== {name} (half-B basis; baseline k1={blk['baseline_k1_half_B']}, "
          f"cluster_B={blk['cluster_mean_half_B']}) ===")
    print(f"{'k':>2} {'realis':>7} {'oracle':>7} {'gap_cl':>7} {'gap_ci95':>18} "
          f"{'%orcl':>7} {'%clsdG':>7}")
    for k in range(1, N2 + 1):
        r = cB[str(k)]
        print(f"{k:>2} {r['best_of_k']:>7.4f} {r['oracle_decoupled_B']:>7.4f} "
              f"{r['gap_to_cluster']:>7.4f} {str(r['gap_ci95']):>18} "
              f"{('' if r['frac_oracle_headroom'] is None else format(r['frac_oracle_headroom'],'.3f')):>7} "
              f"{('' if r['frac_cluster_gap_closed'] is None else format(r['frac_cluster_gap_closed'],'.3f')):>7}")
    print(f"  draws level w/ cluster (point) half-B: {blk['draws_level_k_point_half_B']}  "
          f"full: {blk['draws_level_k_point_full']}  | CI-overlap k half-B: "
          f"{blk['draws_level_k_ci_overlap_half_B']} (oracle & all curves overlap 0 -> low power, n=30)")
    print(f"  max frac oracle headroom realised: {blk['max_frac_oracle_headroom']}  "
          f"@k={N2}: {blk['frac_oracle_headroom_at_kmax']}")

print(json.dumps({"n_queries": len(qs2), "n_samples": N2, "cluster_B": round(cluster_B, 4),
                  "cluster_full": round(cluster_full, 4), "baseline_B": round(baseline_B, 4),
                  "oracle_B_curve": {k: round(v, 4) for k, v in oracle_B.items()},
                  "forward_oracle_matches_stored": True}, indent=1))
summarise("SELF-CONSISTENCY", sc_block)
summarise("HELD-OUT PROXY SCORER", hs_block)
print("\n--- held-out proxy ablations (half-B best_of_k @k=1..12, draws-level point k) ---")
for name, ab in hs_block["ablations"].items():
    vals = [ab["curve_half_B_best_of_k"][str(k)] for k in range(1, N2 + 1)]
    print(f"  {name:>16}: {[round(v,4) for v in vals]}  draws_level_k={ab['draws_level_k_point_half_B']}")
print("\nLanded keys: best_of_n.realisable_selfconsistency, best_of_n.realisable_heldout_scorer")
