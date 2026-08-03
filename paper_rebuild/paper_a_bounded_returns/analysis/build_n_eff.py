#!/usr/bin/env python
"""E3 N-EFF — effective judge count and family-level dependence (Phase 1).

Pre-registration: docs/publication/prereg/prereg_E3.md (registered 2026-06-11). Canonical key `n_eff`
(the existing `irr` key holds pairwise correlations on OVERALL scores; this computes the
effective judge count at CRITERION-VERDICT level and the within- vs cross-family structure).

Substrate (verified on disk): the fully-crossed cell = (pattern x query x criterion_id)
verdicts scored by ALL THREE panel judges (gpt52, claude_opus, claude_sonnet). These align
across judges only on the shared 'general' criteria (judges paraphrase benchmark criteria, so
their ids diverge). Measured: 36,113 crossed cells -> 108,341 verdicts, 106 distinct criteria,
all 9 dimensions. (The plan's cited n=44,425 differs by definition; this script reports its own
disk-computed n.) claude_code excluded: its overlap with opus is n=0 (a labelled secondary
analysis only).

N_eff method: effective number of independent unit-variance measurements when averaging N
correlated judges, N_eff = N^2 / (1' R 1) = N^2 / (N + 2 * sum_{i<j} rho_ij), with rho the phi
coefficient (Pearson on the 0/1 verdicts). Reported overall and per dimension, with the
within-family (opus-sonnet) vs cross-family (gpt52-Claude) agreement that drives it. CARE
(2603.00039) / Ising (2601.22336) dependence-aware aggregation are the deferred refinement;
this Phase-1 number is the standard correlation-based effective count, stated as such.

Determinism: no randomness (closed-form); inputs sorted for reproducibility.
"""
import json, os, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
PANEL = ["gpt52", "claude_opus", "claude_sonnet"]
FAM = {"gpt52": "openai", "claude_opus": "anthropic", "claude_sonnet": "anthropic"}

V = pd.read_parquet(f"{ROOT}/data/analysis/df_verdicts.parquet")
b = V[(V.pattern_family == "base") & (V.judge.isin(PANEL)) & (V.satisfied_is_known)].copy()

# fully-crossed cells (all 3 judges on the same pattern x query x criterion_id)
key = ["pattern", "query_id", "criterion_id"]
cnt = b.groupby(key, observed=True)["judge"].nunique()
crossed_keys = set(cnt[cnt == 3].index)
b["k"] = list(zip(b.pattern, b.query_id, b.criterion_id))
fc = b[b.k.isin(crossed_keys)].copy()

def wide(df):
    w = (df.pivot_table(index=key, columns="judge", values="satisfied", aggfunc="first",
                        observed=True).dropna(subset=PANEL))
    for j in PANEL:
        w[j] = w[j].astype(int)
    return w.sort_index()

def phi(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def kappa(x, y):
    po = float((x == y).mean())
    px1, py1 = x.mean(), y.mean()
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else float("nan")

def ac1(x, y):
    # Gwet's AC1: prevalence-robust chance correction. pe = 2*q*(1-q) with
    # q the pooled positive rate; insensitive to the marginal-prevalence
    # attenuation that suppresses phi when judges have very different positive rates.
    x = np.asarray(x, float); y = np.asarray(y, float)
    po = float((x == y).mean())
    q = float((x.mean() + y.mean()) / 2.0)
    pe = 2.0 * q * (1.0 - q)
    return float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 0 else float("nan")

def holm(pv):
    # reused from build_pairwise.py:12 (step-down Holm-Bonferroni)
    pv = np.asarray(pv, float); idx = np.argsort(pv); m = len(pv)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * pv[i]); adj[i] = min(run, 1.0)
    return adj

def n_eff_from_rhos(rhos):
    # N=3, N_eff = 9 / (3 + 2*sum rho)
    s = sum(r for r in rhos if np.isfinite(r))
    denom = 3 + 2 * s
    return float(9.0 / denom) if denom > 0 else float("nan")

def _within_cross_excess(w, fn):
    """within (opus-sonnet) and mean cross (gpt52-Claude) agreement under metric `fn`,
    plus the within-minus-cross excess. Works for any pairwise agreement statistic."""
    within = fn(w["claude_opus"], w["claude_sonnet"])
    cross = np.nanmean([fn(w["gpt52"], w["claude_opus"]),
                        fn(w["gpt52"], w["claude_sonnet"])])
    return float(within), float(cross), float(within - cross)

def analyse(df, label):
    w = wide(df)
    pairs = list(itertools.combinations(PANEL, 2))
    rho = {f"{a}|{b_}": round(phi(w[a], w[b_]), 4) for a, b_ in pairs}
    kap = {f"{a}|{b_}": round(kappa(w[a], w[b_]), 4) for a, b_ in pairs}
    ac = {f"{a}|{b_}": round(ac1(w[a], w[b_]), 4) for a, b_ in pairs}
    within, cross, excess = _within_cross_excess(w, phi)              # phi (attenuated)
    a_within, a_cross, a_excess = _within_cross_excess(w, ac1)        # AC1 (prevalence-robust)
    neff = n_eff_from_rhos([phi(w[a], w[b_]) for a, b_ in pairs])
    return {"n_cells": int(len(w)), "phi": rho, "kappa": kap, "ac1": ac,
            "within_family_phi_opus_sonnet": round(within, 4),
            "cross_family_phi_gpt52_claude": round(cross, 4),
            "within_minus_cross": round(excess, 4),
            "within_family_ac1_opus_sonnet": round(a_within, 4),
            "cross_family_ac1_gpt52_claude": round(a_cross, 4),
            "within_minus_cross_ac1": round(a_excess, 4),
            "n_eff": round(neff, 4)}

overall = analyse(fc, "overall")
per_dim = {}
for dim, g in fc.groupby("dimension", observed=True):
    if g.k.nunique() >= 30:
        per_dim[str(dim)] = analyse(g, dim)

# --- per-dimension excess significance: paired query-clustered bootstrap + Holm ---
# Cluster = query_id (verdicts within a query are dependent). We resample query_ids
# WITH replacement (the same resampled query-block per replicate -> paired across
# metrics), recompute the within-minus-cross excess on the resampled wide table, and
# read a two-sided p from the bootstrap distribution crossing zero. Done for BOTH phi
# (attenuated) and AC1 (prevalence-robust), then Holm-correct across dimensions.
N_BOOT = 2000
SEED = 20260611  # prereg seed (E3)

def _excess_boot_pvals(per_dim_groups):
    rng = np.random.default_rng(SEED)
    dims = sorted(per_dim_groups)
    wides = {d: wide(per_dim_groups[d]) for d in dims}
    # union of query_ids across dims; resample once, apply to every dim (paired)
    qids_by_dim = {d: w.index.get_level_values("query_id").to_numpy() for d, w in wides.items()}
    all_q = np.array(sorted(set(np.concatenate([np.unique(q) for q in qids_by_dim.values()]))))
    boot = {d: {"phi": np.empty(N_BOOT), "ac1": np.empty(N_BOOT)} for d in dims}
    for it in range(N_BOOT):
        chosen = all_q[rng.integers(0, len(all_q), len(all_q))]
        chosen_set = set(chosen.tolist())
        for d in dims:
            w = wides[d]
            sel = w[np.isin(w.index.get_level_values("query_id"), list(chosen_set))]
            if len(sel) < 4 or sel["claude_opus"].nunique() < 2:
                boot[d]["phi"][it] = np.nan; boot[d]["ac1"][it] = np.nan; continue
            boot[d]["phi"][it] = _within_cross_excess(sel, phi)[2]
            boot[d]["ac1"][it] = _within_cross_excess(sel, ac1)[2]
    pvals = {}
    for d in dims:
        for metric in ("phi", "ac1"):
            bd = boot[d][metric]; bd = bd[np.isfinite(bd)]
            if len(bd) == 0:
                pvals[(d, metric)] = float("nan"); continue
            # two-sided: 2 * smaller tail mass at 0 (paired bootstrap on the excess)
            frac_le0 = float((bd <= 0).mean()); frac_ge0 = float((bd >= 0).mean())
            pvals[(d, metric)] = float(min(1.0, 2.0 * min(frac_le0, frac_ge0)))
    return pvals

# --- overall N_eff: query-clustered bootstrap CI (adversarial review 2026-07-28, round 25:
# the headline N_eff=1.65 was reported as a bare point estimate everywhere it is quoted, while
# its own corroborating Condorcet cross-check carries a 95% CI; this closes that asymmetry with
# the same resampling scheme used above for the per-dimension excess). ---
def _overall_neff_ci(df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    w = wide(df)
    all_q = w.index.get_level_values("query_id").unique().to_numpy()
    pairs = list(itertools.combinations(PANEL, 2))
    boot = np.empty(n_boot)
    for it in range(n_boot):
        chosen = all_q[rng.integers(0, len(all_q), len(all_q))]
        sel = w[np.isin(w.index.get_level_values("query_id"), chosen)]
        if len(sel) < 4 or any(sel[p].nunique() < 2 for p in PANEL):
            boot[it] = np.nan; continue
        boot[it] = n_eff_from_rhos([phi(sel[a], sel[b_]) for a, b_ in pairs])
    boot = boot[np.isfinite(boot)]
    return [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]

overall["n_eff_ci95"] = _overall_neff_ci(fc)

_excess_pvals = _excess_boot_pvals({d: g for d, g in fc.groupby("dimension", observed=True)
                                    if g.k.nunique() >= 30})
_dims = sorted(per_dim)
for metric in ("phi", "ac1"):
    raw = np.array([_excess_pvals[(d, metric)] for d in _dims])
    # Holm across dimensions (NaNs -> p=1 so they never reject)
    adj = holm(np.where(np.isfinite(raw), raw, 1.0))
    for i, d in enumerate(_dims):
        per_dim[d][f"excess_{metric}_p_raw"] = round(float(raw[i]), 4)
        per_dim[d][f"excess_{metric}_p_holm"] = round(float(adj[i]), 4)
        per_dim[d][f"excess_{metric}_sig_holm"] = bool(adj[i] < 0.05)

# which dimension shows the largest within-Claude excess agreement (artefact signature)?
# Ranked on phi for back-compat; AC1 excess + Holm-adjusted p carried alongside.
art = sorted(((d, v["within_minus_cross"]) for d, v in per_dim.items()),
             key=lambda x: -x[1])
art_ac1 = sorted(((d, v["within_minus_cross_ac1"]) for d, v in per_dim.items()),
                 key=lambda x: -x[1])

out = {
    "_note": "E3 effective judge count at criterion-verdict level on the fully-crossed 3-panel "
             "cell. N_eff = 9/(3+2*sum phi). Within-family (opus-sonnet) vs cross-family "
             "(gpt52-Claude) agreement per dimension. CARE/Ising aggregation deferred. "
             "Prereg: prereg_E3.md.",
    "prereg": "docs/publication/prereg/prereg_E3.md",
    "n_crossed_cells": int(len(crossed_keys)),
    "n_crossed_verdicts": int(len(fc)),
    "n_distinct_criteria": int(fc.criterion_id.nunique()),
    "overall": overall,
    "per_dimension": per_dim,
    "artefact_signature_ranking": [
        {"dimension": d, "within_minus_cross": v,
         "within_minus_cross_ac1": per_dim[d]["within_minus_cross_ac1"],
         "excess_phi_p_holm": per_dim[d]["excess_phi_p_holm"],
         "excess_ac1_p_holm": per_dim[d]["excess_ac1_p_holm"],
         "excess_ac1_sig_holm": per_dim[d]["excess_ac1_sig_holm"]}
        for d, v in art],
    "artefact_signature_ranking_ac1": [
        {"dimension": d, "within_minus_cross_ac1": v,
         "excess_ac1_p_holm": per_dim[d]["excess_ac1_p_holm"],
         "excess_ac1_sig_holm": per_dim[d]["excess_ac1_sig_holm"]}
        for d, v in art_ac1],
    "robustness_note_ac1": (
        "phi (Pearson on 0/1 verdicts) is prevalence-attenuated when judges have very "
        "different positive rates; Gwet's AC1 is reported ALONGSIDE as a prevalence-robust "
        "companion for the excess and the per-dimension ranking. Per-dimension excess tests "
        "are paired query-clustered bootstraps (seed=%d, N=%d) Holm-corrected across the %d "
        "dimensions, for both phi and AC1. The overall within-minus-cross excess is "
        "phi=%+.3f / AC1=%+.3f; the artefact signature holds under the prevalence-robust "
        "statistic iff the AC1 ranking + Holm-significant dimensions agree (see "
        "artefact_signature_ranking_ac1)." % (
            SEED, N_BOOT, len(_dims),
            overall["within_minus_cross"], overall["within_minus_cross_ac1"])),
    "ranking_note": (
        f"Top within-minus-cross dimensions: {art[0][0]} ({art[0][1]}) ~ {art[1][0]} ({art[1][1]}) "
        f"at the top (a near-tie), then {art[2][0]} ({art[2][1]}). citation_quality is genuinely "
        "ELEVATED and positive but ranks THIRD, not second/first; a query-clustered bootstrap "
        "(adversarial review 2026-06-11) places it significantly below both dimensions above it. "
        "So headline (iii)'s citation-specific prediction is only PARTIALLY borne out: the two "
        "Claudes are redundant (N_eff<3) and citation IS one of the elevated dimensions, but it is "
        "not THE signature; instruction_following and logical_coherence lead."),
    "interpretation": "N_eff < 3 => the two Claude judges are partially redundant. The dimensions "
                      "with the largest within_minus_cross (instruction_following, logical_coherence) "
                      "are where the two-Claude correlation most exceeds cross-family; citation_quality "
                      "is elevated (third) but not the leading signature (see ranking_note).",
}

DRY_RUN = ("--dry-run" in __import__("sys").argv) or ("--write" not in __import__("sys").argv)
if DRY_RUN:
    print("[dry-run] computed n_eff; NOT writing canonical_numbers.json (pass --write to land).")
else:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    cn["n_eff"] = out
    _tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(_tmp, "w").write(json.dumps(cn, indent=1)); os.replace(_tmp, f"{ANA}/canonical_numbers.json")

print(f"n_eff: crossed_cells={len(crossed_keys)} verdicts={len(fc)} criteria={fc.criterion_id.nunique()}")
print(f"  OVERALL N_eff={overall['n_eff']} (95% CI {overall['n_eff_ci95']})  "
      f"within(opus,sonnet)={overall['within_family_phi_opus_sonnet']} "
      f"cross(gpt52,Claude)={overall['cross_family_phi_gpt52_claude']}")
print(f"  artefact-signature top-3 (within-cross): " +
      ", ".join(f"{d}={v}" for d, v in art[:3]))
print(f"  citation_quality ranks #{[d for d,_ in art].index('citation_quality')+1} "
      f"(elevated +{dict(art)['citation_quality']} but NOT the leading signature)")
print(f"  OVERALL excess: phi={overall['within_minus_cross']:+.4f}  "
      f"AC1(prevalence-robust)={overall['within_minus_cross_ac1']:+.4f}")
print(f"  artefact-signature top-3 by AC1: " +
      ", ".join(f"{d}={v}" for d, v in art_ac1[:3]))
print("  per-dim excess Holm (AC1): " + ", ".join(
    f"{d}={per_dim[d]['within_minus_cross_ac1']:+.3f}"
    f"(p_holm={per_dim[d]['excess_ac1_p_holm']},sig={per_dim[d]['excess_ac1_sig_holm']})"
    for d, _ in art[:5]))
