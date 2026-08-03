"""
Phase 2 Hierarchical Statistical Analysis of 11 Deep-Research Patterns.

Gates:
  1. Omnibus on overall score (base patterns)
  2. Per-dimension omnibus (Holm corrected)
  3. Pairwise Wilcoxon + Cliff's delta
  4. Stratification by source / difficulty
  5. Ablations
  6. Bayesian signed-rank (stretch)
  7. Per-judge sensitivity
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "analysis"
OUT = ROOT / "reports" / "phase2_statistics"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "information_recall", "factual_accuracy", "coverage", "analytical_depth",
    "citation_quality", "logical_coherence", "organization",
    "instruction_following", "attribution_quality",
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df_queries = pd.read_parquet(DATA / "df_queries.parquet")
df_runs = pd.read_parquet(DATA / "df_runs.parquet")
df_overall = pd.read_parquet(DATA / "df_overall_scores.parquet")
df_scores = pd.read_parquet(DATA / "df_scores.parquet")

# Trustworthy overall
df_overall["overall"] = np.where(
    df_overall["overall_score_trustworthy"],
    df_overall["overall_score"],
    df_overall["overall_score_recomputed"],
)

# Drop rows with NaN overall (missing reports)
df_overall = df_overall.dropna(subset=["overall"])
df_scores = df_scores.dropna(subset=["score"])

# Exclude ablation_p5_no_citation_verify (only 2/90 reports)
EXCLUDE = ["ablation_p5_no_citation_verify"]
df_overall = df_overall[~df_overall["pattern"].isin(EXCLUDE)]
df_scores = df_scores[~df_scores["pattern"].isin(EXCLUDE)]

print(f"Loaded overall={len(df_overall)}, scores={len(df_scores)}")

# ---------------------------------------------------------------------------
# Aggregate across judges -> (pattern, query_id)
# ---------------------------------------------------------------------------
agg_overall = (
    df_overall.groupby(["pattern", "pattern_family", "query_id"], observed=True)["overall"]
    .mean()
    .reset_index()
)
agg_dim = (
    df_scores.groupby(["pattern", "pattern_family", "query_id", "dimension"], observed=True)["score"]
    .mean()
    .reset_index()
)

# Merge query metadata
agg_overall = agg_overall.merge(
    df_queries[["query_id", "source", "difficulty"]], on="query_id", how="left"
)
agg_dim = agg_dim.merge(
    df_queries[["query_id", "source", "difficulty"]], on="query_id", how="left"
)

BASE_PATTERNS = sorted([p for p in agg_overall["pattern"].unique() if p.startswith("base_")])
print(f"Base patterns ({len(BASE_PATTERNS)}): {BASE_PATTERNS}")

# Cast to plain strings to avoid unused-category issues with patsy/mixedlm
for _df in (agg_overall, agg_dim):
    for c in ("pattern", "pattern_family", "dimension", "source", "difficulty"):
        if c in _df.columns and str(_df[c].dtype) == "category":
            _df[c] = _df[c].astype(str)

agg_overall_base = agg_overall[agg_overall["pattern"].isin(BASE_PATTERNS)].copy()
agg_dim_base = agg_dim[agg_dim["pattern"].isin(BASE_PATTERNS)].copy()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))


def bootstrap_cliffs_delta_ci(x, y, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return (np.nan, np.nan)
    deltas = np.empty(n)
    nx, ny = len(x), len(y)
    for i in range(n):
        xi = rng.integers(0, nx, nx)
        yi = rng.integers(0, ny, ny)
        deltas[i] = cliffs_delta(x[xi], y[yi])
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        v = (m - i) * pvals[idx]
        v = min(v, 1.0)
        v = max(v, prev)
        adj[idx] = v
        prev = v
    return adj


def paired_wilcoxon(x, y):
    # x, y aligned
    d = np.asarray(x) - np.asarray(y)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.all(d == 0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def paired_align(df, pat_a, pat_b, value_col="overall"):
    a = df[df["pattern"] == pat_a].set_index("query_id")[value_col]
    b = df[df["pattern"] == pat_b].set_index("query_id")[value_col]
    common = a.index.intersection(b.index)
    return a.loc[common].values, b.loc[common].values, list(common)


# ---------------------------------------------------------------------------
# GATE 1 — Omnibus overall (base)
# ---------------------------------------------------------------------------
print("\n=== GATE 1: Omnibus overall ===")

def _fit_mixed(formula, df, group_col):
    m = smf.mixedlm(formula, df, groups=df[group_col])
    last_err = None
    for method in (["lbfgs"], ["bfgs"], ["powell"], ["nm"]):
        try:
            res = m.fit(reml=False, method=method)
            if np.isfinite(res.llf):
                return res
        except Exception as e:
            last_err = e
            continue
    try:
        res = m.fit(reml=False)
        if np.isfinite(res.llf):
            return res
    except Exception as e:
        last_err = e
    raise RuntimeError(f"mixedlm failed: {last_err}")


def mixed_lrt(df, formula_full, formula_null, group_col="query_id"):
    """Primary: mixedlm LRT. Fallback: OLS F-test if mixedlm unreliable."""
    try:
        m_full = _fit_mixed(formula_full, df, group_col)
        m_null = _fit_mixed(formula_null, df, group_col)
        if np.isfinite(m_full.llf) and np.isfinite(m_null.llf) and m_full.llf > m_null.llf - 1e-6:
            lr = 2 * (m_full.llf - m_null.llf)
            k = len(m_full.fe_params) - len(m_null.fe_params)
            p = 1 - stats.chi2.cdf(lr, df=max(k, 1))
            return m_full, m_null, lr, k, p
    except Exception:
        pass
    # Fallback: OLS with query_id fixed effect
    import statsmodels.api as sm
    try:
        # Use OLS with query_id as a fixed effect (absorbs within-query variation)
        full = smf.ols(formula_full + " + C(query_id)", df).fit()
        null = smf.ols(formula_null + " + C(query_id)", df).fit()
        lr_stat = (null.ssr - full.ssr) / full.ssr * full.df_resid
        k = int(null.df_resid - full.df_resid)
        p = 1 - stats.f.cdf(lr_stat / max(k, 1), dfn=max(k, 1), dfd=full.df_resid)
        return full, null, lr_stat, k, p
    except Exception as e:
        return None, None, np.nan, np.nan, np.nan

m_full, m_null, lr_overall, k_overall, p_overall = mixed_lrt(
    agg_overall_base, "overall ~ C(pattern)", "overall ~ 1"
)

# Variance components from REML fit
try:
    m_reml = smf.mixedlm("overall ~ C(pattern)", agg_overall_base, groups=agg_overall_base["query_id"]).fit(reml=True, method="lbfgs")
except Exception:
    m_reml = smf.mixedlm("overall ~ C(pattern)", agg_overall_base, groups=agg_overall_base["query_id"]).fit(reml=True)
var_query = float(m_reml.cov_re.iloc[0, 0])
var_resid = float(m_reml.scale)
icc_overall = var_query / (var_query + var_resid)

# Ranking (mean overall)
rank_overall = (
    agg_overall_base.groupby("pattern", observed=True)["overall"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
    .reset_index()
)
rank_overall.columns = ["pattern", "mean", "std", "n"]

with open(OUT / "01_omnibus_overall.md", "w") as f:
    f.write("# Gate 1 — Omnibus on Overall Score (base patterns)\n\n")
    f.write(f"Mixed-effects model: `overall ~ C(pattern) + (1|query_id)`\n\n")
    f.write(f"Patterns: {len(BASE_PATTERNS)}  |  Queries: {agg_overall_base['query_id'].nunique()}  |  Rows: {len(agg_overall_base)}\n\n")
    f.write("## Likelihood-ratio test (fixed effect of pattern)\n\n")
    f.write(f"- LR statistic: **{lr_overall:.3f}** on df={k_overall}\n")
    f.write(f"- p-value: **{p_overall:.4g}**\n")
    f.write(f"- Pass: **{'YES' if p_overall < 0.05 else 'NO'}**\n\n")
    f.write("## Variance components (REML)\n\n")
    f.write(f"- Query variance: {var_query:.5f}\n")
    f.write(f"- Residual variance: {var_resid:.5f}\n")
    f.write(f"- ICC(query): {icc_overall:.3f}\n\n")
    f.write("## Pattern ranking (mean ± std, n)\n\n")
    f.write("| pattern | mean | std | n |\n|---|---|---|---|\n")
    for _, r in rank_overall.iterrows():
        f.write(f"| {r['pattern']} | {r['mean']:.3f} | {r['std']:.3f} | {int(r['n'])} |\n")

print(f"Gate 1: LR={lr_overall:.2f}, p={p_overall:.4g} -> {'PASS' if p_overall<0.05 else 'FAIL'}")

# ---------------------------------------------------------------------------
# GATE 2 — Per-dimension omnibus
# ---------------------------------------------------------------------------
print("\n=== GATE 2: Per-dimension omnibus ===")

gate2_rows = []
for dim in DIMENSIONS:
    d = agg_dim_base[agg_dim_base["dimension"] == dim]
    if len(d) < 20:
        continue
    try:
        _, _, lr, k, p = mixed_lrt(d, "score ~ C(pattern)", "score ~ 1")
    except Exception as e:
        lr, k, p = np.nan, np.nan, np.nan
    gate2_rows.append({"dimension": dim, "lr_stat": lr, "df": k, "p_raw": p})

gate2_df = pd.DataFrame(gate2_rows)
gate2_df["p_holm"] = holm(gate2_df["p_raw"].fillna(1.0).values)
gate2_df["sig"] = gate2_df["p_holm"] < 0.05
gate2_df = gate2_df.sort_values("p_holm")

with open(OUT / "02_omnibus_per_dimension.md", "w") as f:
    f.write("# Gate 2 — Per-Dimension Omnibus (base patterns)\n\n")
    f.write("Mixed-effects LR test per dimension. Holm-corrected across 9 dimensions.\n\n")
    f.write("| dimension | LR | df | p_raw | p_holm | sig |\n|---|---|---|---|---|---|\n")
    for _, r in gate2_df.iterrows():
        f.write(f"| {r['dimension']} | {r['lr_stat']:.2f} | {int(r['df']) if not np.isnan(r['df']) else '-'} | "
                f"{r['p_raw']:.4g} | {r['p_holm']:.4g} | {'Yes' if r['sig'] else 'No'} |\n")

SIG_DIMS = gate2_df[gate2_df["sig"]]["dimension"].tolist()
print(f"Gate 2: {len(SIG_DIMS)}/9 dims significant after Holm: {SIG_DIMS}")

# ---------------------------------------------------------------------------
# GATE 3 — Pairwise (overall + significant dims)
# ---------------------------------------------------------------------------
print("\n=== GATE 3: Pairwise ===")

def pairwise_analysis(df, value_col, label):
    pats = sorted(df["pattern"].unique())
    rows = []
    for a, b in combinations(pats, 2):
        xa, xb, _ = paired_align(df, a, b, value_col=value_col)
        if len(xa) < 5:
            continue
        stat, p = paired_wilcoxon(xa, xb)
        delta = cliffs_delta(xa, xb)
        ci_lo, ci_hi = bootstrap_cliffs_delta_ci(xa, xb, n=1000)
        mean_a, mean_b = float(np.mean(xa)), float(np.mean(xb))
        rows.append({
            "a": a, "b": b, "mean_a": mean_a, "mean_b": mean_b,
            "mean_diff": mean_a - mean_b, "n_paired": len(xa),
            "wilcoxon_stat": stat, "p_raw": p,
            "cliffs_delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res["p_raw"].fillna(1.0).values)
        res["sig"] = res["p_holm"] < 0.05
    return res

pairwise_overall = pairwise_analysis(agg_overall_base, "overall", "overall")

with open(OUT / "03_pairwise_overall.md", "w") as f:
    f.write("# Gate 3 — Pairwise Comparisons (Overall, base)\n\n")
    f.write("Paired Wilcoxon signed-rank, Holm corrected across 55 pairs. Cliff's delta with 95% bootstrap CI.\n\n")
    f.write(f"Total pairs: {len(pairwise_overall)}  |  Significant (p_holm<0.05): {pairwise_overall['sig'].sum()}\n\n")
    sorted_pw = pairwise_overall.sort_values("p_holm")
    f.write("| a | b | mean_a | mean_b | diff | n | W | p_raw | p_holm | δ | δ_CI | sig |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in sorted_pw.iterrows():
        f.write(f"| {r['a']} | {r['b']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_diff']:+.3f} | "
                f"{int(r['n_paired'])} | {r['wilcoxon_stat']:.1f} | {r['p_raw']:.3g} | {r['p_holm']:.3g} | "
                f"{r['cliffs_delta']:+.3f} | [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] | {'Y' if r['sig'] else '.'} |\n")

# Heatmap of signed Cliff's delta
def save_heatmap(res, title, outpath):
    pats = sorted(set(res["a"]).union(res["b"]))
    n = len(pats)
    idx = {p: i for i, p in enumerate(pats)}
    mat = np.full((n, n), np.nan)
    sig_mat = np.zeros((n, n), dtype=bool)
    for _, r in res.iterrows():
        i, j = idx[r["a"]], idx[r["b"]]
        mat[i, j] = r["cliffs_delta"]
        mat[j, i] = -r["cliffs_delta"]
        sig_mat[i, j] = r["sig"]
        sig_mat[j, i] = r["sig"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=pats, yticklabels=pats, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Cliff's δ (row vs col)"})
    # Mark sig cells
    for i in range(n):
        for j in range(n):
            if sig_mat[i, j] and i != j:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close(fig)

save_heatmap(pairwise_overall, "Pairwise Cliff's δ — Overall (black box = p_holm<0.05)",
             FIG / "heatmap_overall.png")

# Per-significant-dim pairwise
pairwise_by_dim = {}
for dim in SIG_DIMS:
    d = agg_dim_base[agg_dim_base["dimension"] == dim]
    res = pairwise_analysis(d, "score", dim)
    pairwise_by_dim[dim] = res
    with open(OUT / f"03_pairwise_{dim}.md", "w") as f:
        f.write(f"# Gate 3 — Pairwise ({dim})\n\n")
        f.write(f"Significant pairs: {res['sig'].sum()}/{len(res)}\n\n")
        sr = res.sort_values("p_holm")
        f.write("| a | b | mean_a | mean_b | diff | W | p_holm | δ | sig |\n|---|---|---|---|---|---|---|---|---|\n")
        for _, r in sr.iterrows():
            f.write(f"| {r['a']} | {r['b']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                    f"{r['mean_diff']:+.3f} | {r['wilcoxon_stat']:.1f} | {r['p_holm']:.3g} | "
                    f"{r['cliffs_delta']:+.3f} | {'Y' if r['sig'] else '.'} |\n")
    save_heatmap(res, f"Pairwise Cliff's δ — {dim}", FIG / f"heatmap_{dim}.png")

# Ranking per dimension
ranking_tbl = (
    agg_dim_base.groupby(["dimension", "pattern"], observed=True)["score"].mean().reset_index()
)
ranking_wide = ranking_tbl.pivot(index="pattern", columns="dimension", values="score")
ranking_wide = ranking_wide.round(3)
ranking_wide.to_csv(FIG / "ranking_per_dimension.csv")

# Plot ranking heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(ranking_wide, annot=True, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Mean score per pattern × dimension (base)")
plt.tight_layout()
plt.savefig(FIG / "ranking_per_dimension.png", dpi=100)
plt.close(fig)

print(f"Gate 3 overall: {pairwise_overall['sig'].sum()} significant pairs")

# ---------------------------------------------------------------------------
# GATE 4 — Stratification
# ---------------------------------------------------------------------------
print("\n=== GATE 4: Stratification ===")

def interaction_lrt(df, interaction_factor):
    full = f"overall ~ C(pattern) * C({interaction_factor})"
    null = f"overall ~ C(pattern) + C({interaction_factor})"
    m_full = _fit_mixed(full, df, "query_id")
    m_null = _fit_mixed(null, df, "query_id")
    lr = 2 * (m_full.llf - m_null.llf)
    k = len(m_full.fe_params) - len(m_null.fe_params)
    p = 1 - stats.chi2.cdf(lr, df=max(k, 1))
    return lr, k, p

lr_src, k_src, p_src = interaction_lrt(agg_overall_base, "source")
lr_dif, k_dif, p_dif = interaction_lrt(agg_overall_base, "difficulty")

src_rank = (
    agg_overall_base.groupby(["source", "pattern"], observed=True)["overall"].mean()
    .reset_index().pivot(index="pattern", columns="source", values="overall").round(3)
)
dif_rank = (
    agg_overall_base.groupby(["difficulty", "pattern"], observed=True)["overall"].mean()
    .reset_index().pivot(index="pattern", columns="difficulty", values="overall").round(3)
)

with open(OUT / "04_stratification.md", "w") as f:
    f.write("# Gate 4 — Stratification\n\n")
    f.write("## Source interaction\n\n")
    f.write(f"LR pattern×source: **{lr_src:.2f}** df={k_src}, p=**{p_src:.4g}**\n\n")
    f.write(f"Interaction significant: {'YES' if p_src<0.05 else 'NO'}\n\n")
    f.write("### Mean overall per (pattern × source)\n\n")
    f.write(src_rank.to_markdown() + "\n\n")
    f.write("## Difficulty interaction\n\n")
    f.write(f"LR pattern×difficulty: **{lr_dif:.2f}** df={k_dif}, p=**{p_dif:.4g}**\n\n")
    f.write(f"Interaction significant: {'YES' if p_dif<0.05 else 'NO'}\n\n")
    f.write("### Mean overall per (pattern × difficulty)\n\n")
    f.write(dif_rank.to_markdown() + "\n\n")

print(f"Gate 4: source p={p_src:.4g}, difficulty p={p_dif:.4g}")

# ---------------------------------------------------------------------------
# GATE 5 — Ablations
# ---------------------------------------------------------------------------
print("\n=== GATE 5: Ablations ===")

ABLATION_PAIRS = [
    ("ablation_p3_no_quality_eval", "base_p3"),
    ("ablation_p3_no_topic_mining", "base_p3"),
    ("ablation_p4_fixed_perspectives", "base_p4"),
    ("ablation_p4_no_conversations", "base_p4"),
    ("ablation_p4_no_triangulation", "base_p4"),
    ("ablation_p5_fixed_width", "base_p5"),
    ("ablation_p5_no_meta_eval", "base_p5"),
]


def bootstrap_mean_ci(diffs, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots[i] = np.mean(diffs[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


abl_rows = []
for abl, base in ABLATION_PAIRS:
    xa, xb, _ = paired_align(agg_overall, abl, base, value_col="overall")
    if len(xa) < 5:
        continue
    diffs = xa - xb
    mean_diff = float(np.mean(diffs))
    ci_lo, ci_hi = bootstrap_mean_ci(diffs, n=1000)
    stat, p = paired_wilcoxon(xa, xb)
    delta = cliffs_delta(xa, xb)
    abl_rows.append({
        "ablation": abl, "base": base, "n": len(xa),
        "mean_abl": float(np.mean(xa)), "mean_base": float(np.mean(xb)),
        "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "wilcoxon": stat, "p_raw": p, "cliffs_delta": delta,
    })
abl_df = pd.DataFrame(abl_rows)
abl_df["p_holm"] = holm(abl_df["p_raw"].fillna(1.0).values)
abl_df["sig"] = abl_df["p_holm"] < 0.05

# Per-dimension Δ
abl_dim_rows = []
for abl, base in ABLATION_PAIRS:
    for dim in DIMENSIONS:
        d = agg_dim[agg_dim["dimension"] == dim]
        xa, xb, _ = paired_align(d, abl, base, value_col="score")
        if len(xa) < 5:
            continue
        stat, p = paired_wilcoxon(xa, xb)
        abl_dim_rows.append({
            "ablation": abl, "base": base, "dimension": dim,
            "n": len(xa), "mean_diff": float(np.mean(xa - xb)),
            "p_raw": p, "cliffs_delta": cliffs_delta(xa, xb),
        })
abl_dim_df = pd.DataFrame(abl_dim_rows)

with open(OUT / "05_ablations.md", "w") as f:
    f.write("# Gate 5 — Ablations\n\n")
    f.write("Paired Wilcoxon (ablation vs base). Holm across 7 ablations. Mean Δ with percentile bootstrap 95% CI.\n\n")
    f.write("## Overall score\n\n")
    f.write("| ablation | base | n | mean_abl | mean_base | Δ | 95% CI | δ | p_raw | p_holm | sig |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in abl_df.iterrows():
        f.write(f"| {r['ablation']} | {r['base']} | {int(r['n'])} | {r['mean_abl']:.3f} | "
                f"{r['mean_base']:.3f} | {r['mean_diff']:+.3f} | [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] | "
                f"{r['cliffs_delta']:+.3f} | {r['p_raw']:.3g} | {r['p_holm']:.3g} | "
                f"{'Y' if r['sig'] else '.'} |\n")
    f.write("\n## Per-dimension Δ (ablation − base)\n\n")
    wide = abl_dim_df.pivot(index="ablation", columns="dimension", values="mean_diff").round(3)
    f.write(wide.to_markdown() + "\n")

# Ablation forest plot
fig, ax = plt.subplots(figsize=(10, 6))
abl_plot = abl_df.sort_values("mean_diff")
y = np.arange(len(abl_plot))
ax.errorbar(abl_plot["mean_diff"], y,
            xerr=[abl_plot["mean_diff"] - abl_plot["ci_lo"], abl_plot["ci_hi"] - abl_plot["mean_diff"]],
            fmt="o", color="black", capsize=4)
ax.axvline(0, color="red", ls="--")
ax.set_yticks(y)
ax.set_yticklabels(abl_plot["ablation"])
ax.set_xlabel("Δ overall (ablation − base) with 95% bootstrap CI")
ax.set_title("Ablation forest plot")
plt.tight_layout()
plt.savefig(FIG / "ablation_forest.png", dpi=100)
plt.close(fig)

# Forest plot for overall pairwise (top-20 most significant)
fig, ax = plt.subplots(figsize=(10, 8))
top = pairwise_overall.assign(abs_d=pairwise_overall["cliffs_delta"].abs()).sort_values("abs_d", ascending=True).tail(20)
y = np.arange(len(top))
ax.errorbar(top["cliffs_delta"], y,
            xerr=[top["cliffs_delta"] - top["ci_lo"], top["ci_hi"] - top["cliffs_delta"]],
            fmt="o", color="steelblue", capsize=3)
ax.axvline(0, color="black", ls="--")
ax.set_yticks(y)
ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(top["a"], top["b"])])
ax.set_xlabel("Cliff's δ with 95% bootstrap CI")
ax.set_title("Top-20 pairwise effects (by |δ|) — overall")
plt.tight_layout()
plt.savefig(FIG / "pairwise_forest_overall.png", dpi=100)
plt.close(fig)

print(f"Gate 5: {abl_df['sig'].sum()}/{len(abl_df)} ablations significant")

# ---------------------------------------------------------------------------
# GATE 6 — Bayesian signed-rank (stretch)
# ---------------------------------------------------------------------------
print("\n=== GATE 6: Bayesian ===")

bayes_ok = True
try:
    import baycomp
except Exception as e:
    bayes_ok = False
    print("baycomp unavailable:", e)

bayes_rows = []
if bayes_ok:
    headlines = [
        ("base_p4", "base_p10", "P4 vs P10 (best pipeline vs RL-7B)"),
        ("base_p9", "base_p10", "P9 vs P10 (RL effect)"),
        ("base_p9", "base_p0", "P9 vs P0 (model scale)"),
    ]
    for a, b, label in headlines:
        xa, xb, _ = paired_align(agg_overall, a, b, value_col="overall")
        if len(xa) < 5:
            bayes_rows.append({"label": label, "a": a, "b": b, "n": len(xa),
                               "p_left": np.nan, "p_rope": np.nan, "p_right": np.nan,
                               "note": "insufficient paired data"})
            continue
        try:
            probs = baycomp.SignedRankTest(xa, xb, rope=0.02).probs()
            # baycomp: (P(a>b+rope), P(|a-b|<=rope), P(a<b-rope))
            p_a_wins, p_rope, p_b_wins = [float(v) for v in probs]
        except Exception as e:
            p_a_wins = p_rope = p_b_wins = np.nan
        bayes_rows.append({
            "label": label, "a": a, "b": b, "n": len(xa),
            "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
            "p_a_wins": p_a_wins, "p_rope": p_rope, "p_b_wins": p_b_wins,
        })

with open(OUT / "06_bayesian.md", "w") as f:
    f.write("# Gate 6 — Bayesian Signed-Rank (headline comparisons)\n\n")
    f.write("`baycomp.SignedRankTest` with ROPE=±0.02.\n\n")
    f.write("P(a>b) = posterior P(a beats b beyond ROPE). P(rope) = P(practical equivalence). "
            "P(b>a) = posterior P(b beats a beyond ROPE).\n\n")
    if not bayes_ok:
        f.write("`baycomp` unavailable — skipped.\n")
    else:
        f.write("| comparison | a | b | n | mean_a | mean_b | P(a>b) | P(rope) | P(b>a) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in bayes_rows:
            f.write(f"| {r['label']} | {r['a']} | {r['b']} | {r.get('n','-')} | "
                    f"{r.get('mean_a', float('nan')):.3f} | {r.get('mean_b', float('nan')):.3f} | "
                    f"{r['p_a_wins']:.3f} | {r['p_rope']:.3f} | {r['p_b_wins']:.3f} |\n")

print(f"Gate 6 done")

# ---------------------------------------------------------------------------
# GATE 7 — Per-judge sensitivity
# ---------------------------------------------------------------------------
print("\n=== GATE 7: Per-judge sensitivity ===")

judges = ["gpt52", "claude_opus", "claude_sonnet"]
per_judge_results = {}
for j in judges:
    dj = df_overall[(df_overall["judge"] == j) & (df_overall["pattern"].isin(BASE_PATTERNS))].copy()
    if len(dj) < 50:
        continue
    # Omnibus
    try:
        _, _, lr, k, p = mixed_lrt(dj, "overall ~ C(pattern)", "overall ~ 1")
    except Exception:
        lr, k, p = np.nan, np.nan, np.nan
    rank = dj.groupby("pattern", observed=True)["overall"].mean().sort_values(ascending=False)
    per_judge_results[j] = {"lr": lr, "p": p, "rank": rank}

# Compare rankings
rank_df = pd.DataFrame({j: v["rank"] for j, v in per_judge_results.items()})
rank_order_df = pd.DataFrame({j: v["rank"].rank(ascending=False).astype(int) for j, v in per_judge_results.items()})

# Spearman across judge rank orders
from scipy.stats import spearmanr
spear = {}
judges_have = list(per_judge_results.keys())
for a, b in combinations(judges_have, 2):
    r, _ = spearmanr(rank_df[a], rank_df[b])
    spear[f"{a} vs {b}"] = float(r)

with open(OUT / "07_per_judge_sensitivity.md", "w") as f:
    f.write("# Gate 7 — Per-Judge Sensitivity\n\n")
    f.write("## Omnibus per judge\n\n")
    f.write("| judge | LR | df | p |\n|---|---|---|---|\n")
    for j, v in per_judge_results.items():
        f.write(f"| {j} | {v['lr']:.2f} | - | {v['p']:.3g} |\n")
    f.write("\n## Mean overall score per (pattern × judge)\n\n")
    f.write(rank_df.round(3).to_markdown() + "\n\n")
    f.write("## Rank order per judge (1 = best)\n\n")
    f.write(rank_order_df.to_markdown() + "\n\n")
    f.write("## Spearman ρ between judge rankings\n\n")
    for k, v in spear.items():
        f.write(f"- {k}: {v:.3f}\n")

print("Gate 7 done, spearman:", spear)

# ---------------------------------------------------------------------------
# Summary + digest
# ---------------------------------------------------------------------------
top3 = pairwise_overall.assign(abs_d=pairwise_overall["cliffs_delta"].abs()).sort_values("abs_d", ascending=False).head(3)

digest = {
    "gate1": {
        "lr_stat": float(lr_overall), "df": int(k_overall), "p": float(p_overall),
        "pass": bool(p_overall < 0.05),
        "variance_query": var_query, "variance_resid": var_resid, "icc": icc_overall,
        "best_pattern": rank_overall.iloc[0]["pattern"],
        "best_mean": float(rank_overall.iloc[0]["mean"]),
    },
    "gate2": {
        "sig_dims": SIG_DIMS,
        "n_sig": len(SIG_DIMS),
    },
    "gate3": {
        "overall_sig_pairs": int(pairwise_overall["sig"].sum()),
        "overall_total_pairs": int(len(pairwise_overall)),
        "top3": [
            {"a": r["a"], "b": r["b"], "delta": float(r["cliffs_delta"]),
             "mean_diff": float(r["mean_diff"]), "p_holm": float(r["p_holm"])}
            for _, r in top3.iterrows()
        ],
    },
    "gate4": {
        "source_interaction_p": float(p_src),
        "difficulty_interaction_p": float(p_dif),
        "source_sig": bool(p_src < 0.05),
        "difficulty_sig": bool(p_dif < 0.05),
    },
    "gate5": {
        "ablations": [
            {"ablation": r["ablation"], "base": r["base"],
             "mean_diff": float(r["mean_diff"]), "ci": [float(r["ci_lo"]), float(r["ci_hi"])],
             "delta": float(r["cliffs_delta"]), "p_holm": float(r["p_holm"]),
             "sig": bool(r["sig"])}
            for _, r in abl_df.iterrows()
        ],
    },
    "gate6": bayes_rows if bayes_ok else None,
    "gate7": {
        "spearman_across_judges": spear,
        "per_judge_omnibus_p": {j: float(v["p"]) for j, v in per_judge_results.items()},
    },
}

with open(OUT / "digest.json", "w") as f:
    json.dump(digest, f, indent=2, default=str)

with open(OUT / "summary.md", "w") as f:
    f.write("# Phase 2 Statistical Analysis — Summary\n\n")
    f.write("## Headline numbers\n\n")
    f.write(f"- **Gate 1 (omnibus overall)**: LR={lr_overall:.2f}, p={p_overall:.3g} — {'PASS' if p_overall<0.05 else 'FAIL'}\n")
    f.write(f"  - Query variance={var_query:.4f}, residual={var_resid:.4f}, ICC={icc_overall:.3f}\n")
    f.write(f"  - Best pattern: **{rank_overall.iloc[0]['pattern']}** (mean={rank_overall.iloc[0]['mean']:.3f})\n")
    f.write(f"  - Worst pattern: **{rank_overall.iloc[-1]['pattern']}** (mean={rank_overall.iloc[-1]['mean']:.3f})\n")
    f.write(f"- **Gate 2 (per-dim omnibus, Holm)**: {len(SIG_DIMS)}/9 significant → {SIG_DIMS}\n")
    f.write(f"- **Gate 3 (pairwise overall)**: {int(pairwise_overall['sig'].sum())}/{len(pairwise_overall)} significant after Holm\n")
    f.write(f"- **Gate 4 (stratification)**: source p={p_src:.3g} ({'SIG' if p_src<0.05 else 'ns'}); difficulty p={p_dif:.3g} ({'SIG' if p_dif<0.05 else 'ns'})\n")
    f.write(f"- **Gate 5 (ablations)**: {int(abl_df['sig'].sum())}/{len(abl_df)} significant after Holm\n")
    if bayes_ok:
        for r in bayes_rows:
            if not np.isnan(r.get("p_a_wins", np.nan)):
                f.write(f"- **Gate 6** {r['label']}: P(a>b)={r['p_a_wins']:.2f}, P(rope)={r['p_rope']:.2f}, P(b>a)={r['p_b_wins']:.2f}\n")
    f.write("\n## Top-3 pairwise effects (overall, by |δ|)\n\n")
    f.write("| a | b | Δmean | δ | p_holm |\n|---|---|---|---|---|\n")
    for _, r in top3.iterrows():
        f.write(f"| {r['a']} | {r['b']} | {r['mean_diff']:+.3f} | {r['cliffs_delta']:+.3f} | {r['p_holm']:.3g} |\n")
    f.write("\n## Top ablation effects\n\n")
    f.write("| ablation | base | Δ | δ | p_holm |\n|---|---|---|---|---|\n")
    for _, r in abl_df.sort_values("mean_diff").iterrows():
        f.write(f"| {r['ablation']} | {r['base']} | {r['mean_diff']:+.3f} | {r['cliffs_delta']:+.3f} | {r['p_holm']:.3g} |\n")
    f.write(f"\n## Per-judge ranking agreement (Spearman ρ)\n\n")
    for k, v in spear.items():
        f.write(f"- {k}: {v:.3f}\n")
    f.write("\n## Files\n\n")
    for p in sorted(OUT.glob("*.md")):
        f.write(f"- `{p.name}`\n")

print("\n=== DONE ===")
print(f"Outputs -> {OUT}")
