"""
Phase 1: LLM-Judge Validation.

Computes inter-rater reliability, bias audits, ranking concordance, and
aggregation decision for the 3-judge panel (gpt52, claude_opus, claude_sonnet).

Outputs to reports/phase1_judge_validation/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from irrCAC.raw import CAC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "analysis"
OUT = ROOT / "reports" / "phase1_judge_validation"
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

JUDGES = ["gpt52", "claude_opus", "claude_sonnet"]
DIMENSIONS = [
    "information_recall", "factual_accuracy", "coverage",
    "analytical_depth", "citation_quality", "logical_coherence",
    "organization", "instruction_following", "attribution_quality",
]
RNG = np.random.default_rng(42)
N_BOOT = 2000

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
print("Loading parquet files...")
df_queries = pd.read_parquet(DATA / "df_queries.parquet")
df_runs = pd.read_parquet(DATA / "df_runs.parquet")
df_scores = pd.read_parquet(DATA / "df_scores.parquet")
df_overall = pd.read_parquet(DATA / "df_overall_scores.parquet")
df_verdicts = pd.read_parquet(DATA / "df_verdicts.parquet")

# Cast category -> str to avoid merge/boolean headaches
for df in (df_runs, df_scores, df_overall, df_verdicts):
    for c in df.select_dtypes("category"):
        df[c] = df[c].astype(str)

# Trustworthy overall score: use recomputed when flag is False
df_overall["overall"] = np.where(
    df_overall["overall_score_trustworthy"],
    df_overall["overall_score"],
    df_overall["overall_score_recomputed"],
)

# Restrict to base patterns for IRR / bias primary analysis
base_scores = df_scores[df_scores["pattern_family"] == "base"].copy()
base_overall = df_overall[df_overall["pattern_family"] == "base"].copy()
base_verdicts = df_verdicts[
    (df_verdicts["pattern_family"] == "base")
    & (df_verdicts["satisfied_is_known"])
].copy()

print(f"base_scores: {len(base_scores)}  base_overall: {len(base_overall)}  base_verdicts: {len(base_verdicts)}")

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def boot_ci(stat_fn, data, n=N_BOOT, alpha=0.05, cluster_col=None):
    """Bootstrap CI for stat_fn(data).

    If cluster_col is given (e.g. 'query_id'), uses cluster bootstrap: resamples
    unique cluster IDs with replacement, then takes all rows in those clusters.
    This respects nested data structure (same query × multiple patterns).
    """
    if len(data) < 2:
        return (np.nan, np.nan)
    vals = []
    if cluster_col is not None:
        # cluster bootstrap on cluster_col
        if cluster_col in data.index.names:
            groups_arr = data.index.get_level_values(cluster_col).to_numpy()
        else:
            groups_arr = data[cluster_col].to_numpy()
        clusters = np.unique(groups_arr)
        n_clusters = len(clusters)
        for _ in range(n):
            chosen = RNG.choice(clusters, n_clusters, replace=True)
            mask = np.isin(groups_arr, chosen)
            try:
                v = stat_fn(data.loc[mask] if isinstance(mask, np.ndarray) and mask.dtype == bool else data.iloc[mask])
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                continue
    else:
        n_rows = len(data)
        for _ in range(n):
            idx = RNG.integers(0, n_rows, n_rows)
            try:
                v = stat_fn(data.iloc[idx])
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                continue
    if len(vals) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def dim_score_matrix(dim):
    """Return judges x units matrix of scores for a single dimension (base only)."""
    sub = base_scores[base_scores["dimension"] == dim]
    wide = sub.pivot_table(
        index=["pattern", "query_id"], columns="judge", values="score", aggfunc="mean"
    )
    wide = wide.dropna(subset=JUDGES)
    return wide  # index: (pattern, query_id), cols: judges


def krip_alpha(wide):
    # reliability data: rows = coders (judges), cols = units
    arr = wide[JUDGES].T.to_numpy(dtype=float)
    return krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")


def icc_from_wide(wide, kind="ICC(A,k)"):
    """Pingouin ICC for raters x units. kind: 'ICC(A,1)' (single-judge) or 'ICC(A,k)' (averaged panel).
    Returns (point, (lo, hi))."""
    long = wide.reset_index().melt(
        id_vars=["pattern", "query_id"], value_vars=JUDGES,
        var_name="judge", value_name="score",
    )
    long["unit"] = long["pattern"].astype(str) + "|" + long["query_id"].astype(str)
    try:
        res = pg.intraclass_corr(data=long, targets="unit", raters="judge",
                                 ratings="score", nan_policy="omit")
        row = res[res["Type"] == kind].iloc[0]
        ci = row["CI95"]
        return float(row["ICC"]), (float(ci[0]), float(ci[1]))
    except Exception as e:
        return (np.nan, (np.nan, np.nan))


def icc2k_from_wide(wide):
    """Backward-compat wrapper returning ICC(A,k)."""
    return icc_from_wide(wide, "ICC(A,k)")


# --------------------------------------------------------------------------
# 1. Continuous IRR per dimension
# --------------------------------------------------------------------------
print("\n[1] Continuous IRR per dimension (cluster bootstrap on query_id, N=2000)...")
rows_cont = []
for dim in DIMENSIONS:
    wide = dim_score_matrix(dim)
    alpha_point = krip_alpha(wide)
    # cluster bootstrap on query_id (handles same-query × multiple-pattern non-independence)
    lo, hi = boot_ci(lambda d: krip_alpha(d), wide, cluster_col="query_id")
    icc_a1_point, icc_a1_ci = icc_from_wide(wide, "ICC(A,1)")
    icc_ak_point, icc_ak_ci = icc_from_wide(wide, "ICC(A,k)")
    rows_cont.append({
        "dimension": dim,
        "n_units": len(wide),
        "alpha": alpha_point, "alpha_lo": lo, "alpha_hi": hi,
        "ICC_A1": icc_a1_point, "ICC_A1_lo": icc_a1_ci[0], "ICC_A1_hi": icc_a1_ci[1],
        "ICC_Ak": icc_ak_point, "ICC_Ak_lo": icc_ak_ci[0], "ICC_Ak_hi": icc_ak_ci[1],
    })
df_cont = pd.DataFrame(rows_cont)
df_cont.to_csv(OUT / "01_irr_continuous.csv", index=False)
print(df_cont.to_string(index=False))

# Also for overall
wide_overall = base_overall.pivot_table(
    index=["pattern", "query_id"], columns="judge", values="overall", aggfunc="mean"
).dropna(subset=JUDGES)
overall_alpha = krip_alpha(wide_overall)
overall_alpha_ci = boot_ci(lambda d: krip_alpha(d), wide_overall, cluster_col="query_id")
overall_icc_a1, overall_icc_a1_ci = icc_from_wide(wide_overall, "ICC(A,1)")
overall_icc_ak, overall_icc_ak_ci = icc_from_wide(wide_overall, "ICC(A,k)")
print(f"OVERALL alpha={overall_alpha:.3f} ({overall_alpha_ci[0]:.3f},{overall_alpha_ci[1]:.3f}) "
      f"ICC(A,1)={overall_icc_a1:.3f} ({overall_icc_a1_ci[0]:.3f},{overall_icc_a1_ci[1]:.3f}) "
      f"ICC(A,k)={overall_icc_ak:.3f} ({overall_icc_ak_ci[0]:.3f},{overall_icc_ak_ci[1]:.3f})")

with open(OUT / "01_irr_continuous.md", "w") as f:
    f.write("# IRR (Continuous) — Base Patterns (11 patterns × 90 queries, 3 judges)\n\n")
    f.write("Krippendorff's α (interval), ICC(A,1) single-judge absolute agreement, ICC(A,k) averaged-panel absolute agreement.\n")
    f.write(f"Cluster bootstrap on query_id, N={N_BOOT} resamples, for α 95% CI. Pingouin analytic CI for ICC.\n")
    f.write("Judges: gpt52, claude_opus, claude_sonnet. Use ICC(A,1) when reporting reliability of a single judge; ICC(A,k) when reporting the averaged 3-judge composite.\n\n")
    f.write("| Dimension | N | α | α 95% CI | ICC(A,1) | ICC(A,1) CI | ICC(A,k) | ICC(A,k) CI |\n")
    f.write("|---|---:|---:|:---:|---:|:---:|---:|:---:|\n")
    for _, r in df_cont.iterrows():
        f.write(f"| {r['dimension']} | {r['n_units']} | {r['alpha']:.3f} "
                f"| ({r['alpha_lo']:.3f}, {r['alpha_hi']:.3f}) "
                f"| {r['ICC_A1']:.3f} | ({r['ICC_A1_lo']:.3f}, {r['ICC_A1_hi']:.3f}) "
                f"| {r['ICC_Ak']:.3f} | ({r['ICC_Ak_lo']:.3f}, {r['ICC_Ak_hi']:.3f}) |\n")
    f.write(f"| **overall** | {len(wide_overall)} | {overall_alpha:.3f} "
            f"| ({overall_alpha_ci[0]:.3f}, {overall_alpha_ci[1]:.3f}) "
            f"| {overall_icc_a1:.3f} | ({overall_icc_a1_ci[0]:.3f}, {overall_icc_a1_ci[1]:.3f}) "
            f"| {overall_icc_ak:.3f} | ({overall_icc_ak_ci[0]:.3f}, {overall_icc_ak_ci[1]:.3f}) |\n")
    f.write("\nInterpretation benchmarks (Krippendorff, 2004): α≥0.80 strong; 0.67≤α<0.80 tentative; α<0.67 weak.\n")
    f.write("Koo & Li 2016 ICC thresholds: <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, ≥0.9 excellent.\n")

# --------------------------------------------------------------------------
# 2. Binary IRR per dimension (gpt52 vs claude_sonnet only)
# --------------------------------------------------------------------------
print("\n[2] Binary IRR (criterion-level) gpt52 vs claude_sonnet...")
# For claude_opus the criterion_ids are unpinned -> skip
v_gpt = base_verdicts[base_verdicts["judge"] == "gpt52"][
    ["pattern", "query_id", "dimension", "criterion_id", "satisfied"]
].rename(columns={"satisfied": "sat_gpt"})
v_son = base_verdicts[base_verdicts["judge"] == "claude_sonnet"][
    ["pattern", "query_id", "dimension", "criterion_id", "satisfied"]
].rename(columns={"satisfied": "sat_son"})

pair = v_gpt.merge(v_son, on=["pattern", "query_id", "dimension", "criterion_id"], how="inner")
print(f"Paired criterion verdicts: {len(pair)} (of {len(v_gpt)} gpt52, {len(v_son)} sonnet)")

rows_bin = []
for dim in DIMENSIONS:
    sub = pair[pair["dimension"] == dim]
    if len(sub) < 2:
        rows_bin.append({"dimension": dim, "n_pairs": len(sub), "kappa": np.nan, "ac1": np.nan,
                         "prev_gpt": np.nan, "prev_son": np.nan, "agree_rate": np.nan})
        continue
    a = sub["sat_gpt"].astype(int).to_numpy()
    b = sub["sat_son"].astype(int).to_numpy()
    kappa = cohen_kappa_score(a, b)
    try:
        cac = CAC(pd.DataFrame({"r1": a, "r2": b}))
        ac1 = cac.gwet()["est"]["coefficient_value"]
    except Exception:
        ac1 = np.nan
    rows_bin.append({
        "dimension": dim, "n_pairs": len(sub),
        "kappa": float(kappa), "ac1": float(ac1),
        "prev_gpt": float(a.mean()), "prev_son": float(b.mean()),
        "agree_rate": float((a == b).mean()),
    })
df_bin = pd.DataFrame(rows_bin)
df_bin.to_csv(OUT / "02_irr_binary.csv", index=False)
print(df_bin.to_string(index=False))

with open(OUT / "02_irr_binary.md", "w") as f:
    f.write("# IRR (Binary, Criterion-Level) — gpt52 vs claude_sonnet\n\n")
    f.write("Paired on (pattern, query_id, dimension, criterion_id). Claude Opus excluded because its criterion_ids are unpinned (674 distinct vs 106).\n\n")
    f.write("| Dimension | N pairs | Cohen's κ | Gwet's AC1 | Prev (gpt52) | Prev (sonnet) | % Agree |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for _, r in df_bin.iterrows():
        f.write(f"| {r['dimension']} | {r['n_pairs']} | {r['kappa']:.3f} | {r['ac1']:.3f} | "
                f"{r['prev_gpt']:.3f} | {r['prev_son']:.3f} | {r['agree_rate']:.3f} |\n")
    f.write("\nNote: AC1 is preferred over κ when class prevalence is imbalanced (|prev−0.5| large).\n")
    f.write("Cases where AC1 >> κ indicate prevalence-driven κ deflation.\n")

# --------------------------------------------------------------------------
# 3. System-ranking correlation
# --------------------------------------------------------------------------
print("\n[3] System-ranking correlation...")
# Per-judge per-pattern mean, for each dimension + overall
rank_records = []
for dim in DIMENSIONS:
    sub = base_scores[base_scores["dimension"] == dim]
    tbl = sub.groupby(["pattern", "judge"])["score"].mean().reset_index()
    wide = tbl.pivot(index="pattern", columns="judge", values="score")
    rank_records.append((dim, wide))
# overall
ov = base_overall.groupby(["pattern", "judge"])["overall"].mean().reset_index()
ov_wide = ov.pivot(index="pattern", columns="judge", values="overall")
rank_records.append(("overall", ov_wide))

ranking_md_lines = ["# System-Ranking Correlation Across Judges\n",
                    "Per-pattern means (N=11 base patterns). Spearman ρ and Kendall τ for each judge pair.\n\n"]
heatmap_data = {}  # per dim spearman 3x3 matrix
for dim, wide in rank_records:
    ranking_md_lines.append(f"## {dim}\n\n")
    rho_mat = pd.DataFrame(index=JUDGES, columns=JUDGES, dtype=float)
    tau_mat = pd.DataFrame(index=JUDGES, columns=JUDGES, dtype=float)
    for a in JUDGES:
        for b in JUDGES:
            if a == b:
                rho_mat.loc[a, b] = 1.0
                tau_mat.loc[a, b] = 1.0
            else:
                r = stats.spearmanr(wide[a], wide[b]).statistic
                t = stats.kendalltau(wide[a], wide[b]).statistic
                rho_mat.loc[a, b] = r
                tau_mat.loc[a, b] = t
    heatmap_data[dim] = rho_mat.astype(float)
    ranking_md_lines.append("Spearman ρ:\n\n")
    ranking_md_lines.append(rho_mat.astype(float).round(3).to_markdown() + "\n\n")
    ranking_md_lines.append("Kendall τ:\n\n")
    ranking_md_lines.append(tau_mat.astype(float).round(3).to_markdown() + "\n\n")
with open(OUT / "03_ranking_correlation.md", "w") as f:
    f.write("\n".join(ranking_md_lines))

# Figure 8b: small multiples of Spearman heatmaps
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.ravel()
for i, (dim, wide) in enumerate(rank_records):
    sns.heatmap(heatmap_data[dim], annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=axes[i], cbar=(i == 9), square=True,
                xticklabels=JUDGES, yticklabels=JUDGES)
    axes[i].set_title(dim, fontsize=10)
    axes[i].tick_params(axis='x', labelrotation=45, labelsize=8)
    axes[i].tick_params(axis='y', labelrotation=0, labelsize=8)
plt.suptitle("Fig 8b — Spearman ρ on per-pattern means across judges", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "fig_8b.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "fig_8b.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 8a: judge-pair scatter on overall scores per (pattern, query)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs = [("gpt52", "claude_sonnet"), ("gpt52", "claude_opus"), ("claude_sonnet", "claude_opus")]
for ax, (a, b) in zip(axes, pairs):
    x = wide_overall[a].to_numpy()
    y = wide_overall[b].to_numpy()
    r_pear = stats.pearsonr(x, y).statistic
    # ICC(2,1) for this pair
    long = wide_overall[[a, b]].reset_index().melt(
        id_vars=["pattern", "query_id"], value_vars=[a, b],
        var_name="judge", value_name="score"
    )
    long["unit"] = long["pattern"].astype(str) + "|" + long["query_id"].astype(str)
    try:
        res = pg.intraclass_corr(data=long, targets="unit", raters="judge", ratings="score")
        icc21 = float(res[res["Type"] == "ICC2"].iloc[0]["ICC"])
    except Exception:
        icc21 = np.nan
    ax.scatter(x, y, alpha=0.25, s=10)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}\nPearson r={r_pear:.3f}, ICC(2,1)={icc21:.3f}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
plt.suptitle("Fig 8a — Judge pair agreement on overall scores (per pattern×query)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "fig_8a.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "fig_8a.png", dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------
# 4. Bias: length, citations, self-preference
# --------------------------------------------------------------------------
print("\n[4] Bias audits...")

# Attach run metadata
run_feats = df_runs[["pattern", "query_id", "report_word_count", "citations", "pattern_family"]].copy()
base_runs = run_feats[run_feats["pattern_family"] == "base"].copy()
joined = base_overall.merge(
    base_runs[["pattern", "query_id", "report_word_count", "citations"]],
    on=["pattern", "query_id"], how="left"
)
joined["log_wc"] = np.log1p(joined["report_word_count"].fillna(0))
joined["log_cit"] = np.log1p(joined["citations"].fillna(0))
# pattern_family split: gpt4o (p0-p8) vs local7b (p9-p10)
def _model_family(p):
    tail = p.replace("base_", "")
    if tail in ("p9", "p10"):
        return "local7b"
    return "gpt4o"
joined["model_family"] = joined["pattern"].apply(_model_family)
# judge family
def _jf(j):
    return "gpt" if j == "gpt52" else "claude"
joined["judge_family"] = joined["judge"].apply(_jf)
joined["self_pref"] = ((joined["judge_family"] == "gpt") & (joined["model_family"] == "gpt4o")).astype(int)

# Mixed-effects regressions per judge — query_id as random intercept to handle nested data
# (same query observed across 11 patterns is non-independent).
import statsmodels.formula.api as smf
bias_rows = []
for judge in JUDGES:
    sub = joined[joined["judge"] == judge].dropna(subset=["overall", "log_wc", "log_cit"]).copy()
    sub["is_local7b"] = (sub["model_family"] == "local7b").astype(int)
    # Standardize predictors
    for c in ["log_wc", "log_cit"]:
        sub[f"z_{c}"] = (sub[c] - sub[c].mean()) / sub[c].std(ddof=0)
    try:
        model = smf.mixedlm(
            "overall ~ z_log_wc + z_log_cit + is_local7b",
            data=sub, groups=sub["query_id"],
        ).fit(reml=True, method="lbfgs")
        ci = model.conf_int()
        for raw, lbl in [("z_log_wc", "log_wc"), ("z_log_cit", "log_cit"), ("is_local7b", "is_local7b")]:
            bias_rows.append({
                "judge": judge, "predictor": lbl,
                "beta": float(model.params[raw]),
                "ci_lo": float(ci.loc[raw, 0]), "ci_hi": float(ci.loc[raw, 1]),
                "p": float(model.pvalues[raw]),
                "n": int(len(sub)),
                "model": "mixedlm(re=query_id)",
            })
    except Exception as e:
        # Fallback to cluster-robust OLS on query_id
        X = sub[["z_log_wc", "z_log_cit", "is_local7b"]].copy()
        X = sm.add_constant(X)
        m = sm.OLS(sub["overall"], X).fit(cov_type="cluster", cov_kwds={"groups": sub["query_id"]})
        ci = m.conf_int()
        for raw, lbl in [("z_log_wc", "log_wc"), ("z_log_cit", "log_cit"), ("is_local7b", "is_local7b")]:
            bias_rows.append({
                "judge": judge, "predictor": lbl,
                "beta": float(m.params[raw]),
                "ci_lo": float(ci.loc[raw, 0]), "ci_hi": float(ci.loc[raw, 1]),
                "p": float(m.pvalues[raw]), "n": int(len(sub)),
                "model": f"OLS+cluster(query_id) [mixedlm failed: {e}]",
            })

df_bias = pd.DataFrame(bias_rows)
# Benjamini-Hochberg correction across all 9 bias tests
from statsmodels.stats.multitest import multipletests
_, p_bh, _, _ = multipletests(df_bias["p"].fillna(1.0).to_numpy(), method="fdr_bh")
df_bias["p_bh"] = p_bh
df_bias["sig_bh_05"] = df_bias["p_bh"] < 0.05
df_bias.to_csv(OUT / "04_bias_regression.csv", index=False)

# Length bias per dimension
length_rows = []
for judge in JUDGES:
    for dim in DIMENSIONS:
        sub = base_scores[(base_scores["judge"] == judge) & (base_scores["dimension"] == dim)].merge(
            base_runs[["pattern", "query_id", "report_word_count", "citations"]],
            on=["pattern", "query_id"], how="left"
        ).dropna(subset=["score", "report_word_count"])
        if len(sub) < 5:
            continue
        r_p = stats.pearsonr(sub["score"], sub["report_word_count"]).statistic
        r_s = stats.spearmanr(sub["score"], sub["report_word_count"]).statistic
        length_rows.append({"judge": judge, "dimension": dim, "n": len(sub),
                            "pearson_wc": float(r_p), "spearman_wc": float(r_s)})
df_length = pd.DataFrame(length_rows)
df_length.to_csv(OUT / "04_bias_length_by_dim.csv", index=False)

with open(OUT / "04_bias_length.md", "w") as f:
    f.write("# Length & Citation Bias Audit\n\n")
    f.write("## Per-judge mixed-effects regression: overall ~ log_wc + log_cit + is_local7b (standardized) + (1|query_id)\n\n")
    f.write("Query is a random intercept to respect nested structure (same query × 11 patterns). p_bh = Benjamini-Hochberg FDR-corrected across all 9 tests.\n\n")
    f.write("| Judge | Predictor | β (std) | 95% CI | p_raw | p_bh | sig (BH<0.05) | N |\n")
    f.write("|---|---|---:|:---:|---:|---:|:---:|---:|\n")
    for _, r in df_bias.iterrows():
        sig = "✓" if r["sig_bh_05"] else ""
        f.write(f"| {r['judge']} | {r['predictor']} | {r['beta']:+.3f} | "
                f"({r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}) | {r['p']:.4f} | {r['p_bh']:.4f} | {sig} | {r['n']} |\n")
    f.write("\n## Dimension-level length correlation (score vs report_word_count)\n\n")
    f.write("Pearson and Spearman per (judge, dimension). Note: these are simple correlations (not query-clustered).\n\n")
    f.write("| Judge | Dimension | N | Pearson r | Spearman ρ |\n")
    f.write("|---|---|---:|---:|---:|\n")
    for _, r in df_length.iterrows():
        f.write(f"| {r['judge']} | {r['dimension']} | {r['n']} | {r['pearson_wc']:.3f} | {r['spearman_wc']:.3f} |\n")

# Fig bias2: length-bias scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, j in zip(axes, JUDGES):
    sub = joined[joined["judge"] == j].dropna(subset=["overall", "report_word_count"])
    ax.scatter(sub["report_word_count"], sub["overall"], alpha=0.2, s=8)
    # OLS line
    X = sm.add_constant(sub["report_word_count"])
    m = sm.OLS(sub["overall"], X).fit()
    xs = np.linspace(sub["report_word_count"].min(), sub["report_word_count"].max(), 100)
    ax.plot(xs, m.params["const"] + m.params["report_word_count"] * xs, "r-", lw=2)
    r = stats.pearsonr(sub["report_word_count"], sub["overall"]).statistic
    ax.set_title(f"{j}\nPearson r={r:.3f}", fontsize=10)
    ax.set_xlabel("report_word_count"); ax.set_ylabel("overall_score")
plt.suptitle("Fig bias2 — Length bias per judge", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "fig_bias2.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "fig_bias2.png", dpi=300, bbox_inches="tight")
plt.close()

# Self-preference 2x2
print("\n  Self-preference 2x2...")
sp_rows = []
for jf in ["gpt", "claude"]:
    for mf in ["gpt4o", "local7b"]:
        sub = joined[(joined["judge_family"] == jf) & (joined["model_family"] == mf)]
        vals = sub["overall"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        mean = vals.mean()
        # bootstrap CI
        boots = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sp_rows.append({"judge_family": jf, "model_family": mf,
                        "mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi),
                        "n": int(len(vals))})
df_sp = pd.DataFrame(sp_rows)
df_sp.to_csv(OUT / "05_self_preference.csv", index=False)

# Formal test: interaction (judge_family × model_family) on overall score
# Two complementary tests, both pre-registered:
#   (a) Two-way ANOVA on raw scores at the (pattern, judge) means level.
#   (b) Rank-transform within judge, then test the interaction. This removes
#       multiplicative judge-stringency confounding (Conover-Iman style).
pattern_means = joined.groupby(["judge_family", "model_family", "pattern", "judge"])["overall"].mean().reset_index()

try:
    anova = pg.anova(data=pattern_means, dv="overall", between=["judge_family", "model_family"])
    inter_row = anova[anova["Source"] == "judge_family * model_family"]
    interaction_p_raw = float(inter_row["p_unc"].iloc[0])
    interaction_F_raw = float(inter_row["F"].iloc[0])
    interaction_np2_raw = float(inter_row["np2"].iloc[0]) if "np2" in inter_row.columns else float("nan")
except Exception as e:
    interaction_p_raw = float("nan"); interaction_F_raw = float("nan"); interaction_np2_raw = float("nan")
    print(f"  ANOVA(raw) failed: {e}")

# Rank-within-judge before testing interaction (defends against multiplicative
# judge-stringency: Sonnet uniformly more lenient than GPT-5.2 doesn't read as a self-preference effect)
joined_rank = joined.copy()
joined_rank["overall_rank_within_judge"] = (
    joined_rank.groupby("judge")["overall"].rank(method="average", pct=True)
)
pattern_means_rank = joined_rank.groupby(
    ["judge_family", "model_family", "pattern", "judge"]
)["overall_rank_within_judge"].mean().reset_index()
try:
    anova_rank = pg.anova(
        data=pattern_means_rank, dv="overall_rank_within_judge",
        between=["judge_family", "model_family"],
    )
    inter_row = anova_rank[anova_rank["Source"] == "judge_family * model_family"]
    interaction_p_rank = float(inter_row["p_unc"].iloc[0])
    interaction_F_rank = float(inter_row["F"].iloc[0])
    interaction_np2_rank = float(inter_row["np2"].iloc[0]) if "np2" in inter_row.columns else float("nan")
except Exception as e:
    interaction_p_rank = float("nan"); interaction_F_rank = float("nan"); interaction_np2_rank = float("nan")
    print(f"  ANOVA(rank) failed: {e}")

# Permutation test on rank-based gap statistic (cleanest given non-normality + unbalanced cells)
def _gap_rank(df_r):
    def cell(jf, mf):
        s = df_r[(df_r.judge_family==jf)&(df_r.model_family==mf)]["overall_rank_within_judge"]
        return s.mean() if len(s) else np.nan
    return (cell("gpt", "gpt4o") - cell("claude", "gpt4o")) - (cell("gpt", "local7b") - cell("claude", "local7b"))

observed_gap_rank = _gap_rank(joined_rank)
# Permute model_family within judge (preserves judge-stringency, breaks family ↔ score link)
n_perm = 5000
gaps_perm = np.empty(n_perm)
for i in range(n_perm):
    permuted = joined_rank.copy()
    permuted["model_family"] = (
        permuted.groupby("judge")["model_family"]
        .transform(lambda s: s.sample(frac=1.0, random_state=int(RNG.integers(0, 1_000_000))).values)
    )
    gaps_perm[i] = _gap_rank(permuted)
perm_p = float((np.abs(gaps_perm) >= abs(observed_gap_rank)).mean())

# Self-preference gap on raw scores (kept for context but flagged as confounded)
def _cell(jf, mf):
    return float(joined[(joined.judge_family==jf)&(joined.model_family==mf)]["overall"].mean())
gap = (_cell("gpt", "gpt4o") - _cell("claude", "gpt4o")) - (_cell("gpt", "local7b") - _cell("claude", "local7b"))

with open(OUT / "05_bias_self_preference.md", "w") as f:
    f.write("# Self-Preference Bias (Judge Family × Model Family)\n\n")
    f.write("Question: does GPT-5.2 score GPT-4o pipeline outputs higher than Claude judges do, controlling for a non-GPT comparator (local 7B)?\n\n")
    f.write("## 2×2 cell means (raw overall score) with bootstrap 95% CI\n\n")
    f.write("**Caveat:** raw cell means confound a true family-preference effect with simple judge-stringency differences.\n")
    f.write("GPT-5.2 is uniformly harsher than Claude judges across both model families. The rank-within-judge analysis below\n")
    f.write("is the primary self-preference test.\n\n")
    f.write("| Judge family | Model family | N | Mean | 95% CI |\n")
    f.write("|---|---|---:|---:|:---:|\n")
    for _, r in df_sp.iterrows():
        f.write(f"| {r['judge_family']} | {r['model_family']} | {r['n']} | {r['mean']:.3f} "
                f"| ({r['ci_lo']:.3f}, {r['ci_hi']:.3f}) |\n")
    f.write(f"\nRaw self-preference gap = {gap:+.3f}\n\n")
    f.write(f"Two-way ANOVA on raw scores: interaction F = {interaction_F_raw:.3f}, p = {interaction_p_raw:.4f}, η²p = {interaction_np2_raw:.3f}\n\n")
    f.write("## Rank-within-judge analysis (PRIMARY test — confound-corrected)\n\n")
    f.write("Each judge's scores are converted to within-judge percentile ranks before testing. Removes multiplicative judge-stringency confounding.\n\n")
    f.write(f"- **Rank-based self-preference gap: {observed_gap_rank:+.3f}**\n")
    f.write(f"- Two-way ANOVA on ranks: interaction F = {interaction_F_rank:.3f}, p = {interaction_p_rank:.4f}, η²p = {interaction_np2_rank:.3f}\n")
    f.write(f"- **Permutation test (5000 permutations, model_family permuted within judge): p = {perm_p:.4f}**\n\n")
    f.write("Positive gap ⇒ GPT judge over-credits GPT-4o outputs relative to Claude judges, beyond what Claude judges do.\n\n")
    f.write("## Methodological note\n\n")
    f.write("The 2×2 design has only 2 unique local7b patterns (P9, P10) versus 9 gpt4o patterns. Power is structurally low for the interaction term. Treat all p-values as exploratory; the rank-permutation test is the most defensible single statistic.\n")

# Fig bias1: 2x2 bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Raw scores
pivot_sp = df_sp.pivot(index="model_family", columns="judge_family", values="mean")
err = df_sp.copy()
err["err"] = (err["ci_hi"] - err["ci_lo"]) / 2
err_pivot = err.pivot(index="model_family", columns="judge_family", values="err")
pivot_sp.plot(kind="bar", yerr=err_pivot, ax=axes[0], capsize=4, rot=0,
              color=["#2a9d8f", "#e76f51"], legend=True)
axes[0].set_ylabel("Mean overall score (raw)")
axes[0].set_title(f"Raw scores  (gap={gap:+.3f}, ANOVA p={interaction_p_raw:.3f})")
axes[0].set_ylim(0, 1)

# Ranks within judge
rank_means = joined_rank.groupby(["judge_family", "model_family"])["overall_rank_within_judge"].mean().reset_index()
rank_pivot = rank_means.pivot(index="model_family", columns="judge_family", values="overall_rank_within_judge")
rank_pivot.plot(kind="bar", ax=axes[1], rot=0, color=["#2a9d8f", "#e76f51"], legend=True)
axes[1].set_ylabel("Mean rank within judge")
axes[1].set_title(f"Rank-within-judge  (gap={observed_gap_rank:+.3f}, perm p={perm_p:.3f})")
axes[1].set_ylim(0, 1)

plt.suptitle("Fig bias1 — Self-preference (raw vs rank-within-judge)", fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "fig_bias1.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "fig_bias1.png", dpi=300, bbox_inches="tight")
plt.close()

# Citation confound on factual_accuracy
print("\n  Citation confound on factual_accuracy...")
fa = base_scores[base_scores["dimension"] == "factual_accuracy"].merge(
    base_runs[["pattern", "query_id", "citations"]], on=["pattern", "query_id"], how="left"
).dropna(subset=["citations"])
cit_rows = []
for j in JUDGES:
    sub = fa[fa["judge"] == j]
    r = stats.pearsonr(sub["score"], sub["citations"]).statistic
    rho = stats.spearmanr(sub["score"], sub["citations"]).statistic
    cit_rows.append({"judge": j, "n": len(sub), "pearson_fa_cit": float(r),
                     "spearman_fa_cit": float(rho)})
df_cit = pd.DataFrame(cit_rows)
df_cit.to_csv(OUT / "06_citation_confound.csv", index=False)
with open(OUT / "06_bias_citation_confound.md", "w") as f:
    f.write("# Citation-Count Confound on factual_accuracy\n\n")
    f.write("If r > 0.6 the dimension is partially measuring citation density, not truth.\n\n")
    f.write("| Judge | N | Pearson(fa, citations) | Spearman(fa, citations) |\n")
    f.write("|---|---:|---:|---:|\n")
    for _, r in df_cit.iterrows():
        f.write(f"| {r['judge']} | {r['n']} | {r['pearson_fa_cit']:.3f} | {r['spearman_fa_cit']:.3f} |\n")

# --------------------------------------------------------------------------
# 5. Dispersion
# --------------------------------------------------------------------------
print("\n[5] Judge dispersion...")
disp = base_overall.pivot_table(index=["pattern", "query_id"], columns="judge",
                                values="overall", aggfunc="mean").dropna(subset=JUDGES)
disp["std"] = disp[JUDGES].std(axis=1)
disp["range"] = disp[JUDGES].max(axis=1) - disp[JUDGES].min(axis=1)
disp.to_csv(OUT / "07_dispersion_cells.csv")
high = disp[disp["std"] > disp["std"].quantile(0.95)].sort_values("std", ascending=False)
with open(OUT / "07_judge_dispersion.md", "w") as f:
    f.write("# Judge Dispersion (per pattern × query)\n\n")
    f.write(f"N cells: {len(disp)}\n")
    f.write(f"Mean std across 3 judges: {disp['std'].mean():.3f}\n")
    f.write(f"Median std: {disp['std'].median():.3f}\n")
    f.write(f"95th percentile std: {disp['std'].quantile(0.95):.3f}\n")
    f.write(f"Max std: {disp['std'].max():.3f}\n\n")
    f.write("## Top 20 highest-disagreement cells\n\n")
    f.write(high.head(20).round(3).reset_index().to_markdown(index=False) + "\n")

plt.figure(figsize=(8, 5))
plt.hist(disp["std"], bins=40, color="#264653", alpha=0.85)
plt.axvline(disp["std"].mean(), color="red", lw=1, label=f"mean={disp['std'].mean():.3f}")
plt.axvline(disp["std"].quantile(0.95), color="orange", lw=1, label=f"p95={disp['std'].quantile(0.95):.3f}")
plt.xlabel("std of overall score across 3 judges")
plt.ylabel("# (pattern × query) cells")
plt.title("Judge dispersion — std of overall across 3 judges per cell")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "fig_dispersion.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIGS / "fig_dispersion.png", dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------
# 6. Aggregation decision
# --------------------------------------------------------------------------
print("\n[6] Aggregation decision...")
decisions = []
for _, r in df_cont.iterrows():
    a = r["alpha"]
    if a >= 0.80:
        d = "average across judges (strong agreement)"
    elif a >= 0.67:
        d = "average + per-judge sensitivity in appendix (tentative)"
    else:
        d = "per-judge analysis; report ranking concordance only (weak)"
    decisions.append({"dimension": r["dimension"], "alpha": a, "decision": d})
decision_df = pd.DataFrame(decisions)

with open(OUT / "aggregation_decision.md", "w") as f:
    f.write("# Aggregation Decision\n\n")
    f.write("Based on Krippendorff α (interval) across gpt52, claude_opus, claude_sonnet on base patterns (N=983 cells).\n\n")
    f.write("Thresholds: α≥0.80 → average; 0.67≤α<0.80 → average + appendix sensitivity; α<0.67 → per-judge + ranking concordance.\n\n")
    f.write("| Dimension | α | Decision |\n")
    f.write("|---|---:|---|\n")
    for _, r in decision_df.iterrows():
        f.write(f"| {r['dimension']} | {r['alpha']:.3f} | {r['decision']} |\n")
    f.write(f"| **overall** | {overall_alpha:.3f} | ")
    if overall_alpha >= 0.80:
        f.write("average across judges |\n")
    elif overall_alpha >= 0.67:
        f.write("average + per-judge sensitivity in appendix |\n")
    else:
        f.write("per-judge + ranking concordance only |\n")
    # Overall recommendation
    f.write("\n## Overall recommendation\n\n")
    strong = (decision_df["alpha"] >= 0.80).sum()
    tentative = ((decision_df["alpha"] >= 0.67) & (decision_df["alpha"] < 0.80)).sum()
    weak = (decision_df["alpha"] < 0.67).sum()
    f.write(f"- Strong (α≥0.80): {strong} dimensions\n")
    f.write(f"- Tentative (0.67≤α<0.80): {tentative} dimensions\n")
    f.write(f"- Weak (α<0.67): {weak} dimensions\n\n")
    if weak > 4:
        rec = "Primary analysis: **per-judge** across most dimensions. Average only where α≥0.80. Report ranking concordance (Spearman ρ between judges' pattern rankings) as the cross-judge headline."
    elif tentative + weak > strong:
        rec = "Primary analysis: **per-judge tables in main paper + averaged composite in appendix**, with clear flags for weak dimensions. Use ranking concordance to validate the leaderboard."
    else:
        rec = "Primary analysis: **averaged composite** across judges, with per-judge sensitivity in appendix."
    f.write(f"**Recommendation:** {rec}\n")

# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------
print("\n[7] Writing summary...")
with open(OUT / "summary.md", "w") as f:
    f.write("# Phase 1 Judge Validation — Summary\n\n")
    f.write("3 judges (gpt52, claude_opus, claude_sonnet) on 11 base patterns × 90 queries (983 cells).\n\n")
    f.write("## Headline numbers\n\n")
    f.write("### Continuous IRR (per dimension)\n\n")
    f.write("| Dimension | α | ICC(A,1) | ICC(A,k) |\n|---|---:|---:|---:|\n")
    for _, r in df_cont.iterrows():
        f.write(f"| {r['dimension']} | {r['alpha']:.3f} | {r['ICC_A1']:.3f} | {r['ICC_Ak']:.3f} |\n")
    f.write(f"| **overall** | {overall_alpha:.3f} | {overall_icc_a1:.3f} | {overall_icc_ak:.3f} |\n")
    f.write("\n### Binary IRR (gpt52 vs claude_sonnet, criterion-level)\n\n")
    f.write("| Dimension | κ | AC1 | N |\n|---|---:|---:|---:|\n")
    for _, r in df_bin.iterrows():
        f.write(f"| {r['dimension']} | {r['kappa']:.3f} | {r['ac1']:.3f} | {r['n_pairs']} |\n")
    f.write("\n### Bias highlights\n\n")
    f.write("| Judge | β(log_wc) | β(log_cit) | β(is_local7b) |\n|---|---:|---:|---:|\n")
    for j in JUDGES:
        jr = df_bias[df_bias["judge"] == j].set_index("predictor")
        f.write(f"| {j} | {jr.loc['log_wc','beta']:+.3f} | {jr.loc['log_cit','beta']:+.3f} | "
                f"{jr.loc['is_local7b','beta']:+.3f} |\n")
    f.write(f"\n**Self-preference (raw) gap**: {gap:+.3f}; ANOVA(raw) interaction p = {interaction_p_raw:.4f}\n")
    f.write(f"**Self-preference (rank-within-judge) gap**: {observed_gap_rank:+.3f}; permutation p = {perm_p:.4f}\n")
    f.write("\n### Citation confound on factual_accuracy\n\n")
    f.write("| Judge | Pearson r |\n|---|---:|\n")
    for _, r in df_cit.iterrows():
        f.write(f"| {r['judge']} | {r['pearson_fa_cit']:.3f} |\n")
    f.write(f"\n### Dispersion\n\n")
    f.write(f"Mean std across 3 judges per cell: **{disp['std'].mean():.3f}**; p95: {disp['std'].quantile(0.95):.3f}\n")
    f.write("\n## Output files\n\n")
    for p in sorted(OUT.glob("*.md")):
        f.write(f"- `{p.name}`\n")
    for p in sorted(FIGS.glob("*.pdf")):
        f.write(f"- `figures/{p.name}`\n")

# Save JSON digest
digest = {
    "continuous_irr": df_cont.to_dict(orient="records"),
    "overall_alpha": overall_alpha,
    "overall_icc_a1": overall_icc_a1,
    "overall_icc_ak": overall_icc_ak,
    "binary_irr": df_bin.to_dict(orient="records"),
    "bias_regression": df_bias.to_dict(orient="records"),
    "self_preference_cells": df_sp.to_dict(orient="records"),
    "self_preference_gap": gap,
    "interaction_p_raw": interaction_p_raw,
    "interaction_F_raw": interaction_F_raw,
    "interaction_p_rank": interaction_p_rank,
    "interaction_F_rank": interaction_F_rank,
    "self_preference_gap_rank": observed_gap_rank,
    "permutation_p": perm_p,
    "citation_confound": df_cit.to_dict(orient="records"),
    "dispersion": {"mean_std": float(disp["std"].mean()),
                   "median_std": float(disp["std"].median()),
                   "p95_std": float(disp["std"].quantile(0.95)),
                   "n_cells": int(len(disp))},
}
with open(OUT / "digest.json", "w") as f:
    json.dump(digest, f, indent=2, default=str)

print("\nDone. Outputs in", OUT)
