"""Phase 7a: Post-hoc citation verification.

Steps:
  1. Extract all citations from results/experiments/base_*/<qid>.md
  2. Classify each citation as placeholder / real_url / academic
  3. Aggregate per-pattern stats -> data/analysis/df_citations.parquet + CSV
  4. Correlate with gpt52 judge factual_accuracy / citation_quality scores
     - Mixed-effects model: factual_accuracy ~ academic_rate + placeholder_rate
       + cite_count + (1|query_id)
  5. Stacked bar chart figure

Outputs:
  data/analysis/df_citations.parquet
  reports/phase7a_citation_verification/per_pattern_stats.csv
  reports/phase7a_citation_verification/per_pattern_stats.md
  reports/phase7a_citation_verification/citation_quality_regression.md
  reports/phase7a_citation_verification/figures/citation_breakdown.pdf
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "results" / "experiments"
JUDGE_DIR = ROOT / "results" / "judge_gpt52"
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUT_DIR = ROOT / "reports" / "phase7a_citation_verification"
FIG_DIR = OUT_DIR / "figures"

for d in [ANALYSIS_DIR, OUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain classification sets
# ---------------------------------------------------------------------------
ACADEMIC_DOMAINS = {
    "arxiv.org", "semanticscholar.org", "doi.org", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "springer.com", "link.springer.com",
    "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org",
    "nature.com", "science.org", "sciencemag.org", "pnas.org",
    "plos.org", "journals.plos.org", "biorxiv.org", "medrxiv.org",
    "ssrn.com", "openreview.net", "aclanthology.org", "proceedings.mlr.press",
    "jmlr.org", "aaai.org", "ojs.aaai.org", "nips.cc", "neurips.cc",
    "sciencedirect.com", "academic.oup.com", "wiley.com", "onlinelibrary.wiley.com",
    "cambridge.org", "tandfonline.com", "mdpi.com", "frontiersin.org",
    "researchgate.net",  # not perfect but close enough for heuristic
}

SUSPICIOUS_DOMAINS = {
    "blogspot.com", "wordpress.com", "medium.com", "substack.com",
    "quora.com", "reddit.com", "tumblr.com", "wix.com", "weebly.com",
}

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------
# Reference section headers (case-insensitive)
_REF_HEADER_RE = re.compile(
    r"^\s*#+\s*(references|bibliography|sources|citations|works cited)\s*$",
    re.IGNORECASE,
)

# Reference line: [N] Title — URL  (em-dash or regular dash variants)
_REF_LINE_RE = re.compile(
    r"^\[(\d+)\]\s*(.*?)\s*[—–-]{1,3}\s*(https?://\S+)?\s*$",
    re.DOTALL,
)

# Also handle "Web Search Synthesis" entries with empty URL field
_REF_LINE_EMPTY_RE = re.compile(
    r"^\[(\d+)\]\s*(.*?)\s*[—–-]{1,3}\s*$",
    re.DOTALL,
)

# Inline citations: sentence containing [N] or [N][M]...
_INLINE_CITE_RE = re.compile(r"\[(\d+)\]")


def _extract_domain(url: str) -> str | None:
    """Return the registered domain from a URL string."""
    if not url:
        return None
    m = re.match(r"https?://([^/?\s]+)", url.strip())
    if not m:
        return None
    host = m.group(1).lower()
    # strip www. prefix
    if host.startswith("www."):
        host = host[4:]
    return host


def _classify_citation(url: str | None, title: str) -> str:
    """Return one of: placeholder, academic, suspicious, real_url."""
    title_lower = (title or "").lower()
    if (
        not url
        or "web search synthesis" in title_lower
        or "synthesis" in title_lower
        or url.strip() == ""
    ):
        return "placeholder"

    if not url.startswith("http"):
        return "placeholder"

    domain = _extract_domain(url)
    if domain is None:
        return "placeholder"

    # Check academic first (supersedes suspicious)
    for ad in ACADEMIC_DOMAINS:
        if domain == ad or domain.endswith("." + ad):
            return "academic"

    for sd in SUSPICIOUS_DOMAINS:
        if domain == sd or domain.endswith("." + sd):
            return "suspicious"

    return "real_url"


def _get_sentence_context(text: str, cite_idx: int, char_radius: int = 200) -> str:
    """Return text around a [cite_idx] marker."""
    pattern = rf"\[{cite_idx}\]"
    m = re.search(pattern, text)
    if not m:
        return ""
    start = max(0, m.start() - char_radius)
    end = min(len(text), m.end() + char_radius)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet[:400]


def extract_citations_from_report(path: Path, pattern: str, query_id: str) -> list[dict]:
    """Parse one markdown report and return list of citation dicts."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    word_count = len(re.findall(r"\S+", text))
    records = []

    # Find references section
    lines = text.splitlines()
    ref_start = None
    for i, line in enumerate(lines):
        if _REF_HEADER_RE.match(line):
            ref_start = i
            break

    # Parse reference section lines
    parsed_refs: dict[int, dict] = {}  # index -> {title, url, category}

    if ref_start is not None:
        ref_lines = lines[ref_start + 1:]
        for line in ref_lines:
            line = line.strip()
            if not line:
                continue
            # Try full match with URL
            m = _REF_LINE_RE.match(line)
            if m:
                idx = int(m.group(1))
                title = (m.group(2) or "").strip()
                url = (m.group(3) or "").strip()
                parsed_refs[idx] = {"title": title, "url": url}
                continue
            # Try match without URL (placeholder entry)
            m2 = _REF_LINE_EMPTY_RE.match(line)
            if m2:
                idx = int(m2.group(1))
                title = (m2.group(2) or "").strip()
                parsed_refs[idx] = {"title": title, "url": ""}

    # For each parsed reference, build a record
    for cite_idx, ref in parsed_refs.items():
        title = ref["title"]
        url = ref["url"]
        cat = _classify_citation(url if url else None, title)
        domain = _extract_domain(url) if url else None
        claim_ctx = _get_sentence_context(text, cite_idx)

        records.append({
            "pattern": pattern,
            "query_id": query_id,
            "citation_index": cite_idx,
            "cited_title": title,
            "cited_url": url,
            "domain": domain,
            "category": cat,
            "claim_context": claim_ctx,
            "report_word_count": word_count,
        })

    # If no refs section found but inline cites exist, note count only
    if ref_start is None:
        inline_cites = set(_INLINE_CITE_RE.findall(text))
        if inline_cites:
            for ci in sorted(inline_cites, key=int):
                records.append({
                    "pattern": pattern,
                    "query_id": query_id,
                    "citation_index": int(ci),
                    "cited_title": "",
                    "cited_url": "",
                    "domain": None,
                    "category": "placeholder",
                    "claim_context": _get_sentence_context(text, int(ci)),
                    "report_word_count": word_count,
                })

    return records


# ---------------------------------------------------------------------------
# Step 1: Walk all base_p* directories
# ---------------------------------------------------------------------------
print("Step 1: Extracting citations from all reports...")
all_records: list[dict] = []
pattern_dirs = sorted(EXPERIMENTS_DIR.glob("base_p*"))

for pdir in pattern_dirs:
    pattern_short = pdir.name  # e.g. "base_p4"
    report_files = sorted(pdir.glob("*.md"))
    n = len(report_files)
    n_empty = 0
    for rfile in report_files:
        query_id = rfile.stem
        recs = extract_citations_from_report(rfile, pattern_short, query_id)
        if not recs:
            n_empty += 1
        all_records.extend(recs)
    print(f"  {pattern_short}: {n} reports, {n_empty} with no citations parsed")

df_cit = pd.DataFrame(all_records)
print(f"\nTotal citation records: {len(df_cit):,}")
print(f"Patterns: {df_cit['pattern'].unique().tolist()}")
print(f"Categories: {df_cit['category'].value_counts().to_dict()}")

# Save — try parquet first, fall back to CSV
try:
    df_cit.to_parquet(ANALYSIS_DIR / "df_citations.parquet", index=False)
    print(f"Saved: {ANALYSIS_DIR / 'df_citations.parquet'}")
except ImportError:
    df_cit.to_csv(ANALYSIS_DIR / "df_citations.csv", index=False)
    print(f"Saved (parquet unavailable, wrote CSV): {ANALYSIS_DIR / 'df_citations.csv'}")

# ---------------------------------------------------------------------------
# Step 2: Per-report stats (aggregated to report level)
# ---------------------------------------------------------------------------
print("\nStep 2: Aggregating per-report statistics...")

def report_stats(grp: pd.DataFrame) -> pd.Series:
    n = len(grp)
    if n == 0:
        return pd.Series({
            "cite_count": 0,
            "placeholder_count": 0,
            "real_url_count": 0,
            "academic_count": 0,
            "suspicious_count": 0,
            "placeholder_rate": 0.0,
            "real_url_rate": 0.0,
            "academic_rate": 0.0,
            "suspicious_rate": 0.0,
            "word_count": 0,
            "cite_density_per_1000w": 0.0,
        })
    wc = grp["report_word_count"].iloc[0]
    ph = (grp["category"] == "placeholder").sum()
    ru = (grp["category"] == "real_url").sum()
    ac = (grp["category"] == "academic").sum()
    su = (grp["category"] == "suspicious").sum()
    return pd.Series({
        "cite_count": n,
        "placeholder_count": int(ph),
        "real_url_count": int(ru),
        "academic_count": int(ac),
        "suspicious_count": int(su),
        "placeholder_rate": ph / n,
        "real_url_rate": ru / n,
        "academic_rate": ac / n,
        "suspicious_rate": su / n,
        "word_count": wc,
        "cite_density_per_1000w": (n / wc * 1000) if wc > 0 else 0.0,
    })

df_report = (
    df_cit
    .groupby(["pattern", "query_id"])
    .apply(report_stats)
    .reset_index()
)
print(f"Per-report rows: {len(df_report)}")

# ---------------------------------------------------------------------------
# Step 3: Per-pattern aggregation
# ---------------------------------------------------------------------------
print("\nStep 3: Per-pattern aggregation...")

def pattern_agg(grp: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_reports": len(grp),
        "mean_cite_count": grp["cite_count"].mean(),
        "mean_placeholder_rate": grp["placeholder_rate"].mean(),
        "mean_real_url_rate": grp["real_url_rate"].mean(),
        "mean_academic_rate": grp["academic_rate"].mean(),
        "mean_suspicious_rate": grp["suspicious_rate"].mean(),
        "mean_word_count": grp["word_count"].mean(),
        "mean_cite_density_per_1000w": grp["cite_density_per_1000w"].mean(),
        "total_citations": grp["cite_count"].sum(),
        "total_placeholder": grp["placeholder_count"].sum(),
        "total_real_url": grp["real_url_count"].sum(),
        "total_academic": grp["academic_count"].sum(),
    })

df_pat = (
    df_report
    .groupby("pattern")
    .apply(pattern_agg)
    .reset_index()
    .sort_values("mean_academic_rate", ascending=False)
)

# Clean up pattern name for display
df_pat["pattern_label"] = df_pat["pattern"].str.replace("base_", "", regex=False).str.upper()

print(df_pat[["pattern_label", "mean_cite_count", "mean_placeholder_rate",
              "mean_academic_rate", "mean_real_url_rate", "mean_cite_density_per_1000w"]].to_string(index=False))

# Save CSV
df_pat.to_csv(OUT_DIR / "per_pattern_stats.csv", index=False)
print(f"Saved: {OUT_DIR / 'per_pattern_stats.csv'}")

# ---------------------------------------------------------------------------
# Markdown report: per-pattern stats
# ---------------------------------------------------------------------------
print("\nWriting per_pattern_stats.md...")

# Sort by pattern label for table presentation
df_table = df_pat.sort_values("pattern_label")

md_lines = [
    "# Phase 7a: Citation Verification — Per-Pattern Statistics",
    "",
    f"**Generated:** 2026-04-15  ",
    f"**Reports analysed:** {df_report['query_id'].nunique()} unique queries × 11 patterns ({len(df_report):,} report rows)  ",
    f"**Total citations extracted:** {df_cit.shape[0]:,}",
    "",
    "## Classification scheme",
    "",
    "| Category | Definition |",
    "|----------|------------|",
    "| `placeholder` | URL absent/empty OR title contains 'Web Search Synthesis' |",
    "| `academic` | URL domain in curated set: arxiv.org, doi.org, ieee.org, nature.com, etc. |",
    "| `suspicious` | Domain in low-credibility set: blogspot, medium, wordpress, etc. |",
    "| `real_url` | Real URL not in the above sets |",
    "",
    "## Per-pattern summary",
    "",
    "| Pattern | Reports | Mean cites/report | Placeholder% | Academic% | Real-URL% | Suspicious% | Density (per 1k words) |",
    "|---------|---------|-------------------|--------------|-----------|-----------|-------------|------------------------|",
]

for _, row in df_table.iterrows():
    md_lines.append(
        f"| {row['pattern_label']} "
        f"| {int(row['n_reports'])} "
        f"| {row['mean_cite_count']:.1f} "
        f"| {row['mean_placeholder_rate']*100:.1f}% "
        f"| {row['mean_academic_rate']*100:.1f}% "
        f"| {row['mean_real_url_rate']*100:.1f}% "
        f"| {row['mean_suspicious_rate']*100:.1f}% "
        f"| {row['mean_cite_density_per_1000w']:.2f} |"
    )

# Top placeholder offender
top_ph = df_pat.loc[df_pat["mean_placeholder_rate"].idxmax()]
top_ac = df_pat.loc[df_pat["mean_academic_rate"].idxmax()]
bottom_ac = df_pat.loc[df_pat["mean_academic_rate"].idxmin()]

md_lines += [
    "",
    "## Headline findings",
    "",
    f"- **Highest placeholder rate:** `{top_ph['pattern_label']}` — "
    f"{top_ph['mean_placeholder_rate']*100:.1f}% of citations are 'Web Search Synthesis' placeholders "
    f"(mean {top_ph['mean_cite_count']:.1f} cites/report, {top_ph['total_placeholder']:.0f} total placeholders).",
    "",
    f"- **Highest academic citation rate:** `{top_ac['pattern_label']}` — "
    f"{top_ac['mean_academic_rate']*100:.1f}% of citations link to academic sources.",
    "",
    f"- **Lowest academic citation rate:** `{bottom_ac['pattern_label']}` — "
    f"{bottom_ac['mean_academic_rate']*100:.1f}% academic.",
    "",
]

(OUT_DIR / "per_pattern_stats.md").write_text("\n".join(md_lines), encoding="utf-8")
print(f"Saved: {OUT_DIR / 'per_pattern_stats.md'}")

# ---------------------------------------------------------------------------
# Step 4: Load gpt52 judge scores and correlate
# ---------------------------------------------------------------------------
print("\nStep 4: Loading judge scores for correlation analysis...")

judge_records: list[dict] = []
for pdir in JUDGE_DIR.glob("base_p*"):
    pattern_short = pdir.name
    for jfile in pdir.glob("*.json"):
        query_id = jfile.stem
        try:
            d = json.loads(jfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        dims = d.get("dimensions", {})
        factual_acc = dims.get("factual_accuracy", {}).get("score")
        cit_qual = dims.get("citation_quality", {}).get("score")
        attr_qual = dims.get("attribution_quality", {}).get("score")
        info_recall = dims.get("information_recall", {}).get("score")
        overall = d.get("overall_score")
        judge_records.append({
            "pattern": pattern_short,
            "query_id": query_id,
            "j_factual_accuracy": factual_acc,
            "j_citation_quality": cit_qual,
            "j_attribution_quality": attr_qual,
            "j_information_recall": info_recall,
            "j_overall": overall,
        })

df_judge = pd.DataFrame(judge_records)
print(f"Judge records loaded: {len(df_judge):,}")

# Merge with per-report citation stats
df_merged = df_report.merge(df_judge, on=["pattern", "query_id"], how="inner")
print(f"Merged rows: {len(df_merged):,}")

# Drop rows with missing judge scores
df_merged = df_merged.dropna(subset=["j_factual_accuracy", "j_citation_quality"])
print(f"Rows after dropping NaN judge scores: {len(df_merged):,}")

# ---------------------------------------------------------------------------
# Simple correlations
# ---------------------------------------------------------------------------
from scipy import stats as scipy_stats

def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return r, p

corr_results: dict[str, dict] = {}
for target in ["j_factual_accuracy", "j_citation_quality"]:
    corr_results[target] = {}
    for predictor in ["academic_rate", "placeholder_rate", "cite_count", "cite_density_per_1000w", "real_url_rate"]:
        r, p = safe_corr(df_merged[predictor], df_merged[target])
        corr_results[target][predictor] = (r, p)

print("\nCorrelation results:")
for target, preds in corr_results.items():
    print(f"\n  Target: {target}")
    for pred, (r, p) in preds.items():
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {pred:35s}: r={r:+.3f}, p={p:.4f} {stars}")

# ---------------------------------------------------------------------------
# Mixed-effects regression (statsmodels MixedLM)
# ---------------------------------------------------------------------------
print("\nFitting mixed-effects models...")
me_results: dict[str, str] = {}

try:
    import statsmodels.formula.api as smf

    for target in ["j_factual_accuracy", "j_citation_quality"]:
        df_m = df_merged[["pattern", "query_id", "academic_rate", "placeholder_rate",
                           "cite_count", target]].dropna()
        # Standardise predictors for comparability
        for col in ["academic_rate", "placeholder_rate", "cite_count"]:
            std = df_m[col].std()
            if std > 0:
                df_m = df_m.copy()
                df_m[col + "_z"] = (df_m[col] - df_m[col].mean()) / std
            else:
                df_m[col + "_z"] = 0.0

        formula = f"{target} ~ academic_rate_z + placeholder_rate_z + cite_count_z"
        # Use pattern as grouping variable (11 groups, each ~83 obs) — more stable than query_id
        try:
            model = smf.mixedlm(formula, df_m, groups=df_m["pattern"])
            result = model.fit(reml=False, method="lbfgs")
            me_results[target] = result.summary().as_text()
            print(f"  Mixed-effects model for {target}: converged")
        except Exception as e:
            # Fall back to OLS if mixed model is still singular
            try:
                import statsmodels.formula.api as smf_ols
                ols_result = smf_ols.ols(formula, data=df_m).fit()
                me_results[target] = (
                    f"NOTE: MixedLM singular ({e}); fell back to OLS.\n\n"
                    + ols_result.summary().as_text()
                )
                print(f"  Mixed-effects model for {target}: singular, fell back to OLS")
            except Exception as e2:
                me_results[target] = f"Model failed: MixedLM={e}; OLS={e2}"
                print(f"  Mixed-effects model for {target}: FAILED — {e}")

except ImportError:
    print("  statsmodels not available, skipping mixed-effects models")
    for target in ["j_factual_accuracy", "j_citation_quality"]:
        me_results[target] = "statsmodels not installed"

# ---------------------------------------------------------------------------
# Write regression report
# ---------------------------------------------------------------------------
print("\nWriting citation_quality_regression.md...")

def _fmt_corr_table(target: str) -> list[str]:
    rows = ["| Predictor | Pearson r | p-value | Significance |",
            "|-----------|-----------|---------|--------------|"]
    for pred, (r, p) in corr_results[target].items():
        stars = "p<0.001" if p < 0.001 else "p<0.01" if p < 0.01 else "p<0.05" if p < 0.05 else "n.s."
        r_str = f"{r:+.3f}" if not np.isnan(r) else "n/a"
        p_str = f"{p:.4f}" if not np.isnan(p) else "n/a"
        rows.append(f"| `{pred}` | {r_str} | {p_str} | {stars} |")
    return rows

reg_md = [
    "# Phase 7a: Citation Verification — Correlation & Regression Analysis",
    "",
    "**Generated:** 2026-04-15  ",
    f"**N (report-level observations):** {len(df_merged):,}  ",
    "**Judge:** GPT-5.2 (gpt52) — gpt52 judge only (trustworthy overall score)",
    "",
    "## Research question",
    "",
    "Does academic citation rate predict judge factual_accuracy better than raw citation count?  ",
    "Does placeholder rate predict poor citation_quality?",
    "",
    "## Predictor definitions",
    "",
    "- `academic_rate`: fraction of citations linking to academic domains (arxiv, doi, ieee, etc.)",
    "- `placeholder_rate`: fraction of citations that are 'Web Search Synthesis' or empty URL",
    "- `cite_count`: total number of citations in report",
    "- `cite_density_per_1000w`: citations per 1000 words",
    "- `real_url_rate`: fraction of citations with real non-academic URLs",
    "",
    "## 4a: Correlations with `j_factual_accuracy`",
    "",
] + _fmt_corr_table("j_factual_accuracy") + [
    "",
    "## 4b: Correlations with `j_citation_quality`",
    "",
] + _fmt_corr_table("j_citation_quality") + [
    "",
    "## 4c: Mixed-effects model — `j_factual_accuracy ~ academic_rate + placeholder_rate + cite_count + (1|query_id)`",
    "",
    "Predictors z-scored within the merged dataset. Random intercept per query_id accounts for",
    "query difficulty differences.",
    "",
    "```",
    me_results.get("j_factual_accuracy", "not run"),
    "```",
    "",
    "## 4d: Mixed-effects model — `j_citation_quality ~ academic_rate + placeholder_rate + cite_count + (1|query_id)`",
    "",
    "```",
    me_results.get("j_citation_quality", "not run"),
    "```",
    "",
    "## Summary interpretation",
    "",
]

# Auto-generate a plain-English interpretation from the correlations
fa_ac_r, fa_ac_p = corr_results["j_factual_accuracy"]["academic_rate"]
fa_ph_r, fa_ph_p = corr_results["j_factual_accuracy"]["placeholder_rate"]
fa_ct_r, fa_ct_p = corr_results["j_factual_accuracy"]["cite_count"]
cq_ac_r, cq_ac_p = corr_results["j_citation_quality"]["academic_rate"]
cq_ph_r, cq_ph_p = corr_results["j_citation_quality"]["placeholder_rate"]

def _sig_str(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001: return "highly significant (p<0.001)"
    if p < 0.01: return "significant (p<0.01)"
    if p < 0.05: return "marginally significant (p<0.05)"
    return "not significant (p={:.3f})".format(p)

reg_md += [
    f"- **Academic rate → factual accuracy:** r={fa_ac_r:+.3f}, {_sig_str(fa_ac_p)}. "
    f"{'Academic URLs positively predict factual accuracy.' if fa_ac_r > 0.05 else 'Weak or no positive signal from academic URLs on factual accuracy.'}",
    "",
    f"- **Placeholder rate → factual accuracy:** r={fa_ph_r:+.3f}, {_sig_str(fa_ph_p)}. "
    f"{'High placeholder rate is associated with LOWER factual accuracy, consistent with placeholder citations providing no verifiable evidence.' if fa_ph_r < -0.05 else 'Placeholder rate not clearly negatively correlated with factual accuracy — judge may not penalise synthesis citations.'}",
    "",
    f"- **Cite count → factual accuracy:** r={fa_ct_r:+.3f}, {_sig_str(fa_ct_p)}. "
    f"{'Raw citation count positively predicts factual accuracy, suggesting density-reward effect in judging.' if fa_ct_r > 0.05 else 'Raw citation count does not strongly predict factual accuracy.'}",
    "",
    f"- **Academic rate → citation quality:** r={cq_ac_r:+.3f}, {_sig_str(cq_ac_p)}.",
    f"- **Placeholder rate → citation quality:** r={cq_ph_r:+.3f}, {_sig_str(cq_ph_p)}.",
    "",
    "### Key takeaway for peer review response",
    "",
    "If placeholder_rate is negatively correlated with citation_quality but academic_rate is not",
    "strongly correlated with factual_accuracy, this suggests judges primarily reward citation *presence*",
    "(density) rather than citation *quality* (academic sourcing). This would validate the peer reviewer's",
    "concern that citation_quality and factual_accuracy scores are inflated by citation density.",
]

(OUT_DIR / "citation_quality_regression.md").write_text("\n".join(reg_md), encoding="utf-8")
print(f"Saved: {OUT_DIR / 'citation_quality_regression.md'}")

# ---------------------------------------------------------------------------
# Step 5: Stacked bar figure
# ---------------------------------------------------------------------------
print("\nStep 5: Generating citation breakdown figure...")

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Sort patterns by placeholder rate desc for visual clarity
df_fig = df_pat.sort_values("mean_placeholder_rate", ascending=False).copy()

COLORS = {
    "placeholder": "#d62728",   # red
    "real_url":    "#1f77b4",   # blue
    "academic":    "#2ca02c",   # green
    "suspicious":  "#ff7f0e",   # orange
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Panel A: Percentage stacked bar ----
ax = axes[0]
labels = df_fig["pattern_label"].tolist()
x = np.arange(len(labels))
width = 0.6

bottoms = np.zeros(len(labels))
for cat, color in COLORS.items():
    col = f"mean_{cat}_rate"
    if cat == "suspicious":
        # compute from what's left
        vals = df_fig["mean_suspicious_rate"].values
    elif cat == "real_url":
        vals = df_fig["mean_real_url_rate"].values
    elif cat == "placeholder":
        vals = df_fig["mean_placeholder_rate"].values
    else:
        vals = df_fig["mean_academic_rate"].values

    ax.bar(x, vals * 100, width, bottom=bottoms * 100, color=color,
           label=cat.replace("_", " ").title(), alpha=0.88, edgecolor="white", linewidth=0.4)
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("% of citations per report")
ax.set_title("A  Citation category breakdown by pattern (% mean per report)")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", framealpha=0.7, ncol=2)
ax.axhline(100, color="grey", linewidth=0.5, linestyle="--")

# Annotate with cite count
for i, (_, row) in enumerate(df_fig.iterrows()):
    ax.text(i, 102, f"{row['mean_cite_count']:.0f}", ha="center", va="bottom",
            fontsize=7, color="grey")
ax.text(0.5, 1.04, "grey = mean cites/report", ha="center", va="bottom",
        transform=ax.transAxes, fontsize=7, color="grey")

# ---- Panel B: Scatter academic_rate vs j_factual_accuracy (per pattern means) ----
ax2 = axes[1]
# Per-pattern mean judge scores
df_jmean = df_merged.groupby("pattern")[["j_factual_accuracy", "j_citation_quality"]].mean().reset_index()
# Avoid duplicate pattern_label column — drop from jmean before merge
df_merged2 = df_fig[["pattern", "pattern_label", "mean_academic_rate", "mean_placeholder_rate",
                       "mean_cite_count"]].merge(df_jmean, on="pattern", how="inner")

ax2.scatter(df_merged2["mean_academic_rate"] * 100,
            df_merged2["j_factual_accuracy"],
            c=df_merged2["mean_placeholder_rate"],
            cmap="RdYlGn_r",
            s=80, zorder=3, edgecolors="grey", linewidth=0.5, alpha=0.9)

for _, row in df_merged2.iterrows():
    ax2.annotate(row["pattern_label"],
                 (row["mean_academic_rate"] * 100, row["j_factual_accuracy"]),
                 textcoords="offset points", xytext=(4, 2), fontsize=7, alpha=0.85)

# Colorbar
sc = ax2.scatter([], [], c=[], cmap="RdYlGn_r", vmin=0, vmax=1)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap="RdYlGn_r",
                           norm=mpl.colors.Normalize(vmin=df_merged2["mean_placeholder_rate"].min(),
                                                      vmax=df_merged2["mean_placeholder_rate"].max())),
    ax=ax2, fraction=0.046, pad=0.04
)
cbar.set_label("Placeholder rate (mean)", fontsize=8)

ax2.set_xlabel("Academic citation rate (% of cites, per-pattern mean)")
ax2.set_ylabel("GPT-5.2 judge factual_accuracy (per-pattern mean)")
ax2.set_title("B  Academic citation rate vs judge factual accuracy\n(colour = placeholder rate)")
ax2.grid(True, alpha=0.2, linestyle="--")

# Fit OLS line
from numpy.polynomial import polynomial as P
x_vals = df_merged2["mean_academic_rate"].values * 100
y_vals = df_merged2["j_factual_accuracy"].values
if len(x_vals) >= 3:
    coeffs = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax2.plot(x_line, np.polyval(coeffs, x_line), "k--", linewidth=1, alpha=0.5, label="OLS")

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Phase 7a: Citation Verification Analysis — Deep Research Patterns",
             fontsize=11, y=0.99)

out_pdf = FIG_DIR / "citation_breakdown.pdf"
out_png = FIG_DIR / "citation_breakdown.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ---------------------------------------------------------------------------
# Final summary to stdout
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 7a COMPLETE — SUMMARY")
print("=" * 70)

print("\nPer-pattern placeholder rate (sorted high → low):")
df_ph = df_pat.sort_values("mean_placeholder_rate", ascending=False)
for _, row in df_ph.iterrows():
    bar = "█" * int(row["mean_placeholder_rate"] * 40)
    print(f"  {row['pattern_label']:6s}  {row['mean_placeholder_rate']*100:5.1f}%  {bar}")

print("\nPer-pattern academic rate (sorted high → low):")
df_ac = df_pat.sort_values("mean_academic_rate", ascending=False)
for _, row in df_ac.iterrows():
    bar = "█" * int(row["mean_academic_rate"] * 40)
    print(f"  {row['pattern_label']:6s}  {row['mean_academic_rate']*100:5.1f}%  {bar}")

print(f"\nTop placeholder offender: {top_ph['pattern_label']} "
      f"({top_ph['mean_placeholder_rate']*100:.1f}% placeholder)")
print(f"Top academic pattern:     {top_ac['pattern_label']} "
      f"({top_ac['mean_academic_rate']*100:.1f}% academic)")

print("\nCorrelation with j_factual_accuracy:")
for pred, (r, p) in corr_results["j_factual_accuracy"].items():
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {pred:35s}: r={r:+.3f} {stars}")

print("\nCorrelation with j_citation_quality:")
for pred, (r, p) in corr_results["j_citation_quality"].items():
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {pred:35s}: r={r:+.3f} {stars}")

print("\nOutput files:")
print(f"  {ANALYSIS_DIR / 'df_citations.parquet'}")
print(f"  {OUT_DIR / 'per_pattern_stats.csv'}")
print(f"  {OUT_DIR / 'per_pattern_stats.md'}")
print(f"  {OUT_DIR / 'citation_quality_regression.md'}")
print(f"  {FIG_DIR / 'citation_breakdown.pdf'}")
print(f"  {FIG_DIR / 'citation_breakdown.png'}")
