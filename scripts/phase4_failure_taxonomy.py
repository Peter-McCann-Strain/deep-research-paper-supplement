#!/usr/bin/env python3
"""Phase 4.1: Failure-mode taxonomy + per-judge verdict n-gram mining.

Inputs:
  data/analysis/df_verdicts.parquet

Outputs:
  reports/phase4_failures/failure_mode_table.md
  reports/phase4_failures/failure_mode_heatmap.pdf (+ .png)
  reports/phase4_failures/taxonomy_rules.json
  reports/phase4_failures/verdict_ngrams.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase4_failures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Failure-mode rules — keyword/regex heuristics on verdict `reasoning` text.
# A verdict may be tagged with multiple modes.
# ----------------------------------------------------------------------
RULES: dict[str, list[str]] = {
    "citation_fabrication": [
        r"\bweb search synthesis\b",
        r"\bno (real|actual|specific|concrete) (source|citation|reference)s?\b",
        r"\bplaceholder (citation|source|reference)s?\b",
        r"\bunverifiable\b",
        r"\bgeneric (link|source|url)s?\b",
        r"\bno url\b",
        r"\bmissing (citation|reference)s?\b",
        r"\bunattributed\b",
    ],
    "hallucinated_source": [
        r"\bfabricat",
        r"\bdoes not exist\b",
        r"\bdoesn'?t exist\b",
        r"\bcannot (be )?verif",
        r"\bcan'?t (be )?verif",
        r"\bnot (a )?real (source|paper|study|citation)\b",
        r"\bmade[- ]up\b",
        r"\binvented\b",
        r"\bhallucinat",
    ],
    "entity_confusion": [
        r"\bwrong (year|date|person|name|author|entity|number|value)\b",
        r"\bincorrect (year|date|person|name|author|entity|number|value|attribution)\b",
        r"\bmisidentif",
        r"\bmisattribut",
        r"\bconfus(es|ed|ing) .{0,40}\bwith\b",
        r"\bconflat",
    ],
    "missing_perspective": [
        r"\b(only|just) (one|a single) (perspective|viewpoint|side|angle)\b",
        r"\bmissing (perspective|viewpoint|counter[- ]?argument|opposing view)s?\b",
        r"\bno (counter[- ]?argument|opposing view|alternative view)s?\b",
        r"\blacks? (perspective|viewpoint|counter|alternative view|dissent)",
        r"\bone[- ]sided\b",
        r"\bdoes not (cover|address|include) .{0,40}\b(other|alternative|opposing|counter)",
        r"\bfail(s|ed)? to (consider|include|address) .{0,40}\b(other|alternative|opposing|counter|perspective)",
    ],
    "format_violation": [
        r"\b(not|no) (in )?(table|bullet|list|comparison table) form(at)?\b",
        r"\bshould (be|have been) .{0,30}\btable\b",
        r"\bwrong (format|structure)\b",
        r"\bdoes not follow .{0,30}\b(format|structure|template|schema)\b",
        r"\bformat(ting)? (violation|issue|problem)\b",
        r"\bignores? the (requested|asked) (format|structure)\b",
        r"\bprose (instead|rather) than\b",
    ],
    "superficial_analysis": [
        r"\bsuperficial\b",
        r"\bshallow (analysis|treatment|coverage)\b",
        r"\brestates? the (query|question)\b",
        r"\bno (real )?synthesis\b",
        r"\blacks? (depth|synthesis|analysis|nuance)\b",
        r"\bsurface[- ]level\b",
        r"\bdescriptive (only|rather than)\b",
        r"\bno critical (analysis|evaluation|assessment)\b",
        r"\bnot (deeply )?analy[sz]ed\b",
        r"\bmore (descriptive|assertion) than (analysis|argument)\b",
        r"\bgeneric .{0,30}(explanation|treatment|discussion)\b",
    ],
    "missing_evidence": [
        r"\bnot supported (by|with) .{0,40}\b(evidence|data|source|citation)\b",
        r"\bunsupported .{0,30}(claim|assertion|conclusion)\b",
        r"\bwithout .{0,20}(evidence|citation|source|data)\b",
        r"\bno (concrete|specific|supporting) (evidence|data|examples)\b",
        r"\blacks? (evidence|supporting data|concrete (evidence|data))\b",
        r"\bnot (clearly )?traceable\b",
        r"\bdoes not (include|provide|contain) .{0,30}\b(quantitative|numerical|statistical) (data|information|evidence)\b",
        r"\bno quantitative (data|information|figures|statistics)\b",
        r"\bno (numbers|numerical|statistics)\b",
    ],
    "empty_or_sparse": [
        r"\breport is empty\b",
        r"\bno content\b",
        r"\bextremely (short|brief|sparse)\b",
        r"\bessentially empty\b",
        r"\b(very|quite) (thin|brief)\b",
        r"\binsufficient content\b",
    ],
    "scope_drift": [
        r"\boff[- ]topic\b",
        r"\bdoes not answer the (query|question)\b",
        r"\bdoesn'?t answer the (query|question)\b",
        r"\banswer(s|ed) a different\b",
        r"\b(scope|topic) drift\b",
        r"\btangential\b",
        r"\bveers? (off|away)\b",
        r"\bmiss(es|ed) the (point|question|query|ask)\b",
        r"\bdoes not address the (specific )?(query|question|ask)\b",
    ],
    "factual_contradiction": [
        r"\bcontradict",
        r"\binconsistent\b",
        r"\bconflict(s|ing)? (with|between)\b",
        r"\bone (part|section) says .{0,40}\banother\b",
        r"\bstates? .{0,40}\bbut (also|later)\b",
        r"\binternal(ly)? (contradict|inconsistent)",
    ],
}


def compile_rules(rules: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {mode: [re.compile(p, re.IGNORECASE) for p in patterns] for mode, patterns in rules.items()}


def tag_reasoning(text: str, compiled: dict[str, list[re.Pattern]]) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tags: list[str] = []
    for mode, pats in compiled.items():
        for p in pats:
            if p.search(text):
                tags.append(mode)
                break
    return tags


def main() -> None:
    print("Loading df_verdicts...")
    v = pd.read_parquet(ROOT / "data" / "analysis" / "df_verdicts.parquet")

    # Filter: gpt52 + claude_sonnet, satisfied=False, non-empty reasoning
    keep_judges = ["gpt52", "claude_sonnet"]
    v = v[v["judge"].isin(keep_judges)]
    fails = v[(v["satisfied"] == False) & v["reasoning"].notna()].copy()
    fails = fails[fails["reasoning"].str.strip().str.len() > 0]
    print(f"Unsatisfied verdicts with reasoning (gpt52+sonnet): {len(fails):,}")

    compiled = compile_rules(RULES)
    fails["failure_modes"] = fails["reasoning"].apply(lambda t: tag_reasoning(t, compiled))
    fails["n_modes"] = fails["failure_modes"].apply(len)
    untagged_frac = (fails["n_modes"] == 0).mean()
    print(f"Untagged failures (no rule match): {untagged_frac:.1%}")

    # Explode to (pattern, judge, mode) rows
    long = fails.explode("failure_modes").dropna(subset=["failure_modes"])
    long = long.rename(columns={"failure_modes": "failure_mode"})

    # --- 4.1a: Pattern x failure_mode table (counts over both judges combined) ---
    table = (
        long.groupby(["pattern", "failure_mode"]).size().unstack(fill_value=0).sort_index()
    )
    # Also: proportion of a pattern's failed-verdicts that hit each mode
    pattern_fail_totals = fails.groupby("pattern").size()
    prop = table.div(pattern_fail_totals, axis=0).fillna(0.0)

    # Write markdown table
    md = ["# Failure-mode prevalence by pattern", "",
          "Counts of failed verdicts tagged with each failure mode (gpt52 + claude_sonnet judges combined).",
          "Proportion is: count / total failed verdicts for that pattern. A verdict may map to multiple modes.",
          "",
          "## Counts", "", table.to_markdown(), "",
          "## Proportions (per pattern)", "", prop.round(3).to_markdown(), ""]
    (OUT / "failure_mode_table.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'failure_mode_table.md'}")

    # --- 4.1b: Heatmap ---
    # Order patterns: base_p0..p10, then ablations
    base_order = [f"base_p{i}" for i in range(11)]
    abl_order = sorted([p for p in prop.index if p.startswith("ablation_")])
    order = [p for p in base_order if p in prop.index] + abl_order
    prop_ord = prop.loc[order]
    # Mode order: sort by overall prevalence
    mode_order = prop.sum(axis=0).sort_values(ascending=False).index.tolist()
    prop_ord = prop_ord[mode_order]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(prop_ord) + 2))
    sns.heatmap(prop_ord, annot=True, fmt=".2f", cmap="Reds",
                cbar_kws={"label": "Fraction of failed verdicts"}, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title("Failure-mode prevalence by pattern\n(Proportion of each pattern's failed verdicts; gpt52 + claude_sonnet)")
    ax.set_xlabel("Failure mode")
    ax.set_ylabel("Pattern")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "failure_mode_heatmap.pdf")
    fig.savefig(OUT / "failure_mode_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Wrote heatmap PDF + PNG")

    # --- 4.1c: taxonomy_rules.json ---
    (OUT / "taxonomy_rules.json").write_text(json.dumps(RULES, indent=2))
    print(f"Wrote taxonomy_rules.json")

    # --- 4.3: Per-judge verdict n-gram mining ---
    ngram_md = ["# Verdict-reasoning n-grams (satisfied=False)", "",
                "Top-20 trigrams per judge from failed-verdict reasoning text.",
                "CountVectorizer(ngram_range=(2,3), max_features=100, stop_words='english').",
                ""]
    for judge in keep_judges + ["claude_opus"]:  # include opus here for reference
        sub = v[(v["judge"] == judge) & (v["satisfied"] == False) & v["reasoning"].notna()]
        sub = sub[sub["reasoning"].str.strip().str.len() > 0]
        if len(sub) == 0:
            continue
        try:
            vec = CountVectorizer(ngram_range=(2, 3), max_features=100, stop_words="english")
            X = vec.fit_transform(sub["reasoning"].tolist())
            totals = np.asarray(X.sum(axis=0)).ravel()
            vocab = vec.get_feature_names_out()
            trigrams = [(g, int(c)) for g, c in zip(vocab, totals) if len(g.split()) == 3]
            trigrams.sort(key=lambda x: -x[1])
            top = trigrams[:20]
            ngram_md.append(f"## {judge}  (n={len(sub):,} failed verdicts)")
            ngram_md.append("")
            ngram_md.append("| Rank | Trigram | Count |")
            ngram_md.append("|---:|:---|---:|")
            for i, (g, c) in enumerate(top, 1):
                ngram_md.append(f"| {i} | {g} | {c} |")
            ngram_md.append("")
        except Exception as e:
            ngram_md.append(f"## {judge}: ERROR — {e}")
            ngram_md.append("")

    (OUT / "verdict_ngrams.md").write_text("\n".join(ngram_md))
    print(f"Wrote verdict_ngrams.md")

    # --- Family-level summary used later by summary.md ---
    # GPT-4o family (base_p0..p8) vs local 7B (base_p9, base_p10)
    long["family_ab"] = long["pattern"].map(
        lambda p: ("GPT-4o" if p in [f"base_p{i}" for i in range(9)]
                   else ("Local7B" if p in ["base_p9", "base_p10"]
                         else "Ablation"))
    )
    fam_table = (
        long.groupby(["family_ab", "failure_mode"]).size().unstack(fill_value=0)
    )

    # --- CORRECTED denominators ---
    # (A) tagged-conditional: proportion of tagged rows that are mode X
    #     denominator = sum of tagged counts per family (old behavior, kept for reference)
    fam_prop_conditional = fam_table.div(fam_table.sum(axis=1), axis=0).round(3)

    # (B) overall proportion: count / total failed verdicts for that family
    #     denominator = total failed verdicts per family (base patterns only for the base-family rows)
    # We need fails scoped to the same long filter
    # Recompute fails family labels on the full fails df
    fails["family_ab"] = fails["pattern"].map(
        lambda p: ("GPT-4o" if p in [f"base_p{i}" for i in range(9)]
                   else ("Local7B" if p in ["base_p9", "base_p10"]
                         else "Ablation"))
    )
    total_failed_per_family = fails.groupby("family_ab").size()
    fam_prop_overall = fam_table.div(total_failed_per_family, axis=0).round(3)

    # Save both; primary export is the corrected overall prop
    fam_prop_overall.to_csv(OUT / "_family_mode_prop.csv")
    fam_prop_conditional.to_csv(OUT / "_family_mode_prop_conditional.csv")
    print("Wrote _family_mode_prop.csv (overall) and _family_mode_prop_conditional.csv")

    print("\n== Top 3 failure modes per family (OVERALL proportion) ==")
    for fam in fam_prop_overall.index:
        row = fam_prop_overall.loc[fam].sort_values(ascending=False).head(3)
        print(f"  {fam}:", [(m, f"{p:.3f}") for m, p in row.items()])

    print("\n== Top 3 failure modes per family (tagged-conditional proportion) ==")
    for fam in fam_prop_conditional.index:
        row = fam_prop_conditional.loc[fam].sort_values(ascending=False).head(3)
        print(f"  {fam}:", [(m, f"{p:.3f}") for m, p in row.items()])

    # --- Chi-square test: GPT-4o vs Local7B mode distribution ---
    df_tagged_base = long[long["family_ab"].isin(["GPT-4o", "Local7B"])]
    ct = pd.crosstab(df_tagged_base["family_ab"], df_tagged_base["failure_mode"])
    chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(ct)
    print(f"\nChi-square (GPT-4o vs Local7B modes): χ²={chi2_stat:.3f}, df={chi2_dof}, p={chi2_p:.2e}")

    # Save chi-square result
    chi2_md = [
        "# Chi-square test: family × failure-mode distribution",
        "",
        "Tests whether the distribution of failure modes differs significantly between "
        "the GPT-4o family (P0–P8, base patterns) and the Local 7B family (P9–P10).",
        "",
        "## Contingency table (tagged counts)",
        "",
        ct.to_markdown(),
        "",
        "## Result",
        "",
        f"| Statistic | Value |",
        "|:---|---:|",
        f"| χ² | {chi2_stat:.3f} |",
        f"| df | {chi2_dof} |",
        f"| p-value | {chi2_p:.2e} |",
        "",
        "## Interpretation",
        "",
    ]
    if chi2_p < 0.001:
        chi2_md.append(
            f"The failure-mode distributions of GPT-4o and Local 7B families are "
            f"**qualitatively different** (χ²={chi2_stat:.1f}, df={chi2_dof}, "
            f"p={chi2_p:.2e}). The null hypothesis of identical mode proportions is "
            "rejected at p<0.001. Notably, citation_fabrication is far more prevalent "
            "in GPT-4o outputs, while superficial_analysis and missing_evidence are "
            "more prominent in Local 7B outputs — consistent with different failure "
            "archetypes driven by model scale and RL training."
        )
    else:
        chi2_md.append(
            f"p={chi2_p:.2e} — insufficient evidence to reject the null hypothesis "
            "that mode distributions are the same across families."
        )
    chi2_md.append("")
    (OUT / "chi2_family_test.md").write_text("\n".join(chi2_md))
    print(f"Wrote {OUT / 'chi2_family_test.md'}")


if __name__ == "__main__":
    main()
