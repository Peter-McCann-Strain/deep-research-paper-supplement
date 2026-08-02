# Paper A Artifact Index

This index links the public paper supplement to the files that rebuild or verify
the manuscript. It is the starting point when checking whether a claim is backed
by included source, included derived data, or an excluded archival input.

## Public Rebuild Contract

| Artifact | Public path | Producer or command | Inputs | Notes |
|---|---|---|---|---|
| Paper source | `paper_rebuild/paper_a_bounded_returns/main.tex` | edited source, compiled by `deep-research paper rebuild paper-a` | generated `figures/`, `tables/`, bibliography | Final scrubbed source only; drafts and submission bundles are excluded. |
| Paper PDF | `papers/paper_a_bounded_returns/main.pdf` | `deep-research paper rebuild paper-a` | `paper_rebuild/paper_a_bounded_returns/main.pdf` | Convenience copy for citation and browsing. |
| Canonical number store | `paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json` | archived analysis builders plus public table/figure scripts | compact staging blobs and `data/analysis/` | This is the authoritative shipped store for prose/table/figure rebuilds. |
| Capability isoquant and claim-type block | `analysis/build_isoquant_claimtype.py` | `python paper_rebuild/paper_a_bounded_returns/analysis/build_isoquant_claimtype.py --write` | `analysis/staging/isoquant_claimtype.json` | Rebuilds the canonical block that backs the claim-type oracle-lift paragraph from included compact staging data. |
| Rebuild preflight | `deep-research paper rebuild paper-a --check-only` | `deep_research/paper_rebuild.py` | required public sources and generated assets | No provider APIs and no TeX toolchain required. |
| Tables and figures | `deep-research paper rebuild paper-a --skip-compile` | `deep_research/paper_rebuild.py` | `canonical_numbers.json`, `data/analysis/*.parquet` | Regenerates public assets without compiling the PDF. |

## Figures Referenced By `main.tex`

| Figure asset | Producer | Main public inputs |
|---|---|---|
| `figures/fig1_money.pdf` | `analysis/make_money_figure.py` | `analysis/canonical_numbers.json` |
| `figures/fig_judge_gold.pdf` | `analysis/make_judge_gold_figure.py` | `analysis/canonical_numbers.json`, staged judge/gold summaries |
| `figures/fig_cd_clean.pdf` | `analysis/make_cd_diagram.py` | `analysis/canonical_numbers.json` |
| `figures/fig_stratification.pdf` | `analysis/make_stratification_figure.py` | `analysis/canonical_numbers.json`, `data/analysis/df_scores.parquet` |
| `figures/fig_vintage.pdf` | `analysis/make_vintage_figure.py` | `analysis/canonical_numbers.json`, staged vintage summary |
| `figures/fig_cost.pdf` | `analysis/make_cost_figure.py` | `analysis/canonical_numbers.json` |
| `figures/fig_disentanglement.pdf` | `analysis/make_disentanglement_figure.py` | `analysis/canonical_numbers.json`, staged disentanglement summary |
| `figures/fig_e5_dose_response.pdf` | `analysis/make_e5_dose_response.py` | `analysis/canonical_numbers.json`, staged E5 dose summary |
| `figures/fig_oracle.pdf` | `analysis/make_oracle_figure.py` | `analysis/canonical_numbers.json`, staged oracle summary |

PNG siblings are included for inspection where generated. The PDF assets are
the versions consumed by the manuscript.

## Tables Referenced By `main.tex`

| Table asset | Producer | Main public inputs |
|---|---|---|
| `tables/tab_headline_means.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_per_dimension.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_irr.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_verdicts.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json`, `data/analysis/df_verdicts.parquet` |
| `tables/tab_citations.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json`, `data/analysis/df_citations*.parquet` |
| `tables/tab_per_source.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_bing_tavily.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_drjudge.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_ablations.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_single_judge.tex` | `analysis/make_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_p2_neff.tex` | `analysis/make_paper2_tables.py` | `analysis/canonical_numbers.json` |
| `tables/tab_b2.tex` | `analysis/make_b2_bestofn_tables.py` | `analysis/canonical_numbers.json`, `data/b2_subset.json` |
| `tables/tab_bestofn_decoupled.tex` | `analysis/make_b2_bestofn_tables.py` | `analysis/canonical_numbers.json`, `data/b2_subset.json` |

Additional generated appendix tables in `tables/` are retained because they are
useful for inspection and were part of the final paper-support package, even
when they are not directly `\input{}` by `main.tex`.

## Shipped Derived Data

The compact derived data lives in `data/analysis/` and is documented in
`data/analysis/DATA_DICTIONARY.md`:

| File family | Use |
|---|---|
| `df_queries.parquet` | Query metadata, strata, and public identifiers. |
| `df_runs.parquet`, `df_scores.parquet`, `df_overall_scores.parquet` | Pattern/report-level score analysis. |
| `df_verdicts.parquet` | Criterion-level verdict analysis. |
| `df_citations*.parquet` | Citation extraction and citation-faithfulness analyses. |
| `df_c0_*.parquet` | C0 verification support tables. |
| `df_e14_oracle_*.parquet` | Oracle-entailment support tables. |
| `data/analysis/build_manifest.json`, `data/analysis/coverage_report.md` | Public coverage, hash, and known-missing-cell audit. |

`coverage_report.md` is the place to check why some archived pattern rows have
fewer scored observations than the 90 public queries.

## Excluded Archival Inputs

The public GitHub supplement intentionally excludes raw generated report
forests, raw judge-verdict packet directories, caches, model weights,
checkpoints, paper drafts, submission bundles, and personal working notes.
Those inputs are not needed for the public artifact rebuild above. They would
be needed only for a bitwise replay of the historical raw experiment under the
original provider/search snapshots.
