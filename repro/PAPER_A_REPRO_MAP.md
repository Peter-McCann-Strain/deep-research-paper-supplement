# Paper A Public Reproducibility Map

Use this page to choose the right command and to see which paper artifact it checks. The public checkout supports integrity checks, reference inspection, compatible metric comparison, Paper A artifact rebuilds, and current-API demos. It does not bitwise replay the archived raw experiment.

## Public Commands

| Command | Paid APIs | Purpose | Comparable to paper metrics? |
|---|---:|---|---|
| `deep-research reproduce paper-a --mode smoke` | No | Verifies that reference summaries and public inputs are present. | No; integrity check only. |
| `deep-research reproduce paper-a --mode reference` | No | Prints the frozen headline reference: 90 queries, 13 patterns, primary `mean_3judge` ordering, and comparison policy. | Yes; this is the compact reference view. |
| `deep-research reproduce paper-a --mode provenance` | No | Verifies hashes and counts for the public query, headline, and pattern-metric reference files. | Yes; artifact integrity check only. |
| `deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv` | No | Validates the shipped compact pattern-by-judge metrics CSV against the frozen reference JSON. | Yes; pattern-level audit only. |
| `deep-research paper rebuild paper-a --check-only` | No | Verifies that the Paper A rebuild source, canonical store, derived tables, figures, and LaTeX inputs are present. | Yes; rebuild input integrity check. |
| `deep-research paper rebuild paper-a --skip-compile` | No | Regenerates public tables and figures from the shipped canonical store and `data/analysis/`. | Yes; paper artifact rebuild from public derived data. |
| `deep-research paper rebuild paper-a` | No provider APIs | Also compiles `paper_rebuild/paper_a_bounded_returns/main.tex` when `tectonic` is installed. | Yes; manuscript artifact rebuild. |
| `deep-research doctor --verify-api` | Yes | Makes a tiny live generation call to verify current OpenAI/Azure hosted-search entitlement. | No; access check only. |
| `deep-research cost paper-a --full --judge` | No | Estimates call counts and configurable budget guardrails for a full live API rerun. | No; planning only. |
| `deep-research reproduce paper-a --mode api-best-effort --execute --limit N` | Yes | Generates current OpenAI/Azure hosted-search reports for public queries without model downloads and verifies `web_search_call`. | No; this is a live API demo, not the frozen 13-pattern matrix. |
| `deep-research reproduce paper-a --mode api-best-effort --execute --full --judge` | Yes | Generates 90 live reports and scores successful reports with OpenAI plus Anthropic API judges. | Partially; useful for qualitative drift checks, but not a historical equality claim. |
| `deep-research compare paper-a --run-summary RUN.json` | No | Compares candidate pattern-level metrics with the frozen public reference and fails on large metric or rank-order divergence. | Yes only when `RUN.json` contains pattern-level metrics. API-demo summaries fail with a clear not-comparable status. |

## Frozen Reference

- Reference file: `repro/reference/paper_a_headline_numbers.json`
- Compact metrics CSV: `repro/reference/paper_a_pattern_metrics.csv`
- Query count: 90
- Pattern count: 13
- Primary metric: `mean_3judge`
- Top public reference pattern: `base_p1`
- Lowest public reference pattern: `base_p12`

The paper's narrative often focuses on six headline orchestration archetypes (`P0`--`P5`). The compact public table has thirteen `base_p*` rows because it also includes P6 Reactive, P7 Graph, P8 Beam, the local-7B controls P9/P10, and the post-hoc single-judge probes P11/P12. `repro/reference/PATTERN_DICTIONARY.csv` gives the `paper_archetype`, `variant_kind`, and notes for every row.

The public manifest has 90 queries. `n_queries` in the compact metrics CSV is the archived count of scored observations for that pattern, so rows with 87 or 89 observations reflect missing historical outputs or judge cells. Blank judge cells mean that judge metric was unavailable in the archived aggregate for that row; they should not be read as a live API failure.

## Public Data

- `data/eval_queries_v2.json`: compact public query set.
- `data/analysis/`: compact derived analysis tables used by the public paper rebuild.
- `data/public_judge_criteria.json`: small standalone judge-smoke criteria; integrated `--judge` uses each query's full bundled rubric.
- `data/README.md` and `data/DATA_DICTIONARY.md`: sources, licenses, and field definitions.
- `repro/SCRIPT_CATALOG.csv`: one-row-per-script map explaining whether each top-level `scripts/` file is a supported helper, optional workflow, historical analysis helper, local-model/GPU workflow, external download, worker, or raw-artifact rebuild helper.
- `repro/PAPER_A_ARTIFACT_INDEX.md`: one-page map from the manuscript to the shipped tables, figures, producers, and derived data files.
- `docs/huggingface_release.md` and `repro/HUGGINGFACE_DATASET_CARD.md`: Hugging Face dataset-mirror publishing instructions and card text.

## Public Paper Rebuild

- `paper_rebuild/paper_a_bounded_returns/main.tex`: manuscript source for artifact rebuilds.
- `paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json`: canonical number store consumed by tables, figures, and prose checks.
- `paper_rebuild/paper_a_bounded_returns/analysis/`: paper-specific table, figure, statistical, and provenance scripts.
- `paper_rebuild/paper_a_bounded_returns/figures/` and `tables/`: generated assets consumed by the manuscript.
- `papers/paper_a_bounded_returns/main.pdf`: checked final PDF for convenient citation and browsing.

For a table/figure-level map, start from `repro/PAPER_A_ARTIFACT_INDEX.md`. It also points to `data/analysis/coverage_report.md`, which explains known missing archived cells and why some scored-observation counts differ from the 90 public queries.

## Excluded From Public GitHub

Private notes, local agent memory, generated report forests, raw judge verdict trees, caches, model weights, checkpoints, paper drafts, submission bundles, and outreach material stay out of GitHub. The final manuscript source needed for artifact rebuilds is included under `paper_rebuild/`. `deep-research export-public` enforces the allowlist and writes `PUBLIC_EXPORT_REPORT.json` with file hashes and provenance.

## Source Scope

The source tree ships the API-backed reproduction, judging, comparison, export, audit, settings, compact inputs, tests, final PDF, pattern implementations, evaluation modules, script catalog, and paper artifact rebuild package. Optional local-model and GPU experiment code is included for provenance, but the supported public reproduction path is no-model-download and API-only unless a user deliberately opts into optional local experiments.

## Comparable Candidate Schema

A comparable run summary should contain either `primary_ordering`, `pattern_metrics`, or `metrics_by_pattern`. Each pattern row must include a pattern name and numeric `mean_3judge` or equivalent `score` field. A full `success` also requires all reference metric cells available for the overlapping patterns; summaries with only a primary score are reported as `partial`. Example:

```json
{
  "primary_ordering": [
    {"pattern": "base_p1", "mean_3judge": 0.67, "mean_gpt52": 0.46, "mean_opus": 0.76, "mean_sonnet_corrected": 0.79, "ppi_debiased_mean": 0.67},
    {"pattern": "base_p4", "mean_3judge": 0.64, "mean_gpt52": 0.46, "mean_opus": 0.64, "mean_sonnet_corrected": 0.81, "ppi_debiased_mean": 0.64}
  ]
}
```
