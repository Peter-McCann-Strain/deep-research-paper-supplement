# Data Dictionary — data/analysis/*.parquet

This dictionary documents every shipped compact derived parquet table under
`data/analysis/`. It is public supplementary material for the Paper A artifact
rebuild. The raw generated reports, raw judge verdict forests, and local caches
used to build these compact tables are intentionally not shipped.

Paths recorded in provenance fields are repository-relative historical pointers
into the unshipped raw artifact layout. They are not absolute local paths and
are not expected to resolve in a public checkout.

## Conventions

- `pattern`: e.g. `base_p4`, `ablation_p3_no_quality_eval`.
- `pattern_family`: `"base"`, `"ablation"`, `"protocol_a"`, `"variance"`, or `"disentanglement"`.
- `pattern_short`: suffix after `base_` / `ablation_`, e.g. `p4`, `p3_no_quality_eval`.
- `query_id`: unique id from `data/eval_queries_v2.json`.
- `judge`: archived provider/model label such as `gpt52`, `claude_opus`,
  `claude_sonnet`, or `claude_code`. The `claude_code` label is retained only
  for frozen-data provenance; the supported public judge workflow calls the
  Anthropic API directly and does not require Claude Code or local assistant
  sessions.
- `dimension`: one of the 9 rubric-v2 dimensions listed below.

## Rubric v2 dimensions

`information_recall`, `factual_accuracy`, `coverage`, `analytical_depth`,
`citation_quality`, `logical_coherence`, `organization`, `instruction_following`,
`attribution_quality`. Weights are query- and source-dependent; see
`build_manifest.json` for the canonical hash.

## df_queries.parquet

| column | dtype | description | source |
|---|---|---|---|
| query_id | str | Unique query identifier | `data/eval_queries_v2.json` |
| source | category | Benchmark source: `custom`, `draco`, `deepsearch_qa`, `research_qa`, `litqa2` | manifest |
| domain | str | Query subject domain (free-form) | manifest |
| difficulty | category | `simple`, `moderate`, or `complex` from the public manifest | manifest |
| query_text | str | Natural-language query | manifest |
| expected_topics | list[str] | Expected coverage elements | manifest `expected_elements` |
| gold_answer | str | Reference/gold answer if known | manifest `reference_answer` |

## df_runs.parquet

One row per (pattern × query_id) whether or not a report exists.

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | Pattern directory name | historical `results/experiments/` layout |
| pattern_family | category | `base`, `ablation`, `protocol_a`, `variance`, or `disentanglement` | derived |
| pattern_short | category | Suffix after prefix, e.g. `p4` | derived |
| query_id | str | FK to df_queries | manifest |
| status | category | `success` / `failed` / `missing_checkpoint` / `missing_status` | checkpoint JSON (see Issue 5) |
| elapsed_seconds | float64 | Wall-clock seconds for the run | checkpoint |
| total_tokens | float64 | Total LLM tokens consumed | checkpoint |
| total_cost_usd | float64 | Cost recorded by upstream caller (may be missing for local models) | checkpoint |
| sections | float64 | Report section count | checkpoint |
| citations | float64 | Citations emitted | checkpoint |
| timestamp | datetime64 | Run timestamp | checkpoint |
| report_path | str \| null | Repository-relative historical pointer to the `.md` report in the unshipped raw artifact layout | filesystem provenance |
| report_exists | bool | True iff `.md` report file is present | filesystem |
| word_count_is_present | bool | Alias of `report_exists` for explicit gating | filesystem |
| report_word_count | float64 | Word count of `.md`; **NaN when report is missing** (not 0) | filesystem |
| cost_proxy_usd | float64 | GPT-4o patterns: `total_tokens * $5/M`. Local 7B (p9, p10, p12): `elapsed_seconds * $0.0001/sec`. | derived |
| excluded_from_analysis | bool | True for patterns in the exclusion set (currently `ablation_p5_no_citation_verify`, only 2/90 reports) | derived |
| elapsed_is_suspect | bool | True if `elapsed_seconds > 2 * median(elapsed_seconds)` within the same pattern | derived |

## df_scores.parquet

One row per (pattern × query × judge × dimension).

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| dimension | category | One of 9 rubric-v2 dimensions | judge JSON |
| score | float64 | Per-dimension score in [0, 1] | judge JSON `dimensions[dim].score` |
| met | Int64 | Criteria met (normalized across upstream schema variants) | judge JSON |
| total | Int64 | Criteria total (normalized across upstream schema variants) | judge JSON |

## df_overall_scores.parquet

One row per (pattern × query × judge).

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| overall_score | float64 | Stored top-level overall score | judge JSON |
| overall_score_recomputed | float64 | Recomputed using `DIMENSION_WEIGHTS_BY_SOURCE[query.source]` applied to per-dimension scores (falls back to met/total when score is missing) | derived |
| overall_score_per_query_weights | float64 | Recomputed using the per-query `rubric.dimension_weights` from the manifest | derived |
| overall_score_trustworthy | bool | **False for claude_sonnet** — its stored `overall_score` is upstream-corrupted. Downstream analyses MUST use `overall_score_recomputed` or `overall_score_per_query_weights` for those rows. True for gpt52, claude_opus, and the archived `claude_code` provenance label. | policy |
| n_criteria | Int64 | Criteria total across dimensions | judge JSON |
| n_satisfied | Int64 | Criteria satisfied across dimensions | judge JSON |
| judge_tokens | Int64 | Tokens consumed by the judge (gpt52 only) | judge JSON |
| judge_latency_s | float64 | Judge call latency (gpt52 only) | judge JSON |

## df_verdicts.parquet

One row per individual criterion verdict.

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| criterion_index | Int64 | Index within the judge's verdict list | judge JSON |
| dimension | category | Rubric dimension | judge JSON |
| criterion | str | **Normalized** criterion text. Claude Opus uses varied keys — `criterion`, `description`, `text`, `criterion_text`; all are normalized to this column. | judge JSON |
| criterion_id | str | 12-char md5 of `normalize(criterion)` where `normalize = lowercase + strip + whitespace-collapse`. Stable across wording jitter. | derived |
| satisfied | bool | True/False. For claude_opus rows that used `verdict: "SATISFIED"/"NOT_SATISFIED"`, the string is mapped to bool. If the verdict could not be parsed, value is False and `satisfied_is_known` is False. | judge JSON |
| satisfied_is_known | bool | False when neither `satisfied` nor `verdict` could be extracted. Filter on this for trusted analyses. | derived |
| evidence | str | Evidence quote from report | judge JSON |
| reasoning | str | Judge reasoning (from `reasoning` or `reason`) | judge JSON |

## Additional shipped parquet tables

| file | rows | role | producer / source | paper use |
|---|---:|---|---|---|
| `df_citations.parquet` | 22,903 | Citation extraction table for base and analysis patterns | Historical generated reports plus citation extraction scripts | Citation tables, citation-density analyses, oracle/source robustness checks |
| `df_citations_protocol_a.parquet` | 6,241 | Protocol A citation extraction table with backend and short-pattern fields | Protocol A generated reports and citation extraction scripts | Bing/Tavily robustness and citation-volume analyses |
| `df_c0_per_report.parquet` | 269 | Claim-verification aggregate per report | C0 citation/factuality verifier outputs | Claim-level factuality and citation faithfulness support analyses |
| `df_c0_verdicts.parquet` | 3,096 | Claim-level C0 verifier verdicts | C0 verifier outputs | Claim-level factuality diagnostics and audit tables |
| `df_e14_oracle_per_report.parquet` | 327 | Oracle-arm claim-verification aggregate per report | E14 oracle entailment verifier outputs | Oracle/source-ablation paper figures and tables |
| `df_e14_oracle_verdicts.parquet` | 6,365 | Oracle-arm claim-level verifier verdicts | E14 oracle entailment verifier outputs | Oracle/source-ablation diagnostics |

`data/analysis/build_manifest.json` records the historical raw input roots and
producer hash for the core dataframes. Those raw roots are provenance metadata,
not public-file availability claims.

## Known upstream data issues (documented, not silently repaired)

1. **Claude Opus verdict schema heterogeneity** (13 distinct variants observed).
   Normalized per-verdict in this script — see `_extract_verdict_fields`.
2. **Claude Sonnet stored overall_score is corrupted.** Flagged via
   `overall_score_trustworthy = False`. Always use the recomputed columns.
3. **Criterion-id stability.** Criterion text is normalized before hashing so
   whitespace/case jitter across judge runs does not inflate rubric-drift counts.
4. **ablation_p5_no_citation_verify is excluded** from statistical comparisons
   (`excluded_from_analysis = True`) because only 2/90 reports were generated.
5. **Dual Claude-judge baselines (oracle analyses).** The released results contain TWO
   Claude scoring generations: the main-panel runs (judge labels `claude_opus`,
   `claude_sonnet`, feeding `df_scores`/`df_overall_scores`) and the version-bumped
   re-score runs used by the oracle arm (the `*48` result directories, e.g.
   `judge_opus48/`). The two generations differ systematically (e.g. Opus base-pattern
   citation_quality 0.76 main-panel vs 0.95 version-bumped). Any oracle-minus-baseline
   delta on a Claude judge MUST pair the oracle scores with the SAME generation's
   baseline (as `build_oracle_opus.py` does). Pairing oracle `*48` scores against the
   main-panel baseline in `df_scores` produces a spurious negative factual-accuracy
   delta (~-0.15) that is a judge-version artefact, not an oracle effect.
6. **Public redaction pass.** Two DRACO-derived rows were anonymized before
   public release. The same anonymized labels are applied to
   `df_queries.parquet`, `df_citations.parquet`, and `df_verdicts.parquet`; row
   counts and scoring fields are unchanged.

## Build reproducibility

See `build_manifest.json` for script hash, input paths, rubric-weight hash,
python/pandas/pyarrow versions, and per-parquet row counts.
