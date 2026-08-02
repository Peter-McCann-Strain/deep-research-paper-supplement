# Public Data Dictionary

`data/` in the public export is reserved for compact, reviewed inputs. Generated
payloads, caches, logs, checkpoints, local model weights, judge-output forests,
raw human-label assets, and training corpora belong outside GitHub.

| Path | Role | Public status |
|---|---|---|
| `README.md` | Human-readable data overview | Included |
| `DATA_DICTIONARY.md` | This inventory | Included |
| `eval_queries_v2.json` | Canonical 90-query evaluation manifest | Included |
| `all_90_queries.json` | 90-query convenience index of the canonical query set | Included |
| `b2_subset.json` | 12-query B2 subset manifest | Included |
| `protocol_a_stratified_v2.json` | 29-query Protocol A stratification manifest | Included |
| `variance_stratified.json` | 30-query variance experiment stratification manifest | Included |
| `public_judge_criteria.json` | Small API judge smoke/example criteria | Included |
| `benchmarks/` | Upstream benchmark cache directories | Excluded |
| `analysis/` | Generated parquet/csv analysis products | Excluded |
| `human_calibration_pack/` | Human-rater pilot packet | Excluded |
| `human_labels/` | Large local human-label assets with separate licenses | Excluded |
| `dr_judge_training/` | Large local DR-Judge training split | Excluded |
| `*_cache` directories | Live API/search caches | Excluded; runtime writes go under `artifacts/caches/` |
| `oracle_corpus_t1.json` and `e5_oracle_dose/` | Oracle/frozen corpus inputs needing separate review | Excluded |

The public export is governed by `PUBLIC_MANIFEST.json` at the repository root.

## Privacy Review

Rows with `metadata.public_redaction` were edited before release to remove
private details while keeping the benchmark task structure. See `data/README.md`
for the benchmark and professional-advice scope.
