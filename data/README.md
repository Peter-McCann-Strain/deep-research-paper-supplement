# Public Evaluation Data

This directory contains the compact inputs needed by the public CLI. Generated
reports, live-search caches, human-label packets, training data, analysis tables,
and upstream benchmark caches stay out of GitHub; `DATA_LICENSES.md` is the
canonical place for those release boundaries.

## Included Files

- `eval_queries_v2.json`: canonical 90-query evaluation manifest used by the paper.
- `all_90_queries.json`: compact convenience index for the same 90 queries.
- `b2_subset.json`: 12-query B2 subset manifest.
- `protocol_a_stratified_v2.json`: 29-query Protocol A stratification manifest.
- `variance_stratified.json`: 30-query variance experiment stratification manifest.
- `public_judge_criteria.json`: small public criteria file for API judge smoke tests and examples.
- `DATA_DICTIONARY.md`: data inventory for the public export.

## Query Manifest Schema

`eval_queries_v2.json` contains one object per selected query. The original benchmark cache directories are not included; this selected manifest is the reproducible public input.

| Field | Meaning |
|---|---|
| `id` | Stable query identifier used in outputs and filenames. |
| `query` | Prompt shown to the research system. |
| `source` | Source benchmark or `custom`. |
| `domain` | Coarse topic bucket used for stratification. |
| `difficulty` | `simple`, `moderate`, or `complex`. |
| `metadata` | Source-specific bookkeeping; redacted rows carry `public_redaction`. |
| `rubric` | Criteria, dimensions, and weights used by the API judge path. |

Example shape:

```json
{
  "id": "q_example",
  "query": "What evidence answers the research question?",
  "source": "custom",
  "domain": "NLP/AI",
  "difficulty": "moderate",
  "metadata": {"index": 1},
  "rubric": {
    "dimension_weights": {"factual_accuracy": 0.25, "coverage": 0.25},
    "criteria": [{"text": "cites relevant sources", "dimension": "citation_quality"}]
  }
}
```

Two DRACO-derived rows carry `metadata.public_redaction` because sensitive details were anonymized before release.

## Redistribution Notes

The included query manifests are compact research inputs selected for the paper
supplement. `eval_queries_v2.json` is mixed-license by row: DRACO and ResearchQA
rows are MIT, DeepSearchQA rows are Apache-2.0, LitQA2/LAB-Bench rows are
CC-BY-SA-4.0, and custom rows are Apache-2.0. See `DATA_LICENSES.md` for the
source table and upstream links. Before any future expansion of `data/`, review
upstream license and privacy status and update both files with per-file terms.

## Professional-Advice Disclaimer

These files contain research benchmark inputs only. Prompts touching medical, legal, financial, insurance, or policy topics are not professional advice, legal instructions, clinical guidance, or investment recommendations.
