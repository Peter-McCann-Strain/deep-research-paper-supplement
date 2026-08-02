# Data And License Notes

Code in this repository is released under Apache-2.0. Public data files are mixed-license research inputs; treat each row according to its source license rather than assuming the code license applies to all data. `NOTICE` repeats this boundary for GitHub visitors who start from the root license file.

Public releases must keep only compact, reviewed inputs in GitHub. Large generated reports, judge verdict forests, model checkpoints, raw human labels, and cached API/search artifacts stay out of GitHub unless their license and consent status are explicitly documented.

## Included Public Data

| Path | Public role | Release status |
|---|---|---|
| `data/eval_queries_v2.json` | Canonical selected 90-query paper manifest | Included as mixed-license public evaluation input |
| `data/all_90_queries.json` | Compact query index | Included as derived metadata from the same selected query set |
| `data/b2_subset.json` | B2 subset manifest | Included as derived metadata |
| `data/protocol_a_stratified_v2.json` | Protocol A stratification manifest | Included as derived metadata |
| `data/variance_stratified.json` | Variance stratification manifest | Included as derived metadata |
| `data/public_judge_criteria.json` | Public API judge criteria | Included under Apache-2.0 with the repository code |

## Query Source Licenses

`data/eval_queries_v2.json` contains 90 selected prompts and rubrics:

| Source field | Rows | Upstream/public basis | License/status |
|---|---:|---|---|
| `draco` | 40 | https://huggingface.co/datasets/perplexity-ai/draco | MIT |
| `deepsearch_qa` | 20 | https://huggingface.co/datasets/google/deepsearchqa | Apache-2.0 |
| `research_qa` | 15 | https://huggingface.co/datasets/realliyifei/ResearchQA | MIT |
| `litqa2` | 10 | https://huggingface.co/datasets/futurehouse/lab-bench | CC-BY-SA-4.0; preserve attribution and share-alike obligations for these rows |
| `custom` | 5 | Author-created prompts for this supplement | Apache-2.0 |

The public query manifests are research prompts and metadata for reproducing the paper workflow. They do not include generated report corpora, human-rater packets, local API caches, or model-training data.

## Excluded By Policy

The public manifest excludes paper drafts and LaTeX sources, submission bundles, private notes, outreach messages, local paths, generated report forests, cached search/API payloads, human-label packets, model weights, checkpoints, and large upstream benchmark corpora.

The default public reproduction path is API-backed and best-effort. It does not require downloading model weights, frozen caches, or archived verdict corpora.

## Public Redaction

Two DRACO-derived rows were anonymized before public release. Their `metadata.public_redaction` fields mark the change. Both rows remain counted as DRACO-derived because the task structures and rubric sources are retained, but the original private details are not part of this repository.
