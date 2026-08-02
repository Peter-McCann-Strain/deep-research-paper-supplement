# Deep Research Public Supplement

This repository is the executable supplement for the bounded-returns deep-research orchestration paper. Install it from a checkout, not from PyPI, so the CLI can see `data/`, `repro/`, `docs/`, and the final PDF.

| Paper | Details |
|---|---|
| Title | **Bounded Returns to Orchestration: A Synthesis Ceiling Beneath Eleven Controlled Deep-Research Architectures** |
| Author | Peter McCann Strain, independent researcher |
| Manuscript date | July 29, 2026 |
| PDF | `papers/paper_a_bounded_returns/main.pdf` |
| Rebuild source | `paper_rebuild/paper_a_bounded_returns/` |
| Citation metadata | `CITATION.cff` |
| DOI/arXiv | Not assigned in this release; add the identifier here when available. |

Use this repo to:

- run no-cost integrity checks for the shipped paper-reference files;
- inspect the frozen 90-query, 13-pattern headline summaries;
- rebuild the Paper A tables, figures, and manuscript PDF from the shipped derived data;
- inspect the paper's reusable pattern, evaluation, benchmark, tool, and analysis code;
- compare a compatible pattern-metric summary against the frozen reference;
- run a current OpenAI or Azure OpenAI hosted-search demo; and
- add Anthropic API credentials when you want the full Claude judge panel.

It rebuilds the public paper artifacts from the included canonical store and compact derived analysis tables. It does not bitwise replay the exact historical raw run because that would require raw generated report forests, raw judge-verdict packet directories, historical provider/search snapshots, and local infrastructure that are not part of the GitHub release. Live API reruns are useful for checking the workflow and current-model drift, but they are not a bitwise equality claim.

License boundary: Apache-2.0 applies to code. Public data files are mixed-license by row/source; see `NOTICE` and `DATA_LICENSES.md` before redistributing data-derived material.

Research benchmark only: some prompts involve medical, legal, financial, insurance, or policy topics. They are evaluation inputs, not professional advice requests or recommendations.

Claude robustness judges use the Anthropic API directly. The public path does not depend on Claude Code, local assistant sessions, or model downloads.

## Quickstart

Requires Python `>=3.11,<3.13`.

```bash
python3 -m venv venv
[ -f venv/bin/activate ] && source venv/bin/activate
python -m pip install -c constraints-public.txt -e ".[api,paper]"
cp .env.example .env
deep-research quickstart-check
deep-research paper rebuild paper-a --check-only
deep-research doctor
deep-research reproduce paper-a --mode smoke
deep-research reproduce paper-a --mode reference
deep-research reproduce paper-a --mode provenance
```

For maintainer checks, install the test extras and run the offline gate:

```bash
python -m pip install -c constraints-public.txt -e ".[api,paper,dev]"
python -m pytest -q -p no:cacheprovider
ruff check --select F821,F811,B008,B023 --no-cache deep_research tests
```

The lint command is the high-impact gate used by CI. The archival research
scripts are shipped for provenance and are not yet a full-style-clean tree.

`quickstart-check`, `smoke`, `reference`, and `provenance` make no paid API calls. `quickstart-check` runs the full offline first-run path in one command. The reference command prints the compact public headline ordering and comparison policy from `repro/reference/paper_a_headline_numbers.json`. `repro/reference/paper_a_pattern_metrics.csv` gives a compact pattern-by-judge audit table, and `repro/reference/PATTERN_DICTIONARY.csv` maps every frozen `base_p*` identifier to the public reference table. `repro/PAPER_A_REPRO_MAP.md` maps every public command to its paper artifact and states whether it is comparable to the frozen paper metrics. `repro/PAPER_A_ARTIFACT_INDEX.md` maps the manuscript, tables, figures, scripts, and derived data files one by one.

## What This Reproduces

| Workflow | Command | Paid APIs | Paper comparability |
|---|---|---:|---|
| Integrity smoke test | `deep-research reproduce paper-a --mode smoke` | No | Verifies shipped public inputs only. |
| Frozen headline reference | `deep-research reproduce paper-a --mode reference` | No | Shows the paper's 90-query, 13-pattern headline ordering. |
| Compact metric audit | `deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv` | No | Recomputes agreement against the shipped pattern-level metrics table. |
| Paper artifact rebuild | `deep-research paper rebuild paper-a --skip-compile` | No | Rebuilds tables and figures from `data/analysis/` and `paper_rebuild/.../analysis/canonical_numbers.json`. |
| Full manuscript compile | `deep-research paper rebuild paper-a` | No provider APIs | Also compiles `paper_rebuild/paper_a_bounded_returns/main.tex` and refreshes `papers/paper_a_bounded_returns/main.pdf` when `tectonic` is installed. |
| Live API generation demo | `deep-research reproduce paper-a --mode api-best-effort --execute --limit N` | Yes | Best-effort current-API run; requires and verifies hosted web search; not the historical 13-pattern matrix. |
| Live API judging | add `--judge` | Yes | Uses query rubrics and the OpenAI plus Anthropic API panel for current reports; not a historical equality claim. |
| Historical raw replay | Not fully shipped | N/A | Requires private/archival raw reports, raw judge packet directories, and historical model/search snapshots. |

Paper artifact rebuild:

```bash
deep-research paper rebuild paper-a --check-only
deep-research paper rebuild paper-a --skip-compile
deep-research paper rebuild paper-a
```

Use `--skip-compile` if you do not have `tectonic` installed. The command still regenerates the tables and figures consumed by the manuscript source.

For the best-effort API path, fill in standard OpenAI credentials or the Azure OpenAI deployment settings from `.env.example`; add `ANTHROPIC_API_KEY` for the full judge panel. The public CLI uses `OPENAI_MODEL` and `OPENAI_JUDGE_MODEL`. Older archival scripts that import `deep_research.config` still honor `DEFAULT_MODEL` and `JUDGE_MODEL` for backward compatibility.

No-call configuration and budget checks:

```bash
deep-research doctor --require-api --ensure-dirs
deep-research cost paper-a --limit 3
deep-research reproduce paper-a --mode api-best-effort --limit 3 --max-cost-usd 5
```

Optional paid entitlement probe:

```bash
deep-research doctor --verify-api
```

Paid small-subset execution:

```bash
deep-research reproduce paper-a --mode api-best-effort --execute --limit 3 --max-cost-usd 5
```

For a full rerun, use `--execute --full`. Add `--judge` to score generated reports with the OpenAI plus Anthropic API judge panel and the per-query criteria, dimensions, and weights bundled in `data/eval_queries_v2.json`.

Before spending on the full panel, run:

```bash
deep-research doctor --require-judge-panel
deep-research doctor --verify-api --verify-judge-panel
deep-research cost paper-a --full --judge
```

A full generation run makes 90 OpenAI/Azure Responses calls. `--judge` adds up to 90 OpenAI judge calls and 180 Anthropic judge calls. The cost command is a guardrail based on configured per-call rates; provider billing can differ because prices, tokenization, retries, report length, and hosted-search charges change.

Compare pattern-level metric summaries against the paper reference. A full `success` requires the candidate to provide every reference metric cell available for the overlapping patterns; otherwise the command reports `partial` or `diverged`.

The public manifest has 90 queries. The `n_queries` column in `paper_a_pattern_metrics.csv` is the archived count of scored observations for that pattern, so it can be 87 or 89 when historical outputs or judge cells were unavailable. Blank judge cells mean the archived aggregate did not have that judge metric for that row; they are not a current API error.

```bash
deep-research compare paper-a --run-summary path/to/pattern_metrics_summary.json
deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv
```

The live `api-best-effort` summary will intentionally report `not-comparable` because it does not rerun the paper's 13-pattern matrix.

Preview a single API-backed judge call without spending tokens:

```bash
deep-research judge run \
  --query "Example research question" \
  --report-file repro/examples/example_report.md \
  --criteria-file data/public_judge_criteria.json \
  --dry-run
```

Run the judge panel by removing `--dry-run`. The default `paper-a-api` panel uses the OpenAI judge model plus Claude Opus/Sonnet through the Anthropic API directly. The standalone example uses the small smoke criteria file; integrated reproduction with `--judge` writes each query's bundled criteria, dimensions, and weights, validates every provider verdict, and emits current-run dimension-weighted score summaries.

## Provider Contract

| Use | Interface | Required signal | Notes |
|---|---|---|---|
| Generation | OpenAI Responses API or Azure OpenAI v1 Responses API | `web_search_call` must appear in the response output | Azure calls use deployment names and `api-version=v1`. |
| OpenAI judge | Responses API with strict JSON-schema output | valid `evaluations` JSON for every criterion | No Claude Code or local assistant session is used. |
| Claude judges | Anthropic Messages API | valid `evaluations` JSON for every criterion | `ANTHROPIC_API_KEY` is required only for the full `paper-a-api` panel. |

Azure hosted search is backed by Bing grounding. Check your Azure region, compliance boundary, and current pricing before running `doctor --verify-api` or a paid rerun.

## Repository Map

| Path | Contents |
|---|---|
| `deep_research/` | Public package: CLI, API judges, reproduction, pattern implementations, evaluation code, tools, benchmarks, export, audit, and settings |
| `tests/` | Offline tests for release hygiene, docs, comparison, judging, and API request shapes |
| `repro/` | Reference tables, expected outputs, paper-to-command mapping, and the Paper A artifact index |
| `docs/` | Method and human-evaluation documentation |
| `data/` | Compact reusable inputs, query/rubric manifests, and derived analysis tables |
| `paper_rebuild/` | Paper A LaTeX source, bibliography, analysis scripts, generated figures/tables, and compact statistical summaries |
| `scripts/` | Historical execution, analysis, data-preparation, and diagnostic scripts, documented by `scripts/README.md` and classified one-by-one in `repro/SCRIPT_CATALOG.csv` |
| `papers/paper_a_bounded_returns/main.pdf` | Final paper PDF only |
| `artifacts/` | Local generated outputs only; not part of public GitHub |

## Release Policy

Large or regenerable material is local-first and gitignored. The physical home is `artifacts/`; legacy paths such as `results/experiments`, `results/judge_gpt52`, `models`, `logs`, and `checkpoints` are local compatibility paths only.

Do not commit raw caches, local model weights, checkpoints, logs, generated report forests, raw judge packet directories, private notes, outreach messages, scratchpads, drafts, or submission bundles. Commit source, docs, compact canonical inputs, derived public tables, constraints, manifests, tests, the paper rebuild package, and the final paper PDF.

| Exclusion reason | Examples | Public substitute |
|---|---|---|
| Privacy and consent | private notes, human-label packets, local evaluator materials | anonymized summaries and public protocol docs |
| License or redistribution limits | large upstream benchmark caches, raw third-party corpora | selected public query manifests with source/license metadata |
| Size and regenerability | generated report forests, raw judge packet trees, API/search caches | compact derived parquet tables, reference CSV/JSON summaries |
| Provider drift | exact historical model/search snapshots | best-effort current API rerun with model IDs recorded |
| Local infrastructure | model weights, checkpoints, GPU queues, dependency trees | API-only public workflow plus optional local scripts clearly cataloged |

Build and audit a clean candidate tree before publishing:

```bash
deep-research export-public --out /tmp/deep-research-public-export
deep-research release-audit --root /tmp/deep-research-public-export
```

`export-public` refuses a dirty git tree by default so `PUBLIC_EXPORT_REPORT.json`
maps to an exact commit. Use `--allow-dirty` only for local inspection exports.

If you run tests inside an exported candidate, rebuild the export before the
final audit and before publishing. Python bytecode caches are intentionally
rejected by the release audit.

## Reproducibility

See `REPRODUCIBILITY.md` for the public artifact rebuild and no-model-download API workflow. The default live path is a best-effort API rerun. Historical exact raw replay requires frozen raw artifacts and model/search snapshots and is not the default GitHub path.

For a Hugging Face dataset mirror, see `docs/huggingface_release.md`. The helper
script reads `HF_TOKEN` from the environment and never accepts tokens as command
arguments.

## Scope Limits

- The working research workspace may contain private artifacts; publish from `deep-research export-public`, not from the raw workspace.
- Public reruns must record the exact OpenAI and Anthropic model identifiers used.
- Local model/GPU experiment code is included for inspection and provenance, but the supported public reproduction workflow is API-only and no-model-download unless a user deliberately opts into optional local-model experiments.
