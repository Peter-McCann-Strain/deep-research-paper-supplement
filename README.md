# Deep Research Public Supplement

This repository is the executable supplement for the bounded-returns deep-research orchestration paper. Install it from a checkout, not from PyPI, so the CLI can see `data/`, `repro/`, `docs/`, and the final PDF.

Use this repo to:

- run no-cost integrity checks for the shipped paper-reference files;
- inspect the frozen 90-query, 13-pattern headline summaries;
- compare a compatible pattern-metric summary against the frozen reference;
- run a current OpenAI or Azure OpenAI hosted-search demo; and
- add Anthropic API credentials when you want the full Claude judge panel.

It does not regenerate the exact submitted run. That would require archived raw reports, judge verdict trees, historical model/search snapshots, and local infrastructure that are not part of the GitHub release. Live API reruns are useful for checking the workflow and current-model drift, but they are not a bitwise reproduction claim.

License boundary: Apache-2.0 applies to code. Public data files are mixed-license by row/source; see `NOTICE` and `DATA_LICENSES.md` before redistributing data-derived material.

Research benchmark only: some prompts involve medical, legal, financial, insurance, or policy topics. They are evaluation inputs, not professional advice requests or recommendations.

Claude robustness judges use the Anthropic API directly. The public path does not depend on Claude Code, local assistant sessions, or model downloads.

## Quickstart

Requires Python `>=3.11,<3.13`.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -c constraints-public.txt -e ".[api]"
cp .env.example .env
deep-research quickstart-check
deep-research doctor
deep-research reproduce paper-a --mode smoke
deep-research reproduce paper-a --mode reference
deep-research reproduce paper-a --mode provenance
```

For maintainer checks, install the test extras and run the offline gate:

```bash
python -m pip install -c constraints-public.txt -e ".[api,paper,dev]"
python -m pytest -q -p no:cacheprovider
ruff check --no-cache deep_research tests
```

`quickstart-check`, `smoke`, `reference`, and `provenance` make no paid API calls. `quickstart-check` runs the full offline first-run path in one command. The reference command prints the compact public headline ordering and comparison policy from `repro/reference/paper_a_headline_numbers.json`. `repro/reference/paper_a_pattern_metrics.csv` gives a compact pattern-by-judge audit table, and `repro/reference/PATTERN_DICTIONARY.csv` maps every frozen `base_p*` identifier to the public reference table. `repro/PAPER_A_REPRO_MAP.md` maps every public command to its paper artifact and states whether it is comparable to the frozen paper metrics.

## What This Reproduces

| Workflow | Command | Paid APIs | Paper comparability |
|---|---|---:|---|
| Integrity smoke test | `deep-research reproduce paper-a --mode smoke` | No | Verifies shipped public inputs only. |
| Frozen headline reference | `deep-research reproduce paper-a --mode reference` | No | Shows the paper's 90-query, 13-pattern headline ordering. |
| Compact metric audit | `deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv` | No | Recomputes agreement against the shipped pattern-level metrics table. |
| Live API generation demo | `deep-research reproduce paper-a --mode api-best-effort --execute --limit N` | Yes | Best-effort current-API run; requires and verifies hosted web search; not the historical 13-pattern matrix. |
| Live API judging | add `--judge` | Yes | Uses query rubrics and the OpenAI plus Anthropic API panel for current reports; not a historical equality claim. |
| Historical exact rerun | Not shipped | N/A | Requires private/archival infrastructure, raw reports, judge verdict trees, and historical model/search snapshots. |

For the best-effort API path, fill in standard OpenAI credentials or the Azure OpenAI deployment settings from `.env.example`; add `ANTHROPIC_API_KEY` for the full judge panel.

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
| `deep_research/` | CLI and reproduction code: API judges, comparison, export, audit, and settings |
| `tests/` | Offline tests for release hygiene, docs, comparison, judging, and API request shapes |
| `repro/` | Reference tables, expected outputs, and paper-to-command mapping |
| `docs/` | Method and human-evaluation documentation |
| `data/` | Compact reusable inputs and dictionaries only |
| `papers/paper_a_bounded_returns/main.pdf` | Final paper PDF only |
| `artifacts/` | Local generated outputs only; not part of public GitHub |

## Release Policy

Large or regenerable material is local-first and gitignored. The physical home is `artifacts/`; legacy paths such as `results/experiments`, `results/judge_gpt52`, `models`, `logs`, and `checkpoints` are local compatibility paths only.

Do not commit raw caches, local model weights, checkpoints, logs, generated report forests, private notes, outreach messages, scratchpads, or submission bundles. Commit source, docs, compact canonical inputs, constraints, manifests, tests, and the final paper PDF.

Build and audit a clean candidate tree before publishing:

```bash
deep-research export-public --out /tmp/deep-research-public-export
deep-research release-audit --root /tmp/deep-research-public-export
```

If you run tests inside an exported candidate, rebuild the export before the
final audit and before publishing. Python bytecode caches are intentionally
rejected by the release audit.

## Reproducibility

See `REPRODUCIBILITY.md` for the public no-model-download workflow. The default path is a best-effort API rerun. Historical exact reproduction requires frozen artifacts and model/search snapshots and is not the default GitHub path.

## Scope Limits

- The working research workspace may contain private artifacts; publish from `deep-research export-public`, not from the raw workspace.
- Public reruns must record the exact OpenAI and Anthropic model identifiers used.
- Local model/GPU experiments and historical implementation modules are intentionally omitted from the public export; the public workflow is API-only and no-model-download.
