# Reproducibility

This repo supports four public workflows: no-cost checks, inspection of the frozen paper-reference summaries, Paper A artifact rebuilds, and optional current-API reruns. The current-API path uses hosted models and search, so it is best-effort by design.

It does not recreate the private historical raw run bit for bit. Exact raw replay would require archived generated reports, raw judge-verdict packet directories, model/search snapshots, and local infrastructure that are outside this GitHub supplement. The paper-facing artifacts are rebuilt from the included canonical store and compact derived analysis tables.

## Setup

Requires Python `>=3.11,<3.13`.

```bash
python3 -m venv venv
[ -f venv/bin/activate ] && source venv/bin/activate
python -m pip install -c constraints-public.txt -e ".[api,paper]"
cp .env.example .env
```

Fill in either standard OpenAI credentials or Azure OpenAI credentials, plus Anthropic for the full judge panel:

- Standard OpenAI: `OPENAI_API_KEY`
- Azure OpenAI: `USE_AZURE_OPENAI=true`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION=v1`, and `AZURE_OPENAI_DEPLOYMENT`
- Full judge panel: `ANTHROPIC_API_KEY`

Azure OpenAI uses deployment names. Public hosted-search generation expects the OpenAI-compatible v1 Responses API and a deployment entitled for the configured hosted `web_search` tool. The checked `constraints-public.txt` pins the tested direct public dependency set while excluding GPU/local-model stacks.

## Smoke Check

```bash
deep-research quickstart-check
deep-research doctor
deep-research reproduce paper-a --mode smoke
deep-research reproduce paper-a --mode reference
deep-research reproduce paper-a --mode provenance
```

`quickstart-check` runs the offline first-run path in one command. `smoke` validates public reference files. `reference` prints the compact headline ordering, broad score ranges, and comparison policy used for best-effort public reruns. `provenance` verifies checked hashes and counts for the public reference files. None of these commands make paid API calls. See `repro/PAPER_A_REPRO_MAP.md` for the command-to-artifact map and comparability contract, and `repro/PAPER_A_ARTIFACT_INDEX.md` for the manuscript/table/figure/data index.

## Paper Artifact Rebuild

Check that all public rebuild inputs are present:

```bash
deep-research paper rebuild paper-a --check-only
```

Regenerate tables and figures from the shipped canonical store and derived analysis tables:

```bash
deep-research paper rebuild paper-a --skip-compile
```

Compile the manuscript too when `tectonic` is installed:

```bash
deep-research paper rebuild paper-a
```

This writes generated assets under `paper_rebuild/paper_a_bounded_returns/`. A full compile also refreshes `papers/paper_a_bounded_returns/main.pdf` from the rebuilt source PDF for citation and browsing. The artifact rebuild makes no provider API calls.

`paper_rebuild/paper_a_bounded_returns/analysis/rebuild_all.sh` is retained as historical provenance code for rebuilding the canonical store when optional raw result directories are available. It is not the default public command because it depends on raw artifacts that are intentionally not shipped.

For direct script use, consult `repro/SCRIPT_CATALOG.csv` first. It classifies every shipped top-level file under `scripts/` by family, public status, required inputs or services, expected outputs, and a short purpose summary.

## Best-Effort API Rerun

Offline preflight without paid API calls:

```bash
deep-research doctor --require-api --ensure-dirs
deep-research cost paper-a --limit 3
deep-research reproduce paper-a --mode api-best-effort --limit 3 --max-cost-usd 5
```

This preflight writes a local JSON plan under `artifacts/reproduction/`; it does not call provider APIs. `doctor --require-api` checks that required generation credentials are present but does not validate live model/tool entitlement.

Optional paid entitlement probes before execution:

```bash
deep-research doctor --verify-api
deep-research doctor --verify-api --verify-judge-panel
```

`doctor --verify-api` makes a small paid generation request to confirm model/tool access and checks that the provider actually returns a `web_search_call`. Add `--verify-judge-panel` only when you want to validate the full OpenAI plus Anthropic judge-panel configuration. `doctor --require-judge-panel` is a no-call configuration check for the same credential set.

Execute a small no-download API subset:

```bash
deep-research reproduce paper-a --mode api-best-effort --execute --limit 3 --max-cost-usd 5
```

Execute all public queries and run the API judge panel on generated reports:

```bash
deep-research reproduce paper-a --mode api-best-effort --execute --full --judge
```

The run writes generated Markdown reports, per-query JSON status files, query-rubric criteria files for judged runs, and `summary.json` under `artifacts/reproduction/paper_a_api_best_effort/`.

The public API workflow reads `OPENAI_MODEL` and `OPENAI_JUDGE_MODEL` from `.env`. The shipped defaults were release-tested on 2026-08-02 and still require provider entitlement in the account used for the rerun. Legacy archival scripts that import `deep_research.config` retain `DEFAULT_MODEL` and `JUDGE_MODEL` overrides so historical sensitivity runners can be inspected or rerun deliberately.

Every generation call requires hosted web search. If the provider response lacks a `web_search_call`, the query is marked failed instead of being treated as a valid research report.

A full generation run makes 90 OpenAI/Azure Responses calls. A full run with `--judge` adds up to 90 OpenAI judge calls and 180 Anthropic judge calls. Run `deep-research doctor --require-judge-panel`, `deep-research doctor --verify-api --verify-judge-panel`, and `deep-research cost paper-a --full --judge` before spending on the full panel.

The summary records provider usage fields when they are returned, validated provider verdicts, current-run dimension-weighted scores, and an estimated incurred-cost ledger. Exact historical equality is not promised because model and search APIs drift.

Compare a pattern-metric summary against the frozen reference with:

```bash
deep-research compare paper-a --run-summary path/to/pattern_metrics_summary.json
deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv
```

The live `api-best-effort` summary is expected to return `not-comparable` from this command because it does not rerun the paper's 13-pattern matrix. In the frozen CSV, `n_queries` is the historical count of scored observations for that pattern, not a claim that the public manifest has fewer than 90 queries. Blank judge cells mark unavailable archived judge coverage for that row.

## API Judge Panel

The public judge path calls provider APIs directly. It does not require local assistant sessions or subscriptions.

Criteria can be JSON, JSONL, or plain text. JSON may be either a list or an object with a `criteria`, `rubric`, or `items` list.

```bash
deep-research judge run \
  --query "Research question" \
  --report-file repro/examples/example_report.md \
  --criteria-file data/public_judge_criteria.json \
  --panel paper-a-api \
  --dry-run
```

Use `--dry-run` first to validate files and show the provider plan without making API calls. The default `paper-a-api` panel requires OpenAI plus Anthropic API credentials. `--panel openai-only` is useful for cheap debugging and should not be reported as the full Paper A judge panel. Integrated reproduction with `--judge` uses each query's bundled criteria, dimensions, and weights from `data/eval_queries_v2.json`; `data/public_judge_criteria.json` is only a small standalone smoke/example criteria file.

| Provider path | API used by this repo | Extra check |
|---|---|---|
| OpenAI generation | Responses API with hosted `web_search` | response must include `web_search_call` |
| Azure generation | OpenAI-compatible `/openai/v1/responses?api-version=v1` | deployment must be entitled for hosted `web_search` |
| OpenAI judge | Responses API with strict JSON-schema output | all criteria must be returned once |
| Anthropic judges | Messages API | Opus and Sonnet model IDs are recorded in outputs |

Azure hosted search uses Bing grounding. Confirm pricing, data handling, and region/compliance requirements in your Azure account before live verification or a full run.

## Public Export And Audit

Run the offline test gate before publishing:

```bash
python -m pip install -c constraints-public.txt -e ".[api,paper,dev]"
python -m pytest -q -p no:cacheprovider
ruff check --select F821,F811,B008,B023 --no-cache deep_research tests
deep-research paper rebuild paper-a --check-only
```

The lint command is the high-impact gate used by CI. The archival research
scripts are shipped for provenance and are not yet a full-style-clean tree.

Before publishing a candidate tree:

```bash
deep-research export-public --out /tmp/deep-research-public-export
deep-research release-audit --root /tmp/deep-research-public-export
```

The audit fails on private files, local paths, filled secret assignments, generated bundles, model weights, and oversized files. It also enforces `PUBLIC_MANIFEST.json`, so files outside the explicit allowlist are rejected.

For a Hugging Face dataset mirror, run `scripts/publish_huggingface.py --dry-run`
first. The publish helper builds the same audited export, swaps in the Hugging
Face dataset card for the upload copy, refreshes file hashes, and reads any
credential only from `HF_TOKEN` or an existing Hugging Face login.

If you run tests inside an exported candidate tree, rebuild the export before
the final audit and before publishing. Runtime caches and bytecode are not part
of the public artifact.
