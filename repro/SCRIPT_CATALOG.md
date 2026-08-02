# Public Script Catalog

This catalog explains why each top-level file in `scripts/` is shipped. The supported public entry points are the `deep-research ...` CLI commands documented in `README.md` and `REPRODUCIBILITY.md`; most files in `scripts/` are included so the historical research workflow is inspectable, not because every file is a no-input one-command reproduction target.

The machine-readable index is `repro/SCRIPT_CATALOG.csv`. CI checks that every shipped `.py`, `.sh`, or `.js` file under `scripts/` has a catalog row.

## Status Taxonomy

- `prefer public CLI wrapper`: use the documented CLI command first; the script exists for implementation/provenance.
- `supported public helper`: can be run directly for a public validation or diagnostic task.
- `optional non-default workflow`: may be useful, but requires explicit inputs, API keys, or local artifacts.
- `optional external download`: downloads third-party assets; inspect upstream licenses before redistributing outputs.
- `optional GPU/local-model workflow`: requires local model or training infrastructure and is outside the default API-only public path.
- `requires non-public raw artifacts`: reconstructs historical intermediate stores from raw forests, raw verdicts, or evaluator packets that are intentionally not shipped.
- `historical analysis helper`: retained so methods and decisions are inspectable; the supported paper-facing rebuild is `deep-research paper rebuild paper-a`.
- `internal worker helper`: called by a parent script, not intended as a standalone user entry point.

## Source-Scope Decision

The public release includes final, scrubbed Paper A source under `paper_rebuild/` only because the supported artifact rebuild can compile the manuscript PDF. Drafts, submission bundles, outreach material, personal notes, raw judge packets, generated report forests, local caches, model weights, and checkpoints remain excluded by `PUBLIC_MANIFEST.json` and the release audit.

## Default Public Path

For a new reader, start here instead of running scripts directly:

```bash
deep-research quickstart-check
deep-research paper rebuild paper-a --check-only
deep-research paper rebuild paper-a --skip-compile
deep-research reproduce paper-a --mode reference
deep-research judge run --report-file repro/examples/example_report.md --criteria-file data/public_judge_criteria.json --dry-run
```
