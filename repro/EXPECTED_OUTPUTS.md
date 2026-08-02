# Expected Offline Outputs

These commands do not call provider APIs. They are intended as first-run checks after installation and should finish in seconds on a normal laptop.

## `deep-research quickstart-check`

Expected shape:

```json
{
  "offline_ok": true,
  "api_calls_made": false,
  "smoke": {"status": "success"},
  "reference": {"status": "success", "reference_pattern_count": 13},
  "provenance": {"status": "success"},
  "compare": {"status": "success"}
}
```

If this fails, check that the repository was installed from a checkout and that `data/` and `repro/` are present.

## `deep-research doctor`

With no API keys, the command should still exit successfully and report missing paid configuration:

```json
{
  "generation_configured": false,
  "judge_panel_configured": false,
  "missing_generation_configuration": ["OPENAI_API_KEY"],
  "missing_judge_panel_configuration": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
}
```

It does not call APIs unless `--verify-api` is added.

## `deep-research reproduce paper-a --mode smoke`

Expected status: `success`. The check validates that compact reference files are present.

## `deep-research reproduce paper-a --mode reference`

Expected status: `success`; expected counts are 90 public queries and 13 frozen pattern rows. The top pattern should be `base_p1`. The 13 rows map to paper labels through `repro/reference/PATTERN_DICTIONARY.csv`.

## `deep-research reproduce paper-a --mode provenance`

Expected status: `success`. The command verifies hashes and counts in `repro/reference/REFERENCE_MANIFEST.json`. A mismatch usually means a reference file was edited without regenerating the manifest.

## `deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv`

Expected status: `success`; expected overlap is 13/13 patterns, full ordering match, full metric schema, and zero metric deltas. The `n_queries` values are historical scored-observation counts, so 87 or 89 means archived coverage was incomplete for that pattern even though the public manifest contains 90 queries. Blank judge cells mark unavailable archived judge metrics.
