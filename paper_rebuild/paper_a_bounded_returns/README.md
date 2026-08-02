# Paper A Rebuild Package

This directory contains the public source needed to rebuild the Paper A tables,
figures, and manuscript PDF from the compact derived analysis tables shipped in
this repository.

Supported command:

```bash
deep-research paper rebuild paper-a --check-only
deep-research paper rebuild paper-a --skip-compile
deep-research paper rebuild paper-a
```

`--check-only` verifies that the required canonical store, derived parquet files,
tables, and figures are present. `--skip-compile` regenerates tables and figures
without requiring a TeX toolchain. The full command also runs `tectonic main.tex`
when `tectonic` is installed.

The public rebuild contract is deliberately narrower than a historical raw rerun:
it rebuilds the paper artifacts from the included canonical store and compact
derived analysis tables. It does not ship private notes, paper drafts, submission
bundles, raw generated report forests, raw judge-verdict packet directories, or
manual Claude Code judging workflows. Current public judging is available through
the API-backed `deep-research judge run` and `deep-research reproduce ... --judge`
commands.

Directory map:

| Path | Purpose |
|---|---|
| `main.tex` | Final manuscript source used for public PDF rebuilds. |
| `references*.bib` | Bibliography files used by `main.tex`. |
| `analysis/` | Paper-specific analysis, table, figure, and provenance scripts. |
| `analysis/canonical_numbers.json` | Canonical paper number store consumed by tables, figures, and prose checks. |
| `analysis/staging/` | Compact staged statistical outputs used by the canonical store. |
| `figures/` | Generated figure assets consumed by the manuscript. |
| `tables/` | Generated LaTeX tables consumed by the manuscript. |
| `reports/` | Curated compact statistical/failure-analysis summaries; not raw judge output. |
| `supporting_analysis/` | Small helper scripts and staged blobs that support specific appendix claims. |

For the cross-repository map from manuscript tables/figures to producers and
derived data, see `repro/PAPER_A_ARTIFACT_INDEX.md` at the repository root.

For raw API reruns, start from the repository-level reproducibility guide rather
than this directory.
