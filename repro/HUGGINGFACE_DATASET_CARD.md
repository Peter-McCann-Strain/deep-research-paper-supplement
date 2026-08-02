---
license: apache-2.0
language:
  - en
pretty_name: Deep Research Paper Supplement
tags:
  - deep-research
  - reproducibility
  - llm-as-judge
  - research-agents
  - paper-supplement
size_categories:
  - n<1K
task_categories:
  - text-generation
  - text-classification
---

# Deep Research Paper Supplement

This Hugging Face dataset mirror contains the public executable supplement for
the bounded-returns deep-research orchestration paper. It is a code-and-data
supplement, not a model checkpoint. It includes compact public query/rubric
inputs, derived analysis tables, source code, tests, the Paper A artifact
rebuild package, and the final paper PDF.

## Reproducibility Contract

Supported no-cost checks:

```bash
python -m pip install -c constraints-public.txt -e ".[api,paper]"
deep-research quickstart-check
deep-research paper rebuild paper-a --check-only
deep-research reproduce paper-a --mode smoke
deep-research reproduce paper-a --mode reference
deep-research reproduce paper-a --mode provenance
```

The paper tables and figures rebuild from `paper_rebuild/` plus the compact
derived tables in `data/analysis/`. Exact historical raw replay is not claimed:
raw generated reports, raw judge-verdict trees, caches, model weights,
checkpoints, paper drafts, submission bundles, and personal working notes are
excluded from the public release.

## API Path

The current rerun path uses OpenAI or Azure OpenAI hosted-search generation and
an API-backed judge panel. Anthropic judges use the Anthropic API directly; the
public workflow does not require Claude Code or local assistant sessions.

## License And Data Boundary

Apache-2.0 applies to code. Public data files are mixed-license by row/source;
see `NOTICE` and `DATA_LICENSES.md` before redistributing data-derived material.
