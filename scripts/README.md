# Scripts

The public CLI is the supported entry point for reproduction:

```bash
deep-research quickstart-check
deep-research reproduce paper-a --mode api-best-effort --execute --limit 3
deep-research reproduce paper-a --mode api-best-effort --execute --full --judge
deep-research paper rebuild paper-a --skip-compile
```

The scripts in this directory are shipped so the paper code is inspectable beyond
the CLI. They are historical research workflows, not a single default entrypoint.
Many read optional raw result directories such as `results/experiments/`,
`checkpoints/experiments/`, or benchmark corpora that are intentionally not part
of the GitHub tree.

Use this map when navigating the folder:

| Family | Examples | Public status |
|---|---|---|
| API and orchestration runners | `run_pattern.py`, `run_eval_v2.py`, `run_all_experiments.py` | Source for paper-pattern execution; may require provider keys and optional raw output directories. |
| Derived-data builders | `build_analysis_dataframes.py`, `build_compute_ledger.py`, `build_e5_dose_response.py` | Rebuilds compact analysis tables when optional raw artifacts are available; shipped derived tables live under `data/analysis/`. |
| Statistical and figure support | `phase2_statistical_analysis.py`, `phase3_figures.py`, `phase4_failure_taxonomy.py` | Historical analysis code; paper-facing table/figure rebuilds are wrapped by `deep-research paper rebuild`. |
| Benchmark/data adapters | `download_benchmarks.py`, `select_eval_queries.py`, `prepare_human_eval.py` | Optional data-preparation utilities; review upstream licenses before redistributing downloaded data. |
| Local-model and training probes | `train_p12_rl.py`, `finetune_dr_judge.py`, GPU queue scripts | Optional archival code. The public reproduction path does not require model weights or local GPU inference. |
| Diagnostics and reconciliation | `verify_headline_numbers.py`, `reconcile_exclusions.py`, `tool_health_report.py` | Audit/provenance helpers used during paper development. |

Manual Claude Code judging prep/parse scripts are not part of the public release.
Current public judge execution is API-backed through `deep-research judge run` and
`deep-research reproduce ... --judge`.
