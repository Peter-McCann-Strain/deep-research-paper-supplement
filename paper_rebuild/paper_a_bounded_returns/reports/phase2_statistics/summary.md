# Phase 2 Statistical Analysis — Summary

## Headline numbers

- **Gate 1 (omnibus overall)**: LR=982.44, p=0 — PASS
  - Query variance=0.0080, residual=0.0121, ICC=0.398
  - Best pattern: **base_p1** (mean=0.674)
  - Worst pattern: **base_p9** (mean=0.258)
- **Gate 2 (per-dim omnibus, Holm)**: 9/9 significant → ['information_recall', 'factual_accuracy', 'coverage', 'analytical_depth', 'citation_quality', 'logical_coherence', 'organization', 'instruction_following', 'attribution_quality']
- **Gate 3 (pairwise overall)**: 54/66 significant after Holm
- **Gate 4 (stratification)**: source p=0 (SIG); difficulty p=0.033 (SIG)
- **Gate 5 (ablations)**: 5/7 significant after Holm
- **Gate 6** P4 vs P10 (best pipeline vs RL-7B): P(a>b)=1.00, P(rope)=0.00, P(b>a)=0.00
- **Gate 6** P9 vs P10 (RL effect): P(a>b)=0.00, P(rope)=0.00, P(b>a)=1.00
- **Gate 6** P9 vs P0 (model scale): P(a>b)=0.00, P(rope)=0.00, P(b>a)=1.00

## Top-3 pairwise effects (overall, by |δ|)

| a | b | Δmean | δ | p_holm |
|---|---|---|---|---|
| base_p11 | base_p4 | -0.310 | -0.923 | 1.56e-14 |
| base_p4 | base_p9 | +0.382 | +0.905 | 1.15e-14 |
| base_p1 | base_p11 | +0.343 | +0.903 | 1.56e-14 |

## Top ablation effects

| ablation | base | Δ | δ | p_holm |
|---|---|---|---|---|
| ablation_p4_no_triangulation | base_p4 | -0.060 | -0.383 | 2.15e-09 |
| ablation_p5_no_meta_eval | base_p5 | -0.047 | -0.285 | 5.94e-06 |
| ablation_p4_no_conversations | base_p4 | -0.037 | -0.236 | 2.5e-05 |
| ablation_p4_fixed_perspectives | base_p4 | -0.030 | -0.235 | 0.000145 |
| ablation_p5_fixed_width | base_p5 | -0.024 | -0.174 | 0.0215 |
| ablation_p3_no_quality_eval | base_p3 | -0.007 | -0.053 | 0.465 |
| ablation_p3_no_topic_mining | base_p3 | -0.002 | -0.027 | 0.465 |

## Per-judge ranking agreement (Spearman ρ)

- gpt52 vs claude_opus: nan
- gpt52 vs claude_sonnet: nan
- claude_opus vs claude_sonnet: nan

## Files

- `01_omnibus_overall.md`
- `02_omnibus_per_dimension.md`
- `03_pairwise_analytical_depth.md`
- `03_pairwise_attribution_quality.md`
- `03_pairwise_citation_quality.md`
- `03_pairwise_coverage.md`
- `03_pairwise_factual_accuracy.md`
- `03_pairwise_information_recall.md`
- `03_pairwise_instruction_following.md`
- `03_pairwise_logical_coherence.md`
- `03_pairwise_organization.md`
- `03_pairwise_overall.md`
- `04_stratification.md`
- `05_ablations.md`
- `06_bayesian.md`
- `07_per_judge_sensitivity.md`
- `summary.md`
