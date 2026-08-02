# Gate 5 (re-fit) — Ablations with consistent 2-judge (gpt52+sonnet) coverage

Original Phase 2 used 3-judge means, but Opus had near-zero ablation coverage (0-8 cells per ablation), making the original Gate 5 a misleading mix of 3-judge base means vs near-2-judge ablation means.

This re-run uses **gpt52+sonnet 2-judge mean** for both bases and ablations — symmetric coverage.

Plus TOST equivalence test (±0.02 ROPE) for null ablations.

| Ablation | Base | N | Δ | 95% CI | Wilcoxon p_holm | Cliff's δ | TOST p (±0.02) | Verdict |
|---|---|---:|---:|:---:|---:|---:|---:|:---:|
| ablation_p3_no_quality_eval | base_p3 | 87 | +0.002 | (-0.014, +0.018) | 1 | +0.001 | 0.01383 | EQUIVALENT (within ±0.02) |
| ablation_p3_no_topic_mining | base_p3 | 86 | +0.008 | (-0.011, +0.026) | 1 | +0.036 | 0.09768 | indeterminate |
| ablation_p4_fixed_perspectives | base_p4 | 90 | -0.030 | (-0.044, -0.016) | 0.0001236 | -0.218 | 0.9234 | **SIG** (degrades by 0.030) |
| ablation_p4_no_conversations | base_p4 | 90 | -0.036 | (-0.053, -0.019) | 2.526e-05 | -0.231 | 0.969 | **SIG** (degrades by 0.036) |
| ablation_p4_no_triangulation | base_p4 | 90 | -0.060 | (-0.080, -0.041) | 6.725e-09 | -0.385 | 1 | **SIG** (degrades by 0.060) |
| ablation_p5_fixed_width | base_p5 | 82 | -0.023 | (-0.044, -0.003) | 0.06923 | -0.144 | 0.6233 | indeterminate |
| ablation_p5_no_meta_eval | base_p5 | 86 | -0.047 | (-0.064, -0.029) | 2.996e-05 | -0.256 | 0.9975 | **SIG** (degrades by 0.047) |

## Interpretation

- TOST < 0.05 = mean diff is statistically equivalent to zero within ±0.02 ROPE (real null).
- TOST ≥ 0.05 AND Wilcoxon non-sig = underpowered/indeterminate, NOT proven null.
- Holm-significant Wilcoxon = ablation degrades the base.
