# Gate 5 — Ablations

Paired Wilcoxon (ablation vs base). Holm across 7 ablations. Mean Δ with percentile bootstrap 95% CI.

## Overall score

| ablation | base | n | mean_abl | mean_base | Δ | 95% CI | δ | p_raw | p_holm | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| ablation_p3_no_quality_eval | base_p3 | 87 | 0.566 | 0.572 | -0.007 | [-0.021,+0.008] | -0.053 | 0.233 | 0.465 | . |
| ablation_p3_no_topic_mining | base_p3 | 86 | 0.568 | 0.570 | -0.002 | [-0.018,+0.015] | -0.027 | 0.458 | 0.465 | . |
| ablation_p4_fixed_perspectives | base_p4 | 90 | 0.609 | 0.640 | -0.030 | [-0.044,-0.016] | -0.235 | 3.62e-05 | 0.000145 | Y |
| ablation_p4_no_conversations | base_p4 | 90 | 0.603 | 0.640 | -0.037 | [-0.053,-0.020] | -0.236 | 5e-06 | 2.5e-05 | Y |
| ablation_p4_no_triangulation | base_p4 | 90 | 0.579 | 0.640 | -0.060 | [-0.080,-0.042] | -0.383 | 3.07e-10 | 2.15e-09 | Y |
| ablation_p5_fixed_width | base_p5 | 82 | 0.582 | 0.606 | -0.024 | [-0.044,-0.004] | -0.174 | 0.00718 | 0.0215 | Y |
| ablation_p5_no_meta_eval | base_p5 | 86 | 0.556 | 0.604 | -0.047 | [-0.064,-0.029] | -0.285 | 9.89e-07 | 5.94e-06 | Y |

## Per-dimension Δ (ablation − base)

| ablation                       |   analytical_depth |   attribution_quality |   citation_quality |   coverage |   factual_accuracy |   information_recall |   instruction_following |   logical_coherence |   organization |
|:-------------------------------|-------------------:|----------------------:|-------------------:|-----------:|-------------------:|---------------------:|------------------------:|--------------------:|---------------:|
| ablation_p3_no_quality_eval    |              0.011 |                 0.053 |              0.009 |     -0.011 |              0.02  |               -0.053 |                   0.003 |              -0.089 |         -0.024 |
| ablation_p3_no_topic_mining    |              0.02  |                 0.032 |             -0.01  |      0.012 |              0.046 |               -0.06  |                  -0.004 |              -0.077 |         -0.027 |
| ablation_p4_fixed_perspectives |             -0.012 |                 0.041 |              0.034 |     -0.065 |             -0.004 |               -0.093 |                  -0.085 |              -0.058 |          0.006 |
| ablation_p4_no_conversations   |             -0.068 |                 0.03  |              0.008 |     -0.06  |              0     |               -0.072 |                  -0.089 |              -0.051 |         -0.003 |
| ablation_p4_no_triangulation   |             -0.086 |                 0.035 |             -0.006 |     -0.087 |             -0.018 |               -0.108 |                  -0.128 |              -0.067 |         -0.021 |
| ablation_p5_fixed_width        |              0.002 |                 0.045 |              0.005 |     -0.03  |             -0.003 |               -0.088 |                  -0.065 |              -0.05  |         -0.002 |
| ablation_p5_no_meta_eval       |             -0.007 |                 0.002 |             -0.072 |     -0.068 |             -0.025 |               -0.106 |                  -0.059 |              -0.041 |          0.008 |
