# Canonical numbers (regenerated from current parquets)

Single source of truth for the rewrite. sonnet uses overall_score_recomputed.

## Headline per-pattern (3-judge, sonnet-corrected)

| pattern | n_cells | n_q | mean | std | gpt52 | opus | sonnet_corr |
|---|--:|--:|--:|--:|--:|--:|--:|
| base_p1 | 268 | 90 | 0.6734 | 0.2149 | 0.4665 | 0.7629 | 0.7932 |
| base_p4 | 270 | 90 | 0.6397 | 0.1721 | 0.4666 | 0.6404 | 0.8122 |
| base_p6 | 261 | 87 | 0.634 | 0.1997 | 0.4612 | 0.6393 | 0.8016 |
| base_p7 | 270 | 90 | 0.6301 | 0.1898 | 0.4466 | 0.6708 | 0.7729 |
| base_p8 | 270 | 90 | 0.625 | 0.1934 | 0.4322 | 0.6743 | 0.7684 |
| base_p5 | 267 | 89 | 0.6007 | 0.1976 | 0.4283 | 0.6008 | 0.7728 |
| base_p2 | 270 | 90 | 0.5853 | 0.1968 | 0.3976 | 0.6535 | 0.7047 |
| base_p3 | 265 | 89 | 0.5697 | 0.1766 | 0.406 | 0.5917 | 0.7133 |
| base_p0 | 270 | 90 | 0.4884 | 0.247 | 0.3845 | 0.4747 | 0.6059 |
| base_p11 | 169 | 89 | 0.4002 | 0.1762 | 0.3308 | nan | 0.4775 |
| base_p10 | 270 | 90 | 0.3358 | 0.1951 | 0.2131 | 0.3184 | 0.4757 |
| base_p9 | 270 | 90 | 0.2577 | 0.259 | 0.18 | 0.2431 | 0.35 |
| base_p12 | 142 | 90 | 0.241 | 0.1914 | 0.1821 | nan | 0.3428 |

## IRR
Krippendorff α=0.4234, ICC(A,1)=0.4892, ICC(A,k=3)=0.7418 (n=983)

per-dim α: information_recall=0.1055, factual_accuracy=0.1347, coverage=0.542, analytical_depth=0.5721, citation_quality=0.5417, logical_coherence=0.0973, organization=0.8405, instruction_following=0.2761, attribution_quality=-0.102

## Variance components
ICC(query)=0.4548, ICC(judge)=0.1772, σ²resid=0.01832


## Verdicts (TRUE counts)
total=248536; base=131897; ablation=46844; protocol_a=13139; variance=29034; triples≥2=79972; triples=3=50298


## DR-Judge
κ overall=0.4533 (n=3824); undisputed=0.6525; disputed=0.1989; agree=0.7189


## Citations
total=22903; by_category={'real_url': 12117, 'placeholder': 6803, 'academic': 3966, 'suspicious': 17}
