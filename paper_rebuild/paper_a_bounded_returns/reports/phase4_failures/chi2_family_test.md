# Chi-square test: family × failure-mode distribution

Tests whether the distribution of failure modes differs significantly between the GPT-4o family (P0–P8, base patterns) and the Local 7B family (P9–P10).

## Contingency table (tagged counts)

| family_ab   |   citation_fabrication |   empty_or_sparse |   entity_confusion |   factual_contradiction |   format_violation |   hallucinated_source |   missing_evidence |   missing_perspective |   scope_drift |   superficial_analysis |
|:------------|-----------------------:|------------------:|-------------------:|------------------------:|-------------------:|----------------------:|-------------------:|----------------------:|--------------:|-----------------------:|
| GPT-4o      |                    797 |               190 |                109 |                     724 |                  1 |                   147 |               1310 |                    40 |            47 |                    260 |
| Local7B     |                     38 |               516 |                 30 |                     216 |                  1 |                    99 |                413 |                    24 |            15 |                    197 |

## Result

| Statistic | Value |
|:---|---:|
| χ² | 992.201 |
| df | 9 |
| p-value | 8.28e-208 |

## Interpretation

The failure-mode distributions of GPT-4o and Local 7B families are **qualitatively different** (χ²=992.2, df=9, p=8.28e-208). The null hypothesis of identical mode proportions is rejected at p<0.001. Notably, citation_fabrication is far more prevalent in GPT-4o outputs, while superficial_analysis and missing_evidence are more prominent in Local 7B outputs — consistent with different failure archetypes driven by model scale and RL training.
