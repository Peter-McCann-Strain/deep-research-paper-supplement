# Gate 1 — Omnibus on Overall Score (base patterns)

Mixed-effects model: `overall ~ C(pattern) + (1|query_id)`

Patterns: 12  |  Queries: 90  |  Rows: 1074

## Likelihood-ratio test (fixed effect of pattern)

- LR statistic: **982.438** on df=11
- p-value: **0**
- Pass: **YES**

## Variance components (REML)

- Query variance: 0.00799
- Residual variance: 0.01209
- ICC(query): 0.398

## Pattern ranking (mean ± std, n)

| pattern | mean | std | n |
|---|---|---|---|
| base_p1 | 0.674 | 0.125 | 90 |
| base_p4 | 0.640 | 0.076 | 90 |
| base_p6 | 0.634 | 0.129 | 87 |
| base_p7 | 0.630 | 0.105 | 90 |
| base_p8 | 0.625 | 0.107 | 90 |
| base_p5 | 0.601 | 0.118 | 89 |
| base_p2 | 0.585 | 0.115 | 90 |
| base_p3 | 0.566 | 0.113 | 89 |
| base_p0 | 0.488 | 0.211 | 90 |
| base_p10 | 0.336 | 0.149 | 90 |
| base_p11 | 0.331 | 0.147 | 89 |
| base_p9 | 0.258 | 0.227 | 90 |
