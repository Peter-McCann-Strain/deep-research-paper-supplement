# Gate 1 (re-fit) — Crossed Random Effects: query + judge

Model: `overall ~ C(pattern) + (1|query_id) + (1|judge)` on un-aggregated base data (3 judges × 11 patterns × 90 queries).

This re-specification (vs the original aggregated 3-judge mean) properly partitions judge variance.

- N rows: 2949
- LR test of pattern effect: LR = 1879.96, df = 10, p = 0.00e+00
- Variance components (REML):
  - σ²(query) = 0.00116  → ICC(query) = 0.028
  - σ²(judge) = 0.02301  → ICC(judge) = 0.546
  - σ²(residual) = 0.01799

Interpretation: query difficulty and judge stringency are both substantial variance sources, but the pattern effect remains overwhelmingly significant after both are accounted for.
