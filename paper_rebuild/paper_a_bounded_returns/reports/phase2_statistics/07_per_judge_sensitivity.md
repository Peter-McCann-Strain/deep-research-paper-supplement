# Gate 7 — Per-Judge Sensitivity

## Omnibus per judge

| judge | LR | df | p |
|---|---|---|---|
| gpt52 | 974.38 | - | 1.11e-16 |
| claude_opus | 1062.36 | - | 1.11e-16 |
| claude_sonnet | 990.72 | - | 1.11e-16 |

## Mean overall score per (pattern × judge)

| pattern   |   gpt52 |   claude_opus |   claude_sonnet |
|:----------|--------:|--------------:|----------------:|
| base_p0   |   0.385 |         0.475 |           0.606 |
| base_p1   |   0.466 |         0.763 |           0.793 |
| base_p10  |   0.213 |         0.318 |           0.476 |
| base_p11  |   0.331 |       nan     |         nan     |
| base_p2   |   0.398 |         0.653 |           0.705 |
| base_p3   |   0.406 |         0.592 |           0.713 |
| base_p4   |   0.467 |         0.64  |           0.812 |
| base_p5   |   0.428 |         0.601 |           0.773 |
| base_p6   |   0.461 |         0.639 |           0.802 |
| base_p7   |   0.447 |         0.671 |           0.773 |
| base_p8   |   0.432 |         0.674 |           0.768 |
| base_p9   |   0.18  |         0.243 |           0.35  |

## Rank order per judge (1 = best)

| pattern   |   gpt52 |   claude_opus |   claude_sonnet |
|:----------|--------:|--------------:|----------------:|
| base_p0   |       9 |             9 |               9 |
| base_p1   |       2 |             1 |               3 |
| base_p10  |      11 |            10 |              10 |
| base_p11  |      10 |           nan |             nan |
| base_p2   |       8 |             4 |               8 |
| base_p3   |       7 |             8 |               7 |
| base_p4   |       1 |             5 |               1 |
| base_p5   |       6 |             7 |               5 |
| base_p6   |       3 |             6 |               2 |
| base_p7   |       4 |             3 |               4 |
| base_p8   |       5 |             2 |               6 |
| base_p9   |      12 |            11 |              11 |

## Spearman ρ between judge rankings

- gpt52 vs claude_opus: nan
- gpt52 vs claude_sonnet: nan
- claude_opus vs claude_sonnet: nan
