# Gate 6 — Bayesian Signed-Rank (headline comparisons)

`baycomp.SignedRankTest` with ROPE=±0.02.

P(a>b) = posterior P(a beats b beyond ROPE). P(rope) = P(practical equivalence). P(b>a) = posterior P(b beats a beyond ROPE).

| comparison | a | b | n | mean_a | mean_b | P(a>b) | P(rope) | P(b>a) |
|---|---|---|---|---|---|---|---|---|
| P4 vs P10 (best pipeline vs RL-7B) | base_p4 | base_p10 | 90 | 0.640 | 0.336 | 1.000 | 0.000 | 0.000 |
| P9 vs P10 (RL effect) | base_p9 | base_p10 | 90 | 0.258 | 0.336 | 0.000 | 0.000 | 1.000 |
| P9 vs P0 (model scale) | base_p9 | base_p0 | 90 | 0.258 | 0.488 | 0.000 | 0.000 | 1.000 |
