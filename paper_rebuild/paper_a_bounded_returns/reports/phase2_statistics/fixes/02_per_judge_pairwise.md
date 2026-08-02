# Gate 3 (re-fit) — Per-judge pairwise robustness

For each of 55 pairs and each of 3 judges, run paired Wilcoxon (Holm-corrected within judge).
Report which pairs are: (a) Holm-significant in all 3 judges AND (b) directionally consistent.

- 55 pairs total, judges: ['gpt52', 'claude_opus', 'claude_sonnet']
- Pairs Holm-sig in all 3 judges: 26/55
- Pairs sig+direction-consistent (judge-robust): 26/55

## Judge-robust pairs (sig in all 3 + same direction)

| a | b | gpt52 | opus | sonnet |
|---|---|---:|---:|---:|
| base_p0 | base_p1 | -0.081 | -0.290 | -0.190 |
| base_p0 | base_p10 | +0.171 | +0.156 | +0.130 |
| base_p0 | base_p4 | -0.082 | -0.166 | -0.206 |
| base_p0 | base_p6 | -0.073 | -0.166 | -0.197 |
| base_p0 | base_p9 | +0.205 | +0.232 | +0.256 |
| base_p1 | base_p10 | +0.251 | +0.445 | +0.317 |
| base_p1 | base_p2 | +0.068 | +0.110 | +0.089 |
| base_p1 | base_p3 | +0.056 | +0.172 | +0.082 |
| base_p1 | base_p9 | +0.283 | +0.517 | +0.439 |
| base_p10 | base_p2 | -0.184 | -0.335 | -0.229 |
| base_p10 | base_p3 | -0.197 | -0.273 | -0.239 |
| base_p10 | base_p4 | -0.253 | -0.322 | -0.336 |
| base_p10 | base_p5 | -0.215 | -0.283 | -0.298 |
| base_p10 | base_p6 | -0.246 | -0.318 | -0.323 |
| base_p10 | base_p7 | -0.233 | -0.352 | -0.297 |
| base_p10 | base_p8 | -0.219 | -0.356 | -0.293 |
| base_p2 | base_p9 | +0.218 | +0.410 | +0.355 |
| base_p3 | base_p4 | -0.058 | -0.051 | -0.098 |
| base_p3 | base_p6 | -0.053 | -0.045 | -0.087 |
| base_p3 | base_p7 | -0.037 | -0.081 | -0.060 |
| base_p3 | base_p9 | +0.231 | +0.352 | +0.364 |
| base_p4 | base_p9 | +0.287 | +0.397 | +0.462 |
| base_p5 | base_p9 | +0.248 | +0.359 | +0.428 |
| base_p6 | base_p9 | +0.278 | +0.392 | +0.448 |
| base_p7 | base_p9 | +0.267 | +0.428 | +0.423 |
| base_p8 | base_p9 | +0.252 | +0.431 | +0.418 |

## Within-cluster (NOT judge-robust) — top cluster pairs

Of 23 within-top-cluster pairs (P1-P8), only 0 are judge-robust.

This justifies framing P1/P4/P5/P6/P7/P8 as a 'statistically indistinguishable top cluster' rather than ranking them.

## P1 vs P4 case study (the headline reversal)

- gpt52: meanΔ(P1−P4) = -0.0017
- claude_opus: meanΔ(P1−P4) = +0.1235
- claude_sonnet: meanΔ(P1−P4) = -0.0188
- Direction consensus across 3 judges: False

**Interpretation:** the apparent P1>P4 finding in Phase 2 was an Opus artifact. GPT-5.2 and Sonnet show effectively no difference. Reframe as 'P1≈P4'.
