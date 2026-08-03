# J0 - Judge-Version Calibration Bridge (SUMMARY)

**Date:** 2026-07-12 | **Sample:** 40 headline reports (base_p0..base_p10), stratified seed=42, balanced over pattern x complexity (13 simple / 13 moderate / 14 complex).
**Method:** Re-judged BLIND with Opus 4.8 (`claude-opus-4-8`) and Sonnet 5 (`claude-sonnet-5`) using DRACO binary methodology on the SAME banked criteria; aligned NEW vs BANKED on identical (pattern, query_id, criterion_id); overall recomputed as mean satisfied-rate (stored overall_score never trusted). 95% CIs = 10k report-bootstrap, seed=42.

## Headline offset (new minus banked satisfied-rate)

| Family | Banked -> New overall | Overall Δ | 95% CI | Direction | κ | raw agree | aligned criteria |
|---|---|---|---|---|---|---|---|
| **Opus 4.8** vs Opus 4.1 | 0.6444 -> 0.7786 | **+0.1342** | [0.1014, 0.1673] | **more LENIENT** | 0.425 | 0.7588 | 1513 |
| **Sonnet 5** vs Sonnet 4.5 | 0.7463 -> 0.674 | **-0.0723** | [-0.1085, -0.0369] | **stricter** | 0.4086 | 0.755 | 1494 |

**The two new Claude judges move in OPPOSITE directions.** Opus 4.8 marks criteria SATISFIED ~13.4 pp MORE often than Opus 4.1 (significant (CI excludes 0)). Sonnet 5 marks SATISFIED ~7.2 pp LESS often than Sonnet 4.5 (significant (CI excludes 0)). Both offsets are statistically significant. Criterion-level agreement with the banked panel is moderate (κ≈0.41-0.43, raw≈0.76) for both - i.e. the version shift is a real systematic bias, not just noise.

## Per-dimension Δ - Opus 4.8 vs Opus 4.1 (new − banked)
| dimension | banked | new | Δ | 95% CI |
|---|---|---|---|---|
| information_recall | 0.4744 | 0.5897 | +0.1154 | [0.0188, 0.2125] |
| factual_accuracy | 0.6118 | 0.6553 | +0.0435 | [-0.0062, 0.096]  (ns) |
| coverage | 0.6931 | 0.8254 | +0.1323 | [0.0829, 0.1889] |
| analytical_depth | 0.5663 | 0.8133 | +0.2470 | [0.1728, 0.321] |
| citation_quality | 0.5215 | 0.816 | +0.2945 | [0.2134, 0.3758] |
| logical_coherence | 0.8291 | 0.812 | -0.0171 | [-0.1167, 0.075]  (ns) |
| organization | 0.9688 | 1.0 | +0.0312 | [0.0062, 0.0563] |
| instruction_following | 0.7625 | 0.8812 | +0.1188 | [0.0312, 0.2125] |
| attribution_quality | 0.25 | 0.6875 | +0.4375 | [0.325, 0.55] |

Opus 4.8's leniency is concentrated in the *subjective / quality* dimensions: attribution_quality (+0.438), citation_quality (+0.294), analytical_depth (+0.247). It is essentially unchanged on **factual_accuracy** (+0.043, ns) and **logical_coherence** (-0.017, ns), and near-ceiling on organization.

## Per-dimension Δ - Sonnet 5 vs Sonnet 4.5 (new − banked)
| dimension | banked | new | Δ | 95% CI |
|---|---|---|---|---|
| information_recall | 0.6812 | 0.45 | -0.2312 | [-0.3062, -0.1562] |
| factual_accuracy | 0.7438 | 0.6188 | -0.1250 | [-0.1781, -0.075] |
| coverage | 0.7644 | 0.7184 | -0.0460 | [-0.1084, 0.0123]  (ns) |
| analytical_depth | 0.7125 | 0.6562 | -0.0563 | [-0.1437, 0.025]  (ns) |
| citation_quality | 0.55 | 0.6875 | +0.1375 | [0.05, 0.225] |
| logical_coherence | 0.9083 | 0.725 | -0.1833 | [-0.2917, -0.0833] |
| organization | 0.9938 | 1.0 | +0.0062 | [0.0, 0.0188]  (ns) |
| instruction_following | 0.8 | 0.7625 | -0.0375 | [-0.1562, 0.0688]  (ns) |
| attribution_quality | 0.4625 | 0.35 | -0.1125 | [-0.225, -0.0125] |

Sonnet 5's extra strictness is concentrated in **information_recall** (-0.231), **logical_coherence** (-0.183) and **factual_accuracy** (-0.125) - it demands more before crediting recall/consistency. It is slightly MORE lenient only on citation_quality (+0.138).

## Adjustment for downstream analyses (map NEW Claude scores onto the original banked scale)
- **Opus:** subtract **0.1342** from any Opus-4.8 satisfied-rate (add -0.1342); per-dimension adjustments in `judge_version_bridge.json['families']['opus']['adjustment_per_dim']`.
- **Sonnet:** add **0.0723** to any Sonnet-5 satisfied-rate (add +0.0723); per-dimension adjustments in the JSON.

## How much does this shift the paper's Claude-based effect sizes?
The paper's headline effects are **differences between architectures judged by the SAME judge**, so a *uniform* per-judge shift **cancels** in a within-judge contrast (P_i − P_j). J0 therefore does **not** move any existing banked-panel effect size - those stay on the Opus-4.1/Sonnet-4.5 scale. Its role is forward-looking: any NEW experiment (bake-off, gpt-4.1 backbone, B-track) judged by Opus 4.8 / Sonnet 5 must be de-biased by the above before it is pooled or compared against a banked number. The main risk is **level** comparisons (e.g. "is arm X's absolute quality above the banked cohort") - there the ~+13pp Opus / ~−7pp Sonnet gaps are large enough to flip a naive comparison and MUST be applied. Because the two families shift oppositely, a two-Claude panel mean is partially self-cancelling (net ≈ (+0.134−0.072)/2 ≈ +0.031), but the per-family and per-dimension corrections should be used rather than the pooled net.

*Caveats:* offsets estimated on n=40 reports / ~1.5k aligned criteria per family; dimension-level CIs for the smaller dimensions (attribution_quality n≈80) are correspondingly wide. Union-staging dropped 33 (Opus) / 52 (Sonnet) new verdicts that only the other banked family had scored; 0 true hash mismatches.
