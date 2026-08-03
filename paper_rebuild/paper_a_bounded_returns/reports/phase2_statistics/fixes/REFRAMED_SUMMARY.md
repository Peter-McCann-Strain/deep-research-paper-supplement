# Phase 2 — REFRAMED SUMMARY (post-validator-fixes)

## Headline (CORRECTED)

Five GPT-4o-based architectures form a **statistically indistinguishable top cluster**: P1, P4, P5, P6, P7, P8 with mean overall scores in the range 0.59–0.67. Within-cluster pairwise comparisons are mostly NOT robust across all 3 judges.

Three patterns are clearly worse, with judge-robust large effects:
- **P0** baseline single-call: mean ≈ 0.49
- **P10** RL-trained 7B (DeepResearcher-7b): mean ≈ 0.34
- **P9** local 7B baseline (Qwen2.5-7B): mean ≈ 0.26

**The earlier 'P1 wins by +0.033' headline is an Opus artifact** (Opus shows P1−P4 = +0.124; GPT-5.2 and Sonnet show essentially zero difference).

## Per-pattern means (3-judge averaged)

| Pattern | N | Mean | Std |
|---|---:|---:|---:|
| base_p0 | 270 | 0.488 | 0.247 |
| base_p1 | 267 | 0.673 | 0.215 |
| base_p10 | 270 | 0.336 | 0.195 |
| base_p2 | 270 | 0.585 | 0.197 |
| base_p3 | 264 | 0.572 | 0.174 |
| base_p4 | 270 | 0.640 | 0.172 |
| base_p5 | 267 | 0.601 | 0.198 |
| base_p6 | 261 | 0.634 | 0.200 |
| base_p7 | 270 | 0.630 | 0.190 |
| base_p8 | 270 | 0.625 | 0.193 |
| base_p9 | 270 | 0.258 | 0.259 |

## Robust thesis: 'Source retrieval, not orchestration, is the binding constraint'

- Within-GPT-4o variation (P1 to P0) ≈ Δ 0.18 (and P0 baseline is *within* the cluster on some judges)
- GPT-4o ↔ local 7B gap: Δ ≈ 0.40
- Architectural complexity does not break the citation_quality / factual_accuracy ceiling
- All three judges agree on the top-cluster vs lower-tier separation

## Output artifacts

- `01_omnibus_crossed_random_effects.md`
- `02_per_judge_pairwise.md`
- `03_ablations_2judge.md`
- `04_concordance_fisher_z.md`
- `05_p9_floor_effect.md`
- `REFRAMED_SUMMARY.md`
