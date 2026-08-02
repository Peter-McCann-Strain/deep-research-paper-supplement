# Phase 4 Failure Analysis — Summary

> Generated: 2026-04-15. All figures use gpt52 + claude_sonnet judges combined, base patterns only (P0–P10), unless noted.

---

## 1. 10-Mode Taxonomy

Failure modes derived from keyword/regex heuristics on unsatisfied-verdict reasoning text.
A single verdict may be tagged with multiple modes.

| # | Mode | One-line description |
|:---|:---|:---|
| 1 | `citation_fabrication` | Report uses placeholder citations ("Web Search Synthesis"), missing URLs, or unverifiable generic sources |
| 2 | `hallucinated_source` | Cited paper, study, or source does not exist or cannot be verified; fabricated reference |
| 3 | `entity_confusion` | Wrong year, name, author, or numerical value; misidentification or misattribution of entities |
| 4 | `missing_perspective` | One-sided treatment; omits counter-arguments, opposing views, or alternative positions |
| 5 | `format_violation` | Output uses prose when table/list was requested, or violates specified structural template |
| 6 | `superficial_analysis` | Surface-level or descriptive treatment; lacks synthesis, critical evaluation, or nuance |
| 7 | `missing_evidence` | Claims not supported by concrete data, citations, or quantitative figures |
| 8 | `empty_or_sparse` | Report is essentially empty or extremely short; insufficient content to evaluate |
| 9 | `scope_drift` | Answer is off-topic, addresses a different question, or misses the specific ask |
| 10 | `factual_contradiction` | Internal inconsistency; different sections contradict each other |

**Coverage note:** 83.4% of unsatisfied verdicts (49,442 total) received no tag — the rules capture a meaningful subset but the majority of failures are expressed in vocabulary not matched by the current ruleset. All proportions below are fractions of *all* failed verdicts, not just tagged ones.

---

## 2. Corrected Family Proportions

Two normalizations are reported. **Overall** (primary) uses total failed verdicts as denominator — this is the fraction of all failures that manifest as a given mode. **Conditional** uses the sum of tagged counts as denominator — this is the within-tagged-set distribution and sums to 1.0 per family.

### 2a. Overall proportion (tagged count / total failed verdicts)

| Family | N failed | citation_fabrication | empty_or_sparse | factual_contradiction | missing_evidence | superficial_analysis |
|:---|---:|---:|---:|---:|---:|---:|
| GPT-4o (P0–P8) | 22,816 | **0.035** | 0.008 | 0.032 | **0.057** | 0.011 |
| Local 7B (P9–P10) | 8,938 | 0.004 | **0.058** | 0.024 | **0.046** | 0.022 |

Full table: `_family_mode_prop.csv`

### 2b. Conditional proportion (tagged count / sum of tagged counts per family, sums to 1.0)

| Family | citation_fabrication | empty_or_sparse | factual_contradiction | missing_evidence | superficial_analysis |
|:---|---:|---:|---:|---:|---:|
| GPT-4o (P0–P8) | **0.220** | 0.052 | 0.200 | **0.361** | 0.072 |
| Local 7B (P9–P10) | 0.025 | **0.333** | 0.139 | **0.267** | 0.127 |

Full table: `_family_mode_prop_conditional.csv`

---

## 3. Chi-Square Test (family x failure-mode distribution)

Tests whether GPT-4o (P0-P8) and Local 7B (P9-P10) exhibit qualitatively different failure-mode distributions.

| Statistic | Value |
|:---|---:|
| chi-squared | 992.201 |
| df | 9 |
| p-value | 8.28e-208 |

**Interpretation:** The failure-mode distributions are qualitatively different (p<0.001). The null hypothesis of identical proportions is rejected decisively. The divergence is primarily driven by `citation_fabrication` (8.1x higher rate in GPT-4o) and `empty_or_sparse` (7.0x higher rate in Local 7B).

Full result: `chi2_family_test.md`

---

## 4. Top 3 Failure Modes per Family

### 4a. Overall proportion (recommended for cross-family comparison)

**GPT-4o (P0-P8):**
1. `missing_evidence` — 5.7% of all GPT-4o failed verdicts
2. `citation_fabrication` — 3.5%
3. `factual_contradiction` — 3.2%

**Local 7B (P9-P10):**
1. `empty_or_sparse` — 5.8% of all Local 7B failed verdicts
2. `missing_evidence` — 4.6%
3. `factual_contradiction` — 2.4%

### 4b. Conditional proportion (within-tagged distribution)

**GPT-4o:** missing_evidence (36.1%), citation_fabrication (22.0%), factual_contradiction (20.0%)

**Local 7B:** empty_or_sparse (33.3%), missing_evidence (26.7%), factual_contradiction (13.9%)

---

## 5. Citation Fabrication Gap

**GPT-4o citation_fabrication rate (overall): 3.5%**
**Local 7B citation_fabrication rate (overall): 0.4%**
**Ratio: 8.1x higher in GPT-4o family**

This does not mean GPT-4o is worse overall — GPT-4o produces substantive reports that get scrutinized for citation quality. Local 7B patterns more often fail at the output-generation stage (`empty_or_sparse`), so citation quality is rarely assessed.

---

## 6. Five Case Studies (distinct queries)

| CS | Query ID | Pattern contrast | Key finding |
|:---|:---|:---|:---|
| 1 | `f1b0f094-fa7a-4f18-adbd-f4cd86633f77` | P4 (0.716) vs P0 (0.014) | Architecture wins: STORM multi-perspective decomposition lifts delta=0.703 on a multi-facet query |
| 2 | `dsqa_0063` | P1 (0.753) vs P4 (0.396) | Top-cluster tie: Iterative RAG edges STORM by 0.356 on a factual-grounding query |
| 3 | `a45c277e-55d9-4e7f-b1de-37fc2e19daf6` | Best 3 patterns (ceiling=0.610) | Near-floor: no pattern exceeds 0.610; lowest retrieval ceiling in eval set (threshold note below) |
| 4 | `dsqa_0080` | base_p0, std=0.396 | Max judge disagreement: gpt52 vs sonnet diverge on citation-placeholder strictness |
| 5 | `8e99d8d2-f6b9-4800-83a9-6f56829898fe` | P10 (0.342) vs P9 (0.056) | RL effect: DeepResearcher-7b delta=0.286 over Qwen2.5-7B baseline on same backbone |

All 5 case studies use distinct query IDs.

---

## 7. Most Striking Finding (corrected numbers)

**GPT-4o citation fabrication (3.5% of all failed verdicts) is 8.1x the Local 7B rate (0.4%)**, confirmed significant by chi-square (chi2=992, p<10^-200).

This is the headline cross-family contrast. However, Local 7B's dominant failure is `empty_or_sparse` at 5.8% — representing outright output failure rather than hallucination. The two families fail in structurally different ways, not merely at different rates on the same modes.

---

## 8. Methodology Caveats

1. **Judge vocabulary divergence:** The 10-mode ruleset matches only 16.6% of failed verdicts (83.4% untagged). The rules were designed for interpretability, not completeness. Proportions reflect the detectable failure surface, not total failure mass.

2. **83% untagged:** The majority of unsatisfied verdicts do not match any keyword rule. This likely reflects nuanced reasoning that resists keyword classification. The taxonomy should be treated as a characterization of the *most keyword-salient* failure modes, not a comprehensive failure model.

3. **CS3 threshold not met:** The original "universal failure" criterion (max score <=0.40) was not met by any query in the 90-query eval set. CS3 uses the lowest-ceiling query available (max=0.610). The case study is retitled "Near-floor query — retrieval ceiling limits all patterns."

4. **CS5 P9 report quality:** The selected CS5 query (`8e99d8d2-...`) has P9 report length 1,290 chars — above the 1,000-char filter threshold. Earlier candidates had effectively empty P9 reports (<500 chars), which would have made the delta reflect output failure rather than RL training effect.

5. **Conditional vs overall proportions:** The previous `_family_mode_prop.csv` normalized by sum-of-tagged-counts (conditional), not by total failed verdicts. This inflated reported percentages by approximately 6x (e.g., conditional GPT-4o citation_fabrication was 22.0% vs correct overall 3.5%). The overall proportion is now the primary export.

6. **Single-judge filtering:** Only gpt52 and claude_sonnet are used for taxonomy; claude_opus is excluded to avoid triple-counting. This may slightly undercount failures on queries only covered by opus.
