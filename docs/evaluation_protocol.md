# Evaluation Protocol for Comparative Assessment of Deep Research Patterns

**Version:** 2.0
**Date:** March 2026
**Authors:** Peter Strain

---

## Public Export Note

This document records the historical methodology used for the paper. The GitHub
export ships the API workflow in `deep_research/`, compact inputs, frozen
reference summaries, public tests, and the final PDF. It does not ship the
archived execution engine, local/GPU pattern implementations, raw reports, raw
judge verdict trees, or historical model/search snapshots.

Naming note: this protocol describes six orchestration archetypes (`P0`--`P5`).
The public reference table contains thirteen frozen `base_p*` rows because the
final paper comparison includes variants and ablations. Use
`repro/reference/PATTERN_DICTIONARY.csv` and `repro/PAPER_A_REPRO_MAP.md` for the
public command contract.

Historical model references such as GPT-4o/PTU generation and GPT-5.2 judging
describe the submitted-paper run. Public live reruns use the current configured
API model IDs recorded by the CLI output.

---

## Table of Contents

1. [Research Goal and Framing](#1-research-goal-and-framing)
2. [Diagnostic Questions](#2-diagnostic-questions)
3. [Patterns Under Evaluation](#3-patterns-under-evaluation)
4. [Query Corpus and Sampling Strategy](#4-query-corpus-and-sampling-strategy)
5. [Evaluation Dimensions](#5-evaluation-dimensions)
6. [Process Evaluation Metrics](#6-process-evaluation-metrics)
7. [Measurement Methods](#7-measurement-methods)
8. [External Benchmarks](#8-external-benchmarks)
9. [Statistical Analysis Plan](#9-statistical-analysis-plan)
10. [Ablation Studies](#10-ablation-studies)
11. [Human Evaluation Calibration](#11-human-evaluation-calibration)
12. [Concordance Analysis](#12-concordance-analysis)
13. [Execution Infrastructure](#13-execution-infrastructure)
14. [Reporting Standards](#14-reporting-standards)
15. [Literature Grounding](#15-literature-grounding)
16. [Appendix A: Rubric Criteria Inventory](#appendix-a-rubric-criteria-inventory)
17. [Appendix B: Error Taxonomy](#appendix-b-error-taxonomy)

---

## 1. Research Goal and Framing

### 1.1 Purpose

This protocol compares six automated deep research patterns (`P0`--`P5`) under shared tools and model access. The main measurements are report quality, variance, failure modes, cost, and how orchestration choices change those outcomes.

This is exploratory diagnostic work, not a single-hypothesis test. We do not start from a narrow claim such as "P4 outperforms P0." Instead, the eight diagnostic questions in Section 2 break quality into pieces that can be measured and checked against each other.

### 1.2 Scope

The evaluation covers:

- **6 patterns** spanning the complexity spectrum from single-call baseline to three-level hierarchical control (Section 3).
- **90 evaluation queries** drawn from 5 benchmark sources with stratified sampling across domains, difficulty levels, and answer types (Section 4).
- **7 quality dimensions** with 31 general criteria plus task-specific criteria per query (Section 5).
- **Process metrics** at each pipeline stage: planning, query generation, retrieval, and synthesis (Section 6).
- **Multi-judge ensemble evaluation** with inter-rater reliability measurement (Section 7).
- **External benchmark calibration** against published system scores (Section 8).
- **Nonparametric statistical analysis** following Demsar (2006) recommendations (Section 9).
- **Ablation studies** isolating 12 component contributions across 5 patterns (Section 10).

### 1.3 Design Principles

| Principle | Implementation |
|-----------|---------------|
| Controlled comparison | Historical paper runs used a controlled GPT-4o/PTU generation environment with common search/extraction tools and API endpoints. Public API reruns use the current configured standard OpenAI and Anthropic model IDs and are documented as best-effort. |
| Multi-dimensional assessment | Seven quality dimensions with distinct weights rather than a single holistic score. |
| Measurement triangulation | LLM-as-judge verdicts, agentic citation verification, process metrics, human calibration, and concordance analysis. |
| Reproducibility | Fixed random seeds, deterministic query manifest, checkpoint/resume execution, version-locked dependencies. |
| Statistical rigor | Nonparametric tests appropriate for k-system comparison on n tasks, with multiplicity correction and effect sizes. |
| External validity | Queries drawn from five published benchmarks, not only custom test cases. |

---

## 2. Diagnostic Questions

The evaluation is organized around eight primary diagnostic questions. Each maps to specific measurement methods and analysis techniques.

### Table 1. Diagnostic Questions and Analysis Methods

| ID | Diagnostic Question | What it answers | Primary Analysis |
|----|---------------------|-----------------|-----------------|
| DQ1 | Where does quality variance come from? | Whether variance is driven by pattern choice, query difficulty, quality dimension, or their interactions. | Variance decomposition: pattern, query, dimension, and interaction components. Friedman test for pattern effect. Stratified bootstrap for query-difficulty interaction. |
| DQ2 | What goes wrong when a report is bad? | The taxonomy of failure modes and whether different patterns fail differently. | Error categorization (8 categories, 3 severity levels). Failure clustering by pattern. Distribution of critical errors. |
| DQ3 | Which pipeline components contribute most? | Whether retrieval, planning, synthesis, or iteration drives quality variance. | Component-level process metrics (Section 6) regressed against quality scores. Retrieval metrics as predictors of final quality. |
| DQ4 | What is the cost-quality Pareto frontier? | Whether more expensive patterns (higher token usage) produce proportionally better output. | Multi-objective scatter (total tokens vs. overall score). Pareto front identification. Marginal quality per 10K tokens. |
| DQ5 | Do patterns differ in compute-normalized quality? | Fair comparison when patterns consume different resources. | Quality per token, quality per LLM call, quality per wall-clock second. Rank stability under normalization. |
| DQ6 | How do our patterns compare to production systems? | Calibration against published baselines on shared benchmarks. | Score comparison on DRACO (Perplexity DR 70.5%, Gemini DR 59.0%, OpenAI o3 52.1%). Percentile rank on ResearchQA, DeepSearchQA, LitQA2 leaderboards. |
| DQ7 | Which architectural components actually help? | Causal attribution of quality to specific design decisions. | Ablation studies (Section 10): 12 configurations across P1--P5, with Cliff's Delta effect sizes and Wilcoxon tests per ablation. |
| DQ8 | Does query difficulty interact with architecture? | Whether the "best" pattern depends on query characteristics. | Stratified analysis by difficulty level (simple/moderate/complex). Per-stratum rank distributions. Interaction plots. |

---

## 3. Patterns Under Evaluation

Six patterns are evaluated, spanning a complexity spectrum informed by the modular deep research framework taxonomy (arXiv:2508.12752) and the systematic survey of deep research systems (arXiv:2512.02038).

### Table 2. Pattern Architectures

| ID | Pattern Name | Architecture Category | Core Mechanism | Key Components | Orchestration Complexity |
|----|-------------|----------------------|----------------|----------------|------------------------|
| P0 | Baseline | Single-call RAG | Query + search results fed to a single LLM generation call. No iteration. | LLM caller, web search | Minimal |
| P1 | Iterative RAG | Single-agent linear | Seven-stage pipeline: decompose, search, extract, generate, reflect, re-search. Up to 3 reflection loops. | Query decomposer, retriever, generator, reflector, report assembler | Low |
| P2 | Supervisor + Parallel Workers | Flat multi-agent parallel | Supervisor decomposes query into sub-topics; N async workers search and extract independently; quality gate with gap-fill. | Supervisor, worker pool, quality gate, aggregator | Medium |
| P3 | MERIDIAN | Sequential specialist pipeline | Four specialist roles executed sequentially: Searcher, Topic Miner, Writer, 3-Judge Evaluation Panel. | Specialist chain, topic miner, multi-judge evaluation | Medium-High |
| P4 | Perspective STORM | Perspective-driven multi-agent | Discover diverse perspectives, simulate pairwise expert conversations, triangulate claims across perspectives, synthesize. | Perspective discovery, conversation simulator, triangulator, synthesizer, mind map | High |
| P5 | Hierarchical W&D | Three-level hierarchical | Width phase (broad parallel), Depth phase (targeted deep-dives), Meta-evaluation with budget rebalancing, adaptive W(t) decay schedule. | Width controller, depth controller, meta evaluator, budget allocator, WD schedule, citation verifier, planner | Highest |

### 3.1 Architectural Categories

Following the taxonomy from the modular deep research framework survey (arXiv:2508.12752), these patterns map to:

- **P0:** Retrieval-augmented generation (RAG) baseline. Single retrieval pass, single generation pass.
- **P1:** Iterative RAG with self-refinement. Maps to the "iterative retrieval" module in the modular framework.
- **P2:** Supervisor-worker decomposition. Maps to the "query decomposition + parallel execution" pattern identified in multi-agent research (AutoResearcher, GPT-Researcher).
- **P3:** Sequential specialist pipeline. Maps to the "role specialization" pattern (MERIDIAN, AutoSurvey).
- **P4:** Perspective-driven deliberation. Maps to the "multi-perspective synthesis" pattern (STORM, Co-STORM).
- **P5:** Hierarchical meta-controlled system. Maps to the "hierarchical planning with adaptive resource allocation" pattern. Extends the survey taxonomy with explicit width-depth scheduling.

### 3.2 Shared Infrastructure

All patterns share:

- **LLM backbone:** GPT-4o on Azure PTU (provisioned throughput, deterministic latency).
- **Web search:** Bing via Azure OpenAI Responses API.
- **Academic search:** Semantic Scholar API + arXiv.
- **Source extraction:** trafilatura with BeautifulSoup fallback.
- **Rate limiting:** Semaphore(12) + AsyncLimiter(200 RPM) via `_PTURateGate`.
- **Cost tracking:** Per-call token counting with model-specific pricing.
- **Maximum budget:** $2.00 USD per individual query run.

---

## 4. Query Corpus and Sampling Strategy

### 4.1 Composition

The evaluation corpus contains 90 queries drawn from 5 sources via stratified sampling. The target composition balances external benchmark validity with diagnostic coverage.

### Table 3. Query Corpus Composition

| Source | Count | Selection Method | Domain Coverage | What It Calibrates |
|--------|-------|-----------------|-----------------|-------------------|
| Custom test queries | 5 | All (hand-crafted) | NLP/AI | Controlled queries with known expected elements; deep diagnostic value |
| DRACO (Perplexity, 2025) | 40 | Stratified by domain (10 domains, 4 per domain) | Academic, Finance, Shopping, Technology, Medicine, Legal, UX, Needle-in-Haystack, General Knowledge, Personal Assistant | External benchmark calibration against Perplexity DR, Gemini DR, o3 |
| DeepSearchQA (Google, 2025) | 20 | Stratified by problem category | 17 fields including Politics, Media, Education, Health, Geography | Multi-step retrieval requiring cross-source synthesis |
| ResearchQA (Li et al., 2025) | 15 | Stratified by academic field | 75 academic fields (sampled) | Academic research quality; PhD-annotated rubrics |
| LitQA2 (FutureHouse, 2024) | 10 | Random sample | Scientific literature | Factual precision on scientific MCQs; calibration against PaperQA2 |
| **Total** | **90** | | | |

### 4.2 Difficulty Stratification

Queries are classified into three difficulty levels using a keyword heuristic validated against query properties:

| Difficulty | Indicators | Typical Word Count |
|-----------|-----------|-------------------|
| Simple | Starts with "what is", "who is", "define", "list"; single question mark; < 15 words | < 15 words |
| Moderate | Default category; does not match simple or complex indicators | 15--40 words |
| Complex | Contains "compare and contrast", "analyze the tradeoffs", "critically assess"; multiple question marks (>= 2); or > 40 words | > 40 words |

The historical classification heuristic flagged short queries classified as complex and long multi-part queries classified as simple for manual review.

### 4.3 Manifest and Reproducibility

The complete query selection is serialized to `data/eval_queries_v2.json` (version 2.0 manifest). This file contains:

- Query text, ID, source, domain, and difficulty classification.
- Full V2 rubric for each query (criteria, dimension weights, source-specific overrides).
- Expected elements and reference answers where available.
- Metadata from the originating benchmark.

All sampling uses fixed random seed 42 for reproducibility. The manifest is version-controlled and must not be regenerated between evaluation runs.

---

## 5. Evaluation Dimensions

### 5.1 Seven Quality Dimensions

Report quality is assessed across seven dimensions. The historical V2 rubric system assigns 31 general criteria across these dimensions plus task-specific criteria generated per query.

### Table 4. Evaluation Dimensions

| Dimension | Weight | Criteria Count | Description |
|-----------|--------|---------------|-------------|
| Information Recall (Coverage) | 0.25 | 5 general + task-specific | Completeness of topic coverage. Major aspects addressed, both advantages and limitations discussed, recent developments included, multiple perspectives represented, practical implications covered. |
| Factual Accuracy | 0.25 | 8 | Correctness of claims, proper use of technical terminology, accuracy of numbers/dates/benchmarks, correct chronology, supported comparison claims, internal consistency, accurate representation of limitations, correct identification of state-of-the-art. |
| Analytical Depth | 0.15 | 4 | Synthesis across sources (not just summarization), identification of patterns/trade-offs/mechanisms, connections drawn between aspects, distinction between established and emerging claims. |
| Citation and Attribution | 0.15 | 6 (4 citation + 2 attribution) | Inline citations to named sources, consistent formatting, appropriate number of distinct sources (minimum 5), source diversity across sections, traceability of claims to named sources, clear distinction between author analysis and source material. |
| Logical Coherence (Organization) | 0.10 | 4 | Clear introduction, logical section progression, synthesizing conclusion, focused paragraphs with smooth transitions. |
| Instruction Following | 0.10 | 4 | Direct address of the research question, appropriate scope, coverage of all implied sub-questions, format/structure matching query expectations. |
| Organization | 0.05 | 0 (subsumed) | Structural quality is captured within Logical Coherence. The separate 5% weight allows fine-grained scoring in downstream analysis. |

### 5.2 Weight Justification

Dimension weights reflect the relative importance of each quality aspect for research report utility:

- **Information Recall and Factual Accuracy** (0.25 each, total 0.50) dominate because a research report that is incomplete or inaccurate fails its primary purpose regardless of presentation quality.
- **Analytical Depth** (0.15) captures the value-add of synthesis over raw information retrieval. A system that merely copies source material adds less value than one that identifies patterns and trade-offs.
- **Citation and Attribution** (0.15) is elevated relative to V1 (which had 0.15 for citation alone) because attribution quality was identified as the universal bottleneck across all patterns in preliminary evaluation.
- **Logical Coherence** and **Instruction Following** (0.10 each) are important but secondary to content quality.
- **Organization** (0.05) carries minimal weight because structural deficiencies rarely render a report useless if the content is strong.

### 5.3 Source-Specific Weight Overrides

Certain benchmark sources use modified dimension weights to reflect their evaluation priorities:

| Source | Factual Accuracy | Coverage | Analytical Depth | Citation | Organization | Instruction Following | Attribution |
|--------|-----------------|----------|-----------------|----------|-------------|----------------------|-------------|
| Default | 0.25 | 0.25 | 0.15 | 0.10 | 0.10 | 0.10 | 0.05 |
| LitQA2 | 0.35 | 0.15 | 0.20 | 0.10 | 0.05 | 0.10 | 0.05 |
| DeepSearchQA | 0.30 | 0.20 | 0.15 | 0.10 | 0.10 | 0.10 | 0.05 |
| ResearchQA | 0.25 | 0.25 | 0.20 | 0.10 | 0.05 | 0.10 | 0.05 |
| DRACO | 0.25 | 0.25 | 0.15 | 0.10 | 0.10 | 0.10 | 0.05 |

LitQA2 elevates factual accuracy because it tests precise scientific knowledge. ResearchQA elevates analytical depth because it assesses academic synthesis. All source-specific overrides sum to 1.0.

---

## 6. Process Evaluation Metrics

Beyond final report quality, we instrument each pipeline stage to measure component-level performance. These metrics address DQ3 (which pipeline components contribute most?) and DQ4/DQ5 (cost-quality trade-offs).

### 6.1 Planning Stage

| Metric | Description | Patterns |
|--------|-------------|----------|
| Sub-query count | Number of sub-queries generated from decomposition | P1, P2, P3, P5 |
| Sub-query diversity | Semantic diversity of generated sub-queries (embedding cosine distance) | P1, P2, P5 |
| Perspective count | Number of distinct perspectives discovered | P4 |
| Planning token cost | Tokens consumed during planning/decomposition | All |
| Planning wall-clock time | Seconds spent in planning phase | All |

### 6.2 Query Generation Stage

| Metric | Description | Patterns |
|--------|-------------|----------|
| Search queries generated | Total search queries issued to web and academic APIs | All |
| Query specificity | Average word count of search queries (proxy for specificity) | All |
| Academic vs. web query ratio | Proportion of queries directed to academic sources | All |

### 6.3 Retrieval Stage

Retrieval metrics were computed by the historical retrieval evaluator following the DeepResearchBench RACE+FACT model and SurGE three-level framework.

| Metric | Description | Computation |
|--------|-------------|-------------|
| Total sources retrieved | Raw count of source documents fetched | Direct count |
| Unique URLs | Deduplicated source count | URL set cardinality |
| URLs with full content | Sources where extraction yielded > 50 characters | Threshold count |
| Academic source ratio | Proportion from academic domains (arXiv, Semantic Scholar, PubMed, etc.) | Domain classification against 30+ academic domain patterns |
| Source diversity (Shannon entropy) | Diversity of domain distribution; higher entropy = more diverse | H = -sum(p_i * log2(p_i)) over domain proportions |
| Average content length | Mean characters per extracted source | Arithmetic mean |
| Median content length | Median characters per extracted source | 50th percentile |
| Domain distribution | Count of sources per domain | Counter over extracted domains |

### 6.4 Synthesis Stage

| Metric | Description | Computation |
|--------|-------------|-------------|
| Total sections | Number of markdown sections in final report | Regex extraction of `# ` headers |
| Total words | Word count of final report | Split-and-count |
| Total claims | Atomic factual claims extracted by citation verifier | LLM decomposition (max 50) |
| Attributed claims | Claims with inline citation markers | Citation marker regex matching |
| Attribution rate | attributed_claims / total_claims | Ratio |
| Unique sources cited | Distinct source URLs referenced in citations | URL set cardinality |
| Citation density | Citations per 1,000 words | (citation_marker_count / word_count) * 1000 |
| Has abstract | Whether report contains an abstract section | Regex match for `# Abstract` |
| Has conclusion | Whether report contains a conclusion section | Regex match for `# Conclusion` or synonyms |
| Average section length | Mean words per section | Arithmetic mean |

### 6.5 Three-Level Citation Accuracy (SurGE Model)

Following the SurGE framework, each citation is evaluated at three levels of granularity using LLM-as-judge:

| Level | Name | Question | Method |
|-------|------|----------|--------|
| 1 | Doc-Accuracy | Is the cited source thematically relevant to the report topic? | LLM classifies source title + summary against report topic |
| 2 | Section-Accuracy | Is the citation placed in a topically appropriate section? | LLM checks source against section title and content excerpt |
| 3 | Sentence-Accuracy | Does the cited source support the specific claim in the sentence? | NLI-style entailment check: source content against citing sentence |

Up to 20 citations per report are evaluated. All three levels use temperature 0.1 with JSON structured output.

### 6.6 Cost Metrics

| Metric | Description |
|--------|-------------|
| Total tokens | Input + output tokens across all LLM calls |
| Total input tokens | Tokens consumed as input (prompt) |
| Total output tokens | Tokens generated as output (completion) |
| LLM call count | Total number of LLM API calls |
| Cost (USD) | Estimated cost based on model pricing ($0/token for PTU, market rates for standard) |
| Elapsed seconds | Wall-clock execution time |
| Quality per token | Overall quality score / total tokens * 10,000 |
| Quality per LLM call | Overall quality score / LLM call count |

---

## 7. Measurement Methods

### 7.1 Multi-Judge Ensemble Evaluation

The primary quality measurement used a historical LLM-as-judge ensemble to address the known fragility of single-judge evaluation.

#### 7.1.1 Architecture

- **Judge model:** GPT-5.2 on Azure (standard deployment, not PTU), configured via `JUDGE_MODEL`.
- **Ensemble configuration:** Multiple judge instances (potentially different models or endpoints) each execute multiple passes.
- **Passes per judge:** 3 (configurable via `EVAL_PIPELINE.passes_per_judge`).
- **Total evaluations per report:** n_judges * passes_per_judge (e.g., 2 judges * 3 passes = 6 evaluations).
- **Concurrency:** Semaphore(3) limiting concurrent judge API calls.
- **Retry policy:** 8 retries with exponential backoff (2^attempt * 2.0s base, capped at 30s) plus 0--2s random jitter. Handles RateLimitError, APIConnectionError, APITimeoutError, InternalServerError.
- **Temperature:** 0.1 (low but non-zero to permit some response variation across passes).
- **Max tokens:** 8,192 per judge response.
- **Response format:** JSON mode enforced (`response_format: {"type": "json_object"}`).
- **Report truncation:** Reports exceeding 12,000 words are truncated with a `[... report truncated for evaluation ...]` marker.

#### 7.1.2 Scoring Protocol (DRACO Methodology)

Each judge pass evaluates the report against the full rubric using binary DRACO verdicts:

1. The judge receives a system prompt calibrated with worked examples demonstrating SATISFIED and NOT_SATISFIED verdicts.
2. For each criterion, the judge provides:
   - **Verdict:** SATISFIED or NOT_SATISFIED (binary, no partial credit).
   - **Evidence:** Brief quote or reference to specific report content.
   - **Reasoning:** One-sentence explanation of the judgment.
3. Dimension scores are computed as weighted satisfaction rates: `score_dim = sum(|w_i| * met_i) / sum(|w_i|)` where `w_i` is the criterion weight (1.0 for general criteria, variable for DRACO criteria) and `met_i` is 1 if SATISFIED, 0 otherwise.
4. Overall score is the weighted sum of dimension scores: `score_overall = sum(score_dim * weight_dim)`.

Negative-weight criteria (DRACO critical failures) are handled with inverted logic: SATISFIED means the bad behavior is present (penalize), NOT_SATISFIED means it is absent (reward).

#### 7.1.3 Position Bias Mitigation

Criteria order is randomized per judge pass to mitigate position bias (Zheng et al., 2023):

- A deterministic seed is computed from `hash((judge_label, pass_number, query_id))`.
- The criteria list is shuffled using this seed before building the judge prompt.
- A mapping from shuffled index to original criterion index is maintained for verdict parsing.
- Different passes see different criteria orderings, ensuring that no criterion is systematically advantaged or disadvantaged by its position.

#### 7.1.4 Ensemble Aggregation

The ensemble uses majority-vote aggregation (SE-Jury style):

1. For each criterion, count SATISFIED votes across all passes from all judges.
2. A criterion is MET if strictly more than 50% of votes are SATISFIED.
3. Dimension scores and overall score are recomputed from majority-voted criteria.

This is more robust than mean aggregation because it discounts outlier passes.

### 7.2 Inter-Rater Reliability Metrics

Three reliability metrics are computed for every ensemble evaluation:

| Metric | Scope | Interpretation |
|--------|-------|---------------|
| **Intra-judge consistency** (flip rate) | Per judge | Fraction of criteria where a judge gives different verdicts across passes. Low flip rate (< 0.15) indicates stable judgment. |
| **Cohen's kappa / Fleiss' kappa** | Inter-judge | Agreement between judges on majority verdicts. kappa > 0.6 = substantial agreement; > 0.8 = almost perfect. Cohen's kappa for 2 judges; Fleiss' kappa for 3+. |
| **Krippendorff's alpha** | All passes | Agreement across all individual passes (treating each pass as a separate rater). Captures both intra-judge and inter-judge variance. alpha > 0.667 = tentatively acceptable; > 0.8 = reliable. |

Per-dimension agreement is also computed to identify dimensions where judges disagree most, informing the reliability of per-dimension comparisons.

### 7.3 Agentic Citation Verification (SAFE + NLI)

Independent of the rubric-based judge, citation quality was measured by a historical agentic citation verifier following the SAFE (Google, 2024) and FActScore methodologies:

**Pipeline:**

1. **Claim extraction:** An LLM decomposes the report into atomic, independently verifiable factual claims (maximum 50 per report). Opinions, hedged statements, and meta-commentary are excluded.
2. **Citation matching:** Claims with inline citation markers (e.g., `[1]`, `[3]`) are matched to Citation objects from the report metadata. URLs are resolved from the citation reference list.
3. **Cited claim verification:** For each claim with a URL, the source is fetched (via the shared URL extractor), and an NLI (natural language inference) check determines whether the source entails the claim. Verdicts: `supported`, `not_supported`, `unverifiable`, `source_unavailable`.
4. **Uncited claim verification (optional):** If a web searcher is available, uncited claims are verified by searching the web and checking top results for supporting evidence.
5. **DOI tracking:** DOIs are extracted from report text and cited URLs, then compared against reference DOIs (for LitQA2 queries) to compute DOI recall.

**Output metrics:**

| Metric | Definition |
|--------|-----------|
| Citation precision | supported / (supported + not_supported) |
| Citation recall | claims_with_citations / total_claims |
| Attribution accuracy | supported / claims_with_citations |
| Source availability | (total - source_unavailable) / total |
| DOI recall | matched_dois / reference_dois (LitQA2 only) |

The NLI check uses strict entailment: the source must clearly confirm the claim, not merely be topically related. Temperature is set to 0.1 with maximum 1,024 tokens per NLI judgment.

### 7.4 Pairwise Arena (Elo + Bradley-Terry)

For head-to-head comparison, a pairwise arena can be constructed where an LLM judge directly compares two reports on the same query and selects the better one (with ties allowed). From pairwise preferences:

- **Elo ratings** are computed using the standard update rule with K=32.
- **Bradley-Terry model** parameters are estimated via maximum likelihood, providing strength parameters for each pattern on a common scale.

The pairwise arena complements criterion-level scoring by capturing holistic quality judgments that may not decompose neatly into rubric dimensions.

---

## 8. External Benchmarks

Five external benchmarks provide calibration against published system scores. Each benchmark emphasizes different quality aspects.

### Table 5. External Benchmark Calibration

| Benchmark | Source | Size | What It Calibrates | Published Top Scores | Our Metric |
|-----------|--------|------|--------------------|---------------------|------------|
| **DRACO** (Perplexity, 2025) | Industry | 10 domains, weighted rubrics | Overall research quality at production scale. Expert-crafted rubrics with 30--50 criteria per task. Binary MET/UNMET verdicts. | Perplexity DR: 70.5%, Gemini DR: 59.0%, OpenAI o3: 52.1% | Weighted criterion satisfaction rate, mapped to DRACO rubric format via `build_rubric_from_draco()`. |
| **DRB-II** (DeepResearchBench-II) | Academic | Multi-domain, multi-step | Complex multi-step research requiring cross-source reasoning. Evaluates both retrieval (RACE) and generation (FACT). | Varies by system | RACE+FACT decomposition via the historical retrieval evaluator. |
| **DR.BENCH** (Deep Research Benchmark) | Academic | Standardized tasks | Baseline calibration for research system quality. | Varies by task | Standard rubric-based scoring. |
| **ResearchQA** (Li et al., 2025) | Academic | 21,414 queries, 8 fields, 160K+ rubric items | Academic research synthesis quality. PhD-annotated rubrics across 75 fields. Provides the highest-quality human judgment calibration. | Perplexity Sonar DR: 75.29% | Rubric items converted to V2 criteria by the historical conversion scripts. |
| **DeepSearchQA** (Google, 2025) | Industry | 900 prompts, 17 fields | Multi-step retrieval across diverse domains. Expert-validated answers. Includes Set Answer and Free-form answer types. | Varies by system | Answer-type-specific criteria plus standard V2 rubric. |
| **LitQA2** (FutureHouse, 2024) | Academic | 199 MCQs | Scientific literature factual precision. Calibrates against PaperQA2 (85.2% superhuman precision). Tests whether systems can identify specific scientific facts from literature. | PaperQA2: 85.2% (vs. 73.8% human) | Correct answer identification + distractor avoidance. Elevated factual accuracy weight (0.35). |
| **STORM FreshWiki** | Academic | Wikipedia articles | Long-form article generation quality. Calibrates organizational structure and coverage breadth. | STORM: various metrics | Coverage and organization emphasis. |

### 8.1 Benchmark-Specific Rubric Conversion

Each benchmark has its own rubric format. Historical conversion scripts translated these into the unified V2 rubric format:

- **DRACO:** Sections and weighted criteria are preserved. DRACO criterion weights (1--20) map to the `Criterion.weight` field. Negative-weight criteria (critical failures) are handled with inverted scoring logic. All DRACO criteria are mapped to the `coverage` dimension with their original weights.
- **DeepSearchQA:** Answer-type-specific criteria are generated (Set Answer requires comprehensive lists; Free-form requires direct address). Reference answers provide coverage criteria.
- **ResearchQA:** PhD-annotated rubric items are converted to coverage criteria. Field metadata is preserved for stratified analysis.
- **LitQA2:** Multiple-choice structure generates factual accuracy criteria (correct answer support, distractor avoidance). Elevated factual accuracy weight (0.35).

---

## 9. Statistical Analysis Plan

Statistical analysis follows the recommendations of Demsar (2006) for comparing multiple classifiers/systems over multiple datasets, supplemented by modern robust aggregation methods from Agarwal et al. (2021).

### 9.1 Omnibus Test: Friedman + Iman-Davenport

**Purpose:** Test whether there is a statistically significant difference in performance among the k=6 patterns across n=90 queries.

**Procedure:**

1. For each query, rank the 6 patterns by overall score (1 = best). Ties are handled by average ranks.
2. Compute the Friedman statistic: chi2_F = (12n / (k(k+1))) * (sum(R_j^2) - k(k+1)^2/4).
3. Compute the Iman-Davenport F correction: F_F = ((n-1) * chi2_F) / (n(k-1) - chi2_F), which has better statistical power than the raw Friedman chi-squared.
4. Reject H0 (all patterns perform equally) if F_F exceeds the critical value of F(k-1, (k-1)(n-1)) at alpha = 0.05.

The Iman-Davenport correction is preferred because the Friedman chi-squared is known to be conservative (Iman & Davenport, 1980).

### 9.2 Post-Hoc Pairwise Tests

If the omnibus test rejects H0, proceed to pairwise comparisons:

#### 9.2.1 Nemenyi Test with Critical Difference

**Purpose:** Identify which specific pairs of patterns differ significantly.

**Procedure:**

1. Compute the critical difference: CD = q_alpha * sqrt(k(k+1) / (6n)), where q_alpha is the studentized range statistic divided by sqrt(2).
2. Two patterns differ significantly if |R_i - R_j| > CD.
3. Visualize via a critical difference diagram (Demsar, 2006) showing patterns ordered by average rank with horizontal bars connecting groups that are not significantly different.

#### 9.2.2 Wilcoxon Signed-Rank Tests with Holm-Bonferroni Correction

**Purpose:** More powerful pairwise comparison than Nemenyi when specific pairs are of interest.

**Procedure:**

1. For each pair (i,j), compute the Wilcoxon signed-rank statistic on the score differences.
2. Apply Holm-Bonferroni correction for C(k,2) = 15 pairwise comparisons: sort p-values ascending, reject the i-th test if p_i < alpha / (m - i + 1) where m = 15.
3. Report corrected p-values alongside raw p-values.

### 9.3 Effect Sizes: Cliff's Delta

**Purpose:** Quantify the magnitude of pairwise differences (complementing p-values with practical significance).

**Procedure:**

1. For each pair (i,j), compute Cliff's Delta: delta = (count(x_i > x_j) - count(x_i < x_j)) / (n_i * n_j).
2. Interpret using standard thresholds:

| |delta| | Interpretation |
|---------|---------------|
| < 0.147 | Negligible |
| 0.147 -- 0.330 | Small |
| 0.330 -- 0.474 | Medium |
| >= 0.474 | Large |

Cliff's Delta is preferred over Cohen's d because it does not assume normality and is robust to outliers (appropriate for bounded scores on heterogeneous queries).

### 9.4 Bootstrap Confidence Intervals

**Purpose:** Provide uncertainty estimates for all reported statistics.

**Procedure:**

1. Use bias-corrected and accelerated (BCa) bootstrap with 10,000 resamples (`EVAL_PIPELINE.bootstrap_resamples`).
2. If BCa fails (degenerate jackknife), fall back to percentile bootstrap.
3. Report 95% confidence intervals for:
   - Per-pattern overall scores (mean, IQM).
   - Per-pattern per-dimension scores.
   - Pairwise score differences.
   - Rank distributions.

### 9.5 Interquartile Mean (IQM)

**Purpose:** Robust central tendency that is less sensitive to outliers than the arithmetic mean (Agarwal et al., 2021).

**Procedure:**

1. Discard the bottom 25% and top 25% of scores.
2. Compute the mean of the remaining 50%.
3. Report IQM alongside mean for all per-pattern aggregates.
4. Bootstrap the IQM to obtain confidence intervals.

IQM is the recommended aggregate for reinforcement learning benchmarks (Agarwal et al., 2021) and is equally appropriate here because individual query scores may have heavy tails.

### 9.6 Variance Decomposition (DQ1)

**Purpose:** Quantify how much quality variance is attributable to pattern choice, query difficulty, quality dimension, and their interactions.

**Procedure:**

1. Structure the data as a three-way layout: pattern (6) x query difficulty (3) x dimension (7).
2. Compute sum of squares attributable to each factor.
3. Report the proportion of total variance explained by:
   - Pattern main effect.
   - Query difficulty main effect.
   - Dimension main effect.
   - Pattern x difficulty interaction.
   - Residual (within-cell variance).

This is a descriptive decomposition, not a formal ANOVA (the assumptions for ANOVA are unlikely to hold with bounded, non-normal scores).

### 9.7 Stratified Analysis (DQ8)

**Purpose:** Test whether the "best" pattern depends on query characteristics.

**Procedure:**

1. Partition queries by difficulty level (simple, moderate, complex).
2. Within each stratum, repeat the Friedman test and compute per-pattern means with bootstrap CIs.
3. Compare per-stratum rankings to the overall ranking.
4. Generate interaction plots: x-axis = difficulty level, y-axis = mean quality, one line per pattern.
5. Identify crossover interactions (patterns whose relative ranking reverses across strata).

### 9.8 Power Analysis

**Purpose:** Determine whether the sample size (n=90) provides adequate power for the planned comparisons.

**Procedure:**

1. Estimate the minimum detectable effect size for the Friedman test at alpha=0.05, power=0.80.
2. Report the achieved power for the observed effect size.
3. If power < 0.80, note which comparisons are underpowered.

### 9.9 Rank Stability

**Purpose:** Assess how robust pattern rankings are to perturbation.

**Procedure:**

1. Bootstrap rank distributions (10,000 resamples): for each bootstrap sample, rank patterns, accumulate rank counts.
2. Report the probability of each pattern achieving each rank.
3. Compute 95% CI for each pattern's rank.
4. Patterns whose rank CIs do not overlap are robustly distinguishable.

---

## 10. Ablation Studies

### 10.1 Purpose (DQ7)

Ablation studies isolate the contribution of individual architectural components. Each ablation disables or simplifies a single component while leaving the rest of the pattern intact. The resulting quality difference measures the component's contribution.

### 10.2 Ablation Registry

Twelve ablation configurations span five patterns:

### Table 6. Ablation Configurations

| ID | Base Pattern | Component Removed | Modification | Expected Effect |
|----|-------------|-------------------|-------------|-----------------|
| `p4_no_conversations` | P4 Perspective STORM | Conversation simulator | Skip pairwise expert conversations | Lower analytical depth from loss of multi-perspective dialogue |
| `p4_no_triangulation` | P4 Perspective STORM | Triangulator | Include all claims without cross-perspective agreement check | More factual errors, possibly broader coverage |
| `p4_fixed_perspectives` | P4 Perspective STORM | Perspective discovery | Use 3 fixed generic perspectives instead of LLM-discovered ones | Less diverse source retrieval and analysis angles |
| `p3_no_quality_eval` | P3 MERIDIAN | Quality evaluator | Skip evaluation/revision, take first draft | Lower organization and analytical depth |
| `p3_no_topic_mining` | P3 MERIDIAN | Topic miner | Pass extractions directly to writer | Less structured analysis, lower coherence |
| `p5_fixed_width` | P5 Hierarchical W&D | Width-depth schedule decay | Fixed width=2 throughout, alpha=1.0 | Less initial breadth, more uniform resource allocation |
| `p5_no_meta_eval` | P5 Hierarchical W&D | Meta evaluator | Skip meta-evaluation and budget rebalancing | No adaptive gap-filling, possibly incomplete coverage |
| `p5_no_citation_verify` | P5 Hierarchical W&D | Citation verifier | Skip internal citation spot-check | Slightly lower citation quality, faster execution |
| `p2_sequential_workers` | P2 Supervisor Parallel | Parallel dispatch | Process sub-topics sequentially (max_workers=1) | Slower but possibly more coherent |
| `p2_no_quality_gate` | P2 Supervisor Parallel | Quality gate | Skip quality gate evaluation and gap-fill | Lower coverage from no gap-filling |
| `p1_single_iteration` | P1 Iterative RAG | Reflection loop | Single retrieval pass, no reflection (max_iterations=1) | Should approximate P0, demonstrating reflection value |
| *(P0 has no ablatable components -- it is the minimal baseline)* | | | | |

### 10.3 Ablation Execution

- **Query subset:** A representative subset of the 90-query corpus (e.g., 10 queries spanning all difficulty levels).
- **Metrics:** Same V2 rubric evaluation as the full pipeline. Multi-judge ensemble with same configuration.
- **Checkpointing:** Per-ablation, per-query checkpoints (`checkpoints/ablations/{ablation_id}/{query_id}.json`) for resume after interruption.
- **Budget:** $2.00 per ablation run (same as full pattern runs).

### 10.4 Ablation Analysis

For each ablation:

1. Compute mean quality score for the base pattern and the ablated variant across the query subset.
2. Compute score delta: `base_mean - ablated_mean` (positive = component helps).
3. Compute relative change: `(base - ablated) / base * 100%`.
4. Wilcoxon signed-rank test for significance.
5. Cliff's Delta for effect size with standard interpretation (negligible/small/medium/large).
6. Report as `AblationComparison` with explicit expected-vs-observed effect comparison.

---

## 11. Human Evaluation Calibration

Human evaluation serves as a calibration layer for the automated judge, not as the primary evaluation method. The full protocol is defined in `docs/human_evaluation_protocol.md`. Key parameters:

| Parameter | Value |
|-----------|-------|
| Sample size | 15% of reports (~81 from 540 total) |
| Stratification | Minimum 2 reports per pattern per difficulty level |
| Evaluators per report | 3 (for inter-annotator agreement) |
| Evaluator qualification | PhD student/postdoc in relevant domain or 3+ years research experience |
| Blinding | Evaluators blinded to pattern identity |
| Calibration | 3 practice reports before starting |
| Session limit | Maximum 10 reports per session to avoid fatigue |

### 11.1 Evaluation Tasks

1. **Factual accuracy assessment** (~15 min/report): Mark each claim as CORRECT, INCORRECT, or UNVERIFIABLE. Scoring: CORRECT / (CORRECT + INCORRECT).
2. **Citation quality assessment** (~10 min/report): Check source existence, relevance, and support for the citing sentence.
3. **Overall quality rating** (~5 min/report): Holistic 1--5 scale plus forced ranking of all 6 patterns for the same query.

### 11.2 Judge-Human Agreement Metrics

- Per-dimension correlation (Spearman's rho) between judge scores and human scores.
- Per-criterion agreement rate (proportion of criteria where judge and human majority agree).
- Identification of dimensions where the judge is least reliable (lowest kappa).
- Systematic bias analysis: does the judge systematically over- or under-rate specific dimensions?

---

## 12. Concordance Analysis

### 12.1 Purpose

Historical concordance analysis quantified how sensitive pattern rankings are to the choice of evaluation method. If different measurement approaches (rubric-based judge, citation verification, process metrics, human evaluation) produce the same ranking, conclusions are robust. If rankings diverge, the disagreement itself is informative.

### 12.2 Methods

| Metric | What It Measures | Interpretation |
|--------|-----------------|---------------|
| Kendall's W | Overall concordance across all evaluation methods | W = 1: perfect agreement on ranking. W = 0: no agreement. |
| Pairwise Kendall's tau | Agreement between each pair of evaluation methods | tau = 1: identical rankings. tau = -1: reversed rankings. |
| Rank variance per pattern | Stability of each pattern's rank across methods | Low variance = robust conclusion. High variance = method-dependent conclusion. |
| Most stable / volatile patterns | Patterns whose ranking is most / least method-dependent | Identifies patterns about which we can make confident claims. |

### 12.3 Evaluation Methods Compared

The concordance analysis compares rankings produced by:

1. **Multi-judge ensemble overall score** (primary method).
2. **Per-dimension scores** (7 separate rankings).
3. **Citation verification metrics** (citation precision, attribution accuracy).
4. **Process metrics** (retrieval quality, source diversity).
5. **Human evaluation scores** (where available).
6. **Cost-normalized quality** (quality per token).

---

## 13. Execution Infrastructure

### 13.1 Execution Pipeline

The historical private execution pipeline orchestrated 540+ pattern runs (6 patterns x 90 queries):

| Feature | Implementation |
|---------|---------------|
| Checkpoint/resume | Per-pattern-per-query JSON checkpoint files. Completed runs are skipped on restart. |
| Concurrency | Configurable (`max_concurrent_runs = 2` default). Limited by PTU throughput. |
| Error handling | Content filter failures, budget exceeded ($2.00 cap), API timeouts -- all handled with status codes. |
| Progress monitoring | Periodic status reports: completed/total, success/fail counts, elapsed time. |
| Cost tracking | Per-run token counts and estimated USD cost. Total tokens, input tokens, output tokens, LLM call count. |
| Metadata capture | Timestamp, repeat index (for multiple runs per query), environment metadata. |
| Run statuses | `success`, `content_filter`, `budget_exceeded`, `error`, `skipped` |

### 13.2 Judge Pipeline

The historical private judge pipeline orchestrated multi-judge evaluation of all generated reports:

| Feature | Implementation |
|---------|---------------|
| Input | Saved reports from execution pipeline |
| Checkpoint/resume | Per-report evaluation checkpoints |
| Multi-judge | Configurable judge instances via `JudgeConfig` |
| Concurrency | Semaphore(3) for judge API calls |
| Output | `EnsembleResult` per report with reliability metrics |

### 13.3 Rate Limiting Architecture

| Endpoint | Concurrency | Rate Limit | Timeout |
|----------|------------|------------|---------|
| PTU (GPT-4o) | Semaphore(12) | AsyncLimiter(200 RPM) | Read: 300s, Connect: 30s |
| Judge (GPT-5.2) | Semaphore(3) | 8 retries, exponential backoff | Read: 600s, Connect: 30s |
| Citation verifier | Semaphore(5) | Shared with PTU | Read: 300s |

### 13.4 Reproducibility Controls

| Control | Implementation |
|---------|---------------|
| Random seeds | Fixed seed 42 for all sampling, stratification, and difficulty classification |
| Query manifest | `data/eval_queries_v2.json` locked after generation; must not be regenerated between runs |
| Criterion shuffling | Deterministic per-pass shuffle seeds via `hash((judge_label, pass_number, query_id))` |
| Environment capture | `get_environment_metadata()` records Python version, platform, timestamp, model deployments |
| Checkpoint versioning | JSON checkpoints include timestamp and metadata for auditability |

---

## 14. Reporting Standards

### 14.1 Required Tables

All evaluation reports must include:

1. **Overall ranking table:** Pattern, mean score, IQM, 95% CI, rank.
2. **Per-dimension breakdown:** Pattern x dimension heatmap with scores and significance markers.
3. **Pairwise comparison table:** All 15 pairs with Cliff's Delta, corrected p-value, and significance flag.
4. **Cost-quality table:** Pattern, total tokens, LLM calls, elapsed time, quality/token, quality/call.
5. **Reliability table:** Per-evaluation kappa, alpha, flip rate.
6. **Ablation results:** Component, delta, relative change, effect size, significance.

### 14.2 Required Figures

1. **Critical difference diagram** (Demsar, 2006): Patterns ordered by average rank with CD bars.
2. **Bootstrap rank distributions:** Heatmap of P(pattern achieves rank r).
3. **Score distribution violin plots:** Per-pattern score distributions across queries.
4. **Interaction plot:** Difficulty level x pattern mean quality.
5. **Pareto frontier:** Tokens vs. quality scatter with Pareto front highlighted.
6. **Radar charts:** Per-pattern dimension profiles.
7. **Error taxonomy distribution:** Stacked bar chart of error categories by pattern.
8. **Concordance matrix:** Heatmap of pairwise Kendall's tau between evaluation methods.

### 14.3 Uncertainty Reporting

- All point estimates must be accompanied by 95% bootstrap confidence intervals.
- p-values must be reported alongside effect sizes (neither alone is sufficient).
- Statistical significance at alpha=0.05 must be distinguished from practical significance (effect size interpretation).
- The number of statistical tests and the multiplicity correction method must be stated explicitly.
- Non-significant results must be reported without euphemism ("we did not find evidence for a difference" rather than "the patterns performed similarly").

### 14.4 Negative Result Reporting

Negative or null findings (e.g., "no significant difference between P3 and P4") are reported with the same rigor as positive findings. The achieved statistical power for each non-significant comparison must be stated so that readers can distinguish "no effect" from "insufficient power to detect an effect."

---

## 15. Literature Grounding

This evaluation protocol is grounded in the following key references.

### 15.1 Statistical Methodology

- **Demsar, J. (2006).** "Statistical Comparisons of Classifiers over Multiple Data Sets." *Journal of Machine Learning Research*, 7, 1--30. Provides the template for our statistical analysis: Friedman test, Iman-Davenport correction, Nemenyi post-hoc, critical difference diagrams, and Holm-Bonferroni correction. The standard reference for comparing k systems on n tasks in ML.

- **Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021).** "Deep Reinforcement Learning at the Edge of the Statistical Precipice." *NeurIPS 2021*. Introduces interquartile mean (IQM) as a robust aggregate and bootstrap confidence intervals as the primary uncertainty measure. We adopt both recommendations.

- **Iman, R. L. & Davenport, J. M. (1980).** "Approximations of the critical region of the Friedman statistic." *Communications in Statistics - Theory and Methods*, 9(6), 571--595. Provides the F-distribution correction to the Friedman statistic used in our omnibus test.

### 15.2 LLM Evaluation Methodology

- **Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023).** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*. Establishes best practices for LLM-as-judge: position bias, verbosity bias, self-enhancement bias, and calibration techniques.

- **Kim, S., et al. (2024).** "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." *ICML 2024*. Demonstrates the viability of specialized judge models and criterion-level evaluation.

- **Wei, T., et al. (2024).** "Systematic Analysis of LLM Contributions to Deep Research." Surveys how LLMs serve as both research tools and evaluation instruments.

### 15.3 Deep Research Systems

- **arXiv:2508.12752** (Modular Deep Research Framework Survey). Provides the architectural taxonomy used to classify P0--P5: RAG baseline, iterative retrieval, supervisor-worker, role specialization, multi-perspective synthesis, hierarchical planning.

- **arXiv:2512.02038** (Systematic Survey of Deep Research Systems). Comprehensive survey covering GPT-Researcher, STORM, PaperQA2, AutoSurvey, and commercial systems. Identifies shared design patterns and evaluation challenges.

- **Shao, Z., et al. (2024).** "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models." *NAACL 2024 (STORM)*. The direct inspiration for P4 (Perspective STORM). Introduces perspective-driven conversations and claim triangulation.

- **Skarlinski, M., et al. (2024).** "Language agents achieve superhuman synthesis of scientific knowledge." *arXiv preprint (PaperQA2)*. Demonstrates SAFE-style citation verification and the LitQA2 benchmark. Their 85.2% precision provides the calibration target for our LitQA2 evaluation.

### 15.4 Benchmarks

- **DRACO** (Perplexity, 2025). "DRACO: A Dynamic Research Autonomous Capability and Operations Benchmark." Expert-crafted rubrics with weighted binary criteria across 10 domains. The primary external benchmark for calibrating our patterns against production systems.

- **Li, Y., et al. (2025).** "ResearchQA: A Benchmark for Real-World Research Knowledge Assessment." 21,414 queries with 160K+ rubric items authored by 31 PhD annotators across 75 academic fields.

- **DeepSearchQA** (Google, 2025). 900 prompts requiring multi-step retrieval across 17 fields with expert-validated answers.

- **LitQA2** (FutureHouse/Skarlinski et al., 2024). 199 expert-crafted MCQs testing scientific literature comprehension. The strictest factual accuracy benchmark.

### 15.5 Citation Verification

- **SAFE** (Google, 2024). "Search-Augmented Factuality Evaluator." Agentic decompose-search-verify pipeline for factual claim checking. The historical run followed their claim extraction and NLI verification approach.

- **FActScore** (Min et al., 2023). "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." Establishes the atomic claim decomposition methodology we adopt.

- **SurGE** (2024). Three-level citation accuracy model (Doc-Acc, Sec-Acc, Sent-Acc) for evaluating whether citations are thematically relevant, section-appropriate, and sentence-supporting. Implemented in the historical retrieval evaluator.

### 15.6 Effect Size and Reliability

- **Cliff, N. (1993).** "Dominance statistics: Ordinal analyses to answer ordinal questions." *Psychological Bulletin*, 114(3), 494--509. Defines Cliff's Delta, our primary non-parametric effect size measure.

- **Krippendorff, K. (2011).** "Computing Krippendorff's Alpha-Reliability." University of Pennsylvania. Defines the inter-rater reliability coefficient used for our multi-judge evaluation.

- **Cohen, J. (1960).** "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement*, 20(1), 37--46. Defines Cohen's kappa used for pairwise inter-judge agreement.

---

## Appendix A: Rubric Criteria Inventory

### A.1 General Criteria (31 total, applied to all queries)

**Factual Accuracy (8 criteria):**

1. Factual claims are accurate and consistent with current knowledge.
2. Technical terminology is used correctly and precisely.
3. Specific numbers, dates, or benchmarks cited are accurate.
4. Historical timeline and chronology of developments is correct.
5. Comparison claims (X outperforms Y, X predates Y) are supported by cited evidence.
6. No internal contradictions between different sections of the report.
7. Limitations and caveats of described methods or findings are accurately represented.
8. Current state-of-the-art is correctly identified where relevant.

**Coverage / Information Recall (5 criteria):**

9. The report covers the major aspects of the topic.
10. Both advantages and limitations of approaches are discussed.
11. Recent developments (within the last 2 years) are included.
12. Multiple perspectives or schools of thought are represented.
13. Practical implications or applications are addressed.

**Analytical Depth (4 criteria):**

14. The report synthesizes across sources rather than merely summarizing each.
15. Analysis goes beyond surface-level description to identify patterns, trade-offs, or mechanisms.
16. Connections between different aspects of the topic are drawn.
17. The report distinguishes between well-established findings and emerging or contested claims.

**Citation Quality (4 criteria):**

18. Claims are attributed to named sources with inline citations.
19. Citations are formatted consistently throughout the report.
20. The number of distinct sources cited is appropriate for the topic scope (minimum 5).
21. Different sections draw from different sources rather than relying on a single source.

**Organization (4 criteria):**

22. The report has a clear introduction that frames the topic.
23. Sections follow a logical progression.
24. The report has a conclusion that synthesizes key findings.
25. Paragraphs are focused and transitions between topics are smooth.

**Instruction Following (4 criteria):**

26. The report directly addresses the specific research question asked.
27. The scope of the report is appropriate to the query (not too narrow or too broad).
28. The report addresses all sub-questions or dimensions implied by the query.
29. The format and structure match what the query implies (e.g., comparison queries produce comparative analysis).

**Attribution Quality (2 criteria):**

30. Each major claim or finding is traceable to a named source.
31. The report clearly distinguishes between the author's analysis and source material.

### A.2 Task-Specific Criteria

In addition to the 31 general criteria, each query receives task-specific criteria:

- **Custom queries:** Expected elements from hand-crafted test cases, converted to coverage criteria.
- **DRACO queries:** Original DRACO rubric criteria with preserved weights (1--20 per criterion).
- **DeepSearchQA queries:** Answer-type-specific criteria (Set Answer requires comprehensive lists).
- **ResearchQA queries:** PhD-annotated rubric items converted to coverage criteria.
- **LitQA2 queries:** Correct answer support and distractor avoidance criteria.

Total criteria per query ranges from 31 (general only) to approximately 50 (general + task-specific).

---

## Appendix B: Error Taxonomy

### B.1 Error Categories

The historical error-analysis module classified report errors into eight categories:

| Category | Description | Typical Severity | Detection Method |
|----------|-------------|-----------------|-----------------|
| `hallucination` | Generated content with no basis in retrieved sources | Critical | Judge verdict + citation verification |
| `citation_fabrication` | Citations to non-existent or fabricated sources | Critical | Citation verification: source fetch failure + NLI contradiction |
| `topic_drift` | Report deviates from the research question | Minor--Moderate | Judge: instruction following NOT_SATISFIED |
| `factual_error` | Verifiably incorrect factual claim | Critical | Judge: factual accuracy NOT_SATISFIED |
| `missing_coverage` | Important aspect of the topic not addressed | Moderate | Judge: coverage NOT_SATISFIED |
| `synthesis_failure` | Report summarizes rather than synthesizes; or is extremely short (< 500 words) | Moderate--Critical | Judge: analytical depth NOT_SATISFIED; heuristic word count check |
| `source_quality` | Sources are low-quality, outdated, or irrelevant | Moderate | Retrieval metrics: low diversity, low academic ratio |
| `attribution_error` | Dangling citation references (e.g., `[99]` with only 5 references defined) | Minor | Heuristic: citation marker regex vs. reference section regex |

### B.2 Severity Levels

| Severity | Definition | Impact on Score |
|----------|-----------|----------------|
| Critical | Error that fundamentally undermines the report's reliability or utility | Major quality penalty. One critical error can invalidate an entire section. |
| Moderate | Error that degrades quality but does not invalidate the report | Noticeable quality impact. Multiple moderate errors compound. |
| Minor | Error that is noticeable but does not materially affect report utility | Marginal quality impact. Only concerning when systematic. |

### B.3 Error Aggregation

Error profiles are aggregated per pattern to identify dominant failure modes:

- **Category distribution:** Proportion of each error category across all reports for a pattern.
- **Severity distribution:** Proportion of critical/moderate/minor errors.
- **Most common errors:** Ranked list of (category, count) pairs.
- **Failure mode narratives:** Automatically generated descriptions for categories exceeding 25% of a pattern's total errors.

---

*This is the historical evaluation protocol. For runnable public commands, start with `README.md`, `REPRODUCIBILITY.md`, and `repro/PAPER_A_REPRO_MAP.md`.*
