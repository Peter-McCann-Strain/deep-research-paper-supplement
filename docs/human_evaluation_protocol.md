# Human Evaluation Protocol for Deep Research Pattern Comparison

## Purpose

This protocol documents how human reviewers checked a representative subset of
research reports against the LLM-as-judge scores. The point is practical: measure
where judge scores agree with expert readers, and find the dimensions where the
judge is least reliable.

The public repository includes the protocol, not raw evaluator packets, contact
records, or private label exports.

## Sample Selection

- **Target size:** 15% of all generated reports (81 reports from 540 total)
- **Stratification:** Minimum 2 reports per pattern per difficulty level
  - 6 patterns x 3 difficulty levels = 18 strata
  - ~4-5 reports per stratum
- **Randomization:** Reports selected uniformly at random within each stratum
- **Selection seed:** Fixed random seed (42) for reproducibility

## Evaluator Requirements

- PhD student or postdoc in a relevant research domain, OR
- 3+ years of professional research experience in a relevant field
- Minimum 3 evaluators per report for inter-annotator agreement
- Evaluators must complete a calibration session on 3 practice reports before starting
- Evaluators should not evaluate more than 10 reports in a single session to avoid fatigue

## Blinding

- Evaluators are blinded to which pattern generated each report
- Reports are identified only by anonymous ID (e.g., "R-0042")
- The original query text is provided for context
- Evaluators do not see other evaluators' judgments until all evaluations are complete

## Evaluation Tasks

### Task 1: Factual Accuracy Assessment (~15 min/report)

For each factual claim in the report:
1. Mark as CORRECT, INCORRECT, or UNVERIFIABLE
2. Evaluators may use internet search to verify claims
3. Note the specific error for INCORRECT claims
4. Record time spent

Scoring: proportion of CORRECT claims out of (CORRECT + INCORRECT).
UNVERIFIABLE claims are excluded from the denominator.

### Task 2: Citation Quality Assessment (~10 min/report)

For each inline citation:
1. Check if the cited source EXISTS (accessible URL, valid paper title, identifiable publication)
2. Check if the source is RELEVANT to the claim being made
3. Check if the source SUPPORTS the specific claim (not just the general topic)

Scoring:
- Existence rate: proportion of citations where source exists
- Relevance rate: proportion of existing sources that are relevant
- Support rate: proportion of relevant sources that support the claim

### Task 3: Overall Quality Rating (~5 min/report)

Rate on a 1-7 Likert scale for each dimension:

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Very Poor | Fails to address the dimension |
| 2 | Poor | Major deficiencies |
| 3 | Below Average | Notable gaps |
| 4 | Average | Adequate but unremarkable |
| 5 | Above Average | Solid with minor gaps |
| 6 | Good | Comprehensive and well-executed |
| 7 | Excellent | Exemplary quality |

Dimensions rated:
1. Factual accuracy
2. Topic coverage
3. Analytical depth
4. Citation quality
5. Organization
6. Instruction following

## Agreement Metrics

- **Fleiss' kappa** for 3+ raters on binary verdicts (SATISFIED/NOT_SATISFIED)
- **Krippendorff's alpha** for ordinal (1-7) ratings
- **Target:** kappa >= 0.60 (moderate agreement, realistic for long-form text evaluation per STORM's findings)
- If agreement falls below 0.40, conduct a reconciliation session and re-evaluate

## Judge-Human Comparison

After human evaluation is complete:
1. Convert human Likert scores to binary using threshold of 4 (>= 4 = SATISFIED)
2. Compute Cohen's kappa between majority-vote human verdicts and LLM judge verdicts
3. Compute Pearson correlation between average human Likert scores and judge dimension scores
4. Identify dimensions with lowest agreement (kappa < 0.40)
5. Report judge bias (systematic over- or under-rating)

## Data Collection Format

- Use the JSON structure below; the private historical helper module is not shipped in the public export
- Record time spent per criterion (for workload estimation)
- Record confidence per judgment (0-1 scale)
- Save as JSON with evaluator_id anonymized (e.g., "eval_A", "eval_B", "eval_C")
- One JSON file per evaluated report

### Example JSON structure

```json
{
  "report_id": "R-0042",
  "pattern": "p4_perspective_storm",
  "query_id": "q7",
  "evaluators": ["eval_A", "eval_B", "eval_C"],
  "verdicts": [
    {
      "evaluator_id": "eval_A",
      "report_id": "R-0042",
      "criterion": "Factual claims are accurate",
      "dimension": "factual_accuracy",
      "verdict": "SATISFIED",
      "confidence": 0.8,
      "comment": "Most claims verified via Google Scholar",
      "time_seconds": 120
    }
  ]
}
```

## Timeline

1. **Week 1:** Evaluator recruitment and calibration session
2. **Weeks 2-3:** Primary evaluation (81 reports x 3 evaluators = 243 evaluations)
3. **Week 4:** Reconciliation for low-agreement cases, analysis, reporting

## Ethical Considerations

- Evaluators are told the task, expected time, and use of their ratings before they start.
- Evaluators are compensated for their expertise and time.
- Public artifacts use anonymized evaluator IDs such as `eval_A`; raw contact details and scheduling records are not shipped.
- Label files and comments are treated as research-validation data, not as public benchmark prompts or training data.
- Any future public release of human labels should repeat the license, consent, and privacy review before publication.
