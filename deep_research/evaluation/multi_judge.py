"""Multi-judge ensemble evaluator with reliability measurement.

Implements:
- Multiple judge models evaluated in parallel
- Multiple passes per judge for intra-judge consistency
- Cohen's kappa for inter-judge agreement
- Krippendorff's alpha for multi-rater agreement
- SE-Jury style majority-vote aggregation
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import structlog
from openai import AsyncAzureOpenAI

from deep_research.config import (
    AZURE_OPENAI_API_VERSION,
    EVAL_PIPELINE,
    JUDGE,
    RETRY,
    TIMEOUTS,
)
from deep_research.evaluation.rubric_v2 import (
    RubricV2,
    rubric_to_judge_prompt,
    rubric_to_judge_prompt_with_mapping,
)

log = structlog.get_logger()


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class CriterionVerdict:
    """A single criterion's verdict from one judge pass."""

    criterion_text: str
    dimension: str
    satisfied: bool
    reasoning: str
    weight: float = 1.0


@dataclass
class JudgePassResult:
    """Result from a single judge's single pass on one report."""

    judge_label: str
    pass_number: int
    verdicts: list[CriterionVerdict]
    dimension_scores: dict[str, float]
    overall_score: float
    raw_response: str


@dataclass
class EnsembleResult:
    """Aggregated result from multi-judge ensemble."""

    query_id: str
    pattern_name: str
    # Individual results
    individual_passes: list[JudgePassResult]
    # Aggregated scores
    ensemble_overall: float
    ensemble_dimensions: dict[str, float]
    # Reliability metrics
    intra_judge_consistency: dict[str, float]  # judge_label -> per-criterion flip rate
    inter_judge_agreement: float  # Cohen's kappa (for 2 judges) or Fleiss' kappa
    krippendorffs_alpha: float
    per_dimension_agreement: dict[str, float]
    # Metadata
    n_judges: int = 0
    n_passes_per_judge: int = 0
    total_evaluations: int = 0


@dataclass
class JudgeConfig:
    """Configuration for a single judge."""

    label: str
    model: str
    endpoint: str
    api_key: str
    api_version: str = AZURE_OPENAI_API_VERSION
    temperature: float = JUDGE.temperature
    max_tokens: int = JUDGE.max_tokens


# ── Judge prompt constants ───────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are an expert research report evaluator using the DRACO evaluation methodology. You assess whether research reports satisfy specific evaluation criteria.

You will be given:
1. The original research query
2. A research report to evaluate
3. A list of evaluation criteria

For EACH criterion, you must provide:
- VERDICT: "SATISFIED" or "NOT_SATISFIED"
- EVIDENCE: A brief quote or reference to specific content in the report
- REASONING: One sentence explaining your judgment

Rules:
- Only mark SATISFIED if the criterion is clearly and fully met
- Partial or vague coverage counts as NOT_SATISFIED
- For citation criteria, check that actual sources/references are provided
- For factual criteria, verify claims are consistent and reasonable
- Be strict but fair -- do not penalize for minor omissions if the substance is there

## Calibration Examples

The following worked examples illustrate the expected verdict format and calibration level.

Example 1 -- SATISFIED (factual_accuracy):
```json
{
  "criterion_index": 0,
  "verdict": "SATISFIED",
  "evidence": "The report states \\"Transformer architecture was introduced by Vaswani et al. (2017)\\" which is correct and properly attributed to the original paper.",
  "reasoning": "The factual claim is accurate and properly attributed to the correct authors and year."
}
```

Example 2 -- NOT_SATISFIED (citation_quality):
```json
{
  "criterion_index": 3,
  "verdict": "NOT_SATISFIED",
  "evidence": "The report states \\"Research suggests that...\\" and \\"Studies have shown...\\" without naming any specific sources or publications.",
  "reasoning": "Vague attributions like \\"studies show\\" and \\"research suggests\\" without specific named citations do not satisfy the requirement for claims to be attributed to specific named sources."
}
```

Example 3 -- NOT_SATISFIED (coverage):
```json
{
  "criterion_index": 5,
  "verdict": "NOT_SATISFIED",
  "evidence": "The report discusses only the benefits of the approach across all sections, including the introduction, methodology overview, and applications.",
  "reasoning": "No limitations, drawbacks, or counterarguments are presented anywhere in the report, so the criterion requiring both advantages and limitations is not met."
}
```

Respond with valid JSON only."""

_MAX_RETRIES = RETRY.max_retries


# ── Agreement metrics (implemented from scratch) ────────────────────────────


def cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Cohen's kappa for binary agreement between two raters.

    kappa = (p_o - p_e) / (1 - p_e)

    where p_o is observed agreement and p_e is expected agreement by chance.

    Returns:
        kappa in [-1, 1].  1.0 = perfect agreement, 0 = chance, <0 = worse
        than chance.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater lists must be the same length")
    n = len(rater_a)
    if n == 0:
        return 0.0

    # Contingency table counts
    a1b1 = sum(1 for a, b in zip(rater_a, rater_b) if a and b)
    a0b0 = sum(1 for a, b in zip(rater_a, rater_b) if not a and not b)
    a1b0 = sum(1 for a, b in zip(rater_a, rater_b) if a and not b)
    a0b1 = sum(1 for a, b in zip(rater_a, rater_b) if not a and b)

    p_o = (a1b1 + a0b0) / n

    # Marginal probabilities
    p_a1 = (a1b1 + a1b0) / n
    p_b1 = (a1b1 + a0b1) / n
    p_a0 = 1.0 - p_a1
    p_b0 = 1.0 - p_b1

    p_e = p_a1 * p_b1 + p_a0 * p_b0

    if abs(1.0 - p_e) < 1e-12:
        # Both raters always agree (degenerate case) -> perfect agreement
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Fleiss' kappa for multiple raters with binary categories.

    Args:
        ratings_matrix: shape (n_items, n_categories). Each row sums to
            n_raters.  For binary data, n_categories = 2.

    Returns:
        kappa in [-1, 1].
    """
    ratings_matrix = np.asarray(ratings_matrix, dtype=float)
    if ratings_matrix.ndim != 2:
        raise ValueError("ratings_matrix must be 2-dimensional")

    n_items, n_categories = ratings_matrix.shape
    if n_items == 0 or n_categories == 0:
        return 0.0

    n_raters = ratings_matrix[0].sum()
    if n_raters < 2:
        return 0.0

    # Proportion of assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # Per-item agreement: P_i = (1 / (n*(n-1))) * (sum(n_ij^2) - n)
    P_i = (
        (ratings_matrix ** 2).sum(axis=1) - n_raters
    ) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()

    P_e_bar = (p_j ** 2).sum()

    if abs(1.0 - P_e_bar) < 1e-12:
        return 1.0 if abs(P_bar - 1.0) < 1e-12 else 0.0

    return (P_bar - P_e_bar) / (1.0 - P_e_bar)


def krippendorffs_alpha_binary(data: np.ndarray) -> float:
    """Krippendorff's alpha for binary nominal data.

    Args:
        data: shape (n_raters, n_items). Values are 0 or 1. NaN for missing.

    Returns:
        alpha in [-1, 1].  1.0 = perfect agreement, 0 = chance, <0 = worse.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-dimensional (n_raters, n_items)")

    n_raters, n_items = data.shape
    if n_items == 0:
        return 0.0

    # Observed disagreement: D_o
    # For each item, compute the proportion of rater-pair disagreements
    # weighted by the number of valid pairs.
    D_o = 0.0
    total_pairs = 0.0

    for u in range(n_items):
        col = data[:, u]
        valid = col[~np.isnan(col)]
        m_u = len(valid)
        if m_u < 2:
            continue
        # Number of 1s and 0s among valid raters for this item
        n1 = valid.sum()
        n0 = m_u - n1
        # Number of disagreeing pairs = n0 * n1 (each 0 paired with each 1)
        item_disagree = n0 * n1
        # Weight by 1/(m_u - 1) as per Krippendorff
        D_o += item_disagree / (m_u - 1)
        total_pairs += m_u

    if total_pairs < 2:
        return 0.0

    D_o /= total_pairs

    # Expected disagreement: D_e
    # Based on marginal frequencies across all valid assignments
    all_valid = data[~np.isnan(data)]
    n_total = len(all_valid)
    if n_total < 2:
        return 0.0

    n1_total = all_valid.sum()
    n0_total = n_total - n1_total

    # For nominal data with 2 categories:
    # D_e = (n0_total * n1_total) / (n_total * (n_total - 1))
    D_e = (n0_total * n1_total) / (n_total * (n_total - 1))

    if abs(D_e) < 1e-12:
        # No expected disagreement -> perfect agreement by definition
        return 1.0

    return 1.0 - (D_o / D_e)


# ── Multi-Judge class ────────────────────────────────────────────────────────


class MultiJudge:
    """Multi-judge ensemble evaluator."""

    def __init__(
        self,
        judges: list[JudgeConfig],
        passes_per_judge: int = 3,
        max_concurrent: int = 3,
    ):
        self.judges = judges
        self.passes_per_judge = passes_per_judge
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate(
        self,
        query_id: str,
        query_text: str,
        pattern_name: str,
        report_text: str,
        rubric: RubricV2,
    ) -> EnsembleResult:
        """Run all judges with multiple passes and aggregate.

        Total evaluations = len(judges) * passes_per_judge
        """
        start = time.time()
        tasks = []
        for judge in self.judges:
            for pass_num in range(self.passes_per_judge):
                tasks.append(
                    self._run_single_pass(
                        judge, pass_num, query_id, query_text, report_text, rubric
                    )
                )

        all_passes: list[JudgePassResult] = await asyncio.gather(*tasks)

        # Aggregate
        ensemble_overall, ensemble_dimensions = self._aggregate_ensemble(
            all_passes, rubric
        )

        # Reliability metrics
        intra = self._compute_intra_judge_consistency(all_passes)
        inter = self._compute_inter_judge_agreement(all_passes)
        alpha = self._compute_krippendorffs_alpha(all_passes)
        per_dim = self._compute_per_dimension_agreement(all_passes, rubric)

        elapsed = time.time() - start
        log.info(
            "ensemble_complete",
            query_id=query_id,
            pattern=pattern_name,
            overall=f"{ensemble_overall:.3f}",
            kappa=f"{inter:.3f}",
            alpha=f"{alpha:.3f}",
            time=f"{elapsed:.1f}s",
        )

        return EnsembleResult(
            query_id=query_id,
            pattern_name=pattern_name,
            individual_passes=all_passes,
            ensemble_overall=ensemble_overall,
            ensemble_dimensions=ensemble_dimensions,
            intra_judge_consistency=intra,
            inter_judge_agreement=inter,
            krippendorffs_alpha=alpha,
            per_dimension_agreement=per_dim,
            n_judges=len(self.judges),
            n_passes_per_judge=self.passes_per_judge,
            total_evaluations=len(all_passes),
        )

    async def _run_single_pass(
        self,
        judge: JudgeConfig,
        pass_number: int,
        query_id: str,
        query_text: str,
        report_text: str,
        rubric: RubricV2,
    ) -> JudgePassResult:
        """Execute a single judge pass.

        Creates an AsyncAzureOpenAI client per judge config, builds the prompt
        from the rubric, calls the LLM with JSON mode, and parses the response
        into CriterionVerdict objects.

        Criteria are shuffled per pass using a deterministic seed derived from
        the judge label, pass number, and query ID to mitigate position bias.
        """
        client = AsyncAzureOpenAI(
            api_key=judge.api_key,
            azure_endpoint=judge.endpoint,
            api_version=judge.api_version,
            max_retries=JUDGE.sdk_max_retries,
            timeout=httpx.Timeout(
                connect=TIMEOUTS.connect,
                read=JUDGE.read_timeout,
                write=TIMEOUTS.write,
                pool=TIMEOUTS.pool,
            ),
        )

        shuffle_seed = hash((judge.label, pass_number, query_id)) & 0x7FFFFFFF
        criteria_prompt, criterion_mapping = rubric_to_judge_prompt_with_mapping(
            rubric, seed=shuffle_seed
        )

        # Truncate report if very long
        words = report_text.split()
        if len(words) > EVAL_PIPELINE.report_truncation_words:
            log.warning(
                "report_truncated",
                query_id=query_id,
                judge=judge.label,
                original_words=len(words),
                truncated_to=EVAL_PIPELINE.report_truncation_words,
                words_lost=len(words) - EVAL_PIPELINE.report_truncation_words,
            )
            report_text = (
                " ".join(words[:EVAL_PIPELINE.report_truncation_words])
                + "\n\n[... report truncated for evaluation ...]"
            )

        user_msg = (
            f"## Research Query\n{query_text}\n\n"
            f"## Report to Evaluate\n{report_text}\n\n"
            f"## Criteria to Evaluate\n{criteria_prompt}"
        )

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw_content = await self._call_with_retry(
            client, judge, messages
        )

        # Parse the response
        verdicts = self._parse_verdicts(raw_content, rubric, criterion_mapping)

        # Compute dimension scores
        dimension_scores = self._compute_dimension_scores(verdicts, rubric)

        # Overall weighted score
        overall = sum(
            dimension_scores.get(dim, 0.0) * w
            for dim, w in rubric.dimension_weights.items()
        )

        return JudgePassResult(
            judge_label=judge.label,
            pass_number=pass_number,
            verdicts=verdicts,
            dimension_scores=dimension_scores,
            overall_score=overall,
            raw_response=raw_content,
        )

    async def _call_with_retry(
        self,
        client: AsyncAzureOpenAI,
        judge: JudgeConfig,
        messages: list[dict],
    ) -> str:
        """Make a judge API call with rate limiting and retry."""
        from openai import (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        )

        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            async with self._semaphore:
                try:
                    resp = await client.chat.completions.create(
                        model=judge.model,
                        messages=messages,
                        temperature=judge.temperature,
                        response_format={"type": "json_object"},
                        max_completion_tokens=judge.max_tokens,
                    )
                    return resp.choices[0].message.content or "{}"

                except RateLimitError as e:
                    last_exc = e
                    retry_after = 0.0
                    response = getattr(e, "response", None)
                    if response:
                        headers = getattr(response, "headers", {})
                        retry_ms = headers.get("retry-after-ms")
                        if retry_ms:
                            retry_after = int(retry_ms) / 1000.0
                        else:
                            retry_s = headers.get("retry-after")
                            if retry_s:
                                retry_after = float(retry_s)
                    wait = max(retry_after, 2.0 * (2 ** min(attempt, 4))) + random.uniform(0, 2)
                    log.warning(
                        "judge_rate_limited",
                        judge=judge.label,
                        attempt=attempt + 1,
                        wait=f"{wait:.1f}s",
                    )
                    await asyncio.sleep(wait)

                except (
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    ConnectionError,
                    httpx.ReadError,
                    httpx.WriteError,
                ) as e:
                    last_exc = e
                    wait = min(
                        1.0 * (2 ** min(attempt, 4)) + random.uniform(0, 2), 30
                    )
                    log.warning(
                        "judge_connection_error",
                        judge=judge.label,
                        error=type(e).__name__,
                        attempt=attempt + 1,
                        wait=f"{wait:.1f}s",
                    )
                    await asyncio.sleep(wait)

                except Exception as e:
                    log.error(
                        "judge_unexpected_error",
                        judge=judge.label,
                        error=str(e)[:200],
                    )
                    raise

        log.error("judge_retries_exhausted", judge=judge.label, attempts=_MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    def _parse_verdicts(
        self,
        raw_response: str,
        rubric: RubricV2,
        criterion_mapping: list[int] | None = None,
    ) -> list[CriterionVerdict]:
        """Parse the JSON response into CriterionVerdict objects.

        Args:
            raw_response: Raw JSON string from the judge LLM.
            rubric: The rubric used for evaluation.
            criterion_mapping: Optional mapping from shuffled index to original
                ``rubric.criteria`` index.  When provided, the
                ``criterion_index`` returned by the judge is first translated
                through this mapping before looking up the criterion.
        """
        verdicts: list[CriterionVerdict] = []
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            # Return all NOT_SATISFIED on parse failure
            for crit in rubric.criteria:
                verdicts.append(
                    CriterionVerdict(
                        criterion_text=crit.text,
                        dimension=crit.dimension,
                        satisfied=False,
                        reasoning="JSON parse error in judge response",
                        weight=crit.weight,
                    )
                )
            return verdicts

        evals = result.get("evaluations", [])
        for ev in evals:
            idx = ev.get("criterion_index", 0)
            # If a mapping is provided, translate shuffled index -> original index
            if criterion_mapping is not None:
                if 0 <= idx < len(criterion_mapping):
                    orig_idx = criterion_mapping[idx]
                else:
                    orig_idx = 0
            else:
                orig_idx = idx
            if 0 <= orig_idx < len(rubric.criteria):
                crit = rubric.criteria[orig_idx]
            elif rubric.criteria:
                crit = rubric.criteria[0]
            else:
                continue

            verdicts.append(
                CriterionVerdict(
                    criterion_text=crit.text,
                    dimension=crit.dimension,
                    satisfied=ev.get("verdict", "").upper() == "SATISFIED",
                    reasoning=ev.get("reasoning", ev.get("evidence", "")),
                    weight=crit.weight,
                )
            )

        # If we got fewer verdicts than criteria, fill remaining as NOT_SATISFIED
        seen_texts = {v.criterion_text for v in verdicts}
        for crit in rubric.criteria:
            if crit.text not in seen_texts:
                verdicts.append(
                    CriterionVerdict(
                        criterion_text=crit.text,
                        dimension=crit.dimension,
                        satisfied=False,
                        reasoning="No verdict returned by judge",
                        weight=crit.weight,
                    )
                )

        return verdicts

    def _compute_dimension_scores(
        self, verdicts: list[CriterionVerdict], rubric: RubricV2
    ) -> dict[str, float]:
        """Compute per-dimension scores from verdicts using criterion weights.

        Dimension score = sum(weight_i * met_i) / sum(weight_i)
        where met_i is 1 if satisfied, 0 otherwise.  This correctly handles
        DRACO criteria whose weights range from 1 to 20 (or negative for
        penalties).
        """
        scores: dict[str, float] = {}
        for dim in rubric.get_dimensions():
            dim_verdicts = [v for v in verdicts if v.dimension == dim]
            if not dim_verdicts:
                scores[dim] = 0.0
                continue
            weighted_met = 0.0
            for v in dim_verdicts:
                aw = abs(v.weight)
                if v.weight >= 0:
                    # Positive criterion: SATISFIED = good -> add weight
                    weighted_met += aw * (1.0 if v.satisfied else 0.0)
                else:
                    # Negative (penalty) criterion: SATISFIED = bad behavior
                    # present -> penalise; NOT_SATISFIED = bad behavior absent
                    # -> reward with full weight
                    weighted_met += aw * (0.0 if v.satisfied else 1.0)
            total_weight = sum(abs(v.weight) for v in dim_verdicts)
            scores[dim] = weighted_met / total_weight if total_weight > 0 else 0.0
        # Also include dimensions from weights that may not have criteria
        for dim in rubric.dimension_weights:
            if dim not in scores:
                scores[dim] = 0.0
        return scores

    def _aggregate_ensemble(
        self,
        all_passes: list[JudgePassResult],
        rubric: RubricV2,
    ) -> tuple[float, dict[str, float]]:
        """Majority-vote aggregation across all passes.

        For each criterion:
        - Count SATISFIED votes across all passes
        - If > 50% SATISFIED, criterion is MET
        - Dimension score = met_criteria / total_criteria for that dimension
        - Overall = weighted sum of dimension scores
        """
        if not all_passes:
            return 0.0, {}

        n_passes = len(all_passes)

        # Map criterion_text -> list of satisfied bools across all passes
        criterion_votes: dict[str, list[bool]] = {}
        for pass_result in all_passes:
            for v in pass_result.verdicts:
                criterion_votes.setdefault(v.criterion_text, []).append(v.satisfied)

        # Majority vote per criterion
        criterion_met: dict[str, bool] = {}
        for crit_text, votes in criterion_votes.items():
            satisfied_count = sum(votes)
            criterion_met[crit_text] = satisfied_count > len(votes) / 2

        # Compute dimension scores from majority-voted criteria using weights.
        # Dimension score = sum(|weight_i| * met_i) / sum(|weight_i|)
        dimension_scores: dict[str, float] = {}
        for dim in rubric.get_dimensions():
            dim_criteria = rubric.get_criteria_by_dimension(dim)
            if not dim_criteria:
                dimension_scores[dim] = 0.0
                continue
            weighted_met = 0.0
            for c in dim_criteria:
                aw = abs(c.weight)
                met = criterion_met.get(c.text, False)
                if c.weight >= 0:
                    # Positive criterion: SATISFIED = good
                    weighted_met += aw * (1.0 if met else 0.0)
                else:
                    # Negative (penalty) criterion: SATISFIED = bad behavior
                    # present -> penalise; NOT_SATISFIED = good
                    weighted_met += aw * (0.0 if met else 1.0)
            total_weight = sum(abs(c.weight) for c in dim_criteria)
            dimension_scores[dim] = weighted_met / total_weight if total_weight > 0 else 0.0

        # Also cover dimensions from weights
        for dim in rubric.dimension_weights:
            if dim not in dimension_scores:
                dimension_scores[dim] = 0.0

        # Overall weighted score
        overall = sum(
            dimension_scores.get(dim, 0.0) * w
            for dim, w in rubric.dimension_weights.items()
        )

        return overall, dimension_scores

    def _compute_intra_judge_consistency(
        self,
        all_passes: list[JudgePassResult],
    ) -> dict[str, float]:
        """Compute per-criterion flip rate within each judge's passes.

        Flip rate = fraction of criteria where the judge gave different verdicts
        across passes.  Low flip rate = high consistency.

        Returns:
            dict mapping judge_label to flip rate (0.0 = perfectly consistent).
        """
        # Group passes by judge
        judge_passes: dict[str, list[JudgePassResult]] = {}
        for p in all_passes:
            judge_passes.setdefault(p.judge_label, []).append(p)

        result: dict[str, float] = {}
        for label, passes in judge_passes.items():
            if len(passes) < 2:
                result[label] = 0.0
                continue

            # Build criterion -> list of verdicts across passes
            crit_verdicts: dict[str, list[bool]] = {}
            for p in passes:
                for v in p.verdicts:
                    crit_verdicts.setdefault(v.criterion_text, []).append(v.satisfied)

            if not crit_verdicts:
                result[label] = 0.0
                continue

            # A criterion "flipped" if not all verdicts agree
            flipped = sum(
                1
                for votes in crit_verdicts.values()
                if len(set(votes)) > 1
            )
            result[label] = flipped / len(crit_verdicts)

        return result

    def _compute_inter_judge_agreement(
        self,
        all_passes: list[JudgePassResult],
    ) -> float:
        """Cohen's kappa (2 judges) or Fleiss' kappa (3+ judges).

        Computed on the majority verdict from each judge's passes.
        For each judge, we first take the majority vote across their passes
        for each criterion, then compute agreement across judges.
        """
        # Group passes by judge
        judge_passes: dict[str, list[JudgePassResult]] = {}
        for p in all_passes:
            judge_passes.setdefault(p.judge_label, []).append(p)

        judge_labels = list(judge_passes.keys())
        if len(judge_labels) < 2:
            return 1.0  # Single judge is trivially in agreement with itself

        # For each judge, compute majority verdict per criterion
        judge_majorities: dict[str, dict[str, bool]] = {}
        all_criteria_texts: set[str] = set()
        for label, passes in judge_passes.items():
            crit_verdicts: dict[str, list[bool]] = {}
            for p in passes:
                for v in p.verdicts:
                    crit_verdicts.setdefault(v.criterion_text, []).append(v.satisfied)
                    all_criteria_texts.add(v.criterion_text)
            majority: dict[str, bool] = {}
            for crit_text, votes in crit_verdicts.items():
                majority[crit_text] = sum(votes) > len(votes) / 2
            judge_majorities[label] = majority

        # Ensure consistent ordering of criteria
        criteria_list = sorted(all_criteria_texts)
        if not criteria_list:
            return 1.0

        if len(judge_labels) == 2:
            # Cohen's kappa for 2 judges
            rater_a = [
                judge_majorities[judge_labels[0]].get(c, False) for c in criteria_list
            ]
            rater_b = [
                judge_majorities[judge_labels[1]].get(c, False) for c in criteria_list
            ]
            return cohens_kappa(rater_a, rater_b)
        else:
            # Fleiss' kappa for 3+ judges
            n_raters = len(judge_labels)
            # Build ratings_matrix: (n_items, 2) where columns are [NOT_SATISFIED, SATISFIED]
            ratings = np.zeros((len(criteria_list), 2))
            for i, crit in enumerate(criteria_list):
                for label in judge_labels:
                    if judge_majorities.get(label, {}).get(crit, False):
                        ratings[i, 1] += 1
                    else:
                        ratings[i, 0] += 1
            return fleiss_kappa(ratings)

    def _compute_krippendorffs_alpha(
        self,
        all_passes: list[JudgePassResult],
    ) -> float:
        """Krippendorff's alpha for binary verdicts across all passes.

        Each pass is treated as a separate rater. This captures both
        intra-judge and inter-judge variance.
        """
        if not all_passes:
            return 0.0

        # Collect all unique criteria
        all_criteria: set[str] = set()
        for p in all_passes:
            for v in p.verdicts:
                all_criteria.add(v.criterion_text)
        criteria_list = sorted(all_criteria)

        if not criteria_list:
            return 0.0

        n_raters = len(all_passes)
        n_items = len(criteria_list)
        crit_idx = {c: i for i, c in enumerate(criteria_list)}

        # Build data matrix: (n_raters, n_items)
        data = np.full((n_raters, n_items), np.nan)
        for r, p in enumerate(all_passes):
            for v in p.verdicts:
                idx = crit_idx.get(v.criterion_text)
                if idx is not None:
                    data[r, idx] = 1.0 if v.satisfied else 0.0

        return krippendorffs_alpha_binary(data)

    def _compute_per_dimension_agreement(
        self,
        all_passes: list[JudgePassResult],
        rubric: RubricV2,
    ) -> dict[str, float]:
        """Agreement (kappa) computed separately per dimension.

        Uses Cohen's kappa for 2 judges, Fleiss' kappa for 3+, based on
        majority verdicts per judge.
        """
        # Group passes by judge and get majority verdicts
        judge_passes: dict[str, list[JudgePassResult]] = {}
        for p in all_passes:
            judge_passes.setdefault(p.judge_label, []).append(p)

        judge_labels = list(judge_passes.keys())
        if len(judge_labels) < 2:
            return {dim: 1.0 for dim in rubric.get_dimensions()}

        # Compute per-judge majority verdicts
        judge_majorities: dict[str, dict[str, bool]] = {}
        for label, passes in judge_passes.items():
            crit_verdicts: dict[str, list[bool]] = {}
            for p in passes:
                for v in p.verdicts:
                    crit_verdicts.setdefault(v.criterion_text, []).append(v.satisfied)
            majority: dict[str, bool] = {}
            for crit_text, votes in crit_verdicts.items():
                majority[crit_text] = sum(votes) > len(votes) / 2
            judge_majorities[label] = majority

        result: dict[str, float] = {}
        for dim in rubric.get_dimensions():
            dim_criteria = rubric.get_criteria_by_dimension(dim)
            dim_crit_texts = [c.text for c in dim_criteria]
            if not dim_crit_texts:
                result[dim] = 1.0
                continue

            if len(judge_labels) == 2:
                rater_a = [
                    judge_majorities[judge_labels[0]].get(c, False)
                    for c in dim_crit_texts
                ]
                rater_b = [
                    judge_majorities[judge_labels[1]].get(c, False)
                    for c in dim_crit_texts
                ]
                result[dim] = cohens_kappa(rater_a, rater_b)
            else:
                ratings = np.zeros((len(dim_crit_texts), 2))
                for i, crit_text in enumerate(dim_crit_texts):
                    for label in judge_labels:
                        if judge_majorities.get(label, {}).get(crit_text, False):
                            ratings[i, 1] += 1
                        else:
                            ratings[i, 0] += 1
                result[dim] = fleiss_kappa(ratings)

        return result
