"""LLM-as-judge evaluation using GPT-5.2 (V1 single-pass judge).

.. deprecated::
    This module is the V1 single-pass judge, retained for backward
    compatibility with existing verdict files.  New evaluations should use
    :mod:`deep_research.evaluation.multi_judge` which supports multi-judge
    ensemble evaluation, position-bias mitigation via criterion shuffling,
    and the full 9-dimension RubricV2 scoring.

Implements DRACO/ResearchRubrics methodology:
- Binary verdicts (SATISFIED/NOT_SATISFIED) per criterion
- Chain-of-thought justification for each verdict
- Multi-dimensional scoring aligned with RubricV2 (9 dimensions)
- Separate judge endpoint (GPT-5.2) from the pipeline models (gpt-4o PTU)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import structlog
from openai import AsyncAzureOpenAI

from deep_research.config import (
    AZURE_OPENAI_API_VERSION,
    JUDGE,
    JUDGE_MODEL,
    JUDGE_OPENAI_API_KEY,
    JUDGE_OPENAI_ENDPOINT,
    POOL,
    RETRY,
    TIMEOUTS,
)
from deep_research.evaluation.rubric_v2 import DIMENSION_WEIGHTS_V2

log = structlog.get_logger()

# ── Rate limiting for judge endpoint ─────────────────────────────────────────
_JUDGE_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_judge_semaphore() -> asyncio.Semaphore:
    """Limit concurrent judge calls (standard deployment, not PTU)."""
    global _JUDGE_SEMAPHORE
    if _JUDGE_SEMAPHORE is None:
        _JUDGE_SEMAPHORE = asyncio.Semaphore(JUDGE.max_concurrent)
    return _JUDGE_SEMAPHORE


# ── Judge client singleton ───────────────────────────────────────────────────

_judge_client: Optional[AsyncAzureOpenAI] = None


def _get_judge_client() -> AsyncAzureOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncAzureOpenAI(
            api_key=JUDGE_OPENAI_API_KEY,
            azure_endpoint=JUDGE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            max_retries=JUDGE.sdk_max_retries,
            timeout=httpx.Timeout(
                connect=TIMEOUTS.connect,
                read=JUDGE.read_timeout,
                write=TIMEOUTS.write,
                pool=TIMEOUTS.pool,
            ),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=POOL.max_connections,
                    max_keepalive_connections=POOL.max_keepalive_connections,
                    keepalive_expiry=POOL.keepalive_expiry,
                ),
            ),
        )
    return _judge_client


async def _judge_call_with_retry(messages: list, max_tokens: int = JUDGE.max_tokens) -> tuple:
    """Make a judge API call with rate limiting and retry.

    Returns (content_str, total_tokens).
    """
    import random
    from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

    client = _get_judge_client()
    sem = _get_judge_semaphore()
    last_exc = None

    for attempt in range(RETRY.max_retries):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                    temperature=JUDGE.temperature,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                    seed=JUDGE.seed,
                )
                content = resp.choices[0].message.content or "{}"
                tokens = resp.usage.total_tokens if resp.usage else 0
                return content, tokens

            except RateLimitError as e:
                last_exc = e
                # Extract retry-after from headers
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
                log.warning("judge_rate_limited", attempt=attempt + 1,
                            retry_after=f"{retry_after:.1f}s", wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)

            except (APIConnectionError, APITimeoutError, InternalServerError,
                    ConnectionError, httpx.ReadError, httpx.WriteError) as e:
                last_exc = e
                wait = min(1.0 * (2 ** min(attempt, 4)) + random.uniform(0, 2), 30)
                log.warning("judge_connection_error", error=type(e).__name__,
                            attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)

            except Exception as e:
                log.error("judge_unexpected_error", error=str(e)[:200],
                          error_type=type(e).__name__)
                raise

    log.error("judge_retries_exhausted", attempts=RETRY.max_retries)
    raise last_exc


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class CriterionVerdict:
    """Result of judging a single rubric criterion."""
    criterion: str
    dimension: str
    satisfied: bool
    evidence: str = ""
    reasoning: str = ""


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    name: str
    score: float           # 0.0 - 1.0
    criteria_met: int
    criteria_total: int
    details: List[CriterionVerdict] = field(default_factory=list)


@dataclass
class JudgeResult:
    """Complete LLM-as-judge evaluation result."""
    query_id: str
    pattern_name: str
    overall_score: float = 0.0
    dimensions: Dict[str, DimensionScore] = field(default_factory=dict)
    verdicts: List[CriterionVerdict] = field(default_factory=list)
    judge_model: str = ""
    latency_seconds: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "pattern": self.pattern_name,
            "overall_score": round(self.overall_score, 4),
            "judge_model": self.judge_model,
            "latency_s": round(self.latency_seconds, 1),
            "tokens": self.total_tokens,
            "dimensions": {
                name: {
                    "score": round(d.score, 4),
                    "met": d.criteria_met,
                    "total": d.criteria_total,
                }
                for name, d in self.dimensions.items()
            },
            "verdicts": [
                {
                    "criterion": v.criterion,
                    "dimension": v.dimension,
                    "satisfied": v.satisfied,
                    "evidence": v.evidence,
                    "reasoning": v.reasoning,
                }
                for v in self.verdicts
            ],
        }


# ── Rubric definitions ───────────────────────────────────────────────────────

# V1 dimension weights — DEPRECATED.
# Kept for backward compatibility with existing verdict files scored before V2.
# New evaluations should use multi_judge.py with RubricV2 dimension weights.
# Import canonical V2 weights for use in new judge calls.
DIMENSION_WEIGHTS = DIMENSION_WEIGHTS_V2

# General report quality criteria (applied to all queries)
GENERAL_CRITERIA = [
    ("The report directly addresses the research query without going off-topic", "instruction_following"),
    ("The report is well-organized with clear sections, headings, and logical flow", "organization"),
    ("The report has an introduction that frames the topic and a conclusion that synthesizes findings", "organization"),
    ("Claims are supported by citations to specific sources", "citation_quality"),
    ("Citations reference real, verifiable sources (not fabricated)", "citation_quality"),
    ("Sources are from authoritative venues (academic papers, official docs, reputable publications)", "citation_quality"),
    ("The report synthesizes information across multiple sources rather than summarizing one at a time", "analytical_depth"),
    ("The report provides analysis and comparison, not just description", "analytical_depth"),
    ("The report acknowledges limitations, gaps, or areas of disagreement", "analytical_depth"),
    ("The writing is clear, professional, and free of contradictions", "organization"),
    ("The report retrieves and includes the key facts needed to answer the research question", "information_recall"),
    ("Important quantitative data relevant to the query are present in the report", "information_recall"),
    ("The report does not contain internal contradictions between sections or claims", "logical_coherence"),
    ("Conclusions follow logically from the evidence and arguments presented", "logical_coherence"),
    ("Claims are attributed to specific named sources rather than vague references", "attribution_quality"),
    ("Factual claims in the report are accurate and consistent with current knowledge", "factual_accuracy"),
    ("Technical terminology is used correctly throughout the report", "factual_accuracy"),
    ("Specific numbers, dates, or benchmarks cited in the report are accurate", "factual_accuracy"),
]


def build_rubric(query: str, expected_elements: List[str]) -> List[tuple[str, str]]:
    """Build a full rubric combining query-specific and general criteria.

    Returns list of (criterion_text, dimension_name) tuples.
    """
    criteria = []

    # Query-specific coverage criteria
    for element in expected_elements:
        criteria.append((
            f"The report covers: {element}",
            "coverage",
        ))

    # General criteria (includes factual_accuracy, information_recall,
    # logical_coherence, attribution_quality, and all other dimensions)
    criteria.extend(GENERAL_CRITERIA)

    return criteria


# ── Judge prompts ────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert research report evaluator using the DRACO evaluation methodology. You assess whether research reports satisfy specific evaluation criteria.

You will be given:
1. The original research query
2. A research report to evaluate
3. A batch of evaluation criteria

For EACH criterion, you must provide:
- VERDICT: "SATISFIED" or "NOT_SATISFIED"
- EVIDENCE: A brief quote or reference to specific content in the report
- REASONING: One sentence explaining your judgment

Rules:
- Only mark SATISFIED if the criterion is clearly and fully met
- Partial or vague coverage counts as NOT_SATISFIED
- For citation criteria, check that actual sources/references are provided
- For factual criteria, verify claims are consistent and reasonable
- Be strict but fair — do not penalize for minor omissions if the substance is there

Respond with valid JSON only."""

JUDGE_USER_TEMPLATE = """## Research Query
{query}

## Report to Evaluate
{report_text}

## Criteria to Evaluate
Evaluate each of the following criteria. Return a JSON array with one object per criterion.

{criteria_json}

Return JSON in this exact format:
{{
  "evaluations": [
    {{
      "criterion_index": 0,
      "verdict": "SATISFIED" or "NOT_SATISFIED",
      "evidence": "brief quote or reference from the report",
      "reasoning": "one sentence explanation"
    }},
    ...
  ]
}}"""


# ── Core judge logic ─────────────────────────────────────────────────────────

async def _judge_batch(
    query: str,
    report_text: str,
    criteria: List[tuple[str, str]],
    batch_size: int = 8,
) -> List[CriterionVerdict]:
    """Judge criteria in batches to stay within context limits."""
    all_verdicts: List[CriterionVerdict] = []
    total_tokens = 0

    # Truncate report if very long (keep first ~12K words for judge context)
    words = report_text.split()
    if len(words) > 12000:
        report_text = " ".join(words[:12000]) + "\n\n[... report truncated for evaluation ...]"

    for i in range(0, len(criteria), batch_size):
        batch = criteria[i:i + batch_size]
        criteria_json = json.dumps([
            {"index": j, "criterion": c[0], "dimension": c[1]}
            for j, c in enumerate(batch)
        ], indent=2)

        user_msg = JUDGE_USER_TEMPLATE.format(
            query=query,
            report_text=report_text,
            criteria_json=criteria_json,
        )

        try:
            content, batch_tokens = await _judge_call_with_retry(
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=JUDGE.max_tokens,
            )
            total_tokens += batch_tokens

            result = json.loads(content)
            evals = result.get("evaluations", [])

            for ev in evals:
                idx = ev.get("criterion_index", 0)
                if 0 <= idx < len(batch):
                    crit_text, dim = batch[idx]
                else:
                    crit_text, dim = batch[0]

                all_verdicts.append(CriterionVerdict(
                    criterion=crit_text,
                    dimension=dim,
                    satisfied=ev.get("verdict", "").upper() == "SATISFIED",
                    evidence=ev.get("evidence", ""),
                    reasoning=ev.get("reasoning", ""),
                ))

        except Exception as e:
            log.error("judge_batch_failed", error=str(e)[:200],
                      batch_start=i, batch_size=len(batch))
            # Mark all criteria in batch as not evaluated
            for crit_text, dim in batch:
                all_verdicts.append(CriterionVerdict(
                    criterion=crit_text,
                    dimension=dim,
                    satisfied=False,
                    reasoning=f"Judge error: {str(e)[:80]}",
                ))

    return all_verdicts, total_tokens


async def judge_report(
    query: str,
    query_id: str,
    pattern_name: str,
    report_text: str,
    expected_elements: List[str],
) -> JudgeResult:
    """Run full LLM-as-judge evaluation on a research report.

    Uses GPT-5.2 to evaluate each rubric criterion with binary verdicts.
    Returns multi-dimensional scores following DRACO methodology.
    """
    start = time.time()

    # Build rubric
    criteria = build_rubric(query, expected_elements)
    log.info("judge_start", query_id=query_id, pattern=pattern_name,
             criteria_count=len(criteria))

    # Run judge
    verdicts, total_tokens = await _judge_batch(query, report_text, criteria)

    # Compute dimension scores
    dimensions: Dict[str, DimensionScore] = {}
    for dim_name in DIMENSION_WEIGHTS:
        dim_verdicts = [v for v in verdicts if v.dimension == dim_name]
        met = sum(1 for v in dim_verdicts if v.satisfied)
        total = len(dim_verdicts)
        score = met / total if total > 0 else 0.0
        dimensions[dim_name] = DimensionScore(
            name=dim_name,
            score=score,
            criteria_met=met,
            criteria_total=total,
            details=dim_verdicts,
        )

    # Compute weighted overall score
    overall = sum(
        dimensions[dim].score * weight
        for dim, weight in DIMENSION_WEIGHTS.items()
        if dim in dimensions
    )

    elapsed = time.time() - start
    log.info("judge_complete", query_id=query_id, pattern=pattern_name,
             overall=f"{overall:.3f}", tokens=total_tokens,
             time=f"{elapsed:.1f}s")

    return JudgeResult(
        query_id=query_id,
        pattern_name=pattern_name,
        overall_score=overall,
        dimensions=dimensions,
        verdicts=verdicts,
        judge_model=JUDGE_MODEL,
        latency_seconds=elapsed,
        total_tokens=total_tokens,
    )


# ── Benchmark criterion adaptation ──────────────────────────────────────────

async def judge_benchmark_report(
    query: str,
    query_id: str,
    pattern_name: str,
    report_text: str,
    rubric_criteria: List[tuple[str, str]],
) -> JudgeResult:
    """Judge a report against benchmark-provided criteria (DRACO, ResearchQA, etc).

    rubric_criteria: list of (criterion_text, dimension) tuples from the benchmark.
    """
    start = time.time()

    # Add general quality criteria
    all_criteria = list(rubric_criteria) + GENERAL_CRITERIA
    log.info("judge_benchmark_start", query_id=query_id, pattern=pattern_name,
             criteria_count=len(all_criteria))

    verdicts, total_tokens = await _judge_batch(query, report_text, all_criteria)

    # Compute dimension scores
    dimensions: Dict[str, DimensionScore] = {}
    for dim_name in DIMENSION_WEIGHTS:
        dim_verdicts = [v for v in verdicts if v.dimension == dim_name]
        met = sum(1 for v in dim_verdicts if v.satisfied)
        total = len(dim_verdicts)
        score = met / total if total > 0 else 0.0
        dimensions[dim_name] = DimensionScore(
            name=dim_name,
            score=score,
            criteria_met=met,
            criteria_total=total,
            details=dim_verdicts,
        )

    overall = sum(
        dimensions[dim].score * weight
        for dim, weight in DIMENSION_WEIGHTS.items()
        if dim in dimensions
    )

    elapsed = time.time() - start
    log.info("judge_benchmark_complete", query_id=query_id, pattern=pattern_name,
             overall=f"{overall:.3f}", tokens=total_tokens,
             time=f"{elapsed:.1f}s")

    return JudgeResult(
        query_id=query_id,
        pattern_name=pattern_name,
        overall_score=overall,
        dimensions=dimensions,
        verdicts=verdicts,
        judge_model=JUDGE_MODEL,
        latency_seconds=elapsed,
        total_tokens=total_tokens,
    )
