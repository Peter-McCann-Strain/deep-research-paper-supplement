"""12-dimension evaluation rubric for the MERIDIAN quality evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Dimension:
    """A single evaluation dimension."""
    name: str
    description: str
    scoring_guide: str  # what each score band (1-3, 4-6, 7-10) means


# ── The 12-Dimension Rubric ────────────────────────────────────────────────────

DIMENSIONS: List[Dimension] = [
    Dimension(
        name="coverage",
        description="How thoroughly does the report address all facets of the query?",
        scoring_guide=(
            "1-3: Major aspects of the query are missing or barely mentioned. "
            "4-6: Most aspects are addressed but some notable gaps remain. "
            "7-10: All key aspects are thoroughly covered with appropriate depth."
        ),
    ),
    Dimension(
        name="accuracy",
        description="Are the factual claims correct and well-supported by cited sources?",
        scoring_guide=(
            "1-3: Contains multiple factual errors or unsupported claims. "
            "4-6: Mostly accurate but some claims lack adequate support. "
            "7-10: Claims are accurate, well-sourced, and any uncertainty is acknowledged."
        ),
    ),
    Dimension(
        name="depth",
        description="Does the report go beyond surface-level treatment into nuanced analysis?",
        scoring_guide=(
            "1-3: Stays at surface level, merely restating common knowledge. "
            "4-6: Some depth on key topics but analysis remains shallow in places. "
            "7-10: Provides deep, insightful analysis with expert-level nuance."
        ),
    ),
    Dimension(
        name="breadth",
        description="Does the report incorporate diverse sources, perspectives, and disciplines?",
        scoring_guide=(
            "1-3: Relies on a narrow set of sources or a single perspective. "
            "4-6: Draws from multiple sources but misses some relevant viewpoints. "
            "7-10: Incorporates diverse, multi-disciplinary perspectives comprehensively."
        ),
    ),
    Dimension(
        name="coherence",
        description="Is the report logically structured with smooth transitions and a clear narrative?",
        scoring_guide=(
            "1-3: Disorganized, hard to follow, or contradicts itself. "
            "4-6: Generally organized but some sections lack logical flow. "
            "7-10: Seamlessly structured with a compelling narrative thread throughout."
        ),
    ),
    Dimension(
        name="citation_quality",
        description="Are citations relevant, recent, authoritative, and properly integrated?",
        scoring_guide=(
            "1-3: Few or no citations, or citations are irrelevant/low-quality. "
            "4-6: Adequate citations but some are tangential or outdated. "
            "7-10: Authoritative, recent, and precisely placed citations throughout."
        ),
    ),
    Dimension(
        name="specificity",
        description="Does the report provide concrete data, examples, and specific evidence?",
        scoring_guide=(
            "1-3: Vague generalities with no concrete data or examples. "
            "4-6: Some specifics but relies too much on generalizations. "
            "7-10: Rich with concrete data points, statistics, case studies, and examples."
        ),
    ),
    Dimension(
        name="balance",
        description="Does the report present multiple sides fairly without undue bias?",
        scoring_guide=(
            "1-3: Clearly one-sided or ignores significant counter-arguments. "
            "4-6: Acknowledges other views but favors one side disproportionately. "
            "7-10: Fair, balanced treatment of competing perspectives and trade-offs."
        ),
    ),
    Dimension(
        name="methodology",
        description="Does the report explain how evidence was gathered and how conclusions were drawn?",
        scoring_guide=(
            "1-3: No mention of methodology or evidence evaluation approach. "
            "4-6: Some methodology transparency but gaps in explaining reasoning. "
            "7-10: Clear about evidence hierarchy, search strategy, and reasoning process."
        ),
    ),
    Dimension(
        name="recency",
        description="Does the report incorporate the most current available information?",
        scoring_guide=(
            "1-3: Relies primarily on outdated information. "
            "4-6: Mix of recent and older sources; misses some recent developments. "
            "7-10: Up-to-date with the latest findings, trends, and developments."
        ),
    ),
    Dimension(
        name="clarity",
        description="Is the writing clear, precise, and accessible to the intended audience?",
        scoring_guide=(
            "1-3: Confusing, jargon-heavy, or poorly written. "
            "4-6: Generally clear but some passages are hard to parse. "
            "7-10: Exceptionally clear, precise language appropriate for the audience."
        ),
    ),
    Dimension(
        name="actionability",
        description="Does the report provide actionable insights, recommendations, or next steps?",
        scoring_guide=(
            "1-3: No practical takeaways or recommendations. "
            "4-6: Some general recommendations but they lack specificity. "
            "7-10: Clear, specific, actionable recommendations grounded in the evidence."
        ),
    ),
]

DIMENSION_MAP: Dict[str, Dimension] = {d.name: d for d in DIMENSIONS}
DIMENSION_NAMES: List[str] = [d.name for d in DIMENSIONS]

# ── Prompt template for a single judge ─────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an expert research-report evaluator. You will score a research "
    "report across 12 quality dimensions, each on a 1-10 scale. Be rigorous "
    "and calibrated: a score of 7 represents 'good professional quality', "
    "9-10 is reserved for truly exceptional work."
)


def build_judge_prompt(query: str, report_text: str) -> str:
    """Build the evaluation prompt that a judge model will receive."""
    dim_block = "\n".join(
        f"  {i}. **{d.name}** — {d.description}\n     Scoring: {d.scoring_guide}"
        for i, d in enumerate(DIMENSIONS, 1)
    )

    return f"""\
Evaluate the following research report that was generated in response to the query.

## Original Query
{query}

## Research Report
{report_text}

## Evaluation Dimensions
{dim_block}

## Instructions
For EACH dimension, provide:
1. A score from 1 to 10.
2. A brief rationale (1-2 sentences) explaining the score.

Then provide an "overall" score (1-10) summarizing your holistic assessment.

Respond in JSON with this exact structure:
{{
  "dimensions": {{
    "<dimension_name>": {{
      "score": <int 1-10>,
      "rationale": "<string>"
    }},
    ...
  }},
  "overall": {{
    "score": <int 1-10>,
    "rationale": "<string>"
  }}
}}
"""


def build_revision_prompt(
    query: str,
    report_text: str,
    evaluation_feedback: str,
) -> str:
    """Build a revision prompt incorporating evaluator feedback."""
    return f"""\
You are an expert research writer. A panel of judges has evaluated your report and
provided detailed feedback. Revise the report to address their concerns while
maintaining the strengths they identified.

## Original Query
{query}

## Current Report
{report_text}

## Evaluator Feedback
{evaluation_feedback}

## Instructions
1. Address every specific critique from the evaluators.
2. Maintain or improve the areas that scored well.
3. Keep all existing citations and add new ones where needed.
4. Ensure every factual claim is grounded in the source material.
5. Preserve the overall structure unless the evaluators flagged structural issues.

Write the revised report in full (do not summarize or abbreviate sections).
Use markdown formatting with ## section headers.
"""
