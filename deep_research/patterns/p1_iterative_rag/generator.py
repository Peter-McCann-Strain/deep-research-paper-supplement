"""Stage 3: Report generation from structured source extractions."""

from __future__ import annotations

from typing import List

from deep_research.tools import SourceExtraction, format_extractions_as_evidence
from deep_research.tools.llm_caller import LLMCaller
from deep_research.config import DEFAULT_MODEL

GENERATE_PROMPT = """You are a research report writer. Based on the following source extractions,
write a comprehensive research report answering the query.

Requirements:
- Use inline citations [1], [2], etc. corresponding to the source numbers below
- Include an abstract, multiple sections with clear headings, and a conclusion
- Be specific: include numbers, dates, method names, benchmark results where available
- Leverage key findings, data points, and methodology details from the evidence
- Acknowledge limitations and areas of disagreement in the literature
- Aim for 2000-4000 words

Query: {query}

Evidence (numbered):
{evidence}

Write the report in markdown format.
"""


async def generate_report(
    query: str,
    extractions: List[SourceExtraction],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a research report from structured source extractions."""
    evidence = format_extractions_as_evidence(extractions)
    report = await llm.complete(
        GENERATE_PROMPT.format(query=query, evidence=evidence),
        model=model,
        max_tokens=8192,
        temperature=0.3,
    )
    return report
