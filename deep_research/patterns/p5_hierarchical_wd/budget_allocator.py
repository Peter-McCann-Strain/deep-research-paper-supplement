"""Budget allocator: dynamic budget allocation between width and depth.

Uses gpt-5.2 to interpret meta-evaluation feedback and adjust the
width-depth schedule for the next iteration via the WDSchedule rebalance.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from deep_research.tools.llm_caller import LLMCaller
from deep_research.tools.cost_tracker import CostTracker
from deep_research.types import WDAllocation

from .wd_schedule import WDSchedule
from deep_research.config import DEFAULT_MODEL

log = structlog.get_logger()

REBALANCE_PROMPT = """Based on the meta-evaluation of research progress, recommend how to
allocate the remaining budget between broad search (width) and deep analysis (depth).

Current State:
- Step: {step} of {max_steps}
- Budget remaining: ${remaining:.4f}
- Coverage score: {coverage:.2f}
- Quality score: {quality:.2f}
- Overall score: {overall:.2f}
- Uncovered subtopics: {uncovered}
- Improvement suggestions: {suggestions}
- Current width workers: {current_width}
- Current depth iterations: {current_depth}

Recommend allocation adjustments in JSON:
{{
    "coverage_assessment": "brief assessment of whether more breadth or depth is needed",
    "recommended_coverage_score": 0.0-1.0,
    "prioritize": "width" or "depth" or "balanced",
    "rationale": "why this allocation"
}}"""


class BudgetAllocator:
    """Manages dynamic budget reallocation between width and depth phases.

    After each meta-evaluation, the allocator:
    1. Consults gpt-5.2 for rebalancing advice
    2. Translates that into a concrete WDAllocation via WDSchedule.rebalance()
    3. Tracks cumulative spending across phases
    """

    def __init__(
        self,
        schedule: WDSchedule,
        cost_tracker: CostTracker,
        llm: LLMCaller,
    ):
        self.schedule = schedule
        self.cost_tracker = cost_tracker
        self.llm = llm

    async def initial_allocation(self, step: int = 0) -> WDAllocation:
        """Get the initial allocation for step 0 (no meta-eval yet)."""
        remaining = self.schedule.total_budget - self.cost_tracker.total_cost
        alloc = self.schedule.allocate(step, remaining)
        log.info("budget_initial_allocation", step=step)
        return alloc

    async def rebalance(
        self,
        step: int,
        meta_eval: Dict[str, Any],
        current_allocation: WDAllocation,
    ) -> WDAllocation:
        """Rebalance budget allocation based on meta-evaluation feedback.

        Args:
            step: The next step index.
            meta_eval: Results from meta_evaluator.evaluate_progress().
            current_allocation: The allocation used in the most recent step.

        Returns:
            New WDAllocation for the next step.
        """
        remaining = self.schedule.total_budget - self.cost_tracker.total_cost
        if remaining <= 0:
            log.warning("budget_exhausted")
            return WDAllocation(
                step=step,
                width_budget=0,
                depth_budget=0,
                width_workers=0,
                depth_iterations=0,
            )

        coverage = meta_eval.get("coverage_score", 0.5)
        quality = meta_eval.get("quality_score", 0.5)
        overall = meta_eval.get("overall_score", 0.5)
        uncovered = meta_eval.get("uncovered_subtopics", [])
        suggestions = meta_eval.get("improvement_suggestions", [])

        # Ask gpt-5.2 for rebalancing advice
        try:
            advice = await self.llm.complete_json(
                REBALANCE_PROMPT.format(
                    step=step,
                    max_steps=self.schedule.max_steps,
                    remaining=remaining,
                    coverage=coverage,
                    quality=quality,
                    overall=overall,
                    uncovered=", ".join(uncovered[:5]) if uncovered else "None",
                    suggestions=", ".join(suggestions[:3]) if suggestions else "None",
                    current_width=current_allocation.width_workers,
                    current_depth=current_allocation.depth_iterations,
                ),
                model=DEFAULT_MODEL,
                temperature=0.2,
                max_tokens=512,
            )
            # Use the recommended coverage score for rebalancing
            effective_coverage = advice.get("recommended_coverage_score", coverage)
        except Exception as e:
            log.warning("budget_rebalance_llm_error", error=str(e))
            effective_coverage = coverage

        # Apply rebalance via schedule
        alloc = self.schedule.rebalance(
            step=step,
            remaining_budget=remaining,
            coverage_score=effective_coverage,
        )

        log.info(
            "budget_rebalanced",
            step=step,
            remaining=f"${remaining:.4f}",
            coverage=f"{coverage:.2f}",
            effective_coverage=f"{effective_coverage:.2f}",
            new_width=alloc.width_workers,
            new_depth=alloc.depth_iterations,
        )

        return alloc
