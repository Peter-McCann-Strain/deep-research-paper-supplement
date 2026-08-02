"""Width-depth schedule: W(t) = max(W_min, W_0 * alpha^t).

Controls the transition from broad exploration (width) to focused
analysis (depth) over successive research iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import structlog

from deep_research.types import WDAllocation

log = structlog.get_logger()


@dataclass
class WDSchedule:
    """Exponential decay schedule for width/depth resource allocation.

    W(t) = max(W_min, floor(W_0 * alpha^t))

    As t increases, width workers decrease and depth iterations increase,
    shifting focus from broad exploration to deep analysis.
    """

    w_0: int = 4           # initial width workers
    alpha: float = 0.5     # decay factor per step
    w_min: int = 1          # minimum width workers
    max_steps: int = 3      # maximum iteration steps
    total_budget: float = 2.0  # total USD budget for the run

    _history: List[WDAllocation] = field(default_factory=list)

    def width_at(self, step: int) -> int:
        """Compute width workers for a given step: W(t) = max(W_min, W_0 * alpha^t)."""
        raw = self.w_0 * (self.alpha ** step)
        return max(self.w_min, math.floor(raw))

    def depth_iterations_at(self, step: int) -> int:
        """Depth iterations increase as width shrinks.

        Heuristic: depth = max_steps - width_at(step) + 1, clamped to [1, max_steps].
        """
        w = self.width_at(step)
        depth = self.max_steps - w + 1
        return max(1, min(depth, self.max_steps))

    def allocate(self, step: int, remaining_budget: float) -> WDAllocation:
        """Compute width/depth budget split for a given step.

        As width decreases, a larger fraction of the remaining budget
        shifts to depth analysis.
        """
        w = self.width_at(step)
        d = self.depth_iterations_at(step)

        # Budget fraction for width decreases with step
        # At step 0: ~60% width, 40% depth
        # At later steps: width fraction shrinks proportionally
        width_fraction = w / (w + d)
        depth_fraction = 1.0 - width_fraction

        # Reserve 15% of total for citation verification and report generation
        usable = remaining_budget * 0.85
        width_budget = round(usable * width_fraction, 4)
        depth_budget = round(usable * depth_fraction, 4)

        alloc = WDAllocation(
            step=step,
            width_budget=width_budget,
            depth_budget=depth_budget,
            width_workers=w,
            depth_iterations=d,
        )

        self._history.append(alloc)
        log.info(
            "wd_schedule_allocate",
            step=step,
            width_workers=w,
            depth_iterations=d,
            width_budget=f"${width_budget:.4f}",
            depth_budget=f"${depth_budget:.4f}",
        )
        return alloc

    def rebalance(
        self,
        step: int,
        remaining_budget: float,
        coverage_score: float,
    ) -> WDAllocation:
        """Rebalance allocation based on meta-evaluation feedback.

        If coverage is low (< 0.5), keep more width workers.
        If coverage is high (>= 0.7), shift aggressively to depth.
        """
        base_w = self.width_at(step)

        if coverage_score < 0.4:
            # Poor coverage: boost width
            adjusted_w = min(base_w + 2, self.w_0)
        elif coverage_score < 0.6:
            # Moderate coverage: slight width boost
            adjusted_w = min(base_w + 1, self.w_0)
        elif coverage_score >= 0.8:
            # Good coverage: minimize width, maximize depth
            adjusted_w = self.w_min
        else:
            adjusted_w = base_w

        d = max(1, self.max_steps - adjusted_w + 1)

        width_fraction = adjusted_w / (adjusted_w + d)
        depth_fraction = 1.0 - width_fraction

        usable = remaining_budget * 0.85
        width_budget = round(usable * width_fraction, 4)
        depth_budget = round(usable * depth_fraction, 4)

        alloc = WDAllocation(
            step=step,
            width_budget=width_budget,
            depth_budget=depth_budget,
            width_workers=adjusted_w,
            depth_iterations=d,
        )

        self._history.append(alloc)
        log.info(
            "wd_schedule_rebalance",
            step=step,
            coverage=f"{coverage_score:.2f}",
            adjusted_width=adjusted_w,
            depth_iters=d,
        )
        return alloc

    @property
    def history(self) -> List[WDAllocation]:
        return list(self._history)
