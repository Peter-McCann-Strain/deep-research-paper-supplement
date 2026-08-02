"""Token/cost tracking with budget enforcement."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List

import structlog

from deep_research.config import MODELS, MAX_COST_PER_RUN
from deep_research.types import LLMUsage

log = structlog.get_logger()


class BudgetExceeded(Exception):
    """Raised when a run exceeds its cost or token budget."""

    def __init__(self, message: str = "", current: float = 0, limit: float = 0):
        self.current = current
        self.limit = limit
        if not message:
            message = f"Budget exceeded: ${current:.4f} >= ${limit:.4f}"
        super().__init__(message)


@dataclass
class CostTracker:
    """Thread-safe per-run cost tracker with both cost and token budgets."""

    budget_usd: float = MAX_COST_PER_RUN
    total_token_budget: int = 0  # 0 = unlimited
    _records: List[LLMUsage] = field(default_factory=list)
    _total_cost: float = 0.0
    _total_tokens_used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, model: str, input_tokens: int, output_tokens: int,
               call_type: str = "complete") -> LLMUsage:
        spec = MODELS.get(model)
        if spec:
            cost = (
                (input_tokens / 1000) * spec.cost_per_1k_input
                + (output_tokens / 1000) * spec.cost_per_1k_output
            )
        else:
            # Embedding or unknown — estimate conservatively
            cost = (input_tokens + output_tokens) * 0.00001

        usage = LLMUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            call_type=call_type,
        )

        with self._lock:
            self._records.append(usage)
            self._total_cost += cost
            self._total_tokens_used += input_tokens + output_tokens

            # Check token budget inline (for PTU where cost is 0)
            if (self.total_token_budget > 0
                    and self._total_tokens_used > self.total_token_budget):
                raise BudgetExceeded(
                    f"Token budget exceeded: {self._total_tokens_used:,} / "
                    f"{self.total_token_budget:,}"
                )

        log.debug("cost_tracked", model=model, cost=f"${cost:.6f}",
                  total=f"${self._total_cost:.4f}",
                  tokens_used=self._total_tokens_used)
        return usage

    def check_budget(self) -> None:
        with self._lock:
            if self._total_cost >= self.budget_usd:
                raise BudgetExceeded(
                    current=self._total_cost, limit=self.budget_usd
                )
            if (self.total_token_budget > 0
                    and self._total_tokens_used > self.total_token_budget):
                raise BudgetExceeded(
                    f"Token budget exceeded: {self._total_tokens_used:,} / "
                    f"{self.total_token_budget:,}"
                )

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens + r.output_tokens for r in self._records)

    def summary_by_model(self) -> Dict[str, Dict]:
        with self._lock:
            by_model: Dict[str, Dict] = {}
            for r in self._records:
                if r.model not in by_model:
                    by_model[r.model] = {
                        "calls": 0, "input_tokens": 0,
                        "output_tokens": 0, "cost_usd": 0.0,
                    }
                m = by_model[r.model]
                m["calls"] += 1
                m["input_tokens"] += r.input_tokens
                m["output_tokens"] += r.output_tokens
                m["cost_usd"] += r.cost_usd
            return by_model

    def to_dict(self) -> dict:
        """Serialize full cost/token breakdown for checkpointing."""
        with self._lock:
            by_call_type: dict = {}
            by_model: Dict[str, Dict] = {}
            for r in self._records:
                ct = r.call_type
                if ct not in by_call_type:
                    by_call_type[ct] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                by_call_type[ct]["calls"] += 1
                by_call_type[ct]["input_tokens"] += r.input_tokens
                by_call_type[ct]["output_tokens"] += r.output_tokens
                by_call_type[ct]["cost_usd"] += r.cost_usd

                if r.model not in by_model:
                    by_model[r.model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                by_model[r.model]["calls"] += 1
                by_model[r.model]["input_tokens"] += r.input_tokens
                by_model[r.model]["output_tokens"] += r.output_tokens
                by_model[r.model]["cost_usd"] += r.cost_usd
            return {
                "total_cost_usd": self._total_cost,
                "total_input_tokens": sum(r.input_tokens for r in self._records),
                "total_output_tokens": sum(r.output_tokens for r in self._records),
                "llm_call_count": len(self._records),
                "by_model": by_model,
                "by_call_type": by_call_type,
            }

    def summary_text(self) -> str:
        lines = [f"Total: ${self.total_cost:.4f} | {self.total_tokens:,} tokens"]
        for model, stats in self.summary_by_model().items():
            lines.append(
                f"  {model}: {stats['calls']} calls, "
                f"{stats['input_tokens']:,}+{stats['output_tokens']:,} tokens, "
                f"${stats['cost_usd']:.4f}"
            )
        return "\n".join(lines)
