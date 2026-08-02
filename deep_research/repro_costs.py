"""Cost estimation helpers for public API reproduction."""

from __future__ import annotations

from typing import Any

from deep_research.repro_queries import _load_public_queries, _select_queries
from deep_research.settings import PublicSettings


def _cost_component(name: str, provider: str, calls: int, usd_per_call: float) -> dict[str, Any]:
    estimated = calls * usd_per_call
    return {
        "name": name,
        "provider": provider,
        "calls": calls,
        "usd_per_call": round(usd_per_call, 6),
        "estimated_usd": round(estimated, 4),
    }


def _cost_estimate_for_query_count(
    settings: PublicSettings,
    *,
    query_count: int,
    judge: bool,
) -> dict[str, Any]:
    components = [
        _cost_component(
            "openai_generation_responses",
            "azure_openai" if settings.openai.use_azure else "openai",
            query_count,
            settings.cost.openai_generation_usd_per_call,
        ),
        _cost_component(
            "openai_web_search_tool",
            "azure_openai" if settings.openai.use_azure else "openai",
            query_count,
            settings.cost.openai_web_search_usd_per_call,
        ),
    ]
    judge_calls = {"openai": 0, "anthropic_opus": 0, "anthropic_sonnet": 0}
    if judge:
        judge_calls = {
            "openai": query_count,
            "anthropic_opus": query_count,
            "anthropic_sonnet": query_count,
        }
        components.extend(
            [
                _cost_component(
                    "openai_judge",
                    "openai",
                    query_count,
                    settings.cost.openai_judge_usd_per_call,
                ),
                _cost_component(
                    "anthropic_opus_judge",
                    "anthropic",
                    query_count,
                    settings.cost.anthropic_opus_judge_usd_per_call,
                ),
                _cost_component(
                    "anthropic_sonnet_judge",
                    "anthropic",
                    query_count,
                    settings.cost.anthropic_sonnet_judge_usd_per_call,
                ),
            ]
        )

    total = sum(component["estimated_usd"] for component in components)
    return {
        "estimate_version": 1,
        "query_count": query_count,
        "generation_calls": query_count,
        "web_search_tool_calls_estimated": query_count,
        "judge_requested": judge,
        "judge_calls": judge_calls,
        "components": components,
        "estimated_total_usd": round(total, 4),
        "basis": settings.cost.note,
        "overridable_env_vars": [
            "DR_COST_OPENAI_GENERATION_USD_PER_CALL",
            "DR_COST_OPENAI_WEB_SEARCH_USD_PER_CALL",
            "DR_COST_OPENAI_JUDGE_USD_PER_CALL",
            "DR_COST_ANTHROPIC_OPUS_JUDGE_USD_PER_CALL",
            "DR_COST_ANTHROPIC_SONNET_JUDGE_USD_PER_CALL",
        ],
    }


def estimate_api_reproduction_cost(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
) -> dict[str, Any]:
    """Estimate paid API calls and configurable dollar guardrail for a rerun."""
    query_count = len(
        _select_queries(_load_public_queries(settings.paths.project_root), full=full, limit=limit)
    )
    return _cost_estimate_for_query_count(settings, query_count=query_count, judge=judge)


def _cost_block_message(cost_estimate: dict[str, Any], max_cost_usd: float | None) -> str:
    if max_cost_usd is None:
        return ""
    if max_cost_usd < 0:
        raise ValueError("--max-cost-usd must be non-negative")
    estimated = float(cost_estimate["estimated_total_usd"])
    if estimated > max_cost_usd:
        return f"estimated cost ${estimated:.4f} exceeds --max-cost-usd ${max_cost_usd:.4f}"
    return ""
