"""Usage and cost summarization helpers for live reproduction runs."""

from __future__ import annotations

from typing import Any

from deep_research.settings import PublicSettings


def _usage_token_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        usage = item.get("usage", {}) if isinstance(item.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens
    return totals


def _judge_token_totals(judge_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    result_records = 0
    failed_result_records = 0
    for judge_result in judge_results:
        for result in judge_result.get("results", []):
            result_records += 1
            if result.get("status") != "success":
                failed_result_records += 1
                continue
            provider = str(result.get("provider") or "unknown")
            model = str(result.get("model") or result.get("configured_model") or "unknown")
            for buckets, key in ((by_provider, provider), (by_model, model)):
                bucket = buckets.setdefault(
                    key, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                )
                bucket["calls"] += 1
                bucket["input_tokens"] += int(result.get("input_tokens") or 0)
                bucket["output_tokens"] += int(result.get("output_tokens") or 0)
    return {
        "by_provider": by_provider,
        "by_model": by_model,
        "provider_call_count": sum(v["calls"] for v in by_provider.values()),
        "result_records": result_records,
        "failed_result_records": failed_result_records,
    }


def _actual_usage_summary(
    settings: PublicSettings,
    generation_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_attempts = len(generation_results)
    generation_successes = sum(1 for item in generation_results if item.get("status") == "success")
    judge_totals = _judge_token_totals(judge_results)
    estimated_incurred_usd = (
        generation_attempts * settings.cost.openai_generation_usd_per_call
        + generation_successes * settings.cost.openai_web_search_usd_per_call
    )
    provider_calls = judge_totals["by_provider"]
    model_calls = judge_totals["by_model"]
    estimated_incurred_usd += (
        provider_calls.get("openai", {}).get("calls", 0)
        * settings.cost.openai_judge_usd_per_call
    )
    estimated_incurred_usd += (
        model_calls.get(settings.anthropic.opus_model, {}).get("calls", 0)
        * settings.cost.anthropic_opus_judge_usd_per_call
    )
    estimated_incurred_usd += (
        model_calls.get(settings.anthropic.sonnet_model, {}).get("calls", 0)
        * settings.cost.anthropic_sonnet_judge_usd_per_call
    )
    return {
        "generation_attempts": generation_attempts,
        "generation_successes": generation_successes,
        "generation_tokens": _usage_token_totals(generation_results),
        "judge_tokens": judge_totals,
        "estimated_incurred_usd": round(estimated_incurred_usd, 4),
        "basis": "Actual token counts where providers returned usage; dollar values use configured per-call estimates.",
    }
