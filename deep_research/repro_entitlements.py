"""Live API entitlement probes for public reproduction."""

from __future__ import annotations

from typing import Any

from deep_research.api_judges import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    JudgeRequest,
    OpenAIJudgeProvider,
)
from deep_research.repro_checks import _api_generation_unsupported
from deep_research.repro_generation import (
    _openai_client,
    _response_output_types,
    _response_used_web_search,
    _usage_dict,
)
from deep_research.settings import PublicSettings


async def _verify_openai_generation_entitlement(
    settings: PublicSettings,
    *,
    openai_client_factory: Any | None = None,
) -> dict[str, Any]:
    unsupported = _api_generation_unsupported(settings)
    check = {
        "name": "openai_generation_with_hosted_search",
        "provider": "azure_openai" if settings.openai.use_azure else "openai",
        "configured_model": settings.openai.model,
        "call_model_or_deployment": settings.openai.generation_call_model,
        "search_tool": settings.search.openai_web_search_tool,
        "status": "blocked" if unsupported else "pending",
    }
    if unsupported:
        return {**check, "message": "; ".join(unsupported)}
    missing = settings.openai.missing_for_generation()
    if missing:
        return {**check, "status": "blocked", "missing_configuration": missing}
    try:
        client_factory = openai_client_factory or _openai_client
        client, provider_mode, call_model = client_factory(settings)
        response = await client.responses.create(
            model=call_model,
            input="Use web search if available, then reply with the word OK.",
            tools=[{"type": settings.search.openai_web_search_tool}],
            tool_choice="required",
            max_output_tokens=32,
            store=False,
        )
        response_output_types = _response_output_types(response)
        web_search_used = _response_used_web_search(response)
        if not web_search_used:
            return {
                **check,
                "provider_mode": provider_mode,
                "call_model_or_deployment": call_model,
                "status": "failed",
                "error_type": "MissingWebSearchCall",
                "error_message": "Provider response did not include a web_search_call output item",
                "response_output_types": response_output_types,
            }
        return {
            **check,
            "provider_mode": provider_mode,
            "call_model_or_deployment": call_model,
            "status": "success",
            "web_search_used": True,
            "response_output_types": response_output_types,
            "usage": _usage_dict(response),
        }
    except Exception as exc:  # noqa: BLE001 - provider entitlement failures are reported as JSON
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }


async def _verify_openai_judge_entitlement(settings: PublicSettings) -> dict[str, Any]:
    check = {
        "name": "openai_judge",
        "provider": "azure_openai" if settings.openai.use_azure else "openai",
        "configured_model": settings.openai.judge_model,
        "call_model_or_deployment": settings.openai.judge_call_model,
        "status": "pending",
    }
    missing = settings.openai.missing_for_judging()
    if missing:
        return {**check, "status": "blocked", "missing_configuration": missing}
    try:
        provider = OpenAIJudgeProvider(settings.openai)
        response = await provider.evaluate(
            JudgeRequest(
                query="Public judge entitlement probe.",
                report="This report says OK and cites no external claims.",
                criteria=["The report includes the token OK."],
                system_prompt=DEFAULT_JUDGE_SYSTEM_PROMPT,
            )
        )
        return {
            **check,
            "provider_mode": response.provider_mode,
            "call_model_or_deployment": response.call_model_or_deployment,
            "status": "success",
            "usage": {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        }
    except Exception as exc:  # noqa: BLE001 - provider entitlement failures are reported as JSON
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }


async def _verify_anthropic_entitlement(settings: PublicSettings, *, model: str, label: str) -> dict[str, Any]:
    check = {
        "name": label,
        "provider": "anthropic",
        "configured_model": model,
        "call_model_or_deployment": model,
        "status": "pending",
    }
    if not settings.has_anthropic:
        return {**check, "status": "blocked", "missing_configuration": ["ANTHROPIC_API_KEY"]}
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": "Anthropic verification requires the `api` extra: `pip install -e .[api]`.",
        }
    try:
        client = AsyncAnthropic(api_key=settings.anthropic.api_key, max_retries=0)
        response = await client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        usage = getattr(response, "usage", None)
        return {
            **check,
            "status": "success",
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            },
        }
    except Exception as exc:  # noqa: BLE001 - provider entitlement failures are reported as JSON
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
