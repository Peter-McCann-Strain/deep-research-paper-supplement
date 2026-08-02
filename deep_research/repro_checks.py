"""Configuration checks shared by public reproduction commands."""

from __future__ import annotations

from deep_research.repro_queries import _dedupe
from deep_research.settings import PublicSettings


def _api_generation_unsupported(settings: PublicSettings) -> list[str]:
    if settings.openai.use_azure and settings.openai.azure_api_version != "v1":
        return [
            (
                "Azure OpenAI hosted-search generation requires AZURE_OPENAI_API_VERSION=v1 "
                "and a deployment entitled for Responses hosted web_search."
            )
        ]
    return []


def _judge_missing(settings: PublicSettings) -> list[str]:
    missing = list(settings.openai.missing_for_judging())
    if not settings.has_anthropic:
        missing.append("ANTHROPIC_API_KEY")
    return _dedupe(missing)
