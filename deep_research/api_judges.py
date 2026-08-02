"""API-backed judge providers for public reproduction workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from deep_research.settings import AnthropicSettings, OpenAISettings

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are an exacting research-report evaluator.
Evaluate only whether the supplied report satisfies each listed criterion.
Use direct evidence from the report where possible. Do not reward verbosity.
Return JSON only, with an `evaluations` array. Each item must include:
`criterion_index`, `verdict`, `evidence`, and `reasoning`.
Use verdict `SATISFIED` or `NOT_SATISFIED`.
""".strip()


@dataclass(frozen=True)
class JudgeRequest:
    query: str
    report: str
    criteria: list[str]
    system_prompt: str


@dataclass(frozen=True)
class JudgeResponse:
    provider: str
    model: str
    content: str
    parsed: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    provider_mode: str = ""
    configured_model: str = ""
    call_model_or_deployment: str = ""


class JudgeProvider(Protocol):
    label: str
    model: str
    provider_mode: str
    configured_model: str
    call_model_or_deployment: str

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        ...


def build_judge_user_message(request: JudgeRequest) -> str:
    criteria = [{"criterion_index": idx, "criterion": text} for idx, text in enumerate(request.criteria)]
    return (
        "## Research Query\n"
        f"{request.query}\n\n"
        "## Report to Evaluate\n"
        f"{request.report}\n\n"
        "## Criteria to Evaluate\n"
        f"{json.dumps(criteria, indent=2)}\n\n"
        "Return JSON with an `evaluations` array. Each item must include "
        "`criterion_index`, `verdict`, `evidence`, and `reasoning`."
    )


def _extract_json_text(content: str) -> str:
    stripped = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


VALID_JUDGE_VERDICTS = {"SATISFIED", "NOT_SATISFIED"}
OPENAI_JUDGE_RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "research_judge_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["evaluations"],
            "properties": {
                "evaluations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["criterion_index", "verdict", "evidence", "reasoning"],
                        "properties": {
                            "criterion_index": {"type": "integer"},
                            "verdict": {
                                "type": "string",
                                "enum": sorted(VALID_JUDGE_VERDICTS),
                            },
                            "evidence": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                    },
                }
            },
        },
    }
}


def validate_judge_response(
    parsed: dict[str, Any],
    *,
    expected_criteria_count: int | None = None,
) -> None:
    evaluations = parsed.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("Judge response field `evaluations` must be a non-empty list")

    seen_indexes: set[int] = set()
    for position, item in enumerate(evaluations):
        if not isinstance(item, dict):
            raise TypeError(f"evaluation item {position} must be a JSON object")
        criterion_index = item.get("criterion_index")
        if not isinstance(criterion_index, int) or criterion_index < 0:
            raise ValueError(f"evaluation item {position} has invalid criterion_index")
        if criterion_index in seen_indexes:
            raise ValueError(f"duplicate criterion_index {criterion_index}")
        seen_indexes.add(criterion_index)

        verdict = item.get("verdict")
        if verdict not in VALID_JUDGE_VERDICTS:
            raise ValueError(
                f"evaluation item {position} has invalid verdict {verdict!r}; "
                "expected SATISFIED or NOT_SATISFIED"
            )
        for field in ("evidence", "reasoning"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation item {position} must contain non-empty {field}")

    if expected_criteria_count is None:
        return
    expected_indexes = set(range(expected_criteria_count))
    missing = sorted(expected_indexes - seen_indexes)
    unexpected = sorted(seen_indexes - expected_indexes)
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing criterion indexes {missing}")
        if unexpected:
            problems.append(f"unexpected criterion indexes {unexpected}")
        raise ValueError("Judge response criteria mismatch: " + "; ".join(problems))


def parse_json_response(
    content: str,
    *,
    expected_criteria_count: int | None = None,
) -> dict[str, Any]:
    try:
        parsed = json.loads(_extract_json_text(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TypeError("Judge response must be a JSON object")
    validate_judge_response(parsed, expected_criteria_count=expected_criteria_count)
    return parsed


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", "")
    if direct:
        return direct
    texts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _usage_value(usage: Any, *fields: str) -> int:
    for field in fields:
        value = getattr(usage, field, None) if usage else None
        if isinstance(value, int):
            return value
    return 0


class OpenAIJudgeProvider:
    """OpenAI or Azure OpenAI judge provider."""

    label = "openai"

    def __init__(
        self,
        settings: OpenAISettings,
        *,
        model: str | None = None,
        client: Any | None = None,
    ):
        self.settings = settings
        self.model = model or settings.judge_model
        self.configured_model = self.model
        self.provider_mode = "azure_openai" if settings.use_azure else "openai"
        self.call_model_or_deployment = (
            settings.judge_call_model if settings.use_azure else self.model
        )
        if client is not None:
            self._client = client
            return
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI judge support requires the `api` extra: `pip install -e .[api]`."
            ) from exc
        timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0)
        if settings.use_azure:
            deployment = settings.azure_judge_deployment or settings.azure_deployment
            if not deployment:
                raise ValueError(
                    "Azure OpenAI judging requires AZURE_OPENAI_JUDGE_DEPLOYMENT "
                    "or AZURE_OPENAI_DEPLOYMENT"
                )
            if settings.azure_api_version != "v1":
                raise ValueError(
                    "Azure OpenAI judging in the public supplement requires "
                    "AZURE_OPENAI_API_VERSION=v1 because OpenAI judging uses the Responses API"
                )
            self._client = AsyncOpenAI(
                api_key=settings.azure_api_key,
                base_url=settings.azure_v1_base_url,
                timeout=timeout,
                max_retries=0,
                default_query={"api-version": settings.azure_api_version},
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.api_key,
                timeout=timeout,
                max_retries=0,
            )

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        response = await self._client.responses.create(
            model=self.call_model_or_deployment,
            instructions=request.system_prompt,
            input=build_judge_user_message(request),
            text=OPENAI_JUDGE_RESPONSE_FORMAT,
        )
        content = _response_text(response) or "{}"
        usage = getattr(response, "usage", None)
        return JudgeResponse(
            provider=self.label,
            provider_mode=self.provider_mode,
            model=self.model,
            configured_model=self.configured_model,
            call_model_or_deployment=self.call_model_or_deployment,
            content=content,
            parsed=parse_json_response(content, expected_criteria_count=len(request.criteria)),
            input_tokens=_usage_value(usage, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_value(usage, "output_tokens", "completion_tokens"),
        )


class AnthropicJudgeProvider:
    """Anthropic API judge provider for Claude Opus/Sonnet panels."""

    label = "anthropic"
    provider_mode = "anthropic"

    def __init__(self, settings: AnthropicSettings, *, model: str):
        self.settings = settings
        self.model = model
        self.configured_model = model
        self.call_model_or_deployment = model
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic judge support requires the optional `api` extra: "
                "`pip install -e .[api]`."
            ) from exc
        self._client = AsyncAnthropic(api_key=settings.api_key, max_retries=0)

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=request.system_prompt,
            messages=[{"role": "user", "content": build_judge_user_message(request)}],
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return JudgeResponse(
            provider=self.label,
            provider_mode=self.provider_mode,
            model=self.model,
            configured_model=self.configured_model,
            call_model_or_deployment=self.call_model_or_deployment,
            content=content,
            parsed=parse_json_response(content, expected_criteria_count=len(request.criteria)),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )


def public_paper_a_panel(
    openai: OpenAISettings,
    anthropic: AnthropicSettings,
) -> list[JudgeProvider]:
    """Return the public Paper A judge panel: OpenAI anchor plus Claude API judges."""
    providers: list[JudgeProvider] = [OpenAIJudgeProvider(openai)]
    if anthropic.api_key:
        providers.extend(
            [
                AnthropicJudgeProvider(anthropic, model=anthropic.opus_model),
                AnthropicJudgeProvider(anthropic, model=anthropic.sonnet_model),
            ]
        )
    return providers
