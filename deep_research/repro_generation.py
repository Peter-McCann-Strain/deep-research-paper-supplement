"""OpenAI/Azure generation helpers for public reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research.repro_queries import _query_file_stem
from deep_research.settings import PublicSettings


def _generation_prompt(query_record: dict[str, Any]) -> str:
    query = query_record["query"]
    rubric = query_record.get("rubric", {})
    criteria = rubric.get("criteria", []) if isinstance(rubric, dict) else []
    criteria_text = "\n".join(
        f"- {item.get('text', item)}" for item in criteria[:8] if isinstance(item, (dict, str))
    )
    return (
        "Write a concise, citation-rich research report for the paper supplement. "
        "Use current web evidence where the hosted search tool is available. "
        "Be explicit about uncertainty and limitations.\n\n"
        f"Research query:\n{query}\n\n"
        f"Public rubric hints:\n{criteria_text or 'No query-specific rubric hints supplied.'}\n\n"
        "Return Markdown with a title, short abstract, sections, inline citations, and references."
    )


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


def _response_output_items(response: Any) -> list[Any]:
    output = getattr(response, "output", [])
    if isinstance(output, list):
        return output
    return []


def _output_item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type") or "")
    return str(getattr(item, "type", "") or "")


def _response_output_types(response: Any) -> list[str]:
    return [_output_item_type(item) for item in _response_output_items(response)]


def _response_used_web_search(response: Any) -> bool:
    return "web_search_call" in _response_output_types(response)


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    data: dict[str, Any] = {}
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        value = getattr(usage, field, None)
        if value is not None:
            data[field] = value
    return data


def _openai_client(settings: PublicSettings) -> tuple[Any, str, str]:
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI generation requires the `api` extra: `pip install -e .[api]`."
        ) from exc
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    if settings.openai.use_azure:
        return (
            AsyncOpenAI(
                api_key=settings.openai.azure_api_key,
                base_url=settings.openai.azure_v1_base_url,
                timeout=timeout,
                max_retries=0,
                default_query={"api-version": settings.openai.azure_api_version},
            ),
            "azure_openai",
            settings.openai.generation_call_model,
        )
    return (
        AsyncOpenAI(api_key=settings.openai.api_key, timeout=timeout, max_retries=0),
        "openai",
        settings.openai.generation_call_model,
    )


async def _generate_report(
    settings: PublicSettings,
    query_record: dict[str, Any],
    output_dir: Path,
    *,
    openai_client_factory: Any | None = None,
) -> dict[str, Any]:
    query_id = _query_file_stem(query_record)
    markdown_path = output_dir / f"{query_id}.md"
    json_path = output_dir / f"{query_id}.json"
    client_factory = openai_client_factory or _openai_client
    client, provider_mode, call_model = client_factory(settings)
    tool_type = settings.search.openai_web_search_tool
    result = {
        "query_id": query_id,
        "source_query_id": str(query_record.get("id") or ""),
        "query": query_record["query"],
        "provider": provider_mode,
        "provider_mode": provider_mode,
        "configured_model": settings.openai.model,
        "call_model_or_deployment": call_model,
        "web_search_tool": tool_type,
        "web_search_required": True,
        "web_search_used": False,
        "response_output_types": [],
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "status": "failed",
        "error_type": "",
        "error_message": "",
        "usage": {},
    }
    try:
        response = await client.responses.create(
            model=call_model,
            input=_generation_prompt(query_record),
            tools=[{"type": tool_type}],
            tool_choice="required",
            max_output_tokens=4096,
            store=False,
        )
        response_output_types = _response_output_types(response)
        web_search_used = _response_used_web_search(response)
        result.update(
            {
                "web_search_used": web_search_used,
                "response_output_types": response_output_types,
            }
        )
        if not web_search_used:
            raise ValueError("OpenAI response did not include a web_search_call output item")
        markdown = _response_text(response)
        if not markdown:
            raise ValueError("OpenAI response did not contain output text")
    except Exception as exc:  # noqa: BLE001 - per-query provider failures are captured in result JSON
        result.update(
            {
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )
    else:
        markdown_path.write_text(markdown + "\n")
        result.update({"status": "success", "usage": _usage_dict(response)})
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


async def _generate_reports(
    settings: PublicSettings,
    query_records: list[dict[str, Any]],
    output_dir: Path,
    *,
    openai_client_factory: Any | None = None,
) -> list[dict[str, Any]]:
    results = []
    for query_record in query_records:
        results.append(
            await _generate_report(
                settings,
                query_record,
                output_dir,
                openai_client_factory=openai_client_factory,
            )
        )
    return results
