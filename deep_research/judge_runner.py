"""CLI workflow for API-backed public judge runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research.api_judges import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    JudgeProvider,
    JudgeRequest,
    OpenAIJudgeProvider,
    public_paper_a_panel,
)
from deep_research.settings import PublicSettings, ensure_runtime_dirs


@dataclass(frozen=True)
class JudgeRunReport:
    status: str
    panel: str
    created_utc: str
    query: str
    report_file: str
    criteria_file: str
    output_path: str | None
    providers: list[dict[str, Any]]
    criteria_count: int
    missing_configuration: list[str]
    results: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _coerce_criterion(item: Any, *, index: int) -> str:
    if isinstance(item, str):
        text = item.strip()
    elif isinstance(item, dict):
        for key in ("criterion", "text", "description", "question", "rubric"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        else:
            raise ValueError(f"criterion item {index} has no text field")
    else:
        raise TypeError(f"criterion item {index} must be a string or object")
    if not text:
        raise ValueError(f"criterion item {index} is empty")
    return text


def load_criteria(criteria_path: Path) -> list[str]:
    """Load judge criteria from JSON, JSONL, or plain text."""
    suffix = criteria_path.suffix.lower()
    text = criteria_path.read_text()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [_coerce_criterion(item, index=idx) for idx, item in enumerate(rows)]
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("criteria", "rubric", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break
            else:
                raise ValueError("criteria JSON object must contain `criteria`, `rubric`, or `items`")
        if not isinstance(payload, list):
            raise ValueError("criteria JSON must be a list or an object containing a list")
        return [_coerce_criterion(item, index=idx) for idx, item in enumerate(payload)]

    criteria: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        criteria.append(stripped.lstrip("-*").strip())
    if not criteria:
        raise ValueError(f"no criteria found in {criteria_path}")
    return criteria


def _panel_configuration(settings: PublicSettings, panel: str) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    providers: list[dict[str, Any]] = []

    if panel not in {"paper-a-api", "openai-only"}:
        raise ValueError(f"unknown judge panel: {panel}")

    openai_missing = settings.openai.missing_for_judging()
    providers.append(
        {
            "provider": "openai",
            "provider_mode": "azure_openai" if settings.openai.use_azure else "openai",
            "configured_model": settings.openai.judge_model,
            "call_model_or_deployment": settings.openai.judge_call_model,
            "configured": not openai_missing,
        }
    )
    missing.extend(openai_missing)

    if panel == "paper-a-api":
        providers.extend(
            [
                {
                    "provider": "anthropic",
                    "provider_mode": "anthropic",
                    "configured_model": settings.anthropic.opus_model,
                    "call_model_or_deployment": settings.anthropic.opus_model,
                    "configured": settings.has_anthropic,
                },
                {
                    "provider": "anthropic",
                    "provider_mode": "anthropic",
                    "configured_model": settings.anthropic.sonnet_model,
                    "call_model_or_deployment": settings.anthropic.sonnet_model,
                    "configured": settings.has_anthropic,
                },
            ]
        )
        if not settings.has_anthropic:
            missing.append("ANTHROPIC_API_KEY")

    return providers, missing


def _build_providers(settings: PublicSettings, panel: str) -> list[JudgeProvider]:
    if panel == "paper-a-api":
        return public_paper_a_panel(settings.openai, settings.anthropic)
    if panel == "openai-only":
        return [OpenAIJudgeProvider(settings.openai)]
    raise ValueError(f"unknown judge panel: {panel}")


def _provider_metadata(provider: JudgeProvider) -> dict[str, Any]:
    return {
        "provider": getattr(provider, "label", provider.__class__.__name__),
        "provider_mode": getattr(provider, "provider_mode", ""),
        "model": getattr(provider, "model", ""),
        "configured_model": getattr(provider, "configured_model", getattr(provider, "model", "")),
        "call_model_or_deployment": getattr(
            provider,
            "call_model_or_deployment",
            getattr(provider, "model", ""),
        ),
    }


async def _evaluate_all(providers: list[JudgeProvider], request: JudgeRequest) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for provider in providers:
        base = _provider_metadata(provider)
        try:
            response = await provider.evaluate(request)
        except Exception as exc:  # noqa: BLE001 - provider failures are serialized into judge results
            results.append(
                {
                    **base,
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
        else:
            results.append({**asdict(response), "status": "success"})
    return results


def _status_from_results(results: list[dict[str, Any]]) -> str:
    if all(result.get("status") == "success" for result in results):
        return "success"
    if any(result.get("status") == "success" for result in results):
        return "partial"
    return "failed"


def run_judge_file(
    settings: PublicSettings,
    *,
    query: str,
    report_file: Path,
    criteria_file: Path,
    panel: str = "paper-a-api",
    output_path: Path | None = None,
    dry_run: bool = False,
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
) -> JudgeRunReport:
    """Run or preview a public API-backed judge panel on one report."""
    criteria = load_criteria(criteria_file)
    report_text = report_file.read_text()
    providers, missing = _panel_configuration(settings, panel)

    if output_path is None and not dry_run:
        ensure_runtime_dirs(settings)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = settings.paths.artifacts_dir / "judges" / f"judge_run_{stamp}.json"

    status = "dry-run" if dry_run else "blocked" if missing else "success"
    results: list[dict[str, Any]] = []
    if not dry_run and not missing:
        request = JudgeRequest(
            query=query,
            report=report_text,
            criteria=criteria,
            system_prompt=system_prompt,
        )
        try:
            built_providers = _build_providers(settings, panel)
        except Exception as exc:  # noqa: BLE001 - provider construction failures are serialized
            status = "failed"
            results = [
                {
                    "provider": "panel",
                    "provider_mode": panel,
                    "model": "",
                    "configured_model": "",
                    "call_model_or_deployment": "",
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            ]
        else:
            results = asyncio.run(_evaluate_all(built_providers, request))
            status = _status_from_results(results)

    report = JudgeRunReport(
        status=status,
        panel=panel,
        created_utc=datetime.now(UTC).isoformat(),
        query=query,
        report_file=str(report_file),
        criteria_file=str(criteria_file),
        output_path=str(output_path) if output_path else None,
        providers=providers,
        criteria_count=len(criteria),
        missing_configuration=missing,
        results=results,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json() + "\n")

    return report
