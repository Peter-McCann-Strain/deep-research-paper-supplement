"""Public reproduction orchestration facade."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deep_research import (
    repro_compare,
    repro_costs,
    repro_entitlements,
    repro_generation,
    repro_provenance,
    repro_queries,
)
from deep_research.judge_runner import run_judge_file
from deep_research.repro_checks import _api_generation_unsupported, _judge_missing
from deep_research.repro_common import (
    PUBLIC_CRITERIA_PATH,
    REFERENCE_HEADLINE_PATH,
    REFERENCE_PATTERN_METRICS_CSV_PATH,
    REFERENCE_RESULTS_PATH,
    ReproductionReport,
)
from deep_research.repro_scoring import _criterion_metadata, _score_judge_report
from deep_research.repro_usage import _actual_usage_summary
from deep_research.settings import PublicSettings, ensure_runtime_dirs

_openai_client = repro_generation._openai_client


def _safe_file_stem(value: str, *, max_length: int = 96) -> str:
    return repro_queries._safe_file_stem(value, max_length=max_length)


def _query_file_stem(query_record: dict[str, Any]) -> str:
    return repro_queries._query_file_stem(query_record)


def compare_paper_a_run(settings: PublicSettings, run_summary_path: Path) -> ReproductionReport:
    return repro_compare.compare_paper_a_run(settings, run_summary_path)


def estimate_api_reproduction_cost(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
) -> dict[str, Any]:
    return repro_costs.estimate_api_reproduction_cost(
        settings,
        full=full,
        limit=limit,
        judge=judge,
    )


def run_provenance_check(settings: PublicSettings) -> ReproductionReport:
    return repro_provenance.run_provenance_check(settings)


async def _generate_report(
    settings: PublicSettings,
    query_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return await repro_generation._generate_report(
        settings,
        query_record,
        output_dir,
        openai_client_factory=_openai_client,
    )


async def _generate_reports(
    settings: PublicSettings,
    query_records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    return await repro_generation._generate_reports(
        settings,
        query_records,
        output_dir,
        openai_client_factory=_openai_client,
    )


def run_smoke_reproduction(settings: PublicSettings) -> ReproductionReport:
    """Run the no-network public smoke check."""
    reference = repro_queries.load_reference_results(settings.paths.project_root)
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    headline = json.loads(headline_path.read_text()) if headline_path.exists() else {}
    required_keys = {"paper", "reproduction_contract", "reference_metrics"}
    required_headline_keys = {"query_count", "pattern_count", "primary_ordering"}
    missing = sorted(required_keys - set(reference))
    missing_headline = sorted(required_headline_keys - set(headline))
    status = "error" if missing or missing_headline else "success"
    if status == "success":
        message = "reference summaries are present"
    else:
        message = f"missing reference keys: {missing}; missing headline keys: {missing_headline}"
    return ReproductionReport(
        mode="smoke",
        status=status,
        message=message,
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        details={
            "reference_keys": sorted(reference.keys()),
            "headline_reference_path": str(headline_path),
            "headline_reference_keys": sorted(headline.keys()),
        },
    )


def run_reference_summary(settings: PublicSettings) -> ReproductionReport:
    """Return the compact public paper-reference summary without network calls."""
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    if not headline_path.exists():
        return ReproductionReport(
            mode="reference",
            status="error",
            message=f"missing headline reference: {headline_path}",
            created_utc=datetime.now(UTC).isoformat(),
            reference_path=str(headline_path),
        )
    headline = json.loads(headline_path.read_text())
    ordering = headline.get("primary_ordering", [])
    top_patterns = ordering[:5] if isinstance(ordering, list) else []
    details = {
        "paper": headline.get("paper"),
        "query_count": headline.get("query_count"),
        "pattern_count": headline.get("pattern_count"),
        "primary_metric": headline.get("primary_metric"),
        "headline_ranges": headline.get("headline_ranges"),
        "top_patterns": top_patterns,
        "comparison_policy": headline.get("comparison_policy"),
        "pattern_metrics_csv_path": str(
            settings.paths.project_root / REFERENCE_PATTERN_METRICS_CSV_PATH
        ),
    }
    return ReproductionReport(
        mode="reference",
        status="success",
        message="compact paper reference summary is present",
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(headline_path),
        details=details,
    )


async def _verify_openai_generation_entitlement(settings: PublicSettings) -> dict[str, Any]:
    return await repro_entitlements._verify_openai_generation_entitlement(
        settings,
        openai_client_factory=_openai_client,
    )


async def _verify_openai_judge_entitlement(settings: PublicSettings) -> dict[str, Any]:
    return await repro_entitlements._verify_openai_judge_entitlement(settings)


async def _verify_anthropic_entitlement(
    settings: PublicSettings, *, model: str, label: str
) -> dict[str, Any]:
    return await repro_entitlements._verify_anthropic_entitlement(settings, model=model, label=label)


async def _verify_api_entitlements_async(
    settings: PublicSettings,
    *,
    judge: bool = False,
) -> list[dict[str, Any]]:
    checks = [await _verify_openai_generation_entitlement(settings)]
    if judge:
        checks.extend(
            [
                await _verify_openai_judge_entitlement(settings),
                await _verify_anthropic_entitlement(
                    settings, model=settings.anthropic.opus_model, label="anthropic_opus_judge"
                ),
                await _verify_anthropic_entitlement(
                    settings, model=settings.anthropic.sonnet_model, label="anthropic_sonnet_judge"
                ),
            ]
        )
    return checks


def verify_api_entitlements(settings: PublicSettings, *, judge: bool = False) -> dict[str, Any]:
    """Make explicit live API entitlement probes for model/tool access.

    This is intentionally separate from dry-run planning because it may create
    small billable provider requests.
    """
    checks = asyncio.run(_verify_api_entitlements_async(settings, judge=judge))
    statuses = {check.get("status") for check in checks}
    if statuses == {"success"}:
        status = "success"
    elif "success" in statuses:
        status = "partial"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "blocked"
    return {
        "status": status,
        "created_utc": datetime.now(UTC).isoformat(),
        "paid_probe": True,
        "judge_panel_requested": judge,
        "checks": checks,
    }


def plan_api_reproduction(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
    max_cost_usd: float | None = None,
) -> ReproductionReport:
    """Prepare a best-effort API reproduction run without launching paid calls."""
    ensure_runtime_dirs(settings)
    selected = repro_queries._select_queries(
        repro_queries._load_public_queries(settings.paths.project_root), full=full, limit=limit
    )
    cost_estimate = repro_costs._cost_estimate_for_query_count(
        settings,
        query_count=len(selected),
        judge=judge,
    )
    missing = list(settings.openai.missing_for_generation())
    unsupported = _api_generation_unsupported(settings)
    if judge:
        missing.extend(_judge_missing(settings))
    if not selected:
        missing.append("data/eval_queries_v2.json contains no public queries")
    missing = repro_queries._dedupe(missing)
    cost_block = repro_costs._cost_block_message(cost_estimate, max_cost_usd)

    out_dir = settings.paths.artifacts_dir / "reproduction"
    out_path = out_dir / "paper_a_api_reproduction_plan.json"
    details = {
        "full": full,
        "judge_requested": judge,
        "query_count": len(selected),
        "execute_command": (
            "deep-research reproduce paper-a --mode api-best-effort --execute "
            + ("--full" if full else f"--limit {limit}")
            + (" --judge" if judge else "")
            + (f" --max-cost-usd {max_cost_usd}" if max_cost_usd is not None else "")
        ),
        "judge_command_template": (
            "deep-research judge run --query-file <query.txt> --report-file <report.md> "
            "--criteria-file data/public_judge_criteria.json --panel paper-a-api"
        ),
        "openai_provider_mode": "azure_openai" if settings.openai.use_azure else "openai",
        "openai_model": settings.openai.model,
        "openai_generation_call_model": settings.openai.generation_call_model,
        "azure_api_version": settings.openai.azure_api_version if settings.openai.use_azure else "",
        "openai_judge_model": settings.openai.judge_model,
        "anthropic_opus_model": settings.anthropic.opus_model,
        "anthropic_sonnet_model": settings.anthropic.sonnet_model,
        "search_tool": settings.search.openai_web_search_tool,
        "cost_estimate": cost_estimate,
        "max_cost_usd": max_cost_usd,
        "cost_guardrail_ok": not cost_block,
        "unsupported_configuration": unsupported,
        "contract": "live API demo; not the frozen 13-pattern paper matrix",
        "note": "Best-effort rerun. Exact paper equality is not promised because live APIs drift.",
    }
    blocked_reasons = []
    if missing:
        blocked_reasons.append("missing API settings: " + ", ".join(missing))
    if unsupported:
        blocked_reasons.append("unsupported configuration: " + "; ".join(unsupported))
    if cost_block:
        blocked_reasons.append(cost_block)
    report = ReproductionReport(
        mode="api-best-effort",
        status="blocked" if blocked_reasons else "ready",
        message="; ".join(blocked_reasons) if blocked_reasons else "API settings present",
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        output_path=str(out_path),
        details=details,
    )
    out_path.write_text(report.to_json() + "\n")
    return report


def run_api_reproduction(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
    max_cost_usd: float | None = None,
) -> ReproductionReport:
    """Execute a no-download, OpenAI-hosted-search reproduction subset or full run."""
    ensure_runtime_dirs(settings)
    selected = repro_queries._select_queries(
        repro_queries._load_public_queries(settings.paths.project_root), full=full, limit=limit
    )
    cost_estimate = repro_costs._cost_estimate_for_query_count(
        settings,
        query_count=len(selected),
        judge=judge,
    )
    missing = list(settings.openai.missing_for_generation())
    unsupported = _api_generation_unsupported(settings)
    if judge:
        missing.extend(_judge_missing(settings))
    if not selected:
        missing.append("data/eval_queries_v2.json contains no public queries")
    missing = repro_queries._dedupe(missing)
    cost_block = repro_costs._cost_block_message(cost_estimate, max_cost_usd)
    if missing or unsupported or cost_block:
        reasons = []
        if missing:
            reasons.append("missing API settings: " + ", ".join(missing))
        if unsupported:
            reasons.append("unsupported configuration: " + "; ".join(unsupported))
        if cost_block:
            reasons.append(cost_block)
        return ReproductionReport(
            mode="api-best-effort",
            status="blocked",
            message="; ".join(reasons),
            created_utc=datetime.now(UTC).isoformat(),
            reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
            details={
                "missing_configuration": missing,
                "unsupported_configuration": unsupported,
                "judge_requested": judge,
                "cost_estimate": cost_estimate,
                "max_cost_usd": max_cost_usd,
                "cost_guardrail_ok": not cost_block,
            },
        )

    output_dir = settings.paths.artifacts_dir / "reproduction" / "paper_a_api_best_effort"
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_results = asyncio.run(_generate_reports(settings, selected, output_dir))

    judge_results: list[dict[str, Any]] = []
    if judge:
        fallback_criteria_path = settings.paths.project_root / PUBLIC_CRITERIA_PATH
        for query_record, generation in zip(selected, generation_results):
            if generation.get("status") != "success":
                continue
            query_id = generation["query_id"]
            query_criteria = repro_queries._criteria_from_query_record(query_record)
            criteria_source = "query_rubric"
            if query_criteria:
                criteria_path = output_dir / f"{query_id}.criteria.json"
                criteria_path.write_text(
                    json.dumps(
                        {
                            "source": criteria_source,
                            "criteria": query_criteria,
                            "criteria_metadata": _criterion_metadata(query_record, query_criteria),
                            "dimension_weights": query_record.get("rubric", {}).get(
                                "dimension_weights", {}
                            )
                            if isinstance(query_record.get("rubric"), dict)
                            else {},
                        },
                        indent=2,
                    )
                    + "\n"
                )
            else:
                criteria_source = "public_smoke_fallback"
                criteria_path = fallback_criteria_path
            judge_path = output_dir / f"{query_id}.judge.json"
            judge_report = run_judge_file(
                settings,
                query=query_record["query"],
                report_file=Path(generation["markdown_path"]),
                criteria_file=criteria_path,
                panel="paper-a-api",
                output_path=judge_path,
                dry_run=False,
            )
            judge_payload = asdict(judge_report)
            judge_payload["criteria_source"] = criteria_source
            judge_payload["criteria_count_from_query"] = len(query_criteria)
            judge_payload["score_summary"] = _score_judge_report(
                judge_payload, query_record, query_criteria
            )
            judge_results.append(judge_payload)

    summary_path = output_dir / "summary.json"
    success_count = sum(1 for result in generation_results if result.get("status") == "success")
    generation_count = len(generation_results)
    if judge:
        judge_success_count = sum(
            1 for result in judge_results if result.get("status") == "success"
        )
        expected_judge_count = success_count
        if success_count == generation_count and judge_success_count == expected_judge_count:
            status = "success"
        elif success_count or judge_success_count:
            status = "partial"
        else:
            status = "failed"
        message = (
            f"generated {success_count}/{generation_count} public API reports; "
            f"judged {judge_success_count}/{expected_judge_count} successful generations"
        )
    else:
        status = (
            "success"
            if success_count == generation_count and generation_count > 0
            else "partial"
            if success_count
            else "failed"
        )
        message = f"generated {success_count}/{generation_count} public API reports"
    details = {
        "full": full,
        "query_count": len(selected),
        "successful_generations": success_count,
        "failed_generations": generation_count - success_count,
        "judge_requested": judge,
        "successful_judges": sum(
            1 for result in judge_results if result.get("status") == "success"
        ),
        "failed_or_partial_judges": sum(
            1 for result in judge_results if result.get("status") != "success"
        ),
        "cost_estimate": cost_estimate,
        "actual_usage_summary": _actual_usage_summary(settings, generation_results, judge_results),
        "current_api_score_summaries": [
            result.get("score_summary") for result in judge_results if result.get("score_summary")
        ],
        "max_cost_usd": max_cost_usd,
        "cost_guardrail_ok": True,
        "cost_guardrail_strategy": (
            "stop before paid execution if the selected plan exceeds --max-cost-usd"
        ),
        "contract": "live API demo; not the frozen 13-pattern paper matrix",
        "generation_results": generation_results,
        "judge_results": judge_results,
    }
    report = ReproductionReport(
        mode="api-best-effort",
        status=status,
        message=message,
        created_utc=datetime.now(UTC).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        output_path=str(summary_path),
        details=details,
    )
    summary_path.write_text(report.to_json() + "\n")
    return report
