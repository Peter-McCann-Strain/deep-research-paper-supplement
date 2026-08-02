"""Release-facing command line for the public supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deep_research.judge_runner import run_judge_file
from deep_research.paper_rebuild import run_paper_rebuild
from deep_research.public_export import export_public_tree
from deep_research.release_audit import audit_release_tree
from deep_research.reproduce import (
    compare_paper_a_run,
    estimate_api_reproduction_cost,
    plan_api_reproduction,
    run_api_reproduction,
    run_provenance_check,
    run_reference_summary,
    run_smoke_reproduction,
    verify_api_entitlements,
)
from deep_research.settings import ensure_runtime_dirs, load_public_settings


def _settings(args: argparse.Namespace):
    root = Path(args.project_root).resolve() if getattr(args, "project_root", None) else None
    return load_public_settings(project_root=root)


def _read_query(args: argparse.Namespace) -> str:
    if args.query_file:
        return Path(args.query_file).read_text().strip()
    return args.query.strip()


def _import_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _dedupe_for_cli(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    missing_generation = settings.openai.missing_for_generation()
    missing_judging = settings.openai.missing_for_judging()
    missing_judge_panel = _dedupe_for_cli(missing_judging)
    if not settings.has_anthropic:
        missing_judge_panel.append("ANTHROPIC_API_KEY")
    missing_api = sorted(set(missing_generation + missing_judge_panel))
    sdk_checks = {
        "openai": _import_available("openai"),
        "anthropic": _import_available("anthropic"),
    }
    checks = {
        "project_root": str(settings.paths.project_root),
        "data_dir_exists": settings.paths.data_dir.exists(),
        "paper_dir_exists": settings.paths.paper_dir.exists(),
        "generation_configured": not missing_generation,
        "judge_panel_configured": not missing_judge_panel,
        "openai_configured": not missing_generation,
        "anthropic_configured": settings.has_anthropic,
        "azure_enabled": settings.openai.use_azure,
        "azure_public_generation_supported": (
            not settings.openai.use_azure or settings.openai.azure_api_version == "v1"
        ),
        "missing_generation_configuration": missing_generation,
        "missing_judge_panel_configuration": missing_judge_panel,
        "missing_api_configuration": missing_api,
        "sdk_imports": sdk_checks,
        "default_contract": "live API demo plus frozen-reference comparison; not a bitwise rerun",
        "model_contract": (
            "Default model IDs are release-tested/paper-compatible settings. "
            "Provider catalogs change; record exact model IDs for every paid rerun."
        ),
        "azure_contract": (
            "Azure OpenAI is supported through the OpenAI-compatible Responses API endpoint "
            "when the deployment supports the configured hosted web_search tool. Use "
            "doctor --verify-api to test entitlement before a paid rerun."
        ),
        "cost_defaults": {
            "note": settings.cost.note,
            "openai_generation_usd_per_call": settings.cost.openai_generation_usd_per_call,
            "openai_web_search_usd_per_call": settings.cost.openai_web_search_usd_per_call,
            "openai_judge_usd_per_call": settings.cost.openai_judge_usd_per_call,
            "anthropic_opus_judge_usd_per_call": settings.cost.anthropic_opus_judge_usd_per_call,
            "anthropic_sonnet_judge_usd_per_call": settings.cost.anthropic_sonnet_judge_usd_per_call,
        },
    }
    if args.ensure_dirs:
        ensure_runtime_dirs(settings)
        checks["runtime_dirs_created"] = True
    live_verification = None
    if args.verify_api:
        live_verification = verify_api_entitlements(
            settings, judge=args.verify_judge_panel or args.require_judge_panel
        )
        checks["live_api_verification"] = live_verification
    print(json.dumps(checks, indent=2))
    if args.require_api and (missing_generation or not sdk_checks["openai"]):
        return 1
    if args.require_judge_panel and (missing_judge_panel or not all(sdk_checks.values())):
        return 1
    if live_verification and live_verification.get("status") != "success":
        return 1
    return 0


def cmd_release_audit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest_path = None if not args.manifest else Path(args.manifest).resolve()
    result = audit_release_tree(
        root,
        max_file_mb=args.max_file_mb,
        manifest_path=manifest_path,
        enforce_manifest=not args.no_manifest,
    )
    print(result.to_json())
    return 0 if result.ok else 1


def cmd_export_public(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    result = export_public_tree(
        source_root,
        Path(args.out).resolve(),
        manifest_path=Path(args.manifest) if args.manifest else None,
        force=args.force,
        max_file_mb=args.max_file_mb,
    )
    print(result.to_json())
    return 0 if result.ok else 1


def cmd_reproduce(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.mode == "smoke":
        report = run_smoke_reproduction(settings)
    elif args.mode == "reference":
        report = run_reference_summary(settings)
    elif args.mode == "provenance":
        report = run_provenance_check(settings)
    elif args.mode == "api-best-effort" and args.execute:
        report = run_api_reproduction(
            settings,
            full=args.full,
            limit=args.limit,
            judge=args.judge,
            max_cost_usd=args.max_cost_usd,
        )
    elif args.mode == "api-best-effort":
        report = plan_api_reproduction(
            settings,
            full=args.full,
            limit=args.limit,
            judge=args.judge,
            max_cost_usd=args.max_cost_usd,
        )
    else:
        raise ValueError(f"Unknown reproduction mode: {args.mode}")
    print(report.to_json())
    return 0 if report.status in {"success", "ready"} else 1


def cmd_cost(args: argparse.Namespace) -> int:
    settings = _settings(args)
    estimate = estimate_api_reproduction_cost(
        settings,
        full=args.full,
        limit=args.limit,
        judge=args.judge,
    )
    print(json.dumps(estimate, indent=2))
    if args.max_cost_usd is None:
        return 0
    return 0 if estimate["estimated_total_usd"] <= args.max_cost_usd else 1


def cmd_compare(args: argparse.Namespace) -> int:
    settings = _settings(args)
    report = compare_paper_a_run(settings, Path(args.run_summary))
    print(report.to_json())
    return 0 if report.status == "success" else 1


def cmd_quickstart_check(args: argparse.Namespace) -> int:
    settings = _settings(args)
    smoke = run_smoke_reproduction(settings)
    reference = run_reference_summary(settings)
    provenance = run_provenance_check(settings)
    compare = compare_paper_a_run(
        settings, settings.paths.project_root / "repro/reference/paper_a_pattern_metrics.csv"
    )
    offline_ok = all(
        report.status == "success" for report in (smoke, reference, provenance, compare)
    )
    payload = {
        "offline_ok": offline_ok,
        "api_calls_made": False,
        "smoke": {"status": smoke.status, "message": smoke.message},
        "reference": {
            "status": reference.status,
            "query_count": (reference.details or {}).get("query_count"),
            "reference_pattern_count": (reference.details or {}).get("pattern_count"),
            "top_pattern": ((reference.details or {}).get("top_patterns") or [{}])[0].get("pattern"),
        },
        "provenance": {
            "status": provenance.status,
            "counts_match": (provenance.details or {}).get("counts_match"),
        },
        "compare": {
            "status": compare.status,
            "overlap_count": (compare.details or {}).get("overlap_count"),
            "ordering_matches_reference": (compare.details or {}).get("ordering_matches_reference"),
            "score_within_tolerance": (compare.details or {}).get("score_within_tolerance"),
        },
        "optional_paid_next_steps": [
            "deep-research doctor --verify-api",
            "deep-research reproduce paper-a --mode api-best-effort --execute --limit 3 --max-cost-usd 5",
            "deep-research doctor --verify-api --verify-judge-panel",
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if offline_ok else 1


def cmd_judge_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    report = run_judge_file(
        settings,
        query=_read_query(args),
        report_file=Path(args.report_file),
        criteria_file=Path(args.criteria_file),
        panel=args.panel,
        output_path=Path(args.out) if args.out else None,
        dry_run=args.dry_run,
    )
    print(report.to_json())
    if args.dry_run:
        return 0
    return 0 if report.status == "success" else 1


def cmd_paper_rebuild(args: argparse.Namespace) -> int:
    settings = _settings(args)
    report = run_paper_rebuild(
        settings.paths.project_root,
        check_only=args.check_only,
        tables=not args.skip_tables,
        figures=not args.skip_figures,
        compile_pdf=not args.skip_compile,
    )
    print(report.to_json())
    return 0 if report.status == "success" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deep-research")
    parser.add_argument("--project-root", default="", help="Repository root override")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Validate public supplement configuration")
    doctor.add_argument(
        "--require-api",
        action="store_true",
        help="Fail unless generation API credentials and SDK are configured",
    )
    doctor.add_argument(
        "--require-judge-panel",
        action="store_true",
        help="Fail unless the full OpenAI plus Anthropic judge panel is configured",
    )
    doctor.add_argument(
        "--ensure-dirs", action="store_true", help="Create runtime artifact directories"
    )
    doctor.add_argument(
        "--verify-api",
        action="store_true",
        help="Make a small live generation call to verify model and hosted-search entitlement",
    )
    doctor.add_argument(
        "--verify-judge-panel",
        action="store_true",
        help="With --verify-api, also make small live judge-model entitlement calls",
    )
    doctor.set_defaults(func=cmd_doctor)

    quickstart = sub.add_parser(
        "quickstart-check",
        help="Run all no-network first-run checks and print optional paid next steps",
    )
    quickstart.set_defaults(func=cmd_quickstart_check)

    audit = sub.add_parser("release-audit", help="Audit a candidate public release tree")
    audit.add_argument("--root", default=".", help="Tree to audit")
    audit.add_argument(
        "--manifest", default="", help="Manifest path. Defaults to ROOT/PUBLIC_MANIFEST.json"
    )
    audit.add_argument(
        "--no-manifest", action="store_true", help="Disable manifest allowlist enforcement"
    )
    audit.add_argument("--max-file-mb", type=int, default=None, help="Maximum public file size")
    audit.set_defaults(func=cmd_release_audit)

    export = sub.add_parser("export-public", help="Build a sanitized public release tree")
    export.add_argument("--source-root", default=".", help="Source repository root")
    export.add_argument(
        "--out", required=True, help="Output directory outside the source repository"
    )
    export.add_argument("--manifest", default="", help="Manifest path relative to source root")
    export.add_argument(
        "--force", action="store_true", help="Overwrite an earlier export from this command"
    )
    export.add_argument("--max-file-mb", type=int, default=None, help="Maximum public file size")
    export.set_defaults(func=cmd_export_public)

    reproduce = sub.add_parser("reproduce", help="Run or plan public paper reproduction")
    reproduce.add_argument("paper", choices=["paper-a"], help="Paper to reproduce")
    reproduce.add_argument(
        "--mode",
        choices=["smoke", "reference", "provenance", "api-best-effort"],
        default="smoke",
        help="Reproduction mode",
    )
    reproduce.add_argument(
        "--full", action="store_true", help="Use all public queries instead of a small subset"
    )
    reproduce.add_argument(
        "--limit",
        type=_positive_int,
        default=3,
        help="Number of public queries for non-full API mode",
    )
    reproduce.add_argument(
        "--execute", action="store_true", help="Launch the paid API generation workflow"
    )
    reproduce.add_argument(
        "--judge",
        action="store_true",
        help="After generation, score reports with the API judge panel",
    )
    reproduce.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Block execution or preflight if the configurable estimate exceeds this amount",
    )
    reproduce.set_defaults(func=cmd_reproduce)

    cost = sub.add_parser("cost", help="Estimate paid calls for the public API rerun")
    cost.add_argument("paper", choices=["paper-a"], help="Paper to estimate")
    cost.add_argument("--full", action="store_true", help="Estimate all public queries")
    cost.add_argument(
        "--limit",
        type=_positive_int,
        default=3,
        help="Number of public queries for non-full mode",
    )
    cost.add_argument(
        "--judge", action="store_true", help="Include the OpenAI plus Anthropic judge panel"
    )
    cost.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Exit nonzero if the estimate exceeds this amount",
    )
    cost.set_defaults(func=cmd_cost)

    compare = sub.add_parser("compare", help="Compare pattern-level metrics with Paper A reference")
    compare.add_argument("paper", choices=["paper-a"], help="Paper reference to compare against")
    compare.add_argument(
        "--run-summary",
        "--run",
        required=True,
        help="JSON or CSV pattern-metric summary from a comparable run",
    )
    compare.set_defaults(func=cmd_compare)

    judge = sub.add_parser("judge", help="Run public API-backed judge workflows")
    judge_sub = judge.add_subparsers(dest="judge_command", required=True)
    judge_run = judge_sub.add_parser("run", help="Score one report with the public API judge panel")
    query = judge_run.add_mutually_exclusive_group(required=True)
    query.add_argument("--query", default="", help="Research query text")
    query.add_argument("--query-file", default="", help="File containing the research query")
    judge_run.add_argument(
        "--report-file", required=True, help="Report text/Markdown file to evaluate"
    )
    judge_run.add_argument(
        "--criteria-file", required=True, help="Criteria file: JSON, JSONL, or text"
    )
    judge_run.add_argument(
        "--panel",
        choices=["paper-a-api", "openai-only"],
        default="paper-a-api",
        help="Judge panel. `paper-a-api` uses OpenAI plus Anthropic API Claude judges.",
    )
    judge_run.add_argument("--out", default="", help="Optional JSON output path")
    judge_run.add_argument(
        "--dry-run", action="store_true", help="Validate files and show provider plan only"
    )
    judge_run.set_defaults(func=cmd_judge_run)

    paper = sub.add_parser("paper", help="Build and verify paper artifacts")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    rebuild = paper_sub.add_parser(
        "rebuild",
        help="Rebuild Paper A tables, figures, and optionally the PDF",
    )
    rebuild.add_argument("paper", choices=["paper-a"], help="Paper to rebuild")
    rebuild.add_argument(
        "--check-only",
        action="store_true",
        help="Verify public paper inputs and generated assets without running scripts",
    )
    rebuild.add_argument("--skip-tables", action="store_true", help="Do not regenerate tables")
    rebuild.add_argument("--skip-figures", action="store_true", help="Do not regenerate figures")
    rebuild.add_argument(
        "--skip-compile",
        action="store_true",
        help="Do not compile the final PDF with tectonic",
    )
    rebuild.set_defaults(func=cmd_paper_rebuild)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
