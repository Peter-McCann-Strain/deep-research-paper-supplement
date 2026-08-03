"""Offline first-run and provenance checks."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from deep_research.cli import build_parser
from deep_research.reproduce import run_provenance_check
from deep_research.settings import load_public_settings


def test_quickstart_check_is_offline_and_reports_core_status(capsys):
    args = build_parser().parse_args(["quickstart-check"])

    exit_code = args.func(args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["offline_ok"] is True
    assert payload["api_calls_made"] is False
    assert payload["smoke"]["status"] == "success"
    assert payload["reference"]["reference_pattern_count"] == 13
    assert payload["provenance"]["status"] == "success"
    assert payload["compare"]["status"] == "success"
    assert "doctor --verify-api" in payload["optional_paid_next_steps"][0]


def test_provenance_check_detects_modified_reference_file(tmp_path):
    for rel in (
        "data/eval_queries_v2.json",
        "repro/reference/paper_a_headline_numbers.json",
        "repro/reference/paper_a_pattern_metrics.csv",
        "repro/reference/REFERENCE_MANIFEST.json",
    ):
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(rel), destination)
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(project_root=tmp_path, env={})

    assert run_provenance_check(settings).status == "success"

    headline = tmp_path / "repro/reference/paper_a_headline_numbers.json"
    payload = json.loads(headline.read_text())
    payload["pattern_count"] = 999
    headline.write_text(json.dumps(payload))

    report = run_provenance_check(settings)

    assert report.status == "error"
    assert report.details["files"][1]["matches"] is False


def test_expected_outputs_commands_parse():
    doc = Path("repro/EXPECTED_OUTPUTS.md").read_text()
    commands = [
        "deep-research quickstart-check",
        "deep-research doctor",
        "deep-research reproduce paper-a --mode smoke",
        "deep-research reproduce paper-a --mode reference",
        "deep-research reproduce paper-a --mode provenance",
        "deep-research paper rebuild paper-a --check-only",
        "deep-research paper rebuild paper-a --skip-compile",
        "deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv",
    ]
    parser = build_parser()
    for command in commands:
        assert command in doc
        parser.parse_args(shlex.split(command)[1:])
