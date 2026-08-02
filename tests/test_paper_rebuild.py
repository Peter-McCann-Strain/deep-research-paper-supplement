"""Paper artifact rebuild contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research.cli import build_parser
from deep_research.paper_rebuild import run_paper_rebuild


def test_paper_rebuild_check_only_has_required_public_assets():
    report = run_paper_rebuild(Path.cwd(), check_only=True)

    assert report.status == "success"
    assert report.missing_inputs == []
    assert report.missing_pdf_assets == []
    assert report.output_pdf == "paper_rebuild/paper_a_bounded_returns/main.pdf"


def test_paper_rebuild_cli_check_only(capsys):
    args = build_parser().parse_args(["paper", "rebuild", "paper-a", "--check-only"])

    exit_code = args.func(args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["missing_inputs"] == []
    assert payload["missing_pdf_assets"] == []
