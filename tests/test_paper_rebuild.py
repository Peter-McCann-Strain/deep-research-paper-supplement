"""Paper artifact rebuild contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research.cli import build_parser
import deep_research.paper_rebuild as paper_rebuild
from deep_research.paper_rebuild import ScriptResult, run_paper_rebuild


def test_paper_rebuild_check_only_has_required_public_assets():
    report = run_paper_rebuild(Path.cwd(), check_only=True)

    assert report.status == "success"
    assert report.missing_inputs == []
    assert report.missing_pdf_assets == []
    assert report.output_pdf == "paper_rebuild/paper_a_bounded_returns/main.pdf"
    assert report.canonical_store_unchanged is True


def test_paper_rebuild_cli_check_only(capsys):
    args = build_parser().parse_args(["paper", "rebuild", "paper-a", "--check-only"])

    exit_code = args.func(args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["missing_inputs"] == []
    assert payload["missing_pdf_assets"] == []
    assert payload["canonical_store_unchanged"] is True


def test_public_paper_rebuild_fails_if_canonical_store_is_mutated(tmp_path, monkeypatch):
    canonical = tmp_path / paper_rebuild.ANALYSIS_DIR / "canonical_numbers.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"stable": true}\n')

    monkeypatch.setattr(
        paper_rebuild,
        "REQUIRED_INPUTS",
        [paper_rebuild.ANALYSIS_DIR / "canonical_numbers.json"],
    )
    monkeypatch.setattr(paper_rebuild, "REQUIRED_PDF_ASSETS", [])
    monkeypatch.setattr(paper_rebuild, "TABLE_SCRIPTS", ["mutating_builder.py"])
    monkeypatch.setattr(paper_rebuild, "FIGURE_SCRIPTS", [])

    def fake_runner(project_root: Path, script_name: str) -> ScriptResult:
        assert script_name == "mutating_builder.py"
        (project_root / paper_rebuild.ANALYSIS_DIR / "canonical_numbers.json").write_text(
            '{"stable": false}\n'
        )
        return ScriptResult(script="mutating_builder.py", returncode=0, stdout_tail="", stderr_tail="")

    monkeypatch.setattr(paper_rebuild, "_run_python_script", fake_runner)

    report = run_paper_rebuild(tmp_path, compile_pdf=False)

    assert report.status == "failed"
    assert report.message == "paper artifact rebuild mutated canonical_numbers.json"
    assert report.canonical_store_unchanged is False
    assert report.canonical_store_fingerprint_before != report.canonical_store_fingerprint_after
