"""Release audit, export, and packaging contract tests."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from deep_research.public_export import export_public_tree
from deep_research.release_audit import audit_release_tree


def test_release_audit_flags_private_files(tmp_path):
    (tmp_path / ".env").write_text("PRIVATE_CONFIG=1\n")
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "deep_research" / "__init__.py").write_text("")

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    messages = [finding.message for finding in result.findings]
    assert any("forbidden public-release path" in message for message in messages)


def test_release_audit_allows_basic_source_tree(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "deep_research" / "__init__.py").write_text("input_tokens = 42\n")
    (tmp_path / "README.md").write_text("# Public\n")

    result = audit_release_tree(tmp_path, enforce_manifest=False)

    assert result.ok is True


def test_release_audit_fails_closed_without_manifest(tmp_path):
    (tmp_path / "README.md").write_text("# Public\n")

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any("manifest not found" in finding.message for finding in result.findings)


def test_release_audit_enforces_manifest_allowlist(tmp_path):
    (tmp_path / "README.md").write_text("# Public\n")
    (tmp_path / "private_notes.md").write_text("do not ship\n")
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["README.md", "PUBLIC_MANIFEST.json"],
                "exclude_globs": [],
                "required_paths": ["README.md"],
            }
        )
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any(finding.path == "private_notes.md" for finding in result.findings)


def test_release_audit_scans_binary_local_markers(tmp_path):
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["PUBLIC_MANIFEST.json", "paper.pdf"],
                "exclude_globs": [],
                "required_paths": ["paper.pdf"],
            }
        )
    )
    (tmp_path / "paper.pdf").write_bytes(
        b"%PDF-1.7\n/Producer (" + b"/home/" + b"researcher/private/main.tex)"
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any(finding.path == "paper.pdf" for finding in result.findings)


def test_release_audit_ignores_env_template_placeholders(tmp_path):
    openai_key_name = "OPENAI_" + "API_KEY"
    anthropic_key_name = "ANTHROPIC_" + "API_KEY"
    (tmp_path / ".env.example").write_text(f"{openai_key_name}=\n{anthropic_key_name}=<your-key>\n")

    result = audit_release_tree(tmp_path, enforce_manifest=False)

    assert result.ok is True


def test_release_audit_flags_vcs_metadata(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")

    result = audit_release_tree(tmp_path, enforce_manifest=False)

    assert result.ok is False
    assert any(finding.path == ".git/config" for finding in result.findings)


def test_release_audit_requires_catalog_for_shipped_scripts(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "uncataloged.py").write_text("print('x')\n")
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["PUBLIC_MANIFEST.json", "scripts/*.py"],
                "exclude_globs": [],
                "required_paths": ["PUBLIC_MANIFEST.json"],
            }
        )
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any("script catalog missing" in finding.message for finding in result.findings)


def test_release_audit_flags_uncataloged_script_rows(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "covered.py").write_text("print('x')\n")
    (scripts / "missing.py").write_text("print('y')\n")
    repro = tmp_path / "repro"
    repro.mkdir()
    (repro / "SCRIPT_CATALOG.csv").write_text(
        "script,family,public_status,required_inputs_or_services,expected_outputs,summary\n"
        "covered.py,validation,supported public helper,public files,console report,covered helper\n"
    )
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["PUBLIC_MANIFEST.json", "scripts/*.py", "repro/*.csv"],
                "exclude_globs": [],
                "required_paths": ["PUBLIC_MANIFEST.json"],
            }
        )
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any(finding.path == "scripts/missing.py" for finding in result.findings)


def test_release_audit_flags_redaction_leak_phrases(tmp_path):
    (tmp_path / "README.md").write_text("mentions a " + "private" + " planning " + "board\n")

    result = audit_release_tree(tmp_path, enforce_manifest=False)

    assert result.ok is False
    assert any("private/local marker" in finding.message for finding in result.findings)


def test_public_export_copies_allowlist_and_audits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "deep_research").mkdir()
    (source / "deep_research" / "__init__.py").write_text("")
    (source / "README.md").write_text("# Public\n")
    (source / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "max_file_mb": 1,
                "required_paths": ["README.md", "PUBLIC_MANIFEST.json"],
                "include_globs": [
                    "README.md",
                    "PUBLIC_MANIFEST.json",
                    "PUBLIC_EXPORT_REPORT.json",
                    "deep_research/**/*.py",
                ],
                "exclude_globs": ["private/**"],
            }
        )
    )
    (source / "private").mkdir()
    (source / "private" / "notes.md").write_text("secret notes\n")
    output = tmp_path / "export"

    result = export_public_tree(source, output)

    assert result.ok is True
    assert (output / "README.md").exists()
    report = json.loads((output / "PUBLIC_EXPORT_REPORT.json").read_text())
    assert report["manifest_sha256"]
    assert report["file_sha256"]["README.md"]
    assert report["source_git"]["commit"]
    assert report["artifact_file_count"] == report["file_count"] + 1
    assert "PUBLIC_EXPORT_REPORT.json" in report["files_in_artifact"]
    assert not (output / "private" / "notes.md").exists()


def test_notice_and_pattern_metrics_are_manifested():
    manifest = json.loads(Path("PUBLIC_MANIFEST.json").read_text())

    assert Path("NOTICE").exists()
    assert "NOTICE" in manifest["required_paths"]
    assert "deep_research/config.py" in manifest["required_paths"]
    assert "deep_research/types.py" in manifest["required_paths"]
    assert "repro/reference/paper_a_pattern_metrics.csv" in manifest["required_paths"]
    assert "repro/PAPER_A_ARTIFACT_INDEX.md" in manifest["required_paths"]
    assert "repro/SCRIPT_CATALOG.csv" in manifest["required_paths"]
    assert "repro/SCRIPT_CATALOG.md" in manifest["required_paths"]
    assert "Apache-2.0 applies to code" in Path("README.md").read_text()
    assert "mixed-license" in Path("NOTICE").read_text()


def test_public_dependency_profile_uses_api_and_paper_without_local_model_stack():
    project = tomllib.loads(Path("pyproject.toml").read_text())

    base_deps = project["project"]["dependencies"]
    assert "python-dotenv>=1.0.0" in base_deps
    assert any(dep.startswith("pydantic") for dep in base_deps)
    assert any(dep.startswith("structlog") for dep in base_deps)
    api_extra = project["project"]["optional-dependencies"]["api"]
    assert any(dep.startswith("openai") for dep in api_extra)
    assert any(dep.startswith("anthropic") for dep in api_extra)
    assert any(dep.startswith("aiolimiter") for dep in api_extra)
    assert any(dep.startswith("trafilatura") for dep in api_extra)
    assert any(dep.startswith("tavily-python") for dep in api_extra)
    requirements = Path("requirements.txt").read_text()
    assert ".[api,paper]" in requirements
    assert ("tor" + "ch") not in Path("constraints-public.txt").read_text().lower()
    assert "tavily" not in Path("deep_research/cli.py").read_text()
