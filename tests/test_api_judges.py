"""Public API judge parsing and request-shape tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research.api_judges import JudgeRequest, OpenAIJudgeProvider, parse_json_response
from deep_research.cli import build_parser
from deep_research.judge_runner import load_criteria, run_judge_file
from deep_research.settings import load_public_settings


def test_parse_json_response_validates_object():
    payload = {
        "evaluations": [
            {
                "criterion_index": 0,
                "verdict": "SATISFIED",
                "evidence": "The report cites a source.",
                "reasoning": "The citation directly supports the claim.",
            }
        ]
    }

    assert parse_json_response(json.dumps(payload), expected_criteria_count=1) == payload


def test_parse_json_response_rejects_empty_evaluations():
    with pytest.raises(ValueError, match="non-empty list"):
        parse_json_response('{"evaluations": []}')


def test_parse_json_response_rejects_missing_or_invalid_verdicts():
    payload = {
        "evaluations": [
            {
                "criterion_index": 0,
                "verdict": "MAYBE",
                "evidence": "Evidence",
                "reasoning": "Reasoning",
            }
        ]
    }

    with pytest.raises(ValueError, match="invalid verdict"):
        parse_json_response(json.dumps(payload), expected_criteria_count=1)


def test_parse_json_response_rejects_non_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_response("not json")


def test_cli_rejects_non_positive_api_limit():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["reproduce", "paper-a", "--mode", "api-best-effort", "--limit", "0"])


def test_load_criteria_accepts_json_object_and_text(tmp_path):
    json_path = tmp_path / "criteria.json"
    json_path.write_text(json.dumps({"criteria": [{"criterion": "cite sources"}, "be concise"]}))
    text_path = tmp_path / "criteria.txt"
    text_path.write_text("# comment\n- cover tradeoffs\n* state limits\n")

    assert load_criteria(json_path) == ["cite sources", "be concise"]
    assert load_criteria(text_path) == ["cover tradeoffs", "state limits"]


def test_judge_run_dry_run_does_not_require_api_or_anthropic_package(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    report_path = tmp_path / "report.md"
    criteria_path = tmp_path / "criteria.json"
    report_path.write_text("The report cites sources and states limitations.")
    criteria_path.write_text(json.dumps(["cites sources", "states limitations"]))
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_judge_file(
        settings,
        query="What happened?",
        report_file=report_path,
        criteria_file=criteria_path,
        dry_run=True,
    )

    assert report.status == "dry-run"
    assert report.criteria_count == 2
    assert "ANTHROPIC_API_KEY" in report.missing_configuration


def test_parse_json_response_accepts_fenced_json():
    payload = {
        "evaluations": [
            {
                "criterion_index": 0,
                "verdict": "NOT_SATISFIED",
                "evidence": "No relevant evidence found.",
                "reasoning": "The report omits the required point.",
            }
        ]
    }
    content = "```json\n" + json.dumps(payload) + "\n```"

    assert parse_json_response(content, expected_criteria_count=1) == payload


class FakeResponsesClient:
    def __init__(self):
        self.calls = []
        self.responses = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            "evaluations": [
                {
                    "criterion_index": 0,
                    "verdict": "SATISFIED",
                    "evidence": "The report cites a source.",
                    "reasoning": "The evidence addresses the criterion.",
                }
            ]
        }
        return SimpleNamespace(
            output_text=json.dumps(payload),
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


def test_openai_judge_provider_uses_responses_api_with_json_schema(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "OPENAI_JUDGE_MODEL": "judge-model"},
    )
    fake_client = FakeResponsesClient()
    provider = OpenAIJudgeProvider(settings.openai, client=fake_client)

    response = asyncio.run(
        provider.evaluate(
            JudgeRequest(
                query="What happened?",
                report="The report cites a source.",
                criteria=["The report cites a source."],
                system_prompt="Return JSON only.",
            )
        )
    )

    call = fake_client.calls[0]
    assert call["model"] == "judge-model"
    assert call["instructions"] == "Return JSON only."
    assert "Criteria to Evaluate" in call["input"]
    schema_format = call["text"]["format"]
    schema = schema_format["schema"]
    evaluation_schema = schema["properties"]["evaluations"]["items"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["strict"] is True
    assert schema["additionalProperties"] is False
    assert evaluation_schema["additionalProperties"] is False
    assert evaluation_schema["required"] == ["criterion_index", "verdict", "evidence", "reasoning"]
    assert "messages" not in call
    assert response.parsed["evaluations"][0]["verdict"] == "SATISFIED"
    assert response.input_tokens == 11
    assert response.output_tokens == 7


def test_judge_run_records_provider_failure(monkeypatch, tmp_path):
    from deep_research import judge_runner

    class FailingProvider:
        label = "test-provider"
        model = "test-model"
        provider_mode = "test"
        configured_model = "test-model"
        call_model_or_deployment = "test-model"

        async def evaluate(self, request):
            raise RuntimeError("planned failure")

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    report_path = tmp_path / "report.md"
    criteria_path = tmp_path / "criteria.json"
    out_path = tmp_path / "judge.json"
    report_path.write_text("Report text")
    criteria_path.write_text(json.dumps(["criterion"]))
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )
    monkeypatch.setattr(
        judge_runner, "_build_providers", lambda settings, panel: [FailingProvider()]
    )

    report = run_judge_file(
        settings,
        query="Question?",
        report_file=report_path,
        criteria_file=criteria_path,
        panel="openai-only",
        output_path=out_path,
    )

    assert report.status == "failed"
    assert report.results[0]["status"] == "failed"
    assert report.results[0]["error_type"] == "RuntimeError"
    assert out_path.exists()


def test_documented_judge_dry_run_example_loads_without_api_credentials(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_judge_file(
        settings,
        query="Example research question",
        report_file=Path("repro/examples/example_report.md"),
        criteria_file=Path("data/public_judge_criteria.json"),
        dry_run=True,
    )

    assert report.status == "dry-run"
    assert report.criteria_count > 0
    assert "ANTHROPIC_API_KEY" in report.missing_configuration


class FakeTimeout:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class CapturingAsyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = SimpleNamespace()


def _install_fake_openai_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Timeout=FakeTimeout))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=CapturingAsyncOpenAI, AsyncAzureOpenAI=CapturingAsyncOpenAI),
    )


def test_azure_openai_judge_requires_v1_responses_api(monkeypatch, tmp_path):
    _install_fake_openai_modules(monkeypatch)
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
            "AZURE_OPENAI_DEPLOYMENT": "judge-deployment",
        },
    )

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION=v1"):
        OpenAIJudgeProvider(settings.openai)


def test_azure_v1_openai_judge_client_sets_api_version_default_query(monkeypatch, tmp_path):
    _install_fake_openai_modules(monkeypatch)
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "generation-deployment",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT": "judge-deployment",
        },
    )

    provider = OpenAIJudgeProvider(settings.openai)

    assert isinstance(provider._client, CapturingAsyncOpenAI)
    assert provider.call_model_or_deployment == "judge-deployment"
    assert provider._client.kwargs["base_url"] == "https://example.openai.azure.com/openai/v1/"
    assert provider._client.kwargs["default_query"] == {"api-version": "v1"}
