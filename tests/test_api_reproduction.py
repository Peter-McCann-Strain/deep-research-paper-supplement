"""No-download API reproduction planning, execution, and entitlement tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from deep_research.reproduce import (
    _query_file_stem,
    estimate_api_reproduction_cost,
    plan_api_reproduction,
    run_api_reproduction,
    verify_api_entitlements,
)
from deep_research.settings import load_public_settings


def test_plan_api_reproduction_uses_public_queries(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "Question 1?"},
                    {"id": "q2", "query": "Question 2?"},
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )

    report = plan_api_reproduction(settings, limit=1)

    assert report.status == "ready"
    assert report.details["query_count"] == 1
    assert report.details["judge_requested"] is False
    assert "--execute" in report.details["execute_command"]
    assert "--judge" not in report.details["execute_command"]


def test_plan_api_reproduction_blocks_for_judge_without_anthropic(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )

    report = plan_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "blocked"
    assert report.details["judge_requested"] is True
    assert "--judge" in report.details["execute_command"]
    assert "ANTHROPIC_API_KEY" in report.message


def test_cost_estimate_includes_full_judge_components(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "Question 1?"},
                    {"id": "q2", "query": "Question 2?"},
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "OPENAI_API_KEY": "test-openai",
            "ANTHROPIC_API_KEY": "test-anthropic",
            "DR_COST_OPENAI_GENERATION_USD_PER_CALL": "1",
            "DR_COST_OPENAI_WEB_SEARCH_USD_PER_CALL": "0.5",
            "DR_COST_OPENAI_JUDGE_USD_PER_CALL": "0.25",
            "DR_COST_ANTHROPIC_OPUS_JUDGE_USD_PER_CALL": "2",
            "DR_COST_ANTHROPIC_SONNET_JUDGE_USD_PER_CALL": "3",
        },
    )

    estimate = estimate_api_reproduction_cost(settings, full=True, judge=True)

    assert estimate["generation_calls"] == 2
    assert estimate["web_search_tool_calls_estimated"] == 2
    assert estimate["judge_calls"] == {
        "openai": 2,
        "anthropic_opus": 2,
        "anthropic_sonnet": 2,
    }
    assert {component["name"] for component in estimate["components"]} == {
        "openai_generation_responses",
        "openai_web_search_tool",
        "openai_judge",
        "anthropic_opus_judge",
        "anthropic_sonnet_judge",
    }
    assert estimate["estimated_total_usd"] == 13.5


def test_plan_api_reproduction_rejects_non_positive_limit(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(project_root=tmp_path, env={"OPENAI_API_KEY": "test-openai"})

    try:
        plan_api_reproduction(settings, limit=0)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plan_api_reproduction_blocks_when_cost_exceeds_limit(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(project_root=tmp_path, env={"OPENAI_API_KEY": "test-openai"})

    report = plan_api_reproduction(settings, limit=1, max_cost_usd=0.01)

    assert report.status == "blocked"
    assert "estimated cost" in report.message
    assert report.details["cost_guardrail_ok"] is False


def test_run_api_reproduction_blocks_for_judge_without_anthropic(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(project_root=tmp_path, env={"OPENAI_API_KEY": "test-openai"})

    report = run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "blocked"
    assert "ANTHROPIC_API_KEY" in report.message
    assert report.details["judge_requested"] is True


def test_run_api_reproduction_blocks_without_generation_key(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(json.dumps({"queries": []}))
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_api_reproduction(settings, limit=1)

    assert report.status == "blocked"
    assert "OPENAI_API_KEY" in report.message


def test_run_api_reproduction_execute_judge_failure_is_partial(monkeypatch, tmp_path):
    from deep_research import reproduce
    from deep_research.judge_runner import JudgeRunReport

    class FakeUsage:
        input_tokens = 10
        output_tokens = 20
        total_tokens = 30

    class FakeOutputItem:
        type = "web_search_call"

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = FakeUsage()

        def __init__(self):
            self.output = [FakeOutputItem()]

    class FakeResponses:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    def fake_judge_file(*args, **kwargs):
        return JudgeRunReport(
            status="failed",
            panel="paper-a-api",
            created_utc="2026-07-31T00:00:00+00:00",
            query=kwargs["query"],
            report_file=str(kwargs["report_file"]),
            criteria_file=str(kwargs["criteria_file"]),
            output_path=str(kwargs["output_path"]),
            providers=[],
            criteria_count=1,
            missing_configuration=[],
            results=[{"status": "failed", "error_type": "RuntimeError"}],
        )

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    (data_dir / "public_judge_criteria.json").write_text(json.dumps(["criterion"]))
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )
    monkeypatch.setattr(reproduce, "run_judge_file", fake_judge_file)

    report = reproduce.run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "partial"
    assert report.details["successful_generations"] == 1
    assert report.details["successful_judges"] == 0
    assert report.details["failed_or_partial_judges"] == 1
    assert report.details["actual_usage_summary"]["judge_tokens"]["provider_call_count"] == 0
    assert report.details["actual_usage_summary"]["judge_tokens"]["failed_result_records"] == 1


def test_run_api_reproduction_execute_success_with_fake_openai(monkeypatch, tmp_path):
    from deep_research import reproduce

    calls = []

    class FakeUsage:
        input_tokens = 10
        output_tokens = 20
        total_tokens = 30

    class FakeOutputItem:
        type = "web_search_call"

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = FakeUsage()

        def __init__(self):
            self.output = [FakeOutputItem()]

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "Question 1?",
                        "rubric": {"criteria": [{"text": "cite sources"}]},
                    }
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )

    report = reproduce.run_api_reproduction(settings, limit=1)

    assert report.status == "success"
    assert report.details["successful_generations"] == 1
    assert calls[0]["model"] == settings.openai.model
    assert calls[0]["tools"] == [{"type": "web_search"}]
    assert calls[0]["tool_choice"] == "required"
    generation = report.details["generation_results"][0]
    assert generation["web_search_required"] is True
    assert generation["web_search_used"] is True
    assert generation["response_output_types"] == ["web_search_call"]
    assert (tmp_path / "artifacts/reproduction/paper_a_api_best_effort/q1.md").exists()


def test_run_api_reproduction_fails_generation_without_web_search_call(monkeypatch, tmp_path):
    from deep_research import reproduce

    calls = []

    class FakeResponse:
        output_text = "# Generated report\n\nUngrounded summary."
        usage = None

        def __init__(self):
            self.output = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )

    report = reproduce.run_api_reproduction(settings, limit=1)

    assert calls[0]["tool_choice"] == "required"
    assert report.status == "failed"
    generation = report.details["generation_results"][0]
    assert generation["web_search_required"] is True
    assert generation["web_search_used"] is False
    assert generation["error_type"] == "ValueError"
    assert "web_search_call" in generation["error_message"]


def test_query_file_stem_is_safe_and_deterministic():
    assert _query_file_stem({"id": "../../bad id?", "query": "Question?"}) == "bad_id"
    first = _query_file_stem({"query": "Question needing hashed fallback?"})
    second = _query_file_stem({"query": "Question needing hashed fallback?"})

    assert first == second
    assert first.startswith("query_")
    assert "/" not in first


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


def test_azure_v1_generation_client_sets_api_version_default_query(monkeypatch, tmp_path):
    from deep_research import repro_generation

    _install_fake_openai_modules(monkeypatch)
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
        },
    )

    client, provider_mode, call_model = repro_generation._openai_client(settings)

    assert isinstance(client, CapturingAsyncOpenAI)
    assert provider_mode == "azure_openai"
    assert call_model == "deployment"
    assert client.kwargs["base_url"] == "https://example.openai.azure.com/openai/v1/"
    assert client.kwargs["default_query"] == {"api-version": "v1"}


def test_plan_api_reproduction_supports_azure_hosted_search_config(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
        },
    )

    report = plan_api_reproduction(settings, limit=1)

    assert report.status == "ready"
    assert report.details["unsupported_configuration"] == []
    assert report.details["openai_provider_mode"] == "azure_openai"
    assert report.details["openai_generation_call_model"] == "deployment"
    assert report.details["azure_api_version"] == "v1"


def test_verify_api_entitlements_reports_missing_configuration_without_network(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(project_root=tmp_path, env={})

    result = verify_api_entitlements(settings)

    assert result["status"] == "blocked"
    assert result["paid_probe"] is True
    assert result["judge_panel_requested"] is False
    assert len(result["checks"]) == 1
    assert all(check["status"] == "blocked" for check in result["checks"])


def test_verify_api_entitlements_fails_when_search_call_missing(monkeypatch, tmp_path):
    from deep_research import reproduce

    class FakeResponse:
        output_text = "OK"
        usage = None

        def __init__(self):
            self.output = []

    class FakeResponses:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    async def fake_openai_judge(*args, **kwargs):
        return {"name": "openai_judge", "status": "success"}

    async def fake_anthropic(*args, **kwargs):
        return {"status": "success"}

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )
    monkeypatch.setattr(reproduce, "_verify_openai_judge_entitlement", fake_openai_judge)
    monkeypatch.setattr(reproduce, "_verify_anthropic_entitlement", fake_anthropic)

    result = reproduce.verify_api_entitlements(settings, judge=True)

    assert result["status"] == "partial"
    assert result["judge_panel_requested"] is True
    openai_check = result["checks"][0]
    assert openai_check["status"] == "failed"
    assert openai_check["error_type"] == "MissingWebSearchCall"
    assert openai_check["response_output_types"] == []


def test_verify_api_entitlements_judge_panel_includes_openai_judge(monkeypatch, tmp_path):
    from deep_research import reproduce

    calls = []

    async def fake_generation(settings):
        calls.append("generation")
        return {"name": "openai_generation_with_hosted_search", "status": "success"}

    async def fake_openai_judge(settings):
        calls.append("openai_judge")
        return {"name": "openai_judge", "status": "success"}

    async def fake_anthropic(settings, *, model, label):
        calls.append(label)
        return {"name": label, "status": "success"}

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(reproduce, "_verify_openai_generation_entitlement", fake_generation)
    monkeypatch.setattr(reproduce, "_verify_openai_judge_entitlement", fake_openai_judge)
    monkeypatch.setattr(reproduce, "_verify_anthropic_entitlement", fake_anthropic)

    result = reproduce.verify_api_entitlements(settings, judge=True)

    assert result["status"] == "success"
    assert calls == [
        "generation",
        "openai_judge",
        "anthropic_opus_judge",
        "anthropic_sonnet_judge",
    ]


def test_run_api_reproduction_judge_uses_query_rubric(monkeypatch, tmp_path):
    from deep_research import reproduce
    from deep_research.judge_runner import JudgeRunReport

    class FakeOutputItem:
        type = "web_search_call"

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = None

        def __init__(self):
            self.output = [FakeOutputItem()]

    class FakeResponses:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    captured = {}

    def fake_judge_file(*args, **kwargs):
        criteria_payload = json.loads(Path(kwargs["criteria_file"]).read_text())
        captured["criteria"] = criteria_payload["criteria"]
        return JudgeRunReport(
            status="success",
            panel="paper-a-api",
            created_utc="2026-07-31T00:00:00+00:00",
            query=kwargs["query"],
            report_file=str(kwargs["report_file"]),
            criteria_file=str(kwargs["criteria_file"]),
            output_path=str(kwargs["output_path"]),
            providers=[],
            criteria_count=len(criteria_payload["criteria"]),
            missing_configuration=[],
            results=[
                {
                    "provider": "openai",
                    "provider_mode": "openai",
                    "model": "test-model",
                    "call_model_or_deployment": "test-model",
                    "status": "success",
                    "parsed": {
                        "evaluations": [
                            {
                                "criterion_index": 0,
                                "verdict": "SATISFIED",
                                "evidence": "Source cited.",
                                "reasoning": "The report cites sources.",
                            },
                            {
                                "criterion_index": 1,
                                "verdict": "NOT_SATISFIED",
                                "evidence": "No limitation section.",
                                "reasoning": "The report omits limitations.",
                            },
                        ]
                    },
                    "input_tokens": 10,
                    "output_tokens": 5,
                }
            ],
        )

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "Question 1?",
                        "rubric": {
                            "dimension_weights": {"factual_accuracy": 0.25, "coverage": 0.75},
                            "criteria": [
                                {
                                    "text": "cite sources",
                                    "dimension": "factual_accuracy",
                                    "weight": 10,
                                },
                                {
                                    "text": "state limitations",
                                    "dimension": "coverage",
                                    "weight": 1,
                                },
                            ]
                        },
                    }
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )
    monkeypatch.setattr(reproduce, "run_judge_file", fake_judge_file)

    report = reproduce.run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "success"
    assert captured["criteria"] == ["cite sources", "state limitations"]
    assert report.details["judge_results"][0]["criteria_source"] == "query_rubric"
    score_summary = report.details["judge_results"][0]["score_summary"]
    assert score_summary["dimension_weights"] == {"coverage": 0.75, "factual_accuracy": 0.25}
    assert score_summary["mean_panel_score"] == 0.25
    provider_score = score_summary["provider_scores"][0]
    assert provider_score["scoring_method"] == "dimension_weighted"
    assert provider_score["criterion_weighted_score"] == 0.9091
    assert report.details["current_api_score_summaries"][0]["successful_provider_scores"] == 1
    assert report.details["actual_usage_summary"]["generation_attempts"] == 1
    assert report.details["actual_usage_summary"]["judge_tokens"]["provider_call_count"] == 1
