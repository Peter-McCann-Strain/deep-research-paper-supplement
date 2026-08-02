"""Public settings tests for standard OpenAI and Azure configurations."""

from __future__ import annotations

from deep_research.settings import load_public_settings


def test_settings_load_public_api_env(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "OPENAI_API_KEY": "test-openai",
            "ANTHROPIC_API_KEY": "test-anthropic",
            "OPENAI_JUDGE_MODEL": "judge-model",
        },
    )

    assert settings.has_openai is True
    assert settings.has_anthropic is True
    assert settings.openai.judge_model == "judge-model"
    assert settings.paths.project_root == tmp_path.resolve()


def test_settings_azure_requires_endpoint_and_deployment(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
        },
    )

    assert settings.has_openai is False
    assert settings.openai.azure_api_version == "v1"
    assert "AZURE_OPENAI_ENDPOINT" in settings.openai.missing_for_judging()
    assert "AZURE_OPENAI_DEPLOYMENT" in settings.openai.missing_for_generation()


def test_azure_v1_base_url_normalizes_resource_endpoint(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        },
    )

    assert settings.openai.azure_v1_base_url == "https://example.openai.azure.com/openai/v1/"
