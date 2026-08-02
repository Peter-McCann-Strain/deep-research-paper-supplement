"""Public settings model for the release-facing command line.

This module is intentionally side-effect free: importing it does not load .env
files or create directories. CLI commands call :func:`load_public_settings`
explicitly at the boundary.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_ANTHROPIC_OPUS_MODEL = "claude-opus-4-8"
DEFAULT_ANTHROPIC_SONNET_MODEL = "claude-sonnet-4-6"
DEFAULT_COST_NOTE = (
    "Budgeting estimates only. Provider pricing, tokenization, retries, and hosted search "
    "charges change over time; override these rates after checking provider pricing pages."
)


@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    paper_dir: Path


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str = ""
    model: str = "gpt-5.6-sol"
    judge_model: str = "gpt-5.6-sol"
    use_azure: bool = False
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "v1"
    azure_deployment: str = ""
    azure_judge_deployment: str = ""

    @property
    def azure_v1_base_url(self) -> str:
        endpoint = self.azure_endpoint.rstrip("/")
        if endpoint.endswith("/openai/v1"):
            return endpoint + "/"
        if endpoint.endswith("/openai"):
            return endpoint + "/v1/"
        return endpoint + "/openai/v1/"

    @property
    def generation_call_model(self) -> str:
        if self.use_azure:
            return self.azure_deployment
        return self.model

    @property
    def judge_call_model(self) -> str:
        if self.use_azure:
            return self.azure_judge_deployment or self.azure_deployment
        return self.judge_model

    def missing_for_generation(self) -> list[str]:
        if self.use_azure:
            missing = []
            if not self.azure_api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not self.azure_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not self.azure_deployment:
                missing.append("AZURE_OPENAI_DEPLOYMENT")
            return missing
        return [] if self.api_key else ["OPENAI_API_KEY"]

    def missing_for_judging(self) -> list[str]:
        if self.use_azure:
            missing = []
            if not self.azure_api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not self.azure_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not (self.azure_judge_deployment or self.azure_deployment):
                missing.append("AZURE_OPENAI_JUDGE_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT")
            return missing
        return [] if self.api_key else ["OPENAI_API_KEY"]


@dataclass(frozen=True)
class AnthropicSettings:
    api_key: str = ""
    opus_model: str = DEFAULT_ANTHROPIC_OPUS_MODEL
    sonnet_model: str = DEFAULT_ANTHROPIC_SONNET_MODEL


@dataclass(frozen=True)
class SearchSettings:
    backend: str = "openai"
    semantic_scholar_api_key: str = ""
    openai_web_search_tool: str = "web_search"


@dataclass(frozen=True)
class CostSettings:
    openai_generation_usd_per_call: float = 0.20
    openai_web_search_usd_per_call: float = 0.01
    openai_judge_usd_per_call: float = 0.05
    anthropic_opus_judge_usd_per_call: float = 0.30
    anthropic_sonnet_judge_usd_per_call: float = 0.10
    note: str = DEFAULT_COST_NOTE


@dataclass(frozen=True)
class PublicSettings:
    paths: PathSettings
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = field(default_factory=AnthropicSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    cost: CostSettings = field(default_factory=CostSettings)

    @property
    def has_openai(self) -> bool:
        return not self.openai.missing_for_judging()

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic.api_key)


def discover_project_root(start: Path | None = None) -> Path:
    """Find the repository root without assuming a specific checkout path."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "deep_research").is_dir() and (
            (candidate / "pyproject.toml").exists() or (candidate / ".git").exists()
        ):
            return candidate
    return current


def _merged_env(project_root: Path, env: Mapping[str, str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = project_root / ".env"
    if env_path.exists():
        values.update({k: v or "" for k, v in dotenv_values(env_path).items()})
    values.update(dict(os.environ if env is None else env))
    return values


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(values: Mapping[str, str], key: str, default: float) -> float:
    value = values.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def load_public_settings(
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> PublicSettings:
    """Load release-facing settings from .env and process environment."""
    root = discover_project_root(project_root)
    values = _merged_env(root, env)

    paths = PathSettings(
        project_root=root,
        data_dir=root / values.get("DR_DATA_DIR", "data"),
        artifacts_dir=root / values.get("DR_ARTIFACTS_DIR", "artifacts"),
        paper_dir=root / values.get("DR_PAPER_DIR", "papers/paper_a_bounded_returns"),
    )

    openai = OpenAISettings(
        api_key=values.get("OPENAI_API_KEY", ""),
        model=values.get("OPENAI_MODEL", "gpt-5.6-sol"),
        judge_model=values.get("OPENAI_JUDGE_MODEL", values.get("JUDGE_MODEL", "gpt-5.6-sol")),
        use_azure=_bool(values.get("USE_AZURE_OPENAI", "false")),
        azure_endpoint=values.get("AZURE_OPENAI_ENDPOINT", ""),
        azure_api_key=values.get("AZURE_OPENAI_API_KEY", ""),
        azure_api_version=values.get("AZURE_OPENAI_API_VERSION", "v1"),
        azure_deployment=values.get("AZURE_OPENAI_DEPLOYMENT", ""),
        azure_judge_deployment=values.get("AZURE_OPENAI_JUDGE_DEPLOYMENT", ""),
    )

    anthropic = AnthropicSettings(
        api_key=values.get("ANTHROPIC_API_KEY", ""),
        opus_model=values.get("ANTHROPIC_OPUS_MODEL", DEFAULT_ANTHROPIC_OPUS_MODEL),
        sonnet_model=values.get("ANTHROPIC_SONNET_MODEL", DEFAULT_ANTHROPIC_SONNET_MODEL),
    )

    search = SearchSettings(
        backend=values.get("SEARCH_BACKEND", "openai"),
        semantic_scholar_api_key=values.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        openai_web_search_tool=values.get("OPENAI_WEB_SEARCH_TOOL", "web_search"),
    )

    cost = CostSettings(
        openai_generation_usd_per_call=_float(
            values, "DR_COST_OPENAI_GENERATION_USD_PER_CALL", 0.20
        ),
        openai_web_search_usd_per_call=_float(
            values, "DR_COST_OPENAI_WEB_SEARCH_USD_PER_CALL", 0.01
        ),
        openai_judge_usd_per_call=_float(values, "DR_COST_OPENAI_JUDGE_USD_PER_CALL", 0.05),
        anthropic_opus_judge_usd_per_call=_float(
            values,
            "DR_COST_ANTHROPIC_OPUS_JUDGE_USD_PER_CALL",
            0.30,
        ),
        anthropic_sonnet_judge_usd_per_call=_float(
            values,
            "DR_COST_ANTHROPIC_SONNET_JUDGE_USD_PER_CALL",
            0.10,
        ),
    )

    return PublicSettings(paths=paths, openai=openai, anthropic=anthropic, search=search, cost=cost)


def ensure_runtime_dirs(settings: PublicSettings) -> None:
    """Create runtime directories explicitly at command execution time."""
    for path in (
        settings.paths.data_dir,
        settings.paths.artifacts_dir,
        settings.paths.artifacts_dir / "reports",
        settings.paths.artifacts_dir / "judges",
        settings.paths.artifacts_dir / "reproduction",
    ):
        path.mkdir(parents=True, exist_ok=True)
