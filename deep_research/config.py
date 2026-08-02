"""Compatibility configuration for reusable research modules.

The supported public CLI reads :mod:`deep_research.settings`.  This module is a
thin compatibility layer for the reusable pattern, tool, benchmark, and legacy
experiment code that historically imported ``deep_research.config`` directly.

Importing this file reads ``.env`` and process environment values, but it does
not create runtime directories and it contains no private endpoint defaults.
Azure users should set deployment names in ``.env``; standard OpenAI users can
leave deployment fields blank and call by model ID.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from deep_research.settings import load_public_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV = {k: v or "" for k, v in dotenv_values(PROJECT_ROOT / ".env").items()}
_SETTINGS = load_public_settings(project_root=PROJECT_ROOT)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, _DOTENV.get(key, default))


def _env_float(key: str, default: float) -> float:
    value = _env(key, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def _env_int(key: str, default: int) -> int:
    value = _env(key, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _env_bool(key: str, default: bool = False) -> bool:
    value = _env(key, "true" if default else "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Provider credentials and endpoints.  These are read from user-controlled
# environment variables only; public source never bakes in a private resource.
OPENAI_API_KEY: str = _SETTINGS.openai.api_key
USE_AZURE_OPENAI: bool = _SETTINGS.openai.use_azure

AZURE_OPENAI_API_KEY: str = _SETTINGS.openai.azure_api_key
AZURE_OPENAI_ENDPOINT: str = _SETTINGS.openai.azure_endpoint
AZURE_OPENAI_API_VERSION: str = _SETTINGS.openai.azure_api_version
AZURE_OPENAI_DEPLOYMENT: str = _SETTINGS.openai.azure_deployment
AZURE_OPENAI_JUDGE_DEPLOYMENT: str = _SETTINGS.openai.azure_judge_deployment

SEARCH_OPENAI_API_KEY: str = _env(
    "SEARCH_OPENAI_API_KEY",
    OPENAI_API_KEY or AZURE_OPENAI_API_KEY,
)
SEARCH_OPENAI_ENDPOINT: str = _env("SEARCH_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT)

JUDGE_OPENAI_API_KEY: str = _env(
    "JUDGE_OPENAI_API_KEY",
    OPENAI_API_KEY or SEARCH_OPENAI_API_KEY,
)
JUDGE_OPENAI_ENDPOINT: str = _env(
    "JUDGE_OPENAI_ENDPOINT",
    SEARCH_OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT,
)
JUDGE_MODEL: str = _env("JUDGE_MODEL", _SETTINGS.openai.judge_call_model)

ANTHROPIC_API_KEY: str = _SETTINGS.anthropic.api_key
TAVILY_API_KEY: str = _env("TAVILY_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY: str = _SETTINGS.search.semantic_scholar_api_key


@dataclass(frozen=True)
class ModelSpec:
    deployment: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context: int = 128_000
    supports_json: bool = True
    use_max_completion_tokens: bool = False


DEFAULT_MODEL: str = _env("DEFAULT_MODEL", _SETTINGS.openai.generation_call_model)
SEARCH_MODEL: str = _env("SEARCH_MODEL", "gpt-4o-mini")
SEARCH_BACKEND: str = _env("SEARCH_BACKEND", _SETTINGS.search.backend)


def _deployment(model: str, specific_env: str, *, fallback: str | None = None) -> str:
    return _env(specific_env, _env("AZURE_OPENAI_DEPLOYMENT", fallback or model))


MODELS: dict[str, ModelSpec] = {
    "gpt-5.6-sol": ModelSpec(
        deployment=_env("AZURE_OPENAI_GPT56_SOL_DEPLOYMENT", AZURE_OPENAI_DEPLOYMENT or "gpt-5.6-sol"),
        cost_per_1k_input=_env_float("DR_COST_GPT56_SOL_INPUT_PER_1K", 0.003),
        cost_per_1k_output=_env_float("DR_COST_GPT56_SOL_OUTPUT_PER_1K", 0.012),
        use_max_completion_tokens=True,
    ),
    "gpt-5.2": ModelSpec(
        deployment=_env("AZURE_OPENAI_GPT52_DEPLOYMENT", AZURE_OPENAI_JUDGE_DEPLOYMENT or "gpt-5.2"),
        cost_per_1k_input=_env_float("DR_COST_GPT52_INPUT_PER_1K", 0.003),
        cost_per_1k_output=_env_float("DR_COST_GPT52_OUTPUT_PER_1K", 0.012),
        use_max_completion_tokens=True,
    ),
    "gpt-4o": ModelSpec(
        deployment=_deployment("gpt-4o", "AZURE_OPENAI_GPT4O_DEPLOYMENT"),
        cost_per_1k_input=_env_float("DR_COST_GPT4O_INPUT_PER_1K", 0.005),
        cost_per_1k_output=_env_float("DR_COST_GPT4O_OUTPUT_PER_1K", 0.015),
        use_max_completion_tokens=True,
    ),
    "gpt-4o-mini": ModelSpec(
        deployment=_deployment("gpt-4o-mini", "AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT"),
        cost_per_1k_input=_env_float("DR_COST_GPT4O_MINI_INPUT_PER_1K", 0.00015),
        cost_per_1k_output=_env_float("DR_COST_GPT4O_MINI_OUTPUT_PER_1K", 0.0006),
    ),
    "gpt-4.1": ModelSpec(
        deployment=_deployment("gpt-4.1", "AZURE_OPENAI_GPT41_DEPLOYMENT"),
        cost_per_1k_input=_env_float("DR_COST_GPT41_INPUT_PER_1K", 0.002),
        cost_per_1k_output=_env_float("DR_COST_GPT41_OUTPUT_PER_1K", 0.008),
    ),
    "Qwen2.5-7B-Instruct": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
    "DeepResearcher-7b": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
    "DR-Judge-7B": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
}

if DEFAULT_MODEL and DEFAULT_MODEL not in MODELS:
    MODELS[DEFAULT_MODEL] = ModelSpec(
        deployment=AZURE_OPENAI_DEPLOYMENT or DEFAULT_MODEL,
        cost_per_1k_input=_SETTINGS.cost.openai_generation_usd_per_call,
        cost_per_1k_output=_SETTINGS.cost.openai_generation_usd_per_call,
        use_max_completion_tokens=True,
    )
if JUDGE_MODEL and JUDGE_MODEL not in MODELS:
    MODELS[JUDGE_MODEL] = ModelSpec(
        deployment=AZURE_OPENAI_JUDGE_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT or JUDGE_MODEL,
        cost_per_1k_input=_SETTINGS.cost.openai_judge_usd_per_call,
        cost_per_1k_output=_SETTINGS.cost.openai_judge_usd_per_call,
        use_max_completion_tokens=True,
    )


MAX_COST_PER_RUN: float = _env_float("MAX_COST_PER_RUN", 2.00)
MAX_COST_EVAL_RUN: float = _env_float("MAX_COST_EVAL_RUN", 10.00)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = _env_int("DR_RETRY_MAX_RETRIES", 20)
    base_delay: float = _env_float("DR_RETRY_BASE_DELAY", 0.5)
    conn_base_delay: float = _env_float("DR_RETRY_CONN_BASE_DELAY", 1.0)
    max_delay: float = _env_float("DR_RETRY_MAX_DELAY", 60.0)
    jitter_max: float = _env_float("DR_RETRY_JITTER_MAX", 2.0)


@dataclass(frozen=True)
class TimeoutConfig:
    connect: float = _env_float("DR_TIMEOUT_CONNECT", 30.0)
    read: float = _env_float("DR_TIMEOUT_READ", 300.0)
    write: float = _env_float("DR_TIMEOUT_WRITE", 60.0)
    pool: float = _env_float("DR_TIMEOUT_POOL", 30.0)


@dataclass(frozen=True)
class ConnectionPoolConfig:
    max_connections: int = _env_int("DR_POOL_MAX_CONNECTIONS", 50)
    max_keepalive_connections: int = _env_int("DR_POOL_MAX_KEEPALIVE_CONNECTIONS", 20)
    keepalive_expiry: float = _env_float("DR_POOL_KEEPALIVE_EXPIRY", 15.0)


@dataclass(frozen=True)
class RateLimitConfig:
    max_concurrent: int = _env_int("DR_RATE_LIMIT_MAX_CONCURRENT", 12)
    rpm: int = _env_int("DR_RATE_LIMIT_RPM", 200)


@dataclass(frozen=True)
class JudgeDefaults:
    max_concurrent: int = _env_int("DR_JUDGE_MAX_CONCURRENT", 3)
    max_tokens: int = _env_int("DR_JUDGE_MAX_TOKENS", 8192)
    temperature: float = _env_float("DR_JUDGE_TEMPERATURE", 0.1)
    seed: int = _env_int("DR_JUDGE_SEED", 42)
    sdk_max_retries: int = _env_int("DR_JUDGE_SDK_MAX_RETRIES", 0)
    read_timeout: float = _env_float("DR_JUDGE_READ_TIMEOUT", 600.0)


@dataclass(frozen=True)
class EvalPipelineDefaults:
    max_concurrent_runs: int = _env_int("DR_EVAL_MAX_CONCURRENT_RUNS", 2)
    passes_per_judge: int = _env_int("DR_EVAL_PASSES_PER_JUDGE", 3)
    bootstrap_resamples: int = _env_int("DR_EVAL_BOOTSTRAP_RESAMPLES", 10_000)
    statistical_alpha: float = _env_float("DR_EVAL_STATISTICAL_ALPHA", 0.05)
    report_truncation_words: int = _env_int("DR_EVAL_REPORT_TRUNCATION_WORDS", 12_000)
    default_n_repeats: int = _env_int("DR_EVAL_DEFAULT_N_REPEATS", 3)
    default_random_seed: int = _env_int("DR_EVAL_DEFAULT_RANDOM_SEED", 42)


RETRY = RetryConfig()
TIMEOUTS = TimeoutConfig()
POOL = ConnectionPoolConfig()
RATE_LIMIT = RateLimitConfig()
JUDGE = JudgeDefaults()
EVAL_PIPELINE = EvalPipelineDefaults()

DATA_DIR = _SETTINGS.paths.data_dir
ARTIFACTS_DIR = _SETTINGS.paths.artifacts_dir
PAPERS_DIR = PROJECT_ROOT / _env("DR_PAPERS_DIR", "papers")
PAPER_DIR = _SETTINGS.paths.paper_dir
DOCS_DIR = PROJECT_ROOT / _env("DR_DOCS_DIR", "docs")

CHECKPOINTS_DIR = PROJECT_ROOT / _env("DR_CHECKPOINTS_DIR", "checkpoints")
REPORTS_DIR = PROJECT_ROOT / _env("DR_REPORTS_DIR", "reports")
RESULTS_DIR = PROJECT_ROOT / _env("DR_RESULTS_DIR", "results")

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments" / "canonical"
JUDGES_DIR = ARTIFACTS_DIR / "judges"
PHASE_REPORTS_DIR = ARTIFACTS_DIR / "phase_reports"
REPLICATION_DIR = ARTIFACTS_DIR / "replication"
CACHE_DIR = ARTIFACTS_DIR / "caches"
MODELS_DIR = ARTIFACTS_DIR / "models"

RUNTIME_DIRS = (
    DATA_DIR,
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    EXPERIMENTS_DIR,
    JUDGES_DIR,
    PHASE_REPORTS_DIR,
    REPLICATION_DIR,
    CACHE_DIR,
)


def ensure_runtime_dirs() -> None:
    """Create runtime directories explicitly at command/run boundaries."""
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def get_environment_metadata() -> dict[str, str]:
    """Capture non-secret environment metadata for reproducibility records."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "default_model": DEFAULT_MODEL,
        "judge_model": JUDGE_MODEL,
        "search_model": SEARCH_MODEL,
        "search_backend": SEARCH_BACKEND,
        "use_azure_openai": str(USE_AZURE_OPENAI),
    }
    for pkg_name in ["openai", "httpx", "numpy", "scipy", "structlog"]:
        try:
            mod = __import__(pkg_name)
        except ImportError:
            continue
        env[f"{pkg_name}_version"] = str(getattr(mod, "__version__", "unknown"))
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = "unavailable"
    for name, spec in MODELS.items():
        env[f"deployment_{name}"] = spec.deployment
    return env

