"""Release tree audit utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_NAME = "PUBLIC_MANIFEST.json"

SAFE_ENV_TEMPLATES = {".env.example", ".env.template"}

FORBIDDEN_PATH_PATTERNS = [
    ".env",
    ".env.*",
    ".git/**",
    ".claude/**",
    ".cudatk/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.egg-info/**",
    "venv/**",
    ".venv/**",
    "memory-bank/**",
    "scratchpad*/**",
    "artifacts/**",
    "results/**",
    "reports/**",
    "logs/**",
    "models/**",
    "checkpoints/**",
    "external_frameworks/**",
    "archive/**",
    "zenodo_upload/**",
    "analysis/**",
    "papers/drafts/**",
    "papers/archive/**",
    "papers/paper_a_bounded_returns/analysis/**",
    "papers/paper_a_bounded_returns/archive_unused_figures/**",
    "papers/paper_a_bounded_returns/audit_*/**",
    "papers/paper_a_bounded_returns/arxiv_submission*",
    "papers/paper_a_bounded_returns/arxiv_submission*/**",
    "papers/paper_a_bounded_returns/blog/**",
    "papers/paper_a_bounded_returns/docs/**",
    "papers/paper_a_bounded_returns/reports/**",
    "papers/paper_a_bounded_returns/submission_tmlr/**",
    "papers/paper_a_bounded_returns/zenodo_v*/**",
    "papers/paper_a_bounded_returns/personal_website_export/**",
    "papers/paper_a_bounded_returns/public_release/**",
    "papers/paper_a_bounded_returns/*backup*.tex",
    "papers/paper_a_bounded_returns/main.tex",
    "papers/paper_a_bounded_returns/main.txt",
    "papers/paper_a_bounded_returns/*.txt",
    "papers/paper_a_bounded_returns/references*.bib",
    "papers/paper_a_bounded_returns/*outreach*",
    "CODEX_REVIEW_PROMPT.md",
    "REBOOT_STATE_*.md",
    "RESUME_AFTER_REBOOT.sh",
    "*.aux",
    "*.out",
    "*.log",
    "*.blg",
    "*.zip",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "*.gguf",
]

TEXT_SCAN_EXTENSIONS = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LOCAL_OR_PRIVATE_MARKERS = (
    "Deep_" + "Research_Projects",
    "private-" + "planning-" + "board",
    "private" + " planning " + "board",
    "small-" + "business identifier",
    "file" + "://",
)
LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9._-])(?:/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|"
    r"[A-Za-z]:(?:\\+|/+)(?:Users|Documents and Settings)(?:\\+|/+)"
    r"[A-Za-z0-9._ -]+(?:\\+|/+))",
    re.IGNORECASE,
)

PROVIDER_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:OPENAI|ANTHROPIC|AZURE_OPENAI|TAVILY|SEMANTIC_SCHOLAR|HF|"
    r"HUGGINGFACE|WANDB)_[A-Z0-9_]*(?:API_KEY|TOKEN)\b[ \t]*=[ \t]*"
    r"(?P<value>[^\s#]+)"
)
GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|password|secret|token)\b[ \t]*[:=][ \t]*"
    r"(?P<value>['\"]?[A-Za-z0-9][A-Za-z0-9_\-./+=]{15,}['\"]?)",
    re.IGNORECASE,
)

PLACEHOLDER_VALUES = {
    "",
    "''",
    '""',
    "none",
    "null",
    "test",
    "key",
    "k",
    "fake",
    "dummy",
    "changeme",
    "your-api-key",
    "your_api_key",
}


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class AuditResult:
    root: str
    findings: list[AuditFinding]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_json(self) -> str:
        return json.dumps(
            {"root": self.root, "ok": self.ok, "findings": [asdict(f) for f in self.findings]},
            indent=2,
        )


def load_public_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and minimally validate the public release manifest."""
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path} must contain a JSON object")
    return manifest


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_release_files(root: Path):
    """Yield files in a release tree, including VCS files if they are present."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path


@lru_cache(maxsize=4096)
def _compile_public_glob(pattern: str) -> re.Pattern[str]:
    pattern = pattern.replace("\\", "/").lstrip("/")
    if pattern.endswith("/"):
        pattern += "**"
    chunks: list[str] = []
    idx = 0
    while idx < len(pattern):
        char = pattern[idx]
        if char == "*":
            if idx + 1 < len(pattern) and pattern[idx + 1] == "*":
                idx += 2
                if idx < len(pattern) and pattern[idx] == "/":
                    idx += 1
                    chunks.append("(?:.*/)?")
                else:
                    chunks.append(".*")
                continue
            chunks.append("[^/]*")
        elif char == "?":
            chunks.append("[^/]")
        else:
            chunks.append(re.escape(char))
        idx += 1
    return re.compile("^" + "".join(chunks) + "$")


def matches_public_glob(rel_path: str, pattern: str) -> bool:
    return bool(_compile_public_glob(pattern).match(rel_path.replace("\\", "/").lstrip("/")))


def matches_any_public_glob(rel_path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(matches_public_glob(rel_path, pattern) for pattern in patterns)


def manifest_patterns(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return include, exclude, and required path patterns from a manifest.

    The current manifest uses explicit include/exclude globs. Legacy manifests
    from earlier release prep are also accepted to keep the CLI stable.
    """
    includes = list(manifest.get("include_globs", []))
    excludes = list(manifest.get("exclude_globs", []))
    required = list(manifest.get("required_paths", []))

    if not includes:
        includes.extend(manifest.get("allowed_top_level_files", []))
        includes.extend(manifest.get("allowed_paper_files", []))
        includes.extend(f"{root.rstrip('/')}/**" for root in manifest.get("allowed_roots", []))

    return includes, excludes, required


def is_manifest_allowed(rel_path: str, manifest: dict[str, Any]) -> bool:
    includes, excludes, _ = manifest_patterns(manifest)
    if rel_path in SAFE_ENV_TEMPLATES:
        return matches_any_public_glob(rel_path, includes)
    if matches_any_public_glob(rel_path, excludes):
        return False
    return matches_any_public_glob(rel_path, includes)


def _is_forbidden(rel_path: str) -> str | None:
    if rel_path in SAFE_ENV_TEMPLATES:
        return None
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if matches_public_glob(rel_path, pattern):
            return pattern
    return None


def _normalise_secret_value(value: str) -> str:
    return value.strip().strip("'\"").strip()


def _is_placeholder_secret_value(value: str) -> bool:
    cleaned = _normalise_secret_value(value)
    lowered = cleaned.lower()
    if lowered in PLACEHOLDER_VALUES:
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    if lowered.startswith(("your-", "your_", "example-", "example_", "test-", "dummy-")):
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", cleaned):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN)", cleaned):
        return True
    return cleaned.startswith(("values.get(", "os.getenv(", "getenv("))


def _read_scannable_text(path: Path, *, text_hint: bool) -> str:
    if text_hint:
        return path.read_text(errors="ignore")
    data = path.read_bytes()
    # Latin-1 preserves byte values and makes embedded PDF/binary metadata visible
    # without treating arbitrary bytes as a decoding failure.
    return data.decode("latin-1", errors="ignore")


def _find_secret_or_local_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker in LOCAL_OR_PRIVATE_MARKERS:
        if marker.lower() in lowered:
            return f"contains private/local marker `{marker}`"

    match = LOCAL_PATH_RE.search(text)
    if match:
        return "contains absolute local user path"

    for match in PROVIDER_SECRET_ASSIGNMENT_RE.finditer(text):
        if not _is_placeholder_secret_value(match.group("value")):
            return "contains provider API key/token assignment with a non-placeholder value"

    for match in GENERIC_SECRET_ASSIGNMENT_RE.finditer(text):
        if not _is_placeholder_secret_value(match.group("value")):
            return "contains secret-like assignment with a non-placeholder value"

    return None


def audit_release_tree(
    root: Path,
    *,
    max_file_mb: int | None = None,
    manifest_path: Path | None = None,
    enforce_manifest: bool | None = None,
) -> AuditResult:
    """Audit a candidate public release tree."""
    root = root.resolve()
    findings: list[AuditFinding] = []

    manifest: dict[str, Any] | None = None
    candidate_manifest = manifest_path or (root / DEFAULT_MANIFEST_NAME)
    if enforce_manifest is None:
        enforce_manifest = True
    if candidate_manifest.exists():
        manifest = load_public_manifest(candidate_manifest)
    elif enforce_manifest:
        findings.append(
            AuditFinding("error", DEFAULT_MANIFEST_NAME, f"manifest not found: {candidate_manifest}")
        )

    if max_file_mb is None:
        max_file_mb = int(manifest.get("max_file_mb", 10)) if manifest else 10
    max_bytes = max_file_mb * 1024 * 1024

    if not root.exists():
        return AuditResult(
            root=str(root), findings=[AuditFinding("error", ".", "root does not exist")]
        )

    if manifest and enforce_manifest:
        _, _, required = manifest_patterns(manifest)
        for required_path in required:
            if not (root / required_path).exists():
                findings.append(
                    AuditFinding("error", required_path, "required public file is missing")
                )

    for path in iter_release_files(root):
        rel = _relative(path, root)

        if manifest and enforce_manifest and not is_manifest_allowed(rel, manifest):
            findings.append(AuditFinding("error", rel, "not allowed by PUBLIC_MANIFEST"))

        forbidden_pattern = _is_forbidden(rel)
        if forbidden_pattern:
            findings.append(
                AuditFinding("error", rel, f"forbidden public-release path: {forbidden_pattern}")
            )

        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            findings.append(
                AuditFinding("error", rel, f"file exceeds public size limit of {max_file_mb} MB")
            )

        try:
            text_hint = path.suffix.lower() in TEXT_SCAN_EXTENSIONS or path.name.startswith(".env")
            text = _read_scannable_text(path, text_hint=text_hint)
        except OSError:
            continue
        marker = _find_secret_or_local_marker(text)
        if marker:
            findings.append(AuditFinding("error", rel, marker))

    return AuditResult(root=str(root), findings=findings)
