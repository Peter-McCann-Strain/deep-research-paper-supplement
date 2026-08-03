#!/usr/bin/env python3
"""Shared machinery for the THREE frozen-evidence defensive experiments.

These experiments rebut the "clever synthesis wiring / retrieval polish fixes
factual accuracy" line pushed by Argus / TTD-DR / GRACE. They all run on the
FROZEN ORACLE corpus (data/oracle_corpus_t1.json, the 30 variance-stratified
query_ids) so there is NO live web search: no S2/Bing throttle and no contention
with any live generation run. The generation BACKBONE is gpt-4o-mini (cheap,
metered) routed to the endpoint where it actually lives (SEARCH_OPENAI_ENDPOINT);
gpt-4.1 is deliberately NOT used (a gpt-4.1 subset run already owns that endpoint).

This module holds the pieces every runner needs, reusing the VETTED machinery
from the two shipped backbone-swap harnesses verbatim:
  * run_gpt41_backbone.py  (bb) : resolve_safe_out path guard, _load_eval,
                                  oracle_query_ids, SIGALRM hard wall-clock guard,
                                  _HardQueryTimeout.
  * run_gpt4omini_fullpanel.py (mp) : _ensure_mini_endpoint() endpoint routing +
                                  import_pattern_pinned() backbone pinning.

IMPORTANT ORDERING: gpt-4o-mini's generation deployment lives on
SEARCH_OPENAI_ENDPOINT (the legacy PTU AZURE_OPENAI_ENDPOINT 401s for every
model). llm_caller binds the endpoint at IMPORT time via a module-level shared
client singleton, so the endpoint MUST be routed BEFORE the first deep_research
import in the process. route_and_pin() does that and purges deep_research.* so the
config + llm_caller rebind to the routed endpoint and the pinned backbone. Every
runner must call route_and_pin() before it touches deep_research.

Nothing here is a paid/side-effecting import at module load: bb and mp import only
stdlib at top level, so importing this module triggers ZERO deep_research imports.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Vetted machinery (no top-level deep_research import in either module).
import run_gpt41_backbone as bb          # noqa: E402
import run_gpt4omini_fullpanel as mp     # noqa: E402

BACKBONE = "gpt-4o-mini"
CORPUS_SEARCH_MODEL = "gpt-4o-mini"
ORACLE_CORPUS = _REPO_ROOT / "data" / "oracle_corpus_t1.json"

# Re-export the vetted helpers so runners import them from one place.
resolve_safe_out = bb.resolve_safe_out
_load_eval = bb._load_eval
oracle_query_ids = bb.oracle_query_ids
_alarm_handler = bb._alarm_handler
_HardQueryTimeout = bb._HardQueryTimeout
HARD_QUERY_S = bb.HARD_QUERY_S
PER_QUERY_TIMEOUT_S = bb.PER_QUERY_TIMEOUT_S


# ── Endpoint routing + backbone pinning (call FIRST, before any deep_research import)
def route_and_pin() -> str:
    """Route AZURE_OPENAI_ENDPOINT -> where gpt-4o-mini lives, pin the backbone,
    and purge any already-imported deep_research modules so config + llm_caller
    rebind to the routed endpoint and pinned model. Idempotent. Returns the host.
    """
    ep = mp._ensure_mini_endpoint()
    os.environ["DEFAULT_MODEL"] = BACKBONE
    os.environ["SEARCH_MODEL"] = CORPUS_SEARCH_MODEL
    # Never inherit a stray oracle backend from a prior process; runners set it
    # explicitly per-arm via wire_oracle()/set_query().
    for var in ("SEARCH_BACKEND", "ORACLE_CORPUS_PATH", "ORACLE_QUERY_ID", "ORACLE_MAX_DOCS"):
        os.environ.pop(var, None)
    for name in list(sys.modules):
        if name == "deep_research.config" or name.startswith("deep_research."):
            del sys.modules[name]
    return ep


def assert_mini_bound():
    """Hard-assert the backbone bound after route_and_pin() + a deep_research import."""
    import deep_research.config as cfg
    if cfg.DEFAULT_MODEL != BACKBONE:
        raise SystemExit(
            f"BACKBONE MISMATCH: config.DEFAULT_MODEL={cfg.DEFAULT_MODEL!r}, expected "
            f"{BACKBONE!r}. route_and_pin() must run before the first deep_research import."
        )
    spec = cfg.MODELS.get(BACKBONE)
    if not spec or spec.cost_per_1k_input <= 0:
        raise SystemExit(f"gpt-4o-mini must be a METERED model in config.MODELS; got {spec!r}.")
    return cfg


def make_llm(budget_usd: float):
    """Fresh (LLMCaller, CostTracker) bound to the pinned mini backbone."""
    from deep_research.tools import LLMCaller, CostTracker
    tr = CostTracker(budget_usd=budget_usd)
    return LLMCaller(cost_tracker=tr), tr


def import_pattern_pinned(arch: str):
    """Import a real pipeline (p0/p1/p4/p8...) pinned to the mini backbone.

    Delegates to the vetted mp.import_pattern_pinned (routes endpoint, pins model,
    purges config+patterns, HARD-asserts DEFAULT_MODEL==gpt-4o-mini). It pops the
    oracle env vars; the caller re-wires the oracle backend AFTER this via
    wire_oracle() + set_query() (exactly as run_gpt41_backbone.generate_arm does).
    """
    return mp.import_pattern_pinned(arch)


def wire_oracle(corpus_path, max_docs: int):
    """Turn the oracle (frozen-evidence) backend ON for the current arm."""
    os.environ["SEARCH_BACKEND"] = "oracle"
    os.environ["ORACLE_CORPUS_PATH"] = str(corpus_path)
    os.environ["ORACLE_MAX_DOCS"] = str(max_docs)
    import deep_research.config as cfg
    cfg.SEARCH_BACKEND = "oracle"


def set_query(qid: str):
    """Point the oracle searcher at one query's frozen evidence."""
    os.environ["ORACLE_QUERY_ID"] = qid


# ── Frozen-evidence helpers ───────────────────────────────────────────────────
_CORPUS_CACHE: dict[str, dict] = {}


def load_corpus(path=None) -> dict:
    path = str(path or ORACLE_CORPUS)
    if path not in _CORPUS_CACHE:
        _CORPUS_CACHE[path] = json.loads(Path(path).read_text())
    return _CORPUS_CACHE[path]


def frozen_docs(qid: str, corpus=None) -> list[dict]:
    c = corpus if isinstance(corpus, dict) else load_corpus(corpus)
    return c.get(qid, [])


def evidence_block(docs: list[dict], max_chars: int = 120_000, per_doc: int = 4000) -> str:
    """Numbered [i] Title / URL / content block — the IDENTICAL evidence handed to
    every synthesis scaffold so any cross-scaffold delta is pure synthesis wiring."""
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        title = (d.get("title", "") or "").strip()
        url = (d.get("url", "") or "").strip()
        content = (d.get("content", "") or "")[:per_doc].strip()
        blk = f"[{i}] {title}\nURL: {url}\n{content}\n"
        if used + len(blk) > max_chars:
            break
        parts.append(blk)
        used += len(blk)
    return "\n".join(parts)


def references_from_docs(docs: list[dict], cited: set[int] | None = None) -> str:
    """A ## References block keyed to evidence indices (for c0/GPT-5.2 scorability)."""
    lines = ["## References"]
    for i, d in enumerate(docs, 1):
        if cited is not None and i not in cited:
            continue
        title = (d.get("title", "") or "").strip() or "(untitled)"
        url = (d.get("url", "") or "").strip()
        lines.append(f"[{i}] {title} — {url}")
    return "\n".join(lines)


def sources_by_citation(docs: list[dict], cap: int = 4000) -> dict[int, str]:
    """{citation_idx -> source text} for c0_verifier.verify_report over frozen docs."""
    out: dict[int, str] = {}
    for i, d in enumerate(docs, 1):
        title = (d.get("title", "") or "").strip()
        content = (d.get("content", "") or "")[:cap]
        out[i] = f"{title}\n\n{content}"
    return out


# Shared output contract every scaffold-produced report obeys (scorable markdown).
REPORT_CONTRACT = """Requirements:
- Start with a title (# Title)
- Include an abstract (## Abstract)
- Organise into logical sections (## Section Name)
- Use inline numbered citations [1], [2], ... keyed to the evidence numbers
- End with a ## References section listing each cited source as [N] Title — URL
- Use ONLY the provided source evidence; do not invent sources
- Aim for 1500-3000 words"""


def ensure_references(md: str, docs: list[dict]) -> str:
    """Guarantee a References section so downstream c0/judge parsing never fails."""
    if "## References" in md or "\n# References" in md:
        return md
    return md.rstrip() + "\n\n" + references_from_docs(docs)
