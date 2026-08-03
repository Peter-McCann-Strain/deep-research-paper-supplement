"""E8 VINTAGE pattern P13 — Qwen3-8B (2025-04) in the frozen P9 scaffold.

The frozen P9 architecture (P0 pipeline on a local model) run with a NEWER model vintage, to
measure how the frontier-vs-local gap moves across model release dates. Identical tools, prompts,
and scaffold as P9 (Qwen2.5-7B, 2024-09); only the backbone changes, so any score difference is a
pure model-vintage effect. Qwen3.5-9B (2026-03) OOMs on the 16GB GPU (hybrid linear-attention arch
that does not 4-bit cleanly); Qwen3-8B is the plan's sanctioned fallback (dense, ~5GB in 4-bit).

This model is a STUDY SUBJECT (a deep-research agent being graded), never a judge — so it is
unaffected by the no-small-model-judge rule. Writes to its own pattern directory (corpus-safe).
"""
from deep_research.patterns.p9_local_baseline.pipeline import run as _p9_run

VINTAGE_MODEL_ID = "Qwen/Qwen3-8B"  # 2025-04


async def run(query: str, budget_usd: float = 2.0, **kwargs):
    kwargs["model_id"] = VINTAGE_MODEL_ID
    return await _p9_run(query, budget_usd=budget_usd, **kwargs)
