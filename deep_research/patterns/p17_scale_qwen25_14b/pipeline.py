"""P17 — Qwen2.5-14B-Instruct in the FROZEN P9 scaffold (E8 capacity / E9 scale arm).

The frozen P9 architecture (search -> extract -> single report-gen) run on a LARGER local
backbone (14B vs the 7B P9/P10), holding tools/prompts/queries/evidence-limit/judge constant.
Only the BACKBONE changes. Study subject (graded), not a judge.

Backbone path
-------------
The 14B model OOMs at weights-materialisation through the transformers/bnb LocalLLMCaller on
the single 16 GB RTX 5080 (documented in scripts/run_detector_panel.py and build_e8_vintage.py).
So this arm routes through llama.cpp via a GGUF Q4_K_M file (LlamaCppLLMCaller), which full-
offloads ~8.4 GB weights + a 4096-token KV cache comfortably (~10-11 GB peak). The caller is
INJECTED into the frozen P9 body via kwargs['llm'], so P9 never imports transformers for this
arm and the scaffold stays frozen — the ONLY change vs P9/P14 is the backbone.

  backbone='gguf'         -> LlamaCppLLMCaller(model_path=GGUF)   [default; the only path that fits]
  backbone='transformers' -> fall through to P9's LocalLLMCaller   [will OOM for 14B on this card]

Provenance label is kept as 'Qwen/Qwen2.5-14B-Instruct' regardless of backbone, so judged rows
read the same as a transformers arm would have. evidence_word_limit / temperature(0) /
max_tokens=4096 are inherited unchanged from P9.
"""
from deep_research.patterns.p9_local_baseline.pipeline import run as _p9_run

SCALE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


async def run(query: str, budget_usd: float = 2.0, **kwargs):
    """Run the frozen P9 scaffold on the Qwen2.5-14B backbone.

    Default backbone is the GGUF / llama.cpp path (the only one that fits 16 GB). Pass
    backbone='transformers' to force the (OOM-prone) transformers LocalLLMCaller path.
    """
    backbone = kwargs.pop("backbone", "gguf")
    kwargs["model_id"] = SCALE_MODEL_ID

    if backbone == "gguf" and kwargs.get("llm") is None:
        # Import lazily so importing this module never triggers the native
        # llama.cpp extension load until a run actually needs it.
        from deep_research.tools.llamacpp_llm_caller import (
            DEFAULT_GGUF_PATH,
            LlamaCppLLMCaller,
        )

        gguf_path = kwargs.pop("gguf_path", DEFAULT_GGUF_PATH)
        # CostTracker is rebound inside _p9_run to the per-run tracker.
        kwargs["llm"] = LlamaCppLLMCaller(
            model_path=gguf_path,
            model_id=SCALE_MODEL_ID,
        )

    return await _p9_run(query, budget_usd=budget_usd, **kwargs)
