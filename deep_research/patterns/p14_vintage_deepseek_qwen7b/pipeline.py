"""E8 VINTAGE pattern P14 — DeepSeek-R1-Distill-Qwen-7B (2025-01) in the frozen P9 scaffold.

A second vintage point on the same frozen P9 architecture, for the model-vintage gap curve
(Qwen2.5-7B 2024-09 = P9; this 2025-01 distill = P14; Qwen3-8B 2025-04 = P13). Study subject,
not a judge. Corpus-safe (own pattern directory).

Note: this is a Qwen-FAMILY distill, so under the judge-independence rule it could never JUDGE
P9/P10/P13 — but here it is only a graded subject, so the rule does not bind.
"""
from deep_research.patterns.p9_local_baseline.pipeline import run as _p9_run

VINTAGE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 2025-01


async def run(query: str, budget_usd: float = 2.0, **kwargs):
    kwargs["model_id"] = VINTAGE_MODEL_ID
    return await _p9_run(query, budget_usd=budget_usd, **kwargs)
