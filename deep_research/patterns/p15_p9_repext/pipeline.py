"""P15 — P9 replicate extension (Qwen2.5-7B-Instruct, 2024-09).

Re-runs the proven P9 local-baseline scaffold (identical model + tools) to a SEPARATE pattern
directory, generating additional replicates of the local 7B baseline for the E2 variance
extension (frontier-vs-local variance on the local arm). Corpus-safe (own dir); proven to fit
16GB. Study subject, not a judge.
"""
from deep_research.patterns.p9_local_baseline.pipeline import run as _run
async def run(query: str, budget_usd: float = 2.0, **kwargs):
    return await _run(query, budget_usd=budget_usd, **kwargs)
