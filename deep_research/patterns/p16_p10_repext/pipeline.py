"""P16 — P10 replicate extension (DeepResearcher-7b RL agent, 2025).

Re-runs the proven P10 RL-agent scaffold to a SEPARATE pattern directory, generating additional
replicates of the RL-trained local agent for the E2 variance extension (the plan's "top up P10 to
12 replicates"). Corpus-safe (own dir); proven to fit 16GB. Study subject, not a judge.
"""
from deep_research.patterns.p10_deep_researcher.pipeline import run as _run
async def run(query: str, budget_usd: float = 2.0, **kwargs):
    return await _run(query, budget_usd=budget_usd, **kwargs)
