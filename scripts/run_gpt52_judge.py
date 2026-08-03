#!/usr/bin/env python3
"""Run GPT-5.2 LLM-as-judge evaluation on all experiment reports.

Sends ALL criteria in a single API call per evaluation (GPT-5.2's 128K context
handles ~4.5K input tokens easily). This is 5x faster and 4x cheaper on input
tokens compared to batching at 8 criteria per call.

Usage:
    python scripts/run_gpt52_judge.py                    # Run all pending
    python scripts/run_gpt52_judge.py --resume           # Skip already-scored
    python scripts/run_gpt52_judge.py --patterns p0,p1   # Only specific patterns
    python scripts/run_gpt52_judge.py --concurrency 5    # Override concurrency
    python scripts/run_gpt52_judge.py --dry-run           # Show what would be run
"""

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import structlog
from openai import (
    AsyncAzureOpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

from deep_research.config import (
    AZURE_OPENAI_API_VERSION,
    JUDGE,
    JUDGE_MODEL,
    JUDGE_OPENAI_API_KEY,
    JUDGE_OPENAI_ENDPOINT,
    MODELS,
    POOL,
    RETRY,
    TIMEOUTS,
)
from deep_research.evaluation.rubric_v2 import (
    build_general_criteria,
    build_rubric_v2,
    rubric_to_judge_prompt,
    Criterion,
    DIMENSION_WEIGHTS_V2,
)

log = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_BASE = Path("results/experiments")
JUDGE_OUT = Path("results/judge_gpt52")
EVAL_QUERIES = Path("data/eval_queries_v2.json")

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert research report evaluator using the DRACO evaluation methodology.
You assess whether research reports satisfy specific evaluation criteria.

You will be given:
1. The original research query
2. A research report to evaluate
3. A list of evaluation criteria

For EACH criterion, you must provide:
- VERDICT: "SATISFIED" or "NOT_SATISFIED"
- EVIDENCE: A brief quote or reference to specific content in the report
- REASONING: One sentence explaining your judgment

Rules:
- Only mark SATISFIED if the criterion is clearly and fully met
- Partial or vague coverage counts as NOT_SATISFIED
- For citation criteria, check that actual sources/references are provided
- For factual criteria, verify claims are consistent and reasonable
- Be strict but fair -- do not penalize for minor omissions if the substance is there

Respond with valid JSON only."""


# ── Client singleton ──────────────────────────────────────────────────────────
_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=JUDGE_OPENAI_API_KEY,
            azure_endpoint=JUDGE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            max_retries=0,
            timeout=httpx.Timeout(
                connect=TIMEOUTS.connect,
                read=JUDGE.read_timeout,
                write=TIMEOUTS.write,
                pool=TIMEOUTS.pool,
            ),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=POOL.max_connections,
                    max_keepalive_connections=POOL.max_keepalive_connections,
                    keepalive_expiry=POOL.keepalive_expiry,
                ),
            ),
        )
    return _client


# ── Rate-limited call with retry ──────────────────────────────────────────────

async def _judge_call(
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    max_tokens: int = JUDGE.max_tokens,
) -> tuple[str, int]:
    """Single judge API call with rate limiting and retry. Returns (content, tokens)."""
    client = _get_client()
    spec = MODELS.get(JUDGE_MODEL)
    deployment = spec.deployment if spec else JUDGE_MODEL
    last_exc = None

    for attempt in range(RETRY.max_retries):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=JUDGE.temperature,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or "{}"
                tokens = resp.usage.total_tokens if resp.usage else 0
                return content, tokens

            except RateLimitError as e:
                last_exc = e
                retry_after = 0.0
                response = getattr(e, "response", None)
                if response:
                    headers = getattr(response, "headers", {})
                    retry_ms = headers.get("retry-after-ms")
                    if retry_ms:
                        retry_after = int(retry_ms) / 1000.0
                    else:
                        retry_s = headers.get("retry-after")
                        if retry_s:
                            retry_after = float(retry_s)
                wait = max(retry_after, 2.0 * (2 ** min(attempt, 5))) + random.uniform(0, 2)
                log.warning("judge_rate_limited", attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)

            except (APIConnectionError, APITimeoutError, InternalServerError,
                    ConnectionError, httpx.ReadError, httpx.WriteError,
                    httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                wait = min(1.0 * (2 ** min(attempt, 4)) + random.uniform(0, 2), 30)
                log.warning("judge_conn_error", error=type(e).__name__,
                            attempt=attempt + 1, wait=f"{wait:.1f}s")
                await asyncio.sleep(wait)

            except Exception as e:
                log.error("judge_fatal", error=str(e)[:200])
                raise

    raise last_exc


# ── Build rubric for a query ──────────────────────────────────────────────────

def load_queries() -> dict[str, dict]:
    """Load all queries from eval manifest."""
    with open(EVAL_QUERIES) as f:
        data = json.load(f)
    return {q["id"]: q for q in data["queries"]}


def build_rubric_for_query(query: dict) -> tuple[list[str], list[str], dict[str, float], str]:
    """Build criteria list for a query.

    Returns (criteria_texts, criteria_dimensions, dimension_weights, criteria_prompt).
    """
    coverage_criteria = None
    if query.get("expected_elements"):
        coverage_criteria = [
            Criterion(
                text=f"The report covers: {elem}",
                dimension="coverage",
                source="task_specific",
            )
            for elem in query["expected_elements"]
        ]

    source_type = query.get("source", "default")
    rubric = build_rubric_v2(
        query_id=query["id"],
        query_text=query["query"],
        coverage_criteria=coverage_criteria,
        source_type=source_type,
    )

    criteria_texts = [c.text for c in rubric.criteria]
    criteria_dims = [c.dimension for c in rubric.criteria]
    criteria_prompt = rubric_to_judge_prompt(rubric)

    return criteria_texts, criteria_dims, rubric.dimension_weights, criteria_prompt


# ── Evaluate one report ──────────────────────────────────────────────────────

async def evaluate_one(
    semaphore: asyncio.Semaphore,
    pattern: str,
    query_id: str,
    query: dict,
    report_text: str,
) -> dict:
    """Evaluate a single report. Returns result dict."""
    criteria_texts, criteria_dims, dim_weights, criteria_prompt = build_rubric_for_query(query)

    # Truncate very long reports
    words = report_text.split()
    if len(words) > 12000:
        report_text = " ".join(words[:12000]) + "\n\n[... report truncated for evaluation ...]"

    user_msg = f"""## Research Query
{query['query']}

## Report to Evaluate
{report_text}

## Criteria to Evaluate
{criteria_prompt}"""

    t0 = time.time()
    content, total_tokens = await _judge_call(
        semaphore,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    latency = time.time() - t0

    # Parse response
    result_json = json.loads(content)
    evaluations = result_json.get("evaluations", [])

    # Score per dimension
    dim_stats: dict[str, dict] = {}
    for dim in set(criteria_dims):
        dim_stats[dim] = {"met": 0, "total": 0}

    verdicts = []
    for ev in evaluations:
        idx = ev.get("criterion_index", 0)
        if idx >= len(criteria_texts):
            continue
        satisfied = ev.get("verdict", "").upper() == "SATISFIED"
        dim = criteria_dims[idx]
        dim_stats[dim]["total"] += 1
        if satisfied:
            dim_stats[dim]["met"] += 1
        verdicts.append({
            "criterion": criteria_texts[idx],
            "dimension": dim,
            "satisfied": satisfied,
            "evidence": ev.get("evidence", ""),
            "reasoning": ev.get("reasoning", ""),
        })

    # Compute dimension scores
    dimensions = {}
    for dim, stats in dim_stats.items():
        score = stats["met"] / stats["total"] if stats["total"] > 0 else 0.0
        dimensions[dim] = {
            "score": round(score, 4),
            "met": stats["met"],
            "total": stats["total"],
        }

    # Weighted overall
    overall = sum(
        dimensions.get(dim, {}).get("score", 0) * w
        for dim, w in dim_weights.items()
    )

    return {
        "query_id": query_id,
        "pattern": pattern,
        "judge_model": JUDGE_MODEL,
        "overall_score": round(overall, 4),
        "dimensions": dimensions,
        "verdicts": verdicts,
        "n_criteria": len(verdicts),
        "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
        "tokens": total_tokens,
        "latency_s": round(latency, 1),
    }


# ── Main runner ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Run GPT-5.2 judge on all experiments")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored reports")
    parser.add_argument("--patterns", type=str, default="",
                        help="Comma-separated pattern numbers, e.g. '0,1,8'")
    parser.add_argument("--ablations", type=str, default="",
                        help="Comma-separated ablation dir names (without 'ablation_' prefix), "
                             "e.g. 'p3_no_topic_mining,p4_fixed_perspectives'. "
                             "Use 'all' to run all ablation_* directories under results/experiments/.")
    parser.add_argument("--patterns-raw", type=str, default="",
                        help="Verbatim comma-separated experiment_ids — for non-base, non-ablation dirs "
                             "such as 'protocol_a_bing_p4,protocol_a_tavily_p4,base_p4_v1'. "
                             "Pass 'all-protocol-a', 'all-variance', 'top-cluster-variance', "
                             "or 'all-disentanglement' for common batches.")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent judge calls (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    args = parser.parse_args()

    # Load query manifest
    queries = load_queries()
    print(f"Loaded {len(queries)} queries from {EVAL_QUERIES}")

    # Determine patterns to process
    if args.patterns_raw:
        raw_alias = args.patterns_raw.strip().lower()
        if raw_alias == "all-protocol-a":
            patterns = sorted(
                p.name for p in RESULTS_BASE.iterdir()
                if p.is_dir() and p.name.startswith("protocol_a_")
            )
        elif raw_alias == "all-variance":
            patterns = sorted(
                p.name for p in RESULTS_BASE.iterdir()
                if p.is_dir() and p.name.startswith("base_") and "_v" in p.name
            )
        elif raw_alias == "top-cluster-variance":
            patterns = [
                f"base_p{p}_v{v}"
                for p in (5, 6, 8)
                for v in (1, 2, 3)
            ]
        elif raw_alias == "all-disentanglement":
            patterns = sorted(
                p.name for p in RESULTS_BASE.iterdir()
                if p.is_dir() and p.name.startswith("disentangle_")
            )
        else:
            patterns = [n.strip() for n in args.patterns_raw.split(",") if n.strip()]
    elif args.ablations:
        if args.ablations.strip().lower() == "all":
            patterns = sorted(
                p.name for p in RESULTS_BASE.iterdir()
                if p.is_dir() and p.name.startswith("ablation_")
            )
        else:
            names = [n.strip() for n in args.ablations.split(",") if n.strip()]
            patterns = [
                n if n.startswith("ablation_") else f"ablation_{n}"
                for n in names
            ]
    elif args.patterns:
        pattern_nums = [int(p.strip()) for p in args.patterns.split(",")]
        patterns = [f"base_p{n}" for n in pattern_nums]
    else:
        patterns = [f"base_p{i}" for i in range(13)]  # include p11/p12

    # Build work list
    work = []
    for pattern in patterns:
        exp_dir = RESULTS_BASE / pattern
        if not exp_dir.exists():
            continue
        for report_file in sorted(exp_dir.glob("*.md")):
            query_id = report_file.stem
            if query_id not in queries:
                continue

            # Check if already scored
            if args.resume:
                result_path = JUDGE_OUT / pattern / f"{query_id}.json"
                if result_path.exists():
                    continue

            work.append((pattern, query_id, report_file))

    print(f"\nGPT-5.2 Judge Evaluation")
    print(f"  Patterns: {', '.join(patterns)}")
    print(f"  Total pending: {len(work)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Output: {JUDGE_OUT}")
    print(f"  Resume: {args.resume}")

    if args.dry_run:
        by_pattern = defaultdict(int)
        for p, q, _ in work:
            by_pattern[p] += 1
        for p in patterns:
            print(f"    {p}: {by_pattern.get(p, 0)} reports")
        print(f"\n  Estimated cost: ${len(work) * 0.08:.2f}")
        print(f"  Estimated time: {len(work) * 20 / args.concurrency / 60:.1f} min")
        return

    if not work:
        print("  Nothing to do — all reports already scored.")
        return

    JUDGE_OUT.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Progress tracking
    completed = 0
    failed = 0
    total_tokens = 0
    total_cost = 0.0
    start_time = time.time()

    # Process in batches to show progress
    async def process_one(pattern, query_id, report_file, idx):
        nonlocal completed, failed, total_tokens, total_cost

        report_text = report_file.read_text()
        query = queries[query_id]

        try:
            result = await evaluate_one(semaphore, pattern, query_id, query, report_text)

            # Save result
            out_dir = JUDGE_OUT / pattern
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{query_id}.json").write_text(json.dumps(result, indent=2))

            completed += 1
            total_tokens += result.get("tokens", 0)
            spec = MODELS.get(JUDGE_MODEL)
            if spec:
                cost = (result.get("tokens", 0) / 1000) * (spec.cost_per_1k_input + spec.cost_per_1k_output) / 2
                total_cost += cost

            elapsed = time.time() - start_time
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            remaining = (len(work) - completed - failed) / rate if rate > 0 else 0

            print(f"  [{completed + failed}/{len(work)}] {pattern}/{query_id}: "
                  f"{result['overall_score']:.3f} "
                  f"({result['n_satisfied']}/{result['n_criteria']}) "
                  f"{result['tokens']}tok {result['latency_s']}s "
                  f"[{rate:.1f}/min, ~{remaining:.0f}min left]")

        except Exception as e:
            failed += 1
            log.error("eval_failed", pattern=pattern, query_id=query_id,
                      error=type(e).__name__, msg=str(e)[:200])
            print(f"  [{completed + failed}/{len(work)}] {pattern}/{query_id}: FAILED - {e}")

    # Launch all tasks concurrently (semaphore controls actual concurrency)
    tasks = [
        process_one(pattern, query_id, report_file, i)
        for i, (pattern, query_id, report_file) in enumerate(work)
    ]
    await asyncio.gather(*tasks)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  Evaluated: {completed}/{len(work)} ({failed} failed)")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Rate: {completed / elapsed * 60:.1f} evals/min")
    print(f"{'='*70}")

    # Per-pattern summary
    print(f"\nPer-pattern results:")
    for pattern in patterns:
        out_dir = JUDGE_OUT / pattern
        if not out_dir.exists():
            continue
        results = []
        for f in sorted(out_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except:
                pass
        if results:
            scores = [r["overall_score"] for r in results]
            import numpy as np
            print(f"  {pattern}: n={len(results)}, mean={np.mean(scores):.3f}, "
                  f"median={np.median(scores):.3f}, std={np.std(scores):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
