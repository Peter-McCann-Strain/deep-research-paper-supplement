#!/usr/bin/env python3
"""Run GPT-5.2 LLM-as-judge evaluation — CORPUS-SAFE, judge-NAMESPACED variant.

This is a namespaced sibling of ``scripts/run_gpt52_judge.py``.  It behaves
identically (same GPT-5.2 client/config, same DRACO single-call rubric path,
same output JSON shape) BUT writes to a configurable ``--judge-out`` directory
that defaults to a BRAND-NEW top-level dir (``results/judge_gpt52_v2``), NOT to
``results/judge_gpt52``.

Why this exists
---------------
``results/judge_gpt52`` holds the irreplaceable ~248k-report GPT-5.2 verdict
corpus.  The original runner HARDCODES that path with no override, and a
without-``--resume`` run whose pattern name collides with an existing subdir
overwrites individual verdict JSONs in place.  This variant lets E4/E6/E12 (or
any new experiment) be judged with GPT-5.2 without ever touching the corpus, by
isolating ALL writes under a distinct JUDGE_OUT root.

Two-layer namespacing
---------------------
1. A NEW JUDGE_OUT root (primary isolation): every write is
   ``JUDGE_OUT/<pattern>/<query_id>.json`` so a distinct root fully isolates
   writes regardless of pattern name.
2. Optional ``--experiment-tag`` that namespaces each pattern subdir as
   ``<pattern>__<tag>`` so even an accidental same-root run cannot collide with
   established pattern names.

Hard safety guards (refuse to run if any tripped)
-------------------------------------------------
* ``--judge-out`` may NEVER resolve to ``results/judge_gpt52`` (the corpus),
  nor be a parent/ancestor of it, nor land inside any other protected path
  (``results/experiments``, ``data/analysis``, ``reports/eval_v2/verdicts``).
* ``results/experiments`` (the input reports) is treated strictly READ-ONLY.
* GPT-5.2 is the ONLY judge wired here; the JUDGE_MODEL / endpoint config is
  untouched.  GPT-4o / small / local models are NEVER wired as a judge.

This script BUILDS and VERIFIES only.  Use ``--dry-run`` for a zero-API-spend
cost estimate.  The paid run is launched separately by the human.

Usage:
    python scripts/run_gpt52_judge_namespaced.py --dry-run --patterns 4
    python scripts/run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52_e4 --patterns-raw e4_... --dry-run
    python scripts/run_gpt52_judge_namespaced.py --resume --experiment-tag e6 --patterns 4,6
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

# MUST be present so `python scripts/run_gpt52_judge_namespaced.py` does not
# crash with ModuleNotFoundError (this is exactly what crashed the detector
# panel).  Do NOT remove.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

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
# RESULTS_BASE is the READ-ONLY input root of generated reports.  Never written.
# Defaults to the corpus dir results/experiments (unchanged behaviour).  May be
# REDIRECTED via the JUDGE_RESULTS_BASE env var to a NEW dir so experiments whose
# reports live OUTSIDE the corpus (e.g. E9 under results/experiments_e9_scale, or
# a symlink-staging dir results/experiments_e9_link) can be judged WITHOUT ever
# creating entries inside the protected corpus directory.  This is READ-only
# regardless of where it points.
import os as _os
RESULTS_BASE = Path(_os.environ.get("JUDGE_RESULTS_BASE", "results/experiments"))
EVAL_QUERIES = Path("data/eval_queries_v2.json")

# Default judge-output root — a BRAND-NEW top-level dir, deliberately NOT
# results/judge_gpt52.  Overridable via --judge-out (still guarded below).
DEFAULT_JUDGE_OUT = Path("results/judge_gpt52_v2")

# ── Protected (READ-ONLY / never-write) paths — corpus is irreplaceable ───────
# Resolved relative to the repo root so guards work regardless of cwd.
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

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


# ── Output-path safety guard ──────────────────────────────────────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    """True if `child` is `parent` or lives somewhere inside it (Py3.8-safe)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_judge_out(raw: str) -> Path:
    """Resolve --judge-out and HARD-REFUSE any path that endangers the corpus.

    Refuses if the resolved JUDGE_OUT root:
      * equals a protected path, or
      * lives inside a protected path, or
      * is an ancestor/parent of a protected path (a run rooted there could
        traverse into it).
    Returns the validated absolute Path on success; raises SystemExit otherwise.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    candidate = candidate.resolve()

    for protected in PROTECTED_PATHS:
        prot = protected.resolve()
        if candidate == prot:
            raise SystemExit(
                f"REFUSING: --judge-out resolves to protected corpus path {prot}. "
                f"This runner must NEVER write there. Choose a new dir."
            )
        if _is_relative_to(candidate, prot):
            raise SystemExit(
                f"REFUSING: --judge-out {candidate} is INSIDE protected path {prot}. "
                f"Choose a top-level dir outside all protected paths."
            )
        if _is_relative_to(prot, candidate):
            raise SystemExit(
                f"REFUSING: --judge-out {candidate} is a PARENT of protected path {prot}; "
                f"a run rooted there could traverse into the corpus. Choose a sibling dir."
            )
    return candidate


def assert_safe_pattern_names(patterns: list[str], judge_out: Path) -> None:
    """Belt-and-braces: refuse pattern names that (combined with JUDGE_OUT) would
    resolve a write target into any protected path. With a safe JUDGE_OUT root
    this can only trip on path-traversal aliases (e.g. '../judge_gpt52')."""
    for pat in patterns:
        target = (judge_out / pat).resolve()
        for protected in PROTECTED_PATHS:
            prot = protected.resolve()
            if target == prot or _is_relative_to(target, prot):
                raise SystemExit(
                    f"REFUSING: pattern/experiment name {pat!r} resolves write target "
                    f"{target} into protected path {prot}. Rename it."
                )


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
    # Stream progress live even when stdout/stderr are pipes (the harness / a
    # `subprocess.run` parent / a `> log` redirect).  Without this, Python
    # BLOCK-buffers stdout when it is not a TTY, so per-report progress lines are
    # only flushed at process exit — and are LOST entirely if the run is killed
    # by a timeout, which looks exactly like a silent "zero verdicts, empty log"
    # failure even while verdicts are being written to disk.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description="Run GPT-5.2 judge — CORPUS-SAFE namespaced variant (writes to --judge-out)"
    )
    parser.add_argument("--judge-out", type=str, default=str(DEFAULT_JUDGE_OUT),
                        help=f"Output root for verdicts (default: {DEFAULT_JUDGE_OUT}). "
                             f"HARD-REFUSES to resolve to results/judge_gpt52 or any protected path.")
    parser.add_argument("--experiment-tag", type=str, default="",
                        help="Optional second namespacing axis: writes to "
                             "<judge-out>/<pattern>__<tag>/ so even a same-root run "
                             "cannot collide with established pattern names.")
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
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, judge at most this many (post-resume) reports. "
                             "For small proof-of-fix samples; 0 = no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run (zero API spend)")
    args = parser.parse_args()

    # Resolve and GUARD the output root BEFORE doing any work.
    judge_out = resolve_safe_judge_out(args.judge_out)

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

    # The READ axis is always results/experiments/<pattern>.
    # The WRITE axis is judge_out/<out_pattern> where out_pattern optionally
    # carries the experiment tag for a second namespacing layer.
    tag = args.experiment_tag.strip()

    def out_pattern_for(pattern: str) -> str:
        return f"{pattern}__{tag}" if tag else pattern

    # Belt-and-braces: refuse any pattern/tag combo that traverses into a
    # protected path (only possible via '../' aliases).
    assert_safe_pattern_names([out_pattern_for(p) for p in patterns], judge_out)

    # Build work list
    work = []
    for pattern in patterns:
        exp_dir = RESULTS_BASE / pattern
        if not exp_dir.exists():
            continue
        out_pattern = out_pattern_for(pattern)
        for report_file in sorted(exp_dir.glob("*.md")):
            query_id = report_file.stem
            if query_id not in queries:
                continue

            # Check if already scored
            if args.resume:
                result_path = judge_out / out_pattern / f"{query_id}.json"
                if result_path.exists():
                    continue

            work.append((pattern, out_pattern, query_id, report_file))

    # Optional cap for small proof-of-fix samples (applied AFTER --resume so it
    # always advances through still-pending reports rather than re-checking done).
    if args.limit and args.limit > 0:
        work = work[: args.limit]

    print(f"\nGPT-5.2 Judge Evaluation (CORPUS-SAFE namespaced variant)")
    print(f"  Judge model: {JUDGE_MODEL} @ {JUDGE_OPENAI_ENDPOINT or '<unset>'}")
    print(f"  Read from (READ-ONLY): {RESULTS_BASE}")
    print(f"  Patterns: {', '.join(patterns)}")
    if tag:
        print(f"  Experiment tag: {tag}  ->  write subdirs '<pattern>__{tag}'")
    print(f"  Total pending: {len(work)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Output (WRITE): {judge_out}")
    print(f"  Resume: {args.resume}")
    print(f"  Corpus protected (never written): {PROTECTED_PATHS[0].resolve()}")

    if args.dry_run:
        by_pattern = defaultdict(int)
        for _p, op, _q, _f in work:
            by_pattern[op] += 1
        for pattern in patterns:
            op = out_pattern_for(pattern)
            print(f"    {pattern} -> {judge_out.name}/{op}: {by_pattern.get(op, 0)} reports")
        print(f"\n  Estimated cost: ${len(work) * 0.08:.2f}")
        print(f"  Estimated time: {len(work) * 20 / args.concurrency / 60:.1f} min")
        print(f"  [DRY RUN] No API calls made, nothing written.")
        return

    if not work:
        print("  Nothing to do — all reports already scored.")
        return

    judge_out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Progress tracking
    completed = 0
    failed = 0
    total_tokens = 0
    total_cost = 0.0
    start_time = time.time()

    # Process in batches to show progress
    async def process_one(pattern, out_pattern, query_id, report_file, idx):
        nonlocal completed, failed, total_tokens, total_cost

        report_text = report_file.read_text()
        query = queries[query_id]

        try:
            result = await evaluate_one(semaphore, pattern, query_id, query, report_text)

            # Save result under the namespaced WRITE path.
            out_dir = judge_out / out_pattern
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

            print(f"  [{completed + failed}/{len(work)}] {out_pattern}/{query_id}: "
                  f"{result['overall_score']:.3f} "
                  f"({result['n_satisfied']}/{result['n_criteria']}) "
                  f"{result['tokens']}tok {result['latency_s']}s "
                  f"[{rate:.1f}/min, ~{remaining:.0f}min left]")

        except Exception as e:
            failed += 1
            log.error("eval_failed", pattern=out_pattern, query_id=query_id,
                      error=type(e).__name__, msg=str(e)[:200])
            print(f"  [{completed + failed}/{len(work)}] {out_pattern}/{query_id}: FAILED - {e}")

    # Launch all tasks concurrently (semaphore controls actual concurrency)
    tasks = [
        process_one(pattern, out_pattern, query_id, report_file, i)
        for i, (pattern, out_pattern, query_id, report_file) in enumerate(work)
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
    seen_out = sorted({op for _p, op, _q, _f in work})
    for out_pattern in seen_out:
        out_dir = judge_out / out_pattern
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
            print(f"  {out_pattern}: n={len(results)}, mean={np.mean(scores):.3f}, "
                  f"median={np.median(scores):.3f}, std={np.std(scores):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
