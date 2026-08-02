#!/usr/bin/env python3
"""T1_within_openai_neff — FULL-CORPUS gpt-4.1 / gpt-4o judge runner (JUDGE endpoint only).

Purpose
-------
Add gpt-4.1 and gpt-4o as FULL-corpus secondary judges over the SAME base-pattern
reports the GPT-5.2 / Claude panel already scored, so E3's within-OpenAI N_eff cell
(gpt52 x gpt-4.1 x gpt-4o) can be recomputed and contrasted with the within-Anthropic
pair (opus x sonnet). This separates *same-lab* judge redundancy from *family-level*
redundancy (the E3 prereg Phase-2 "symmetric within-OpenAI cell", prereg_E3.md line 41).

This is a NEW runner cloned from ``scripts/run_gpt52_judge_namespaced.py`` (same DRACO
single-call rubric, same JSON output shape, same rate-limit/retry, same dimension
scoring, same corpus-safety guards). It differs in exactly TWO ways:

  1. ``--judge {gpt-4.1,gpt-4o}`` selects which OpenAI secondary judge to run. The
     deployment is resolved for the JUDGE Azure resource (JUDGE_OPENAI_ENDPOINT /
     JUDGE_OPENAI_API_KEY) ONLY. The PTU resource (AZURE_OPENAI_ENDPOINT) is NEVER
     used here — judges go via the cloud JUDGE endpoint, never the PTU.
  2. The default WRITE root is per-judge: results/judge_gpt41 (gpt-4.1) or
     results/judge_gpt4o (gpt-4o). These are the names build_analysis_dataframes.py
     will read once its JUDGE_DIRS gains the two entries (see the T1 run-card), so the
     verdicts land where the dataframe builder + N_eff recompute expect them.

The READ axis is the corpus report layout results/experiments/<pattern>/<qid>.md
(READ-ONLY), same as the namespaced runner. Output layout is
  <judge-out>/<pattern>/<qid>.json
so it is schema-identical to the gpt52 corpus and drops straight into the existing
dataframe builder.

Safety
------
* Corpus (results/judge_gpt52), source reports (results/experiments), analysis
  parquets and reports/eval_v2/verdicts are PROTECTED: never written; --judge-out is
  hard-refused if it resolves inside/over any of them. The per-judge defaults
  (results/judge_gpt41, results/judge_gpt4o) are brand-new top-level dirs.
* JUDGE endpoint ONLY. There is no "main"/PTU route in this runner by construction.
* --resume skips already-written <pattern>/<qid>.json (clobber-safe; idempotent).
* --dry-run makes ZERO API calls and writes nothing.

Deployment-name note (READ BEFORE THE PAID RUN)
-----------------------------------------------
gpt-4.1's config deployment id is "gpt-4.1", which is a deployment ON the JUDGE
resource (confirmed used as the E4 gpt-4.1 panel model id). gpt-4o's config
deployment id in deep_research.config.MODELS is "sthree-ptu-02" — that is the *PTU*
deployment and is NOT valid on the JUDGE endpoint. This runner therefore resolves the
JUDGE-endpoint deployment names from a local route table with env overrides:
    JUDGE_GPT41_DEPLOYMENT (default "gpt-4.1")
    JUDGE_GPT4O_DEPLOYMENT (default "gpt-4o")
Confirm the JUDGE-endpoint gpt-4o deployment name with the endpoint owner and, if it
differs from "gpt-4o", export JUDGE_GPT4O_DEPLOYMENT=<name> before the paid run. The
--dry-run prints the resolved deployment so the name can be eyeballed with zero spend.

Usage
-----
    python scripts/run_openai_panel_judge_fullcorpus.py --judge gpt-4.1 --dry-run
    python scripts/run_openai_panel_judge_fullcorpus.py --judge gpt-4o  --dry-run
    # paid run (human-launched, overnight, resumable):
    python scripts/run_openai_panel_judge_fullcorpus.py --judge gpt-4.1 --resume --concurrency 5
    python scripts/run_openai_panel_judge_fullcorpus.py --judge gpt-4o  --resume --concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))  # MUST: avoids ModuleNotFoundError when run as a script

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
    JUDGE_OPENAI_API_KEY,
    JUDGE_OPENAI_ENDPOINT,
    MODELS,
    POOL,
    RETRY,
    TIMEOUTS,
)
from deep_research.evaluation.rubric_v2 import (
    build_rubric_v2,
    rubric_to_judge_prompt,
    Criterion,
)

log = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────────
# READ axis: the corpus reports (READ-ONLY). Same layout as the gpt52 corpus runner.
RESULTS_BASE = Path(os.environ.get("JUDGE_RESULTS_BASE", str(_REPO_ROOT / "results" / "experiments")))
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"

# ── OpenAI secondary-judge route table (JUDGE resource ONLY) ──────────────────
# model         = config MODELS key (for cost + max_completion_tokens flag)
# deployment    = JUDGE-endpoint deployment name (env-overridable; see module docstring)
# out_root      = default WRITE dir name under results/ (matches the dataframe builder's
#                 JUDGE_DIRS keys gpt41 / gpt4o once those entries are added).
JUDGE_ROUTES = {
    "gpt-4.1": {
        "model": "gpt-4.1",
        "deployment": os.environ.get("JUDGE_GPT41_DEPLOYMENT", "gpt-4.1"),
        "out_root": "judge_gpt41",
        "judge_name": "gpt41",
    },
    "gpt-4o": {
        "model": "gpt-4o",
        "deployment": os.environ.get("JUDGE_GPT4O_DEPLOYMENT", "gpt-4o"),
        "out_root": "judge_gpt4o",
        "judge_name": "gpt4o",
    },
}

# ── Protected (READ-ONLY / never-write) paths — corpus is irreplaceable ───────
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

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


# ── Output-path safety guard (mirrors the namespaced runner) ──────────────────

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_judge_out(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    candidate = candidate.resolve()
    for protected in PROTECTED_PATHS:
        prot = protected.resolve()
        if candidate == prot:
            raise SystemExit(
                f"REFUSING: --judge-out resolves to protected corpus path {prot}.")
        if _is_relative_to(candidate, prot):
            raise SystemExit(
                f"REFUSING: --judge-out {candidate} is INSIDE protected path {prot}.")
        if _is_relative_to(prot, candidate):
            raise SystemExit(
                f"REFUSING: --judge-out {candidate} is a PARENT of protected path {prot}.")
    return candidate


# ── Client singleton (JUDGE resource ONLY) ────────────────────────────────────
_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    """Azure client bound to the dedicated JUDGE resource (endpoint + key).

    There is deliberately NO 'main'/PTU route in this runner: secondary OpenAI
    judges run on the cloud JUDGE endpoint only, never the PTU.
    """
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=JUDGE_OPENAI_API_KEY,
            azure_endpoint=JUDGE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            max_retries=0,
            timeout=httpx.Timeout(
                connect=TIMEOUTS.connect, read=JUDGE.read_timeout,
                write=TIMEOUTS.write, pool=TIMEOUTS.pool),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=POOL.max_connections,
                    max_keepalive_connections=POOL.max_keepalive_connections,
                    keepalive_expiry=POOL.keepalive_expiry)),
        )
    return _client


# ── Rate-limited call with retry (model-aware token param) ────────────────────

async def _judge_call(
    semaphore: asyncio.Semaphore,
    route: dict,
    messages: list[dict],
    max_tokens: int = JUDGE.max_tokens,
) -> tuple[str, int]:
    client = _get_client()
    deployment = route["deployment"]
    spec = MODELS.get(route["model"])
    # gpt-4.1 uses the legacy max_tokens param; gpt-4o uses max_completion_tokens.
    use_mct = bool(spec.use_max_completion_tokens) if spec else False
    last_exc = None

    for attempt in range(RETRY.max_retries):
        async with semaphore:
            try:
                kwargs = dict(
                    model=deployment,
                    messages=messages,
                    temperature=JUDGE.temperature,
                    response_format={"type": "json_object"},
                )
                if use_mct:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                resp = await client.chat.completions.create(**kwargs)
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


# ── Build rubric for a query (identical to the corpus runner) ─────────────────

def load_queries() -> dict[str, dict]:
    data = json.loads(EVAL_QUERIES.read_text())
    return {q["id"]: q for q in data["queries"]}


def build_rubric_for_query(query: dict):
    coverage_criteria = None
    if query.get("expected_elements"):
        coverage_criteria = [
            Criterion(text=f"The report covers: {elem}", dimension="coverage",
                      source="task_specific")
            for elem in query["expected_elements"]
        ]
    rubric = build_rubric_v2(
        query_id=query["id"], query_text=query["query"],
        coverage_criteria=coverage_criteria, source_type=query.get("source", "default"))
    return ([c.text for c in rubric.criteria],
            [c.dimension for c in rubric.criteria],
            rubric.dimension_weights,
            rubric_to_judge_prompt(rubric))


async def evaluate_one(semaphore, route, pattern, query_id, query, report_text):
    criteria_texts, criteria_dims, dim_weights, criteria_prompt = build_rubric_for_query(query)
    words = report_text.split()
    if len(words) > 12000:
        report_text = " ".join(words[:12000]) + "\n\n[... report truncated for evaluation ...]"
    user_msg = (f"## Research Query\n{query['query']}\n\n"
                f"## Report to Evaluate\n{report_text}\n\n"
                f"## Criteria to Evaluate\n{criteria_prompt}")
    t0 = time.time()
    content, total_tokens = await _judge_call(
        semaphore, route,
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}])
    latency = time.time() - t0
    result_json = json.loads(content)
    evaluations = result_json.get("evaluations", [])
    dim_stats = {dim: {"met": 0, "total": 0} for dim in set(criteria_dims)}
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
        verdicts.append({"criterion": criteria_texts[idx], "dimension": dim,
                         "satisfied": satisfied, "evidence": ev.get("evidence", ""),
                         "reasoning": ev.get("reasoning", "")})
    dimensions = {dim: {"score": round(s["met"] / s["total"], 4) if s["total"] else 0.0,
                        "met": s["met"], "total": s["total"]}
                  for dim, s in dim_stats.items()}
    overall = sum(dimensions.get(dim, {}).get("score", 0) * w for dim, w in dim_weights.items())
    return {"query_id": query_id, "pattern": pattern, "judge_model": route["model"],
            "overall_score": round(overall, 4), "dimensions": dimensions,
            "verdicts": verdicts, "n_criteria": len(verdicts),
            "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
            "tokens": total_tokens, "latency_s": round(latency, 1)}


# ── Pattern discovery (mirror the namespaced runner's selectors) ──────────────

def select_patterns(args) -> list[str]:
    """Return the list of <pattern> subdir names to judge under RESULTS_BASE.

    Default = the core base set base_p0..base_p12 (the patterns the gpt52/Claude
    panel scored in the N_eff base cell). --patterns overrides with explicit numbers;
    --patterns-raw overrides with verbatim dir names.
    """
    if args.patterns_raw:
        return [n.strip() for n in args.patterns_raw.split(",") if n.strip()]
    if args.patterns:
        nums = [int(p.strip()) for p in args.patterns.split(",")]
        return [f"base_p{n}" for n in nums]
    return [f"base_p{i}" for i in range(13)]  # base_p0 .. base_p12


async def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(
        description="FULL-CORPUS gpt-4.1/gpt-4o secondary judge (JUDGE endpoint only).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=list(JUDGE_ROUTES.keys()), required=True,
                    help="Which OpenAI secondary judge to run (gpt-4.1 or gpt-4o).")
    ap.add_argument("--judge-out", type=str, default="",
                    help="WRITE root (guarded). Default: results/<route out_root> "
                         "(results/judge_gpt41 or results/judge_gpt4o).")
    ap.add_argument("--patterns", type=str, default="",
                    help="Comma-separated pattern numbers, e.g. '0,1,10'. "
                         "Default: base_p0..base_p12 (the N_eff base cell).")
    ap.add_argument("--patterns-raw", type=str, default="",
                    help="Verbatim comma-separated dir names, overrides --patterns.")
    ap.add_argument("--resume", action="store_true", help="Skip already-judged reports.")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap work to this many reports (smoke-test sizing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="ZERO API spend: print the plan + cost + resolved deployment, write nothing.")
    args = ap.parse_args()

    route = JUDGE_ROUTES[args.judge]
    default_out = _REPO_ROOT / "results" / route["out_root"]
    judge_out = resolve_safe_judge_out(args.judge_out or str(default_out))

    queries = load_queries()
    patterns = select_patterns(args)

    # Build work list over the corpus report layout.
    work = []
    for pattern in patterns:
        exp_dir = RESULTS_BASE / pattern
        if not exp_dir.exists():
            continue
        for report_file in sorted(exp_dir.glob("*.md")):
            query_id = report_file.stem
            if query_id not in queries:
                continue
            if args.resume:
                out_path = judge_out / pattern / f"{query_id}.json"
                if out_path.exists():
                    continue
            work.append((pattern, query_id, report_file))
    if args.limit and args.limit > 0:
        work = work[:args.limit]

    print(f"Loaded {len(queries)} queries from {EVAL_QUERIES.name}")
    print(f"OpenAI within-family secondary judge  |  --judge {args.judge}")
    print(f"  model={route['model']}  deployment(JUDGE-endpoint)={route['deployment']!r}")
    print(f"  Endpoint (JUDGE, never PTU): {JUDGE_OPENAI_ENDPOINT or '<unset>'}")
    print(f"  Read (READ-ONLY): {RESULTS_BASE}/<pattern>/<qid>.md")
    print(f"  Write: {judge_out}/<pattern>/<qid>.json")
    print(f"  Patterns: {', '.join(patterns)}")
    print(f"  Total pending: {len(work)}  |  concurrency={args.concurrency}  resume={args.resume}")
    print(f"  Corpus protected (never written): {PROTECTED_PATHS[0]}")

    spec = MODELS.get(route["model"])
    # ~5k in + ~3k out tokens per report, single DRACO call.
    per_call = (5.0 * spec.cost_per_1k_input) + (3.0 * spec.cost_per_1k_output) if spec else 0.08

    if args.dry_run:
        by_pat = defaultdict(int)
        for p, _q, _f in work:
            by_pat[p] += 1
        for pattern in patterns:
            print(f"    {pattern} -> {judge_out.name}/{pattern}: {by_pat.get(pattern, 0)} reports")
        print(f"\n  Est cost ({route['model']}): ${len(work) * per_call:.2f} (~${per_call:.4f}/report)")
        print(f"  Est time: {len(work) * 20 / max(args.concurrency,1) / 60:.1f} min @ conc {args.concurrency}")
        print(f"  [DRY RUN] No API calls made, nothing written.")
        print(f"  NOTE: confirm deployment {route['deployment']!r} exists on the JUDGE endpoint; "
              f"override via JUDGE_{'GPT41' if args.judge=='gpt-4.1' else 'GPT4O'}_DEPLOYMENT if not.")
        return

    if not work:
        print("  Nothing to do — all reports already judged.")
        return

    judge_out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = failed = 0
    total_tokens = 0
    start = time.time()

    async def process_one(pattern, query_id, report_file):
        nonlocal completed, failed, total_tokens
        try:
            result = await evaluate_one(
                semaphore, route, pattern, query_id, queries[query_id], report_file.read_text())
            out_dir = judge_out / pattern
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{query_id}.json").write_text(json.dumps(result, indent=2))
            completed += 1
            total_tokens += result.get("tokens", 0)
            elapsed = time.time() - start
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{completed + failed}/{len(work)}] {pattern}/{query_id}: "
                  f"{result['overall_score']:.3f} "
                  f"({result['n_satisfied']}/{result['n_criteria']}) {result['tokens']}tok "
                  f"[{rate:.1f}/min]")
        except Exception as e:
            failed += 1
            log.error("eval_failed", pattern=pattern, query_id=query_id,
                      error=type(e).__name__, msg=str(e)[:200])
            print(f"  [{completed + failed}/{len(work)}] {pattern}/{query_id}: FAILED - {e}")

    await asyncio.gather(*[process_one(p, q, f) for p, q, f in work])
    elapsed = time.time() - start
    print(f"\n{'='*70}\n  COMPLETE  evaluated {completed}/{len(work)} ({failed} failed) "
          f"in {elapsed/60:.1f} min, {total_tokens:,} tokens\n{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
