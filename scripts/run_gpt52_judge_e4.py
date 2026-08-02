#!/usr/bin/env python3
"""E4 CITE-CAUSAL — Step 4: re-judge the citation-perturbed reports (GPT-5.2 PRIMARY,
plus gpt-4.1 / gpt-4o panel routing).

This is a NEW runner cloned from ``scripts/run_gpt52_judge.py`` /
``run_gpt52_judge_namespaced.py``.  The original HARDCODES JUDGE_OUT=results/judge_gpt52
(the irreplaceable ~248k-report corpus) and is left UNTOUCHED.  This variant differs
in exactly three ways, everything else (DRACO single-call rubric, JSON output shape,
rate-limit/retry, dimension scoring) is identical:

  1. READ axis is the E4 transformed-report layout:
        results/experiments_e4_cite/{condition}/{pattern}/{query_id}.md
     (NOT results/experiments/{pattern}/...).
  2. WRITE axis is a NEW, guarded root, defaulting to results/judge_gpt52_e4/, laid out
        <judge-out>/{condition}/{pattern}/{query_id}.json
     so the corpus is never touched.  HARD-REFUSES any --judge-out that resolves to a
     protected path.
  3. ``--judge {gpt52,gpt-4.1,gpt-4o,dr_judge}`` routes to the right endpoint/model.
     GPT-5.2 is the ONLY authoritative judge.  gpt-4.1 / gpt-4o are wired ONLY as panel
     comparators here at the user's explicit E4 spec; they are NEVER used as the
     authoritative score.  ``dr_judge`` is a LOCAL 7B model: it REFUSES to run in this
     build phase (GPU off-limits) — the route exists but is gated off.

Safety
------
* Corpus (results/judge_gpt52), source reports (results/experiments), analysis parquets
  and reports/eval_v2/verdicts are PROTECTED: never written, --judge-out guarded against
  resolving anywhere inside/over them.
* No local model is loaded; --judge dr_judge aborts before any GPU/model touch.
* --dry-run makes ZERO API calls and writes nothing.

Usage:
    python scripts/run_gpt52_judge_e4.py --help
    python scripts/run_gpt52_judge_e4.py --dry-run                         # all judges? no — pick one
    python scripts/run_gpt52_judge_e4.py --judge gpt52   --dry-run
    python scripts/run_gpt52_judge_e4.py --judge gpt-4.1 --dry-run --conditions C0,C2,C3
    # paid run (human-launched later):
    python scripts/run_gpt52_judge_e4.py --judge gpt52 --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
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
    build_rubric_v2,
    rubric_to_judge_prompt,
    Criterion,
)

log = structlog.get_logger()

# ── Paths ─────────────────────────────────────────────────────────────────────
# READ axis: the E4 transformed reports (NEW dir, produced by build_e4_transforms.py).
E4_REPORTS = _REPO_ROOT / "results" / "experiments_e4_cite"
SAMPLE_MANIFEST = E4_REPORTS / "sample_manifest.json"
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]

# Default WRITE root — a BRAND-NEW dir, deliberately NOT results/judge_gpt52.
DEFAULT_JUDGE_OUT = _REPO_ROOT / "results" / "judge_gpt52_e4"

# ── Judge routing table ───────────────────────────────────────────────────────
# Each route names the config model id and which Azure resource (endpoint/key) hosts
# its deployment.  GPT-5.2 is the only AUTHORITATIVE judge; gpt-4.1/gpt-4o are PANEL
# comparators per E4 spec only.  dr_judge is local => refused in this build phase.
JUDGE_ROUTES = {
    "gpt52":   {"model": "gpt-5.2", "resource": "judge",  "authoritative": True,
                "out_root": "judge_gpt52_e4"},
    "gpt-4.1": {"model": "gpt-4.1", "resource": "main",   "authoritative": False,
                "out_root": "judge_gpt41_e4"},
    "gpt-4o":  {"model": "gpt-4o",  "resource": "main",   "authoritative": False,
                "out_root": "judge_gpt4o_e4"},
    "dr_judge":{"model": "DR-Judge-7B", "resource": "local", "authoritative": False,
                "out_root": "judge_drjudge_e4"},
}

# ── Protected (READ-ONLY / never-write) paths ─────────────────────────────────
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


# ── Client singletons (one per resource) ──────────────────────────────────────
_clients: dict[str, AsyncAzureOpenAI] = {}


def _get_client(resource: str) -> AsyncAzureOpenAI:
    """Return the Azure client for the given resource ('judge' vs 'main').

    GPT-5.2 lives on the dedicated JUDGE Azure resource; gpt-4.1/gpt-4o deployments are
    on the main PTU resource.  Each gets its own endpoint+key.
    """
    if resource in _clients:
        return _clients[resource]
    if resource == "judge":
        endpoint, key = JUDGE_OPENAI_ENDPOINT, JUDGE_OPENAI_API_KEY
    elif resource == "main":
        endpoint, key = AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    else:
        raise SystemExit(f"Unknown resource {resource!r} (no remote endpoint).")
    client = AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
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
    _clients[resource] = client
    return client


# ── Rate-limited call with retry (model-aware token param) ────────────────────

async def _judge_call(
    semaphore: asyncio.Semaphore,
    route: dict,
    messages: list[dict],
    max_tokens: int = JUDGE.max_tokens,
) -> tuple[str, int]:
    client = _get_client(route["resource"])
    spec = MODELS.get(route["model"])
    deployment = spec.deployment if spec else route["model"]
    # gpt-4.1 uses the legacy max_tokens param; gpt-5.2/gpt-4o use max_completion_tokens.
    use_mct = bool(spec.use_max_completion_tokens) if spec else True
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


async def evaluate_one(semaphore, route, query_id, query, report_text):
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
    return {"query_id": query_id, "judge_model": route["model"],
            "overall_score": round(overall, 4), "dimensions": dimensions,
            "verdicts": verdicts, "n_criteria": len(verdicts),
            "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
            "tokens": total_tokens, "latency_s": round(latency, 1)}


# ── Work list from the E4 transformed layout ──────────────────────────────────

def build_work(conditions, patterns_filter, queries, judge_out, resume):
    """Discover {condition}/{pattern}/{query_id}.md under E4_REPORTS."""
    work = []
    for cond in conditions:
        cond_dir = E4_REPORTS / cond
        if not cond_dir.exists():
            continue
        for pat_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            pattern = pat_dir.name
            if patterns_filter and pattern not in patterns_filter:
                continue
            for report_file in sorted(pat_dir.glob("*.md")):
                query_id = report_file.stem
                if query_id not in queries:
                    continue
                if resume:
                    out_path = judge_out / cond / pattern / f"{query_id}.json"
                    if out_path.exists():
                        continue
                work.append((cond, pattern, query_id, report_file))
    return work


async def main():
    ap = argparse.ArgumentParser(
        description="E4 CITE-CAUSAL re-judge runner (GPT-5.2 primary; gpt-4.1/gpt-4o panel).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=list(JUDGE_ROUTES.keys()), default="gpt52",
                    help="Which judge route to run (default gpt52, the authoritative judge).")
    ap.add_argument("--judge-out", type=str, default="",
                    help="WRITE root (guarded). Default: results/<route out_root>_e4.")
    ap.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                    help="Comma-separated subset of C0,C1,C2,C3,C4 (default all).")
    ap.add_argument("--patterns", type=str, default="",
                    help="Comma-separated pattern numbers, e.g. '0,1,10' (default: all present).")
    ap.add_argument("--resume", action="store_true", help="Skip already-judged reports.")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap work to this many reports (smoke-test sizing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="ZERO API spend: print the plan + cost estimate, write nothing.")
    args = ap.parse_args()

    route = JUDGE_ROUTES[args.judge]

    # Local DR-Judge is GPU-bound and OUT OF SCOPE for this build phase.
    if route["resource"] == "local":
        print(f"REFUSING: --judge {args.judge} is a LOCAL 7B model (DR-Judge-7B).")
        print("  The GPU is off-limits in this build phase (another project is using it).")
        print("  The route is wired for completeness but will not load any model here.")
        print("  Run the OpenAI routes (gpt52 / gpt-4.1 / gpt-4o) instead.")
        return

    default_out = _REPO_ROOT / "results" / f"{route['out_root']}"
    judge_out = resolve_safe_judge_out(args.judge_out or str(default_out))

    if not SAMPLE_MANIFEST.exists() and not args.dry_run:
        print(f"WARNING: {SAMPLE_MANIFEST} not found — run build_e4_transforms.py --build first.")
    if not E4_REPORTS.exists():
        print(f"NOTE: {E4_REPORTS} does not exist yet. Run build_e4_transforms.py --build "
              f"(or --dry-run here just prints the plan).")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    patterns_filter = None
    if args.patterns:
        patterns_filter = {f"base_p{n.strip()}" for n in args.patterns.split(",") if n.strip()}

    queries = load_queries()
    print(f"Loaded {len(queries)} queries from {EVAL_QUERIES.name}")
    print(f"E4 re-judge runner  |  judge route: {args.judge}  "
          f"(model={route['model']}, resource={route['resource']}, "
          f"authoritative={route['authoritative']})")
    if not route["authoritative"]:
        print("  NOTE: this is a PANEL comparator only — NOT the authoritative score "
              "(GPT-5.2 is the sole authoritative judge).")
    print(f"  Read (READ-ONLY): {E4_REPORTS}/<condition>/<pattern>/<qid>.md")
    print(f"  Write: {judge_out}/<condition>/<pattern>/<qid>.json")
    print(f"  Conditions: {conditions}")
    print(f"  Corpus protected (never written): {PROTECTED_PATHS[0]}")

    work = build_work(conditions, patterns_filter, queries, judge_out, args.resume)
    if args.limit and args.limit > 0:
        work = work[:args.limit]

    # Cost estimate (GPT-5.2 ~$0.08-0.40/report single-call all-criteria; use spec).
    spec = MODELS.get(route["model"])
    per_call = 0.08
    if spec:
        # ~5k in + ~3k out tokens per report, single call.
        per_call = (5.0 * spec.cost_per_1k_input) + (3.0 * spec.cost_per_1k_output)
    print(f"\n  Total reports to judge: {len(work)}")
    by_cond = defaultdict(int)
    for c, _p, _q, _f in work:
        by_cond[c] += 1
    for c in conditions:
        print(f"    {c}: {by_cond.get(c, 0)} reports")
    print(f"  Est cost ({route['model']}): ${len(work) * per_call:.2f} "
          f"(~${per_call:.3f}/report)")

    if args.dry_run:
        print(f"  Est time: {len(work) * 20 / max(args.concurrency,1) / 60:.1f} min @ conc {args.concurrency}")
        print("  [DRY RUN] No API calls made, nothing written.")
        return

    if not work:
        print("  Nothing to do — all reports already judged (or E4 reports not built).")
        return

    judge_out.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = failed = 0
    total_tokens = 0
    start = time.time()

    async def process_one(cond, pattern, query_id, report_file):
        nonlocal completed, failed, total_tokens
        try:
            result = await evaluate_one(
                semaphore, route, query_id, queries[query_id], report_file.read_text())
            result["pattern"] = pattern
            result["condition"] = cond
            out_dir = judge_out / cond / pattern
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{query_id}.json").write_text(json.dumps(result, indent=2))
            completed += 1
            total_tokens += result.get("tokens", 0)
            print(f"  [{completed + failed}/{len(work)}] {cond}/{pattern}/{query_id}: "
                  f"{result['overall_score']:.3f} "
                  f"({result['n_satisfied']}/{result['n_criteria']}) {result['tokens']}tok")
        except Exception as e:
            failed += 1
            log.error("eval_failed", cond=cond, pattern=pattern, query_id=query_id,
                      error=type(e).__name__, msg=str(e)[:200])
            print(f"  [{completed + failed}/{len(work)}] {cond}/{pattern}/{query_id}: FAILED - {e}")

    await asyncio.gather(*[process_one(c, p, q, f) for c, p, q, f in work])
    elapsed = time.time() - start
    print(f"\n{'='*70}\n  COMPLETE  evaluated {completed}/{len(work)} ({failed} failed) "
          f"in {elapsed/60:.1f} min, {total_tokens:,} tokens\n{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
