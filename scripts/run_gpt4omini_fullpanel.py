#!/usr/bin/env python3
"""GPT-4o-mini FULL-PANEL cheap-backbone generation (experiment C2).

Companion to run_gpt41_fullpanel.py (C1). Re-runs the 9 GPT prompt-engineered
pipelines (p0..p8) on the CHEAP gpt-4o-mini backbone over the full 90-query
manifest, so the paper gets a THIRD backbone point (gpt-4o corpus, gpt-4.1,
gpt-4o-mini) at a fraction of gpt-4.1's cost. p9/p10 are already local ($0 GPU)
and are NOT part of this run.

MODEL/ENDPOINT REALITY (determined empirically, 2026-07-03):
  * gpt-4o-mini is CHEAP-METERED, not free: config.MODELS['gpt-4o-mini'] =
    $0.00015/1k input + $0.0006/1k output ($0.15 / $0.60 per 1M). The PTU
    gpt-4o (deployment sthree-ptu-02, $0) is a DIFFERENT model.
  * The legacy default AZURE_OPENAI_ENDPOINT (the old PTU host) is DEAD: every
    model 401s there now. gpt-4o-mini's generation deployment ('gpt-4o-mini')
    lives on the SAME endpoint config already uses for SEARCH_MODEL=gpt-4o-mini
    (SEARCH_OPENAI_ENDPOINT, the services.ai.azure.com resource). Verified: a
    1-token completion there returns OK and bills real tokens.
  * So this runner routes AZURE_OPENAI_ENDPOINT/KEY -> SEARCH_OPENAI_ENDPOINT/KEY
    (where mini's generation deployment actually is). This is NOT a judge swap;
    it is the endpoint the config already binds for this exact model. Generation
    (gpt-4o-mini) and web search (gpt-4o-mini) then share one working endpoint.
    Judge-independence still holds: the authoritative judge is GPT-5.2, a
    different model; co-location on one Azure resource is how C1 already runs.

MECHANISM (copied faithfully from run_gpt41_fullpanel / run_gpt41_backbone):
  * set DEFAULT_MODEL=gpt-4o-mini + SEARCH_MODEL=gpt-4o-mini and route the
    generation endpoint BEFORE the first config/pattern import, purge
    deep_research.config + deep_research.patterns.* from sys.modules so the
    backbone re-binds, then HARD-ASSERT config.DEFAULT_MODEL == 'gpt-4o-mini'
    (a 'BACKBONE MISMATCH' abort) so a mispinned run can never masquerade as
    a gpt-4o-mini run.
  * per-query SIGALRM hard wall-clock backstop, graceful asyncio.wait_for
    timeout, resumable JSON checkpoints -- all reused from run_gpt41_backbone.

SAFETY:
  Writes ONLY to NEW top-level dirs, both guarded by the reused resolve_safe_out
  (refuses any path in/under/over the protected corpus):
    results/experiments_gpt4omini_fullpanel/<pattern>/
    checkpoints/gpt4omini_fullpanel/<pattern>/
  GENERATES only; the authoritative judge stays GPT-5.2 via the corpus-safe
  namespaced runner.

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/run_gpt4omini_fullpanel.py --dry-run
  python scripts/run_gpt4omini_fullpanel.py --smoke --patterns p0     # 1 query, paid, tiny
  python scripts/run_gpt4omini_fullpanel.py --run --patterns p0,p1,p2,p3,p4,p5,p6,p7,p8
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the vetted machinery verbatim (path safety, SIGALRM guard, checkpoint /
# result paths, manifest loader, constants). Importing it has NO side effects
# (its main() is __main__-guarded and it does not import deep_research.config).
import run_gpt41_backbone as bb  # noqa: E402

resolve_safe_out = bb.resolve_safe_out
_load_eval = bb._load_eval
_result_path = bb._result_path
_ckpt_path = bb._ckpt_path
is_completed = bb.is_completed
report_body_ok = bb.report_body_ok
_strip_leading_h1 = bb._strip_leading_h1
_alarm_handler = bb._alarm_handler
_HardQueryTimeout = bb._HardQueryTimeout
HARD_QUERY_S = bb.HARD_QUERY_S
PER_QUERY_TIMEOUT_S = bb.PER_QUERY_TIMEOUT_S

# ── The cheap backbone (the ONLY generation change vs the corpus) ─────────────
BACKBONE = "gpt-4o-mini"
CORPUS_SEARCH_MODEL = "gpt-4o-mini"      # corpus value; also the generation model here

# ── Full pattern map (p0..p8 only; p9/p10 are already local & excluded) ───────
PATTERNS = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p2": "deep_research.patterns.p2_supervisor_parallel.pipeline",
    "p3": "deep_research.patterns.p3_meridian.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
    "p5": "deep_research.patterns.p5_hierarchical_wd.pipeline",
    "p6": "deep_research.patterns.p6_reactive_interleaved.pipeline",
    "p7": "deep_research.patterns.p7_graph_decomposition.pipeline",
    "p8": "deep_research.patterns.p8_beam_search.pipeline",
}
ALL_PATTERNS = [f"p{i}" for i in range(9)]     # p0..p8

# ── Corpus-safe NEW output roots (siblings of, never inside, the corpus) ──────
DEFAULT_RESULTS_ROOT = "results/experiments_gpt4omini_fullpanel"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/gpt4omini_fullpanel"
JUDGE_OUT_ROOT = "results/judge_gpt52_gpt4omini_fullpanel"
BUILD_KEY = "third_backbone_gpt4omini_fullpanel"

# Budget: gpt-4o-mini is ~13x cheaper than gpt-4.1, so the heaviest pattern (p4)
# lands well under $1/report; a $5 cap never binds (kept for runaway safety).
BUDGET_USD_DEFAULT = 5.0

# Per-pattern gpt-4o corpus token anchors (mean total_tokens/report on the gpt-4o
# corpus). mini tokenises ~like gpt-4o, so these drive the cost projection.
_CORPUS_TOK = {"p0": 64692, "p1": 781068, "p2": 182061, "p3": 316073,
               "p4": 1132213, "p5": 714602, "p6": 529175, "p7": 485001,
               "p8": 579816}
# Blended $/1M gpt-4o-mini tokens. Bounds are the raw rates ($0.15 in / $0.60 out);
# 'exp' uses the output fraction implied by the measured gpt-4.1 p0 blend (~0.37).
_RATE_MINI = {"low": 0.20, "exp": 0.32, "high": 0.55}   # $/1M tokens, blended


# ── Route the generation endpoint to where gpt-4o-mini actually lives ─────────
def _ensure_mini_endpoint() -> str:
    """Point AZURE_OPENAI_ENDPOINT/KEY at the endpoint that serves gpt-4o-mini.

    LLMCaller (generation + source extraction) uses AZURE_OPENAI_ENDPOINT/KEY for
    EVERY model. The legacy PTU AZURE_OPENAI_ENDPOINT is dead (401 for all
    models); gpt-4o-mini's generation deployment lives on SEARCH_OPENAI_ENDPOINT
    (the services.ai.azure.com resource config already binds for the identical
    SEARCH_MODEL=gpt-4o-mini). Route there BEFORE config/llm_caller first bind so
    the shared client is created against a WORKING endpoint. Idempotent; the
    web-search client (its own SEARCH_OPENAI_ENDPOINT) is untouched. Returns the
    bound endpoint host.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except Exception:
        pass
    search_ep = os.environ.get("SEARCH_OPENAI_ENDPOINT")
    search_key = os.environ.get("SEARCH_OPENAI_API_KEY")
    # Fall back to the judge endpoint, which is the SAME Azure resource here and
    # also serves gpt-4o-mini, if SEARCH_* is not explicitly exported.
    if not (search_ep and search_key):
        search_ep = search_ep or os.environ.get("JUDGE_OPENAI_ENDPOINT")
        search_key = search_key or os.environ.get("JUDGE_OPENAI_API_KEY")
    if not (search_ep and search_key):
        raise SystemExit(
            "SEARCH_OPENAI_ENDPOINT / SEARCH_OPENAI_API_KEY not found in .env; cannot "
            f"route the {BACKBONE!r} generation backbone (it lives on the search-model "
            "endpoint). Export AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY to that "
            "endpoint before running."
        )
    os.environ["AZURE_OPENAI_ENDPOINT"] = search_ep
    os.environ["AZURE_OPENAI_API_KEY"] = search_key
    return search_ep


# ── Backbone-pinned import (faithful copy of bb.import_pattern_pinned) ─────────
def import_pattern_pinned(arch: str):
    """Pin DEFAULT_MODEL=gpt-4o-mini, route endpoint, purge, re-import, verify."""
    if arch not in PATTERNS:
        raise SystemExit(f"Unknown arch {arch!r}. Known: {sorted(PATTERNS)}")
    _ensure_mini_endpoint()
    os.environ["DEFAULT_MODEL"] = BACKBONE
    os.environ["SEARCH_MODEL"] = CORPUS_SEARCH_MODEL
    # never let an oracle backend leak in from a prior process/env
    for var in ("SEARCH_BACKEND", "ORACLE_CORPUS_PATH", "ORACLE_QUERY_ID", "ORACLE_MAX_DOCS"):
        os.environ.pop(var, None)
    for name in list(sys.modules):
        if name == "deep_research.config" or name.startswith("deep_research.patterns."):
            del sys.modules[name]
    mod = importlib.import_module(PATTERNS[arch])
    import deep_research.config as _cfg
    bound = getattr(_cfg, "DEFAULT_MODEL", None)
    if bound != BACKBONE:
        raise SystemExit(
            f"BACKBONE MISMATCH: deep_research.config.DEFAULT_MODEL={bound!r} but "
            f"expected {BACKBONE!r}. The env var must be set before the first config/"
            f"pattern import. Aborting to protect comparability."
        )
    if _cfg.SEARCH_MODEL != CORPUS_SEARCH_MODEL:
        raise SystemExit(
            f"SEARCH_MODEL drift: {_cfg.SEARCH_MODEL!r}, must stay {CORPUS_SEARCH_MODEL!r}."
        )
    spec = _cfg.MODELS.get(BACKBONE)
    if not spec:
        raise SystemExit(f"No ModelSpec for backbone {BACKBONE!r} in config.MODELS.")
    return mod


# ── Per-pattern generation (faithful adaptation of bb.generate_arm, no oracle) ─
async def generate_pattern(
    pattern: str,
    qids: list[str],
    qmap: dict,
    results_root: Path,
    ckpt_root: Path,
    budget: float,
    resume: bool,
) -> dict:
    mod = import_pattern_pinned(pattern)

    n_ok = n_fail = n_skip = 0
    print(f"\n### PATTERN {pattern}  backbone={BACKBONE}  queries={len(qids)} ###",
          flush=True)

    for qid in qids:
        if resume and is_completed(ckpt_root, pattern, qid):
            n_skip += 1
            continue
        q = qmap[qid]
        rp = _result_path(results_root, pattern, qid)
        cp = _ckpt_path(ckpt_root, pattern, qid)
        rp.parent.mkdir(parents=True, exist_ok=True)
        cp.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        _alarm_armed = False
        try:
            try:
                signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(HARD_QUERY_S)
                _alarm_armed = True
            except (ValueError, AttributeError):
                _alarm_armed = False  # not main thread / unsupported platform
            report = await asyncio.wait_for(
                mod.run(q["query"], budget_usd=budget, query_id=qid),
                timeout=PER_QUERY_TIMEOUT_S,
            )
            if _alarm_armed:
                signal.alarm(0)
                _alarm_armed = False
            text = report.full_text()
            if not text.strip():
                text = f"# (empty report)\n\nQuery: {q['query']}\n"
            # ── Non-empty-body guard: refusals must FAIL loudly, never silent 0 ──
            ok_body, why = report_body_ok(text, report)
            if not ok_body:
                elapsed = time.time() - t0
                fail_meta = {
                    "experiment": "GPT4OMINI_FULLPANEL",
                    "pattern": pattern, "backbone": BACKBONE,
                    "query": qid, "status": "empty_body", "reason": why,
                    "elapsed_seconds": round(elapsed, 1),
                    "body_words": len(_strip_leading_h1(text).split()),
                    "sections": len(getattr(report, "sections", None) or []),
                    "chars": len(text),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                marker = rp.parent / (rp.stem + ".failed.json")
                marker.write_text(json.dumps(fail_meta, indent=2, default=str))
                # status != "success" -> is_completed() False -> --resume retries
                cp.write_text(json.dumps(fail_meta, indent=2, default=str))
                if rp.exists():
                    rp.unlink()  # never leave a title-only .md that scores ~0
                n_fail += 1
                print(f"    WARN {qid[:28]}: empty_body ({why}) -> wrote "
                      f"{marker.name}; --resume will retry", flush=True)
                continue
            rp.write_text(text)
            elapsed = time.time() - t0
            import deep_research.config as _cfg
            meta = {
                "experiment": "GPT4OMINI_FULLPANEL",
                "pattern": pattern,
                "backbone": BACKBONE,
                "search_model": _cfg.SEARCH_MODEL,
                "search_backend": getattr(_cfg, "SEARCH_BACKEND", "live"),
                "query": qid,
                "status": "success",
                "elapsed_seconds": round(elapsed, 1),
                "chars": len(text),
                "total_tokens": getattr(report, "total_tokens", 0),
                "total_cost_usd": getattr(report, "total_cost_usd", 0.0),
                "sections": len(getattr(report, "sections", [])),
                "citations": len(getattr(report, "citations", [])),
                "default_model_bound": _cfg.DEFAULT_MODEL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cp.write_text(json.dumps(meta, indent=2, default=str))
            n_ok += 1
            print(f"    OK   {qid[:28]}: {meta['elapsed_seconds']}s "
                  f"{meta['chars']}ch {meta['citations']}cit "
                  f"{meta['total_tokens']}tok "
                  f"${meta['total_cost_usd']:.4f} (model={meta['default_model_bound']})",
                  flush=True)
        except (asyncio.TimeoutError, _HardQueryTimeout, Exception) as e:  # noqa: BLE001
            elapsed = time.time() - t0
            status = "timeout" if isinstance(e, (asyncio.TimeoutError, _HardQueryTimeout)) else "error"
            meta = {
                "experiment": "GPT4OMINI_FULLPANEL",
                "pattern": pattern, "backbone": BACKBONE,
                "query": qid, "status": status,
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e)[:300] or "per-query timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cp.write_text(json.dumps(meta, indent=2, default=str))
            n_fail += 1
            print(f"    FAIL {qid[:28]}: {status} {meta['elapsed_seconds']}s "
                  f"-- {meta['error'][:120]}", flush=True)
        finally:
            if _alarm_armed:
                signal.alarm(0)

    return {"pattern": pattern, "ok": n_ok, "fail": n_fail, "skip": n_skip}


# ── Planning / costing ────────────────────────────────────────────────────────
def project_cost(patterns: list[str], n_queries: int) -> dict:
    out = {sc: {} for sc in ("low", "exp", "high")}
    for p in patterns:
        ct = _CORPUS_TOK[p]
        for sc in out:
            out[sc][p] = ct * _RATE_MINI[sc] / 1e6 * n_queries
    return out


def _print_judge_and_build(patterns: list[str]):
    raw = ",".join(patterns)
    print("\n  Now judge (GPT-5.2, corpus-safe namespaced runner):")
    print(f"    JUDGE_RESULTS_BASE={DEFAULT_RESULTS_ROOT} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} \\")
    print(f"      --patterns-raw {raw} --resume --concurrency 3")
    print("\n  Authoritative judge stays GPT-5.2; gpt-4o-mini is the generation")
    print("  BACKBONE only, never the judge -> judge-independence holds.")


# ── Main ──────────────────────────────────────────────────────────────────────
async def amain():
    ap = argparse.ArgumentParser(
        description="GPT-4o-mini FULL-PANEL cheap-backbone generation (corpus-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + per-pattern cost projection. ZERO API calls.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run 1 query per selected pattern (paid, tiny). Confirms the "
                         "backbone binds, endpoint works, and reports write.")
    ap.add_argument("--run", action="store_true",
                    help="Full generation over the selected patterns/queries. Resumes.")
    ap.add_argument("--patterns", default=",".join(ALL_PATTERNS),
                    help="Comma-separated pattern filter (default p0..p8).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap queries per pattern (debug). 0 = full 90-query manifest.")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    ap.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--self-test", action="store_true",
                    help="In-process unit checks (no API). Exit 0/1.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    unknown = [p for p in patterns if p not in PATTERNS]
    if unknown:
        raise SystemExit(f"Unknown patterns: {unknown}. Known: {list(PATTERNS)}")

    results_root = resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = resolve_safe_out(args.checkpoint_root, "--checkpoint-root")

    qmap = _load_eval()
    all_ids = sorted(qmap)  # full 90-query manifest
    if args.smoke:
        qids_full = all_ids[:1]
    elif args.limit:
        qids_full = all_ids[:args.limit]
    else:
        qids_full = all_ids
    n_q = len(qids_full)

    proj = project_cost(patterns, n_q)

    print("=" * 74)
    print("GPT-4o-mini FULL-PANEL CHEAP-BACKBONE GENERATION (C2)")
    print("=" * 74)
    print(f"  Backbone (metered): DEFAULT_MODEL={BACKBONE}  "
          f"($0.15/1M in, $0.60/1M out -- cheap-metered, NOT free)")
    print(f"  Generation endpoint: SEARCH_OPENAI_ENDPOINT (where {BACKBONE} lives; "
          f"legacy PTU endpoint is dead)")
    print(f"  SEARCH_MODEL:       {CORPUS_SEARCH_MODEL} (same model; corpus value)")
    print(f"  Patterns ({len(patterns)}):  {', '.join(patterns)}")
    print(f"  Queries/pattern:    {n_q} (full manifest = 90)")
    print(f"  WRITE results root (NEW):    {results_root}")
    print(f"  WRITE checkpoint root (NEW): {ckpt_root}")
    print(f"\n  gpt-4o-mini cost projection ({len(patterns)} patterns x {n_q} q):")
    print(f"    {'pattern':8} {'corpusTok':>10} {'low$':>8} {'exp$':>8} {'high$':>8}")
    for p in patterns:
        ct = _CORPUS_TOK.get(p)
        print(f"    {p:8} {ct:>10,} {proj['low'][p]:>8.1f} "
              f"{proj['exp'][p]:>8.1f} {proj['high'][p]:>8.1f}")
    print(f"    {'-'*46}")
    print(f"    {'TOTAL':8} {'':>10} {sum(proj['low'].values()):>8.1f} "
          f"{sum(proj['exp'].values()):>8.1f} {sum(proj['high'].values()):>8.1f}")

    if args.dry_run:
        print("\n  [DRY RUN] No API calls made, nothing written.")
        _print_judge_and_build(patterns)
        return

    if not (args.smoke or args.run):
        print("\n  Nothing to do. Pass --dry-run, --smoke, or --run.")
        _print_judge_and_build(patterns)
        return

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE (1/pattern)" if args.smoke else "FULL"
    print(f"\n  >>> {mode} GENERATION starting (backbone={BACKBONE}) <<<\n")

    summary = []
    for pattern in patterns:
        res = await generate_pattern(
            pattern=pattern, qids=qids_full, qmap=qmap,
            results_root=results_root, ckpt_root=ckpt_root,
            budget=args.budget, resume=args.resume,
        )
        summary.append(res)

    print("\n" + "=" * 74)
    print(f"  GPT-4o-mini FULL-PANEL GENERATION COMPLETE ({mode})")
    print("=" * 74)
    for r in summary:
        print(f"  [{r['pattern']}] ok={r['ok']} fail={r['fail']} skip={r['skip']}")
    total_ok = sum(r["ok"] for r in summary)
    total_fail = sum(r["fail"] for r in summary)
    print(f"  TOTAL ok={total_ok} fail={total_fail}")
    if args.smoke and total_ok >= 1:
        print(f"\n  SMOKE OK: backbone bound + {total_ok}/{len(patterns)} pattern(s) "
              f"produced a report under {results_root}.")
    _print_judge_and_build(patterns)


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    check("9 gpt-4o-mini arms p0..p8", ALL_PATTERNS == [f"p{i}" for i in range(9)])
    check("no local arms in map", all(p not in PATTERNS for p in ("p9", "p10")))
    for p, mp in PATTERNS.items():
        pf = _REPO_ROOT / (mp.replace(".", "/") + ".py")
        check(f"pipeline exists {p} -> {mp}", pf.exists())
    qmap = _load_eval()
    check("manifest == 90 queries", len(qmap) == 90)
    # mini must be a METERED model (cost > 0), else the whole premise is wrong
    import deep_research.config as _cfg
    spec = _cfg.MODELS.get(BACKBONE)
    check("gpt-4o-mini in config.MODELS", spec is not None)
    if spec:
        check("gpt-4o-mini is METERED (cost>0)",
              spec.cost_per_1k_input > 0 and spec.cost_per_1k_output > 0)
        check("gpt-4o-mini deployment name", spec.deployment == "gpt-4o-mini")
    # cost projection sane
    proj = project_cost(ALL_PATTERNS, 90)
    tl, te, th = (sum(proj[s].values()) for s in ("low", "exp", "high"))
    check("cost low <= exp <= high", tl <= te <= th)
    check("full-panel cheaper than gpt-4.1 (<$400)", th < 400)
    # safety guards (reused resolve_safe_out)
    for bad in ["results/experiments", "results/judge_gpt52", "data/analysis", "results"]:
        try:
            resolve_safe_out(bad, "--results-root")
            check(f"refuse protected {bad}", False)
        except SystemExit:
            check(f"refuse protected {bad}", True)
    try:
        p = resolve_safe_out(DEFAULT_RESULTS_ROOT, "--results-root")
        check("accept NEW mini root", p.name == "experiments_gpt4omini_fullpanel")
    except SystemExit:
        check("accept NEW mini root", False)
    # endpoint routing must resolve to a real endpoint (not the dead PTU default)
    try:
        ep = _ensure_mini_endpoint()
        check("mini endpoint resolves", bool(ep) and "services.ai.azure" in ep)
    except SystemExit:
        check("mini endpoint resolves", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
