#!/usr/bin/env python3
"""GPT-4.1 FULL-PANEL second-backbone replication (experiment C1).

Extends the 3-arm run_gpt41_backbone.py (p0_base / p4_base / p4_oracle on a
45-query subset) to the FULL PANEL so the paper can show BOTH headline shapes
replicate on a second frontier backbone:

  * the FLAT TOP CLUSTER  (p1..p8 all bunch just above p0), and
  * the P0 -> cluster GAP  (bounded orchestration lift),

on the SAME 90-query manifest the gpt-4o corpus uses.

PANEL (11 patterns, 90 queries each):
  * p0..p8   the 9 GPT-4o pipelines, re-run on the gpt-4.1 backbone (metered).
  * p9, p10  the local 7B patterns, on their OWN local backbones (free, GPU).
             These do NOT touch DEFAULT_MODEL -- they call LocalLLMCaller
             directly, so they are imported WITHOUT the gpt-4.1 pin.

MECHANISM (identical to run_gpt41_backbone.py, reused/copied faithfully):
  * gpt-4.1 patterns: set DEFAULT_MODEL=gpt-4.1 + SEARCH_MODEL=gpt-4o-mini
    BEFORE the first config/pattern import, purge deep_research.config +
    deep_research.patterns.* from sys.modules so the backbone re-binds, then
    HARD-ASSERT config.DEFAULT_MODEL == "gpt-4.1" ("BACKBONE MISMATCH" abort).
  * local patterns: purge the same modules and re-import with the gpt-4.1 pin
    REMOVED (DEFAULT_MODEL restored to the corpus value gpt-4o) so nothing
    leaks the metered backbone into the free local arms.
  * per-query SIGALRM hard wall-clock backstop, graceful asyncio.wait_for
    timeout, and resumable JSON checkpoints -- all reused from the 3-arm script.

SAFETY:
  Writes ONLY to NEW top-level dirs (results/experiments_gpt41_fullpanel/<pattern>/
  and checkpoints/gpt41_fullpanel/<pattern>/), both guarded by resolve_safe_out
  (refuses any path in/under/over the protected corpus).  GENERATES only; the
  authoritative judge stays GPT-5.2 via the corpus-safe namespaced runner.

REUSE:
  --seed-backbone copies the already-generated gpt-4.1 p0_base/p4_base reports
  (results/experiments_gpt41_backbone) into the full-panel p0/p4 dirs so the run
  RESUMES over them instead of regenerating (45 p0 + 32 p4 already on disk).

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate

  # zero-API plan + per-pattern cost projection
  python scripts/run_gpt41_fullpanel.py --dry-run

  # smoke: 1 query for p0 on gpt-4.1 (~1 cent) -- confirms the backbone binds
  python scripts/run_gpt41_fullpanel.py --smoke --patterns p0

  # reuse existing gpt-4.1 p0/p4 work, then full run (paid; launch in background)
  python scripts/run_gpt41_fullpanel.py --seed-backbone
  python scripts/run_gpt41_fullpanel.py --run
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse the 3-arm script's vetted machinery verbatim (path safety, SIGALRM guard,
# checkpoint/result paths, manifest loader, constants).  Importing it has no side
# effects (its main() is __main__-guarded).
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
BACKBONE = bb.BACKBONE                      # "gpt-4.1"
CORPUS_SEARCH_MODEL = bb.CORPUS_SEARCH_MODEL  # "gpt-4o-mini"
CORPUS_DEFAULT_MODEL = "gpt-4o"             # corpus backbone the local arms restore to

# ── Full pattern map: module path per pattern (verified against pipeline.py) ──
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
    "p9": "deep_research.patterns.p9_local_baseline.pipeline",
    "p10": "deep_research.patterns.p10_deep_researcher.pipeline",
}
# gpt-4.1 metered backbone arms (the 9 GPT-4o pipelines re-run on gpt-4.1):
GPT41_PATTERNS = [f"p{i}" for i in range(9)]        # p0..p8
# free local-GPU backbone arms (do NOT pin DEFAULT_MODEL):
LOCAL_PATTERNS = ["p9", "p10"]
LOCAL_BACKBONE_LABEL = {
    "p9": "Qwen2.5-7B-Instruct",
    "p10": "DeepResearcher-7b",
}
ALL_PATTERNS = GPT41_PATTERNS + LOCAL_PATTERNS

# ── Corpus-safe NEW output roots (siblings of, never inside, the corpus) ──────
DEFAULT_RESULTS_ROOT = "results/experiments_gpt41_fullpanel"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/gpt41_fullpanel"
JUDGE_OUT_ROOT = "results/judge_gpt52_gpt41_fullpanel"
BUILD_KEY = "second_backbone_gpt41_fullpanel"

# Existing 3-arm gpt-4.1 output to reuse for p0/p4 (--seed-backbone):
BACKBONE_RESULTS_ROOT = "results/experiments_gpt41_backbone"
BACKBONE_CKPT_ROOT = "checkpoints/gpt41_backbone"
SEED_MAP = {"p0": "p0_base", "p4": "p4_base"}  # fullpanel pattern -> backbone arm

BUDGET_USD_DEFAULT = bb.BUDGET_USD_DEFAULT   # $5.00 (headroom for full P4 work)

# Per-pattern gpt-4.1 cost projection anchors (measured / corpus-token-scaled).
#   corpus_tok = mean total_tokens/report on the gpt-4o corpus (df_runs.parquet,
#                successful reports).  gpt-4.1 uses ~1.21x these tokens (observed
#                on p0: 79.4k/64.7k, p4: 1.35M/1.13M).  Blended $/1M gpt-4.1 token
#                ~2.35 (low) / 2.70 (exp) / 3.30 (high).
_CORPUS_TOK = {"p0": 64692, "p1": 781068, "p2": 182061, "p3": 316073,
               "p4": 1132213, "p5": 714602, "p6": 529175, "p7": 485001,
               "p8": 579816}
_MEASURED_USD = {"p0": 0.2724, "p4": 3.3920}  # measured gpt-4.1 mean cost/report
_INFL = {"low": 1.10, "exp": 1.21, "high": 1.35}
_RATE = {"low": 2.35, "exp": 2.70, "high": 3.30}  # $/1M gpt-4.1 tokens, blended


# ── Route the metered backbone to the endpoint that actually serves gpt-4.1 ───
def _ensure_backbone_endpoint() -> str:
    """gpt-4.1 is deployed ONLY on the JUDGE Azure endpoint, but LLMCaller
    (generation + source extraction) uses AZURE_OPENAI_ENDPOINT/KEY for EVERY
    model.  The corpus default AZURE_OPENAI_* points at the PTU endpoint (gpt-4o
    only), so an un-rerouted gpt-4.1 call 401s.  Point AZURE_OPENAI_ENDPOINT/KEY
    at the judge endpoint (loaded from .env) BEFORE config/llm_caller first bind
    and before the shared client is created -- mirroring how the 3-arm backbone
    run was launched (AZURE_OPENAI_* exported to the judge endpoint), just made
    self-contained.  Idempotent; the web-search client (SEARCH_OPENAI_ENDPOINT)
    is a separate client and is untouched.  Returns the bound endpoint.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except Exception:
        pass
    judge_ep = os.environ.get("JUDGE_OPENAI_ENDPOINT")
    judge_key = os.environ.get("JUDGE_OPENAI_API_KEY")
    if not (judge_ep and judge_key):
        raise SystemExit(
            "JUDGE_OPENAI_ENDPOINT / JUDGE_OPENAI_API_KEY not found in .env; cannot "
            f"route the {BACKBONE!r} backbone (it lives on the judge endpoint). Export "
            "AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY to that endpoint before running."
        )
    os.environ["AZURE_OPENAI_ENDPOINT"] = judge_ep
    os.environ["AZURE_OPENAI_API_KEY"] = judge_key
    return judge_ep


# ── Backbone-pinned import (faithful copy of bb.import_pattern_pinned, full map)
def import_pattern_pinned(arch: str):
    """Pin DEFAULT_MODEL=gpt-4.1, purge config/pattern modules, re-import, verify."""
    if arch not in PATTERNS:
        raise SystemExit(f"Unknown arch {arch!r}. Known: {sorted(PATTERNS)}")
    _ensure_backbone_endpoint()
    os.environ["DEFAULT_MODEL"] = BACKBONE
    os.environ["SEARCH_MODEL"] = CORPUS_SEARCH_MODEL
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


def import_pattern_local(arch: str):
    """Import a local-backbone pattern (p9/p10) WITHOUT the gpt-4.1 pin.

    Restores DEFAULT_MODEL to the corpus value and clears any oracle/backbone env
    leakage from a preceding gpt-4.1 arm, then purges + re-imports so the pattern
    binds its own local caller.  p9/p10 use LocalLLMCaller directly (GPU, $0), so
    the metered backbone must never bind for these arms.
    """
    if arch not in PATTERNS:
        raise SystemExit(f"Unknown arch {arch!r}. Known: {sorted(PATTERNS)}")
    os.environ["DEFAULT_MODEL"] = CORPUS_DEFAULT_MODEL
    os.environ["SEARCH_MODEL"] = CORPUS_SEARCH_MODEL
    for var in ("SEARCH_BACKEND", "ORACLE_CORPUS_PATH", "ORACLE_QUERY_ID", "ORACLE_MAX_DOCS"):
        os.environ.pop(var, None)
    for name in list(sys.modules):
        if name == "deep_research.config" or name.startswith("deep_research.patterns."):
            del sys.modules[name]
    mod = importlib.import_module(PATTERNS[arch])
    import deep_research.config as _cfg
    if getattr(_cfg, "DEFAULT_MODEL", None) == BACKBONE:
        raise SystemExit(
            f"LOCAL-ARM LEAK: DEFAULT_MODEL still {BACKBONE!r} for local pattern {arch!r}; "
            f"the metered backbone must NOT bind for the free local arms."
        )
    return mod


# ── Per-pattern generation (faithful adaptation of bb.generate_arm) ───────────
async def generate_pattern(
    pattern: str,
    qids: list[str],
    qmap: dict,
    results_root: Path,
    ckpt_root: Path,
    budget: float,
    resume: bool,
) -> dict:
    is_local = pattern in LOCAL_PATTERNS
    if is_local:
        mod = import_pattern_local(pattern)
        backbone_label = LOCAL_BACKBONE_LABEL[pattern]
    else:
        mod = import_pattern_pinned(pattern)
        backbone_label = BACKBONE

    n_ok = n_fail = n_skip = 0
    print(f"\n### PATTERN {pattern}  backbone={backbone_label}  "
          f"local={is_local}  queries={len(qids)} ###", flush=True)

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
                    "experiment": "GPT41_FULLPANEL",
                    "pattern": pattern, "local": is_local, "backbone": backbone_label,
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
                "experiment": "GPT41_FULLPANEL",
                "pattern": pattern,
                "local": is_local,
                "backbone": backbone_label,
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
                  f"${meta['total_cost_usd']:.3f} (model={meta['default_model_bound']})",
                  flush=True)
        except (asyncio.TimeoutError, _HardQueryTimeout, Exception) as e:  # noqa: BLE001
            elapsed = time.time() - t0
            status = "timeout" if isinstance(e, (asyncio.TimeoutError, _HardQueryTimeout)) else "error"
            meta = {
                "experiment": "GPT41_FULLPANEL",
                "pattern": pattern, "local": is_local, "backbone": backbone_label,
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


# ── Reuse existing gpt-4.1 p0/p4 work ─────────────────────────────────────────
def seed_from_backbone(results_root: Path, ckpt_root: Path) -> None:
    """Copy successful gpt-4.1 p0_base/p4_base reports+checkpoints into the
    full-panel p0/p4 dirs so --run RESUMES over them (no regeneration).

    Only copies status==success entries whose target is missing.  Idempotent.
    """
    src_res = resolve_safe_out(BACKBONE_RESULTS_ROOT, "backbone results (read)")
    src_ckpt = resolve_safe_out(BACKBONE_CKPT_ROOT, "backbone checkpoints (read)")
    total = 0
    for pat, arm in SEED_MAP.items():
        arm_ckpt_dir = src_ckpt / arm
        if not arm_ckpt_dir.exists():
            print(f"  [seed] {arm}: no backbone checkpoints, skipping")
            continue
        seeded = 0
        for cpf in sorted(arm_ckpt_dir.glob("*.json")):
            try:
                d = json.loads(cpf.read_text())
            except Exception:
                continue
            if d.get("status") != "success":
                continue
            qid = d.get("query")
            src_md = src_res / arm / cpf.name.replace(".json", ".md")
            if not src_md.exists():
                continue
            dst_md = _result_path(results_root, pat, qid)
            dst_cp = _ckpt_path(ckpt_root, pat, qid)
            if dst_cp.exists() and is_completed(ckpt_root, pat, qid):
                continue  # already present
            dst_md.parent.mkdir(parents=True, exist_ok=True)
            dst_cp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_md, dst_md)
            # rewrite the checkpoint under the fullpanel schema (pattern-keyed)
            meta = dict(d)
            meta["experiment"] = "GPT41_FULLPANEL"
            meta["pattern"] = pat
            meta["seeded_from"] = f"{BACKBONE_RESULTS_ROOT}/{arm}"
            dst_cp.write_text(json.dumps(meta, indent=2, default=str))
            seeded += 1
        total += seeded
        print(f"  [seed] {pat} <- {arm}: {seeded} reports reused")
    print(f"  [seed] total reused: {total}")


# ── Planning / costing ────────────────────────────────────────────────────────
def project_cost(patterns: list[str], n_queries: int) -> dict:
    """Return {scenario: {pattern: usd}} for the gpt-4.1 arms (local = $0)."""
    out = {sc: {} for sc in ("low", "exp", "high")}
    for p in patterns:
        if p in LOCAL_PATTERNS:
            for sc in out:
                out[sc][p] = 0.0
            continue
        ct = _CORPUS_TOK[p]
        for sc in out:
            if p in _MEASURED_USD and sc == "exp":
                per = _MEASURED_USD[p]
            else:
                per = ct * _INFL[sc] * _RATE[sc] / 1e6
            out[sc][p] = per * n_queries
    return out


def _print_judge_and_build(results_root: Path, patterns: list[str]):
    raw = ",".join(patterns)
    print("\n  Now judge (GPT-5.2, corpus-safe namespaced runner):")
    print(f"    JUDGE_RESULTS_BASE={DEFAULT_RESULTS_ROOT} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} \\")
    print(f"      --patterns-raw {raw} --resume --concurrency 3")
    print("\n  Authoritative judge stays GPT-5.2; gpt-4.1 is the generation BACKBONE")
    print("  only, never the judge -> judge-independence holds.")


# ── Main ──────────────────────────────────────────────────────────────────────
async def amain():
    ap = argparse.ArgumentParser(
        description="GPT-4.1 FULL-PANEL second-backbone replication (corpus-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + per-pattern cost projection. ZERO API calls.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run 1 query per selected pattern (paid, tiny). Confirms "
                         "the backbone binds and reports write.")
    ap.add_argument("--run", action="store_true",
                    help="Full generation over the selected patterns/queries. Resumes.")
    ap.add_argument("--seed-backbone", action="store_true",
                    help="Copy existing gpt-4.1 p0_base/p4_base reports into the "
                         "full-panel p0/p4 dirs so --run resumes over them. Then exits "
                         "unless combined with --run.")
    ap.add_argument("--patterns", default=",".join(ALL_PATTERNS),
                    help="Comma-separated pattern filter (default p0..p10).")
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

    # Seeding happens up-front so the plan/resume reflects reused work.
    if args.seed_backbone:
        print("\n  Seeding full-panel p0/p4 from existing gpt-4.1 backbone work:")
        results_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        seed_from_backbone(results_root, ckpt_root)
        if not (args.run or args.smoke):
            print("\n  Seed complete. Re-run with --run to generate the remainder.")
            return

    n_gpt41 = sum(1 for p in patterns if p in GPT41_PATTERNS)
    proj = project_cost(patterns, n_q)

    print("=" * 74)
    print("GPT-4.1 FULL-PANEL SECOND-BACKBONE REPLICATION (C1)")
    print("=" * 74)
    print(f"  Backbone (metered arms): DEFAULT_MODEL={BACKBONE}  (p0..p8)")
    print(f"  Local arms (free GPU):   {', '.join(LOCAL_PATTERNS)} "
          f"({', '.join(LOCAL_BACKBONE_LABEL.values())})")
    print(f"  SEARCH_MODEL pinned:     {CORPUS_SEARCH_MODEL} (corpus value, untouched)")
    print(f"  Patterns ({len(patterns)}):  {', '.join(patterns)}")
    print(f"  Queries/pattern:         {n_q} (full manifest = 90)")
    print(f"  WRITE results root (NEW): {results_root}")
    print(f"  WRITE checkpoint root (NEW): {ckpt_root}")
    print(f"\n  gpt-4.1 generation cost projection ({n_gpt41} metered patterns x {n_q} q):")
    print(f"    {'pattern':8} {'corpusTok':>10} {'low$':>8} {'exp$':>8} {'high$':>8}")
    for p in patterns:
        ct = _CORPUS_TOK.get(p)
        cts = f"{ct:,}" if ct else "local"
        print(f"    {p:8} {cts:>10} {proj['low'][p]:>8.0f} "
              f"{proj['exp'][p]:>8.0f} {proj['high'][p]:>8.0f}")
    print(f"    {'-'*46}")
    print(f"    {'TOTAL':8} {'':>10} {sum(proj['low'].values()):>8.0f} "
          f"{sum(proj['exp'].values()):>8.0f} {sum(proj['high'].values()):>8.0f}")
    print(f"  (local p9/p10 = $0 API; GPU/electricity only)")

    if args.dry_run:
        print("\n  [DRY RUN] No API calls made, nothing written.")
        _print_judge_and_build(results_root, patterns)
        return

    if not (args.smoke or args.run):
        print("\n  Nothing to do. Pass --dry-run, --smoke, --seed-backbone, or --run.")
        _print_judge_and_build(results_root, patterns)
        return

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE (1/pattern)" if args.smoke else "FULL"
    print(f"\n  >>> {mode} GENERATION starting <<<\n")

    summary = []
    for pattern in patterns:
        res = await generate_pattern(
            pattern=pattern, qids=qids_full, qmap=qmap,
            results_root=results_root, ckpt_root=ckpt_root,
            budget=args.budget, resume=args.resume,
        )
        summary.append(res)

    print("\n" + "=" * 74)
    print(f"  GPT-4.1 FULL-PANEL GENERATION COMPLETE ({mode})")
    print("=" * 74)
    for r in summary:
        print(f"  [{r['pattern']}] ok={r['ok']} fail={r['fail']} skip={r['skip']}")
    total_ok = sum(r["ok"] for r in summary)
    total_fail = sum(r["fail"] for r in summary)
    print(f"  TOTAL ok={total_ok} fail={total_fail}")
    if args.smoke and total_ok >= 1:
        print(f"\n  SMOKE OK: backbone bound + {total_ok}/{len(patterns)} pattern(s) "
              f"produced a report under {results_root}.")
    _print_judge_and_build(results_root, patterns)


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    # module map complete + importable dotted paths
    check("11 patterns mapped p0..p10", set(PATTERNS) == {f"p{i}" for i in range(11)})
    check("9 gpt-4.1 arms p0..p8", GPT41_PATTERNS == [f"p{i}" for i in range(9)])
    check("2 local arms p9/p10", LOCAL_PATTERNS == ["p9", "p10"])
    for p, mp in PATTERNS.items():
        pf = _REPO_ROOT / (mp.replace(".", "/") + ".py")
        check(f"pipeline exists {p} -> {mp}", pf.exists())
    # 90-query manifest
    qmap = _load_eval()
    check("manifest == 90 queries", len(qmap) == 90)
    # cost projection sane (exp within [low, high]; p0/p4 anchored to measured)
    proj = project_cost(ALL_PATTERNS, 90)
    tl, te, th = (sum(proj[s].values()) for s in ("low", "exp", "high"))
    check("cost low <= exp <= high", tl <= te <= th)
    check("p0 exp anchored to measured", abs(proj["exp"]["p0"] - 0.2724 * 90) < 1e-6)
    check("p4 exp anchored to measured", abs(proj["exp"]["p4"] - 3.3920 * 90) < 1e-6)
    check("local arms $0", proj["exp"]["p9"] == 0.0 and proj["exp"]["p10"] == 0.0)
    # safety guards (reused resolve_safe_out)
    for bad in ["results/experiments", "results/judge_gpt52", "data/analysis", "results"]:
        try:
            resolve_safe_out(bad, "--results-root")
            check(f"refuse protected {bad}", False)
        except SystemExit:
            check(f"refuse protected {bad}", True)
    try:
        p = resolve_safe_out(DEFAULT_RESULTS_ROOT, "--results-root")
        check("accept NEW fullpanel root", p.name == "experiments_gpt41_fullpanel")
    except SystemExit:
        check("accept NEW fullpanel root", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
