#!/usr/bin/env python3
"""GPT-4.1 SECOND-BACKBONE ROBUSTNESS ARM — does the headline replicate?

Closes the single-backbone (GPT-4o-only) reviewer gap.  Re-runs the CORE
contrast on a SECOND frontier backbone we have, gpt-4.1, to show the two
headline findings of "Bounded Returns to Orchestration" REPLICATE off the
gpt-4o corpus:

  (a) ORCHESTRATION'S BOUNDED GAIN.  best pipeline (P4 perspective_storm,
      the canonical GPT-5.2 headline winner: base_p4 0.46655 >= base_p1
      0.46648 on df_overall_scores) buys only a small lift over the bare
      P0 baseline.  Arm contrast: p4_base  vs  p0_base.

  (b) THE ORACLE RETRIEVAL/SYNTHESIS BOTTLENECK.  handing the best pipeline a
      FIXED, pooled gold corpus (oracle) and removing all retrieval variance
      lifts it far more than the architecture did -- i.e. the bottleneck is
      retrieval+synthesis, not orchestration.  Arm contrast: p4_oracle  vs
      p4_base (matched queries).

Backbone is the ONE thing we change vs the corpus: DEFAULT_MODEL=gpt-4.1.
gpt-4.1 is NOT the judge (judge stays GPT-5.2), so judge-independence holds.

THREE ARMS (the minimal set that tests (a) and (b)):
  * p0_base       P0 baseline,            live web search, gpt-4.1 backbone
  * p4_base       P4 perspective_storm,   live web search, gpt-4.1 backbone
  * p4_oracle     P4 perspective_storm,   FROZEN oracle corpus, gpt-4.1 backbone

MECHANISM (mirrors E9 backbone swap + E5 oracle wiring exactly):
  * Backbone IV   : set os.environ["DEFAULT_MODEL"]="gpt-4.1" BEFORE importing
                    config/any pattern, purge deep_research.config +
                    deep_research.patterns.* from sys.modules so the backbone
                    re-binds at import, then HARD-ASSERT config.DEFAULT_MODEL
                    == "gpt-4.1" (a "BACKBONE MISMATCH" abort).
  * SEARCH_MODEL  : pinned to the CORPUS value gpt-4o-mini (deployed; never
                    touched) -- search-query authoring is held fixed so the
                    only generation-side change is the synthesis backbone.
  * Oracle arm    : SEARCH_BACKEND=oracle + ORACLE_CORPUS_PATH (the pooled
                    Tier-1 oracle_corpus_t1.json) + ORACLE_QUERY_ID per query,
                    exactly as run_e5_oracle_dose.py wires it.

QUERY SUBSET (deterministic, cost-control ~45 of 90):
  The pooled oracle corpus (data/oracle_corpus_t1.json) only covers the 30
  variance-stratified query_ids (all in the eval set, all >=5 gold docs).  So:
    * p4_oracle runs on those 30 (the matched oracle set).
    * p0_base / p4_base run on a 45-query subset = those SAME 30 + 15 more
      drawn STRATIFIED-BY-DIFFICULTY (deterministic, seed=7) from the other 60.
  This guarantees the (b) oracle-vs-base contrast is matched on 30 queries while
  the (a) bounded-gain contrast uses the full 45, and selection is reproducible.

SAFETY (never touches the irreplaceable corpus):
  Writes ONLY to NEW top-level dirs (results/experiments_gpt41_backbone/<arm>/
  and checkpoints/gpt41_backbone/<arm>/).  resolve_safe_out REFUSES any output
  root that equals / is inside / is a parent of the protected corpus paths.
  --dry-run and --smoke make ZERO / minimal API calls.  This script GENERATES
  only; authoritative judging is GPT-5.2 via the corpus-safe namespaced runner
  (the exact invocation is printed; this script never imports/runs the judge).

USAGE:
  # zero-API plan + cost estimate + judge/build commands
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/run_gpt41_backbone.py --dry-run

  # smoke test: 1 query per arm (paid, tiny) -- confirms gpt-4.1 backbone binds
  python scripts/run_gpt41_backbone.py --smoke

  # full 45-query run (paid PTU/Azure; launched in background by the human)
  python scripts/run_gpt41_backbone.py --run

Then judge (GPT-5.2, corpus-safe) -- printed by --dry-run:
  JUDGE_RESULTS_BASE=results/experiments_gpt41_backbone \\
    python scripts/run_gpt52_judge_namespaced.py \\
      --judge-out results/judge_gpt52_gpt41_backbone \\
      --patterns-raw p0_base,p4_base,p4_oracle --resume --concurrency 3
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── The second backbone (the ONLY change vs the gpt-4o corpus) ────────────────
BACKBONE = "gpt-4.1"
CORPUS_SEARCH_MODEL = "gpt-4o-mini"   # corpus value, deployed, NEVER changed

# ── Best pipeline = canonical GPT-5.2 headline winner ─────────────────────────
# base_p4 (0.46655) >= base_p1 (0.46648) on df_overall_scores.parquet (gpt52).
BEST_PIPELINE = "p4"

# ── The three arms (arch, oracle?) ────────────────────────────────────────────
PATTERNS = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
}
# arm-name -> (arch, oracle)
ARMS = {
    "p0_base":   ("p0", False),
    f"{BEST_PIPELINE}_base":   (BEST_PIPELINE, False),
    f"{BEST_PIPELINE}_oracle": (BEST_PIPELINE, True),
}

# ── Corpus-safe NEW output roots ──────────────────────────────────────────────
DEFAULT_RESULTS_ROOT = "results/experiments_gpt41_backbone"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/gpt41_backbone"
JUDGE_OUT_ROOT = "results/judge_gpt52_gpt41_backbone"
BUILD_KEY = "second_backbone_gpt41"

# ── Inputs (READ-ONLY) ────────────────────────────────────────────────────────
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"
ORACLE_CORPUS = _REPO_ROOT / "data" / "oracle_corpus_t1.json"

# ── Subset sizing / reproducibility ───────────────────────────────────────────
SUBSET_SIZE = 45        # base arms: oracle-30 + 15 more stratified
SEED = 7                # matches variance_stratified.json seed
# BUDGET RATIONALE (comparability, not generosity):
#   The corpus backbone gpt-4o runs on PTU (cost_per_1k = $0.00 in config.MODELS),
#   so the CostTracker measures $0 and the budget NEVER binds — corpus P4 always
#   does its full pipeline work. gpt-4.1 is billed PER TOKEN (~$0.002/$0.008), so
#   the IDENTICAL P4 work costs ~$1.9–$2.3 real dollars and a $2.00 cap TRUNCATES
#   it mid-synthesis (observed: 71/75 P4 queries died "Budget exceeded: $2.00").
#   That truncation is a pure pricing artifact, not the corpus's intent. To keep
#   gpt-4.1 P4 comparable to the (free) corpus P4 we raise the DOLLAR ceiling to
#   give the same work headroom. Empirically, FULL P4 work on gpt-4.1 reaches
#   ~1.6M tokens ≈ $4.0/query (the short truncated "successes" at <$2 were not
#   representative); a $4.00 cap clips the heaviest queries right at the line, so
#   we set $5.00 to give the largest legitimate P4 runs headroom to land while
#   still guarding genuine runaways. P0 (cheap, ~$0.1) is unaffected.
BUDGET_USD_DEFAULT = 5.0
# PER-QUERY WALL-CLOCK RATIONALE:
#   P4 perspective_storm on gpt-4.1 is genuinely long: 5 perspectives × (search +
#   extract + multi-turn conversations + triangulation + synthesis), ~600K tokens.
#   On the FREE PTU corpus backbone these finished, but at gpt-4.1 standard-endpoint
#   latency the heaviest queries run PAST 600s and were being abandoned mid-work
#   (observed: consecutive "timeout 600.0s" on full, non-hung queries that were
#   making steady LLM progress). 600s was sized for hung-fetch detection, not for
#   P4's legitimate runtime. Observed full-pipeline timings at gpt-4.1 latency:
#   light queries ~320-580s, heavy queries reach stage_5_triangulate / synthesis
#   only at ~1000-1300s (one full, non-hung query was still in synthesis at 1200s).
#   A 1200s cap was still abandoning genuinely-progressing heavy queries at ~$3
#   each AND dropping them from the matched-query contrasts (oracle-vs-base needs
#   matched coverage), so we raise to 1800s (30 min) to let nearly all full P4
#   work land. A genuinely hung query is still bounded: each url fetch is hard-
#   capped at 30s, the SIGALRM backstop fires at 1800+90s, and cost is capped by
#   the $5 budget regardless — so a runaway can cost at most ~$5 and ~31 min.
PER_QUERY_TIMEOUT_S = 1800

PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

GPT41_COST_PER_REPORT_USD = 0.08   # standard-endpoint gpt-4.1 synthesis (E9 const)
GPT52_COST_PER_REPORT_USD = 0.08   # GPT-5.2 judge, per namespaced runner estimate


# ── Path safety (verbatim from E9) ────────────────────────────────────────────
def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_relative_to_lex(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_safe_out(raw: str, label: str) -> Path:
    candidate_lex = Path(raw)
    if not candidate_lex.is_absolute():
        candidate_lex = _REPO_ROOT / candidate_lex
    candidate = candidate_lex.resolve()
    for prot in PROTECTED_PATHS:
        for protected in {prot, prot.resolve()}:
            if candidate == protected or candidate_lex == protected:
                raise SystemExit(
                    f"REFUSING: {label} resolves to protected corpus path "
                    f"{protected}. This harness must NEVER write there. Choose a NEW dir."
                )
            if _is_relative_to(candidate, protected) or _is_relative_to_lex(candidate_lex, protected):
                raise SystemExit(
                    f"REFUSING: {label} ({candidate}) is INSIDE protected path "
                    f"{protected}. Choose a top-level dir outside all protected paths."
                )
            if _is_relative_to(protected, candidate) or _is_relative_to_lex(protected, candidate_lex):
                raise SystemExit(
                    f"REFUSING: {label} ({candidate}) is a PARENT of protected path "
                    f"{protected}; a run rooted there could traverse into the corpus. "
                    f"Choose a sibling dir."
                )
    return candidate


# ── Deterministic stratified query selection ──────────────────────────────────
def _load_eval() -> dict:
    data = json.loads(EVAL_QUERIES.read_text())
    items = data["queries"] if isinstance(data, dict) else data
    return {q["id"]: q for q in items}


def oracle_query_ids() -> list[str]:
    """The query_ids the pooled oracle corpus actually covers (>=1 doc), sorted."""
    oc = json.loads(ORACLE_CORPUS.read_text())
    qmap = _load_eval()
    ids = [qid for qid, docs in oc.items() if docs and qid in qmap]
    return sorted(ids)


def select_subset() -> tuple[list[str], list[str]]:
    """Return (base_ids[45], oracle_ids[30]).

    oracle_ids = exactly the oracle-corpus-covered queries.
    base_ids   = oracle_ids + 15 more, drawn STRATIFIED-by-difficulty and
                 deterministically (seed=SEED) from the remaining eval queries,
                 in proportion to the remaining pool's difficulty mix.
    """
    qmap = _load_eval()
    oracle_ids = oracle_query_ids()
    oracle_set = set(oracle_ids)
    need = SUBSET_SIZE - len(oracle_ids)
    if need <= 0:
        return sorted(oracle_ids)[:SUBSET_SIZE], oracle_ids

    remaining = [qid for qid in qmap if qid not in oracle_set]
    by_diff: dict[str, list[str]] = collections.defaultdict(list)
    for qid in remaining:
        by_diff[qmap[qid]["difficulty"]].append(qid)
    for d in by_diff:
        by_diff[d].sort()  # stable base order

    # proportional allocation across difficulties present in the remaining pool
    pool_total = len(remaining)
    rng = random.Random(SEED)
    alloc: dict[str, int] = {}
    # largest-remainder apportionment so the 15 split matches the pool mix
    quotas = {d: need * len(ids) / pool_total for d, ids in by_diff.items()}
    floors = {d: int(q) for d, q in quotas.items()}
    assigned = sum(floors.values())
    rema = sorted(by_diff, key=lambda d: (-(quotas[d] - floors[d]), d))
    for d in rema:
        if assigned >= need:
            break
        floors[d] += 1
        assigned += 1
    alloc = {d: min(n, len(by_diff[d])) for d, n in floors.items()}

    extra: list[str] = []
    for d, n in alloc.items():
        picks = by_diff[d][:]
        rng.shuffle(picks)
        extra.extend(picks[:n])
    # top up if any difficulty was short
    if len(extra) < need:
        leftover = [q for q in remaining if q not in set(extra)]
        leftover.sort()
        rng.shuffle(leftover)
        extra.extend(leftover[: need - len(extra)])

    base_ids = sorted(set(oracle_ids) | set(extra[:need]))
    return base_ids, oracle_ids


# ── Backbone-pinned import (E9 mechanism) ─────────────────────────────────────
def import_pattern_pinned(arch: str):
    """Pin DEFAULT_MODEL=gpt-4.1, purge config/pattern modules, re-import, verify."""
    if arch not in PATTERNS:
        raise SystemExit(f"Unknown arch {arch!r}. Known: {sorted(PATTERNS)}")
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


# ── Cell paths ────────────────────────────────────────────────────────────────
def _result_path(results_root: Path, arm: str, qid: str) -> Path:
    safe = str(qid).replace("/", "_").replace("\\", "_")
    return results_root / arm / f"{safe}.md"


def _ckpt_path(ckpt_root: Path, arm: str, qid: str) -> Path:
    safe = str(qid).replace("/", "_").replace("\\", "_")
    return ckpt_root / arm / f"{safe}.json"


# ── Bulletproof hard per-query wall-clock guard (SIGALRM backstop) ────────────
# asyncio.wait_for cannot interrupt a SYNCHRONOUS blocking call (e.g. a wedged
# url fetch running in a thread executor) — the await is cancelled but the thread
# runs on, so the per-query timeout never actually fires and the run hangs. The
# url_extractor now bounds each fetch, but as a belt-and-braces backstop we also
# arm a SIGALRM on the main thread: if a single query exceeds HARD_QUERY_S of
# wall-clock, the alarm raises into whatever is running (sync or async) and the
# query is abandoned + recorded as a timeout (resumable), instead of wedging the
# whole multi-hour run. Fires LATER than PER_QUERY_TIMEOUT_S so the graceful
# asyncio timeout gets first chance; this only catches a genuine sync wedge.
HARD_QUERY_S = PER_QUERY_TIMEOUT_S + 90


class _HardQueryTimeout(Exception):
    """Raised by SIGALRM when a single query blows the hard wall-clock backstop."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _HardQueryTimeout(f"hard wall-clock backstop {HARD_QUERY_S}s exceeded")


def is_completed(ckpt_root: Path, arm: str, qid: str) -> bool:
    cp = _ckpt_path(ckpt_root, arm, qid)
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "success"
    except Exception:
        return False


# ── Non-empty-body guard ──────────────────────────────────────────────────────
# A genuine research report has a substantive body. Content-filter REFUSALS and
# near-empty generations produce a title-only .md (the H1 == the query prompt
# echoed back, no sections) whose text is still non-empty, so the naive save path
# wrote it as status=="success" and --resume skipped it forever -> it silently
# scored ~0 and wrecked the arm's baseline. report_body_ok() lets callers detect
# this: on failure they write a .failed.json marker (not a success .md), so
# --resume retries the query and a persistently-empty query is a genuine refusal
# to flag for exclusion rather than score as a zero.
MIN_BODY_WORDS = 50


def _strip_leading_h1(text: str) -> str:
    """Drop the first Markdown H1 (report title / echoed query) so the guard
    measures the BODY, not the prompt a refusal hands straight back."""
    out, dropped = [], False
    for ln in text.splitlines():
        if not dropped and ln.lstrip().startswith("# "):
            dropped = True
            continue
        out.append(ln)
    return "\n".join(out)


def report_body_ok(text: str, report) -> tuple[bool, str]:
    """Return (ok, reason). A report FAILS the guard when its body (title
    stripped) has fewer than MIN_BODY_WORDS words OR the pipeline produced no
    sections -- both hallmarks of a content-filter refusal / empty generation.
    Every genuine report in the corpus carries >=1 section and thousands of body
    words, so the thresholds have a wide safety margin."""
    body_words = len(_strip_leading_h1(text).split())
    n_sections = len(getattr(report, "sections", None) or [])
    if body_words < MIN_BODY_WORDS:
        return False, f"body_only_{body_words}w(<{MIN_BODY_WORDS})"
    if n_sections == 0:
        return False, f"no_sections(body_{body_words}w)"
    return True, "ok"


# ── Per-arm generation ────────────────────────────────────────────────────────
async def generate_arm(
    arm: str,
    arch: str,
    oracle: bool,
    qids: list[str],
    qmap: dict,
    results_root: Path,
    ckpt_root: Path,
    budget: float,
    resume: bool,
) -> dict:
    mod = import_pattern_pinned(arch)

    # wire / unwire the oracle backend for this whole arm
    if oracle:
        os.environ["SEARCH_BACKEND"] = "oracle"
        os.environ["ORACLE_CORPUS_PATH"] = str(ORACLE_CORPUS)
        os.environ["ORACLE_MAX_DOCS"] = os.environ.get("ORACLE_MAX_DOCS", "30")
        import deep_research.config as _cfg
        _cfg.SEARCH_BACKEND = "oracle"
    else:
        os.environ.pop("SEARCH_BACKEND", None)
        os.environ.pop("ORACLE_QUERY_ID", None)

    n_ok = n_fail = n_skip = 0
    print(f"\n### ARM {arm}  arch={arch}  backbone={BACKBONE}  "
          f"oracle={oracle}  queries={len(qids)} ###", flush=True)

    for qid in qids:
        if resume and is_completed(ckpt_root, arm, qid):
            n_skip += 1
            continue
        q = qmap[qid]
        rp = _result_path(results_root, arm, qid)
        cp = _ckpt_path(ckpt_root, arm, qid)
        rp.parent.mkdir(parents=True, exist_ok=True)
        cp.parent.mkdir(parents=True, exist_ok=True)

        if oracle:
            os.environ["ORACLE_QUERY_ID"] = qid

        t0 = time.time()
        _alarm_armed = False
        try:
            # Arm the hard SIGALRM backstop (main-thread only; asyncio.run uses it).
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
                signal.alarm(0)  # disarm: query completed within the backstop
                _alarm_armed = False
            text = report.full_text()
            if not text.strip():
                text = f"# (empty report)\n\nQuery: {q['query']}\n"
            rp.write_text(text)
            elapsed = time.time() - t0
            import deep_research.config as _cfg
            meta = {
                "experiment": "GPT41_SECOND_BACKBONE",
                "arm": arm,
                "architecture": arch,
                "oracle": oracle,
                "backbone": BACKBONE,
                "search_model": _cfg.SEARCH_MODEL,
                "search_backend": _cfg.SEARCH_BACKEND if oracle else "live",
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
                  f"{meta['total_tokens']}tok (model={meta['default_model_bound']})",
                  flush=True)
        except (asyncio.TimeoutError, _HardQueryTimeout, Exception) as e:  # noqa: BLE001
            elapsed = time.time() - t0
            if isinstance(e, (asyncio.TimeoutError, _HardQueryTimeout)):
                status = "timeout"
            else:
                status = "error"
            meta = {
                "experiment": "GPT41_SECOND_BACKBONE",
                "arm": arm, "architecture": arch, "oracle": oracle,
                "backbone": BACKBONE, "query": qid, "status": status,
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e)[:300] or "per-query timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cp.write_text(json.dumps(meta, indent=2, default=str))
            n_fail += 1
            print(f"    FAIL {qid[:28]}: {status} {meta['elapsed_seconds']}s "
                  f"— {meta['error'][:120]}", flush=True)
        finally:
            if _alarm_armed:
                signal.alarm(0)  # always disarm before next query

    return {"arm": arm, "ok": n_ok, "fail": n_fail, "skip": n_skip}


# ── Planning / costing ────────────────────────────────────────────────────────
def build_plan(base_ids, oracle_ids):
    """[(arm, arch, oracle, qids)] for the 3 arms."""
    plan = []
    for arm, (arch, oracle) in ARMS.items():
        qids = oracle_ids if oracle else base_ids
        plan.append((arm, arch, oracle, qids))
    return plan


def _print_judge_and_build(results_root: Path):
    arms = ",".join(ARMS.keys())
    print("\n  Now judge (GPT-5.2, corpus-safe namespaced runner):")
    print(f"    JUDGE_RESULTS_BASE={DEFAULT_RESULTS_ROOT} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} \\")
    print(f"      --patterns-raw {arms} --resume --concurrency 3")
    print("\n  Then build the replication result (POST-judge step):")
    print(f"    python scripts/build_second_backbone.py "
          f"--judge-out {JUDGE_OUT_ROOT}   # key '{BUILD_KEY}'")
    print("\n  Authoritative judge stays GPT-5.2 (JUDGE_MODEL=gpt-5.2); gpt-4.1 is")
    print("  the generation BACKBONE only, never the judge -> judge-independence holds.")


# ── Main ──────────────────────────────────────────────────────────────────────
async def amain():
    ap = argparse.ArgumentParser(
        description="GPT-4.1 second-backbone robustness arm (corpus-safe generation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + cost + judge/build commands. ZERO API calls.")
    ap.add_argument("--smoke", action="store_true",
                    help="Generate 1 query per arm (paid, tiny) to confirm the "
                         "gpt-4.1 backbone binds and reports write.")
    ap.add_argument("--run", action="store_true",
                    help="Full 45-query run (paid PTU/Azure). Resumes by default.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap queries per arm (debug). 0 = full subset.")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    ap.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--self-test", action="store_true",
                    help="In-process unit checks (no API). Exit 0/1.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    results_root = resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = resolve_safe_out(args.checkpoint_root, "--checkpoint-root")

    qmap = _load_eval()
    base_ids, oracle_ids = select_subset()
    if args.smoke:
        base_ids, oracle_ids = base_ids[:1], oracle_ids[:1]
    elif args.limit:
        base_ids, oracle_ids = base_ids[:args.limit], oracle_ids[:args.limit]

    plan = build_plan(base_ids, oracle_ids)
    total_reports = sum(len(qids) for _, _, _, qids in plan)
    gen_usd = total_reports * GPT41_COST_PER_REPORT_USD
    judge_usd = total_reports * GPT52_COST_PER_REPORT_USD

    diff = {qid: qmap[qid]["difficulty"] for qid in qmap}
    base_dist = dict(collections.Counter(diff[i] for i in base_ids))
    oracle_dist = dict(collections.Counter(diff[i] for i in oracle_ids))

    print("=" * 74)
    print("GPT-4.1 SECOND-BACKBONE ROBUSTNESS ARM")
    print("=" * 74)
    print(f"  Backbone (IV, ONLY change vs corpus): DEFAULT_MODEL={BACKBONE}")
    print(f"  SEARCH_MODEL pinned to corpus value:  {CORPUS_SEARCH_MODEL} (deployed; untouched)")
    print(f"  Best pipeline (canonical headline):   {BEST_PIPELINE} (base_p4 0.46655 >= base_p1 0.46648)")
    print(f"  Arms: {', '.join(ARMS.keys())}")
    print(f"    (a) bounded gain  : {BEST_PIPELINE}_base vs p0_base  (45 queries)")
    print(f"    (b) oracle bottleneck: {BEST_PIPELINE}_oracle vs {BEST_PIPELINE}_base (matched 30)")
    print(f"  base subset: {len(base_ids)} queries  difficulty={base_dist}")
    print(f"  oracle subset: {len(oracle_ids)} queries  difficulty={oracle_dist}")
    print(f"  WRITE results root (NEW): {results_root}")
    print(f"  WRITE checkpoint root (NEW): {ckpt_root}")
    print(f"  total reports: {total_reports}")
    print(f"  est. gpt-4.1 generation cost: ${gen_usd:.2f} (standard endpoint)")
    print(f"  est. GPT-5.2 judge cost (post): ${judge_usd:.2f}")
    for arm, arch, oracle, qids in plan:
        print(f"    [{arm}] arch={arch} oracle={oracle} reports={len(qids)}")

    if args.dry_run:
        print("\n  [DRY RUN] No API calls made, nothing written.")
        _print_judge_and_build(results_root)
        return

    if not (args.smoke or args.run):
        print("\n  Nothing to do. Pass --dry-run, --smoke, or --run.")
        _print_judge_and_build(results_root)
        return

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE (1/arm)" if args.smoke else "FULL"
    print(f"\n  >>> {mode} GENERATION starting (backbone={BACKBONE}) <<<\n")

    summary = []
    for arm, arch, oracle, qids in plan:
        res = await generate_arm(
            arm=arm, arch=arch, oracle=oracle, qids=qids, qmap=qmap,
            results_root=results_root, ckpt_root=ckpt_root,
            budget=args.budget, resume=args.resume,
        )
        summary.append(res)

    print("\n" + "=" * 74)
    print(f"  GPT-4.1 BACKBONE GENERATION COMPLETE ({mode})")
    print("=" * 74)
    for r in summary:
        print(f"  [{r['arm']}] ok={r['ok']} fail={r['fail']} skip={r['skip']}")
    total_ok = sum(r["ok"] for r in summary)
    total_fail = sum(r["fail"] for r in summary)
    print(f"  TOTAL ok={total_ok} fail={total_fail}")
    if args.smoke:
        if total_ok == len(plan):
            print(f"\n  SMOKE OK: gpt-4.1 backbone bound + all {len(plan)} arms generated.")
        else:
            print(f"\n  SMOKE INCOMPLETE: {total_ok}/{len(plan)} arms produced a report. "
                  f"Inspect checkpoints under {ckpt_root}.")
    _print_judge_and_build(results_root)


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    base_ids, oracle_ids = select_subset()
    check("oracle subset == 30 (corpus coverage)", len(oracle_ids) == 30)
    check("base subset == 45", len(base_ids) == SUBSET_SIZE)
    check("oracle subset subset-of base", set(oracle_ids).issubset(set(base_ids)))
    check("deterministic selection", select_subset()[0] == base_ids)
    qmap = _load_eval()
    diff = {q: qmap[q]["difficulty"] for q in qmap}
    base_dist = collections.Counter(diff[i] for i in base_ids)
    check("base subset spans >=2 difficulty levels", len(base_dist) >= 2)
    check("3 arms", len(ARMS) == 3)
    check("best pipeline is p4", BEST_PIPELINE == "p4")
    check("p4_oracle arm present + oracle flag", ARMS.get("p4_oracle") == ("p4", True))
    # safety guards
    for bad in ["results/experiments", "results/judge_gpt52", "data/analysis", "results"]:
        try:
            resolve_safe_out(bad, "--results-root")
            check(f"refuse protected {bad}", False)
        except SystemExit:
            check(f"refuse protected {bad}", True)
    try:
        p = resolve_safe_out(DEFAULT_RESULTS_ROOT, "--results-root")
        check("accept NEW backbone root", p.name == "experiments_gpt41_backbone")
    except SystemExit:
        check("accept NEW backbone root", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
