#!/usr/bin/env python3
"""FROZEN-EVIDENCE DEFENCE 3 — DISTRACTOR-DOSE.

Rebuts "cluster/orchestrated pipelines are robust because they filter noise". We
KEEP all the gold frozen evidence and INJECT off-topic distractor passages at
0 / 20 / 40 / 70% of the corpus, then re-run P0 (baseline) plus two cluster
pipelines (P1 iterative-RAG, P4 perspective-storm) on gpt-4o-mini. If factual
accuracy degrades with distractor dose at the SAME rate for the bare baseline and
the orchestrated pipelines, the orchestration buys no noise-robustness — another
crown-jewel point against the Argus/TTD-DR/GRACE "better wiring" story.

Distractors are REAL passages drawn from OTHER oracle queries' evidence (guaranteed
off-topic, real text), deduped against this query's gold URLs, injected and shuffled
in deterministically (seed=7). Gold is never removed — dose = distractor fraction of
the final set. This is DISTINCT from E5 (which varied the gold fraction).

  dose d  ->  n_distract = round(d/(1-d) * n_gold);  final = gold + distract, shuffled
  GOLD_BASE=15 gold docs/query, so d70 => 15 gold + 35 distract = 50 docs.

BACKBONE: gpt-4o-mini (routed to SEARCH_OPENAI_ENDPOINT). Frozen; NO live search.
WRITES (all NEW, corpus-safe):
  data/experiments_distractor/corpus/d{000,020,040,070}.json   (doped corpora)
  data/experiments_distractor/corpus/_stats.json
  results/experiments_distractor/<pattern>_d<NNN>/<qid>.md      (scorable reports)
  checkpoints/distractor/<pattern>_d<NNN>/<qid>.json            (resume + metrics)
JUDGING is a LATER step (GPT-5.2 namespaced + panel); this GENERATES only.

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/run_distractor_dose.py --self-test
  python scripts/run_distractor_dose.py --build-corpus-only     # local, free
  python scripts/run_distractor_dose.py --dry-run
  python scripts/run_distractor_dose.py --smoke --limit 1       # 1 query x 3 patterns x 4 doses (paid, tiny)
  python scripts/run_distractor_dose.py --run                   # full: 3 patterns x 4 doses x 30 queries
  python scripts/run_distractor_dose.py --run --patterns p0 --doses 0.0,0.7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import frozen_defence_common as fd

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "experiments_distractor"
CORPUS_DIR = DATA_DIR / "corpus"
RESULTS_ROOT = "results/experiments_distractor"
CKPT_ROOT = "checkpoints/distractor"
JUDGE_OUT_ROOT = "results/judge_gpt52_distractor"

PATTERNS = ["p0", "p1", "p4"]          # P0 baseline + 2 cluster pipelines
DOSES = [0.0, 0.20, 0.40, 0.70]
GOLD_BASE = 15                         # gold docs kept per query (all doses)
SEED = 7
BUDGET_USD_DEFAULT = 3.0
MODEL = fd.BACKBONE
PER_QUERY_TIMEOUT_S = 900


def dose_tag(d) -> str:
    return f"d{int(round(d * 100)):03d}"


def _n_distract(n_gold, dose):
    if dose <= 0:
        return 0
    return int(round(dose / (1.0 - dose) * n_gold))


def max_docs_for(doses) -> int:
    return max(GOLD_BASE + _n_distract(GOLD_BASE, d) for d in doses)


# ── Build doped corpora (local, free) ─────────────────────────────────────────
def build_corpora(ids, doses, verbose=True) -> dict:
    corpus = fd.load_corpus()
    # global off-topic pool: (url -> doc) from ALL queries, used as cross-topic noise
    pool_by_url = {}
    gold_urls_by_q = {}
    for qid in ids:
        gurls = set()
        for d in corpus.get(qid, [])[:GOLD_BASE]:
            u = (d.get("url", "") or "").strip()
            if u:
                gurls.add(u)
        gold_urls_by_q[qid] = gurls
    for qid, docs in corpus.items():
        for d in docs:
            u = (d.get("url", "") or "").strip()
            if u and u not in pool_by_url:
                pool_by_url[u] = (qid, d)

    out = {dose_tag(d): {} for d in doses}
    stats = {"gold_base": GOLD_BASE, "doses": doses, "n_queries": len(ids),
             "max_docs": max_docs_for(doses), "per_query": {}}
    for qid in ids:
        gold = corpus.get(qid, [])[:GOLD_BASE]
        n_gold = len(gold)
        # off-topic candidates = pool docs whose SOURCE query != qid and whose url
        # is not among this query's gold urls
        cand = [(u, d) for u, (src_q, d) in pool_by_url.items()
                if src_q != qid and u not in gold_urls_by_q[qid]]
        rnd = random.Random(f"{SEED}:{qid}")
        rnd.shuffle(cand)
        per = {}
        for d in doses:
            nd = min(_n_distract(n_gold, d), len(cand))
            distract = []
            for u, doc in cand[:nd]:
                dd = dict(doc)
                md = dict(dd.get("metadata", {}) or {})
                md["distractor"] = True
                dd["metadata"] = md
                distract.append(dd)
            gold_marked = []
            for doc in gold:
                dd = dict(doc)
                md = dict(dd.get("metadata", {}) or {})
                md["distractor"] = False
                dd["metadata"] = md
                gold_marked.append(dd)
            final = gold_marked + distract
            random.Random(f"{SEED}:{qid}:{d}").shuffle(final)
            out[dose_tag(d)][qid] = final
            per[dose_tag(d)] = {"n_gold": n_gold, "n_distract": nd, "total": len(final),
                                "distract_frac": round(nd / len(final), 3) if final else 0.0}
        stats["per_query"][qid] = per

    fd.resolve_safe_out(str(CORPUS_DIR), "corpus dir")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for tag, cmap in out.items():
        p = CORPUS_DIR / f"{tag}.json"
        p.write_text(json.dumps(cmap))
        written[tag] = str(p)
    stats["files"] = written
    (CORPUS_DIR / "_stats.json").write_text(json.dumps(stats, indent=2))
    if verbose:
        print(f"  [build] {len(ids)} queries x {len(doses)} doses  gold_base={GOLD_BASE} "
              f"max_docs={stats['max_docs']}")
        for d in doses:
            samp = next(iter(stats["per_query"].values()))[dose_tag(d)]
            print(f"    {dose_tag(d)}: gold={samp['n_gold']} distract={samp['n_distract']} "
                  f"total={samp['total']} frac~{samp['distract_frac']}")
        for tag, p in written.items():
            print(f"    wrote {tag} -> {p}")
    return stats


# ── Cell IO ───────────────────────────────────────────────────────────────────
def arm_name(pattern, dose):
    return f"{pattern}_{dose_tag(dose)}"


def _rp(results_root, arm, qid):
    return results_root / arm / f"{qid}.md"


def _cp(ckpt_root, arm, qid):
    return ckpt_root / arm / f"{qid}.json"


def is_done(ckpt_root, arm, qid):
    cp = _cp(ckpt_root, arm, qid)
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "success"
    except Exception:
        return False


async def _run_with_retry(mod, query, qid, budget, retries):
    """Run a pipeline, retrying transient failures (e.g. gpt-4o-mini malformed-JSON
    in P4/P-cluster structured prompts, which are stochastic). Timeouts are NOT
    retried (they would just re-block). Returns the report or re-raises."""
    last = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(
                mod.run(query, budget_usd=budget, query_id=qid), timeout=PER_QUERY_TIMEOUT_S)
        except (asyncio.TimeoutError, fd._HardQueryTimeout):
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                print(f"      retry {attempt + 1}/{retries} after: {str(e)[:80]}", flush=True)
    raise last


async def gen_pattern(pattern, doses, ids, qmap, results_root, ckpt_root, budget, resume,
                      max_docs, retries=1):
    mod = fd.import_pattern_pinned(pattern)   # pins gpt-4o-mini, pops oracle env
    fd.assert_mini_bound()
    n_ok = n_fail = n_skip = 0
    for dose in doses:
        arm = arm_name(pattern, dose)
        corpus_path = CORPUS_DIR / f"{dose_tag(dose)}.json"
        fd.wire_oracle(corpus_path, max_docs)
        print(f"\n### {arm}  (backbone={MODEL}, frozen+distractor, max_docs={max_docs}) ###", flush=True)
        for qid in ids:
            if resume and is_done(ckpt_root, arm, qid):
                n_skip += 1
                continue
            rp, cp = _rp(results_root, arm, qid), _cp(ckpt_root, arm, qid)
            rp.parent.mkdir(parents=True, exist_ok=True)
            cp.parent.mkdir(parents=True, exist_ok=True)
            fd.set_query(qid)
            t0 = time.time()
            armed = False
            try:
                try:
                    signal.signal(signal.SIGALRM, fd._alarm_handler)
                    signal.alarm(PER_QUERY_TIMEOUT_S + 90)
                    armed = True
                except (ValueError, AttributeError):
                    armed = False
                report = await _run_with_retry(mod, qmap[qid]["query"], qid, budget, retries)
                if armed:
                    signal.alarm(0); armed = False
                text = report.full_text()
                if not text.strip():
                    text = f"# (empty report)\n\nQuery: {qmap[qid]['query']}\n"
                rp.write_text(text)
                meta = {"experiment": "DISTRACTOR_DOSE", "pattern": pattern, "arm": arm,
                        "dose": dose, "query": qid, "backbone": MODEL, "status": "success",
                        "elapsed_seconds": round(time.time() - t0, 1), "chars": len(text),
                        "total_tokens": getattr(report, "total_tokens", 0),
                        "total_cost_usd": round(getattr(report, "total_cost_usd", 0.0), 6),
                        "citations": len(getattr(report, "citations", [])),
                        "sections": len(getattr(report, "sections", [])),
                        "timestamp": datetime.now(timezone.utc).isoformat()}
                cp.write_text(json.dumps(meta, indent=2, default=str))
                n_ok += 1
                print(f"    OK   {qid[:26]:26s} {meta['elapsed_seconds']:6.1f}s "
                      f"{meta['chars']:6d}ch {meta['citations']:3d}cit "
                      f"{meta['total_tokens']:8d}tok ${meta['total_cost_usd']:.5f}", flush=True)
            except Exception as e:  # noqa: BLE001
                meta = {"experiment": "DISTRACTOR_DOSE", "pattern": pattern, "arm": arm,
                        "dose": dose, "query": qid, "status": "error",
                        "elapsed_seconds": round(time.time() - t0, 1), "error": str(e)[:300],
                        "timestamp": datetime.now(timezone.utc).isoformat()}
                cp.write_text(json.dumps(meta, indent=2, default=str))
                n_fail += 1
                print(f"    FAIL {qid[:26]:26s} -- {str(e)[:110]}", flush=True)
            finally:
                if armed:
                    signal.alarm(0)
    return {"pattern": pattern, "ok": n_ok, "fail": n_fail, "skip": n_skip}


def _print_judge(patterns, doses):
    arms = ",".join(arm_name(p, d) for p in patterns for d in doses)
    print("\n  Judge later (GPT-5.2, corpus-safe namespaced runner):")
    print(f"    JUDGE_RESULTS_BASE={RESULTS_ROOT} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} --patterns-raw {arms} --resume --concurrency 3")


async def amain():
    ap = argparse.ArgumentParser(description="Distractor-dose robustness (frozen evidence, gpt-4o-mini).")
    ap.add_argument("--build-corpus-only", action="store_true", help="Build doped corpora (local, free).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Run --limit queries (default 1), paid+tiny.")
    ap.add_argument("--run", action="store_true", help="Full: patterns x doses x oracle-30.")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--doses", default=",".join(str(d) for d in DOSES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--checkpoint-root", default=CKPT_ROOT)
    ap.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT)
    ap.add_argument("--retries", type=int, default=2,
                    help="Retry a failed cell N times (transient gpt-4o-mini JSON errors, "
                         "notably P4 perspective-discovery). Default 2.")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    bad = [p for p in patterns if p not in PATTERNS]
    if bad:
        raise SystemExit(f"Unknown patterns {bad}. Known: {PATTERNS}")
    doses = [float(x) for x in args.doses.split(",") if x.strip()]

    results_root = fd.resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = fd.resolve_safe_out(args.checkpoint_root, "--checkpoint-root")
    fd.resolve_safe_out(str(DATA_DIR), "data corpus dir")

    fd.route_and_pin()
    fd.assert_mini_bound()

    qmap = fd._load_eval()
    ids = fd.oracle_query_ids()
    if args.smoke and not args.limit:
        ids = ids[:1]
    elif args.limit:
        ids = ids[:args.limit]

    md = max_docs_for(doses)
    total = len(patterns) * len(doses) * len(ids)
    print("=" * 74)
    print("FROZEN-EVIDENCE DEFENCE 3 — DISTRACTOR-DOSE")
    print("=" * 74)
    print(f"  backbone: {MODEL}; evidence: frozen oracle-30 + injected off-topic distractors")
    print(f"  patterns: {patterns}  doses: {doses}  gold_base={GOLD_BASE}  max_docs={md}")
    print(f"  queries: {len(ids)}   total reports: {total}")
    print(f"  WRITE corpora (NEW): {CORPUS_DIR}")
    print(f"  WRITE results (NEW): {results_root}")

    # BUILD corpora is always safe/local
    if args.build_corpus_only or args.dry_run or args.smoke or args.run:
        print("\n  [build] doped corpora ...")
        build_corpora(ids, doses)

    if args.build_corpus_only:
        print("\n  Corpora built (local I/O only). No API calls.")
        return
    if args.dry_run:
        pend = sum(1 for p in patterns for d in doses for q in ids
                   if not (args.resume and is_done(ckpt_root, arm_name(p, d), q)))
        print(f"\n  [DRY RUN] pending after resume: {pend}/{total}. ZERO generation API calls.")
        _print_judge(patterns, doses)
        return
    if not (args.smoke or args.run):
        print("\n  Nothing to do. Pass --build-corpus-only, --dry-run, --smoke, or --run.")
        _print_judge(patterns, doses)
        return

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    mode = "SMOKE" if args.smoke else "FULL"
    print(f"\n  >>> {mode} GENERATION (backbone={MODEL}) <<<")

    summary = []
    for pattern in patterns:
        res = await gen_pattern(pattern, doses, ids, qmap, results_root, ckpt_root,
                                args.budget, args.resume, md, args.retries)
        summary.append(res)

    print("\n" + "=" * 74)
    print(f"  DONE ({mode})")
    for r in summary:
        print(f"  [{r['pattern']}] ok={r['ok']} fail={r['fail']} skip={r['skip']}")
    total_ok = sum(r["ok"] for r in summary)
    if args.smoke and total_ok >= 1:
        print(f"  SMOKE OK: {total_ok} scorable report(s) under {results_root}")
    _print_judge(patterns, doses)


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    check("patterns = P0 + 2 cluster", patterns_ok := (PATTERNS == ["p0", "p1", "p4"]))
    check("doses 0/20/40/70", DOSES == [0.0, 0.20, 0.40, 0.70])
    check("n_distract(15,0.0)=0", _n_distract(15, 0.0) == 0)
    check("n_distract(15,0.7)=35", _n_distract(15, 0.70) == 35)
    check("n_distract(15,0.5)=15", _n_distract(15, 0.50) == 15)
    check("max_docs=50", max_docs_for(DOSES) == 50)
    # build a 2-query corpus and verify structure + determinism
    ids = fd.oracle_query_ids()[:2]
    s1 = build_corpora(ids, DOSES, verbose=False)
    tag70 = dose_tag(0.70)
    c70 = json.loads((CORPUS_DIR / f"{tag70}.json").read_text())
    q0 = ids[0]
    docs = c70[q0]
    n_gold = sum(1 for d in docs if not d.get("metadata", {}).get("distractor"))
    n_dis = sum(1 for d in docs if d.get("metadata", {}).get("distractor"))
    check("d70 has gold+distract", n_gold >= 1 and n_dis >= 1)
    check("d70 fraction ~0.70", abs(n_dis / len(docs) - 0.70) < 0.06)
    d0 = json.loads((CORPUS_DIR / f"{dose_tag(0.0)}.json").read_text())[q0]
    check("d0 has no distractors", all(not d.get("metadata", {}).get("distractor") for d in d0))
    # distractors are off-topic (drawn from other queries' urls, not gold urls)
    gold_urls = {(d.get("url") or "") for d in fd.frozen_docs(q0)[:GOLD_BASE]}
    dis_urls = {(d.get("url") or "") for d in docs if d.get("metadata", {}).get("distractor")}
    check("distractor urls disjoint from gold", gold_urls.isdisjoint(dis_urls))
    s2 = build_corpora(ids, DOSES, verbose=False)
    check("deterministic build", s1["per_query"] == s2["per_query"])
    for bad in ["results/experiments", "data/analysis", "results"]:
        try:
            fd.resolve_safe_out(bad, "x"); check(f"refuse {bad}", False)
        except SystemExit:
            check(f"refuse {bad}", True)
    try:
        check("accept NEW root", fd.resolve_safe_out(RESULTS_ROOT, "x").name == "experiments_distractor")
    except SystemExit:
        check("accept NEW root", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    asyncio.run(amain())
