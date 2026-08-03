#!/usr/bin/env python3
"""FROZEN-EVIDENCE DEFENCE 1 — SYNTHESIS-ARCHITECTURE ABLATION.

Rebuts the Argus / TTD-DR / GRACE claim that a cleverer WRITING scaffold lifts
factual accuracy. We hand the SAME frozen oracle evidence (identical numbered
block, byte-for-byte) to five different synthesis scaffolds, all on gpt-4o-mini,
and let the later GPT-5.2 + panel judging test whether ANY wiring moves factual
accuracy when the evidence is held constant.

Because retrieval AND per-doc extraction are removed (every scaffold consumes one
pre-built evidence block from the frozen corpus), the ONLY thing that varies is the
synthesis wiring — this is the clean identity test the reviewers ask for.

FIVE SCAFFOLDS (all gpt-4o-mini, identical evidence):
  single_pass     one-shot report            (the P0 synthesis step)
  draft_revise    draft -> self-critique -> revise        (TTD-DR-style loop)
  map_reduce      per-group digest -> reduce to report     (Argus-style modular)
  beam            B drafts -> self-score -> refine winner  (P8 explore-then-exploit)
  verifier_select B drafts -> c0 entailment vfa -> pick best (verifier-in-the-loop)

BACKBONE: gpt-4o-mini, routed to SEARCH_OPENAI_ENDPOINT (see frozen_defence_common).
EVIDENCE: data/oracle_corpus_t1.json, the 30 variance-stratified oracle query_ids.
WRITES (all NEW, corpus-safe):
  results/experiments_synthablation/<scaffold>/<qid>.md      (scorable report)
  checkpoints/synthablation/<scaffold>/<qid>.json            (resume + metrics)
JUDGING is a LATER step (GPT-5.2 namespaced + panel); this script GENERATES only.

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/run_synth_ablation.py --self-test          # no API
  python scripts/run_synth_ablation.py --dry-run            # plan, no API
  python scripts/run_synth_ablation.py --smoke --limit 1    # 1 query x 5 scaffolds (paid, tiny)
  python scripts/run_synth_ablation.py --run                # full oracle-30 x 5 scaffolds
  python scripts/run_synth_ablation.py --run --scaffolds single_pass,map_reduce
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import frozen_defence_common as fd

_REPO_ROOT = Path(__file__).resolve().parent.parent

SCAFFOLDS = ["single_pass", "draft_revise", "map_reduce", "beam", "verifier_select"]
DEFAULT_RESULTS_ROOT = "results/experiments_synthablation"
DEFAULT_CKPT_ROOT = "checkpoints/synthablation"
JUDGE_OUT_ROOT = "results/judge_gpt52_synthablation"

BUDGET_USD_DEFAULT = 3.0     # gpt-4o-mini; a scaffold report lands well under $0.10
BEAM_B = 3                   # candidate drafts for beam / verifier_select
VERIFY_MAX_CLAIMS = 10       # c0 claims per candidate in verifier_select (cost cap)
MODEL = fd.BACKBONE
PER_QUERY_TIMEOUT_S = 900


# ── Prompts ───────────────────────────────────────────────────────────────────
def _single_pass_prompt(query, evidence):
    return (
        "You are a research analyst. Write a comprehensive, well-structured research "
        f"report answering the query below, using ONLY the provided source evidence.\n\n"
        f"Research query: {query}\n\nSource evidence:\n{evidence}\n\n{fd.REPORT_CONTRACT}\n\n"
        "Write the full research report:"
    )


def _draft_prompt(query, evidence):
    return (
        "Write a first-draft research report answering the query using ONLY the source "
        f"evidence. Cite with [N].\n\nQuery: {query}\n\nEvidence:\n{evidence}\n\n"
        f"{fd.REPORT_CONTRACT}\n\nDraft report:"
    )


def _critique_prompt(query, evidence, draft):
    return (
        "You are a meticulous fact-checker. Given the source evidence and a draft report, "
        "list the draft's specific weaknesses: (a) claims NOT supported by the evidence, "
        "(b) important evidence points the draft omitted, (c) citations that do not match "
        "the evidence they cite. Be concrete and terse (bullet list).\n\n"
        f"Query: {query}\n\nEvidence:\n{evidence}\n\nDraft:\n{draft}\n\nWeaknesses:"
    )


def _revise_prompt(query, evidence, draft, critique):
    return (
        "Revise the draft into a final research report that fixes every weakness listed, "
        "using ONLY the source evidence. Remove unsupported claims, add the omitted "
        "evidence, and correct citations.\n\n"
        f"Query: {query}\n\nEvidence:\n{evidence}\n\nDraft:\n{draft}\n\n"
        f"Weaknesses to fix:\n{critique}\n\n{fd.REPORT_CONTRACT}\n\nFinal revised report:"
    )


def _map_prompt(query, group_block):
    return (
        "Summarise the following subset of sources into a dense, faithful digest of the "
        "facts and findings relevant to the query. Preserve the [N] citation numbers "
        "exactly as given.\n\n"
        f"Query: {query}\n\nSources:\n{group_block}\n\nDigest (with [N] citations preserved):"
    )


def _reduce_prompt(query, digests, full_refs):
    return (
        "Synthesise the per-group digests below into one comprehensive research report "
        "answering the query. Use ONLY facts from the digests; keep the [N] citation "
        "numbers consistent.\n\n"
        f"Query: {query}\n\nDigests:\n{digests}\n\nCitation key (for the References "
        f"section):\n{full_refs}\n\n{fd.REPORT_CONTRACT}\n\nFinal synthesised report:"
    )


def _score_prompt(query, evidence, candidate):
    return (
        "Rate this candidate research report against the source evidence on two axes, "
        "0-10 each: FAITHFULNESS (every claim supported by the evidence) and COVERAGE "
        "(uses the important evidence). Respond with strict JSON "
        '{"faithfulness": <int>, "coverage": <int>}.\n\n'
        f"Query: {query}\n\nEvidence:\n{evidence}\n\nCandidate:\n{candidate}\n\nJSON:"
    )


def _refine_prompt(query, evidence, winner):
    return (
        "Improve the report below into its final form: tighten structure, remove any "
        "claim not supported by the evidence, and ensure every section is well cited. "
        "Use ONLY the source evidence.\n\n"
        f"Query: {query}\n\nEvidence:\n{evidence}\n\nReport:\n{winner}\n\n"
        f"{fd.REPORT_CONTRACT}\n\nFinal report:"
    )


# ── Scaffold implementations (each returns (markdown, extra_metrics)) ──────────
async def scaffold_single_pass(llm, query, docs, evidence):
    md = await llm.complete(_single_pass_prompt(query, evidence), model=MODEL,
                            max_tokens=8192, temperature=0.3)
    return md, {"llm_calls": 1}


async def scaffold_draft_revise(llm, query, docs, evidence):
    draft = await llm.complete(_draft_prompt(query, evidence), model=MODEL,
                               max_tokens=8192, temperature=0.4)
    critique = await llm.complete(_critique_prompt(query, evidence, draft), model=MODEL,
                                  max_tokens=1500, temperature=0.2)
    final = await llm.complete(_revise_prompt(query, evidence, draft, critique), model=MODEL,
                               max_tokens=8192, temperature=0.3)
    return final, {"llm_calls": 3, "critique_chars": len(critique)}


async def scaffold_map_reduce(llm, query, docs, evidence):
    group_size = 5
    groups = [docs[i:i + group_size] for i in range(0, len(docs), group_size)]
    digests = []
    for gi, g in enumerate(groups):
        # keep the ORIGINAL [N] numbering so citations stay globally consistent
        start = gi * group_size
        block = fd.evidence_block(g, max_chars=40_000)
        # re-number the block to the global indices
        block = _renumber_block(g, start)
        d = await llm.complete(_map_prompt(query, block), model=MODEL,
                               max_tokens=2000, temperature=0.3)
        digests.append(f"### Group {gi + 1}\n{d}")
    reduced = await llm.complete(
        _reduce_prompt(query, "\n\n".join(digests), fd.references_from_docs(docs)),
        model=MODEL, max_tokens=8192, temperature=0.3)
    return reduced, {"llm_calls": len(groups) + 1, "n_groups": len(groups)}


def _renumber_block(group_docs, start_idx):
    parts = []
    for j, d in enumerate(group_docs):
        i = start_idx + j + 1
        title = (d.get("title", "") or "").strip()
        url = (d.get("url", "") or "").strip()
        content = (d.get("content", "") or "")[:4000].strip()
        parts.append(f"[{i}] {title}\nURL: {url}\n{content}\n")
    return "\n".join(parts)


async def scaffold_beam(llm, query, docs, evidence, b=BEAM_B):
    cands = await asyncio.gather(*[
        llm.complete(_draft_prompt(query, evidence), model=MODEL, max_tokens=8192,
                     temperature=0.7)
        for _ in range(b)
    ])
    scores = []
    for c in cands:
        try:
            js = await llm.complete_json(_score_prompt(query, evidence, c), model=MODEL,
                                         max_tokens=200, temperature=0.0)
            s = float(js.get("faithfulness", 0)) + float(js.get("coverage", 0))
        except Exception:
            s = 0.0
        scores.append(s)
    winner = cands[max(range(len(cands)), key=lambda i: scores[i])]
    final = await llm.complete(_refine_prompt(query, evidence, winner), model=MODEL,
                               max_tokens=8192, temperature=0.3)
    return final, {"llm_calls": 2 * b + 1, "beam_scores": scores}


async def scaffold_verifier_select(llm, query, docs, evidence, b=BEAM_B):
    from deep_research.evaluation.c0_verifier import verify_report
    src = fd.sources_by_citation(docs)
    cands = await asyncio.gather(*[
        llm.complete(_draft_prompt(query, evidence), model=MODEL, max_tokens=8192,
                     temperature=0.5)
        for _ in range(b)
    ])
    vfas = []
    for ci, c in enumerate(cands):
        try:
            res = await verify_report(llm, "synthablation", f"cand{ci}", c, src,
                                      model=MODEL, max_claims=VERIFY_MAX_CLAIMS)
            vfas.append(res.verified_factual_accuracy)
        except Exception:
            vfas.append(0.0)
    winner_idx = max(range(len(cands)), key=lambda i: vfas[i])
    return cands[winner_idx], {"llm_calls": b + b * (VERIFY_MAX_CLAIMS + 4),
                               "candidate_vfa": vfas, "winner_vfa": vfas[winner_idx]}


SCAFFOLD_FNS = {
    "single_pass": scaffold_single_pass,
    "draft_revise": scaffold_draft_revise,
    "map_reduce": scaffold_map_reduce,
    "beam": scaffold_beam,
    "verifier_select": scaffold_verifier_select,
}


# ── Cell IO ───────────────────────────────────────────────────────────────────
def _rp(results_root, scaffold, qid):
    return results_root / scaffold / f"{qid}.md"


def _cp(ckpt_root, scaffold, qid):
    return ckpt_root / scaffold / f"{qid}.json"


def is_done(ckpt_root, scaffold, qid):
    cp = _cp(ckpt_root, scaffold, qid)
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "success"
    except Exception:
        return False


async def run_cell(scaffold, qid, query, docs, evidence, results_root, ckpt_root, budget):
    rp, cp = _rp(results_root, scaffold, qid), _cp(ckpt_root, scaffold, qid)
    rp.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    llm, tracker = fd.make_llm(budget)
    t0 = time.time()
    armed = False
    try:
        try:
            signal.signal(signal.SIGALRM, fd._alarm_handler)
            signal.alarm(PER_QUERY_TIMEOUT_S + 90)
            armed = True
        except (ValueError, AttributeError):
            armed = False
        md, extra = await asyncio.wait_for(
            SCAFFOLD_FNS[scaffold](llm, query, docs, evidence), timeout=PER_QUERY_TIMEOUT_S)
        if armed:
            signal.alarm(0)
            armed = False
        md = fd.ensure_references(md.strip() or f"# (empty)\n\nQuery: {query}\n", docs)
        rp.write_text(md)
        elapsed = time.time() - t0
        meta = {
            "experiment": "SYNTH_ABLATION", "scaffold": scaffold, "query": qid,
            "backbone": MODEL, "search_backend": "oracle_frozen", "status": "success",
            "elapsed_seconds": round(elapsed, 1), "chars": len(md),
            "total_tokens": tracker.total_tokens, "total_cost_usd": round(tracker.total_cost, 6),
            "evidence_chars": len(evidence), "n_docs": len(docs),
            "timestamp": datetime.now(timezone.utc).isoformat(), **extra,
        }
        cp.write_text(json.dumps(meta, indent=2, default=str))
        print(f"    OK   {scaffold:16s} {qid[:26]:26s} {elapsed:6.1f}s "
              f"{len(md):6d}ch {tracker.total_tokens:8d}tok ${tracker.total_cost:.5f}", flush=True)
        return meta
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        meta = {"experiment": "SYNTH_ABLATION", "scaffold": scaffold, "query": qid,
                "status": "error", "elapsed_seconds": round(elapsed, 1),
                "error": str(e)[:300], "total_cost_usd": round(tracker.total_cost, 6),
                "timestamp": datetime.now(timezone.utc).isoformat()}
        cp.write_text(json.dumps(meta, indent=2, default=str))
        print(f"    FAIL {scaffold:16s} {qid[:26]:26s} {elapsed:6.1f}s -- {str(e)[:110]}", flush=True)
        return meta
    finally:
        if armed:
            signal.alarm(0)


def _print_judge(results_root, scaffolds):
    raw = ",".join(scaffolds)
    print("\n  Judge later (GPT-5.2, corpus-safe namespaced runner):")
    print(f"    JUDGE_RESULTS_BASE={DEFAULT_RESULTS_ROOT} \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print(f"      --judge-out {JUDGE_OUT_ROOT} --patterns-raw {raw} --resume --concurrency 3")


async def amain():
    ap = argparse.ArgumentParser(description="Synthesis-architecture ablation (frozen evidence, gpt-4o-mini).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Run --limit queries (default 1), paid+tiny.")
    ap.add_argument("--run", action="store_true", help="Full oracle-30 x selected scaffolds.")
    ap.add_argument("--scaffolds", default=",".join(SCAFFOLDS))
    ap.add_argument("--limit", type=int, default=0, help="Cap queries (0=all 30; smoke defaults to 1).")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--checkpoint-root", default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    scaffolds = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
    bad = [s for s in scaffolds if s not in SCAFFOLD_FNS]
    if bad:
        raise SystemExit(f"Unknown scaffolds {bad}. Known: {SCAFFOLDS}")

    results_root = fd.resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = fd.resolve_safe_out(args.checkpoint_root, "--checkpoint-root")

    fd.route_and_pin()
    fd.assert_mini_bound()

    qmap = fd._load_eval()
    ids = fd.oracle_query_ids()
    if args.smoke and not args.limit:
        ids = ids[:1]
    elif args.limit:
        ids = ids[:args.limit]

    total = len(scaffolds) * len(ids)
    print("=" * 74)
    print("FROZEN-EVIDENCE DEFENCE 1 — SYNTHESIS-ARCHITECTURE ABLATION")
    print("=" * 74)
    print(f"  backbone: {MODEL} (routed to SEARCH_OPENAI_ENDPOINT); evidence: frozen oracle-30")
    print(f"  scaffolds ({len(scaffolds)}): {', '.join(scaffolds)}")
    print(f"  queries: {len(ids)}   total reports: {total}")
    print(f"  WRITE results (NEW): {results_root}")
    print(f"  WRITE ckpts   (NEW): {ckpt_root}")

    if args.dry_run:
        pend = sum(1 for s in scaffolds for q in ids if not (args.resume and is_done(ckpt_root, s, q)))
        print(f"\n  [DRY RUN] pending after resume: {pend}/{total}. ZERO API calls.")
        _print_judge(results_root, scaffolds)
        return
    if not (args.smoke or args.run):
        print("\n  Nothing to do. Pass --dry-run, --smoke, or --run.")
        _print_judge(results_root, scaffolds)
        return

    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    corpus = fd.load_corpus()
    mode = "SMOKE" if args.smoke else "FULL"
    print(f"\n  >>> {mode} GENERATION (backbone={MODEL}) <<<\n")

    n_ok = n_fail = n_skip = 0
    for scaffold in scaffolds:
        print(f"### scaffold {scaffold} ###", flush=True)
        for qid in ids:
            if args.resume and is_done(ckpt_root, scaffold, qid):
                n_skip += 1
                continue
            docs = fd.frozen_docs(qid, corpus)
            evidence = fd.evidence_block(docs)
            meta = await run_cell(scaffold, qid, qmap[qid]["query"], docs, evidence,
                                  results_root, ckpt_root, args.budget)
            if meta.get("status") == "success":
                n_ok += 1
            else:
                n_fail += 1

    print("\n" + "=" * 74)
    print(f"  DONE ({mode}) ok={n_ok} fail={n_fail} skip={n_skip}")
    if args.smoke and n_ok >= 1:
        print(f"  SMOKE OK: {n_ok} scorable report(s) under {results_root}")
    _print_judge(results_root, scaffolds)


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    check("5 scaffolds mapped", set(SCAFFOLDS) == set(SCAFFOLD_FNS))
    ids = fd.oracle_query_ids()
    check("oracle-30 ids", len(ids) == 30)
    corpus = fd.load_corpus()
    docs = fd.frozen_docs(ids[0], corpus)
    ev = fd.evidence_block(docs)
    check("evidence block non-empty + numbered", ev.startswith("[1]") and len(ev) > 500)
    check("identical evidence across scaffolds (pure fn)", fd.evidence_block(docs) == ev)
    check("references block parses", "## References" in fd.references_from_docs(docs))
    for bad in ["results/experiments", "results/judge_gpt52", "data/analysis", "results"]:
        try:
            fd.resolve_safe_out(bad, "x"); check(f"refuse {bad}", False)
        except SystemExit:
            check(f"refuse {bad}", True)
    try:
        p = fd.resolve_safe_out(DEFAULT_RESULTS_ROOT, "x")
        check("accept NEW root", p.name == "experiments_synthablation")
    except SystemExit:
        check("accept NEW root", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    asyncio.run(amain())
