#!/usr/bin/env python3
"""FROZEN-EVIDENCE DEFENCE 2 — COUNTERFACTUAL / OVERRIDE-RATE.

Rebuts "the model just faithfully synthesises whatever it retrieves". We edit the
frozen oracle evidence so it CONTRADICTS the model's likely prior — swap one true
date / number / name for a plausible counterfactual, everything else byte-identical
— then measure P(model reports its PRIOR | the evidence says otherwise). A high
override rate means factual accuracy is prior-bound, not evidence-bound, so no
retrieval/synthesis wiring can fix it (the crown-jewel point against Argus/GRACE).

PIPELINE (all gpt-4o-mini; frozen evidence; NO live search):
  build     For each oracle query, gpt-4o-mini proposes ONE salient factual atom
            {attribute, value_type, true_value, counterfactual_value, probe} that
            (a) appears verbatim in the evidence and (b) a trained model plausibly
            already knows. A RULE CHECK confirms the true value is really in the
            evidence, the counterfactual differs, and the swap edits >=1 span. The
            counterfactual corpus replaces every true_value span -> counterfactual
            (case-insensitive), leaving all other text identical.
  gen       Synthesise a full P0-style report over the CONTRADICTING evidence
            (scorable later by GPT-5.2 for evidence-faithfulness) AND ask a direct
            probe: "per the sources, what is <attribute>?".
  classify  Rule-detect whether the probe answer / report states the counterfactual
            (faithful to evidence) or the true value (prior OVERRIDE) or neither.
  override_rate = mean(prior_override) over valid queries.

WRITES (all NEW, corpus-safe):
  data/experiments_counterfactual/corpus_cf.json          (edited frozen corpus)
  data/experiments_counterfactual/manifest.json           (atoms + rule-check)
  results/experiments_counterfactual/reports/<qid>.md     (scorable report)
  results/experiments_counterfactual/probes/<qid>.json    (probe + classification)
  results/experiments_counterfactual/override_summary.json
  checkpoints/counterfactual/<qid>.json

USAGE:
  [ -f venv/bin/activate ] && source venv/bin/activate
  python scripts/run_counterfactual_override.py --self-test
  python scripts/run_counterfactual_override.py --dry-run
  python scripts/run_counterfactual_override.py --smoke --limit 2   # 2 queries end-to-end (paid, tiny)
  python scripts/run_counterfactual_override.py --run              # full oracle-30
  python scripts/run_counterfactual_override.py --stage classify   # re-score on-disk reports only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import frozen_defence_common as fd

_REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "experiments_counterfactual"
RESULTS_ROOT = "results/experiments_counterfactual"
CKPT_ROOT = "checkpoints/counterfactual"
JUDGE_OUT_ROOT = "results/judge_gpt52_counterfactual"

CORPUS_CF = DATA_DIR / "corpus_cf.json"
MANIFEST = DATA_DIR / "manifest.json"

BUDGET_USD_DEFAULT = 3.0
MODEL = fd.BACKBONE
PER_QUERY_TIMEOUT_S = 600


# ── Prompts ───────────────────────────────────────────────────────────────────
def _propose_prompt(query, evidence):
    return (
        "You are constructing a counterfactual robustness probe. From the source "
        "evidence below, choose ONE crisp factual atom that (a) appears VERBATIM in "
        "the evidence as a short string, (b) is the kind of canonical fact a "
        "well-trained language model would already know from pre-training (a specific "
        "year/date, a specific number/count, or a specific proper name), and (c) is "
        "central to answering the query. Then invent a PLAUSIBLE BUT FALSE value of "
        "the SAME type (a different year, a different number, or a different real name). "
        "Prefer short, unambiguous string values that are easy to find by exact match.\n\n"
        f"Query: {query}\n\nEvidence:\n{evidence[:60000]}\n\n"
        "Respond with STRICT JSON only:\n"
        '{"attribute": "<what the value denotes, <=12 words>", '
        '"value_type": "date|number|name", '
        '"true_value": "<exact substring from the evidence>", '
        '"counterfactual_value": "<plausible false value, same type>", '
        '"probe_question": "<a direct question whose answer is that value>"}'
    )


def _report_prompt(query, evidence):
    return (
        "You are a research analyst. Write a comprehensive research report answering the "
        "query using ONLY the provided source evidence. Where the sources state specific "
        "dates, numbers, or names, report them EXACTLY as the sources give them, and cite "
        f"with [N].\n\nResearch query: {query}\n\nSource evidence:\n{evidence}\n\n"
        f"{fd.REPORT_CONTRACT}\n\nWrite the full research report:"
    )


def _probe_prompt(attribute, probe_question, evidence):
    q = probe_question or f"What is the {attribute}?"
    return (
        "Answer the question using ONLY the source evidence below. The sources are "
        "authoritative for this task; do not correct them from your own knowledge. "
        "Reply with ONLY the value (a few words at most), nothing else.\n\n"
        f"Question: {q}\n\nSource evidence:\n{evidence[:60000]}\n\nValue:"
    )


# ── Build: propose atom + rule check + edit corpus ────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _count_occurrences(text: str, needle: str) -> int:
    if not needle:
        return 0
    return len(re.findall(re.escape(needle), text, flags=re.IGNORECASE))


def _apply_swap(docs, true_value, cf_value):
    """Replace every case-insensitive true_value span with cf_value in title+content.
    Returns (edited_docs, n_replacements)."""
    pat = re.compile(re.escape(true_value), flags=re.IGNORECASE)
    edited, n = [], 0
    for d in docs:
        d2 = dict(d)
        for field in ("title", "content"):
            val = d2.get(field, "") or ""
            new, k = pat.subn(cf_value, val)
            d2[field] = new
            n += k
        md = dict(d2.get("metadata", {}) or {})
        md["counterfactual_edited"] = True
        d2["metadata"] = md
        edited.append(d2)
    return edited, n


def validate_atom(atom, docs) -> tuple[bool, str]:
    tv = (atom or {}).get("true_value", "")
    cf = (atom or {}).get("counterfactual_value", "")
    if not tv or len(tv.strip()) < 2:
        return False, "true_value too short/empty"
    if not cf or len(cf.strip()) < 1:
        return False, "counterfactual empty"
    if _norm(tv) == _norm(cf):
        return False, "counterfactual equals true_value"
    joined = "\n".join((d.get("title", "") or "") + " " + (d.get("content", "") or "") for d in docs)
    if _count_occurrences(joined, tv) < 1:
        return False, "true_value not found verbatim in evidence"
    return True, "ok"


async def build_one(llm, qid, query, docs):
    evidence = fd.evidence_block(docs)
    atom = {}
    for attempt in range(2):
        try:
            atom = await llm.complete_json(_propose_prompt(query, evidence), model=MODEL,
                                           max_tokens=400, temperature=0.2 + 0.2 * attempt)
        except Exception as e:  # noqa: BLE001
            atom = {"error": str(e)[:120]}
        valid, reason = validate_atom(atom, docs)
        if valid:
            break
    valid, reason = validate_atom(atom, docs)
    if not valid:
        return {"query_id": qid, "valid": False, "reason": reason, "atom": atom}, None
    edited, n = _apply_swap(docs, atom["true_value"], atom["counterfactual_value"])
    cf_already = _count_occurrences(
        "\n".join((d.get("content", "") or "") for d in docs), atom["counterfactual_value"])
    entry = {
        "query_id": qid, "valid": n >= 1, "reason": "ok" if n >= 1 else "no spans edited",
        "attribute": atom.get("attribute", ""), "value_type": atom.get("value_type", ""),
        "true_value": atom["true_value"], "counterfactual_value": atom["counterfactual_value"],
        "probe_question": atom.get("probe_question", ""),
        "n_replacements": n, "cf_preexisting_in_evidence": cf_already,
    }
    return entry, edited


# ── Gen: report + probe over the CONTRADICTING evidence ───────────────────────
def _distinctive_cores(true_value, cf_value):
    """Tokens that DISTINGUISH the true value from the counterfactual, so a
    paraphrased answer ('1980s' for 'mid-1980s', 'Bahdanau' for 'D. Bahdanau')
    still classifies correctly. Prefer digit-runs (>=3 digits); else word tokens
    (len>=4). Returns (true_only, cf_only) as lowercased token sets."""
    t_dig = set(re.findall(r"\d{3,}", true_value))
    c_dig = set(re.findall(r"\d{3,}", cf_value))
    t_only, c_only = t_dig - c_dig, c_dig - t_dig
    if t_only or c_only:
        return t_only, c_only
    t_tok = {w for w in re.findall(r"[a-z0-9]{4,}", true_value.lower())}
    c_tok = {w for w in re.findall(r"[a-z0-9]{4,}", cf_value.lower())}
    return t_tok - c_tok, c_tok - t_tok


def _cls_one(text, true_value, cf_value):
    n = _norm(text)
    has_t = _norm(true_value) in n
    has_c = _norm(cf_value) in n
    if not (has_t or has_c):  # paraphrase fallback via distinctive cores
        t_only, c_only = _distinctive_cores(true_value, cf_value)
        a_dig = set(re.findall(r"\d{3,}", text))
        a_tok = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
        a_all = a_dig | a_tok
        has_t = bool(t_only) and bool(t_only & a_all)
        has_c = bool(c_only) and bool(c_only & a_all)
    if has_t and not has_c:
        return "prior_override"
    if has_c and not has_t:
        return "faithful"
    if has_c and has_t:
        return "ambiguous"
    return "other"


def classify(answer_text, report_text, true_value, cf_value):
    """Primary signal = probe answer; report text is a secondary check."""
    return {"probe_class": _cls_one(answer_text, true_value, cf_value),
            "report_class": _cls_one(report_text, true_value, cf_value)}


def reclassify_on_disk(results_root) -> int:
    """Recompute probe_class/report_class for every stored probe from its saved
    answer + the on-disk report (no API calls). Idempotent; returns count."""
    probes_dir = results_root / "probes"
    reports_dir = results_root / "reports"
    if not probes_dir.exists():
        return 0
    n = 0
    for pf in sorted(probes_dir.glob("*.json")):
        rec = json.loads(pf.read_text())
        rep = reports_dir / f"{rec['query_id']}.md"
        rep_text = rep.read_text() if rep.exists() else ""
        cls = classify(rec.get("probe_answer", ""), rep_text,
                       rec["true_value"], rec["counterfactual_value"])
        rec.update(cls)
        pf.write_text(json.dumps(rec, indent=2))
        n += 1
    return n


async def gen_one(llm, qid, query, cf_docs, entry, results_root):
    evidence = fd.evidence_block(cf_docs)
    report = await llm.complete(_report_prompt(query, evidence), model=MODEL,
                                max_tokens=8192, temperature=0.3)
    report = fd.ensure_references(report.strip() or f"# (empty)\n\nQuery: {query}\n", cf_docs)
    probe = await llm.complete(
        _probe_prompt(entry["attribute"], entry["probe_question"], evidence),
        model=MODEL, max_tokens=60, temperature=0.0)
    cls = classify(probe, report, entry["true_value"], entry["counterfactual_value"])
    (results_root / "reports").mkdir(parents=True, exist_ok=True)
    (results_root / "reports" / f"{qid}.md").write_text(report)
    probe_rec = {"query_id": qid, "attribute": entry["attribute"],
                 "true_value": entry["true_value"], "counterfactual_value": entry["counterfactual_value"],
                 "probe_answer": probe.strip(), **cls}
    (results_root / "probes").mkdir(parents=True, exist_ok=True)
    (results_root / "probes" / f"{qid}.json").write_text(json.dumps(probe_rec, indent=2))
    return probe_rec, len(report)


def _cp(ckpt_root, qid):
    return ckpt_root / f"{qid}.json"


def gen_done(ckpt_root, qid):
    cp = _cp(ckpt_root, qid)
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "success"
    except Exception:
        return False


def summarise(results_root):
    probes_dir = results_root / "probes"
    recs = [json.loads(p.read_text()) for p in sorted(probes_dir.glob("*.json"))] if probes_dir.exists() else []
    n = len(recs)
    counts = {}
    for r in recs:
        counts[r["probe_class"]] = counts.get(r["probe_class"], 0) + 1
    decided = counts.get("prior_override", 0) + counts.get("faithful", 0)
    override_rate = counts.get("prior_override", 0) / decided if decided else None
    out = {"n_probes": n, "class_counts": counts,
           "override_rate_probe": override_rate,
           "override_rate_note": "prior_override / (prior_override + faithful); ambiguous/other excluded",
           "timestamp": datetime.now(timezone.utc).isoformat()}
    (results_root / "override_summary.json").write_text(json.dumps(out, indent=2))
    return out


def _print_judge():
    print("\n  Judge later (GPT-5.2, corpus-safe): score reports for EVIDENCE-faithfulness")
    print(f"    JUDGE_RESULTS_BASE={RESULTS_ROOT}/reports \\")
    print(f"    python scripts/run_gpt52_judge_namespaced.py --judge-out {JUDGE_OUT_ROOT} "
          f"--patterns-raw . --resume --concurrency 3")
    print("  (the override_rate itself is computed here by rule; GPT-5.2 adjudicates faithfulness.)")


async def amain():
    ap = argparse.ArgumentParser(description="Counterfactual / override-rate (frozen evidence, gpt-4o-mini).")
    ap.add_argument("--stage", choices=["build", "gen", "classify", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Run --limit queries (default 2) end-to-end.")
    ap.add_argument("--run", action="store_true", help="Full oracle-30.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--checkpoint-root", default=CKPT_ROOT)
    ap.add_argument("--budget", type=float, default=BUDGET_USD_DEFAULT)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    results_root = fd.resolve_safe_out(args.results_root, "--results-root")
    ckpt_root = fd.resolve_safe_out(args.checkpoint_root, "--checkpoint-root")
    fd.resolve_safe_out(str(DATA_DIR), "data corpus dir")

    fd.route_and_pin()
    fd.assert_mini_bound()

    qmap = fd._load_eval()
    ids = fd.oracle_query_ids()
    if args.smoke and not args.limit:
        ids = ids[:2]
    elif args.limit:
        ids = ids[:args.limit]

    stage = args.stage
    if args.smoke or args.run:
        stage = "all"

    print("=" * 74)
    print("FROZEN-EVIDENCE DEFENCE 2 — COUNTERFACTUAL / OVERRIDE-RATE")
    print("=" * 74)
    print(f"  backbone: {MODEL}; evidence: frozen oracle-30 (edited to contradict prior)")
    print(f"  queries: {len(ids)}  stage: {stage}")
    print(f"  WRITE corpus/manifest (NEW): {DATA_DIR}")
    print(f"  WRITE results (NEW): {results_root}")

    if args.dry_run:
        print(f"\n  [DRY RUN] would build+gen+classify {len(ids)} queries. ZERO API calls.")
        _print_judge()
        return
    if not (args.smoke or args.run) and args.stage == "all":
        print("\n  Nothing to do. Pass --dry-run, --smoke, --run, or --stage <build|gen|classify>.")
        _print_judge()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    corpus = fd.load_corpus()
    llm, tracker = fd.make_llm(args.budget)

    # ── BUILD ──────────────────────────────────────────────────────────────────
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    cf_corpus = json.loads(CORPUS_CF.read_text()) if CORPUS_CF.exists() else {}
    if stage in ("build", "all"):
        print("\n  [build] proposing atoms + editing corpus ...", flush=True)
        for qid in ids:
            if args.resume and qid in manifest and manifest[qid].get("valid"):
                continue
            entry, edited = await build_one(llm, qid, qmap[qid]["query"], fd.frozen_docs(qid, corpus))
            manifest[qid] = entry
            if edited is not None:
                cf_corpus[qid] = edited
            tag = "OK " if entry.get("valid") else "SKIP"
            print(f"    {tag} {qid[:30]:30s} true={entry.get('true_value','')!r:>18} "
                  f"-> cf={entry.get('counterfactual_value','')!r:>18} "
                  f"repl={entry.get('n_replacements','-')}", flush=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        CORPUS_CF.write_text(json.dumps(cf_corpus))
        n_valid = sum(1 for q in ids if manifest.get(q, {}).get("valid"))
        print(f"  [build] valid atoms: {n_valid}/{len(ids)}  ${tracker.total_cost:.5f}")

    # ── GEN ────────────────────────────────────────────────────────────────────
    if stage in ("gen", "all"):
        print("\n  [gen] synthesising over contradicting evidence + probing ...", flush=True)
        for qid in ids:
            entry = manifest.get(qid, {})
            if not entry.get("valid"):
                continue
            if args.resume and gen_done(ckpt_root, qid):
                continue
            cf_docs = cf_corpus.get(qid) or fd.frozen_docs(qid, corpus)
            t0 = time.time()
            armed = False
            try:
                try:
                    signal.signal(signal.SIGALRM, fd._alarm_handler)
                    signal.alarm(PER_QUERY_TIMEOUT_S + 90)
                    armed = True
                except (ValueError, AttributeError):
                    armed = False
                probe_rec, rchars = await asyncio.wait_for(
                    gen_one(llm, qid, qmap[qid]["query"], cf_docs, entry, results_root),
                    timeout=PER_QUERY_TIMEOUT_S)
                if armed:
                    signal.alarm(0); armed = False
                _cp(ckpt_root, qid).write_text(json.dumps(
                    {"query_id": qid, "status": "success", "report_chars": rchars,
                     "probe_class": probe_rec["probe_class"], "report_class": probe_rec["report_class"],
                     "elapsed_seconds": round(time.time() - t0, 1),
                     "total_cost_usd": round(tracker.total_cost, 6),
                     "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2))
                print(f"    OK   {qid[:30]:30s} probe={probe_rec['probe_class']:14s} "
                      f"ans={probe_rec['probe_answer'][:24]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                _cp(ckpt_root, qid).write_text(json.dumps(
                    {"query_id": qid, "status": "error", "error": str(e)[:300]}, indent=2))
                print(f"    FAIL {qid[:30]:30s} -- {str(e)[:110]}", flush=True)
            finally:
                if armed:
                    signal.alarm(0)

    # ── CLASSIFY / SUMMARISE ───────────────────────────────────────────────────
    if stage in ("classify", "gen", "all"):
        nrc = reclassify_on_disk(results_root)
        if stage == "classify":
            print(f"  [classify] reclassified {nrc} probes from disk (no API calls)")
        summ = summarise(results_root)
        print(f"\n  [summary] probes={summ['n_probes']} classes={summ['class_counts']}")
        print(f"  [summary] OVERRIDE RATE (probe) = {summ['override_rate_probe']}")

    print(f"\n  total spend this run: ${tracker.total_cost:.5f}  ({tracker.total_tokens} tokens)")
    _print_judge()


def _self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    ids = fd.oracle_query_ids()
    check("oracle-30", len(ids) == 30)
    docs = fd.frozen_docs(ids[0])
    # swap logic
    edited, n = _apply_swap(docs, "the", "XZQ")
    check("swap edits spans", n > 0 and any("XZQ" in (d.get("content", "") or "") for d in edited))
    check("swap leaves doc count", len(edited) == len(docs))
    # validate
    v, _ = validate_atom({"true_value": "2017", "counterfactual_value": "2019"},
                         [{"title": "", "content": "published in 2017 by"}])
    check("validate accepts present atom", v)
    v2, _ = validate_atom({"true_value": "zzz_absent", "counterfactual_value": "2019"},
                          [{"title": "", "content": "nothing here"}])
    check("validate rejects absent atom", not v2)
    v3, _ = validate_atom({"true_value": "2017", "counterfactual_value": "2017"},
                          [{"title": "", "content": "2017"}])
    check("validate rejects equal cf", not v3)
    # classify
    c = classify("2017", "the year was 2017", "2017", "2019")
    check("classify prior_override", c["probe_class"] == "prior_override")
    c2 = classify("2019", "the year was 2019", "2017", "2019")
    check("classify faithful", c2["probe_class"] == "faithful")
    # paraphrase robustness: '1980s' must match counterfactual 'mid-1980s'
    c3 = classify("1980s", "", "mid-1970s", "mid-1980s")
    check("classify paraphrase faithful (1980s~mid-1980s)", c3["probe_class"] == "faithful")
    c4 = classify("the 1970s", "", "mid-1970s", "mid-1980s")
    check("classify paraphrase override (1970s~mid-1970s)", c4["probe_class"] == "prior_override")
    c5 = classify("Bahdanau", "", "D. Bahdanau", "A. Vaswani")
    check("classify name faithful (surname)", c5["probe_class"] == "prior_override")
    for bad in ["results/experiments", "data/analysis", "results"]:
        try:
            fd.resolve_safe_out(bad, "x"); check(f"refuse {bad}", False)
        except SystemExit:
            check(f"refuse {bad}", True)
    try:
        check("accept NEW root", fd.resolve_safe_out(RESULTS_ROOT, "x").name == "experiments_counterfactual")
    except SystemExit:
        check("accept NEW root", False)
    print("\n  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    asyncio.run(amain())
