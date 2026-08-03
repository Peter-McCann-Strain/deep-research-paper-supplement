#!/usr/bin/env python3
"""E5 ORACLE-DOSE — graded retrieval-quality dose-response harness.

Implements the E5 experiment from reports/RESEARCH_PLAN_2026H2.md (priority 8):

    Question. Is citation quality monotonically retrieval-bound while factual
    accuracy stays flat at every gold-fraction tier -- and does interleaved
    (vs one-shot) oracle delivery rescue factuality (the 2601.19827 context-
    overload mechanism)?

    Design. 3 architectures (P0, P1, P4) x 30 variance-stratified queries x 6
    cells: gold fraction 0/25/50/75/100% (hard negatives mined from our own
    logged non-cited retrievals) + an interleaved-oracle cell.  540 PTU runs.
    Judges: GPT-5.2 (authoritative) here; gpt-4.1 + DR-Judge-7B + Claude pair
    are queued separately (out of scope for this harness).
    Pre-register: citation monotone, factual flat.

This is a SELF-CONTAINED, CORPUS-SAFE generation + GPT-5.2-judging harness.

WHY A NEW SCRIPT (not just run_all_experiments.py).
  run_all_experiments.py writes generated reports straight into
  results/experiments/<exp_id>/ -- a PROTECTED, READ-ONLY corpus dir.  E5 must
  write ONLY to new dirs.  So this harness reuses the SAME pattern run()
  entrypoints and the SAME oracle backend mechanism (SEARCH_BACKEND=oracle +
  ORACLE_CORPUS_PATH + ORACLE_QUERY_ID), but routes every write to NEW dirs.
  The authoritative GPT-5.2 judge is reused VERBATIM by importing the call path
  (evaluate_one / load_queries / config wiring) out of the corpus-safe
  scripts/run_gpt52_judge_namespaced.py -- never gpt-4o / gpt-4.1 / any other
  model as the authoritative judge.

CONSISTENCY (hard requirement).
  * Generation backbone: gpt-4o on PTU deployment "sthree-ptu-02"
    (config DEFAULT_MODEL=gpt-4o) -- the SAME backbone as the 248k-report
    corpus.  This harness NEVER overrides DEFAULT_MODEL; it asserts it is gpt-4o.
  * SEARCH_MODEL stays gpt-4o-mini (corpus default; never touched).
  * Authoritative judge: GPT-5.2 (JUDGE_MODEL=gpt-5.2) via the namespaced runner.

SAFETY.
  NEVER writes to/modifies results/judge_gpt52/, results/experiments/,
  data/analysis/*.parquet, reports/eval_v2/verdicts/.  All E5 writes land under
  NEW dirs (defaults below).  A hard guard refuses any output path that
  resolves into a protected location.  --dry-run / --limit make ZERO API calls.

DIRECTORY LAYOUT (all NEW, none protected):
  data/e5_oracle_dose/corpus/<cell>.json     graded-dose oracle corpora
  results/e5_oracle_dose/gen/<exp_id>/*.md   generated reports + .json checkpoints
  results/judge_gpt52_e5/<exp_id>/*.json     GPT-5.2 verdicts (corpus-safe root)

USAGE:
  # zero-API smoke test (build corpora is real/local; gen+judge are simulated):
  python scripts/run_e5_oracle_dose.py --dry-run
  python scripts/run_e5_oracle_dose.py --limit 2 --dry-run

  # build the graded-dose corpora only (local, free):
  python scripts/run_e5_oracle_dose.py --build-corpus-only

  # the paid run (launched separately by the human, NOT here):
  python scripts/run_e5_oracle_dose.py --stage gen                 # 540 PTU runs
  python scripts/run_e5_oracle_dose.py --stage judge               # GPT-5.2 verdicts
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# REQUIRED so `python scripts/run_e5_oracle_dose.py` imports deep_research.*
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# Also put scripts/ on the path so the judge stage's
# importlib.import_module("run_gpt52_judge_namespaced") resolves regardless of
# how this file is launched (python -m, alternate cwd, etc.).  When launched as
# `python scripts/run_e5_oracle_dose.py` Python already adds scripts/ to
# sys.path[0]; this makes it explicit and robust.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── Constants / paths (ALL NEW dirs) ──────────────────────────────────────────

ARCHITECTURES = ["p0", "p1", "p4"]  # fixed architectures; spec: P0/P1/P4
GOLD_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]  # graded dose tiers
INTERLEAVED_CELL = "interleaved"  # 6th cell: 100% gold, delivered progressively
DOCS_PER_CELL = 12  # total docs held fixed across tiers (gold + hard-neg mix)
DEFAULT_SEED = 7    # match variance_stratified.json seed


def stable_url_id(prefix: str, url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"

PATTERNS = {
    "p0": "deep_research.patterns.p0_baseline.pipeline",
    "p1": "deep_research.patterns.p1_iterative_rag.pipeline",
    "p4": "deep_research.patterns.p4_perspective_storm.pipeline",
}

# NEW output roots — none of these is a protected corpus path.
CORPUS_DIR = _REPO_ROOT / "data" / "e5_oracle_dose" / "corpus"
GEN_DIR = _REPO_ROOT / "results" / "e5_oracle_dose" / "gen"
DEFAULT_JUDGE_OUT = _REPO_ROOT / "results" / "judge_gpt52_e5"

# Inputs (READ-ONLY).
VARIANCE_FILE = _REPO_ROOT / "data" / "variance_stratified.json"
ORACLE_CORPUS = _REPO_ROOT / "data" / "oracle_corpus_t1.json"
C0_URL_INDEX = _REPO_ROOT / "data" / "c0_url_index.json"
CITATIONS_PARQUET = _REPO_ROOT / "data" / "analysis" / "df_citations.parquet"
QUERIES_PARQUET = _REPO_ROOT / "data" / "analysis" / "df_queries.parquet"
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"

# ── Protected paths — refuse to ever write inside these ───────────────────────
PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_safe_write(path: Path) -> Path:
    """Hard-refuse any write target that equals, lives inside, or is an ancestor
    of a protected corpus path."""
    p = path if path.is_absolute() else _REPO_ROOT / path
    p = p.resolve()
    for prot in PROTECTED_PATHS:
        prot = prot.resolve()
        if p == prot or _is_relative_to(p, prot) or _is_relative_to(prot, p):
            raise SystemExit(
                f"REFUSING: write target {p} collides with protected corpus path "
                f"{prot}. E5 must write ONLY to new dirs."
            )
    return p


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ── Cell naming ───────────────────────────────────────────────────────────────

def cell_name(cell) -> str:
    """gold-fraction float -> 'g000'..'g100'; INTERLEAVED_CELL -> 'interleaved'."""
    if cell == INTERLEAVED_CELL:
        return INTERLEAVED_CELL
    return f"g{int(round(cell * 100)):03d}"


ALL_CELLS = [*GOLD_FRACTIONS, INTERLEAVED_CELL]


def exp_id_for(pattern: str, cell) -> str:
    return f"e5_oracle_dose_{pattern}_{cell_name(cell)}"


# ── Stage 1: build graded-dose oracle corpora (local, free) ───────────────────

def _norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/")


def load_variance_ids() -> list[str]:
    return json.loads(VARIANCE_FILE.read_text())["query_ids"]


def build_dose_corpora(limit: int | None = None, verbose: bool = True) -> dict:
    """Build one oracle corpus JSON per (gold_fraction) cell + the interleaved cell.

    For each variance query we hold the TOTAL doc count fixed at DOCS_PER_CELL and
    vary the gold:hard-negative ratio:
      * gold docs   = the pooled, real-cited oracle docs (oracle_corpus_t1.json),
                      i.e. evidence the architectures actually grounded citations on.
      * hard negs   = URLs in the C0 retrieval cache that were RETRIEVED into the
                      pipeline but NEVER cited for this query -> realistic
                      "plausible but unhelpful" distractors mined from our own logs.

    gold_fraction f -> round(f * DOCS_PER_CELL) gold docs + the rest hard negs.
    The interleaved cell uses the SAME 100%-gold doc set as g100 (delivery order
    is what differs; that is handled at generation time, not in the corpus).

    Each cell corpus is written to CORPUS_DIR/<cell>.json as {query_id: [Document]}.
    Returns a stats dict.  Pure local I/O; no API calls.
    """
    import pandas as pd

    set_seed(DEFAULT_SEED)
    ids = load_variance_ids()
    if limit:
        ids = ids[:limit]

    # gold pool: the pooled-existing oracle corpus (real cited evidence).
    oracle = json.loads(ORACLE_CORPUS.read_text())

    # C0 cache: 42k retrieved URLs -> {title, content}.  Source of hard negatives.
    c0 = json.loads(C0_URL_INDEX.read_text())
    c0n = {_norm_url(k): v for k, v in c0.items()}

    # cited-URL set per query (so hard negs exclude anything ever cited here).
    cit = pd.read_parquet(CITATIONS_PARQUET)
    cit = cit[cit.cited_url.fillna("").str.startswith("http")]
    cited_by_q: dict[str, set] = {}
    for qid, sub in cit.groupby("query_id"):
        cited_by_q[qid] = {_norm_url(u) for u in sub.cited_url}

    # gold-answer strings for leakage redaction (mirror build_oracle_corpus.py).
    gold_str = {}
    if QUERIES_PARQUET.exists():
        q = pd.read_parquet(QUERIES_PARQUET)
        gold_str = dict(zip(q.query_id, q.get("gold_answer", pd.Series([""] * len(q))).fillna("")))

    # A deterministic, query-agnostic hard-negative pool: C0 URLs never cited by
    # ANY query (global distractors) -- plus per-query we exclude that query's
    # cited URLs.  Ranked by a fixed hash so selection is reproducible.
    all_cited = set().union(*cited_by_q.values()) if cited_by_q else set()
    global_negs = [u for u in c0n.keys() if u and u not in all_cited]
    global_negs.sort(key=lambda u: f"{DEFAULT_SEED}:{u}")  # deterministic order

    def make_doc(url: str, is_gold: bool, qid: str, src: dict | None = None) -> dict:
        rec = src if src is not None else c0n.get(url, {})
        content = (rec.get("content", "") or "")
        gstr = (gold_str.get(qid, "") or "").strip()
        if gstr and len(gstr) > 8:
            content = re.sub(re.escape(gstr), "[redacted]", content, flags=re.I)
        return {
            "id": stable_url_id(f"e5_{'gold' if is_gold else 'neg'}", url),
            "title": rec.get("title", "") or "",
            "content": content,
            "url": url,
            "source_type": "web",
            "metadata": {"oracle": True, "e5_gold": is_gold},
        }

    # build per-cell corpora
    corpora: dict[str, dict] = {c: {} for c in [cell_name(f) for f in GOLD_FRACTIONS] + [INTERLEAVED_CELL]}
    per_query_stats = []

    for qid in ids:
        gold_docs_raw = oracle.get(qid, [])[:DOCS_PER_CELL]  # already ranked by pool_freq
        # per-query hard-neg pool: global negs minus this query's cited URLs,
        # seeded by qid for variety while staying deterministic.
        qnegs = [u for u in global_negs if u not in cited_by_q.get(qid, set())]
        rnd = random.Random(f"{DEFAULT_SEED}:{qid}")
        rnd.shuffle(qnegs)

        n_gold_avail = len(gold_docs_raw)
        for f in GOLD_FRACTIONS:
            n_gold = min(int(round(f * DOCS_PER_CELL)), n_gold_avail)
            n_neg = DOCS_PER_CELL - n_gold
            docs = [dict(d, metadata={**d.get("metadata", {}), "e5_gold": True}) for d in gold_docs_raw[:n_gold]]
            for u in qnegs[:n_neg]:
                docs.append(make_doc(u, is_gold=False, qid=qid))
            corpora[cell_name(f)][qid] = docs

        # interleaved cell = same doc set as g100 (max gold); order matters at gen.
        corpora[INTERLEAVED_CELL][qid] = [
            dict(d, metadata={**d.get("metadata", {}), "e5_gold": True})
            for d in gold_docs_raw[:DOCS_PER_CELL]
        ]
        per_query_stats.append((qid, n_gold_avail, len(qnegs)))

    # write corpora to NEW dir
    out_dir = assert_safe_write(CORPUS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for cell, corpus in corpora.items():
        path = assert_safe_write(out_dir / f"{cell}.json")
        path.write_text(json.dumps(corpus))
        written[cell] = str(path)

    stats = {
        "n_queries": len(ids),
        "docs_per_cell": DOCS_PER_CELL,
        "gold_fractions": GOLD_FRACTIONS,
        "cells": list(corpora.keys()),
        "corpus_files": written,
        "queries_with_zero_gold": [q for q, g, _ in per_query_stats if g == 0],
        "mean_gold_available": round(sum(g for _, g, _ in per_query_stats) / max(len(per_query_stats), 1), 1),
        "mean_hardneg_pool": round(sum(n for _, _, n in per_query_stats) / max(len(per_query_stats), 1), 1),
    }
    if verbose:
        print(f"[build-corpus] queries={stats['n_queries']} docs/cell={DOCS_PER_CELL} "
              f"cells={len(corpora)} mean_gold_avail={stats['mean_gold_available']} "
              f"hardneg_pool~{stats['mean_hardneg_pool']:.0f}")
        if stats["queries_with_zero_gold"]:
            print(f"[build-corpus] WARNING zero-gold queries: "
                  f"{[q[:18] for q in stats['queries_with_zero_gold']]}")
        for cell, p in written.items():
            print(f"  wrote {cell:>11s} -> {p}")
    return stats


# ── Interleaved oracle searcher (process-local monkey-patch; no file edits) ───

class _InterleavedOracleSearcher:
    """Serves the frozen gold corpus PROGRESSIVELY across successive search calls.

    The vanilla OracleSearcher returns the ENTIRE corpus on the first call
    (one-shot context dump).  For the interleaved cell we instead hand out a
    rolling window of docs per call, so a pattern that searches K times receives
    the gold evidence spread across K turns -- the 2601.19827 context-overload
    rescue condition.  State is per-process and resets each run via env query id.
    """

    def __init__(self):
        from deep_research.config import DATA_DIR  # noqa: F401  (keep parity)
        self.corpus_path = os.environ["ORACLE_CORPUS_PATH"]
        self.window = int(os.environ.get("E5_INTERLEAVE_WINDOW", "3"))
        self._cursor = 0
        self._corpus = json.loads(Path(self.corpus_path).read_text())

    def _all(self):
        from deep_research.types import Document, SourceType
        qid = os.environ.get("ORACLE_QUERY_ID", "")
        raw = self._corpus.get(qid, [])
        out = []
        for d in raw:
            try:
                out.append(Document(**d))
            except Exception:
                out.append(Document(
                    id=str(d.get("id", "")), title=str(d.get("title", "")),
                    content=str(d.get("content", "")), url=str(d.get("url", "")),
                    source_type=SourceType.WEB, metadata={"oracle": True},
                ))
        return out

    def _next_window(self):
        docs = self._all()
        if not docs:
            return []
        start = self._cursor % len(docs)
        self._cursor += self.window
        win = docs[start:start + self.window]
        if len(win) < self.window:  # wrap
            win += docs[: self.window - len(win)]
        return win

    async def search(self, query: str, max_results: int = 10, **kwargs):
        return self._next_window()

    async def search_batch(self, queries, max_results_per: int = 5, **kwargs):
        return self._next_window()


def _install_interleaved_searcher():
    """Monkey-patch tools.get_web_searcher to return the interleaved searcher when
    SEARCH_BACKEND=oracle AND E5_INTERLEAVE=1.  Process-local only; touches no files."""
    import deep_research.tools as tools
    _orig = tools.get_web_searcher

    def _patched(backend: str | None = None):
        if os.environ.get("E5_INTERLEAVE") == "1" and (
            backend == "oracle" or os.environ.get("SEARCH_BACKEND") == "oracle"
        ):
            return _InterleavedOracleSearcher()
        return _orig(backend)

    tools.get_web_searcher = _patched
    # patch the name already imported into each pattern module, if present
    for mod_path in PATTERNS.values():
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, "get_web_searcher"):
                m.get_web_searcher = _patched
        except Exception:
            pass


# ── Stage 2: generation (gpt-4o on PTU; writes to NEW gen dir) ────────────────

def _assert_generation_backbone():
    """Guarantee generation uses the corpus backbone gpt-4o on PTU sthree-ptu-02."""
    import deep_research.config as cfg
    if cfg.DEFAULT_MODEL != "gpt-4o":
        raise SystemExit(
            f"REFUSING: DEFAULT_MODEL={cfg.DEFAULT_MODEL!r}, must be 'gpt-4o' "
            f"(corpus backbone). E5 never overrides the generation model."
        )
    spec = cfg.MODELS.get("gpt-4o")
    if not spec or spec.deployment != "sthree-ptu-02":
        raise SystemExit(
            f"REFUSING: gpt-4o deployment={getattr(spec, 'deployment', None)!r}, "
            f"must be 'sthree-ptu-02' (the corpus PTU)."
        )
    if cfg.SEARCH_MODEL != "gpt-4o-mini":
        raise SystemExit(
            f"REFUSING: SEARCH_MODEL={cfg.SEARCH_MODEL!r}, must stay 'gpt-4o-mini'."
        )


def _gen_paths(exp_id: str, query_id: str) -> tuple[Path, Path]:
    base = assert_safe_write(GEN_DIR / exp_id)
    return base / f"{query_id}.md", base / f"{query_id}.json"


def gen_is_completed(exp_id: str, query_id: str) -> bool:
    _, cp = _gen_paths(exp_id, query_id)
    if not cp.exists():
        return False
    try:
        return json.loads(cp.read_text()).get("status") == "success"
    except Exception:
        return False


async def run_one_gen(pattern: str, cell, query: dict, budget: float) -> dict:
    """Generate one (pattern, cell, query) report with gpt-4o on PTU, oracle backend.
    Writes the report + checkpoint into the NEW gen dir. Returns a result dict."""
    exp_id = exp_id_for(pattern, cell)
    qid = query["id"]
    md_path, cp_path = _gen_paths(exp_id, qid)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # wire the oracle backend for THIS cell
    os.environ["SEARCH_BACKEND"] = "oracle"
    os.environ["ORACLE_CORPUS_PATH"] = str(CORPUS_DIR / f"{cell_name(cell)}.json")
    os.environ["ORACLE_QUERY_ID"] = qid
    os.environ["ORACLE_MAX_DOCS"] = str(DOCS_PER_CELL)
    if cell == INTERLEAVED_CELL:
        os.environ["E5_INTERLEAVE"] = "1"
    else:
        os.environ.pop("E5_INTERLEAVE", None)

    import deep_research.config as cfg
    cfg.SEARCH_BACKEND = "oracle"

    mod = importlib.import_module(PATTERNS[pattern])
    t0 = time.time()
    try:
        report = await mod.run(query["query"], budget_usd=budget, query_id=qid)
        elapsed = time.time() - t0
        md_path.write_text(report.full_text())
        result = {
            "status": "success", "experiment_id": exp_id, "pattern": pattern,
            "cell": cell_name(cell), "query_id": qid, "elapsed_seconds": elapsed,
            "total_tokens": report.total_tokens, "total_cost_usd": report.total_cost_usd,
            "sections": len(report.sections), "citations": len(report.citations),
            "gen_model": cfg.DEFAULT_MODEL, "search_model": cfg.SEARCH_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        cp_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  [{pattern}/{cell_name(cell)}] OK {qid[:18]} {elapsed:.0f}s "
              f"{report.total_tokens:,}tok {len(report.citations)}cit")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "status": "error", "experiment_id": exp_id, "pattern": pattern,
            "cell": cell_name(cell), "query_id": qid, "elapsed_seconds": elapsed,
            "error": str(e)[:500], "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        cp_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  [{pattern}/{cell_name(cell)}] FAILED {qid[:18]} {str(e)[:90]}")
        return result


def build_gen_plan(patterns, cells, queries, resume: bool) -> list:
    plan = []
    for pattern in patterns:
        for cell in cells:
            for q in queries:
                if resume and gen_is_completed(exp_id_for(pattern, cell), q["id"]):
                    continue
                plan.append((pattern, cell, q))
    return plan


# ── Stage 3: GPT-5.2 judging (reuse the namespaced runner's call path) ────────

async def run_judge_stage(patterns, cells, queries, judge_out: Path,
                          concurrency: int, resume: bool) -> dict:
    """Judge E5 reports with GPT-5.2 by REUSING the corpus-safe namespaced runner.

    We import evaluate_one / load_queries / the GPT-5.2 client + config straight
    out of scripts/run_gpt52_judge_namespaced.py (the authoritative, corpus-safe
    judge), but READ from the E5 gen dir and WRITE to a NEW judge_out root.  The
    namespaced runner itself only reads results/experiments; here we drive its
    evaluate_one() over E5's own (new) report dir instead.  No protected path is
    read or written; GPT-5.2 is the only judge wired.
    """
    judge_out = assert_safe_write(judge_out)
    nj = importlib.import_module("run_gpt52_judge_namespaced")

    # sanity: the authoritative judge must be GPT-5.2
    from deep_research.config import JUDGE_MODEL
    if JUDGE_MODEL != "gpt-5.2":
        raise SystemExit(
            f"REFUSING: JUDGE_MODEL={JUDGE_MODEL!r}, authoritative judge must be 'gpt-5.2'."
        )

    qmap = {q["id"]: q for q in queries}
    semaphore = asyncio.Semaphore(concurrency)

    work = []
    for pattern in patterns:
        for cell in cells:
            exp_id = exp_id_for(pattern, cell)
            gen_dir = GEN_DIR / exp_id
            if not gen_dir.exists():
                continue
            for md in sorted(gen_dir.glob("*.md")):
                qid = md.stem
                if qid not in qmap:
                    continue
                out_path = judge_out / exp_id / f"{qid}.json"
                if resume and out_path.exists():
                    continue
                work.append((exp_id, qid, md))

    print(f"[judge] GPT-5.2 ({JUDGE_MODEL}) pending={len(work)} -> {judge_out}")
    judge_out.mkdir(parents=True, exist_ok=True)
    done = {"completed": 0, "failed": 0}

    async def _one(exp_id, qid, md):
        try:
            res = await nj.evaluate_one(
                semaphore, exp_id, qid, qmap[qid], md.read_text()
            )
            out_dir = assert_safe_write(judge_out / exp_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{qid}.json").write_text(json.dumps(res, indent=2))
            done["completed"] += 1
            print(f"  [{done['completed'] + done['failed']}/{len(work)}] "
                  f"{exp_id}/{qid[:18]}: {res['overall_score']:.3f}")
        except Exception as e:
            done["failed"] += 1
            print(f"  FAILED {exp_id}/{qid[:18]}: {str(e)[:120]}")

    await asyncio.gather(*[_one(*w) for w in work])
    print(f"[judge] complete: {done['completed']} ok, {done['failed']} failed")
    return done


# ── Stage 4: dose-response fit (local; runs on E5 verdicts only) ──────────────

def fit_dose_response(judge_out: Path) -> dict:
    """Pre-registered primary endpoint: mixed-effects dose-response slope model
    factual_accuracy ~ gold_fraction with query random effects, pooled over the 3
    architectures, with a one-sided slope CI bound; plus citation_quality slope.

    Reads ONLY E5 verdict JSONs from judge_out (a NEW dir). Falls back to an OLS
    slope if statsmodels MixedLM is unavailable. Returns a results dict and writes
    it next to the verdicts. No API calls."""
    judge_out = Path(judge_out)
    rows = []
    frac_of = {cell_name(f): f for f in GOLD_FRACTIONS}
    for exp_dir in sorted(judge_out.glob("e5_oracle_dose_*")):
        m = re.match(r"e5_oracle_dose_(p\d+)_(g\d{3}|interleaved)$", exp_dir.name)
        if not m:
            continue
        pattern, cell = m.group(1), m.group(2)
        for jf in exp_dir.glob("*.json"):
            try:
                v = json.loads(jf.read_text())
            except Exception:
                continue
            dims = v.get("dimensions", {})
            rows.append({
                "pattern": pattern, "cell": cell, "query_id": jf.stem,
                "gold_fraction": frac_of.get(cell),  # None for interleaved
                "factual_accuracy": dims.get("factual_accuracy", {}).get("score"),
                "citation_quality": dims.get("citation_quality", {}).get("score"),
                "overall": v.get("overall_score"),
            })
    if not rows:
        return {"status": "no_verdicts", "judge_out": str(judge_out)}

    import pandas as pd
    df = pd.DataFrame(rows)
    dose = df[df.gold_fraction.notna()].copy()
    result = {"n_verdicts": len(df), "n_dose_points": len(dose),
              "cells_present": sorted(df.cell.unique().tolist())}

    def _slope(dim):
        d = dose.dropna(subset=[dim, "gold_fraction"])
        if len(d) < 8 or d.gold_fraction.nunique() < 2:
            return {"status": "insufficient", "n": len(d)}
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm(f"{dim} ~ gold_fraction", d, groups=d["query_id"])
            fit = md.fit(reml=True, method="lbfgs")
            slope = float(fit.params["gold_fraction"])
            ci = fit.conf_int().loc["gold_fraction"].tolist()
            return {"status": "mixedlm", "slope": slope,
                    "ci95": [float(ci[0]), float(ci[1])],
                    "p_value": float(fit.pvalues["gold_fraction"]), "n": len(d)}
        except Exception:
            import numpy as np
            x = d.gold_fraction.to_numpy(); y = d[dim].to_numpy()
            b, a = np.polyfit(x, y, 1)
            return {"status": "ols_fallback", "slope": float(b), "intercept": float(a), "n": len(d)}

    result["factual_accuracy_slope"] = _slope("factual_accuracy")
    result["citation_quality_slope"] = _slope("citation_quality")

    # interleaved vs g100 contrast (one-shot vs progressive at 100% gold)
    g100 = df[df.cell == "g100"]; inter = df[df.cell == "interleaved"]
    if len(g100) and len(inter):
        import numpy as np
        result["interleaved_vs_g100"] = {
            "factual_g100_mean": float(np.nanmean(g100.factual_accuracy)),
            "factual_interleaved_mean": float(np.nanmean(inter.factual_accuracy)),
            "delta": float(np.nanmean(inter.factual_accuracy) - np.nanmean(g100.factual_accuracy)),
        }
    out = assert_safe_write(judge_out / "e5_dose_response_fit.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[fit] wrote {out}")
    print(f"[fit] factual slope: {result['factual_accuracy_slope']}")
    print(f"[fit] citation slope: {result['citation_quality_slope']}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

async def amain():
    ap = argparse.ArgumentParser(description="E5 ORACLE-DOSE harness (corpus-safe)")
    ap.add_argument("--stage", choices=["build", "gen", "judge", "fit", "all"],
                    default="all", help="Which stage to run (default: all)")
    ap.add_argument("--build-corpus-only", action="store_true",
                    help="Alias for --stage build (local, free).")
    ap.add_argument("--patterns", default=",".join(ARCHITECTURES),
                    help=f"Comma-sep architectures (default {','.join(ARCHITECTURES)})")
    ap.add_argument("--cells", default="all",
                    help="Comma-sep cells (g000,g025,g050,g075,g100,interleaved) or 'all'")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of queries (smoke testing). 0 = all 30.")
    ap.add_argument("--budget", type=float, default=2.0, help="USD budget per gen run")
    ap.add_argument("--judge-out", default=str(DEFAULT_JUDGE_OUT),
                    help=f"GPT-5.2 verdict root (default {DEFAULT_JUDGE_OUT}).")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Judge concurrency (default 3, matches GPT-5.2 semaphore).")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip completed cells (default).")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="ZERO API calls: build corpora (local) then print the plan + estimates.")
    args = ap.parse_args()

    stage = "build" if args.build_corpus_only else args.stage

    # patterns / cells
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for p in patterns:
        if p not in PATTERNS:
            raise SystemExit(f"Unknown pattern {p!r}; allowed: {list(PATTERNS)}")
    if args.cells.strip().lower() == "all":
        cells = list(ALL_CELLS)
    else:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in ALL_CELLS if cell_name(c) in wanted]
        if not cells:
            raise SystemExit(f"No valid cells in {args.cells!r}; allowed: "
                             f"{[cell_name(c) for c in ALL_CELLS]}")

    # queries
    qmap = {q["id"]: q for q in json.loads(EVAL_QUERIES.read_text())["queries"]}
    ids = load_variance_ids()
    if args.limit:
        ids = ids[: args.limit]
    queries = [qmap[i] for i in ids if i in qmap]

    print("=" * 64)
    print("E5 ORACLE-DOSE  (graded retrieval dose-response)")
    print("=" * 64)
    print(f"architectures: {patterns}")
    print(f"cells:         {[cell_name(c) for c in cells]}")
    print(f"queries:       {len(queries)} (variance-stratified{' LIMITED' if args.limit else ''})")
    print(f"gen backbone:  gpt-4o @ sthree-ptu-02 (PTU)  | search: gpt-4o-mini")
    print(f"judge:         gpt-5.2 (authoritative)")
    print(f"judge-out:     {assert_safe_write(Path(args.judge_out))}")
    print(f"dry-run:       {args.dry_run}")
    print()

    total_cells = len(patterns) * len(cells) * len(queries)

    # ── BUILD (always safe to run; local) ─────────────────────────────────────
    if stage in ("build", "all"):
        build_dose_corpora(limit=args.limit or None)

    # ── DRY-RUN: print plan + estimates, ZERO API calls ──────────────────────
    if args.dry_run:
        gen_plan = build_gen_plan(patterns, cells, queries, args.resume)
        # judge estimate = one GPT-5.2 call per generated report
        est_gpt52_calls = total_cells
        est_cost = est_gpt52_calls * 0.08  # namespaced runner's per-report estimate
        print("[DRY RUN] generation plan (no API calls):")
        print(f"  total gen cells (pattern x cell x query): {total_cells}")
        print(f"  pending after resume:                     {len(gen_plan)}")
        print(f"  est GPT-5.2 judge calls (1/report):       {est_gpt52_calls}")
        print(f"  est GPT-5.2 cost @ $0.08/report:          ${est_cost:.2f}")
        print(f"  generation cost (gpt-4o PTU):             $0.00 (pre-paid PTU)")
        print(f"  search cost (gpt-4o-mini):                negligible")
        # show a few example cells
        for pattern, cell, q in gen_plan[:6]:
            print(f"    would gen: {exp_id_for(pattern, cell)} x {q['id'][:18]}  "
                  f"(corpus={CORPUS_DIR.name}/{cell_name(cell)}.json)")
        print("\n[DRY RUN] No API calls made. Corpora WERE built (local I/O only).")
        return

    # ── GEN (paid PTU; launched separately by human) ─────────────────────────
    if stage in ("gen", "all"):
        _assert_generation_backbone()
        if INTERLEAVED_CELL in [c for c in cells]:
            _install_interleaved_searcher()
        plan = build_gen_plan(patterns, cells, queries, args.resume)
        print(f"[gen] {len(plan)}/{total_cells} cells to generate "
              f"(gpt-4o PTU; resume={args.resume})")
        for i, (pattern, cell, q) in enumerate(plan, 1):
            set_seed(DEFAULT_SEED)
            print(f"[{i}/{len(plan)}] {exp_id_for(pattern, cell)} x {q['id'][:18]}")
            await run_one_gen(pattern, cell, q, args.budget)

    # ── JUDGE (GPT-5.2; paid; launched separately by human) ──────────────────
    if stage in ("judge", "all"):
        await run_judge_stage(patterns, cells, queries,
                              Path(args.judge_out), args.concurrency, args.resume)

    # ── FIT (local) ──────────────────────────────────────────────────────────
    if stage in ("fit", "all"):
        fit_dose_response(Path(args.judge_out))


if __name__ == "__main__":
    asyncio.run(amain())
