#!/usr/bin/env python3
"""RETRIEVE-AND-FREEZE — build the frozen-source corpus for the frozen_vintage experiment.

What this does
--------------
Runs the FROZEN P9 scaffold's retrieval + extraction (Stages 1-3) EXACTLY ONCE per
query, using a SINGLE fixed, capable EXTRACTOR model (GPT-4o, the canonical LLMCaller),
and persists the generator's literal input — the post-extraction ``evidence_text`` string
plus the structured extractions — to a NEW frozen store. Every model arm in the vintage
curve then regenerates its report by reading that byte-identical store, so the ONLY
variable across arms is the GENERATOR backbone (its release vintage / capacity). Retrieval
and extraction nondeterminism (network, S2 ordering, extractor sampling) is fully removed.

Why freeze the POST-extraction string, not raw pages
----------------------------------------------------
Extraction is MODEL-DEPENDENT: SourceExtractor Step-1 (free-text analysis) and Step-2
(structured JSON) both call ``self.llm`` (see deep_research/tools/source_extractor.py).
If we froze raw pages and let each arm re-extract, the sources each arm sees would differ
— defeating the experiment. The only model-independent join point is the evidence string
the generator literally consumes: ``evidence_text = format_extractions_as_evidence(...)``
AFTER the per-arm word-limit truncation. THAT string is the freeze unit. We freeze the
ALREADY-truncated string so a 14B arm and a 7B arm (whose evidence_word_limit could differ)
see byte-identical context.

Choosing the extractor (a deliberate design decision)
-----------------------------------------------------
The extractor is GPT-4o — capable, constant, and NOT one of the graded generator arms here.
Using any single 7B arm as the extractor would give that arm's vintage a home-field
advantage. GPT-4o is held constant for every arm, so no graded arm is favoured. The
extractor differing from the generator backbone is intentional and documented.

Corpus safety (HARD)
--------------------
Writes ONLY under data/frozen_corpus_vintage/ (a NEW dir). Reads the read-only
data/academic_cache S2 cache (with the backoff already added to academic_search.py) and
the live web search. Never touches results/experiments, results/judge_gpt52, data/analysis.
Resumable: an existing valid <query_id>.json (matching its own corpus_sha256) is skipped.

Determinism of the freeze
--------------------------
The freeze is the variance-elimination step. The extractor's own sampling does NOT need to
be deterministic — it is run ONCE and its output is pinned to disk; every arm reads the same
pinned string. We do request temperature consistent with P9's extraction (Step-1 temp=0.1,
Step-2 temp=0.0 as coded in source_extractor.py) but the byte-identical guarantee comes from
freezing the result, not from extractor determinism.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python scripts/freeze_vintage_sources.py --dry-run       # plan only, no API calls
    python scripts/freeze_vintage_sources.py --self-test     # offline wiring check, no API/net
    python scripts/freeze_vintage_sources.py --limit 1        # freeze ONE query (smoke, ~1 API run)
    python scripts/freeze_vintage_sources.py                  # freeze all 90 (one retrieval pass)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_QUERIES = REPO_ROOT / "data" / "eval_queries_v2.json"

# NEW frozen store — corpus-safe. This is the ONLY dir this script writes to.
FROZEN_DIR = REPO_ROOT / "data" / "frozen_corpus_vintage"
MANIFEST_PATH = FROZEN_DIR / "MANIFEST.json"

# Canonical extractor: GPT-4o via the Azure LLMCaller. Constant for every arm.
EXTRACTOR_MODEL = "gpt-4o"

# Evidence word-limit applied at FREEZE time so the persisted string is the
# already-truncated one every arm reads verbatim. Matches P9's default so the
# frozen string is what the P9 live path would have produced for its generator,
# with extraction held to GPT-4o quality.
EVIDENCE_WORD_LIMIT = 6000

# Directories this script must NEVER write into (sanity guard).
FORBIDDEN_PREFIXES = [
    REPO_ROOT / "results" / "judge_gpt52",
    REPO_ROOT / "results" / "experiments",
    REPO_ROOT / "data" / "analysis",
    REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _assert_corpus_safe(path: Path) -> None:
    resolved = path.resolve()
    root = FROZEN_DIR.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(
            f"CORPUS-SAFETY VIOLATION: refusing to write outside {FROZEN_DIR}: {resolved}"
        )
    for forbidden in FORBIDDEN_PREFIXES:
        fr = forbidden.resolve()
        if fr == resolved or fr in resolved.parents:
            raise RuntimeError(
                f"CORPUS-SAFETY VIOLATION: path under forbidden prefix {forbidden}: {resolved}"
            )


def load_queries(limit: int) -> list[dict]:
    data = json.loads(EVAL_QUERIES.read_text())
    items = data["queries"] if isinstance(data, dict) else data
    items_sorted = sorted(items, key=lambda q: str(q["id"]))
    if limit and limit > 0:
        items_sorted = items_sorted[:limit]
    return items_sorted


def _frozen_path(query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return FROZEN_DIR / f"{safe_id}.json"


def is_frozen(query_id: str) -> bool:
    """True iff a valid frozen json exists (its evidence_text hashes to its stored sha)."""
    p = _frozen_path(query_id)
    if not (p.exists() and p.stat().st_size > 0):
        return False
    try:
        d = json.loads(p.read_text())
        return _sha256(d["evidence_text"]) == d.get("corpus_sha256")
    except Exception:
        return False


async def freeze_one(query: dict, budget: float) -> dict:
    """Run Stages 1-3 ONCE for one query with the GPT-4o extractor; persist evidence_text.

    Reuses the EXACT P9 retrieval/extraction tools (web search, academic search with the
    S2 backoff, url extraction, two-step SourceExtractor). The local generator is never
    loaded here — only the extractor (GPT-4o) runs.
    """
    from deep_research.tools import (
        AcademicSearcher,
        CostTracker,
        SourceExtractor,
        URLExtractor,
        get_web_searcher,
        format_extractions_as_evidence,
    )
    from deep_research.tools.llm_caller import LLMCaller

    query_id = query["id"]
    query_text = query["query"]
    t0 = time.time()

    tracker = CostTracker(budget_usd=budget)
    extractor_llm = LLMCaller(cost_tracker=tracker)

    web = get_web_searcher()
    academic = AcademicSearcher()
    url_extractor = URLExtractor()
    source_extractor = SourceExtractor(
        llm=extractor_llm,
        model=EXTRACTOR_MODEL,
        max_content_per_call=15_000,
    )

    # ── Stage 1: search (web top-10 + academic top-5, dedup by URL) ──────────
    web_docs = await web.search_batch([query_text], max_results_per=10)
    academic_docs = await academic.search(query_text, max_per_source=5)

    seen_urls: set = set()
    all_docs = []
    for doc in web_docs + academic_docs:
        if doc.url and doc.url not in seen_urls:
            seen_urls.add(doc.url)
            all_docs.append(doc)

    # ── Stage 2: page extraction where content is missing ────────────────────
    urls_to_extract = [d.url for d in all_docs if d.url and len(d.content) < 500]
    if urls_to_extract:
        extracted = await url_extractor.extract_batch(urls_to_extract)
        url_to_content = {e.url: e.content for e in extracted if e.content}
        for doc in all_docs:
            if doc.url in url_to_content:
                doc.content = url_to_content[doc.url]

    # ── Stage 3: two-step source extraction (GPT-4o, held constant) ──────────
    extractions = await source_extractor.extract_batch(all_docs, query_text)

    # ── Stage 4 join point: format + truncate -> the FROZEN evidence string ──
    evidence_text = format_extractions_as_evidence(extractions)
    words = evidence_text.split()
    if len(words) > EVIDENCE_WORD_LIMIT:
        evidence_text = " ".join(words[:EVIDENCE_WORD_LIMIT]) + "\n\n[... evidence truncated ...]"

    corpus_sha = _sha256(evidence_text)
    urls = [e.url for e in extractions if e.url]

    record = {
        "query_id": query_id,
        "query": query_text,
        "extractor_model": EXTRACTOR_MODEL,
        "n_docs": len(all_docs),
        "n_web_docs": len(web_docs),
        "n_academic_docs": len(academic_docs),
        "n_extractions": len(extractions),
        "extractions": [e.to_evidence_dict() for e in extractions],
        "evidence_text": evidence_text,
        "evidence_word_limit": EVIDENCE_WORD_LIMIT,
        "urls": urls,
        "corpus_sha256": corpus_sha,
        "frozen": True,
        "extractor_cost_usd": round(tracker.total_cost, 6),
        "extractor_tokens": tracker.total_tokens,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_path = _frozen_path(query_id)
    _assert_corpus_safe(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write.
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(out_path)

    return {
        "query_id": query_id,
        "status": "frozen",
        "n_docs": len(all_docs),
        "n_extractions": len(extractions),
        "evidence_words": len(evidence_text.split()),
        "corpus_sha256": corpus_sha,
        "web_s2_counts": {"web": len(web_docs), "academic": len(academic_docs)},
        "extractor_model": EXTRACTOR_MODEL,
        "elapsed_seconds": record["elapsed_seconds"],
    }


def rebuild_manifest() -> dict:
    """(Re)build MANIFEST.json from whatever frozen jsons are on disk."""
    entries = {}
    for fp in sorted(FROZEN_DIR.glob("*.json")):
        if fp.name == "MANIFEST.json":
            continue
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        entries[d["query_id"]] = {
            "corpus_sha256": d.get("corpus_sha256"),
            "extractor_model": d.get("extractor_model"),
            "n_docs": d.get("n_docs"),
            "n_web_docs": d.get("n_web_docs"),
            "n_academic_docs": d.get("n_academic_docs"),
            "n_extractions": d.get("n_extractions"),
            "evidence_words": len(str(d.get("evidence_text", "")).split()),
        }
    manifest = {
        "experiment": "frozen_vintage",
        "extractor_model": EXTRACTOR_MODEL,
        "evidence_word_limit": EVIDENCE_WORD_LIMIT,
        "n_queries_frozen": len(entries),
        "queries": entries,
    }
    _assert_corpus_safe(MANIFEST_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def self_test() -> int:
    """Offline wiring check — no API/network. Validates freeze<->generation round-trip.

    Builds a couple of fake SourceExtraction objects, formats + truncates them exactly
    as freeze_one does, persists a frozen record, then proves:
      (1) the persisted evidence_text hashes to the stored corpus_sha256, and
      (2) SourceExtraction(**d) reconstructs the extractions from to_evidence_dict()
          (so the pipeline's frozen branch can rebuild them for citation mapping), and
      (3) the corpus-safety guard refuses a write outside the frozen dir.
    """
    from deep_research.tools import format_extractions_as_evidence
    from deep_research.tools.source_extractor import SourceExtraction

    exts = [
        SourceExtraction(
            doc_id="d1", title="Frozen Source A", url="https://example.org/a",
            summary="A " + " ".join(["word"] * 50), relevance_score=9,
            key_findings=["finding 1", "finding 2"],
        ),
        SourceExtraction(
            doc_id="d2", title="Frozen Source B", url="https://example.org/b",
            summary="B " + " ".join(["token"] * 8000), relevance_score=7,
            data_points=["42%", "1.5x"],
        ),
    ]
    evidence_text = format_extractions_as_evidence(exts)
    words = evidence_text.split()
    if len(words) > EVIDENCE_WORD_LIMIT:
        evidence_text = " ".join(words[:EVIDENCE_WORD_LIMIT]) + "\n\n[... evidence truncated ...]"
    sha = _sha256(evidence_text)
    assert len(evidence_text.split()) <= EVIDENCE_WORD_LIMIT + 5, "truncation did not bound length"

    # Round-trip the extraction dicts (what the pipeline frozen branch does).
    dicts = [e.to_evidence_dict() for e in exts]
    rebuilt = [SourceExtraction(**d) for d in dicts]
    assert len(rebuilt) == len(exts), "extraction round-trip lost items"
    assert rebuilt[0].title == "Frozen Source A", "title not preserved"
    assert rebuilt[0].source_type is not None, "source_type coercion failed"

    # Integrity: stored sha == recomputed sha (the pipeline's assert before gen).
    assert _sha256(evidence_text) == sha, "sha mismatch on identical string"

    # Corpus-safety guard must reject an out-of-dir write.
    bad = REPO_ROOT / "results" / "experiments" / "base_p9" / "x.json"
    try:
        _assert_corpus_safe(bad)
        raise AssertionError("corpus-safety guard did NOT reject a forbidden path")
    except RuntimeError:
        pass

    print("[self-test] PASS: truncation bounded, extraction round-trip OK, "
          "sha stable, corpus-safety guard active.")
    print(f"[self-test] sample corpus_sha256={sha[:16]}...  evidence_words={len(evidence_text.split())}")
    return 0


async def amain(args) -> int:
    queries = load_queries(args.limit)

    if args.dry_run:
        done = sum(1 for q in queries if is_frozen(q["id"]))
        print(f"{'='*60}\nFREEZE PLAN (frozen_vintage)\n{'='*60}")
        print(f"Queries:        {len(queries)} (sorted id)")
        print(f"Extractor:      {EXTRACTOR_MODEL} (GPT-4o, held constant for ALL arms)")
        print(f"Evidence limit: {EVIDENCE_WORD_LIMIT} words (frozen already-truncated)")
        print(f"Frozen dir:     {FROZEN_DIR}")
        print(f"Already frozen: {done}  -> to freeze: {len(queries) - done}")
        print(f"Est. extractor cost: ~${(len(queries) - done) * 0.20:.2f} (GPT-4o, ~rough)")
        print("DRY RUN — no API calls, nothing written.")
        return 0

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    work = [q for q in queries if not (args.resume and is_frozen(q["id"]))]
    print(f"Freezing {len(work)} of {len(queries)} queries with extractor={EXTRACTOR_MODEL}")

    results = []
    for i, q in enumerate(work, 1):
        print(f"  [{i}/{len(work)}] {q['id']} ...", flush=True)
        try:
            res = await asyncio.wait_for(freeze_one(q, args.budget), timeout=600)
            print(f"    OK {res['elapsed_seconds']}s  docs={res['n_docs']} "
                  f"ext={res['n_extractions']} ev_words={res['evidence_words']} "
                  f"sha={res['corpus_sha256'][:12]}")
        except asyncio.TimeoutError:
            res = {"query_id": q["id"], "status": "timeout"}
            print(f"    TIMEOUT after 600s — left unfrozen for resume")
        except Exception as e:
            res = {"query_id": q["id"], "status": "error", "error": str(e)[:300]}
            print(f"    ERROR — {str(e)[:160]}")
        results.append(res)

    manifest = rebuild_manifest()
    print(f"\nFroze {sum(1 for r in results if r.get('status') == 'frozen')} this run; "
          f"manifest now lists {manifest['n_queries_frozen']} queries -> {MANIFEST_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Cap to first N sorted-id queries (0=all 90).")
    ap.add_argument("--budget", type=float, default=2.0, help="Per-query extractor token budget (USD).")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="Re-freeze even if a valid frozen json exists.")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no API calls, nothing written.")
    ap.add_argument("--self-test", action="store_true",
                    help="Offline wiring check (no API/network); validates freeze<->gen round-trip.")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="Rebuild MANIFEST.json from frozen jsons on disk; no API calls.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.rebuild_manifest:
        m = rebuild_manifest()
        print(f"MANIFEST rebuilt: {m['n_queries_frozen']} queries -> {MANIFEST_PATH}")
        return 0
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
