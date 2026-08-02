#!/usr/bin/env python
"""E6 STC-AUDIT — STEP 2: deterministic URL/domain regex contamination gate.

Part of E6 (prereg docs/publication/prereg/prereg_E6.md; canonical key `contamination`). Robustness
appendix, NOT a headline (2606.05241 owns the framing).

What this is
------------
A purely DETERMINISTIC, OFFLINE, ZERO-API-CALL gate that flags a logged snippet as
benchmark-contamination-exposed by URL/domain, implementing the 2606.05241 leakage taxonomy:

  - metadata_host        : the snippet's host is a benchmark/dataset/leaderboard host (the
                           dataset's own arXiv page, HF dataset card, paperswithcode/kaggle
                           leaderboard, the benchmark source hosts for draco / deepsearch_qa /
                           research_qa / litqa2). Retrieving the benchmark's OWN page is the
                           clearest metadata leak.
  - question_context     : the URL path/host signals a Q&A / solutions / answer-key surface
                           (stackexchange/-overflow Q&A, quizlet/chegg/coursehero, '/answers',
                           '/solutions', '/answer-key', 'flashcards').
  - explicit_answer_leak : the URL itself names the dataset AND an answer/solution surface
                           (e.g. a '.../litqa2/.../answers' style path) — the strongest leak.

A snippet may match several buckets; we record the FULL set plus a single ``regex_bucket``
(highest-severity: explicit_answer_leak > question_context > metadata_host) and a boolean
``regex_contaminated``. The domain list and the regexes are FIXED in this file and echoed by
--dry-run so the gate is reproducible and pre-registerable (no model, no network, no clock).

Inputs / outputs (READ-ONLY on parquets; writes ONLY under results/contamination_e6/)
-------------------------------------------------------------------------------------
Reads results/contamination_e6/snippets_citation.parquet (and snippets_search.parquet if
present), produced by build_contamination_telemetry.py. Writes
results/contamination_e6/regex_flags_<basis>.parquet with the per-snippet flag + bucket, and
a regex_summary.json with per-(pattern, bucket) counts and per-host hit tallies.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python scripts/contamination_regex_gate.py --dry-run   # print the fixed taxonomy, no I/O
    python scripts/contamination_regex_gate.py             # flag the snippet tables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Repo-root import guard (the detector-panel ModuleNotFoundError lesson). Do NOT remove.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "contamination_e6"

PROTECTED = [
    ROOT / "results" / "judge_gpt52",
    ROOT / "results" / "experiments",
    ROOT / "data" / "analysis",
    ROOT / "reports" / "eval_v2" / "verdicts",
]

# ─── FIXED benchmark-host domain list (2606.05241 metadata-host bucket) ───
# These hosts publish the benchmarks/datasets/leaderboards themselves; retrieving one of them
# is a metadata-leak signal. Frozen here and echoed by --dry-run for pre-registration.
BENCHMARK_HOSTS = frozenset({
    # paper / dataset / model registries (a benchmark's OWN page lives here)
    "arxiv.org", "ar5iv.org", "ar5iv.labs.arxiv.org",
    "semanticscholar.org", "www.semanticscholar.org",
    "huggingface.co", "www.huggingface.co",
    "paperswithcode.com", "www.paperswithcode.com",
    "kaggle.com", "www.kaggle.com",
    "github.com", "raw.githubusercontent.com", "gist.github.com",
    "openreview.net", "www.openreview.net",
    "zenodo.org", "www.zenodo.org",
    "aclanthology.org", "www.aclanthology.org",
    # the four E6 benchmark source hosts (draco / deepsearch_qa / research_qa / litqa2)
    "deepmind.com", "deepmind.google", "storage.googleapis.com",   # deepsearch_qa
    "futurehouse.org", "www.futurehouse.org",                       # litqa2 (PaperQA/LAB-Bench)
    "github.io",                                                    # research_qa / draco repos
})

# Substrings that, if present in the HOST, mark a benchmark/dataset/leaderboard surface even
# when the exact host is not enumerated above. Matched on the registered domain only.
BENCHMARK_HOST_SUBSTRINGS = (
    "leaderboard", "benchmark", "dataset",
)

# Dataset / benchmark NAME tokens. When one of these appears in the URL it raises the chance
# the page is the benchmark's own surface; combined with an answer-surface token it becomes an
# explicit-answer leak.
BENCHMARK_NAME_TOKENS = (
    "draco", "deepsearch", "deepsearchqa", "researchqa", "research-qa",
    "litqa", "litqa2", "lab-bench", "labbench", "paperqa",
)

# Q&A / answer-key / solutions surfaces (question_context bucket).
QA_HOST_SUBSTRINGS = (
    "stackexchange.com", "stackoverflow.com", "superuser.com", "serverfault.com",
    "quizlet.com", "chegg.com", "coursehero.com", "brainly.com", "brainly.in",
    "studocu.com", "scribd.com", "slader.com", "sparknotes.com", "gradesaver.com",
    "answers.com", "quora.com",
)
QA_PATH_PATTERNS = (
    re.compile(r"/answers?(?:/|$|[-_.?])", re.I),
    re.compile(r"/solutions?(?:/|$|[-_.?])", re.I),
    re.compile(r"answer[-_]?key", re.I),
    re.compile(r"flash[-_]?cards?", re.I),
    re.compile(r"/qa/|/q-and-a/|/questions?-and-answers?/", re.I),
)

# Explicit-answer-leak: an answer/solution surface AND a benchmark name token in the URL.
ANSWER_SURFACE_PATTERNS = (
    re.compile(r"answer", re.I),
    re.compile(r"solution", re.I),
    re.compile(r"ground[-_]?truth", re.I),
    re.compile(r"gold[-_]?(?:answer|label|set)", re.I),
    re.compile(r"reference[-_]?answer", re.I),
)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_output_safe(path: Path) -> None:
    rp = path.resolve()
    for prot in PROTECTED:
        p = prot.resolve()
        if rp == p or _is_relative_to(rp, p):
            raise SystemExit(
                f"REFUSING: output {rp} is inside protected path {p}. "
                f"E6 writes ONLY under {OUT_DIR}.")


def registered_domain(url: str, fallback_domain: str = "") -> str:
    """Best-effort host of a URL; falls back to a provided domain string."""
    try:
        host = urlparse(url if "://" in url else "http://" + url).netloc.lower()
    except Exception:
        host = ""
    host = host.split("@")[-1].split(":")[0]  # strip creds/port
    if not host and fallback_domain:
        host = fallback_domain.strip().lower()
    return host


def classify_url(url: str, domain: str = "") -> Tuple[bool, str, List[str]]:
    """Return (contaminated, top_bucket, all_buckets) for ONE snippet URL/domain.

    Pure function of the URL/domain + the frozen lists above — deterministic and offline.
    """
    host = registered_domain(url, domain)
    full = (url or "").lower()
    buckets: List[str] = []

    # metadata_host
    host_hit = host in BENCHMARK_HOSTS or any(
        host.endswith(h) for h in BENCHMARK_HOSTS)
    host_hit = host_hit or any(s in host for s in BENCHMARK_HOST_SUBSTRINGS)
    if host_hit:
        buckets.append("metadata_host")

    # question_context
    qa_host = any(s in host for s in QA_HOST_SUBSTRINGS)
    qa_path = any(p.search(full) for p in QA_PATH_PATTERNS)
    if qa_host or qa_path:
        buckets.append("question_context")

    # explicit_answer_leak: a benchmark name token AND an answer/solution surface in the URL
    name_hit = any(t in full for t in BENCHMARK_NAME_TOKENS)
    answer_hit = any(p.search(full) for p in ANSWER_SURFACE_PATTERNS)
    if name_hit and answer_hit:
        buckets.append("explicit_answer_leak")

    if not buckets:
        return False, "", []
    # severity order: explicit_answer_leak > question_context > metadata_host
    for b in ("explicit_answer_leak", "question_context", "metadata_host"):
        if b in buckets:
            return True, b, sorted(set(buckets))
    return True, buckets[0], sorted(set(buckets))


def _print_taxonomy() -> None:
    print("FROZEN regex taxonomy (2606.05241 buckets) — no model, no network:")
    print(f"  metadata_host hosts        : {len(BENCHMARK_HOSTS)} fixed "
          f"(arxiv, semanticscholar, huggingface, paperswithcode, kaggle, github, "
          f"openreview, zenodo, aclanthology + 4 benchmark source hosts)")
    print(f"  metadata_host substrings   : {BENCHMARK_HOST_SUBSTRINGS}")
    print(f"  benchmark name tokens      : {BENCHMARK_NAME_TOKENS}")
    print(f"  question_context hosts     : {len(QA_HOST_SUBSTRINGS)} (stackexchange, quizlet, "
          f"chegg, coursehero, quora, ...)")
    print(f"  question_context paths     : /answers /solutions answer-key flashcards /qa/")
    print(f"  explicit_answer_leak       : (benchmark name token) AND (answer/solution/"
          f"ground-truth/gold/reference-answer in URL)")
    print(f"  severity order             : explicit_answer_leak > question_context > "
          f"metadata_host")


def run_basis(in_path: Path, out_dir: Path, basis: str) -> Optional[dict]:
    import pandas as pd
    if not in_path.exists():
        print(f"  [{basis}] input absent ({in_path.name}) -> skipped "
              f"(run build_contamination_telemetry.py first).")
        return None
    df = pd.read_parquet(in_path)
    flags = df.apply(
        lambda r: classify_url(str(r.get("url", "")), str(r.get("domain", ""))), axis=1)
    df["regex_contaminated"] = [f[0] for f in flags]
    df["regex_bucket"] = [f[1] for f in flags]
    df["regex_buckets_all"] = ["|".join(f[2]) for f in flags]
    # fill domain where missing (search basis leaves it blank)
    if "domain" in df.columns:
        df["domain"] = df.apply(
            lambda r: r["domain"] or registered_domain(str(r.get("url", ""))), axis=1)

    out_path = out_dir / f"regex_flags_{basis}.parquet"
    df.to_parquet(out_path, index=False)

    pub = df[df["is_public"]] if "is_public" in df.columns else df
    summary = {
        "basis": basis,
        "n_snippets": int(len(df)),
        "n_public": int(len(pub)),
        "n_contaminated": int(df["regex_contaminated"].sum()),
        "n_contaminated_public": int(pub["regex_contaminated"].sum()),
        "bucket_counts": df["regex_bucket"].value_counts().to_dict(),
        "per_pattern_rate": {
            str(p): round(float(g["regex_contaminated"].mean()), 4)
            for p, g in df.groupby("pattern")
        },
        "top_contaminated_hosts": (
            df.loc[df["regex_contaminated"], "domain"].value_counts().head(25).to_dict()
            if "domain" in df.columns else {}),
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the frozen taxonomy and a couple of self-checks; write NOTHING")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    _assert_output_safe(out_dir)

    print("=" * 70)
    print("E6 STC-AUDIT — STEP 2 regex gate (deterministic, offline, $0)")
    _print_taxonomy()
    # tiny built-in self-checks so --dry-run validates the classifier logic with no inputs
    checks = [
        # benchmark host, no answer surface -> metadata_host
        ("https://arxiv.org/abs/2606.05241", "metadata_host", True),
        ("https://huggingface.co/datasets/litqa2", "metadata_host", True),
        # benchmark name token AND answer-surface token in URL -> explicit_answer_leak
        ("https://quizlet.com/123/litqa2-answers-flash-cards", "explicit_answer_leak", True),
        ("https://example.org/draco/reference-answer-key.json", "explicit_answer_leak", True),
        # Q&A / answer-key surface without a benchmark name -> question_context
        ("https://stackoverflow.com/questions/42/foo", "question_context", True),
        ("https://quizlet.com/123/biology-flash-cards", "question_context", True),
        # ordinary topical source material -> clean
        ("https://en.wikipedia.org/wiki/BERT", "", False),
        ("https://www.nature.com/articles/s41586-021", "", False),
    ]
    print("  self-checks (url -> expected bucket / contaminated):")
    all_ok = True
    for url, exp_bucket, exp_contam in checks:
        contam, bucket, _ = classify_url(url)
        ok = (contam == exp_contam) and (bucket == exp_bucket)
        all_ok = all_ok and ok
        print(f"    [{'OK ' if ok else 'BAD'}] {url[:60]:60s} -> "
              f"{bucket or '(clean)'} / {contam}")
    print("=" * 70)

    if args.dry_run:
        print(f"[dry-run] self-checks {'PASS' if all_ok else 'FAIL'}; no files written.")
        return 0 if all_ok else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for basis, fname in (("citation", "snippets_citation.parquet"),
                         ("search", "snippets_search.parquet")):
        s = run_basis(out_dir / fname, out_dir, basis)
        if s is not None:
            summaries[basis] = s
            print(f"  [{basis}] {s['n_contaminated']}/{s['n_snippets']} flagged "
                  f"({s['n_contaminated_public']}/{s['n_public']} public); "
                  f"buckets={s['bucket_counts']}")
    (out_dir / "regex_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"wrote regex_flags_*.parquet + regex_summary.json under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
