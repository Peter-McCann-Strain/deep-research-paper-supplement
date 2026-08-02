#!/usr/bin/env python3
"""E14: Claim-level entailment over the ORACLE reports (retrieval-vs-utilisation arm).

This is the C0/FActScore pipeline (atomic-claim extraction + per-claim entailment)
RE-RUN over the oracle_t1_* reports instead of the base_* reports. Pairing the oracle
verified-factual-accuracy against the base verified-factual-accuracy (the existing
`data/analysis/df_c0_*` parquets) lets A2 decompose the factual channel into:

  * a RETRIEVAL-bound component  (gap between perfect-retrieval oracle and base), and
  * a UTILISATION ceiling        (oracle vfa itself: how well a report grounds its claims
                                  even when every cited source is the cached real page).

Endpoint: PTU gpt-4o (deep_research.config.DEFAULT_MODEL == "gpt-4o",
deployment "sthree-ptu-02", cost_per_1k == 0.0 -> $0 marginal). NO Opus, NO paid judge,
NO local 7B (the 7B is P9/P10's own base model and would be circular over the
oracle_t1_p9 / oracle_t1_p10 reports).

CLOBBER SAFETY: writes to DEDICATED output parquets
    data/analysis/df_e14_oracle_verdicts.parquet
    data/analysis/df_e14_oracle_per_report.parquet
which do NOT exist yet, so the base-pattern df_c0_* parquets are never touched.
`--resume` (default ON) reloads any existing E14 output and skips already-verified
(pattern, query_id) cells, so re-invocation never re-spends or clobbers prior verdicts.

Outputs:
  data/analysis/df_e14_oracle_verdicts.parquet   — per-claim entailment verdicts
  data/analysis/df_e14_oracle_per_report.parquet — per-report verified_factual_accuracy
  reports/phase_e14_oracle/per_pattern_summary.md

Usage:
  ./venv/bin/python scripts/run_e14_oracle_entail.py --patterns all --max-claims 20 --concurrency 3
  ./venv/bin/python scripts/run_e14_oracle_entail.py --dry-run          # plan only, no API calls
  ./venv/bin/python scripts/run_e14_oracle_entail.py --no-resume        # (NOT recommended) full re-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import LLMCaller, CostTracker
from deep_research.evaluation.c0_verifier import verify_report

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "results" / "experiments"
OUT_DIR = ROOT / "reports" / "phase_e14_oracle"
PER_REPORT = ROOT / "data" / "analysis" / "df_e14_oracle_per_report.parquet"
PER_VERDICT = ROOT / "data" / "analysis" / "df_e14_oracle_verdicts.parquet"
URL_INDEX_PATH = ROOT / "data" / "c0_url_index.json"

_url_index: dict[str, dict] | None = None


def _get_url_index() -> dict[str, dict]:
    global _url_index
    if _url_index is None:
        if URL_INDEX_PATH.exists():
            print(f"Loading URL index from {URL_INDEX_PATH} ...")
            _url_index = json.loads(URL_INDEX_PATH.read_text())
            print(f"  loaded {len(_url_index):,} URLs")
        else:
            print(f"WARN: {URL_INDEX_PATH} missing — run scripts/c0_url_index.py first")
            _url_index = {}
    return _url_index


_URL_RE = re.compile(r"https?://[^\s)\]]+")


def _load_sources_for(pattern_dir_name: str, qid: str) -> dict[int, str]:
    """Resolve [N] citation indices -> cached source page text (same idiom as run_c0_verifier)."""
    rep_path = EXP / pattern_dir_name / f"{qid}.md"
    if not rep_path.exists():
        return {}
    text = rep_path.read_text()
    refs: dict[int, str] = {}
    url_index = _get_url_index()

    m = re.search(r"(?:^|\n)#+\s*(?:References|Bibliography|Sources?)\s*\n(.*?)(?=\n#+\s|\Z)",
                  text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return {}
    refs_block = m.group(1)
    for chunk in re.split(r"\n(?=\[\d+\]|\d+\.\s)", refs_block):
        num_match = re.match(r"\s*(?:\[(\d+)\]|(\d+)\.)\s*(.*)", chunk, re.DOTALL)
        if not num_match:
            continue
        idx = int(num_match.group(1) or num_match.group(2))
        body = num_match.group(3).strip()
        urls = _URL_RE.findall(body)
        resolved = ""
        for url in urls:
            url = url.rstrip(".,;)]")
            if url in url_index:
                cached = url_index[url]
                resolved = f"[Cached page] {cached.get('title','')}\n\n{cached.get('content','')}"
                break
        if resolved:
            refs[idx] = resolved[:4000]
        else:
            refs[idx] = body[:1200] if len(body) >= 80 else ""
    return refs


def _discover_oracle_patterns(spec: str) -> list[str]:
    if spec == "all":
        return sorted(d.name for d in EXP.iterdir()
                      if d.is_dir() and re.match(r"^oracle_t1_p\d+$", d.name))
    return [p.strip() for p in spec.split(",") if p.strip()]


def _load_done_cells() -> set[tuple[str, str]]:
    """Resume guard: (pattern, query_id) cells already present in the E14 per-report parquet."""
    if not PER_REPORT.exists():
        return set()
    df = pd.read_parquet(PER_REPORT)
    return set(zip(df.pattern.astype(str), df.query_id.astype(str)))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", default="all",
                        help="Comma-separated oracle_t1_* ids, or 'all' for every oracle_t1_pN dir")
    parser.add_argument("--limit-per-pattern", type=int, default=0)
    parser.add_argument("--max-claims", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="(default) skip (pattern,qid) cells already in the E14 parquet")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan the cell list and print counts; make NO API calls and write nothing.")
    args = parser.parse_args()

    patterns = _discover_oracle_patterns(args.patterns)
    done = _load_done_cells() if args.resume else set()

    # Build the cell list, honouring resume + per-pattern caps.
    cells: list[tuple[str, str]] = []
    for p in patterns:
        pdir = EXP / p
        if not pdir.exists():
            continue
        files = sorted(pdir.glob("*.md"))
        if args.limit_per_pattern:
            files = files[: args.limit_per_pattern]
        for f in files:
            if (p, f.stem) in done:
                continue
            cells.append((p, f.stem))

    print(f"E14 oracle entailment | patterns={len(patterns)} | resume={args.resume} | "
          f"already_done={len(done)} | to_verify={len(cells)} | "
          f"max_claims={args.max_claims} | model={DEFAULT_MODEL} (PTU)")

    if args.dry_run:
        by_pat: dict[str, int] = {}
        for p, _ in cells:
            by_pat[p] = by_pat.get(p, 0) + 1
        print("[DRY RUN] no API calls. Cells to verify per pattern:")
        for p in patterns:
            print(f"  {p}: {by_pat.get(p, 0)}")
        est_calls = len(cells) * (args.max_claims + 6)  # ~max_claims entailment + ~6 extraction calls
        print(f"[DRY RUN] rough PTU call estimate (cap): ~{est_calls:,} "
              f"(<= {len(cells)} reports x ({args.max_claims}+~6) calls); $0 marginal on PTU.")
        return

    if not cells:
        print("Nothing to do (all oracle cells already verified). Resume guard left outputs untouched.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = CostTracker(budget_usd=10.0)
    llm = LLMCaller(cost_tracker=tracker)
    sem = asyncio.Semaphore(args.concurrency)

    new_per_report: list[dict] = []
    new_verdicts: list[dict] = []

    async def _verify(pattern_dir: str, qid: str):
        async with sem:
            rep_path = EXP / pattern_dir / f"{qid}.md"
            text = rep_path.read_text()
            sources = _load_sources_for(pattern_dir, qid)
            try:
                res = await verify_report(
                    llm, pattern_dir, qid, text, sources,
                    model=DEFAULT_MODEL, max_claims=args.max_claims,
                )
                new_per_report.append({
                    "pattern": pattern_dir, "query_id": qid,
                    "n_claims": res.n_claims, "n_supports": res.n_supports,
                    "n_neutral": res.n_neutral, "n_contradicts": res.n_contradicts,
                    "n_no_source": res.n_no_source,
                    "verified_factual_accuracy": res.verified_factual_accuracy,
                })
                for v in res.verdicts:
                    new_verdicts.append({
                        "pattern": pattern_dir, "query_id": qid,
                        "claim": v.claim, "citation_idx": v.citation_idx,
                        "verdict": v.verdict, "evidence_quote": v.evidence_quote,
                    })
                print(f"  [{pattern_dir}/{qid[:30]}] vfa={res.verified_factual_accuracy:.3f} "
                      f"({res.n_supports}/{res.n_claims})", flush=True)
            except Exception as e:
                print(f"  [{pattern_dir}/{qid[:30]}] FAIL: {str(e)[:120]}", flush=True)

    t0 = time.time()
    await asyncio.gather(*[_verify(p, q) for p, q in cells])
    elapsed = time.time() - t0

    # Idempotent merge: union new cells with any prior E14 output, then write.
    df_pr_new = pd.DataFrame(new_per_report)
    df_v_new = pd.DataFrame(new_verdicts)
    if args.resume and PER_REPORT.exists():
        df_pr = pd.concat([pd.read_parquet(PER_REPORT), df_pr_new], ignore_index=True)
        df_pr = df_pr.drop_duplicates(subset=["pattern", "query_id"], keep="last")
    else:
        df_pr = df_pr_new
    if args.resume and PER_VERDICT.exists():
        df_v = pd.concat([pd.read_parquet(PER_VERDICT), df_v_new], ignore_index=True)
    else:
        df_v = df_v_new

    PER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df_pr.to_parquet(PER_REPORT, index=False)
    df_v.to_parquet(PER_VERDICT, index=False)

    by_pat = df_pr.groupby("pattern")["verified_factual_accuracy"].agg(
        ["mean", "median", "std", "count"]).round(3)
    by_pat.to_csv(OUT_DIR / "per_pattern_summary.csv")
    md = ["# E14 — Oracle Verified Factual Accuracy (claim-level entailment, PTU gpt-4o)",
          f"\nTotal oracle reports verified (cumulative): {len(df_pr)} "
          f"(+{len(df_pr_new)} this run) in {elapsed/60:.1f} min",
          "\nPer-pattern verified_factual_accuracy:\n",
          "| Pattern | mean | median | std | N |",
          "|---|---:|---:|---:|---:|"]
    for pat, row in by_pat.sort_values("mean", ascending=False).iterrows():
        md.append(f"| {pat} | {row['mean']:.3f} | {row['median']:.3f} | {row['std']:.3f} | {row['count']:.0f} |")
    (OUT_DIR / "per_pattern_summary.md").write_text("\n".join(md))
    print(f"\nDone. Wrote: {PER_REPORT} | {PER_VERDICT} | {OUT_DIR/'per_pattern_summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
