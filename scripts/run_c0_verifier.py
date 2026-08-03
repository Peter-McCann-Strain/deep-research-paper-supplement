#!/usr/bin/env python3
"""E9: Run C0 citation-grounded verification across all base + protocol_a reports.

Outputs:
  data/analysis/df_c0_verdicts.parquet  — per-claim verdicts
  data/analysis/df_c0_per_report.parquet — per-report verified_factual_accuracy
  reports/phase14_c0/per_pattern_summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import LLMCaller, CostTracker
from deep_research.evaluation.c0_verifier import verify_report

EXP = Path("results/experiments")
CKPT = Path("checkpoints/experiments")
OUT_DIR = Path("reports/phase14_c0")
PER_REPORT = Path("data/analysis/df_c0_per_report.parquet")
PER_VERDICT = Path("data/analysis/df_c0_verdicts.parquet")
URL_INDEX_PATH = Path("data/c0_url_index.json")

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
    """Resolve citation indices to actual page content via the URL cache index.

    Strategy:
      1. Parse the report's References section to map [N] -> URL.
      2. Look up URL in the global URL index (`data/c0_url_index.json`,
         built from bing_cache + tavily_cache + academic_cache).
      3. Fall back to the title/snippet text if URL not found.
    """
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

    # Split into per-citation chunks, tag with index
    for chunk in re.split(r"\n(?=\[\d+\]|\d+\.\s)", refs_block):
        num_match = re.match(r"\s*(?:\[(\d+)\]|(\d+)\.)\s*(.*)", chunk, re.DOTALL)
        if not num_match:
            continue
        idx = int(num_match.group(1) or num_match.group(2))
        body = num_match.group(3).strip()
        # Prefer cached page content via URL lookup
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
            # Fall back to the title/snippet body itself; downstream entailment
            # will treat short bodies as effectively no-source.
            refs[idx] = body[:1200] if len(body) >= 80 else ""
    return refs


import re


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", default="all",
                        help="Comma-separated experiment_ids, or 'all' for all base + protocol_a_*")
    parser.add_argument("--limit-per-pattern", type=int, default=0,
                        help="Cap reports per pattern (0 = all)")
    parser.add_argument("--max-claims", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover patterns
    if args.patterns == "all":
        patterns = sorted([
            d.name for d in EXP.iterdir()
            if d.is_dir() and (d.name.startswith("base_") or d.name.startswith("protocol_a_"))
        ])
    else:
        patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    print(f"C0 verification: {len(patterns)} patterns, max_claims={args.max_claims}, conc={args.concurrency}")

    tracker = CostTracker(budget_usd=10.0)
    llm = LLMCaller(cost_tracker=tracker)

    sem = asyncio.Semaphore(args.concurrency)
    all_per_report: list[dict] = []
    all_verdicts: list[dict] = []

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
                all_per_report.append({
                    "pattern": pattern_dir, "query_id": qid,
                    "n_claims": res.n_claims, "n_supports": res.n_supports,
                    "n_neutral": res.n_neutral, "n_contradicts": res.n_contradicts,
                    "n_no_source": res.n_no_source,
                    "verified_factual_accuracy": res.verified_factual_accuracy,
                })
                for v in res.verdicts:
                    all_verdicts.append({
                        "pattern": pattern_dir, "query_id": qid,
                        "claim": v.claim, "citation_idx": v.citation_idx,
                        "verdict": v.verdict, "evidence_quote": v.evidence_quote,
                    })
                print(f"  [{pattern_dir}/{qid[:30]}] vfa={res.verified_factual_accuracy:.3f} "
                      f"({res.n_supports}/{res.n_claims})", flush=True)
            except Exception as e:
                print(f"  [{pattern_dir}/{qid[:30]}] FAIL: {str(e)[:120]}", flush=True)

    tasks = []
    for p in patterns:
        pdir = EXP / p
        if not pdir.exists():
            continue
        files = sorted(pdir.glob("*.md"))
        if args.limit_per_pattern:
            files = files[: args.limit_per_pattern]
        for f in files:
            tasks.append(_verify(p, f.stem))

    print(f"Total reports to verify: {len(tasks)}")
    overall_t0 = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - overall_t0

    # Save
    df_pr = pd.DataFrame(all_per_report)
    df_v = pd.DataFrame(all_verdicts)
    PER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df_pr.to_parquet(PER_REPORT, index=False)
    df_v.to_parquet(PER_VERDICT, index=False)

    # Summary
    by_pat = df_pr.groupby("pattern")["verified_factual_accuracy"].agg(["mean", "median", "std", "count"]).round(3)
    by_pat.to_csv(OUT_DIR / "per_pattern_summary.csv")
    md = ["# E9 — C0 Verified Factual Accuracy",
          f"\nTotal reports verified: {len(df_pr)} in {elapsed/60:.1f} min",
          f"\nPer-pattern verified_factual_accuracy:\n",
          "| Pattern | mean | median | std | N |",
          "|---|---:|---:|---:|---:|"]
    for pat, row in by_pat.sort_values("mean", ascending=False).iterrows():
        md.append(f"| {pat} | {row['mean']:.3f} | {row['median']:.3f} | {row['std']:.3f} | {row['count']:.0f} |")
    (OUT_DIR / "per_pattern_summary.md").write_text("\n".join(md))
    print(f"\nDone. Wrote: {OUT_DIR / 'per_pattern_summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
