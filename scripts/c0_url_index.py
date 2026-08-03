#!/usr/bin/env python3
"""Build a URL→content index from the bing_cache + tavily_cache so the C0
verifier can resolve citation URLs to actual page content.

Reads:
  data/bing_cache/*.json   (each file = list of Document-like dicts)
  data/tavily_cache/*.json (same)
  data/academic_cache/*.json (same)

Writes:
  data/c0_url_index.json   (dict: url -> {title, content})

Index is intentionally small (≤ 4KB per content, dropping placeholders) so
it fits in memory at C0 time. Run once; rebuild only when caches grow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHES = [
    ROOT / "data" / "tavily_cache",
    ROOT / "data" / "academic_cache",
    ROOT / "data" / "bing_cache",
]
OUT_PATH = ROOT / "data" / "c0_url_index.json"
MAX_CONTENT_CHARS = 4000


def main():
    index: dict[str, dict] = {}
    for cache_dir in CACHES:
        if not cache_dir.exists():
            continue
        files = list(cache_dir.glob("*.json"))
        print(f"Scanning {cache_dir.name}: {len(files):,} files")
        n_added = 0
        for i, f in enumerate(files):
            try:
                docs = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(docs, list):
                continue
            for d in docs:
                if not isinstance(d, dict):
                    continue
                url = (d.get("url") or "").strip()
                content = (d.get("content") or "").strip()
                if not url or len(content) < 100:
                    continue
                # Keep the longer-content version when same URL seen twice
                existing = index.get(url)
                if existing and len(existing.get("content", "")) >= len(content):
                    continue
                index[url] = {
                    "title": (d.get("title") or "").strip()[:200],
                    "content": content[:MAX_CONTENT_CHARS],
                }
                n_added += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(files):,} files  ({len(index):,} URLs total)")
        print(f"  → added {n_added:,} URLs from {cache_dir.name}; index size {len(index):,}")

    OUT_PATH.write_text(json.dumps(index, indent=None, separators=(",", ":")))
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nWrote {len(index):,} unique URLs → {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
