#!/usr/bin/env python3
"""Generic STAGING -> canonical_numbers.json merge, atomic, idempotent.

Usage: python merge_staging_key.py <staging_json_path> <canonical_key>

Reads a JSON blob written by one of this repo's STAGING-only build_*.py
scripts (each of which explicitly declines to touch canonical_numbers.json
itself) and assigns it whole under <canonical_key>, using the same
merge-preserving atomic-write pattern as build_numbers.py: load the existing
store, update exactly one top-level key, write to a .tmp file, os.replace.
Never touches sibling keys, so it is safe to call once per key in sequence
from rebuild_all.sh. staging_json_path may be relative (resolved against the
current working directory, i.e. the repo root when run from rebuild_all.sh)
or absolute.
"""
import json
import sys
from pathlib import Path

AN = Path(__file__).resolve().parent
CANONICAL = AN / "canonical_numbers.json"


def main():
    if len(sys.argv) != 3:
        print("usage: merge_staging_key.py <staging_json_path> <canonical_key>", file=sys.stderr)
        return 1
    staging_path = Path(sys.argv[1])
    if not staging_path.is_absolute():
        staging_path = staging_path.resolve()
    key = sys.argv[2]

    staging = json.loads(staging_path.read_text())
    canon = json.loads(CANONICAL.read_text()) if CANONICAL.exists() else {}
    canon[key] = staging

    tmp = CANONICAL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canon, indent=1))
    tmp.replace(CANONICAL)
    print(f"merged {staging_path} -> canonical['{key}'] ({len(canon)} total keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
