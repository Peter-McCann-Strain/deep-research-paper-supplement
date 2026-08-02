#!/usr/bin/env python3
"""Build or verify the public isoquant/claim-type canonical block.

This paper claim was originally staged from a larger archival analysis. The
public supplement ships the compact staging blob, so this script can reproduce
the canonical ``capability_isoquant_and_claimtype`` block without raw report
forests, private packet directories, provider APIs, or local model downloads.

Default mode is check-only. Pass ``--write`` to refresh the canonical store from
``analysis/staging/isoquant_claimtype.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "canonical_numbers.json"
STAGING = HERE / "staging" / "isoquant_claimtype.json"
KEY = "capability_isoquant_and_claimtype"
PUBLIC_SCRIPT = "paper_rebuild/paper_a_bounded_returns/analysis/build_isoquant_claimtype.py"


def _normalise_public_metadata(block: dict[str, Any]) -> dict[str, Any]:
    normalised = json.loads(json.dumps(block))
    meta = dict(normalised.get("_meta", {}))
    meta.update(
        {
            "script": PUBLIC_SCRIPT,
            "provenance": (
                "reads included analysis/staging/isoquant_claimtype.json and merges "
                "it into canonical_numbers.json; no provider APIs, no raw report forests, "
                "no private judge packets, CPU-only"
            ),
            "warning": (
                "Public compact rebuild from included staging data. Raw claim-packet "
                "and archival report inputs are intentionally not shipped."
            ),
        }
    )
    normalised["_meta"] = meta
    return normalised


def load_public_block() -> dict[str, Any]:
    if not STAGING.exists():
        raise FileNotFoundError(f"missing public staging file: {STAGING}")
    return _normalise_public_metadata(json.loads(STAGING.read_text(encoding="utf-8")))


def load_canonical() -> dict[str, Any]:
    if not CANONICAL.exists():
        raise FileNotFoundError(f"missing canonical store: {CANONICAL}")
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def write_canonical(canonical: dict[str, Any]) -> None:
    tmp = CANONICAL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canonical, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, CANONICAL)


def build_report(*, write: bool) -> dict[str, Any]:
    canonical = load_canonical()
    public_block = load_public_block()
    before = canonical.get(KEY)
    matches_before = before == public_block
    if write and not matches_before:
        canonical[KEY] = public_block
        write_canonical(canonical)
    return {
        "key": KEY,
        "status": "success" if write or matches_before else "failed",
        "mode": "write" if write else "check",
        "canonical_matches_public_staging": bool(matches_before or write),
        "canonical_path": str(CANONICAL),
        "staging_path": str(STAGING),
        "script": PUBLIC_SCRIPT,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh canonical_numbers.json")
    args = parser.parse_args(argv)

    report = build_report(write=args.write)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
