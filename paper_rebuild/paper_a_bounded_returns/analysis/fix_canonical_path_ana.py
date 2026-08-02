#!/usr/bin/env python3
"""B3_e13_drbrace prerequisite: fix the dead ``ANA`` canonical path.

The canonical store was MOVED by commit 0a80ba6 from
``reports/paper_world_class/analysis/`` to
``paper_rebuild/paper_a_bounded_returns/analysis/``. ``build_judge_vs_human.py`` (and
several siblings) still hard-code ``ANA = f"{ROOT}/reports/paper_world_class/
analysis"``; that directory no longer exists, so every ``--write`` / build()
call crashes at ``json.load(open(f"{ANA}/canonical_numbers.json"))`` before it
can land a single number.

This script rewrites the ``ANA = ...`` assignment line of one target file IN
PLACE to point at the live analysis dir (the directory THIS script lives in),
derived from ``Path(__file__)`` so it never hard-codes the new path either.

It is:
  * idempotent  — re-running on an already-fixed file is a no-op (and says so);
  * surgical    — only the single ``ANA = f"{ROOT}...analysis"`` line changes;
  * read-only-safe to dry-run (``--check`` prints the diff and exits non-zero if
    a fix is still needed, zero if already correct);
  * corpus-safe — it edits a *source script*, never canonical_numbers.json.

Usage:
    # dry run (no write) — exits 1 if a fix is still needed, 0 if already good
    python paper_rebuild/paper_a_bounded_returns/analysis/fix_canonical_path_ana.py --check

    # apply the fix in place (writes the .py, NOT the canonical json)
    python paper_rebuild/paper_a_bounded_returns/analysis/fix_canonical_path_ana.py --apply

    # target a different sibling that has the same bug
    python .../fix_canonical_path_ana.py --apply --target build_judge_vs_human.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # the LIVE analysis dir (paper_a)

# Match the broken assignment regardless of quoting / spacing. Group 1 is the
# left side up to and including '=', so we can rebuild deterministically.
_OLD_RE = re.compile(
    r'^(?P<lhs>ANA\s*=\s*)'
    r'f?["\'].*reports/paper_world_class/analysis["\']\s*$',
    re.MULTILINE,
)

# The canonical correct line: derive ANA from the file's own location so it can
# never drift again, whatever the absolute repo path.
_NEW_LINE = (
    'ANA = str(Path(__file__).resolve().parent)  '
    '# live analysis dir (paper_a); was reports/paper_world_class/analysis (moved 0a80ba6)'
)


def fix_text(text: str) -> tuple[str, int]:
    """Return (new_text, n_subs)."""
    new, n = _OLD_RE.subn(_NEW_LINE, text)
    return new, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="build_judge_vs_human.py",
                    help="script (relative to this analysis dir) whose ANA line to fix")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="dry run: print whether a fix is needed; exit 1 if so")
    g.add_argument("--apply", action="store_true",
                   help="rewrite the target file in place")
    args = ap.parse_args()

    target = (HERE / args.target).resolve()
    if not target.is_file():
        print(f"ERROR: target not found: {target}", file=sys.stderr)
        return 2

    text = target.read_text(encoding="utf-8")
    new, n = fix_text(text)

    if "Path(__file__)" not in (re.search(r'^import .*|^from .*', text) or [""]):
        pass  # build_judge_vs_human already imports Path; documented assumption

    if n == 0:
        # Either already fixed, or the line is not present at all.
        if "reports/paper_world_class/analysis" in text:
            print(f"WARN: found the old path but not the expected ANA line in {target}")
            return 1
        print(f"OK: {target.name} already points ANA at the live analysis dir (no change).")
        return 0

    if args.check or not args.apply:
        print(f"FIX NEEDED in {target.name}: {n} ANA line(s) point at the dead "
              f"reports/paper_world_class/analysis path.")
        print(f"  -> would become: {_NEW_LINE}")
        return 1

    # --apply
    target.write_text(new, encoding="utf-8")
    print(f"FIXED {target}: rewrote {n} ANA line(s) to derive from __file__ "
          f"(live dir = {HERE}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
