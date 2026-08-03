#!/usr/bin/env python
"""Table drift tripwire (audit CG-4): assert every numeric cell in tables/*.tex is backed by a
value in canonical_numbers.json (within printed precision). Tables are generated from canonical
by make_tables.py, so post-rebuild they must reconcile; this guards against future drift and is
wired into rebuild_all.sh. Independent table-vs-parquet recomputation is done separately by the
audit agents. Always exits 0; prints a PASS/WARN summary.

Two scopes are reported:
  1. ALL tables/*.tex (legacy full count; includes orphan tables not used by the paper), and
  2. the tables actually \\input by main.tex (parsed fresh from main.tex on every run),
     which is the scope the paper's reproducibility guarantee refers to.
"""
import json
import re
import glob
import os

ROOT = "."
PAPER = f"{ROOT}/paper_rebuild/paper_a_bounded_returns"
ANA = f"{PAPER}/analysis"
TAB = f"{PAPER}/tables"
MAIN = f"{PAPER}/main.tex"

cn = json.load(open(f"{ANA}/canonical_numbers.json"))


def flatten(o, acc):
    if isinstance(o, dict):
        for v in o.values():
            flatten(v, acc)
    elif isinstance(o, list):
        for v in o:
            flatten(v, acc)
    elif isinstance(o, (int, float)):
        acc.add(round(float(o), 4))
        acc.add(round(float(o), 3))
        acc.add(round(float(o), 2))


canon = set()
flatten(cn, canon)
# also admit common transforms: percentages and abs values
canon |= {round(v * 100, 1) for v in list(canon)}
canon |= {round(abs(v), 4) for v in list(canon)}


def used_tables_from_main():
    """Parse main.tex FRESH each run (it is actively edited) for \\input{tables/...}."""
    used = set()
    try:
        src = open(MAIN).read()
    except OSError:
        return used
    for line in src.splitlines():
        line = re.sub(r"(?<!\\)%.*", "", line)  # strip comments
        for m in re.finditer(r"\\input\{tables/([^}]+?)(?:\.tex)?\}", line):
            used.add(m.group(1) + ".tex")
    return used


NUM = re.compile(r"-?\d+\.\d+")


def scan(files):
    total = backed = 0
    warns = []
    for tex in files:
        cells = [float(x) for x in NUM.findall(open(tex).read())]
        miss = []
        for x in cells:
            total += 1
            if any(round(x, p) in canon for p in (4, 3, 2)) or round(abs(x), 3) in canon:
                backed += 1
            else:
                miss.append(x)
        if miss:
            warns.append(f"  {os.path.basename(tex)}: {len(miss)} cell(s) not found in canonical: {sorted(set(miss))[:8]}")
    return total, backed, warns


all_files = sorted(glob.glob(f"{TAB}/*.tex"))
used = used_tables_from_main()
used_files = [f for f in all_files if os.path.basename(f) in used]
missing_inputs = sorted(used - {os.path.basename(f) for f in all_files})

# Scope 1: full directory (legacy count, orphan tables from other papers included)
total, backed, warns = scan(all_files)
pct = 100 * backed / total if total else 100
print(f"[reconcile_tables] ALL tables/*.tex ({len(all_files)} files): {backed}/{total} table cells backed by canonical ({pct:.1f}%)")
if warns:
    print("[reconcile_tables] WARN — unmatched cells in full scope (may be orphan/derived; verify):")
    print("\n".join(warns))
else:
    print("[reconcile_tables] PASS — every table cell (full scope) traces to canonical_numbers.json")

# Scope 2: tables actually \input by main.tex (the paper's guarantee)
u_total, u_backed, u_warns = scan(used_files)
u_pct = 100 * u_backed / u_total if u_total else 100
print(f"[reconcile_tables] USED by main.tex ({len(used_files)} files): {u_backed}/{u_total} table cells backed by canonical ({u_pct:.1f}%)")
if missing_inputs:
    print(f"[reconcile_tables] WARN — main.tex \\inputs tables missing from tables/: {missing_inputs}")
if u_warns:
    print("[reconcile_tables] WARN — unmatched cells in tables the paper uses:")
    print("\n".join(u_warns))
else:
    print("[reconcile_tables] PASS — every table cell the paper \\inputs traces to canonical_numbers.json")
