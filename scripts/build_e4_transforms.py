#!/usr/bin/env python3
"""E4 CITE-CAUSAL — Step 1+2+3: stratified sample, deterministic citation transforms,
and content-preservation verification.

Builds a 100-report stratified sample from the READ-ONLY report corpus, applies five
deterministic, content-fixed citation-density transforms (C0..C4), and verifies that
the non-citation prose is byte-identical between every transformed report and its
original (citation markers stripped from BOTH sides).

CONDITIONS (content held fixed; only citation tokens change)
  C0  original          — verbatim copy of the source report
  C1  strip-all         — remove every inline marker + the references/sources list
  C2  halve-density     — drop every other DISTINCT inline marker (seeded sorted)
  C3  double-density    — duplicate each inline marker in place ([3] -> [3][3])
  C4  shuffle-mapping   — permute which ref-id attaches to each inline slot (seeded);
                          density unchanged, claim<->citation mapping scrambled

MARKER STYLES handled by the tokenizer (all observed in the corpus):
  bare        [N]              (p0, p2-p10)
  md-link     [N](#N)          (p1)
  adjacent    [N][M]           (rendered as two tokens)
  comma       [N],[M]          (rendered as two tokens + literal comma)

base_p10 caveat: RL-agent reports carry near-zero inline markers (27/90 have <=3).
C1-C4 are near no-ops there; each transformed report is flagged
``near_null_transform: true`` in the per-report sidecar and in the manifest so the
analysis (Step 6) can down-weight or exclude that arm's density contrast.

ALL writes land under results/experiments_e4_cite/ (a NEW dir). The source corpus
under results/experiments/ is treated strictly READ-ONLY. No API calls here except
the OPTIONAL Step-3 GPT-4o prose-identity classifier (transform/classifier TOOL only,
never a judge), which is gated behind --gpt4o-check and is OFF by default and OFF
entirely under --dry-run.

Usage:
    # zero-API smoke test (tiny sample, scratch out dir)
    python scripts/build_e4_transforms.py --dry-run

    # full build (deterministic; no API unless --gpt4o-check)
    python scripts/build_e4_transforms.py --build

    # build the sample manifest only (Step 1)
    python scripts/build_e4_transforms.py --sample-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))  # MUST: avoids ModuleNotFoundError when run as a script

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_BASE = _REPO_ROOT / "results" / "experiments"        # READ-ONLY source
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"
OUT_DIR = _REPO_ROOT / "results" / "experiments_e4_cite"     # NEW write root
SAMPLE_MANIFEST = OUT_DIR / "sample_manifest.json"
PRESERVATION_REPORT = OUT_DIR / "preservation_report.json"

CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
ARMS = [f"base_p{i}" for i in range(11)]            # p0..p10
SEED = 4  # E4
SAMPLE_N = 100
QUARANTINE = {"82de3e92"}  # see quarantine_82de3e92.md (judge-specific; flagged not dropped)

# Protected paths — never write here (belt-and-braces guard mirrors the namespaced runner)
PROTECTED = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]


def _assert_safe_out(out: Path) -> None:
    out = out.resolve()
    for prot in PROTECTED:
        p = prot.resolve()
        if out == p or _is_rel(out, p) or _is_rel(p, out):
            raise SystemExit(f"REFUSING: output {out} collides with protected path {p}")


def _is_rel(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# ── Citation tokenizer ────────────────────────────────────────────────────────
# A single regex that matches each inline citation SLOT as one token. Order matters:
# the md-link form must be tried before the bare form so we never split [N](#N).
# Adjacent [N][M] and comma [N],[M] are matched as repeated single-id slots, so the
# tokenizer naturally yields one slot per id.
_MD_LINK = r"\[(\d+)\]\(#\d+\)"          # [4](#4)
_BARE = r"\[(\d+)\]"                       # [3]
MARKER_RE = re.compile(rf"(?:{_MD_LINK})|(?:{_BARE})")

# References / sources list header (start of the list section, to EOF or next H1/H2).
_REF_HEADER = re.compile(
    r"(?im)^\s{0,3}#{0,3}\s*(references|sources|bibliography|works cited|citations)\s*:?\s*$"
)


def find_inline_markers(text: str) -> list[tuple[int, int, str, int]]:
    """Return [(start, end, raw, ref_id), ...] for every inline citation slot, in order.

    `raw` is the exact matched substring (so we can replace it verbatim); `ref_id` is
    the integer cited id. Handles md-link, bare, adjacent and comma forms uniformly.
    """
    out = []
    for m in MARKER_RE.finditer(text):
        rid = m.group(1) if m.group(1) is not None else m.group(2)
        out.append((m.start(), m.end(), m.group(0), int(rid)))
    return out


def _strip_markers_only(text: str) -> str:
    """Remove inline citation markers but keep all other prose. Used by BOTH the C1
    transform and the preservation check (so prose comparison ignores markers)."""
    # Replace markers first (longest/md-link-first via MARKER_RE alternation order).
    s = MARKER_RE.sub("", text)
    # Tidy now-orphaned punctuation produced by removing a marker:  "fact ." / "a ,b"
    s = re.sub(r"[ \t]+([.,;:)])", r"\1", s)
    s = re.sub(r"\(\s*\)", "", s)          # empty parens left by a lone ([N](#N))
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def _split_off_references(text: str) -> tuple[str, str]:
    """Split into (body, references_block). references_block is '' if no list found."""
    m = _REF_HEADER.search(text)
    if not m:
        return text, ""
    # The reference list runs from the header to EOF (reports put it last). If a later
    # top-level heading exists, cut there to be safe.
    start = m.start()
    tail = text[m.end():]
    nxt = re.search(r"(?m)^\s{0,3}#{1,2}\s+\S", tail)
    if nxt:
        end = m.end() + nxt.start()
        return text[:start] + text[end:], text[start:end]
    return text[:start], text[start:]


# ── The five conditions ───────────────────────────────────────────────────────

def transform_C0(text: str, rng: random.Random) -> str:
    return text


def transform_C1(text: str, rng: random.Random) -> str:
    """Strip ALL inline markers and the entire references list."""
    body, _refs = _split_off_references(text)
    body = _strip_markers_only(body)
    return body.rstrip() + "\n"


def transform_C2(text: str, rng: random.Random) -> str:
    """Halve density: drop every other DISTINCT inline marker id (seeded sorted)."""
    body, refs = _split_off_references(text)
    markers = find_inline_markers(body)
    distinct = sorted({rid for _, _, _, rid in markers})
    if not distinct:
        return text
    # Deterministic: keep the even-indexed half of the sorted distinct ids.
    drop = {rid for i, rid in enumerate(distinct) if i % 2 == 1}
    # Rebuild body, removing markers whose id is in `drop`.
    parts, last = [], 0
    for s, e, raw, rid in markers:
        parts.append(body[last:s])
        if rid not in drop:
            parts.append(raw)
        last = e
    parts.append(body[last:])
    new_body = _tidy("".join(parts))
    return new_body + refs


def transform_C3(text: str, rng: random.Random) -> str:
    """Double density: duplicate each inline marker in place ([3] -> [3][3])."""
    body, refs = _split_off_references(text)
    markers = find_inline_markers(body)
    parts, last = [], 0
    for s, e, raw, rid in markers:
        parts.append(body[last:s])
        parts.append(raw + raw)
        last = e
    parts.append(body[last:])
    return "".join(parts) + refs


def transform_C4(text: str, rng: random.Random) -> str:
    """Shuffle claim<->citation mapping: permute which ref-id sits in each inline slot.

    Density (#slots) is unchanged; only the id attached to each slot is permuted via a
    seeded permutation of the multiset of ids. Replaces with the SAME marker style that
    occupied the slot (md-link slots stay md-link, bare stay bare) so the prose diff is
    citation-only.
    """
    body, refs = _split_off_references(text)
    markers = find_inline_markers(body)
    if len(markers) < 2:
        return text
    ids = [rid for _, _, _, rid in markers]
    perm = ids[:]
    rng.shuffle(perm)
    if perm == ids and len(set(ids)) > 1:
        perm = perm[1:] + perm[:1]  # force a non-identity perm when possible
    parts, last = [], 0
    for (s, e, raw, _rid), new_id in zip(markers, perm):
        parts.append(body[last:s])
        # Preserve the slot's original style (md-link vs bare) but swap the id.
        if raw.startswith("[") and "](#" in raw:
            parts.append(f"[{new_id}](#{new_id})")
        else:
            parts.append(f"[{new_id}]")
        last = e
    parts.append(body[last:])
    return "".join(parts) + refs


TRANSFORMS = {
    "C0": transform_C0, "C1": transform_C1, "C2": transform_C2,
    "C3": transform_C3, "C4": transform_C4,
}


def _tidy(s: str) -> str:
    s = re.sub(r"[ \t]+([.,;:)])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def _per_report_seed(pattern: str, query_id: str, condition: str) -> int:
    h = hashlib.sha256(f"{SEED}|{pattern}|{query_id}|{condition}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Step 1: stratified sample ─────────────────────────────────────────────────

def load_queries() -> dict[str, dict]:
    data = json.loads(EVAL_QUERIES.read_text())
    return {q["id"]: q for q in data["queries"]}


def build_sample(queries: dict[str, dict], n: int = SAMPLE_N, seed: int = SEED) -> list[dict]:
    """Stratify by (architecture arm, source family). Deterministic, seeded.

    Cells = ARMS x source-families. We round-robin across cells to hit ~n, drawing the
    seeded-shuffled report list within each cell. base_p10 cells are retained but each
    drawn report is tagged near_null if it has <=3 inline markers (the density contrast
    is weak there; the analysis decides to down-weight vs substitute)."""
    rng = random.Random(seed)
    # Build per-cell candidate lists: (arm, source) -> [query_id,...] that exist on disk.
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for arm in ARMS:
        arm_dir = RESULTS_BASE / arm
        if not arm_dir.exists():
            continue
        for f in sorted(arm_dir.glob("*.md")):
            qid = f.stem
            q = queries.get(qid)
            if q is None:
                continue
            cells[(arm, q.get("source", "default"))].append(qid)
    for k in cells:
        rng.shuffle(cells[k])

    ordered_cells = sorted(cells.keys())
    chosen: list[dict] = []
    seen = set()
    # Round-robin draw to keep BOTH arm and source family balanced.
    while len(chosen) < n and ordered_cells:
        progressed = False
        for cell in ordered_cells:
            pool = cells[cell]
            while pool:
                qid = pool.pop()
                key = (cell[0], qid)
                if key in seen:
                    continue
                seen.add(key)
                arm, source = cell
                src_path = RESULTS_BASE / arm / f"{qid}.md"
                text = src_path.read_text(errors="ignore")
                n_markers = len(find_inline_markers(text))
                chosen.append({
                    "pattern": arm,
                    "query_id": qid,
                    "source": source,
                    "src_path": str(src_path.relative_to(_REPO_ROOT)),
                    "n_inline_markers": n_markers,
                    "near_null_transform": (arm == "base_p10" and n_markers <= 3),
                    "quarantine": qid in QUARANTINE,
                })
                progressed = True
                break
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return chosen


def write_sample_manifest(sample: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arm_counts = defaultdict(int)
    src_counts = defaultdict(int)
    near_null = 0
    for r in sample:
        arm_counts[r["pattern"]] += 1
        src_counts[r["source"]] += 1
        near_null += int(r["near_null_transform"])
    payload = {
        "experiment": "E4 CITE-CAUSAL",
        "seed": SEED,
        "n_reports": len(sample),
        "conditions": CONDITIONS,
        "stratified_by": ["architecture_arm", "source_family"],
        "by_arm": dict(sorted(arm_counts.items())),
        "by_source": dict(sorted(src_counts.items())),
        "n_near_null_p10": near_null,
        "quarantine_flagged": sorted({r["query_id"] for r in sample if r["quarantine"]}),
        "reports": sample,
        "notes": (
            "READ-ONLY source: results/experiments/. base_p10 near_null reports have "
            "<=3 inline markers; C1-C4 are near no-ops there (flagged for down-weight)."
        ),
    }
    SAMPLE_MANIFEST.write_text(json.dumps(payload, indent=2))
    print(f"  wrote sample manifest -> {SAMPLE_MANIFEST} (n={len(sample)})")
    print(f"  by arm: {dict(sorted(arm_counts.items()))}")
    print(f"  by source: {dict(sorted(src_counts.items()))}")
    print(f"  near-null p10 reports flagged: {near_null}")


# ── Step 2: emit transformed reports ──────────────────────────────────────────

def emit_transforms(sample: list[dict], out_root: Path) -> list[dict]:
    """Write out_root/{condition}/{pattern}/{query_id}.md for every report x condition."""
    records = []
    for r in sample:
        src = _REPO_ROOT / r["src_path"]
        original = src.read_text(errors="ignore")
        for cond in CONDITIONS:
            rng = random.Random(_per_report_seed(r["pattern"], r["query_id"], cond))
            transformed = TRANSFORMS[cond](original, rng)
            dst = out_root / cond / r["pattern"] / f"{r['query_id']}.md"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(transformed)
            records.append({
                "condition": cond, "pattern": r["pattern"], "query_id": r["query_id"],
                "near_null_transform": r["near_null_transform"],
                "n_markers_in": len(find_inline_markers(original)),
                "n_markers_out": len(find_inline_markers(transformed)),
                "dst": str(dst.relative_to(_REPO_ROOT)),
            })
    return records


# ── Step 3: content-preservation verification ─────────────────────────────────

def verify_preservation(records: list[dict], out_root: Path,
                        gpt4o_check: bool = False) -> dict:
    """Assert the non-citation prose is identical (markers stripped from BOTH sides).

    Gate: any report whose stripped-prose diff is non-empty is recorded as FAILED; the
    caller aborts the build on any failure. C1 additionally strips the references list,
    so for C1 we compare body-without-refs-without-markers on both sides.
    """
    failures = []
    checked = 0
    gpt4o_checks = []
    for rec in records:
        src = RESULTS_BASE / rec["pattern"] / f"{rec['query_id']}.md"
        dst = out_root / rec["condition"] / rec["pattern"] / f"{rec['query_id']}.md"
        orig = src.read_text(errors="ignore")
        new = dst.read_text(errors="ignore")
        if rec["condition"] == "C1":
            o_body, _ = _split_off_references(orig)
            n_body, _ = _split_off_references(new)
            o_norm = _norm(_strip_markers_only(o_body))
            n_norm = _norm(_strip_markers_only(n_body))
        else:
            o_norm = _norm(_strip_markers_only(orig))
            n_norm = _norm(_strip_markers_only(new))
        checked += 1
        if o_norm != n_norm:
            failures.append({
                "condition": rec["condition"], "pattern": rec["pattern"],
                "query_id": rec["query_id"],
                "first_diff": _first_diff(o_norm, n_norm),
            })
    report = {
        "n_checked": checked,
        "n_failures": len(failures),
        "gate_passed": len(failures) == 0,
        "method": (
            "Strip citation markers from BOTH original and transformed (and the refs "
            "list for C1), normalise whitespace, assert byte-equal. Non-empty prose "
            "diff => abort."
        ),
        "failures": failures[:50],
        "gpt4o_prose_identity_check": "skipped (use --gpt4o-check to enable; OFF in --dry-run)",
    }
    if gpt4o_check:
        report["gpt4o_prose_identity_check"] = _gpt4o_prose_check(records, out_root)
    PRESERVATION_REPORT.write_text(json.dumps(report, indent=2))
    return report


def _norm(s: str) -> str:
    """Whitespace-insensitive normalisation for prose equality."""
    return re.sub(r"\s+", " ", s).strip()


def _first_diff(a: str, b: str) -> str:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            lo = max(0, i - 30)
            return f"@{i}: ...{a[lo:i+30]!r} != ...{b[lo:i+30]!r}"
    if len(a) != len(b):
        return f"length differs {len(a)} vs {len(b)}: tail={a[min(len(a),len(b)):][:60]!r}/{b[min(len(a),len(b)):][:60]!r}"
    return ""


def _gpt4o_prose_check(records, out_root) -> dict:
    """OPTIONAL deterministic GPT-4o yes/no 'is the non-citation prose identical?' check.

    GPT-4o is used here strictly as a transform/classifier TOOL (never a judge). Imports
    the project LLM caller lazily so --dry-run / --help never construct a client. Returns
    a summary; any 'no' is surfaced for manual review (the byte-diff gate is authoritative).
    """
    try:
        import asyncio
        from deep_research.tools.llm_caller import LLMCaller  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local config
        return {"status": "unavailable", "error": str(e)[:200]}
    # LLMCaller() takes no model arg; the model is selected PER CALL via complete(model=...).
    # complete() is ASYNC and its message text goes in the positional `prompt` arg (there is
    # no `user=` kwarg). GPT-4o is used here strictly as a transform/classifier TOOL, never a
    # judge: it emits a YES/NO prose-identity flag, not a quality score.
    caller = LLMCaller()
    sysmsg = ("You are a deterministic text classifier. Two snippets differ ONLY in "
              "bracketed citation markers like [3] or [4](#4). Answer strictly 'YES' if "
              "the non-citation prose is identical, else 'NO'.")

    async def _classify_all() -> list[dict]:
        out = []
        for rec in records:
            if rec["condition"] == "C0":
                continue
            src = RESULTS_BASE / rec["pattern"] / f"{rec['query_id']}.md"
            dst = out_root / rec["condition"] / rec["pattern"] / f"{rec['query_id']}.md"
            a = _strip_markers_only(src.read_text(errors="ignore"))[:6000]
            b = _strip_markers_only(dst.read_text(errors="ignore"))[:6000]
            prompt = f"A:\n{a}\n\nB:\n{b}\n\nIdentical prose? YES/NO"
            ans = await caller.complete(prompt=prompt, model="gpt-4o", system=sysmsg,
                                        temperature=0.0, max_tokens=8)
            out.append({"condition": rec["condition"], "pattern": rec["pattern"],
                        "query_id": rec["query_id"], "answer": ans.strip()[:5].upper()})
        return out

    results = asyncio.run(_classify_all())
    n_no = sum(1 for r in results if not r["answer"].startswith("Y"))
    return {"status": "ran", "n_checked": len(results), "n_no": n_no,
            "flagged": [r for r in results if not r["answer"].startswith("Y")][:50]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="Full build: sample + emit C0..C4 + verify preservation.")
    ap.add_argument("--sample-only", action="store_true",
                    help="Step 1 only: write sample_manifest.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Tiny seeded sample (--limit), write to a SCRATCH out dir, ZERO API.")
    ap.add_argument("--limit", type=int, default=6,
                    help="Reports in --dry-run mode (default 6).")
    ap.add_argument("--n", type=int, default=SAMPLE_N, help="Sample size for --build.")
    ap.add_argument("--out", type=str, default=str(OUT_DIR),
                    help="Output root (guarded; default results/experiments_e4_cite).")
    ap.add_argument("--gpt4o-check", action="store_true",
                    help="ALSO run the GPT-4o prose-identity classifier (TOOL, not judge; "
                         "makes API calls; ignored under --dry-run).")
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root

    queries = load_queries()
    print(f"Loaded {len(queries)} queries from {EVAL_QUERIES.name}")

    if args.dry_run:
        out_root = _REPO_ROOT / "results" / "_scratch_e4_dryrun"
        _assert_safe_out(out_root)
        # Clean scratch so the smoke test is reproducible.
        import shutil
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        global SAMPLE_MANIFEST, PRESERVATION_REPORT
        SAMPLE_MANIFEST = out_root / "sample_manifest.json"
        PRESERVATION_REPORT = out_root / "preservation_report.json"
        sample = build_sample(queries, n=args.limit, seed=SEED)
        print(f"[DRY RUN] sample n={len(sample)} -> scratch {out_root}")
        OUT_DIR_BACKUP = OUT_DIR
        # write manifest to scratch
        arm_counts = defaultdict(int)
        for r in sample:
            arm_counts[r["pattern"]] += 1
        SAMPLE_MANIFEST.write_text(json.dumps(
            {"experiment": "E4 CITE-CAUSAL (DRY RUN)", "seed": SEED,
             "n_reports": len(sample), "conditions": CONDITIONS,
             "by_arm": dict(sorted(arm_counts.items())), "reports": sample}, indent=2))
        records = emit_transforms(sample, out_root)
        report = verify_preservation(records, out_root, gpt4o_check=False)
        print(f"[DRY RUN] emitted {len(records)} transformed reports "
              f"({len(sample)} x {len(CONDITIONS)} conditions)")
        print(f"[DRY RUN] preservation gate: "
              f"{'PASS' if report['gate_passed'] else 'FAIL'} "
              f"({report['n_failures']}/{report['n_checked']} prose diffs)")
        print(f"[DRY RUN] wrote {SAMPLE_MANIFEST.name}, {PRESERVATION_REPORT.name} under {out_root}")
        print("[DRY RUN] ZERO API calls. Nothing written outside the scratch dir.")
        return 0 if report["gate_passed"] else 1

    _assert_safe_out(out_root)

    if args.sample_only:
        sample = build_sample(queries, n=args.n, seed=SEED)
        write_sample_manifest(sample)
        return 0

    if args.build:
        sample = build_sample(queries, n=args.n, seed=SEED)
        write_sample_manifest(sample)
        records = emit_transforms(sample, out_root)
        print(f"  emitted {len(records)} transformed reports -> {out_root}/<condition>/<pattern>/")
        report = verify_preservation(records, out_root, gpt4o_check=args.gpt4o_check)
        print(f"  preservation gate: {'PASS' if report['gate_passed'] else 'FAIL'} "
              f"({report['n_failures']}/{report['n_checked']} prose diffs) -> {PRESERVATION_REPORT}")
        if not report["gate_passed"]:
            print("  ABORT: prose diff detected; transforms are not content-fixed for some reports.")
            return 1
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
