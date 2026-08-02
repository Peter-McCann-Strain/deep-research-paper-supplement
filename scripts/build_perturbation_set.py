#!/usr/bin/env python3
"""E13' — Constructed ground-truth perturbation set for detection ROC.

Build a corpus of *deliberately defected* reports whose injected defects are
GOLD LABELS for defect-detection evaluation.  Small models are NEVER used here;
this script only CONSTRUCTS the labelled set.  Downstream, small models appear
solely as binary DETECTORS (did-it-find-the-injected-defect) reported as ROC,
while GPT-5.2 remains the authoritative quality judge.

Pipeline
--------
1. SELECT 60 base reports stratified by (pattern x source) from df_runs, keeping
   only canonical ``base_p*`` patterns with ``report_exists`` under
   ``results/experiments/``.  Selection is deterministic: seeded sampling over
   sorted inputs.
2. ROTATE one defect TYPE per report across the sorted selection:
   (a) numeric_flip       — change k specific numbers/dates to wrong values
   (b) fabricated_citation — insert k well-formed but non-existent cites/DOIs/URLs
   (c) deleted_evidence    — remove the evidence sentence behind k supported claims
   (d) contradiction       — inject k internal contradictions
   k = 3 injected defects per report.
3. TRANSFORM with GPT-4o on the PTU (free, $0) using a DETERMINISTIC, seeded
   recipe (temperature=0, fixed seed derived per-report).  GPT-4o is a TRANSFORM
   TOOL here, never a judge.  It proposes precise, machine-applicable edits as a
   structured JSON edit-list.
4. VERIFY every edit landed and nothing else changed:
   - apply edits by EXACT string replacement (deterministic, anchored),
   - text-diff check: exactly k spans changed, all other bytes identical,
   - one GPT-4o confirm call per edit (did this single edit realise the defect?).
   Reports failing verification are rejected (logged, never written).
5. WRITE to a NEW directory ``reports/perturbation_set/``:
   - ``perturbed/{report_id}.md`` — the defected report,
   - ``ground_truth.jsonl``       — one row per injected defect.

Corpus safety
-------------
Writes ONLY under ``reports/perturbation_set/``.  Never touches
``results/experiments/``, ``results/judge_*/`` or ``data/analysis/*.parquet``.

Determinism & idempotency
--------------------------
Seeded generators over sorted inputs; per-report seed = sha256(report_id, seed).
``--resume`` (default) skips reports already verified+written.  ``--dry-run``
performs selection + planning and prints the manifest WITHOUT any API calls or
writes.  Re-running with the same seed reproduces identical selection and edits.

Self-test
---------
    python scripts/build_perturbation_set.py --dry-run
    python scripts/build_perturbation_set.py --limit 1            # 1 report, live

Full run (DO NOT auto-launch):
    python scripts/build_perturbation_set.py --n 60
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import random
import re
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import deep_research.config as C  # noqa: E402  (loads .env)

# ── Constants ────────────────────────────────────────────────────────────────

DEFECT_TYPES = (
    "numeric_flip",
    "fabricated_citation",
    "deleted_evidence",
    "contradiction",
)
K_DEFECTS = 3                       # injected defects per report
DEFAULT_N = 60                      # base reports to perturb
DEFAULT_SEED = 42
TRANSFORM_MODEL_KEY = "gpt-4o"      # PTU deployment "sthree-ptu-02", $0
TRANSFORM_DEPLOYMENT = "sthree-ptu-02"
MAX_REPORT_CHARS = 48_000          # truncate huge reports for the prompt window

OUT_DIR = ROOT / "reports" / "perturbation_set"
PERTURBED_DIR = OUT_DIR / "perturbed"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth.jsonl"
MANIFEST_PATH = OUT_DIR / "selection_manifest.json"
REJECTS_PATH = OUT_DIR / "rejects.jsonl"

DF_RUNS = ROOT / "data" / "analysis" / "df_runs.parquet"
EVAL_QUERIES = ROOT / "data" / "eval_queries_v2.json"

CANON_BASE = re.compile(r"base_p\d+$")


# ── Selection ────────────────────────────────────────────────────────────────

def _load_source_map() -> Dict[str, str]:
    data = json.loads(EVAL_QUERIES.read_text())
    queries = data["queries"] if isinstance(data, dict) else data
    return {q["id"]: q["source"] for q in queries}


@dataclass
class Selected:
    report_id: str            # stable id: "{pattern}__{query_id}"
    base_pattern: str
    query_id: str
    source: str
    report_path: str
    defect_type: str = ""     # assigned during rotation


def select_reports(n: int, seed: int) -> List[Selected]:
    """Stratified (pattern x source) selection, deterministic over sorted inputs.

    Allocates ``n`` slots proportionally across (pattern, source) cells using a
    deterministic largest-remainder method, then samples within each cell with a
    seeded RNG on the cell's sorted candidate list.  Defect TYPE is rotated by
    cycling DEFECT_TYPES over the globally-sorted final selection so every type
    is evenly represented and the assignment is reproducible.
    """
    import pandas as pd

    df = pd.read_parquet(DF_RUNS)
    src_map = _load_source_map()

    df = df[df["pattern"].astype(str).str.fullmatch(CANON_BASE.pattern.rstrip("$") + "$")]
    df = df[df["report_exists"] == True]  # noqa: E712
    if "excluded_from_analysis" in df.columns:
        df = df[df["excluded_from_analysis"] != True]  # noqa: E712
    df = df[df["report_path"].astype(str).str.contains("/results/experiments/")]

    cand: List[Selected] = []
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        source = src_map.get(qid)
        if source is None:
            continue
        rpath = str(row["report_path"])
        if not Path(rpath).is_file():
            continue
        pat = str(row["pattern"])
        cand.append(
            Selected(
                report_id=f"{pat}__{qid}",
                base_pattern=pat,
                query_id=qid,
                source=source,
                report_path=rpath,
            )
        )

    # Group into (pattern, source) cells, sorted for determinism.
    cells: Dict[Tuple[str, str], List[Selected]] = {}
    for s in cand:
        cells.setdefault((s.base_pattern, s.source), []).append(s)
    for k in cells:
        cells[k].sort(key=lambda s: s.report_id)

    cell_keys = sorted(cells.keys())
    total = sum(len(v) for v in cells.values())
    if total == 0:
        return []
    n = min(n, total)

    # Largest-remainder proportional allocation across cells.
    raw = {k: (len(cells[k]) / total) * n for k in cell_keys}
    alloc = {k: min(int(raw[k]), len(cells[k])) for k in cell_keys}
    assigned = sum(alloc.values())
    remainder = n - assigned
    # Distribute leftover by descending fractional part (ties broken by sorted key).
    frac_order = sorted(
        cell_keys, key=lambda k: (-(raw[k] - int(raw[k])), k)
    )
    i = 0
    while remainder > 0 and frac_order:
        k = frac_order[i % len(frac_order)]
        if alloc[k] < len(cells[k]):
            alloc[k] += 1
            remainder -= 1
        i += 1
        if i > 100_000:  # safety
            break

    chosen: List[Selected] = []
    for k in cell_keys:
        take = alloc[k]
        if take <= 0:
            continue
        pool = cells[k]
        rng = random.Random(f"{seed}|{k[0]}|{k[1]}")
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        for j in idxs[:take]:
            chosen.append(pool[j])

    chosen.sort(key=lambda s: s.report_id)

    # Rotate defect TYPE across the sorted final selection.
    for i, s in enumerate(chosen):
        s.defect_type = DEFECT_TYPES[i % len(DEFECT_TYPES)]
    return chosen


# ── Transform prompts (GPT-4o = deterministic transform tool, NOT a judge) ───

_EDIT_LIST_INSTRUCTIONS = """You are a precise TEXT-TRANSFORM tool, not a judge and not an author.
You will be given a research report and asked to specify EXACTLY {k} edits that
inject defects of ONE type. Return ONLY machine-applicable edits.

HARD RULES
- Produce EXACTLY {k} edits. No more, no fewer.
- Each edit replaces one `original_text` span (verbatim substring of the report)
  with `perturbed_text`. The `original_text` MUST appear character-for-character
  in the report exactly once (choose a long enough span to be unique).
- Edits must be DISJOINT (non-overlapping) and target DIFFERENT parts of the report.
- Change NOTHING except the {k} specified spans. Do not fix typos, reflow, or
  reword anything else.
- `original_text` and `perturbed_text` must differ.
- Keep edits small and surgical (a clause, a number, a citation, one sentence).

Return STRICT JSON:
{{"edits": [{{"original_text": "...", "perturbed_text": "...", "rationale": "..."}}]}}
"""

_DEFECT_SPECS = {
    "numeric_flip": (
        "DEFECT TYPE: numeric_flip.\n"
        "For each edit, pick a specific factual NUMBER or DATE in the report "
        "(a statistic, count, percentage, year, measurement, score) and change it "
        "to a DIFFERENT, plausible-but-WRONG value. Keep surrounding words identical; "
        "alter only the numeric/date token(s). Pick numbers that carry factual weight."
    ),
    "fabricated_citation": (
        "DEFECT TYPE: fabricated_citation.\n"
        "For each edit, INSERT a well-formed but NON-EXISTENT citation, DOI, or URL "
        "into an existing sentence (e.g. append a parenthetical reference, a fake "
        "arXiv id, a fabricated DOI like 10.XXXX/..., or a plausible-looking URL). "
        "The citation must look real but refer to nothing. Choose `original_text` as "
        "an existing sentence and `perturbed_text` as that sentence with the fake "
        "reference inserted. Do NOT change any other content of the sentence."
    ),
    "deleted_evidence": (
        "DEFECT TYPE: deleted_evidence.\n"
        "For each edit, find a claim that is SUPPORTED by an adjacent evidence "
        "sentence (a sentence giving the source, data, quotation, or specific "
        "justification for the claim) and DELETE that evidence sentence, leaving the "
        "claim unsupported. `original_text` = the supported claim sentence immediately "
        "followed by its evidence sentence (verbatim, including the separating space). "
        "`perturbed_text` = the claim sentence ALONE (evidence removed). Remove only "
        "the evidence sentence; keep the claim sentence byte-for-byte."
    ),
    "contradiction": (
        "DEFECT TYPE: contradiction.\n"
        "For each edit, alter one statement so it CONTRADICTS another statement that "
        "remains elsewhere in the report (e.g. reverse a direction, negate a "
        "conclusion, flip a comparison) so the report now asserts two incompatible "
        "things. `original_text` = the sentence to alter; `perturbed_text` = the "
        "contradicting version. Keep the contradiction internally checkable from the "
        "report text alone."
    ),
}

_CONFIRM_INSTRUCTIONS = """You are a verification tool, not a judge of quality.
Given an ORIGINAL text span, a PERTURBED text span, and a DEFECT TYPE, decide a
single binary question: does replacing ORIGINAL with PERTURBED realise EXACTLY a
defect of the stated type (and not some unrelated change)?

DEFECT TYPE MEANINGS
- numeric_flip: a specific number/date was changed to a wrong value.
- fabricated_citation: a non-existent citation/DOI/URL was inserted.
- deleted_evidence: an evidence sentence supporting a claim was removed.
- contradiction: the statement now contradicts the report's other claims.

Return STRICT JSON: {"realised": true|false, "reason": "<one sentence>"}
"""


def _per_report_seed(report_id: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{report_id}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Azure PTU (gpt-4o) transform client — deterministic, self-contained ──────

def _build_ptu_client():
    import httpx
    from openai import AsyncAzureOpenAI

    return AsyncAzureOpenAI(
        api_key=C.AZURE_OPENAI_API_KEY,
        azure_endpoint=C.AZURE_OPENAI_ENDPOINT,
        api_version=C.AZURE_OPENAI_API_VERSION,
        max_retries=0,
        timeout=httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0),
    )


async def _ptu_json(client, sem, messages: List[Dict[str, str]], seed: int,
                    max_tokens: int = 4096) -> Dict[str, Any]:
    """One deterministic JSON transform call on the PTU gpt-4o (temp=0, seeded)."""
    from openai import (
        RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(8):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=TRANSFORM_DEPLOYMENT,
                    messages=messages,
                    temperature=0.0,
                    seed=seed,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except (RateLimitError, APIConnectionError, APITimeoutError,
                    InternalServerError) as e:
                last_exc = e
                await asyncio.sleep(min(2.0 * (2 ** attempt), 30.0))
            except json.JSONDecodeError as e:
                last_exc = e
                await asyncio.sleep(1.0)
    raise RuntimeError(f"PTU transform failed after retries: {last_exc}")


# ── Edit application + verification ──────────────────────────────────────────

@dataclass
class Defect:
    report_id: str
    base_pattern: str
    source: str
    query_id: str
    defect_type: str
    defect_index: int
    location: Dict[str, Any]          # char offsets in the perturbed file
    snippet: str                      # short context window around the edit
    original_text: str
    perturbed_text: str


def _validate_edits(raw_edits: List[Dict[str, Any]], text: str) -> List[Dict[str, str]]:
    """Filter to well-formed, uniquely-anchored, disjoint edits in source order."""
    cleaned: List[Tuple[int, Dict[str, str]]] = []
    used_spans: List[Tuple[int, int]] = []
    for e in raw_edits:
        orig = e.get("original_text", "")
        pert = e.get("perturbed_text", "")
        if not isinstance(orig, str) or not isinstance(pert, str):
            continue
        if not orig or orig == pert:
            continue
        # Must appear exactly once.
        first = text.find(orig)
        if first < 0 or text.find(orig, first + 1) != -1:
            continue
        span = (first, first + len(orig))
        if any(not (span[1] <= a or span[0] >= b) for a, b in used_spans):
            continue  # overlaps a prior edit
        used_spans.append(span)
        cleaned.append((first, {"original_text": orig, "perturbed_text": pert,
                                "rationale": str(e.get("rationale", ""))}))
    cleaned.sort(key=lambda t: t[0])
    return [c for _, c in cleaned]


def _apply_edits(text: str, edits: List[Dict[str, str]]) -> str:
    """Apply non-overlapping edits by exact replacement (left-to-right, once each)."""
    out = text
    # Apply from the end so earlier offsets stay valid.
    located = sorted(
        ((out.find(e["original_text"]), e) for e in edits),
        key=lambda t: t[0], reverse=True,
    )
    for pos, e in located:
        if pos < 0:
            raise RuntimeError("edit anchor disappeared during application")
        out = out[:pos] + e["perturbed_text"] + out[pos + len(e["original_text"]):]
    return out


def _diff_changed_blocks(original: str, perturbed: str) -> List[Tuple[str, str]]:
    """Return the list of (removed, added) replacement blocks between two strings.

    Used to prove that EXACTLY the intended spans changed and nothing else.
    """
    sm = difflib.SequenceMatcher(a=original, b=perturbed, autojunk=False)
    blocks: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        blocks.append((original[i1:i2], perturbed[j1:j2]))
    return blocks


def _verify_diff(original: str, perturbed: str, edits: List[Dict[str, str]]) -> Tuple[bool, str]:
    """Prove that EXACTLY the k intended edits changed — nothing else.

    Strategy (robust to difflib splitting one edit into adjacent sub-blocks):
    1. Locate each intended edit's span in ``original`` (offsets are unique because
       ``_validate_edits`` requires a unique anchor and disjoint spans).
    2. Reconstruct the perturbed text by applying ONLY those spans; assert it
       equals the supplied ``perturbed`` byte-for-byte. This guarantees the
       perturbed text is precisely original-with-the-k-edits and nothing more.
    3. Independently, assert every changed region of the diff (in original-space)
       is contained within exactly one intended edit span, and that each edit span
       carries at least one change. This catches stray edits the model may have
       sneaked outside the declared spans.
    """
    # (1) anchor each edit (unique, disjoint by construction).
    spans: List[Tuple[int, int]] = []
    for e in edits:
        pos = original.find(e["original_text"])
        if pos < 0:
            return False, "intended edit not found in original"
        spans.append((pos, pos + len(e["original_text"])))
    spans.sort()

    # (2) reconstruct using only the intended spans; must equal `perturbed`.
    by_pos = {original.find(e["original_text"]): e for e in edits}
    rebuilt = original
    for pos in sorted(by_pos, reverse=True):
        e = by_pos[pos]
        rebuilt = rebuilt[:pos] + e["perturbed_text"] + rebuilt[pos + len(e["original_text"]):]
    if rebuilt != perturbed:
        return False, "reconstruction from intended edits != perturbed text"

    # (3) every diff change (original-space) inside exactly one edit span.
    sm = difflib.SequenceMatcher(a=original, b=perturbed, autojunk=False)
    touched = [False] * len(spans)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # For insertions i1==i2; treat as a zero-width point at i1.
        lo, hi = i1, max(i2, i1)
        inside = None
        for idx, (a, b) in enumerate(spans):
            if a <= lo and hi <= b:        # change region within an edit span
                inside = idx
                break
        if inside is None:
            return False, f"change outside any intended span at original[{i1}:{i2}]"
        touched[inside] = True
    if not all(touched):
        return False, "an intended edit produced no change"
    return True, "ok"


def _locate(perturbed: str, snippet_text: str, ctx: int = 80) -> Tuple[Dict[str, Any], str]:
    pos = perturbed.find(snippet_text)
    if pos < 0:
        return {"char_start": -1, "char_end": -1}, ""
    start = max(0, pos - ctx)
    end = min(len(perturbed), pos + len(snippet_text) + ctx)
    return ({"char_start": pos, "char_end": pos + len(snippet_text)},
            perturbed[start:end])


async def _confirm_edit(client, sem, defect_type: str, orig: str, pert: str,
                        seed: int) -> Tuple[bool, str]:
    messages = [
        {"role": "system", "content": _CONFIRM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(
            {"defect_type": defect_type, "original": orig, "perturbed": pert},
            ensure_ascii=False)},
    ]
    res = await _ptu_json(client, sem, messages, seed=seed, max_tokens=256)
    return bool(res.get("realised", False)), str(res.get("reason", ""))


async def perturb_one(client, sem, sel: Selected, base_seed: int,
                      dry_run: bool) -> Tuple[Optional[str], List[Defect], Dict[str, Any]]:
    """Returns (perturbed_text or None, defects, reject_info)."""
    text = Path(sel.report_path).read_text()
    prompt_text = text[:MAX_REPORT_CHARS]
    seed = _per_report_seed(sel.report_id, base_seed)

    if dry_run:
        return None, [], {}

    sys_prompt = _EDIT_LIST_INSTRUCTIONS.format(k=K_DEFECTS) + "\n\n" + \
        _DEFECT_SPECS[sel.defect_type]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"REPORT:\n\n{prompt_text}"},
    ]
    plan = await _ptu_json(client, sem, messages, seed=seed, max_tokens=4096)
    raw_edits = plan.get("edits", []) if isinstance(plan, dict) else []
    edits = _validate_edits(raw_edits, text)

    if len(edits) != K_DEFECTS:
        return None, [], {
            "report_id": sel.report_id, "defect_type": sel.defect_type,
            "reason": f"validated {len(edits)} edits, need {K_DEFECTS}",
        }

    perturbed = _apply_edits(text, edits)
    ok, why = _verify_diff(text, perturbed, edits)
    if not ok:
        return None, [], {
            "report_id": sel.report_id, "defect_type": sel.defect_type,
            "reason": f"diff verify failed: {why}",
        }

    # GPT-4o confirm each edit realised the defect.
    defects: List[Defect] = []
    for i, e in enumerate(edits):
        realised, reason = await _confirm_edit(
            client, sem, sel.defect_type, e["original_text"], e["perturbed_text"],
            seed=seed + i + 1,
        )
        if not realised:
            return None, [], {
                "report_id": sel.report_id, "defect_type": sel.defect_type,
                "reason": f"edit {i} not confirmed: {reason}",
            }
        loc, snip = _locate(perturbed, e["perturbed_text"])
        defects.append(Defect(
            report_id=sel.report_id, base_pattern=sel.base_pattern,
            source=sel.source, query_id=sel.query_id,
            defect_type=sel.defect_type, defect_index=i,
            location=loc, snippet=snip,
            original_text=e["original_text"], perturbed_text=e["perturbed_text"],
        ))
    return perturbed, defects, {}


# ── Idempotency helpers ──────────────────────────────────────────────────────

def _load_done_ids() -> set:
    done = set()
    if GROUND_TRUTH_PATH.exists():
        for line in GROUND_TRUTH_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["report_id"])
            except Exception:
                continue
    # Only count as done if the perturbed file also exists.
    return {rid for rid in done if (PERTURBED_DIR / f"{rid}.md").exists()}


# ── Main ─────────────────────────────────────────────────────────────────────

async def run(args) -> int:
    selected = select_reports(args.n, args.seed)
    if args.limit is not None:
        selected = selected[: args.limit]

    # Manifest summary
    from collections import Counter
    by_type = Counter(s.defect_type for s in selected)
    by_pat = Counter(s.base_pattern for s in selected)
    by_src = Counter(s.source for s in selected)
    print(f"[select] {len(selected)} reports | seed={args.seed} k={K_DEFECTS}")
    print(f"[select] defect_type: {dict(sorted(by_type.items()))}")
    print(f"[select] base_pattern: {dict(sorted(by_pat.items()))}")
    print(f"[select] source: {dict(sorted(by_src.items()))}")

    if args.dry_run:
        print("[dry-run] no API calls, no writes. First 5 planned:")
        for s in selected[:5]:
            print(f"  - {s.report_id}  type={s.defect_type}  src={s.source}")
        # Write nothing in dry-run.
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PERTURBED_DIR.mkdir(parents=True, exist_ok=True)

    # Persist selection manifest (idempotent overwrite — selection is deterministic).
    MANIFEST_PATH.write_text(json.dumps(
        {"seed": args.seed, "n": len(selected), "k": K_DEFECTS,
         "defect_types": list(DEFECT_TYPES),
         "selected": [asdict(s) for s in selected]}, indent=2))

    done = _load_done_ids() if args.resume else set()
    todo = [s for s in selected if s.report_id not in done]
    print(f"[run] {len(done)} already done, {len(todo)} to process")

    client = _build_ptu_client()
    sem = asyncio.Semaphore(args.concurrency)

    gt_lines: List[str] = []
    reject_lines: List[str] = []
    n_ok = 0
    n_rej = 0

    async def worker(sel: Selected):
        nonlocal n_ok, n_rej
        try:
            perturbed, defects, reject = await perturb_one(
                client, sem, sel, args.seed, dry_run=False)
        except Exception as e:  # noqa: BLE001
            reject = {"report_id": sel.report_id,
                      "defect_type": sel.defect_type, "reason": f"exception: {e}"}
            perturbed, defects = None, []
        if perturbed is None:
            n_rej += 1
            reject_lines.append(json.dumps(reject, ensure_ascii=False))
            print(f"[reject] {sel.report_id}: {reject.get('reason')}")
            return
        # Write perturbed file.
        (PERTURBED_DIR / f"{sel.report_id}.md").write_text(perturbed)
        for d in defects:
            gt_lines.append(json.dumps(asdict(d), ensure_ascii=False))
        n_ok += 1
        print(f"[ok] {sel.report_id}  type={sel.defect_type}  defects={len(defects)}")

    # Bounded fan-out (semaphore already gates API; gather is fine).
    await asyncio.gather(*(worker(s) for s in todo))

    # Append (idempotent: resume skips already-done; rewrite when not resuming).
    mode_existing = GROUND_TRUTH_PATH.exists()
    if args.resume and mode_existing:
        with GROUND_TRUTH_PATH.open("a") as f:
            for ln in gt_lines:
                f.write(ln + "\n")
        with REJECTS_PATH.open("a") as f:
            for ln in reject_lines:
                f.write(ln + "\n")
    else:
        GROUND_TRUTH_PATH.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""))
        REJECTS_PATH.write_text("\n".join(reject_lines) + ("\n" if reject_lines else ""))

    print(f"[done] perturbed={n_ok} rejected={n_rej} "
          f"gt_rows+={len(gt_lines)} -> {GROUND_TRUTH_PATH}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="base reports to perturb (default 60)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N selected (self-test)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="select + plan only; no API calls, no writes")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="skip reports already written (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="rebuild from scratch (overwrites outputs)")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
