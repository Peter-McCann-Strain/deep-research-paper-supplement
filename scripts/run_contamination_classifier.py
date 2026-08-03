#!/usr/bin/env python
"""E6 STC-AUDIT — STEP 3: GPT-4o snippet contamination CLASSIFIER (a transform tool, NOT a judge).

Part of E6 (prereg docs/publication/prereg/prereg_E6.md; canonical key `contamination`). Robustness
appendix, NOT a headline (2606.05241 owns the framing).

What this is — and is NOT
-------------------------
This runs ONE GPT-4o pass over logged retrieval snippets and asks a single, narrow,
DETERMINISTIC classification question per snippet: "does this snippet leak benchmark
metadata / question-context / an explicit answer (2606.05241 taxonomy)?". GPT-4o is used
here ONLY as a deterministic transform/classifier tool (temperature 0), NEVER as a quality
judge. GPT-5.2 remains THE one authoritative judge and is untouched by E6. No small/local
model is ever wired as a judge. (Hard constraint; see memory feedback_judge_independence.)

It is a NEW runner variant. It does NOT import or modify scripts/run_gpt52_judge.py (which
HARDCODES JUDGE_OUT=results/judge_gpt52). It writes ONLY under
results/contamination_e6/classifier/ — never to results/judge_gpt52, results/experiments,
data/analysis, or reports/eval_v2/verdicts.

Frozen-before-paid-run contract (prereg requirement)
----------------------------------------------------
The classifier PROMPT (CLASSIFIER_SYSTEM + CLASSIFIER_TEMPLATE), the decision THRESHOLD
(CONTAM_THRESHOLD), the model (gpt-4o), and the temperature (0.0) are FIXED in this file and
echoed verbatim by --dry-run. They must not change between this build/verify pass and the
human-launched paid pass.

Coverage basis (STEP-0 dual-basis decision; authoritative choice recorded in the prereg note
and build_contamination.py header):
  --basis citation : df_citations cited-URL/title/claim_context snippets, ALL 11/12
                     architectures (the uniform rate-regression basis; CITED subset only).
  --basis search   : search.json full extractions (url/title/summary/key_findings), P0/P1/
                     P9/P12 only — the higher-recall 4-pattern sensitivity check.

This workflow runs --dry-run ONLY (zero API calls, zero spend, writes nothing). The free PTU
GPT-4o pass is launched LATER by the human.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python scripts/run_contamination_classifier.py --dry-run                 # plan, $0, no writes
    python scripts/run_contamination_classifier.py --dry-run --basis search  # other basis
    # (human, later) the real free pass — NOT run by this workflow:
    python scripts/run_contamination_classifier.py --basis citation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Repo-root import guard (the detector-panel ModuleNotFoundError lesson). Do NOT remove.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "contamination_e6"
CLASSIFIER_OUT = OUT_DIR / "classifier"

PROTECTED = [
    ROOT / "results" / "judge_gpt52",
    ROOT / "results" / "experiments",
    ROOT / "data" / "analysis",
    ROOT / "reports" / "eval_v2" / "verdicts",
]

# Classifier model = GPT-4o on PTU (free), as a deterministic transform tool ONLY.
CLASSIFIER_MODEL = "gpt-4o"
CLASSIFIER_TEMPERATURE = 0.0
# A snippet is counted contaminated if the classifier's leakage probability >= threshold.
# (The classifier emits a 0/1 label AND a 0-1 probability; the threshold gates the probability
# for a calibrated rate, and ties go to the explicit label.)
CONTAM_THRESHOLD = 0.5

# ─── FROZEN classifier prompt (do NOT change between build-verify and paid pass) ───
CLASSIFIER_SYSTEM = (
    "You are a deterministic data-labelling TOOL implementing the contamination taxonomy of "
    "the paper 'A Survey on Benchmark Contamination' (2606.05241). You are NOT a quality "
    "judge and you do NOT score report quality. For ONE retrieved web snippet you decide "
    "only whether the snippet itself constitutes benchmark-contamination exposure, i.e. it "
    "leaks the evaluation benchmark's metadata, its question/context, or its explicit answer. "
    "Respond with valid JSON only."
)

CLASSIFIER_TEMPLATE = (
    "Decide whether the RETRIEVED SNIPPET below leaks information about the evaluation "
    "benchmark/dataset, per the 2606.05241 taxonomy:\n"
    "  - metadata_leak: the snippet is the benchmark/dataset's own page, card, leaderboard, "
    "paper, or repository (it names or describes the benchmark/dataset itself).\n"
    "  - question_context_leak: the snippet reproduces the benchmark question, its options, "
    "or a near-duplicate Q&A / solutions / answer-key surface for it.\n"
    "  - explicit_answer_leak: the snippet states the benchmark's ground-truth answer, gold "
    "label, or reference solution for the task.\n"
    "A snippet that is ordinary topical source material (an encyclopedia article, a primary "
    "research paper used as evidence, a news page) is NOT contamination — label it clean.\n\n"
    "BENCHMARK CONTEXT (for reference; the snippet is contamination only if it leaks one of "
    "the above about a benchmark of this kind):\n"
    "  source partition: {source}\n\n"
    "RETRIEVED SNIPPET:\n"
    "  url       : {url}\n"
    "  title     : {title}\n"
    "  text      : {text}\n\n"
    "Respond with exactly this JSON object and nothing else:\n"
    '{{\n'
    '  "contaminated": 0 or 1,            // 1 if ANY leak bucket applies, else 0\n'
    '  "probability": 0.0 to 1.0,         // your confidence that it is contamination\n'
    '  "bucket": "metadata_leak" | "question_context_leak" | "explicit_answer_leak" | "none",\n'
    '  "reason": "<=20 words"\n'
    '}}'
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
                f"REFUSING: classifier out {rp} is inside protected path {p}. "
                f"E6 writes ONLY under {CLASSIFIER_OUT}.")
        if _is_relative_to(p, rp):
            raise SystemExit(
                f"REFUSING: out {rp} is a PARENT of protected path {p}.")


def _snippet_text(row: dict) -> str:
    parts = [str(row.get("summary") or ""), str(row.get("key_findings") or "")]
    return " ".join(p for p in parts if p).strip()[:3000]


def build_prompt(row: dict) -> str:
    return CLASSIFIER_TEMPLATE.format(
        source=str(row.get("source") or "unknown"),
        url=str(row.get("url") or "")[:400],
        title=str(row.get("title") or "")[:300],
        text=_snippet_text(row) or "(no snippet text on disk for this basis)",
    )


def parse_label(obj: dict) -> Dict[str, object]:
    """Map a classifier JSON response -> {contaminated, probability, bucket, reason}."""
    try:
        prob = float(obj.get("probability"))
    except (TypeError, ValueError):
        prob = None
    label = obj.get("contaminated")
    try:
        label = int(label)
    except (TypeError, ValueError):
        label = None
    # Decision: explicit label if present, else threshold on probability.
    if label in (0, 1):
        contaminated = label
    elif prob is not None:
        contaminated = int(prob >= CONTAM_THRESHOLD)
    else:
        contaminated = 0
    bucket = str(obj.get("bucket") or "none")
    return {
        "contaminated": int(contaminated),
        "probability": prob if prob is not None else (1.0 if contaminated else 0.0),
        "bucket": bucket,
        "reason": str(obj.get("reason") or "")[:200],
    }


def load_snippets(basis: str) -> "List[dict]":
    import pandas as pd
    fname = {"citation": "snippets_citation.parquet",
             "search": "snippets_search.parquet"}[basis]
    path = OUT_DIR / fname
    if not path.exists():
        raise SystemExit(
            f"snippet table {path} not found — run "
            f"scripts/build_contamination_telemetry.py first.")
    df = pd.read_parquet(path)
    return df.to_dict("records")


async def _classify_all(rows: List[dict], out_path: Path, limit: Optional[int]) -> dict:
    """The REAL pass (human-launched). Uses LLMCaller(gpt-4o) as a transform tool."""
    from deep_research.tools.llm_caller import LLMCaller
    caller = LLMCaller()
    sem = asyncio.Semaphore(5)
    results: List[dict] = []

    async def one(i: int, row: dict):
        async with sem:
            prompt = build_prompt(row)
            try:
                obj = await caller.complete_json(
                    prompt=prompt, model=CLASSIFIER_MODEL,
                    system=CLASSIFIER_SYSTEM, temperature=CLASSIFIER_TEMPERATURE,
                    max_tokens=200)
            except Exception as e:  # never let one snippet kill the pass
                obj = {"contaminated": 0, "probability": 0.0, "bucket": "none",
                       "reason": f"error:{type(e).__name__}"}
            lab = parse_label(obj if isinstance(obj, dict) else {})
            lab.update({
                "row_index": i,
                "pattern": row.get("pattern"),
                "query_id": row.get("query_id") or row.get("trace_query_id"),
                "url": row.get("url"),
                "basis": row.get("basis"),
            })
            results.append(lab)

    work = rows[:limit] if limit else rows
    await asyncio.gather(*(one(i, r) for i, r in enumerate(work)))
    results.sort(key=lambda d: d["row_index"])
    out_path.write_text(json.dumps(
        {"model": CLASSIFIER_MODEL, "temperature": CLASSIFIER_TEMPERATURE,
         "threshold": CONTAM_THRESHOLD, "n": len(results), "labels": results}, indent=2))
    n_contam = sum(r["contaminated"] for r in results)
    return {"n": len(results), "n_contaminated": n_contam, "out": str(out_path)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", choices=["citation", "search"], default="citation",
                    help="snippet basis to classify (default: citation = uniform 11/12-arch)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: classifier/labels_<basis>.json under the new dir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap snippets for a tiny check (the paid pass omits this)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the FROZEN prompt/threshold/model + snippet count; ZERO API "
                         "calls; writes NOTHING")
    args = ap.parse_args(argv)

    out_path = Path(args.out) if args.out else (CLASSIFIER_OUT / f"labels_{args.basis}.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    _assert_output_safe(out_path)

    # Snippet count (cheap; the snippet table is read-only and may be absent pre-STEP-1).
    snip_path = OUT_DIR / {"citation": "snippets_citation.parquet",
                           "search": "snippets_search.parquet"}[args.basis]
    n_snip = "?"
    if snip_path.exists():
        import pandas as pd
        n_snip = len(pd.read_parquet(snip_path))

    print("=" * 70)
    print("E6 STC-AUDIT — STEP 3 GPT-4o contamination CLASSIFIER (transform tool, NOT a judge)")
    print(f"  model              : {CLASSIFIER_MODEL} (PTU; classifier tool only)")
    print(f"  temperature        : {CLASSIFIER_TEMPERATURE} (deterministic)")
    print(f"  decision threshold : {CONTAM_THRESHOLD}")
    print(f"  basis              : {args.basis}  ({snip_path.name})")
    print(f"  snippet count      : {n_snip}")
    print(f"  target out path    : {out_path}")
    print(f"  corpus protected   : {PROTECTED[0]} (NEVER written)")
    print("-" * 70)
    print("FROZEN classifier SYSTEM prompt:")
    print(CLASSIFIER_SYSTEM)
    print("-" * 70)
    print("FROZEN classifier USER template (raw, with {placeholders}):")
    print(CLASSIFIER_TEMPLATE)
    print("-" * 70)
    print("RENDERED example (one synthetic snippet, to show the exact per-call text):")
    print(build_prompt({
        "source": "litqa2",
        "url": "https://huggingface.co/datasets/litqa2",
        "title": "LitQA2 dataset card",
        "summary": "LitQA2 is a scientific literature QA benchmark with reference answers.",
        "key_findings": "",
    }))
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] ZERO API calls, nothing written. "
              "The free PTU GPT-4o pass is launched LATER by the human.")
        return 0

    rows = load_snippets(args.basis)
    CLASSIFIER_OUT.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(_classify_all(rows, out_path, args.limit))
    print(f"classified {res['n']} snippets ({res['n_contaminated']} contaminated) -> {res['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
