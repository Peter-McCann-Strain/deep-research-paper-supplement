#!/usr/bin/env python
"""E13' LOCAL DETECTOR PANEL — constructed-ground-truth defect DETECTORS (NOT judges).

What this is
------------
Part of E13' PERTURB-TRUTH (validity without annotators). The perturbation set
(built separately by ``build_perturbation_set.py``) takes real reports and injects
known REFLECT-style defects: ``fact_flip`` (flip a factual claim),
``citation_fabrication`` (invent / mangle a citation), ``evidence_deletion``
(delete a load-bearing evidential passage). The injected defect IS the gold label.

For each LOCAL model family in the panel we run a *binary, locational DETECTION*
task: "Does this report contain a <defect_type> defect, and if so, where?" We then
compute per-family, per-defect_type DETECTION accuracy and ROC-AUC against the gold
injected-defect labels, and drop any family whose AUC < 0.60 (a pre-specified floor).

HARD RULE — NO SMALL-MODEL JUDGES
---------------------------------
These local models appear ONLY as constructed-ground-truth DETECTORS. We report
their DETECTION ROC / accuracy on items whose answer is known by construction. We
NEVER treat their output as an authoritative quality score, never fold their quality
opinions into adjudication, and we drop any family below the floor. GPT-5.2 is the
real judge; GPT-4o may be a deterministic transform tool, never an authoritative
judge. (See memory: feedback_no_small_model_judges.) This script does NOT call any
small model as a scorer — only as a detector with a floor.

This RUNS ON GPU. Chain it AFTER the oracle queue (one model in VRAM at a time;
PYTORCH_ALLOC_CONF=expandable_segments:True; explicit unload between families).
Every family loads in 4-bit nf4 (bf16 compute, double-quant). The panel is three
7B families that each fit a 16GB card in 4-bit (peaks ~14.1-14.7 GiB on an 18k-char
report). phi-4 (14B) and the Llama-8B R1 distill were swapped out because both OOM at
the weights-materialisation step on this card; see DETECTOR_FAMILIES for the swaps.

Determinism
-----------
Inputs are sorted; a single fixed seed; detection temperature pinned to ~0. The
positive-class probability used for ROC is derived deterministically from the model's
YES/NO decision + an optional self-reported confidence token, so re-runs are stable.

Idempotency
-----------
Per-(family, item) detection verdicts are cached as JSONL under
``reports/perturbation_set/detector_cache/<family_slug>.jsonl``. ``--resume`` (default)
skips cached items; ``--fresh`` ignores the cache. The aggregate
``detector_results.json`` is rewritten atomically.

Corpus safety
-------------
Reads the perturbation set + (optionally) the read-only report corpus / rubric.
Writes ONLY under ``reports/perturbation_set/`` (a NEW dir owned by E13'). Never
writes to results/experiments/, results/judge_*/, or data/analysis/*.parquet.

Usage
-----
    [ -f venv/bin/activate ] && source venv/bin/activate
    # tiny self-test (no GPU, no models loaded — exercises plumbing on a synthetic item):
    python scripts/run_detector_panel.py --self-test
    # plan only, no model loads:
    python scripts/run_detector_panel.py --dry-run
    # FULL run (loads each model 4-bit, sequentially) — run AFTER the oracle queue:
    python scripts/run_detector_panel.py
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo root importable when launched as `python scripts/run_detector_panel.py`
# (sys.path[0] would otherwise be scripts/, breaking `from deep_research...`). Matches
# the sibling run_all_experiments.py. This was the ModuleNotFoundError that crashed queue4.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reduce CUDA fragmentation BEFORE torch is imported anywhere (matches LocalLLMCaller).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(".")
PSET_DIR = ROOT / "reports" / "perturbation_set"
GROUND_TRUTH = PSET_DIR / "ground_truth.jsonl"
PERTURBED_DIR = PSET_DIR / "perturbed"          # perturbed/{report_id}.md (defected)
EXPERIMENTS_DIR = ROOT / "results" / "experiments"  # clean originals (READ-ONLY)
CACHE_DIR = PSET_DIR / "detector_cache"
OUT_PATH = PSET_DIR / "detector_results.json"

# ── Detector panel (LOCAL models, 4-bit). These are DETECTORS, never judges. ──
#
# VRAM fit (RTX 5080, 15.47 GiB usable). Each family is loaded in 4-bit nf4 +
# double-quant, ONE at a time, fully unloaded between families. Measured peaks on
# the production path (18k-char report + DETECTOR_SYSTEM, max_new_tokens=256):
#   Qwen2.5-7B-Instruct           14.59 GiB   (fits)
#   Mistral-7B-Instruct-v0.3      14.13 GiB   (fits)
#   DeepSeek-R1-Distill-Qwen-7B   14.74 GiB   (fits)
#
# FAMILY SWAPS (2026-06: OOM on this 16GB card under transformers 5.2 + bnb 0.49):
#   microsoft/phi-4 (14B)  ->  Qwen/Qwen2.5-7B-Instruct
#       phi-4 OOMs at the WEIGHTS-materialisation step even in 4-bit (~14.1 GiB
#       allocated before the KV cache is even built; its 40-layer / 5120-hidden
#       KV cache for a ~6k-token report would need a further ~13-16 GiB). 14B does
#       not fit 16 GiB here. Qwen2.5-7B is P9's base model and definitely fits.
#   deepseek-ai/DeepSeek-R1-Distill-Llama-8B  ->  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#       The Llama-8B distill also OOMs at the weights-materialisation step alone in
#       a fresh process (~14.1 GiB; a transformers-5.2 core_model_loading spike that
#       device_map="auto"/max_memory do not tame for this checkpoint). The Qwen-7B
#       distill is the same R1-distillation lineage on a smaller base and fits.
DETECTOR_FAMILIES: List[str] = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

# Pre-specified detection floor: drop any family below this AUC.
AUC_FLOOR = 0.60

SEED = 1313  # E13'

# REFLECT-style defect types, matching build_perturbation_set.py exactly. Each
# ground_truth.jsonl row is ONE injected defect (a POSITIVE); the clean original
# report (results/experiments/{base_pattern}/{query_id}.md) is the matched NEGATIVE.
DEFECT_TYPES = (
    "numeric_flip",
    "fabricated_citation",
    "deleted_evidence",
    "contradiction",
)

DEFECT_DESCRIPTIONS = {
    "numeric_flip": (
        "a NUMERIC / DATE ERROR: a specific factual number or date (a statistic, "
        "count, percentage, year, measurement, or score) has been changed to a "
        "different, plausible-but-WRONG value"
    ),
    "fabricated_citation": (
        "a FABRICATED CITATION: a well-formed but NON-EXISTENT citation, DOI, arXiv "
        "id, or URL has been inserted into a sentence; it looks real but refers to "
        "nothing"
    ),
    "deleted_evidence": (
        "DELETED EVIDENCE: a claim's supporting evidence sentence (its source, data, "
        "quotation, or specific justification) has been removed, leaving the claim "
        "asserted but unsupported"
    ),
    "contradiction": (
        "an INTERNAL CONTRADICTION: a statement has been altered so the report now "
        "asserts two incompatible things (a reversed direction, negated conclusion, "
        "or flipped comparison) that are checkable from the report text alone"
    ),
}

# Truncate reports so prompts fit local context comfortably (chars, not tokens).
# The builder uses 48k for the transform; detection prompts are tighter to fit 4-bit
# local KV caches on a 16GB GPU.
MAX_REPORT_CHARS = 18000


# ─────────────────────────── ground-truth loading ───────────────────────────


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _first(d: Dict[str, Any], *keys: str, default=None):
    """Return the first present, non-None key from a tolerant set of aliases."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _original_report_path(base_pattern: str, query_id: str,
                          explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the clean original report (the matched negative for a perturbed report)."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / explicit
        if p.exists():
            return p
    if base_pattern and query_id:
        p = EXPERIMENTS_DIR / base_pattern / f"{query_id}.md"
        if p.exists():
            return p
    return None


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    """Build matched POSITIVE/NEGATIVE detection items from ground_truth.jsonl.

    Schema (from build_perturbation_set.py, one row per injected defect = asdict(Defect)):
        report_id, base_pattern, source, query_id, defect_type, defect_index,
        location:{...}, snippet, original_text, perturbed_text

    Every row is a POSITIVE (a defect present in perturbed/{report_id}.md). We
    construct the matched NEGATIVE from the clean original report
    (results/experiments/{base_pattern}/{query_id}.md), which by construction does
    NOT contain that defect. One report may carry K injected defects of a single
    defect_type; we collapse them to one POSITIVE detection item per (report, defect_type)
    (the locational gold is retained as provenance for error analysis).

    Yields detection items of shape:
        {item_id, report_id, defect_type, label(1/0), report_path, report_kind,
         query_id, base_pattern, source, gold_locations:[...]}
    Determinism: sorted by (defect_type, report_id, label-desc).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"perturbation ground truth not found: {path}\n"
            "Build it first with scripts/build_perturbation_set.py, "
            "or pass --self-test for a synthetic dry self-test."
        )

    # Group defect rows by (report_id, defect_type).
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            defect = str(_first(rec, "defect_type", "defect", default="")).strip().lower()
            if defect not in DEFECT_TYPES:
                continue  # unknown -> skip rather than mis-score
            report_id = str(_first(rec, "report_id", "id", default=""))
            if not report_id:
                continue
            key = (report_id, defect)
            g = groups.setdefault(key, {
                "report_id": report_id,
                "defect_type": defect,
                "base_pattern": _first(rec, "base_pattern", "pattern"),
                "query_id": _first(rec, "query_id", "qid"),
                "source": _first(rec, "source"),
                "perturbed_path": _first(rec, "perturbed_path"),
                "original_path": _first(rec, "original_path", "report_path"),
                "gold_locations": [],
            })
            g["gold_locations"].append({
                "defect_index": _first(rec, "defect_index"),
                "location": _first(rec, "location"),
                "snippet": _first(rec, "snippet"),
                "perturbed_text": _first(rec, "perturbed_text"),
            })

    items: List[Dict[str, Any]] = []
    for (report_id, defect), g in groups.items():
        # POSITIVE: the perturbed (defected) report.
        pert = g.get("perturbed_path")
        pert_path = Path(pert) if pert else (PERTURBED_DIR / f"{report_id}.md")
        if not pert_path.is_absolute():
            pert_path = ROOT / pert_path
        # NEGATIVE: the matched clean original.
        orig_path = _original_report_path(
            g.get("base_pattern") or "", g.get("query_id") or "",
            g.get("original_path"))

        if pert_path.exists():
            items.append({
                "item_id": f"{report_id}::{defect}::pos",
                "report_id": report_id, "defect_type": defect, "label": 1,
                "report_path": str(pert_path), "report_kind": "perturbed",
                "query_id": g.get("query_id"), "base_pattern": g.get("base_pattern"),
                "source": g.get("source"), "gold_locations": g["gold_locations"],
            })
        if orig_path is not None and orig_path.exists():
            items.append({
                "item_id": f"{report_id}::{defect}::neg",
                "report_id": report_id, "defect_type": defect, "label": 0,
                "report_path": str(orig_path), "report_kind": "original",
                "query_id": g.get("query_id"), "base_pattern": g.get("base_pattern"),
                "source": g.get("source"), "gold_locations": [],
            })

    items.sort(key=lambda d: (d["defect_type"], d["report_id"], -d["label"]))
    return items


def resolve_report_text(item: Dict[str, Any]) -> str:
    rp = item.get("report_path")
    if rp:
        p = Path(rp)
        if not p.is_absolute():
            p = ROOT / rp
        if p.exists():
            return p.read_text(errors="replace")[:MAX_REPORT_CHARS]
    return ""


# ─────────────────────────── detection prompt ───────────────────────────────


DETECTOR_SYSTEM = (
    "You are a careful error-DETECTOR for research reports. You are NOT scoring "
    "quality. Your only job is to decide, for ONE specific defect type, whether that "
    "defect is present in the report, and if so to point to where. Answer strictly "
    "in the requested JSON format."
)


def build_detection_prompt(report_text: str, defect_type: str) -> str:
    desc = DEFECT_DESCRIPTIONS[defect_type]
    return (
        f"Defect type to look for: **{defect_type}** — {desc}.\n\n"
        "Read the report below. Decide ONLY whether this specific defect type is "
        "present somewhere in it. Do not comment on overall quality, style, or other "
        "kinds of problems.\n\n"
        "Respond with a single JSON object and nothing else:\n"
        '{\n'
        '  "present": "YES" or "NO",   // is the defect present?\n'
        '  "confidence": 0.0 to 1.0,   // how confident you are in your YES/NO\n'
        '  "location": "<short quote or section, or empty if NO>"\n'
        '}\n\n'
        "=== REPORT START ===\n"
        f"{report_text}\n"
        "=== REPORT END ===\n"
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_detection(raw: str) -> Tuple[int, float, str]:
    """Parse a detector response into (decision 0/1, score in [0,1], location).

    The ROC score is a deterministic, monotone function of the YES/NO decision and
    the self-reported confidence, so that score-ordering matches decision-ordering:
      YES -> 0.5 + 0.5*conf  (in [0.5, 1.0])
      NO  -> 0.5 - 0.5*conf  (in [0.0, 0.5])
    Unparseable -> abstain at the midpoint (decision from any YES/NO keyword, else 0).
    """
    text = raw or ""
    obj: Dict[str, Any] = {}
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = {}

    present_raw = obj.get("present") if isinstance(obj, dict) else None
    conf_raw = obj.get("confidence") if isinstance(obj, dict) else None
    location = ""
    if isinstance(obj, dict) and obj.get("location") is not None:
        location = str(obj.get("location"))[:300]

    decision: Optional[int] = None
    if present_raw is not None:
        s = str(present_raw).strip().lower()
        if s in {"yes", "true", "1", "present"}:
            decision = 1
        elif s in {"no", "false", "0", "absent"}:
            decision = 0
    if decision is None:
        # Fall back to scanning free text for a YES/NO keyword.
        low = text.lower()
        if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
            decision = 1
        elif re.search(r"\bno\b", low) and not re.search(r"\byes\b", low):
            decision = 0
        else:
            decision = 0  # conservative default: defect not detected

    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    score = (0.5 + 0.5 * conf) if decision == 1 else (0.5 - 0.5 * conf)
    return decision, score, location


# ─────────────────────────── metrics ────────────────────────────────────────


def roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """ROC-AUC. None if only one class present (undefined)."""
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0 or neg == 0:
        return None
    try:
        from sklearn.metrics import roc_auc_score  # local import: keep --help cheap
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann–Whitney U fallback (rank-based), midrank ties.
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
        auc = (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
        return float(auc)


def accuracy(labels: List[int], decisions: List[int]) -> Optional[float]:
    if not labels:
        return None
    correct = sum(1 for y, d in zip(labels, decisions) if y == d)
    return correct / len(labels)


# ─────────────────────────── cache I/O ──────────────────────────────────────


def cache_path(family: str) -> Path:
    return CACHE_DIR / f"{_slug(family)}.jsonl"


def load_cache(family: str) -> Dict[str, Dict[str, Any]]:
    p = cache_path(family)
    out: Dict[str, Dict[str, Any]] = {}
    if not p.exists():
        return out
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = f"{rec.get('item_id')}|{rec.get('defect_type')}"
            out[key] = rec
    return out


def append_cache(family: str, rec: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(family).open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────────── per-family run ─────────────────────────────────


def run_family(
    family: str,
    items: List[Dict[str, Any]],
    *,
    resume: bool,
    max_new_tokens: int,
    self_test: bool,
) -> Dict[str, Any]:
    """Run one local detector family over all items. Loads/unloads exactly one model."""
    import asyncio

    cache = load_cache(family) if resume else {}
    verdicts: List[Dict[str, Any]] = []

    caller = None
    loaded = False

    async def _detect(prompt: str) -> str:
        return await caller.complete(
            prompt=prompt,
            system=DETECTOR_SYSTEM,
            temperature=0.0,          # deterministic detection
            max_tokens=max_new_tokens,
        )

    try:
        for it in items:
            key = f"{it['item_id']}|{it['defect_type']}"
            if key in cache:
                verdicts.append(cache[key])
                continue

            report_text = resolve_report_text(it)
            prompt = build_detection_prompt(report_text, it["defect_type"])

            if self_test:
                # Synthetic detector: no model loaded. Exercises the full pipeline
                # (load->parse->metrics->floor->write) with well-defined metrics.
                # It detects the defect iff the item is the perturbed report — i.e. a
                # perfect oracle, purely to validate plumbing. NEVER a real result.
                hit = it.get("report_kind") == "perturbed"
                raw = json.dumps({
                    "present": "YES" if hit else "NO",
                    "confidence": 0.9 if hit else 0.85,
                    "location": "999 million parameters" if hit else "",
                })
            else:
                if caller is None:
                    from deep_research.tools.local_llm_caller import LocalLLMCaller
                    print(f"[load] {family} (4-bit) ...", flush=True)
                    caller = LocalLLMCaller(model_id=family, quantize_4bit=True)
                    loaded = True
                raw = asyncio.run(_detect(prompt))

            decision, score, location = parse_detection(raw)
            rec = {
                "item_id": it["item_id"],
                "report_id": it.get("report_id"),
                "defect_type": it["defect_type"],
                "label": it["label"],
                "report_kind": it.get("report_kind"),
                "decision": decision,
                "score": score,
                "location": location,
                "query_id": it.get("query_id"),
                "base_pattern": it.get("base_pattern"),
            }
            verdicts.append(rec)
            if not self_test:
                append_cache(family, rec)
    finally:
        if loaded:
            try:
                from deep_research.tools.local_llm_caller import unload_model
                unload_model()
            except Exception:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Aggregate per defect_type.
    per_defect: Dict[str, Any] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for v in verdicts:
        by_type.setdefault(v["defect_type"], []).append(v)

    family_above_floor_types: List[str] = []
    for dt in sorted(by_type):
        rows = by_type[dt]
        labels = [int(r["label"]) for r in rows]
        decisions = [int(r["decision"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        auc = roc_auc(labels, scores)
        acc = accuracy(labels, decisions)
        above = (auc is not None) and (auc >= AUC_FLOOR)
        if above:
            family_above_floor_types.append(dt)
        per_defect[dt] = {
            "n": len(rows),
            "n_pos": sum(labels),
            "n_neg": len(labels) - sum(labels),
            "accuracy": None if acc is None else round(acc, 4),
            "auc": None if auc is None else round(auc, 4),
            "above_floor": bool(above),
            "floor": AUC_FLOOR,
        }

    # Family-level AUC across all defect types pooled (provenance summary).
    all_labels = [int(v["label"]) for v in verdicts]
    all_scores = [float(v["score"]) for v in verdicts]
    pooled_auc = roc_auc(all_labels, all_scores)
    family_above_floor = (pooled_auc is not None) and (pooled_auc >= AUC_FLOOR)

    return {
        "family": family,
        "n_items": len(verdicts),
        "pooled_auc": None if pooled_auc is None else round(pooled_auc, 4),
        "above_floor": bool(family_above_floor),
        "above_floor_defect_types": family_above_floor_types,
        "floor": AUC_FLOOR,
        "per_defect_type": per_defect,
    }


# ─────────────────────────── self-test fixture ──────────────────────────────


def write_self_test_fixture() -> Path:
    """Write a tiny synthetic perturbation set (real builder schema) to a temp dir.

    Lays out a temp PSET with the same shape build_perturbation_set.py produces:
      <tmp>/ground_truth.jsonl                 (one asdict(Defect) row = a POSITIVE)
      <tmp>/perturbed/<report_id>.md           (defected report)
      <tmp>/experiments/<pattern>/<qid>.md     (clean original = matched NEGATIVE)
    Returns the ground_truth path. Patches module globals so the loader resolves
    paths inside the temp dir (restored is unnecessary; process is short-lived).
    """
    global PSET_DIR, PERTURBED_DIR, EXPERIMENTS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="detector_selftest_"))
    pert_dir = tmp / "perturbed"
    exp_dir = tmp / "experiments" / "selftest_pattern"
    pert_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    report_id = "selftest_pattern__q_selftest"
    qid = "q_selftest"
    orig_text = (
        "# Self-test report\n\n"
        "BERT was released in 2018 and the base model has 110 million parameters. "
        "It is widely used for classification, as shown by benchmark results [1].\n"
    )
    pert_text = (
        "# Self-test report\n\n"
        "BERT was released in 2018 and the base model has 999 million parameters. "
        "It is widely used for classification, as shown by benchmark results [1].\n"
    )
    (exp_dir / f"{qid}.md").write_text(orig_text)
    (pert_dir / f"{report_id}.md").write_text(pert_text)

    gt = tmp / "ground_truth.jsonl"
    row = {
        "report_id": report_id, "base_pattern": "selftest_pattern",
        "source": "selftest", "query_id": qid, "defect_type": "numeric_flip",
        "defect_index": 0,
        "location": {"start": 0, "end": 0}, "snippet": "999 million parameters",
        "original_text": "110 million parameters",
        "perturbed_text": "999 million parameters",
    }
    gt.write_text(json.dumps(row) + "\n")

    # Point the loader at the temp layout.
    PERTURBED_DIR = pert_dir
    EXPERIMENTS_DIR = tmp / "experiments"
    PSET_DIR = tmp
    return gt


# ─────────────────────────── main ───────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground-truth", default=str(GROUND_TRUTH),
                    help="path to perturbation ground_truth.jsonl")
    ap.add_argument("--out", default=str(OUT_PATH),
                    help="path to write detector_results.json")
    ap.add_argument("--families", nargs="*", default=None,
                    help="override the detector family list (model ids)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of detection items (per family) for smoke runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + normalise ground truth and print the plan; no models")
    ap.add_argument("--self-test", action="store_true",
                    help="run the full pipeline on a 2-item synthetic set, no GPU/models")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--resume", dest="resume", action="store_true", default=True,
                     help="skip items already in the per-family cache (default)")
    grp.add_argument("--fresh", dest="resume", action="store_false",
                     help="ignore the cache and recompute every item")
    args = ap.parse_args(argv)

    random.seed(SEED)
    families = args.families or DETECTOR_FAMILIES

    # Resolve ground truth (self-test uses a synthetic fixture).
    if args.self_test:
        gt_path = write_self_test_fixture()
    else:
        gt_path = Path(args.ground_truth)

    try:
        items = load_ground_truth(gt_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.limit is not None:
        items = items[: args.limit]

    # Plan summary.
    by_dt: Dict[str, Dict[str, int]] = {}
    for it in items:
        d = by_dt.setdefault(it["defect_type"], {"pos": 0, "neg": 0})
        d["pos" if it["label"] == 1 else "neg"] += 1
    print("=" * 64)
    print("E13' LOCAL DETECTOR PANEL (detectors, NOT judges)")
    print(f"  ground truth : {gt_path}")
    print(f"  items        : {len(items)}")
    print(f"  defect types : {sorted(by_dt)}")
    for dt in sorted(by_dt):
        print(f"     {dt:22s} pos={by_dt[dt]['pos']:4d}  neg={by_dt[dt]['neg']:4d}")
    print(f"  families     : {families}")
    print(f"  AUC floor    : {AUC_FLOOR} (drop family/defect below)")
    print(f"  out          : {args.out}")
    print(f"  resume       : {args.resume}")
    print("=" * 64)

    if args.dry_run:
        # Validate that both classes exist per defect type; warn if a cell is one-class.
        warned = False
        for dt in sorted(by_dt):
            if by_dt[dt]["pos"] == 0 or by_dt[dt]["neg"] == 0:
                print(f"  WARNING: defect '{dt}' has a single class -> AUC undefined.")
                warned = True
        if not warned:
            print("  plan OK: every defect type has both classes.")
        print("[dry-run] no models loaded, nothing written.")
        return 0

    # Run each family sequentially — exactly one model in VRAM at a time.
    family_results: Dict[str, Any] = {}
    dropped: List[Dict[str, Any]] = []
    for fam in families:
        print(f"\n--- detector family: {fam} ---", flush=True)
        res = run_family(
            fam, items,
            resume=args.resume,
            max_new_tokens=args.max_new_tokens,
            self_test=args.self_test,
        )
        family_results[fam] = res
        # Apply the floor: report drops explicitly (per family + per defect type).
        if not res["above_floor"]:
            dropped.append({"family": fam, "level": "family",
                            "pooled_auc": res["pooled_auc"], "floor": AUC_FLOOR})
            print(f"  DROPPED family (pooled AUC {res['pooled_auc']} < {AUC_FLOOR})")
        for dt, m in res["per_defect_type"].items():
            tag = "OK " if m["above_floor"] else "DROP"
            print(f"  [{tag}] {dt:22s} n={m['n']:4d} acc={m['accuracy']} auc={m['auc']}")
            if not m["above_floor"]:
                dropped.append({"family": fam, "level": "defect_type",
                                "defect_type": dt, "auc": m["auc"], "floor": AUC_FLOOR})

    out_obj = {
        "schema": "e13prime_detector_panel/v1",
        "note": (
            "Local models are constructed-ground-truth DETECTORS, NOT authoritative "
            "quality judges. Reported values are DETECTION accuracy/ROC-AUC against "
            "injected-defect gold labels. Any family or defect cell with AUC < floor "
            "is dropped. GPT-5.2 is the real judge."
        ),
        "seed": SEED,
        "auc_floor": AUC_FLOOR,
        "ground_truth_path": str(gt_path),
        "n_items": len(items),
        "defect_types": sorted(by_dt),
        "families": list(families),
        "results": family_results,
        "dropped_below_floor": dropped,
        "self_test": bool(args.self_test),
    }

    # Self-test writes to a temp out path to keep the corpus pristine.
    out_path = Path(args.out)
    if args.self_test and out_path == OUT_PATH:
        out_path = Path(gt_path).parent / "detector_results.selftest.json"
    atomic_write_json(out_path, out_obj)
    print(f"\nwrote {out_path}")
    if dropped:
        print(f"dropped {len(dropped)} cell(s) below AUC floor {AUC_FLOOR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
