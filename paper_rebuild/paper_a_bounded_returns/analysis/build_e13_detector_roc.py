#!/usr/bin/env python
"""E13' perturbation-detector ROC -> canonical_numbers.json['e13_detector_roc'].

What this lands
---------------
Reads the TWO already-produced, read-only artefacts of the E13' PERTURB-TRUTH track
and reduces them to ONE canonical key, ``e13_detector_roc``:

  1. LOCAL DETECTOR PANEL  (GPU, constructed-ground-truth DETECTORS, NOT judges)
         reports/perturbation_set/detector_results.json
     Per local family (Qwen2.5-7B, Mistral-7B-v0.3, DeepSeek-R1-Distill-Qwen-7B):
     binary defect-DETECTION accuracy + ROC-AUC against injected-defect gold labels,
     per defect_type and pooled, with the pre-registered AUC>=0.60 drop floor.

  2. FRONTIER GPT-5.2 INJECTION-ROC  (cloud judge, the authoritative judge)
         results/judge_gpt52/<pattern>/<qid>.json          (matched CLEAN originals)
         results/judge_gpt52_perturb/<pattern>/<qid>.json   (PERTURBED reports)
     The injection-ROC variant asks: does GPT-5.2's per-dimension score DROP on the
     perturbed report relative to its matched clean original, on exactly the dimension
     the defect targets? Mapping (pre-registered, see prereg_E13prime_injection_roc.md):
         numeric_flip, deleted_evidence, contradiction  ->  factual_accuracy
         fabricated_citation                            ->  citation_quality
     The DETECTION SCORE for the ROC is the score-DROP (orig_dim - pert_dim); the
     gold label is 1 for the perturbed item and 0 for its matched clean original
     (a matched-pair design). We report per-family (== per defect-dimension)
     detection-ROC on factual_accuracy and citation_quality.

This builder PURELY reduces existing JSON; it calls NO model and NO paid API. The
expensive steps (the GPU panel run and the GPT-5.2 perturbed-set judging) are run
SEPARATELY and committed as the two artefacts above; see the run-card / prereg.
Idempotent: re-running with the same inputs reproduces the same key byte-for-byte.

GPT-5.2 perturbed verdicts not yet present -> the gpt52 block is written as
``{"status": "pending", ...}`` and the local panel block is still emitted, so this
builder is safe to wire into rebuild_all.sh before the cloud run completes.

Path note (2026-06-22): the canonical store was MOVED to
paper_rebuild/paper_a_bounded_returns/analysis/. This builder uses the NEW path (unlike the
legacy builders that still hardcode paper_rebuild/paper_a_bounded_returns/analysis and crash on
write); the fix to those is applied separately by the assembler.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = "."
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"

PSET = Path(ROOT) / "reports" / "perturbation_set"
DETECTOR_RESULTS = PSET / "detector_results.json"
GROUND_TRUTH = PSET / "ground_truth.jsonl"

JUDGE_ORIG = Path(ROOT) / "results" / "judge_gpt52"            # matched clean originals
JUDGE_PERT = Path(ROOT) / "results" / "judge_gpt52_perturb"    # perturbed reports

AUC_FLOOR = 0.60
PREREG = "docs/publication/prereg/prereg_E13prime_injection_roc.md"

# Defect -> the GPT-5.2 dimension whose score should DROP when the defect is injected.
DEFECT_DIMENSION = {
    "numeric_flip": "factual_accuracy",
    "deleted_evidence": "factual_accuracy",
    "contradiction": "factual_accuracy",
    "fabricated_citation": "citation_quality",
}


# ── ROC (self-contained; no sklearn dependency at build time) ─────────────────

def roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0 or neg == 0:
        return None
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
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


# ── (1) Local detector panel reduction ───────────────────────────────────────

def reduce_local_panel() -> Dict[str, Any]:
    if not DETECTOR_RESULTS.exists():
        return {"status": "pending",
                "reason": f"{DETECTOR_RESULTS} not found; run scripts/run_detector_panel.py"}
    d = json.loads(DETECTOR_RESULTS.read_text())
    families: Dict[str, Any] = {}
    for fam, res in d.get("results", {}).items():
        per_defect = {}
        for dt, m in res.get("per_defect_type", {}).items():
            per_defect[dt] = {
                "n": m.get("n"), "n_pos": m.get("n_pos"), "n_neg": m.get("n_neg"),
                "accuracy": m.get("accuracy"), "auc": m.get("auc"),
                "above_floor": m.get("above_floor"),
            }
        families[fam] = {
            "n_items": res.get("n_items"),
            "pooled_auc": res.get("pooled_auc"),
            "above_floor": res.get("above_floor"),
            "above_floor_defect_types": res.get("above_floor_defect_types", []),
            "per_defect_type": per_defect,
        }
    return {
        "status": "done",
        "schema": d.get("schema"),
        "seed": d.get("seed"),
        "auc_floor": d.get("auc_floor", AUC_FLOOR),
        "n_items": d.get("n_items"),
        "defect_types": d.get("defect_types"),
        "families": families,
        "dropped_below_floor": d.get("dropped_below_floor", []),
        "all_families_below_floor": all(
            not f.get("above_floor") for f in families.values()) if families else None,
    }


# ── (2) GPT-5.2 injection-ROC reduction (matched-pair score-drop) ─────────────

def _load_gt_pairs() -> List[Dict[str, str]]:
    """One matched pair per (report_id, defect_type): the perturbed report + its
    clean original share base_pattern/query_id; the defect_type fixes the dimension."""
    if not GROUND_TRUTH.exists():
        return []
    seen = set()
    pairs: List[Dict[str, str]] = []
    for line in GROUND_TRUTH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = r.get("report_id"); dt = r.get("defect_type")
        if not rid or dt not in DEFECT_DIMENSION:
            continue
        key = (rid, dt)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "report_id": rid, "defect_type": dt,
            "base_pattern": r.get("base_pattern"), "query_id": r.get("query_id"),
            "dimension": DEFECT_DIMENSION[dt],
        })
    pairs.sort(key=lambda p: (p["defect_type"], p["report_id"]))
    return pairs


def _dim_score(verdict_path: Path, dim: str) -> Optional[float]:
    if not verdict_path.exists():
        return None
    try:
        v = json.loads(verdict_path.read_text())
    except json.JSONDecodeError:
        return None
    cell = v.get("dimensions", {}).get(dim)
    return None if cell is None else cell.get("score")


def reduce_gpt52_injection_roc() -> Dict[str, Any]:
    pairs = _load_gt_pairs()
    if not pairs:
        return {"status": "pending", "reason": "ground_truth.jsonl empty/missing"}
    if not JUDGE_PERT.exists():
        return {"status": "pending",
                "reason": f"{JUDGE_PERT} not found; run GPT-5.2 over the perturbed set "
                          f"(run_gpt52_judge_namespaced.py --judge-out results/judge_gpt52_perturb)"}

    # Build matched-pair detection rows per targeted dimension.
    # label 1 = perturbed item, label 0 = matched clean original; score = score DROP.
    by_dim: Dict[str, List[Tuple[int, float]]] = {"factual_accuracy": [], "citation_quality": []}
    n_pairs_used = 0
    n_pairs_missing = 0
    pert_root_names: List[str] = sorted({p.name for p in JUDGE_PERT.iterdir() if p.is_dir()}) \
        if JUDGE_PERT.exists() else []

    for p in pairs:
        pat, qid, dim = p["base_pattern"], p["query_id"], p["dimension"]
        orig_s = _dim_score(JUDGE_ORIG / pat / f"{qid}.json", dim)
        # Perturbed verdicts are namespaced; accept either flat <pattern> or report_id stems.
        pert_v = JUDGE_PERT / pat / f"{qid}.json"
        if not pert_v.exists():
            # fallback: report_id-keyed staging
            alt = JUDGE_PERT / p["report_id"] / f"{qid}.json"
            pert_v = alt if alt.exists() else pert_v
        pert_s = _dim_score(pert_v, dim)
        if orig_s is None or pert_s is None:
            n_pairs_missing += 1
            continue
        n_pairs_used += 1
        drop = orig_s - pert_s
        by_dim[dim].append((1, drop))    # perturbed = positive, score = the drop
        by_dim[dim].append((0, 0.0))     # matched original = negative, drop baseline 0

    out_dims: Dict[str, Any] = {}
    for dim, rows in by_dim.items():
        if not rows:
            out_dims[dim] = {"n": 0, "auc": None, "above_floor": None}
            continue
        labels = [r[0] for r in rows]
        scores = [r[1] for r in rows]
        auc = roc_auc(labels, scores)
        n_pos = sum(labels)
        out_dims[dim] = {
            "n": len(rows), "n_pos": n_pos, "n_neg": len(rows) - n_pos,
            "mean_drop_pos": round(sum(s for l, s in rows if l == 1) / max(n_pos, 1), 4),
            "auc": None if auc is None else round(auc, 4),
            "above_floor": None if auc is None else bool(auc >= AUC_FLOOR),
            "floor": AUC_FLOOR,
        }

    return {
        "status": "done",
        "design": "matched-pair score-drop (perturbed vs clean original)",
        "defect_dimension_map": DEFECT_DIMENSION,
        "n_pairs": len(pairs),
        "n_pairs_used": n_pairs_used,
        "n_pairs_missing_verdict": n_pairs_missing,
        "per_dimension": out_dims,
        "judge": "gpt-5.2",
        "perturbed_judge_root": str(JUDGE_PERT.relative_to(ROOT)),
    }


def main() -> int:
    local = reduce_local_panel()
    gpt52 = reduce_gpt52_injection_roc()

    out = {
        "_note": (
            "E13' perturbation-detector ROC. Local 7B models appear ONLY as "
            "constructed-ground-truth DETECTORS (binary did-it-find-the-injected-defect, "
            "reported as ROC-AUC vs gold, dropped below AUC 0.60); they are NEVER judges. "
            "GPT-5.2 is the authoritative judge and supplies the injection-ROC (matched-pair "
            "score-drop on the defect-targeted dimension). NO Opus anywhere. "
            "Prereg: " + PREREG + "."
        ),
        "prereg": PREREG,
        "auc_floor": AUC_FLOOR,
        "local_detector_panel": local,
        "gpt52_injection_roc": gpt52,
    }

    cn_path = f"{ANA}/canonical_numbers.json"
    cn = json.load(open(cn_path))
    cn["e13_detector_roc"] = out
    # Atomic write so a crash mid-dump can never truncate the canonical store
    # (lesson learned 2026-06-11).
    tmp = f"{cn_path}.tmp"
    open(tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(tmp, cn_path)
    print(f"[build_e13_detector_roc] wrote canonical key 'e13_detector_roc' "
          f"(local={local.get('status')}, gpt52={gpt52.get('status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
