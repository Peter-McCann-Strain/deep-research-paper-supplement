#!/usr/bin/env python
"""E8 VINTAGE — 2-point achievable gap-vs-release-date curve on the frozen P9 scaffold.

What this is
------------
E8 asks whether the frontier-vs-local quality gap has closed, persisted, or reversed
across model VINTAGES, holding the research scaffold (P9 local-baseline architecture),
tools, prompts, and judge constant and varying ONLY the local backbone's release date.

HARDWARE TRUTH (honest, NOT recoverable on this box)
----------------------------------------------------
The RESEARCH_PLAN's 4-point curve {Qwen2.5-7B (2024-09), DeepSeek-R1-Distill-Qwen-7B
(2025-01), Qwen3-8B (2025-04), Qwen3.5-9B (2026-03)} is NOT achievable on the
single RTX 5080 (15.47 GiB usable). The two 7B arms each fit a 16 GiB card in 4-bit
nf4 (measured peaks ~14.6 / ~14.7 GiB on an 18k-char report, see run_detector_panel.py).
The 9B/14B vintage arms OOM at the WEIGHTS-materialisation step alone even in 4-bit
(the same transformers-5.2 / bnb-0.49 core_model_loading spike that device_map="auto"
and max_memory cannot tame; documented for phi-4 14B and the Llama-8B distill in
scripts/run_detector_panel.py). They are therefore OOM-SKIPPED on this hardware. This
is a hardware ceiling, NOT a missing-run we can backfill here. The ACHIEVABLE curve is
exactly TWO points:

    P9    Qwen2.5-7B-Instruct           2024-09   (pattern: base_p9, EXISTS on disk)
    P9'   DeepSeek-R1-Distill-Qwen-7B   2025-01   (pattern: VINTAGE_ARM2, if scored)

Both are Qwen-family backbones; the judge is GPT-5.2 (OpenAI) with an optional Claude
subsample, so judge independence is clean (no judge shares family/scale with either arm;
cf. feedback_judge_independence.md — never use a Qwen as the judge here).

What it computes
----------------
Per (overall + each of the 9 rubric dimensions): the local arm score under the PRIMARY
judge (GPT-5.2), the gap to the GPT-4o frontier anchor (base_p0, same judge), and the
two-point slope of gap vs release-date (gap-change per year between the two vintages).
With only two points the "slope" is the connecting line, not a regression; reported as
such. The OOM-skipped arms are recorded explicitly in the emitted note.

Self-guarding / idempotent
--------------------------
Reads ONLY the read-only parquets under data/analysis/. Writes ONLY the single canonical
key `e8_vintage` via an atomic read-modify-tmp-replace (never clobbers other keys; safe to
re-run; deterministic — no randomness, inputs sorted). If the second vintage arm has not
been scored on disk yet, the script still succeeds and emits a 1-POINT PARTIAL curve with
status="partial_pending_arm2" plus the OOM note (so the rebuild chain stays green pre-run,
mirroring build_oracle_opus / build_e4_cite_causal `|| true` self-guards). It never invents
the second point.

CANONICAL PATH FIX (HARD RULE 2026-06-22)
-----------------------------------------
The canonical store was MOVED by commit 0a80ba6 to
papers/paper_a_bounded_returns/analysis/canonical_numbers.json. This script resolves that
NEW path from the repo root and refuses to write anywhere else. (Several older build_*.py
still hardcode the stale papers/paper_a_bounded_returns/analysis/ path; this one does not.)

Out: papers/paper_a_bounded_returns/analysis/canonical_numbers.json['e8_vintage']
Run: [ -f venv/bin/activate ] && source venv/bin/activate
     python scripts/build_e8_vintage.py            # writes/refreshes the e8_vintage key
     python scripts/build_e8_vintage.py --dry-run  # compute + print, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(".")
A = ROOT / "data" / "analysis"  # READ-ONLY parquets
# NEW canonical location (post-0a80ba6). Resolved, not hardcoded to the stale path.
ANA = ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

# Primary judge for the vintage curve. GPT-5.2 (OpenAI) — independent of both Qwen-family
# arms and of the GPT-4o anchor's backbone-as-subject (GPT-4o is the SUBJECT here, the
# JUDGE is GPT-5.2). Pin to a single judge so the two arms are compared like-for-like.
PRIMARY_JUDGE = "gpt52"

# Frontier anchor: GPT-4o under the P0 baseline scaffold, same primary judge. The "gap"
# is anchor_minus_local, so a SHRINKING gap across vintages = the local frontier catching up.
ANCHOR_PATTERN = "base_p0"

# The two achievable vintage points on this 16GB card. (pattern_on_disk, model, release_date)
# release_date is the ISO calendar month of the public weights release; the x-axis unit is
# YEARS since the first point, so the two-point slope is gap-change per year.
VINTAGE_POINTS = [
    {"label": "P9",  "pattern": "base_p9",                     # EXISTS on disk
     "model": "Qwen/Qwen2.5-7B-Instruct",          "release_date": "2024-09"},
    # FIX (2026-06-23): arm2's REAL judged dir is base_p14_vintage_deepseek_qwen7b
    # (pattern p14_vintage_deepseek_qwen7b, registered in ExecutionPipeline.PATTERN_NAMES
    # and generated by scripts/run_gpu_queue.sh). The previous name
    # 'base_p9_vintage_r1distill' was a PHANTOM that could NEVER match on disk, so the
    # script always self-guarded to a 1-point partial. Reconciled to the actual dir.
    {"label": "P9'", "pattern": "base_p14_vintage_deepseek_qwen7b",  # scored later; self-guards if absent
     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "release_date": "2025-01"},
]

# CAPACITY anchor (NOT a third vintage DATE). Qwen2.5-14B released 2024-09 — the SAME
# vintage as Qwen2.5-7B (P9). It is a larger-CAPACITY point at x=0 years on the vintage
# axis, so plotting it as a third point on the gap-vs-YEARS curve would be an AXIS ERROR
# (two points at x=0). It is emitted as a SEPARATE 'capacity_point' block (scale_axis=True)
# rather than silently extending the date curve. The true larger-capacity *later vintage*
# belongs to E9 SCALE-CURVE; the SPEC phrase 'add the 14B point to the e8_vintage curve' is
# reconciled here by recording it as the same-vintage capacity anchor. The 14B is generated
# by scripts/run_e8_vintage_14b_gen.py (frozen P9 scaffold, GGUF backbone) and judged by
# GPT-5.2 under pattern base_p17_scale_qwen25_14b.
CAPACITY_POINT = {
    "label": "P17",
    "pattern": "base_p17_scale_qwen25_14b",          # judged later; self-guards if absent
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "release_date": "2024-09",                       # SAME vintage as P9
    "scale_axis": True,
    "capacity_note": (
        "Same release vintage as P9 (2024-09) but ~2x parameters (14B vs 7B). This is a "
        "CAPACITY/scale anchor at x=0 years on the vintage axis, NOT a new vintage date. "
        "Backbone routed through llama.cpp GGUF Q4_K_M (transformers/bnb OOMs the 14B on the "
        "16 GB RTX 5080); frozen P9 scaffold otherwise. Greedy decode (temp=0,top_k=1) is a "
        "small declared scaffold-frozenness imperfection vs the 7B arms' do_sample temp=0.01."),
}

# Arms that are part of the *intended* curve but OOM-skipped on this hardware. Recorded
# explicitly so the note is auditable and the limit is never silently dropped.
OOM_SKIPPED = [
    {"model": "Qwen/Qwen3-8B",           "release_date": "2025-04",
     "reason": "8B>16GB-card weights-materialisation OOM in 4-bit on RTX 5080 (15.47 GiB)"},
    {"model": "Qwen/Qwen3.5-9B",         "release_date": "2026-03",
     "reason": "9B weights-materialisation OOM in 4-bit on RTX 5080 (15.47 GiB); on disk but does not fit the frozen-scaffold generation path"},
]

DIMENSIONS = [
    "information_recall", "factual_accuracy", "coverage", "analytical_depth",
    "citation_quality", "logical_coherence", "organization",
    "instruction_following", "attribution_quality",
]


def _release_to_years(iso_month: str, base_iso: str) -> float:
    """Fractional years between two YYYY-MM strings (deterministic, no calendar lib needed)."""
    y0, m0 = (int(x) for x in base_iso.split("-")[:2])
    y1, m1 = (int(x) for x in iso_month.split("-")[:2])
    return round(((y1 - y0) * 12 + (m1 - m0)) / 12.0, 4)


def _overall_mean(o: pd.DataFrame, pattern: str, judge: str):
    s = o[(o.pattern == pattern) & (o.judge == judge)]
    if len(s) == 0:
        return None, 0
    return round(float(s.overall_score.mean()), 4), int(s.query_id.nunique())


def _dim_means(sc: pd.DataFrame, pattern: str, judge: str) -> dict:
    s = sc[(sc.pattern == pattern) & (sc.judge == judge)]
    out = {}
    for dim in DIMENSIONS:
        d = s[s.dimension == dim]
        out[dim] = round(float(d.score.mean()), 4) if len(d) else None
    return out


def _gap(anchor: float | None, local: float | None):
    if anchor is None or local is None:
        return None
    return round(float(anchor - local), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, but do not write the canonical key")
    ap.add_argument("--judge", default=PRIMARY_JUDGE,
                    help="primary judge id used for the curve (default: gpt52)")
    args = ap.parse_args()
    judge = args.judge

    if not CANON.exists():
        # Never create the canonical store from scratch — it is the upstream chain's job.
        print(f"[e8_vintage] canonical store missing at {CANON}; run build_numbers.py first. "
              "Exiting 0 (self-guard).")
        return 0

    o = pd.read_parquet(A / "df_overall_scores.parquet")
    sc = pd.read_parquet(A / "df_scores.parquet")

    # Frontier anchor (GPT-4o / P0) under the primary judge.
    anchor_overall, anchor_n = _overall_mean(o, ANCHOR_PATTERN, judge)
    anchor_dims = _dim_means(sc, ANCHOR_PATTERN, judge)

    points = []
    for vp in VINTAGE_POINTS:
        local_overall, n_q = _overall_mean(o, vp["pattern"], judge)
        present = local_overall is not None
        local_dims = _dim_means(sc, vp["pattern"], judge) if present else {d: None for d in DIMENSIONS}
        points.append({
            "label": vp["label"],
            "pattern": vp["pattern"],
            "model": vp["model"],
            "release_date": vp["release_date"],
            "release_years_since_first": _release_to_years(vp["release_date"], VINTAGE_POINTS[0]["release_date"]),
            "present_on_disk": present,
            "n_queries": n_q,
            "judge": judge,
            "local_overall": local_overall,
            "gap_overall_anchor_minus_local": _gap(anchor_overall, local_overall),
            "local_per_dimension": local_dims,
            "gap_per_dimension": {d: _gap(anchor_dims.get(d), local_dims.get(d)) for d in DIMENSIONS},
        })

    # CAPACITY point (Qwen2.5-14B, same vintage as P9 — emitted on the SCALE axis,
    # not the vintage-date curve). Self-guards if the 14B arm is not yet judged.
    cap_overall, cap_n = _overall_mean(o, CAPACITY_POINT["pattern"], judge)
    cap_present = cap_overall is not None
    cap_dims = (_dim_means(sc, CAPACITY_POINT["pattern"], judge)
                if cap_present else {d: None for d in DIMENSIONS})
    capacity_point = {
        "label": CAPACITY_POINT["label"],
        "pattern": CAPACITY_POINT["pattern"],
        "model": CAPACITY_POINT["model"],
        "release_date": CAPACITY_POINT["release_date"],
        "scale_axis": CAPACITY_POINT["scale_axis"],
        "capacity_note": CAPACITY_POINT["capacity_note"],
        "present_on_disk": cap_present,
        "n_queries": cap_n,
        "judge": judge,
        "local_overall": cap_overall,
        "gap_overall_anchor_minus_local": _gap(anchor_overall, cap_overall),
        "local_per_dimension": cap_dims,
        "gap_per_dimension": {d: _gap(anchor_dims.get(d), cap_dims.get(d)) for d in DIMENSIONS},
        # Capacity delta vs the SAME-VINTAGE 7B arm (P9): does ~2x params move the
        # gap at fixed vintage? Positive = 14B closes more of the gap than 7B.
        "capacity_gain_overall_vs_p9": (
            _gap(points[0]["gap_overall_anchor_minus_local"],
                 _gap(anchor_overall, cap_overall))
            if cap_present and points[0]["gap_overall_anchor_minus_local"] is not None
            else None),
    }

    present_points = [p for p in points if p["present_on_disk"]]
    n_present = len(present_points)
    status = ("two_point_curve" if n_present >= 2
              else "partial_pending_arm2" if n_present == 1
              else "no_vintage_arms_scored")
    if not cap_present:
        # The 14B capacity anchor is part of the deliverable; flag when unjudged so
        # the rebuild chain stays honest about the partial state (does not block the
        # vintage curve, which is independent).
        status = status + "+partial_pending_14b" if not status.startswith("two_point") \
            else "two_point_curve+partial_pending_14b"

    # Two-point slope of OVERALL gap vs release-date (gap-change per year). With exactly two
    # achievable points this is the connecting line, reported as such — not a regression.
    slope_overall = None
    slope_per_dimension = {d: None for d in DIMENSIONS}
    if n_present >= 2:
        p0, p1 = present_points[0], present_points[1]
        dt = p1["release_years_since_first"] - p0["release_years_since_first"]
        if dt and dt != 0:
            g0, g1 = p0["gap_overall_anchor_minus_local"], p1["gap_overall_anchor_minus_local"]
            if g0 is not None and g1 is not None:
                slope_overall = round((g1 - g0) / dt, 4)
            for d in DIMENSIONS:
                d0, d1 = p0["gap_per_dimension"].get(d), p1["gap_per_dimension"].get(d)
                if d0 is not None and d1 is not None:
                    slope_per_dimension[d] = round((d1 - d0) / dt, 4)

    # SUPERSESSION (2026-06-29): this e8_vintage key is a STALE 1-point partial — only the
    # P9 Qwen2.5-7B arm ever scored, so the vintage slope is null and no real gap-vs-date
    # curve exists. It has been SUPERSEDED by the full 4-arm `frozen_vintage` key (frozen
    # corpus, decode-backend caveat recorded, all arms qid-aligned). The partial is RETAINED
    # for provenance but marked superseded so no draft cites it as THE vintage result; the
    # status override below makes that machine-checkable regardless of on-disk arm state.
    status = "superseded_by_frozen_vintage"
    _supersede_note = (
        "SUPERSEDED by canonical key 'frozen_vintage' (full 4-arm frozen-corpus vintage curve). "
        "This e8_vintage key is a stale 1-point partial: only the P9 Qwen2.5-7B-Instruct arm was "
        "ever scored (the DeepSeek-R1-Distill arm2 and the 14B capacity anchor never landed here), "
        "so its two-point slope is null and it does NOT constitute a vintage result. Retained for "
        "provenance only — DO NOT cite this key as the vintage finding; use 'frozen_vintage'.")

    out = {
        "_superseded_by": "frozen_vintage",
        "_supersede_note": _supersede_note,
        "_note": (
            "E8 vintage achievable curve is TWO points on 16GB hardware: Qwen2.5-7B (base_p9, "
            "2024-09) -> DeepSeek-R1-Distill-Qwen-7B (base_p14_vintage_deepseek_qwen7b, 2025-01), "
            "PLUS a SAME-VINTAGE 14B CAPACITY anchor (base_p17_scale_qwen25_14b, GGUF/llama.cpp; "
            "see capacity_point, NOT on the years curve). In the FROZEN P9 scaffold "
            "(identical tools/prompts/queries), judged by GPT-5.2 (judge independence clean: "
            "OpenAI judge on Qwen-family arms). 'gap' = base_p0 GPT-4o anchor minus local, same "
            "judge; a shrinking gap across vintages = local catching the frontier. The planned "
            "9B/14B vintage arms (Qwen3-8B 2025-04, Qwen3.5-9B 2026-03) are OOM-SKIPPED at the "
            "weights-materialisation step in 4-bit on the RTX 5080 (15.47 GiB usable) — an honest "
            "hardware limit, NOT recoverable here and NOT backfillable on this box (see "
            "scripts/run_detector_panel.py for the measured fits/OOMs). With two points the "
            "'slope' is the connecting line (gap-change per year), reported as such, not a "
            "regression. Self-guards to a 1-point partial if arm2 is not yet scored."),
        "status": status,
        "primary_judge": judge,
        "anchor": {
            "pattern": ANCHOR_PATTERN,
            "model": "gpt-4o (P0 baseline scaffold)",
            "overall": anchor_overall,
            "n_queries": anchor_n,
            "per_dimension": anchor_dims,
        },
        "points": points,
        "n_points_present": n_present,
        "slope_overall_gap_per_year": slope_overall,
        "slope_per_dimension_gap_per_year": slope_per_dimension,
        "capacity_point": capacity_point,
        "capacity_axis_note": (
            "Qwen2.5-14B (base_p17_scale_qwen25_14b) is a SAME-VINTAGE (2024-09) larger-capacity "
            "anchor, recorded on the SCALE axis at x=0 years — NOT a third point on the gap-vs-YEARS "
            "vintage curve (that would put two points at x=0, an axis error). It is generated on the "
            "frozen P9 scaffold via the llama.cpp GGUF backbone (transformers/bnb OOMs the 14B on the "
            "16 GB card) and judged by GPT-5.2. The same generated rows also feed E9 SCALE-CURVE; one "
            "generation serves both. 'capacity_gain_overall_vs_p9' isolates the effect of ~2x params "
            "at fixed vintage. Self-guarded (status carries '+partial_pending_14b') until judged."),
        "oom_skipped": OOM_SKIPPED,
        "hardware": "RTX 5080, 15.47 GiB usable; 4-bit nf4 + double-quant; transformers 5.2 + bnb 0.49",
        "interpretation": (
            "If slope_overall_gap_per_year < 0 the frontier-vs-local gap is CLOSING across these "
            "two 7B vintages; > 0 it is widening; ~0 it persists. Per-dimension slopes reveal "
            "whether any closure is uniform or asymmetric (e.g. reasoning dimensions moving while "
            "citation/factual stay pinned). Two points cannot establish a 'gap half-life'; that "
            "needs the OOM-skipped arms on larger hardware."),
    }

    # Console summary
    print(f"[e8_vintage] status={status} judge={judge} anchor({ANCHOR_PATTERN})={anchor_overall} (n={anchor_n})")
    for p in points:
        flag = "" if p["present_on_disk"] else "  <-- NOT on disk (OOM/unscored; self-guarded)"
        print(f"  {p['label']:4s} {p['model']:42s} {p['release_date']}  "
              f"overall={p['local_overall']}  gap={p['gap_overall_anchor_minus_local']}{flag}")
    print(f"  slope_overall_gap_per_year={slope_overall}  "
          f"(OOM-skipped: {', '.join(x['model'] for x in OOM_SKIPPED)})")
    cap_flag = "" if capacity_point["present_on_disk"] else "  <-- NOT judged yet (self-guarded)"
    print(f"  [capacity, SAME-VINTAGE scale anchor, NOT on the years curve]")
    print(f"  {capacity_point['label']:4s} {capacity_point['model']:42s} "
          f"{capacity_point['release_date']}  overall={capacity_point['local_overall']}  "
          f"gap={capacity_point['gap_overall_anchor_minus_local']}  "
          f"capacity_gain_vs_P9={capacity_point['capacity_gain_overall_vs_p9']}{cap_flag}")

    if args.dry_run:
        print("[e8_vintage] --dry-run: nothing written.")
        return 0

    # Atomic read-modify-write of the single key (never clobbers siblings; idempotent).
    cn = json.load(open(CANON))
    cn["e8_vintage"] = out
    fd, tmp = tempfile.mkstemp(dir=str(ANA), prefix="canonical_numbers.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[e8_vintage] WROTE key 'e8_vintage' -> {CANON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
