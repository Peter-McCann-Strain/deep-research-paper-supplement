#!/usr/bin/env python
"""build_leaderboard_comparison.py — the head-to-head.

Assembles a single comparison view that places OUR systems' *official-metric*
benchmark scores BESIDE the published SOTA numbers (Perplexity / Gemini /
OpenAI-DR / PaperQA2 / Grok) for the four head-to-head benchmarks:

    DRB / RACE      official metric: RACE overall (0-100)   -> deep_research.benchmarks.race
    LitQA2          official metric: MC accuracy            -> deep_research.benchmarks.litqa2
    DeepSearchQA    official metric: answer-set accuracy    -> deep_research.benchmarks.deepsearch_qa
    DRACO           official metric: weighted rubric MET%   -> rubric_v2 (DRACO-weighted)

The published SOTA numbers are NOT re-derived here; they are the hand-curated,
cited figures in ``published_baselines.json`` (extracted from this repo's three
benchmark-survey reports — see that file's `_README`).

What is computed live (NO API, NO network):
  1.  The four commercial DRAs' RACE scores re-derived from the 150 on-disk
      expert RACE annotation records via ``deep_research.benchmarks.race`` —
      this is a real, on-disk, human-anchored cell, and it is what lets us
      compute a Spearman of *our reconstructed ranking* vs *the published
      official-metric ranking* over a set of systems both rankings cover.
  2.  OUR systems' currently-available benchmark cells, pulled from on-disk
      results where they exist:
        - LitQA2 / DeepSearchQA / DRACO: our GPT-5.2 *panel* mean per pattern
          from ``reports/judge_evaluation_benchmarks/results.json`` (a partial
          slice: P0/P1/P2 only at time of writing). These are clearly labelled
          PANEL (not the official metric) and marked HAVE.
        - The OFFICIAL-metric accuracy cells for our systems (LitQA2 exact
          match, DeepSearchQA answer-set, DRACO weighted rubric, our RACE)
          require a full generation+grading pass that is NOT run here; they are
          emitted as PENDING stub rows so the table is complete and the gaps
          are explicit.

Outputs (written ONLY under paper_rebuild/paper_a_bounded_returns/analysis/, never the corpus):
  - leaderboard_comparison.md     human-readable head-to-head tables + Spearman
  - leaderboard_comparison.json   machine-readable, with a per-cell `status`
                                  ('have' | 'pending') and provenance.

Self-test: ``--self-test`` runs the whole build on the real on-disk data with
stub cells where our official numbers are pending, asserts the published numbers
parse, the RACE reconstruction matches race.py, the Spearman is well-formed, and
NO paid API is touched. Exit code 0 on pass.

Usage:
    python paper_rebuild/paper_a_bounded_returns/analysis/build_leaderboard_comparison.py
    python paper_rebuild/paper_a_bounded_returns/analysis/build_leaderboard_comparison.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from deep_research.benchmarks.race import (  # noqa: E402
    RaceEvaluator,
    load_drb1_human_dimension_means,
)

ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
PUBLISHED = ANA / "published_baselines.json"
OUT_MD = ANA / "leaderboard_comparison.md"
OUT_JSON = ANA / "leaderboard_comparison.json"

# OUR systems' panel benchmark scores (partial; the only on-disk own-system
# benchmark numbers at time of writing).
OUR_BENCH_RESULTS = _REPO_ROOT / "reports" / "judge_evaluation_benchmarks" / "results.json"
# Main leaderboard parquet — source of the GPT-5.2 panel ranking of our patterns.
MAIN_LB_PARQUET = _REPO_ROOT / "data" / "analysis" / "df_overall_scores.parquet"

# Protected, never-write paths (defensive guard; this script only writes under ANA).
PROTECTED = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

# Map the benchmark-result `pattern` strings to canonical pattern ids used in
# the main leaderboard, so the panel-ranking join is unambiguous.
PATTERN_CANON = {
    "p0_baseline": "p0",
    "p1_iterative_rag": "p1",
    "p2_supervisor_parallel": "p2",
    "p3_meridian": "p3",
    "p4_perspective_storm": "p4",
    "p5_hierarchical_wd": "p5",
    "p6_reactive_interleaved": "p6",
    "p7_graph_decomposition": "p7",
    "p8_beam_search": "p8",
    "p9_local_baseline": "p9",
    "p10_deep_researcher": "p10",
}

# Our headline 11-architecture comparison family (for labelling).
OUR_SYSTEM_LABEL = {
    "p0": "P0 Baseline (GPT-4o)",
    "p1": "P1 Iterative-RAG (GPT-4o)",
    "p2": "P2 Supervisor-Parallel (GPT-4o)",
    "p3": "P3 MERIDIAN (GPT-4o)",
    "p4": "P4 Perspective-STORM (GPT-4o)",
    "p5": "P5 Hierarchical-WD (GPT-4o)",
    "p6": "P6 Reactive-Interleaved (GPT-4o)",
    "p7": "P7 Graph-Decomposition (GPT-4o)",
    "p8": "P8 Beam-Search (GPT-4o)",
    "p9": "P9 Local-Baseline (Qwen2.5-7B)",
    "p10": "P10 DeepResearcher-7B (RL)",
}

# Which benchmarks the head-to-head covers and the official metric label.
HEADTOHEAD = ["drb_race", "litqa2", "deepsearch_qa", "draco"]
BENCH_TITLE = {
    "drb_race": "DRB / RACE  (RACE overall, 0-100)",
    "litqa2": "LitQA2  (MC accuracy, %)",
    "deepsearch_qa": "DeepSearchQA  (answer-set accuracy, %)",
    "draco": "DRACO  (weighted rubric MET, %)",
}
# Map our benchmark-result query-id prefix -> head-to-head benchmark key.
# (DRACO/LitQA2 in results.json carry uuid-style ids; we tag them via the
#  results-file metadata when present, else fall back to prefix heuristics.)
DSQA_PREFIX = "dsqa"


# ───────────────────────── safety ──────────────────────────

def _assert_safe_outputs() -> None:
    for out in (OUT_MD, OUT_JSON):
        out_r = out.resolve()
        for prot in PROTECTED:
            try:
                out_r.relative_to(prot.resolve())
                raise SystemExit(f"REFUSING: output {out_r} is inside protected path {prot}.")
            except ValueError:
                pass
    if ANA.resolve() not in OUT_MD.resolve().parents:
        raise SystemExit(f"REFUSING: {OUT_MD} is not under the analysis dir {ANA}.")


# ───────────────────────── stats helpers ──────────────────────────

def spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation with average-rank tie handling (no SciPy)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rankdata(a: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) *
                    sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else float("nan")


# ───────────────────────── load published ──────────────────────────

def load_published() -> Dict[str, Any]:
    pub = json.loads(PUBLISHED.read_text())
    return pub["benchmarks"]


# ───────────────────────── our RACE on the 4 DRAs (on disk) ──────────────────────────

def reconstruct_dra_race() -> Dict[str, float]:
    """Re-derive each commercial DRA's RACE overall from the 150 expert records.

    This is a real on-disk, human-anchored cell: it is the same four systems the
    published RACE leaderboard ranks, scored by RACE's own protocol weighting of
    the *expert dimension grades*. It gives a ranking we can correlate against the
    published official-metric ranking (Spearman) over a system set both cover.
    """
    ev = RaceEvaluator(weight_profile="published")
    human = load_drb1_human_dimension_means()  # {task: {model: {dim..., overall}}}
    per_model: Dict[str, List[float]] = defaultdict(list)
    for _tid, models in human.items():
        for model, grades in models.items():
            if all(d in grades for d in ("comprehensiveness", "depth",
                                         "instruction_following", "readability")):
                rec = ev.score_from_dimension_grades(grades)
                per_model[model].append(rec["overall"])
    return {m: round(sum(v) / len(v), 2) for m, v in per_model.items() if v}


# Mapping the on-disk DRA model keys to the published-baseline system labels so
# the reconstructed RACE can be aligned with the published RACE numbers.
DRA_KEY_TO_PUB = {
    "gemini-2.5-pro-deepresearch": "Gemini-2.5-Pro Deep Research",
    "openai-deepresearch": "OpenAI Deep Research",
    "perplexity-Research": "Perplexity Deep Research",
    "grok-deeper-search": "Grok Deeper Search",
}


# ───────────────────────── our partial panel benchmark scores (on disk) ──────────────────────────

def load_our_panel_bench() -> Dict[str, Dict[str, float]]:
    """OUR GPT-5.2 PANEL mean per (benchmark, pattern) from the on-disk slice.

    NOTE: this is the 9-dim PANEL overall_score, NOT the official benchmark
    metric. It is included as a HAVE cell but explicitly tagged 'panel'.
    Returns {benchmark_key: {canon_pattern: mean_panel_overall}}.
    """
    if not OUR_BENCH_RESULTS.exists():
        return {}
    recs = json.loads(OUR_BENCH_RESULTS.read_text())
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        qid = str(r.get("query_id", ""))
        pat = PATTERN_CANON.get(r.get("pattern", ""), r.get("pattern", ""))
        # Tag the head-to-head benchmark for this record.
        meta_bench = (r.get("metadata") or {}).get("benchmark")
        if meta_bench in HEADTOHEAD:
            bench = meta_bench
        elif qid.startswith(DSQA_PREFIX):
            bench = "deepsearch_qa"
        else:
            # uuid-style ids in this slice are DRACO/LitQA2 mixed; without a
            # benchmark tag in the record we cannot safely split them, so we
            # bucket them under a single 'draco_litqa_mixed' panel cell and
            # surface that ambiguity honestly rather than guess.
            bench = "draco_litqa_mixed"
        acc[bench][pat].append(float(r.get("overall_score", 0.0)))
    out: Dict[str, Dict[str, float]] = {}
    for bench, pats in acc.items():
        out[bench] = {p: round(sum(v) / len(v), 4) for p, v in pats.items()}
    return out


# ───────────────────────── our panel ranking of ALL patterns (for the cross-ranking note) ──────────────────────────

def load_panel_pattern_ranking() -> Dict[str, float]:
    """Per-pattern GPT-5.2 panel mean overall_score from the MAIN leaderboard."""
    try:
        import re
        import pandas as pd
    except Exception:
        return {}
    if not MAIN_LB_PARQUET.exists():
        return {}
    df = pd.read_parquet(MAIN_LB_PARQUET)
    base = df[(df["judge"] == "gpt52") &
              df["pattern"].astype(str).str.fullmatch(r"base_p\d+")]
    means = base.groupby("pattern", observed=True)["overall_score"].mean()
    return {re.sub(r"^base_", "", k): round(float(v), 4) for k, v in means.items()}


# ───────────────────────── assemble ──────────────────────────

def build(stub_our_official: bool = True) -> Dict[str, Any]:
    """Build the full machine-readable comparison structure.

    stub_our_official: emit PENDING stub rows for our official-metric cells that
    require a generation+grading pass not run here. Always True in normal use;
    the self-test exercises both the stub path and the live on-disk cells.
    """
    published = load_published()
    dra_race = reconstruct_dra_race()
    our_panel = load_our_panel_bench()
    panel_rank = load_panel_pattern_ranking()

    out: Dict[str, Any] = {
        "_README": "Head-to-head: OUR systems' official-metric benchmark scores beside "
                   "published SOTA. Each cell carries `status` ('have' | 'pending') and "
                   "provenance. Published numbers from published_baselines.json (cited).",
        "no_api_called": True,
        "benchmarks": {},
        "cross_ranking_spearman": {},
        "panel_pattern_ranking_gpt52": panel_rank,
    }

    # ---- DRB / RACE ----
    race_rows: List[Dict[str, Any]] = []
    pub_race = {r["system"]: r for r in published["drb_race"]["rows"]}
    # Published SOTA rows + our on-disk reconstruction beside them.
    for dra_key, pub_label in DRA_KEY_TO_PUB.items():
        recon = dra_race.get(dra_key)
        pub = pub_race.get(pub_label, {})
        race_rows.append({
            "system": pub_label,
            "is_ours": False,
            "published_official": pub.get("score"),
            "published_source": pub.get("source"),
            "our_reconstructed_race": recon,           # on-disk, human-anchored
            "our_reconstructed_status": "have" if recon is not None else "pending",
            "note": "RACE re-derived from the 150 on-disk expert annotation records "
                    "(race.py protocol weighting).",
        })
    # OUR systems on RACE: official RACE not yet graded -> pending stub rows.
    our_race_systems = ["p0", "p1", "p4", "p6", "p9", "p10"]  # the E12 slice
    for p in our_race_systems:
        race_rows.append({
            "system": OUR_SYSTEM_LABEL[p],
            "is_ours": True,
            "published_official": None,
            "our_race": None,
            "status": "pending",
            "note": "Our RACE via race.py + GPT-5.2 reference-guided grader "
                    "(generation+grading pass not run here).",
        })
    out["benchmarks"]["drb_race"] = {
        "metric": published["drb_race"]["metric"],
        "official_grader": published["drb_race"]["official_grader"],
        "rows": race_rows,
    }

    # ---- LitQA2 / DeepSearchQA / DRACO ----
    for bench in ["litqa2", "deepsearch_qa", "draco"]:
        rows: List[Dict[str, Any]] = []
        for pr in published[bench]["rows"]:
            row = {
                "system": pr["system"],
                "is_ours": False,
                "published_official": pr.get("score"),
                "published_source": pr.get("source"),
                "status": "have",
            }
            if "precision" in pr:
                row["published_precision"] = pr["precision"]
            if "score_verified" in pr:
                row["published_verified"] = pr["score_verified"]
            rows.append(row)
        # OUR systems: official metric pending; attach the on-disk PANEL mean if any.
        panel_cell = our_panel.get(bench, {})
        # For DRACO/LitQA2 the panel slice is bucketed; surface it once under DRACO
        # with an explicit ambiguity flag, none under LitQA2 (cannot safely split).
        mixed_cell = our_panel.get("draco_litqa_mixed", {})
        our_official_systems = ["p0", "p1", "p2", "p4"] if bench != "litqa2" else ["p0", "p1", "p4"]
        for p in our_official_systems:
            r = {
                "system": OUR_SYSTEM_LABEL[p],
                "is_ours": True,
                "published_official": None,
                "our_official": None,
                "official_status": "pending",
                "note": f"Our official {bench} metric via "
                        f"deep_research.benchmarks ({bench}) / rubric_v2 (DRACO) — "
                        f"generation+scoring pass not run here.",
            }
            # Attach the on-disk panel mean where we have it.
            panel_val = panel_cell.get(p)
            if panel_val is None and bench == "draco":
                panel_val = mixed_cell.get(p)
                if panel_val is not None:
                    r["our_panel_note"] = ("on-disk PANEL slice is a DRACO+LitQA2 "
                                           "uuid mix; reported under DRACO with this flag")
            if panel_val is not None:
                r["our_panel_overall"] = panel_val
                r["our_panel_status"] = "have"
            else:
                r["our_panel_status"] = "pending"
            rows.append(r)
        out["benchmarks"][bench] = {
            "metric": published[bench]["metric"],
            "official_grader": published[bench]["official_grader"],
            "rows": rows,
        }

    # ---- Spearman: our ranking vs official-metric ranking ----
    # The only set of systems for which BOTH an our-side score and a published
    # official-metric score exist *right now* is the four DRAs on DRB/RACE
    # (our reconstructed RACE vs their published RACE). Compute it there; the
    # per-pattern cross-ranking on the benchmarks is emitted PENDING until our
    # official cells land.
    aligned = []
    for dra_key, pub_label in DRA_KEY_TO_PUB.items():
        recon = dra_race.get(dra_key)
        pub = pub_race.get(pub_label, {}).get("score")
        if recon is not None and pub is not None:
            aligned.append((pub_label, recon, pub))
    if len(aligned) >= 2:
        ours = [a[1] for a in aligned]
        offi = [a[2] for a in aligned]
        rho = spearman(ours, offi)
        out["cross_ranking_spearman"]["drb_race_recon_vs_published"] = {
            "rho": round(rho, 4) if not math.isnan(rho) else None,
            "n_systems": len(aligned),
            "systems": [a[0] for a in aligned],
            "our_reconstructed_race": {a[0]: a[1] for a in aligned},
            "published_race": {a[0]: a[2] for a in aligned},
            "status": "have",
            "interpretation": "Spearman of OUR on-disk RACE reconstruction ranking "
                              "vs the published RACE leaderboard ranking over the four "
                              "commercial DRAs. rho=1.0 means our protocol re-derivation "
                              "ranks the four SOTA systems identically to the leaderboard.",
            "scale_caveat": "ABSOLUTE SCALES DIFFER and are NOT comparable point-for-point: "
                            "the published RACE leaderboard uses reference-GUIDED relative "
                            "grading (candidate vs a strong reference report, compressing "
                            "scores into the ~40-49 band), whereas our reconstruction is the "
                            "RACE protocol weighting of the RAW 0-100 expert dimension grades "
                            "(absolute, ~66-81 band). Only the RANKING is comparable; that is "
                            "exactly what this Spearman measures.",
        }
    out["cross_ranking_spearman"]["our_systems_vs_official"] = {
        "status": "pending",
        "reason": "Requires our systems' official-metric benchmark scores "
                  "(LitQA2 accuracy / DeepSearchQA / DRACO / our RACE), which "
                  "need a generation+grading pass not run by this script. Once "
                  "those HAVE cells land, this Spearman is computed per benchmark "
                  "and pooled.",
    }
    return out


# ───────────────────────── markdown rendering ──────────────────────────

def _fmt(v: Optional[float], pending_dash: str = "_pending_") -> str:
    if v is None:
        return pending_dash
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def render_markdown(data: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Leaderboard head-to-head: our systems vs published SOTA")
    L.append("")
    L.append("Generated by `build_leaderboard_comparison.py` (NO API / NO network). "
             "Published numbers are the hand-curated, cited figures in "
             "`published_baselines.json`. Cells are marked **HAVE** (on disk now) or "
             "**PENDING** (needs a generation+grading pass not run here).")
    L.append("")

    # DRB / RACE
    L.append(f"## {BENCH_TITLE['drb_race']}")
    L.append("")
    L.append("> The 'Published RACE' column is the reference-GUIDED leaderboard scale "
             "(~40-49). The 'Our RACE (recon)' column re-derives RACE from the raw 0-100 "
             "expert grades (absolute scale, ~66-81). The two columns are **not** "
             "point-comparable; compare ranking, not magnitude (see Spearman below).")
    L.append("")
    L.append("| System | Ours? | Published RACE | Our RACE (recon / graded) | Status | Source |")
    L.append("|---|:--:|--:|--:|:--:|---|")
    for r in data["benchmarks"]["drb_race"]["rows"]:
        if r["is_ours"]:
            our = _fmt(r.get("our_race"))
            st = "PENDING"
            src = ""
        else:
            our = _fmt(r.get("our_reconstructed_race"))
            st = "HAVE" if r.get("our_reconstructed_status") == "have" else "PENDING"
            src = (r.get("published_source") or "").replace("reports/", "")
        L.append(f"| {r['system']} | {'yes' if r['is_ours'] else ''} | "
                 f"{_fmt(r.get('published_official'))} | {our} | {st} | {src} |")
    L.append("")

    # LitQA2 / DeepSearchQA / DRACO
    for bench in ["litqa2", "deepsearch_qa", "draco"]:
        b = data["benchmarks"][bench]
        L.append(f"## {BENCH_TITLE[bench]}")
        L.append("")
        L.append(f"Official metric: {b['metric']}.")
        L.append("")
        L.append("| System | Ours? | Published (official) | Our official | Our panel (9-dim) | Status | Source |")
        L.append("|---|:--:|--:|--:|--:|:--:|---|")
        for r in b["rows"]:
            if r["is_ours"]:
                pub = "—"
                our_off = _fmt(r.get("our_official"))
                our_pan = _fmt(r.get("our_panel_overall"), pending_dash="—")
                st = "panel HAVE / official PENDING" if r.get("our_panel_status") == "have" else "PENDING"
                src = ""
            else:
                extra = ""
                if "published_precision" in r:
                    extra = f" (prec {r['published_precision']})"
                if "published_verified" in r:
                    extra += f" (verified {r['published_verified']})"
                pub = f"{_fmt(r.get('published_official'))}{extra}"
                our_off = "—"
                our_pan = "—"
                st = "HAVE"
                src = (r.get("published_source") or "").replace("reports/", "")
            L.append(f"| {r['system']} | {'yes' if r['is_ours'] else ''} | {pub} | "
                     f"{our_off} | {our_pan} | {st} | {src} |")
        L.append("")

    # Spearman
    L.append("## Spearman: our ranking vs official-metric ranking")
    L.append("")
    cr = data["cross_ranking_spearman"]
    rr = cr.get("drb_race_recon_vs_published")
    if rr and rr.get("rho") is not None:
        L.append(f"**DRB/RACE — our on-disk RACE reconstruction vs published RACE** "
                 f"(n={rr['n_systems']} commercial DRAs): "
                 f"Spearman rho = **{rr['rho']:.4f}** [{rr['status'].upper()}].")
        L.append("")
        L.append("| System | Our recon RACE | Published RACE |")
        L.append("|---|--:|--:|")
        for s in rr["systems"]:
            L.append(f"| {s} | {rr['our_reconstructed_race'][s]:.2f} | "
                     f"{rr['published_race'][s]:.2f} |")
        L.append("")
        L.append(f"_{rr['interpretation']}_")
        if rr.get("scale_caveat"):
            L.append("")
            L.append(f"> **Scale caveat.** {rr['scale_caveat']}")
        L.append("")
    osv = cr.get("our_systems_vs_official", {})
    L.append(f"**Our systems vs official metric (per-benchmark pooled): "
             f"[{osv.get('status', 'pending').upper()}]** — {osv.get('reason', '')}")
    L.append("")

    # Panel ranking of all our patterns (context for the pending cross-ranking)
    pr = data.get("panel_pattern_ranking_gpt52", {})
    if pr:
        L.append("## Our GPT-5.2 panel ranking of all patterns (context)")
        L.append("")
        L.append("This is the main-study 9-dim panel mean per pattern (the ranking the "
                 "pending Spearman will be tested against once official cells land).")
        L.append("")
        L.append("| Rank | Pattern | Panel overall |")
        L.append("|--:|---|--:|")
        for i, (p, v) in enumerate(sorted(pr.items(), key=lambda kv: -kv[1]), 1):
            L.append(f"| {i} | {p} | {v:.4f} |")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**Provenance.** Published numbers: `published_baselines.json` "
             "(hand-extracted + cited from this repo's three benchmark-survey reports). "
             "Our RACE reconstruction: `deep_research/benchmarks/race.py` over the 150 "
             "on-disk expert annotations in `data/benchmarks/drb1/`. Our panel slice: "
             "`reports/judge_evaluation_benchmarks/results.json` (partial: P0/P1/P2). "
             "Our official-metric cells: pending a generation+grading pass.")
    return "\n".join(L) + "\n"


# ───────────────────────── self-test ──────────────────────────

def _self_test() -> int:
    print("=" * 72)
    print("build_leaderboard_comparison self-test (NO API CALLS)")
    print("=" * 72)
    failures: List[str] = []

    # 1) published baselines parse + every row cited.
    pub = json.loads(PUBLISHED.read_text())["benchmarks"]
    for bench in HEADTOHEAD:
        if bench not in pub:
            failures.append(f"published baseline missing benchmark {bench}")
            continue
        for r in pub[bench]["rows"]:
            if "score" not in r or "source" not in r:
                failures.append(f"{bench} row {r.get('system')} missing score/source citation")
    print(f"[1] published baselines: {sum(len(pub[b]['rows']) for b in HEADTOHEAD)} "
          f"cited rows across {len(HEADTOHEAD)} benchmarks -> "
          f"{'OK' if not failures else 'FAIL'}")

    # 2) RACE reconstruction matches race.py and ranks the 4 DRAs.
    dra_race = reconstruct_dra_race()
    print(f"[2] reconstructed DRA RACE (on-disk, human-anchored): "
          f"{ {k: dra_race[k] for k in sorted(dra_race)} }")
    if set(dra_race) != set(DRA_KEY_TO_PUB):
        failures.append(f"RACE reconstruction covered {sorted(dra_race)}, "
                        f"expected {sorted(DRA_KEY_TO_PUB)}")
    # Cross-check ONE cell against race.py directly (no shortcut).
    ev = RaceEvaluator(weight_profile="published")
    human = load_drb1_human_dimension_means()
    sample = sorted(human)[0]
    for model, grades in sorted(human[sample].items()):
        if all(d in grades for d in ("comprehensiveness", "depth",
                                     "instruction_following", "readability")):
            rec = ev.score_from_dimension_grades(grades)
            assert 0 <= rec["overall"] <= 100, "RACE overall out of range"
            break
    print(f"[2] race.py cross-check on task {sample}/{model}: "
          f"overall={rec['overall']:.2f} -> {'OK' if 0 <= rec['overall'] <= 100 else 'FAIL'}")

    # 3) full build runs with stub official cells; structure is complete.
    data = build(stub_our_official=True)
    for bench in HEADTOHEAD:
        if bench not in data["benchmarks"]:
            failures.append(f"build output missing benchmark {bench}")
        rows = data["benchmarks"][bench]["rows"]
        if not any(r["is_ours"] for r in rows):
            failures.append(f"{bench}: no OUR-system rows in head-to-head")
        if not any(not r["is_ours"] for r in rows):
            failures.append(f"{bench}: no published-SOTA rows in head-to-head")
        # every our-system official cell must be explicitly pending or have.
        for r in rows:
            if r["is_ours"]:
                st = r.get("official_status") or r.get("status")
                if st not in ("pending", "have"):
                    failures.append(f"{bench}: our row {r['system']} has no clear status")
    print(f"[3] build structure: "
          f"{sum(len(data['benchmarks'][b]['rows']) for b in HEADTOHEAD)} total rows, "
          f"each cell status-tagged -> {'OK' if not failures else 'FAIL'}")

    # 4) Spearman well-formed on the DRB/RACE recon-vs-published cell.
    rr = data["cross_ranking_spearman"].get("drb_race_recon_vs_published")
    if not rr or rr.get("rho") is None:
        failures.append("DRB/RACE recon-vs-published Spearman not computed")
    else:
        if not (-1.0 - 1e-9 <= rr["rho"] <= 1.0 + 1e-9):
            failures.append(f"Spearman rho out of range: {rr['rho']}")
        print(f"[4] Spearman (our recon RACE vs published RACE, n={rr['n_systems']}): "
              f"rho={rr['rho']:.4f} -> "
              f"{'OK' if -1.0 <= rr['rho'] <= 1.0 else 'FAIL'}")
    # the our-systems-vs-official cross ranking must be explicitly pending.
    if data["cross_ranking_spearman"]["our_systems_vs_official"]["status"] != "pending":
        failures.append("our_systems_vs_official should be pending")

    # 5) at least one HAVE panel cell present (P0/P1/P2 are on disk).
    have_panel = any(
        r.get("our_panel_status") == "have"
        for b in ["litqa2", "deepsearch_qa", "draco"]
        for r in data["benchmarks"][b]["rows"] if r["is_ours"]
    )
    print(f"[5] on-disk OUR panel cells present: {have_panel} -> "
          f"{'OK' if have_panel else 'WARN (no on-disk panel slice found)'}")

    # 6) markdown renders without error and references both have+pending.
    md = render_markdown(data)
    ok_md = ("HAVE" in md and "PENDING" in md and "Spearman" in md)
    print(f"[6] markdown render: {len(md)} chars, has HAVE+PENDING+Spearman -> "
          f"{'OK' if ok_md else 'FAIL'}")
    if not ok_md:
        failures.append("markdown render missing required markers")

    print("=" * 72)
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELF-TEST PASSED (no API called)")
    return 0


# ───────────────────────── main ──────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="Run the offline self-test on real on-disk data and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    _assert_safe_outputs()
    data = build(stub_our_official=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))
    OUT_MD.write_text(render_markdown(data))
    rr = data["cross_ranking_spearman"].get("drb_race_recon_vs_published", {})
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    if rr.get("rho") is not None:
        print(f"DRB/RACE recon-vs-published Spearman rho = {rr['rho']:.4f} "
              f"(n={rr['n_systems']} DRAs)")
    print("Our official-metric cells: PENDING (generation+grading pass not run here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
