#!/usr/bin/env python3
"""B3_e13_drbrace: project the DRB-RACE judge<->human correlation into
``judge_vs_human.by_family[fam].drb_race`` for the panel families
(gpt52, claude_sonnet, local), reusing the verified kernels in
``build_judge_vs_human.py``.

WHY THIS EXISTS
---------------
``build_judge_vs_human.build()`` populates ``by_family[fam][drb_race]`` via
``load_family_verdicts()``, which only ever consults an item-id-keyed
``results/<dir>/_human_drb_race/*.json`` store that the DRB-RACE pipeline never
writes — so those cells are permanently ``awaiting_panel``. The REAL DRB-RACE
verdicts live in ``results/drbrace/<judge>/drbrace_<system>/drb1_<task>.json``
and are consumed by ``run_drb_race()`` into ``drb_race_correlation[judge]``.

This adapter closes that gap WITHOUT touching the shared build: for each family
that has a DRB verdict store on disk it calls the existing (read-only, no-API)
``run_drb_race()`` and writes a compact agreement-style cell into
``by_family[fam].drb_race``, and (re)writes the full result into
``drb_race_correlation[<judge_label>]``. Families with no store yet (sonnet
before its 76 joinable responses finish; local before the Track-C GPU run) are
left as ``{"status": "awaiting_panel"}`` so the chain stays green and slots in
unchanged once verdicts land.

NO API. NO Opus (Opus store intentionally absent per the no-Opus rule and is
NOT consulted here). Reads verdicts + gold read-only; writes ONLY
``canonical_numbers.json['judge_vs_human']`` under the LIVE analysis dir.

Usage:
    # dry run — print every family cell, write nothing
    python paper_rebuild/paper_a_bounded_returns/analysis/build_drb_race_by_family.py

    # land the cells into canonical_numbers.json (atomic; merge-only)
    python paper_rebuild/paper_a_bounded_returns/analysis/build_drb_race_by_family.py --write
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_judge_vs_human as bjh  # noqa: E402  (verified read-only kernels)

# Live canonical store == this dir. (Independent of bjh.ANA so we are immune to
# the ANA-path bug even if fix_canonical_path_ana.py has not been applied yet.)
CN_PATH = HERE / "canonical_numbers.json"

# Panel family -> (judge label used as the drb_race_correlation key,
#                  ordered candidate DRB verdict stores). NO Opus by rule.
FAMILY_DRB_STORES: dict[str, tuple[str, list[str]]] = {
    "gpt52": ("gpt52", ["results/drbrace/judge_gpt52"]),
    "claude_sonnet": ("claude_sonnet",
                      ["results/drbrace/judge_claude_sonnet48",
                       "results/drbrace/judge_claude_sonnet"]),
    "local": ("local",
              ["results/drbrace/judge_local",
               "results/drbrace/judge_local_7b"]),
}

REPO_ROOT = bjh._REPO_ROOT  # noqa: SLF001  (module constant)


def _first_existing_store(candidates: list[str]) -> str | None:
    """Return the first candidate dir that exists AND holds >=1 verdict file."""
    for d in candidates:
        p = REPO_ROOT / d
        if p.is_dir() and any(p.glob("drbrace_*/drb1_*.json")):
            return d
    return None


def _cell_from_result(res: dict) -> dict:
    """Compact ``by_family`` cell derived from a run_drb_race() result."""
    ov = res.get("overall", {})
    return {
        "status": "scored",
        "source": "drb_race",
        "verdicts_dir": res.get("verdicts_dir"),
        "n_joined_system_task_cells": res.get("n_joined_system_task_cells"),
        "systems_judged": res.get("systems_judged"),
        "systems_pending": res.get("systems_pending"),
        "overall": {
            "n": ov.get("n"),
            "n_cells": ov.get("n_cells"),
            "n_tasks": ov.get("n_tasks"),
            "spearman": ov.get("spearman"),
            "pearson": ov.get("pearson"),
        },
        "per_dimension": {
            d: {"n": v.get("n"), "spearman": v.get("spearman")}
            for d, v in (res.get("per_dimension") or {}).items()
        },
        "note": "projected from drb_race_correlation (run_drb_race); see that "
                "block for the full per-dim CIs.",
    }


def build(write: bool = False) -> dict:
    summary: dict[str, dict] = {}
    corr_updates: dict[str, dict] = {}
    by_family_cells: dict[str, dict] = {}

    for fam, (judge_label, candidates) in FAMILY_DRB_STORES.items():
        store = _first_existing_store(candidates)
        if store is None:
            by_family_cells[fam] = {"status": "awaiting_panel"}
            summary[fam] = {"status": "awaiting_panel",
                            "reason": "no DRB verdict store on disk yet",
                            "candidates": candidates}
            continue
        # read-only, no-API correlation over real verdicts
        res = bjh.run_drb_race(store, judge=judge_label, write=False)
        corr_updates[judge_label] = res
        by_family_cells[fam] = _cell_from_result(res)
        ov = res.get("overall", {})
        summary[fam] = {
            "status": "scored",
            "store": store,
            "judge_label": judge_label,
            "n_joined": res.get("n_joined_system_task_cells"),
            "systems_judged": res.get("systems_judged"),
            "overall_spearman": (ov.get("spearman") or {}).get("estimate")
            if isinstance(ov.get("spearman"), dict) else ov.get("spearman"),
        }

    print("=" * 78)
    print("DRB-RACE by_family projection (no API; no Opus)")
    print("=" * 78)
    for fam, s in summary.items():
        print(f"  {fam:14s} {s.get('status'):14s} "
              f"n_joined={s.get('n_joined')!s:>5} "
              f"systems={s.get('systems_judged')}")
    print(f"  canonical: {CN_PATH}")
    print(f"  write:     {write}")

    if write:
        _merge_write(by_family_cells, corr_updates)
        print("WROTE judge_vs_human.by_family[*].drb_race + "
              "drb_race_correlation[gpt52|claude_sonnet|local present-only]")
    return {"by_family": by_family_cells, "drb_race_correlation": corr_updates,
            "summary": summary}


def _merge_write(by_family_cells: dict, corr_updates: dict) -> None:
    """Atomic, MERGE-ONLY write into canonical_numbers.json['judge_vs_human'].

    Touches exactly two leaves:
      * judge_vs_human.by_family[fam]['drb_race']     (per family above)
      * judge_vs_human.drb_race_correlation[judge]    (per scored family)
    Everything else in the canonical file is preserved byte-for-byte structurally
    (re-serialised with the same indent the writers in build_judge_vs_human use).
    NEVER clobbers an existing scored cell with an awaiting_panel one.
    """
    cn = json.load(open(CN_PATH))
    jvh = cn.get("judge_vs_human")
    if not isinstance(jvh, dict):
        jvh = {}
    bf = jvh.setdefault("by_family", {})
    for fam, cell in by_family_cells.items():
        existing = bf.get(fam, {}).get("drb_race")
        # resume-guard: do not downgrade a previously scored cell to awaiting_panel
        if (isinstance(existing, dict)
                and existing.get("status") == "scored"
                and cell.get("status") != "scored"):
            continue
        bf.setdefault(fam, {})["drb_race"] = cell
    drc = jvh.setdefault("drb_race_correlation", {})
    for judge_label, res in corr_updates.items():
        drc[judge_label] = res
    jvh["by_family"] = bf
    cn["judge_vs_human"] = jvh

    tmp = str(CN_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cn, f, indent=1)
    os.replace(tmp, CN_PATH)


if __name__ == "__main__":
    raise SystemExit(0 if build(write=("--write" in sys.argv)) is not None else 1)
