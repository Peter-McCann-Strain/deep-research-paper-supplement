#!/usr/bin/env python
"""E6 STC-AUDIT — STEP 1: assemble the contamination telemetry + per-snippet tables.

Part of E6 (RESEARCH_PLAN_2026H2 §E6; prereg docs/publication/prereg/prereg_E6.md; canonical key
`contamination`). Robustness appendix, NOT a headline (2606.05241 owns the framing).

What this builds (READ-ONLY on every input; writes ONLY under results/contamination_e6/)
----------------------------------------------------------------------------------------
1. ``telemetry.parquet`` / ``telemetry.csv`` — one row per (pattern, query_id) that has a
   trace.json checkpoint with a CANONICAL query_id, carrying the regressors:
     - search_count          (trace data.n_search_queries — the rate-regression regressor)
     - n_unique_urls_visited (trace data.n_unique_urls_visited)
     - n_iterations          (trace data.n_iterations)
   joined to df_queries.source (public-benchmark partition) and df_runs
   (excluded_from_analysis). CUSTOM-source queries are dropped; EXCLUSIONS.md cells are
   flagged (not silently removed — the flag column lets the downstream regression decide).

2. ``snippets.parquet`` / a row-count summary — the per-snippet table that the regex gate
   (STEP 2) and the GPT-4o classifier (STEP 3) consume. TWO bases, per the STEP-0 dual-basis
   decision (see the build_contamination.py header and prereg note for the authoritative
   choice):
     (a) basis="citation"  — df_citations cited_url/domain, ALL 12 logged patterns, the only
         uniform 11-architecture signal (but the CITED subset only — under-counts
         retrieved-but-uncited contaminated snippets).
     (b) basis="search"    — search.json data.extractions[] full snippet text
         (url/title/summary/key_findings), present on disk ONLY for P0/P1/P9/P12 — the
         higher-recall sensitivity basis.

Coverage realities recorded in the manifest (prereg/blocker honesty):
  - trace.json query_ids are only ~partially canonical UUIDs; the search_count regressor is
    available ONLY for the canonical-id subset. P11 (react) has ZERO trace.json files.
  - Checkpoint coverage is a partial replicate sample (~22-31 distinct canonical query_ids
    per pattern), NOT the full 85 public queries. The uniform 85/995-cell basis is
    df_citations; trace.json supplies search_count only where present.

No model calls. Determinism: sorted inputs, sorted outputs. Verify with --dry-run.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python scripts/build_contamination_telemetry.py --dry-run     # plan, no writes
    python scripts/build_contamination_telemetry.py               # build the tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when launched as `python scripts/build_contamination_telemetry.py`
# (sys.path[0] is scripts/ otherwise, breaking `from deep_research...`). A missing one of
# these is exactly the ModuleNotFoundError that crashed the detector panel.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"
ANA = ROOT / "data" / "analysis"
DF_CITATIONS = ANA / "df_citations.parquet"
DF_QUERIES = ANA / "df_queries.parquet"
DF_RUNS = ANA / "df_runs.parquet"

# NEW output dir — every write lands here, never in a protected path.
OUT_DIR = ROOT / "results" / "contamination_e6"

# Protected, strictly-READ-ONLY paths. We never write to or under any of these.
PROTECTED = [
    ROOT / "results" / "judge_gpt52",
    ROOT / "results" / "experiments",
    ROOT / "data" / "analysis",
    ROOT / "reports" / "eval_v2" / "verdicts",
]

# Public-benchmark partition (df_queries.source). 'custom' is excluded per the spec.
PUBLIC_SOURCES = {"draco", "deepsearch_qa", "research_qa", "litqa2"}

# Pattern dir -> canonical pattern label (matches df_citations 'pattern' column).
PATTERN_LABEL = {
    "p0_baseline": "base_p0",
    "p1_iterative_rag": "base_p1",
    "p2_supervisor_parallel": "base_p2",
    "p3_meridian": "base_p3",
    "p4_perspective_storm": "base_p4",
    "p5_hierarchical_wd": "base_p5",
    "p6_reactive_interleaved": "base_p6",
    "p7_graph_decomposition": "base_p7",
    "p8_beam_search": "base_p8",
    "p9_local_baseline": "base_p9",
    "p10_deep_researcher": "base_p10",
    "p11_react": "base_p11",
    "p12_rl_trained": "base_p12",
}


def _assert_output_safe(path: Path) -> None:
    """Refuse to write anywhere inside a protected corpus path."""
    rp = path.resolve()
    for prot in PROTECTED:
        p = prot.resolve()
        if rp == p or _is_relative_to(rp, p):
            raise SystemExit(
                f"REFUSING: output {rp} is inside protected path {p}. "
                f"E6 writes ONLY under {OUT_DIR}."
            )


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_trace_telemetry(canonical_qids: set) -> List[Dict[str, Any]]:
    """Walk checkpoints/<pattern>/<run_id>/trace.json -> per-(pattern,query_id) regressors.

    trace.json shape: {"pattern","run_id","stage","timestamp","data":{...}}, with
    data carrying: query_id, query, n_search_queries, n_unique_urls_visited, n_iterations,
    tool_calls[]. The data.query_id is OFTEN a non-canonical run id (empty, '174539...-s34',
    'q4_...'). We keep a row for every trace, tagging canonical_query_id = qid if it is one of
    the 90 manifest UUIDs else None. The rate regression (build_contamination.py) keys on
    canonical_query_id; non-canonical rows are retained for provenance/auditing only.

    When a (pattern, canonical_query_id) has multiple trace files (replicates), the row with
    the MAX search_count is kept (deterministic, conservative for an exposure regressor); the
    replicate count is recorded.
    """
    rows: Dict[tuple, Dict[str, Any]] = {}
    for patt_dir in sorted(CHECKPOINTS.glob("p*/")):
        label = PATTERN_LABEL.get(patt_dir.name)
        if label is None:
            continue  # non-pattern checkpoint dir (e.g. test_pattern) -> skip
        for trace in sorted(patt_dir.glob("*/trace.json")):
            try:
                d = json.loads(trace.read_text()).get("data", {}) or {}
            except (json.JSONDecodeError, OSError):
                continue
            qid = str(d.get("query_id") or "")
            canonical = qid if qid in canonical_qids else None
            search_count = int(d.get("n_search_queries") or 0)
            rec = {
                "pattern": label,
                "pattern_dir": patt_dir.name,
                "trace_query_id": qid,
                "canonical_query_id": canonical,
                "search_count": search_count,
                "n_unique_urls_visited": int(d.get("n_unique_urls_visited") or 0),
                "n_iterations": int(d.get("n_iterations") or 0),
                "n_tool_calls": len(d.get("tool_calls") or []),
                "run_id": trace.parent.name,
                "is_canonical": canonical is not None,
                "n_trace_replicates": 1,
            }
            # Collapse replicates per (pattern, canonical_query_id); keep max-search row.
            if canonical is not None:
                key = (label, canonical)
                prev = rows.get(key)
                if prev is None:
                    rows[key] = rec
                else:
                    prev["n_trace_replicates"] += 1
                    if search_count > prev["search_count"]:
                        # carry replicate count forward onto the kept row
                        n = prev["n_trace_replicates"]
                        rec["n_trace_replicates"] = n
                        rows[key] = rec
            else:
                # non-canonical: keep every row keyed by (pattern, run_id) to avoid clobber
                rows[(label, trace.parent.name)] = rec
    return list(rows.values())


def _load_search_snippets() -> List[Dict[str, Any]]:
    """search.json data.extractions[] -> full-snippet rows (basis='search').

    Present on disk only for P0/P1/P9/P12. Each extraction carries url/title/summary/
    key_findings — the higher-recall classifier basis. We DO NOT call any model here; we only
    assemble the text the human-launched GPT-4o classifier (STEP 3) will read.
    """
    out: List[Dict[str, Any]] = []
    for patt_dir in sorted(CHECKPOINTS.glob("p*/")):
        label = PATTERN_LABEL.get(patt_dir.name)
        if label is None:
            continue
        for sj in sorted(patt_dir.glob("*/search.json")):
            try:
                data = json.loads(sj.read_text()).get("data", {}) or {}
            except (json.JSONDecodeError, OSError):
                continue
            # query_id lives in the sibling trace.json data; read it if present.
            qid = ""
            tj = sj.parent / "trace.json"
            if tj.exists():
                try:
                    qid = str(json.loads(tj.read_text()).get("data", {}).get("query_id") or "")
                except (json.JSONDecodeError, OSError):
                    qid = ""
            for i, ext in enumerate(data.get("extractions") or []):
                kf = ext.get("key_findings")
                if isinstance(kf, list):
                    kf = " ".join(str(x) for x in kf)
                out.append({
                    "basis": "search",
                    "pattern": label,
                    "trace_query_id": qid,
                    "run_id": sj.parent.name,
                    "snippet_index": i,
                    "url": str(ext.get("url") or ""),
                    "domain": "",  # filled by the regex gate (STEP 2) via URL parse
                    "title": str(ext.get("title") or ""),
                    "summary": str(ext.get("summary") or "")[:4000],
                    "key_findings": str(kf or "")[:4000],
                })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help=f"output dir (default {OUT_DIR}); refuses any protected path")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the tables in-memory and print the plan; write NOTHING")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    _assert_output_safe(out_dir)

    import pandas as pd

    # ---- load READ-ONLY parquets ----
    queries = pd.read_parquet(DF_QUERIES)
    citations = pd.read_parquet(DF_CITATIONS)
    runs = pd.read_parquet(DF_RUNS)

    canonical_qids = set(queries["query_id"].astype(str))
    public_qids = set(
        queries.loc[queries["source"].isin(PUBLIC_SOURCES), "query_id"].astype(str)
    )
    source_map = dict(zip(queries["query_id"].astype(str), queries["source"].astype(str)))

    # ---- telemetry table from trace.json ----
    tele_rows = _load_trace_telemetry(canonical_qids)
    tele = pd.DataFrame(tele_rows)
    if not tele.empty:
        tele["source"] = tele["canonical_query_id"].map(source_map)
        tele["is_public"] = tele["canonical_query_id"].isin(public_qids)
        # exclusion flag from df_runs.excluded_from_analysis (join on pattern+query_id)
        excl = runs[["pattern", "query_id", "excluded_from_analysis"]].copy()
        excl["query_id"] = excl["query_id"].astype(str)
        tele = tele.merge(
            excl, left_on=["pattern", "canonical_query_id"],
            right_on=["pattern", "query_id"], how="left").drop(columns=["query_id"])
        tele["excluded_from_analysis"] = (
            tele["excluded_from_analysis"].astype("boolean").fillna(False).astype(bool))
        tele = tele.sort_values(
            ["pattern", "canonical_query_id", "trace_query_id"], na_position="last"
        ).reset_index(drop=True)

    # ---- per-snippet table, basis='citation' (uniform 11-arch, CITED subset) ----
    cit = citations.copy()
    cit["query_id"] = cit["query_id"].astype(str)
    cit["source"] = cit["query_id"].map(source_map)
    cit["is_public"] = cit["query_id"].isin(public_qids)
    cit_snip = pd.DataFrame({
        "basis": "citation",
        "pattern": cit["pattern"].astype(str),
        "query_id": cit["query_id"],
        "source": cit["source"],
        "is_public": cit["is_public"],
        "snippet_index": cit["citation_index"],
        "url": cit["cited_url"].astype(str),
        "domain": cit["domain"].astype(str),
        "title": cit["cited_title"].astype(str),
        "summary": cit["claim_context"].astype(str),
        "key_findings": "",
        "category": cit["category"].astype(str),
    }).sort_values(["pattern", "query_id", "snippet_index"]).reset_index(drop=True)

    # ---- per-snippet table, basis='search' (P0/P1/P9/P12 full snippets) ----
    search_rows = _load_search_snippets()
    search_snip = pd.DataFrame(search_rows)
    if not search_snip.empty:
        search_snip["query_id"] = search_snip["trace_query_id"].where(
            search_snip["trace_query_id"].isin(canonical_qids), other="")
        search_snip["source"] = search_snip["query_id"].map(source_map)
        search_snip["is_public"] = search_snip["query_id"].isin(public_qids)
        search_snip = search_snip.sort_values(
            ["pattern", "run_id", "snippet_index"]).reset_index(drop=True)

    # ---- plan / manifest summary ----
    n_canon_tele = int(tele["is_canonical"].sum()) if not tele.empty else 0
    patt_with_trace = sorted(tele["pattern"].unique()) if not tele.empty else []
    cit_public = cit_snip[cit_snip["is_public"]]
    manifest = {
        "experiment": "E6 STC-AUDIT (step 1 telemetry build)",
        "prereg": "docs/publication/prereg/prereg_E6.md",
        "public_sources": sorted(PUBLIC_SOURCES),
        "n_public_queries": len(public_qids),
        "telemetry": {
            "n_trace_rows": int(len(tele)),
            "n_canonical_id_rows": n_canon_tele,
            "patterns_with_trace": patt_with_trace,
            "p11_react_trace_files": int(
                (tele["pattern"] == "base_p11").sum()) if not tele.empty else 0,
            "note": ("search_count regressor available ONLY for canonical-id rows; "
                     "P11 has ZERO trace.json (its rows, if any, come via df_citations "
                     "with search_count imputed/omitted downstream)."),
        },
        "snippets_citation_basis": {
            "n_rows_total": int(len(cit_snip)),
            "n_rows_public": int(len(cit_public)),
            "n_patterns": int(cit_snip["pattern"].nunique()),
            "n_cells": int(cit_snip.groupby(["pattern", "query_id"]).ngroups),
            "note": ("UNIFORM 11/12-architecture rate-regression basis; CITED subset only "
                     "(under-counts retrieved-but-uncited contaminated snippets)."),
        },
        "snippets_search_basis": {
            "n_rows_total": int(len(search_snip)),
            "patterns": sorted(search_snip["pattern"].unique()) if not search_snip.empty else [],
            "note": ("Higher-recall full-snippet sensitivity basis; on disk ONLY for "
                     "P0/P1/P9/P12 (search.json extractions)."),
        },
    }

    print("=" * 70)
    print("E6 STC-AUDIT — STEP 1 telemetry build")
    print(f"  out dir              : {out_dir}")
    print(f"  public queries       : {manifest['n_public_queries']} "
          f"({', '.join(sorted(PUBLIC_SOURCES))})")
    print(f"  trace rows           : {manifest['telemetry']['n_trace_rows']} "
          f"(canonical-id: {n_canon_tele})")
    print(f"  patterns w/ trace    : {patt_with_trace}")
    print(f"  P11 trace files      : {manifest['telemetry']['p11_react_trace_files']} "
          f"(blocker: P11 has no trace.json)")
    print(f"  citation snippets    : {len(cit_snip)} total / {len(cit_public)} public "
          f"({manifest['snippets_citation_basis']['n_patterns']} patterns, "
          f"{manifest['snippets_citation_basis']['n_cells']} cells)")
    print(f"  search  snippets     : {len(search_snip)} "
          f"({manifest['snippets_search_basis']['patterns']})")
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] no files written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    tele.to_parquet(out_dir / "telemetry.parquet", index=False)
    tele.to_csv(out_dir / "telemetry.csv", index=False)
    cit_snip.to_parquet(out_dir / "snippets_citation.parquet", index=False)
    if not search_snip.empty:
        search_snip.to_parquet(out_dir / "snippets_search.parquet", index=False)
    (out_dir / "telemetry_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote telemetry.parquet ({len(tele)} rows), "
          f"snippets_citation.parquet ({len(cit_snip)} rows), "
          f"snippets_search.parquet ({len(search_snip)} rows) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
