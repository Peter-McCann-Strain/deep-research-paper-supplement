#!/usr/bin/env python3
"""reconcile_exclusions.py — Audit CG-8 exclusion reconciliation.

For each analysis arm, compute the expected full grid of judging cells
(patterns x queries x judges) and reconcile it against what is actually
observed in ``data/analysis/df_verdicts.parquet``.

A "cell" is a unique (pattern, query_id, judge) triple. A cell is "observed"
iff at least one criterion verdict exists for that triple in df_verdicts.

The reconciliation classifies every *missing* cell (expected - observed) into:

  D1  documented 82de3e92 Claude-panel exclusion (AUP false positive),
      sourced from the quarantine.json files referenced by EXCLUSIONS.md E1.
  D2  documented claude_code reproducibility probe (the 10,600-row probe is a
      *judge that is not part of any arm's intended panel*; its absence on the
      panel arms is by design).
  P   report-generation failure: the underlying report does not exist
      (df_runs.report_exists == False), so NO judge could score it. These are
      pipeline failures, not judging gaps; they are the headline n_queries
      holes (e.g. P6=87, P5=89, P3=89).
  B   by-design partial-panel arm: the arm was never intended to carry this
      judge for every query (single-judge variance / disentanglement arms, the
      sonnet-spot-check + claude_code protocol_a probe, opus-not-run on the
      late-added P11/P12, etc.). Encoded per-arm in ARM_SPECS.
  D3a documented P11/P12 reduced Claude-judging subset (EXCLUSIONS.md E3a): the
      two late-added 7B patterns were Claude-judged only via the manual packet
      `20260604_e2_p11_p12`, on a reduced query subset (P11=80, P12=52). Queries
      outside that packet subset are a documented by-design gap.
  D3b documented incidental Claude-API backfill gap (EXCLUSIONS.md E3b): a base
      report that exists and was GPT-5.2-judged but has no Claude verdict, and
      is NOT an AUP refusal (DOCUMENTED_CLAUDE_BACKFILL_GAPS).
  R   RESIDUAL — a missing cell that is NOT covered by any of the above. The
      report exists, the judge is part of the arm's intended panel, and there
      is no documented exclusion. These are the cells the auditor wants
      surfaced: either to be added to EXCLUSIONS.md (if legitimate) or flagged
      as truly unexplained.

Exit code is 0 iff observed gaps == documented gaps for the panel arms (i.e.
no class-R residual remains after D/P/B accounting). Non-zero otherwise.

Audit CG-8, 2026-06-10.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VERDICTS = REPO / "data" / "analysis" / "df_verdicts.parquet"
RUNS = REPO / "data" / "analysis" / "df_runs.parquet"
EXCLUSIONS_MD = REPO / "reports" / "paper_world_class" / "EXCLUSIONS.md"
QUARANTINE_FILES = [
    REPO / "reports" / "claude_code_judging" / "20260608_base_cluster_opus" / "quarantine.json",
    REPO / "reports" / "claude_code_judging" / "20260608_oracle_3judge" / "quarantine.json",
]

PANEL_JUDGES = ("gpt52", "claude_opus", "claude_sonnet")
CLAUDE_PANEL = ("claude_opus", "claude_sonnet")

# --------------------------------------------------------------------------- #
# Documented Claude-panel coverage holes (EXCLUSIONS.md E3b). These are reports
# that exist and were scored by GPT-5.2 but have NO Claude verdict from either
# Opus or Sonnet, and are NOT AUP refusals (distinct from E1) — incidental
# Claude-API backfill gaps. Documented in EXCLUSIONS.md; listed here so the
# reconciliation recognizes them as accounted-for rather than class-R residual.
# --------------------------------------------------------------------------- #
DOCUMENTED_CLAUDE_BACKFILL_GAPS = {
    ("base_p1", "8e99d8d2-f6b9-4800-83a9-6f56829898fe", "claude_opus"),
    ("base_p1", "8e99d8d2-f6b9-4800-83a9-6f56829898fe", "claude_sonnet"),
    ("base_p3", "b3c576e7-dfc6-403f-90e7-53c011884d5c", "claude_opus"),
    ("base_p3", "b3c576e7-dfc6-403f-90e7-53c011884d5c", "claude_sonnet"),
}


# --------------------------------------------------------------------------- #
# Arm specifications: which (patterns, judges) grid each arm is *intended* to
# fill. Judges not listed for an arm are treated as "not part of this arm" and
# their absence is never counted as a gap. Patterns are matched by regex on the
# pattern name within the given pattern_family.
# --------------------------------------------------------------------------- #
ARM_SPECS = [
    {
        "arm": "base_panel",
        "family": "base",
        "pattern_re": r"^base_p\d+$",          # 13 headline patterns P0-P12
        "judges": list(PANEL_JUDGES),
        # P11/P12 were added late and Claude *Opus* was never run on them; the
        # intended Claude judge for those two is sonnet only (via claude_code
        # path -> recorded under claude_sonnet). Encode as a per-pattern judge
        # exception so opus-absence on p11/p12 is class-B, not class-R.
        "judge_exceptions": {
            "base_p11": list(PANEL_JUDGES[:1]) + ["claude_sonnet"],  # gpt52 + sonnet
            "base_p12": list(PANEL_JUDGES[:1]) + ["claude_sonnet"],  # gpt52 + sonnet
        },
    },
    {
        "arm": "oracle_panel",
        "family": "oracle",
        "pattern_re": r"^oracle_t1_p\d+$",
        "judges": list(PANEL_JUDGES),
        # oracle p2 and p3 were GPT-5.2-only by design (no Claude re-judge packet
        # was ever produced for them); Claude panel is intended only for the
        # 7-pattern oracle_3judge packet {p0,p1,p4,p5,p6,p7,p8}.
        "judge_exceptions": {
            "oracle_t1_p2": ["gpt52"],
            "oracle_t1_p3": ["gpt52"],
        },
    },
    # ----- single-judge / probe arms (entirely by design; no class-R possible) -----
    {
        "arm": "variance_gpt52_only",
        "family": "variance",
        "pattern_re": r".*",
        "judges": ["gpt52"],
        "judge_exceptions": {},
        "by_design": True,
    },
    {
        "arm": "disentanglement_gpt52_only",
        "family": "disentanglement",
        "pattern_re": r".*",
        "judges": ["gpt52"],
        "judge_exceptions": {},
        "by_design": True,
    },
    {
        "arm": "ablation_panel",
        "family": "ablation",
        "pattern_re": r".*",
        # Ablations were judged primarily by gpt52 + sonnet; opus is a sparse
        # spot-check. Intended panel is gpt52 + sonnet; opus is opportunistic.
        "judges": ["gpt52", "claude_sonnet"],
        "judge_exceptions": {},
        "by_design": True,
    },
    {
        "arm": "protocol_a_probe",
        "family": "protocol_a",
        "pattern_re": r".*",
        # Bing-vs-Tavily robustness probe: gpt52 + claude_code, with a 4-query
        # sonnet spot-check. By design partial.
        "judges": ["gpt52", "claude_code", "claude_sonnet"],
        "judge_exceptions": {},
        "by_design": True,
    },
]


def load_quarantine_cells() -> set[tuple[str, str, str]]:
    """Return the set of (pattern, query_id, judge) cells documented as the
    82de3e92 AUP exclusion, expanded from the quarantine.json excluded_tasks x
    affected_judges. EXCLUSIONS.md counts these as 13 (packet x pattern) cells;
    expanded over the two affected Claude judges that is 26 (pattern,query,judge)
    cells. Only those whose report actually appears as a missing cell will be
    matched (the base-cluster ones are present via the separate base mechanism).
    """
    cells: set[tuple[str, str, str]] = set()
    docs = []
    for qf in QUARANTINE_FILES:
        d = json.loads(qf.read_text())
        docs.append((qf, d))
        qid = d["query_id"]
        judges = d["affected_judges"]
        for task in d["excluded_tasks"]:
            # task id form: "<pattern>__<query_id>__claude-code-manual"
            pattern = task.split("__", 1)[0]
            for j in judges:
                cells.add((pattern, qid, j))
    return cells, docs


# Packet that defines the reduced P11/P12 Claude-judging query subset (EXCLUSIONS.md E3a).
P11_P12_PACKET = (
    REPO / "reports" / "claude_code_judging" / "20260604_e2_p11_p12" / "parsed" / "parse_summary.csv"
)


def load_p11_p12_judged_subset() -> dict[str, set[str]]:
    """Return {pattern -> set(query_id)} that the P11/P12 manual Claude-Code
    judging packet actually covered. P11/P12 (the late-added 7B patterns) were
    Claude-judged ONLY via this packet, on a reduced subset by design; queries
    outside the subset are a documented by-design gap (E3a), not class-R.
    Returns {} if the packet is absent (then E3a cells fall back to class-R and
    are surfaced, which is the safe behaviour)."""
    if not P11_P12_PACKET.exists():
        return {}
    df = pd.read_csv(P11_P12_PACKET)
    out: dict[str, set[str]] = {}
    for pat, g in df.groupby("pattern"):
        out[str(pat)] = set(g["query_id"].astype(str))
    return out


def main() -> int:
    v = pd.read_parquet(VERDICTS, columns=["pattern", "pattern_family", "query_id", "judge"])
    v["pattern"] = v["pattern"].astype(str)
    v["pattern_family"] = v["pattern_family"].astype(str)
    v["judge"] = v["judge"].astype(str)

    runs = pd.read_parquet(RUNS, columns=["pattern", "query_id", "report_exists"])
    runs["pattern"] = runs["pattern"].astype(str)
    report_exists = set(
        (r.pattern, r.query_id) for r in runs.itertuples() if bool(r.report_exists)
    )

    observed = set(
        map(tuple, v[["pattern", "query_id", "judge"]].drop_duplicates().to_numpy())
    )

    # Universe of queries per arm = the queries that appear anywhere in that arm's
    # family (a report was attempted for them). Using the family-local query set
    # avoids inventing cells for queries an arm never ran.
    quarantine_cells, quarantine_docs = load_quarantine_cells()
    p11_p12_subset = load_p11_p12_judged_subset()

    print("=" * 78)
    print("EXCLUSION RECONCILIATION (audit CG-8)")
    print("=" * 78)
    print(f"df_verdicts cells (pattern,query,judge): {len(observed):,}")
    print(f"df_verdicts rows: {len(v):,}")
    cc_rows = int((v["judge"] == "claude_code").sum())
    print(f"claude_code probe rows (documented 10,600): {cc_rows:,}")
    print()
    print("Documented 82de3e92 quarantine cells (packet x pattern x affected-judge):")
    for qf, d in quarantine_docs:
        n = len(d["excluded_tasks"]) * len(d["affected_judges"])
        print(
            f"  {qf.parent.name}: {len(d['excluded_tasks'])} patterns "
            f"x {len(d['affected_judges'])} judges = {n} cells"
        )
    print(f"  -> total expanded quarantine cells: {len(quarantine_cells)}")
    print()

    all_residuals: list[tuple[str, str, str, str]] = []  # (arm, pattern, query, judge)
    summary_rows = []

    for spec in ARM_SPECS:
        fam = spec["family"]
        d = v[v["pattern_family"] == fam]
        pats = sorted(p for p in d["pattern"].unique() if re.match(spec["pattern_re"], p))
        arm_queries = sorted(d["query_id"].unique())
        judge_exc = spec.get("judge_exceptions", {})

        expected = set()
        for p in pats:
            judges = judge_exc.get(p, spec["judges"])
            for q in arm_queries:
                for j in judges:
                    expected.add((p, q, j))

        obs_arm = {(p, q, j) for (p, q, j) in observed if p in pats}
        missing = expected - obs_arm

        # classify. Documented categories (D*) and design categories (P/B/D3a)
        # are accounted-for; only class R is an undocumented residual.
        cls = {"D1": [], "D2": [], "P": [], "B": [], "D3a": [], "D3b": [], "R": []}
        for (p, q, j) in sorted(missing):
            if (p, q, j) in quarantine_cells:
                cls["D1"].append((p, q, j))           # E1: 82de3e92 AUP refusal
            elif j == "claude_code":
                cls["D2"].append((p, q, j))           # claude_code probe (not a panel judge)
            elif (p, q) not in report_exists:
                cls["P"].append((p, q, j))            # E2: report-generation failure
            elif (
                p in ("base_p11", "base_p12")
                and j == "claude_sonnet"
                and p in p11_p12_subset
                and q not in p11_p12_subset[p]
            ):
                cls["D3a"].append((p, q, j))          # E3a: P11/P12 reduced Claude subset
            elif (p, q, j) in DOCUMENTED_CLAUDE_BACKFILL_GAPS:
                cls["D3b"].append((p, q, j))          # E3b: documented Claude backfill gap
            elif spec.get("by_design"):
                cls["B"].append((p, q, j))            # single/dual-judge arm by design
            else:
                cls["R"].append((p, q, j))            # UNDOCUMENTED residual

        for (p, q, j) in cls["R"]:
            all_residuals.append((spec["arm"], p, q, j))

        summary_rows.append(
            {
                "arm": spec["arm"],
                "patterns": len(pats),
                "queries": len(arm_queries),
                "judges": len(spec["judges"]),
                "expected": len(expected),
                "observed": len(obs_arm),
                "missing": len(missing),
                "D1_quarantine": len(cls["D1"]),
                "D2_cc_probe": len(cls["D2"]),
                "P_missing_report": len(cls["P"]),
                "D3a_p11p12_subset": len(cls["D3a"]),
                "D3b_backfill": len(cls["D3b"]),
                "B_by_design": len(cls["B"]),
                "R_residual": len(cls["R"]),
            }
        )

        print("-" * 78)
        print(f"ARM: {spec['arm']}  (family={fam})")
        print(
            f"  patterns={len(pats)} queries={len(arm_queries)} "
            f"intended_judges={spec['judges']}"
        )
        print(
            f"  expected={len(expected):,}  observed={len(obs_arm):,}  "
            f"missing={len(missing):,}"
        )
        print(
            f"  classified: D1(82de3e92)={len(cls['D1'])}  "
            f"D2(cc-probe)={len(cls['D2'])}  P(missing-report)={len(cls['P'])}  "
            f"D3a(p11/p12-subset)={len(cls['D3a'])}  D3b(backfill)={len(cls['D3b'])}  "
            f"B(by-design)={len(cls['B'])}  R(RESIDUAL)={len(cls['R'])}"
        )
        if cls["P"]:
            # report-failure holes collapse to (pattern,query) -> show those
            pq = sorted({(p, q) for (p, q, _) in cls["P"]})
            print(f"  P report-failure (pattern,query) holes [{len(pq)}]:")
            for (p, q) in pq:
                print(f"      {p}  {q}")
        if cls["R"]:
            print(f"  *** RESIDUAL cells [{len(cls['R'])}] (report exists, judge in panel, NOT documented): ***")
            for (p, q, j) in cls["R"]:
                print(f"      {p}  {q}  {j}")

    print()
    print("=" * 78)
    print("PER-ARM SUMMARY")
    print("=" * 78)
    sdf = pd.DataFrame(summary_rows)
    print(sdf.to_string(index=False))

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    n_resid = len(all_residuals)
    if n_resid == 0:
        print("PASS: observed gaps == documented gaps.")
        print("Every missing cell is accounted for by: the documented 82de3e92")
        print("quarantine (E1), the claude_code probe, a missing-report generation")
        print("failure (E2), the P11/P12 reduced Claude-judging subset (E3a), a")
        print("documented Claude backfill gap (E3b), or a by-design partial-panel")
        print("arm. No undocumented residual.")
    else:
        print(f"RESIDUAL FOUND: {n_resid} undocumented missing cell(s):")
        for (arm, p, q, j) in all_residuals:
            print(f"  [{arm}] {p}  {q}  {j}")
        print()
        print("These are report-exists / judge-in-panel cells with NO documented")
        print("exclusion. They must be added to EXCLUSIONS.md (if legitimate) or")
        print("flagged as truly unexplained.")

    return 0 if n_resid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
