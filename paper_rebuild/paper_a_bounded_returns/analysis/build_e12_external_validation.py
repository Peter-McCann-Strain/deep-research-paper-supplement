#!/usr/bin/env python3
"""E12 EXTERNAL VALIDATION — land the four pre-registered external tests.

Reads the E12 GPT-5.2 verdicts + generated reports + on-disk benchmark gold
answers produced by ``scripts/run_e12_extval.py`` (generate -> judge phases) and
the main leaderboard parquet, then computes and LANDS the four pre-registered
external-validation tests under canonical key ``external_validation_e12``:

  (1) rank_concordance        — Spearman rho between E12 GPT-5.2 per-pattern means
                                (pooled over the five external benchmarks) and the
                                MAIN leaderboard per-pattern means (gpt52, base_p*).
                                Also reports the flat-top-cluster survival (P1~P4
                                above P0) and the P0-over-7B (P9/P10) tiering, both
                                referenced against the main leaderboard means.
  (2) exact_match_tiering     — On the objective-answer benchmarks (litqa2,
                                deepsearch_qa) score each report by whether the gold
                                ``reference_answer`` / ``rubric.ideal`` string (and,
                                for MCQ, the correct option) appears in the report
                                text; tier patterns by exact-match accuracy and test
                                whether the P1/P4 > P0 ordering survives this
                                judge-independent metric.
  (3) gold_source_factuality  — Per-pattern mean of the GPT-5.2 ``factual_accuracy``
                                dimension, pooled over benchmarks AND restricted to
                                the gold-anchored benchmarks (draco/litqa2/research_qa
                                whose rubrics carry human reference answers); tests
                                whether the factual ordering matches the main study.
  (4) citation_triangulation  — On benchmarks that ship ``expected_citations`` (gold
                                DOIs/URLs: litqa2, freshwiki, research_qa), measure the
                                fraction of reports whose ``## References`` block hits
                                >=1 gold source (domain/DOI match), per pattern; tests
                                whether citation grounding tiers patterns the same way.

This script is BUILD+VERIFY ONLY and READ-ONLY on every protected corpus path and
on the E12 verdict/report trees.  Its single write is the canonical JSON (an
analysis artefact).  It is fully resume-safe: it derives everything from on-disk
verdicts/reports and overwrites only the ``external_validation_e12`` key via a
read-merge-write (every other canonical key is preserved byte-for-byte).  If the
E12 verdicts are incomplete it self-guards: it computes what it can, marks the
block ``status="partial"`` with per-pattern coverage counts, and (unless
``--allow-partial``) does NOT write canonical so a half-finished judge run can
never silently land a drifted number.

CANONICAL PATH FIX (binding, 2026-06-22): canonical_numbers.json was MOVED by
commit 0a80ba6 to paper_rebuild/paper_a_bounded_returns/analysis/.  This script targets
the CORRECT post-move path; it does NOT use the stale paper_rebuild/paper_a_bounded_returns
path that many sibling builders still hardcode and crash on.

Usage:
    # zero-write preview (prints would-be block, never writes canonical):
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e12_external_validation.py --dry-run

    # land canonical['external_validation_e12'] (requires complete E12 verdicts):
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e12_external_validation.py

    # land even if verdicts are incomplete (records coverage; status=partial):
    python paper_rebuild/paper_a_bounded_returns/analysis/build_e12_external_validation.py --allow-partial
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# parents[3]: paper_rebuild/paper_a_bounded_returns/analysis/<this> -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

# CORRECT post-move canonical location (NOT paper_rebuild/paper_a_bounded_returns/analysis).
ANA = _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis"
CANON = ANA / "canonical_numbers.json"

# E12 artefacts produced by scripts/run_e12_extval.py (READ-ONLY here).
E12_ROOT = _REPO_ROOT / "results" / "e12_extval"
GEN_OUT = E12_ROOT / "reports"                  # reports/<bench>/<pat>/<e12_id>.md
JUDGE_OUT = E12_ROOT / "judge_gpt52"            # judge_gpt52/<bench>__<pat>/<e12_id>.json
E12_MANIFEST = E12_ROOT / "e12_query_manifest.json"   # frozen held-out selection

# Benchmark gold (READ-ONLY) — the same files run_e12_extval.py selects from.
BENCH_DIR = _REPO_ROOT / "data" / "benchmarks"
BENCH_FILES = {
    "deepsearch_qa": BENCH_DIR / "deepsearch_qa" / "deepsearch_qa_queries.json",
    "research_qa": BENCH_DIR / "research_qa" / "research_qa_queries.json",
    "draco": BENCH_DIR / "draco" / "draco_queries.json",
    "litqa2": BENCH_DIR / "litqa2" / "litqa2_queries.json",
    "freshwiki": BENCH_DIR / "freshwiki" / "freshwiki_queries.json",
}

# Main leaderboard parquet (READ-ONLY).
MAIN_LB_PARQUET = _REPO_ROOT / "data" / "analysis" / "df_overall_scores.parquet"

E12_GEN_PATTERNS = ["p0", "p1", "p4"]

# Which benchmarks anchor each judge-independent test.
EXACT_MATCH_BENCHES = ["litqa2", "deepsearch_qa"]          # objective short answers
GOLD_FACT_BENCHES = ["draco", "litqa2", "research_qa"]      # human reference answers
# Only benchmarks whose HELD-OUT items actually ship gold expected_citations are
# usable for triangulation. Audited on disk (2026-06-22): litqa2 = 40/40 gold
# DOIs; freshwiki/research_qa held-out items carry none. We list all three but the
# loader self-skips items without gold sources, so this is litqa2-anchored in
# practice and records that in the block's per-pattern n_scored counts.
CITATION_BENCHES = ["litqa2", "freshwiki", "research_qa"]   # ship expected_citations


# ── small stats helpers (no SciPy dependency, matching run_e12_extval.py) ──────

def _spearman(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def rankdata(a: list[float]) -> list[float]:
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


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


# ── loaders ────────────────────────────────────────────────────────────────────

def load_main_leaderboard() -> dict[str, float]:
    import pandas as pd
    df = pd.read_parquet(MAIN_LB_PARQUET)
    base = df[(df["judge"] == "gpt52") &
              df["pattern"].astype(str).str.fullmatch(r"base_p\d+")]
    means = base.groupby("pattern", observed=True)["overall_score"].mean()
    return {re.sub(r"^base_", "", k): float(v) for k, v in means.items()}


def load_manifest() -> dict[str, list[dict]]:
    return json.loads(E12_MANIFEST.read_text())


def load_gold() -> dict[str, dict[str, dict]]:
    """{bench: {orig_id: item}} from the benchmark query files."""
    gold: dict[str, dict[str, dict]] = {}
    for bench, path in BENCH_FILES.items():
        items = json.loads(path.read_text())
        if isinstance(items, dict):
            items = items.get("queries", items)
        gold[bench] = {str(it.get("id", "")): it for it in items}
    return gold


def load_verdicts() -> dict[tuple[str, str], dict[str, dict]]:
    """{(bench, pat): {e12_id: verdict}} from JUDGE_OUT/<bench>__<pat>/<e12_id>.json."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    if not JUDGE_OUT.exists():
        return out
    for sub in sorted(JUDGE_OUT.iterdir()):
        if not sub.is_dir() or "__" not in sub.name:
            continue
        bench, pat = sub.name.split("__", 1)
        for jf in sub.glob("*.json"):
            try:
                out[(bench, pat)][jf.stem] = json.loads(jf.read_text())
            except Exception:
                continue
    return out


def load_report_text(bench: str, pat: str, e12_id: str) -> str | None:
    p = GEN_OUT / bench / pat / f"{e12_id}.md"
    if not p.exists():
        return None
    try:
        return p.read_text()
    except Exception:
        return None


# ── per-id orig-id lookup from frozen manifest ──────────────────────────────────

def build_e12_to_orig(manifest: dict) -> dict[tuple[str, str], str]:
    """{(bench, e12_id): orig_id} so verdicts/reports map back to gold items."""
    m: dict[tuple[str, str], str] = {}
    for bench, rows in manifest.items():
        for r in rows:
            m[(bench, r["e12_id"])] = str(r.get("orig_id", ""))
    return m


# ── Test 1: rank concordance + cluster/tiering survival ────────────────────────

def test_rank_concordance(verdicts, main_lb) -> dict:
    e12_overall: dict[str, list[float]] = defaultdict(list)
    for (bench, pat), recs in verdicts.items():
        for rec in recs.values():
            try:
                e12_overall[pat].append(float(rec["overall_score"]))
            except Exception:
                continue
    e12_means = {p: _mean(v) for p, v in e12_overall.items()}
    shared = sorted(p for p in e12_means if p in main_lb and not math.isnan(e12_means[p]))
    rho = _spearman([main_lb[p] for p in shared], [e12_means[p] for p in shared])

    flat_top = None
    if all(p in e12_means and not math.isnan(e12_means[p]) for p in ("p0", "p1", "p4")):
        p0, p1, p4 = e12_means["p0"], e12_means["p1"], e12_means["p4"]
        flat_top = {
            "p0": p0, "p1": p1, "p4": p4,
            "p1_p4_gap": abs(p1 - p4),
            "p1_above_p0": p1 > p0,
            "p4_above_p0": p4 > p0,
            "top_cluster_flat": abs(p1 - p4) < 0.02,
            "survives": (abs(p1 - p4) < 0.02) and (p1 > p0) and (p4 > p0),
        }

    tiering = None
    if "p0" in e12_means and not math.isnan(e12_means["p0"]):
        p0e = e12_means["p0"]
        p9, p10 = main_lb.get("p9"), main_lb.get("p10")
        tiering = {
            "p0_e12": p0e, "p9_main_lb": p9, "p10_main_lb": p10,
            "p0_above_p9": (p9 is not None and p0e > p9),
            "p0_above_p10": (p10 is not None and p0e > p10),
            "note": "7B arms referenced from main leaderboard means; not regenerated.",
        }
    return {
        "spearman_rho": rho,
        "n_patterns": len(shared),
        "patterns": shared,
        "e12_means": {p: e12_means[p] for p in shared},
        "main_lb_means": {p: main_lb[p] for p in shared},
        "n_reports_per_pattern": {p: len(v) for p, v in e12_overall.items()},
        "flat_top_cluster": flat_top,
        "p0_7b_tiering": tiering,
    }


# ── Test 2: exact-match tiering (judge-independent) ────────────────────────────

def _gold_answer_strings(item: dict) -> list[str]:
    out = []
    ref = (item.get("reference_answer") or "").strip()
    rub = item.get("rubric") or {}
    ideal = (rub.get("ideal") or rub.get("expected_answer") or "").strip()
    for s in (ref, ideal):
        if s and len(s) <= 120:        # short objective answer only
            out.append(s)
    return list(dict.fromkeys(out))


def _exact_match(report_text: str, gold_strings: list[str]) -> bool | None:
    if not gold_strings:
        return None
    hay = report_text.lower()
    return any(g.lower() in hay for g in gold_strings)


def test_exact_match_tiering(manifest, gold, e2o) -> dict:
    per_pat_hits: dict[str, list[int]] = defaultdict(list)
    for bench in EXACT_MATCH_BENCHES:
        for r in manifest.get(bench, []):
            eid, oid = r["e12_id"], str(r.get("orig_id", ""))
            item = gold.get(bench, {}).get(oid)
            if not item:
                continue
            gs = _gold_answer_strings(item)
            if not gs:
                continue
            for pat in E12_GEN_PATTERNS:
                txt = load_report_text(bench, pat, eid)
                if txt is None:
                    continue
                m = _exact_match(txt, gs)
                if m is not None:
                    per_pat_hits[pat].append(1 if m else 0)
    acc = {p: _mean(v) for p, v in per_pat_hits.items()}
    survives = None
    if all(p in acc and not math.isnan(acc[p]) for p in ("p0", "p1", "p4")):
        survives = (acc["p1"] >= acc["p0"]) and (acc["p4"] >= acc["p0"])
    return {
        "benchmarks": EXACT_MATCH_BENCHES,
        "exact_match_accuracy": acc,
        "n_scored_per_pattern": {p: len(v) for p, v in per_pat_hits.items()},
        "p1_p4_ge_p0_survives": survives,
        "metric": "case-insensitive substring containment of gold short answer",
    }


# ── Test 3: gold-source factuality (GPT-5.2 factual_accuracy dim) ───────────────

def test_gold_source_factuality(verdicts) -> dict:
    pooled: dict[str, list[float]] = defaultdict(list)
    gold_anchored: dict[str, list[float]] = defaultdict(list)
    for (bench, pat), recs in verdicts.items():
        for rec in recs.values():
            dims = rec.get("dimensions") or {}
            fa = dims.get("factual_accuracy") or {}
            sc = fa.get("score")
            if sc is None:
                continue
            pooled[pat].append(float(sc))
            if bench in GOLD_FACT_BENCHES:
                gold_anchored[pat].append(float(sc))
    pooled_mean = {p: _mean(v) for p, v in pooled.items()}
    gold_mean = {p: _mean(v) for p, v in gold_anchored.items()}
    ordering_survives = None
    if all(p in gold_mean and not math.isnan(gold_mean[p]) for p in ("p0", "p1", "p4")):
        ordering_survives = (gold_mean["p1"] >= gold_mean["p0"]) and \
                            (gold_mean["p4"] >= gold_mean["p0"])
    return {
        "factual_accuracy_pooled_mean": pooled_mean,
        "factual_accuracy_gold_anchored_mean": gold_mean,
        "gold_anchored_benchmarks": GOLD_FACT_BENCHES,
        "n_pooled_per_pattern": {p: len(v) for p, v in pooled.items()},
        "p1_p4_ge_p0_survives": ordering_survives,
    }


# ── Test 4: citation triangulation against gold sources ────────────────────────

_URL_RE = re.compile(r"https?://[^\s\)\]]+", re.I)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\)\];,]+", re.I)


def _domain(u: str) -> str:
    m = re.match(r"https?://([^/]+)/?", u, re.I)
    d = (m.group(1) if m else u).lower()
    return d[4:] if d.startswith("www.") else d


def _report_sources(report_text: str) -> tuple[set[str], set[str]]:
    """Return (domains, dois) found in the report's reference/citation links."""
    domains, dois = set(), set()
    for u in _URL_RE.findall(report_text):
        domains.add(_domain(u))
        dm = _DOI_RE.search(u)
        if dm:
            dois.add(dm.group(0).lower().rstrip(".,;)"))
    for d in _DOI_RE.findall(report_text):
        dois.add(d.lower().rstrip(".,;)"))
    return domains, dois


def _gold_sources(item: dict) -> tuple[set[str], set[str]]:
    domains, dois = set(), set()
    for c in (item.get("expected_citations") or []):
        c = str(c)
        domains.add(_domain(c))
        dm = _DOI_RE.search(c)
        if dm:
            dois.add(dm.group(0).lower().rstrip(".,;)"))
    return domains, dois


def test_citation_triangulation(manifest, gold) -> dict:
    per_pat_hit: dict[str, list[int]] = defaultdict(list)
    for bench in CITATION_BENCHES:
        for r in manifest.get(bench, []):
            eid, oid = r["e12_id"], str(r.get("orig_id", ""))
            item = gold.get(bench, {}).get(oid)
            if not item:
                continue
            gdom, gdoi = _gold_sources(item)
            if not gdom and not gdoi:
                continue
            for pat in E12_GEN_PATTERNS:
                txt = load_report_text(bench, pat, eid)
                if txt is None:
                    continue
                rdom, rdoi = _report_sources(txt)
                hit = bool(rdoi & gdoi) or bool(rdom & gdom)
                per_pat_hit[pat].append(1 if hit else 0)
    rate = {p: _mean(v) for p, v in per_pat_hit.items()}
    survives = None
    if all(p in rate and not math.isnan(rate[p]) for p in ("p0", "p1", "p4")):
        survives = (rate["p1"] >= rate["p0"]) and (rate["p4"] >= rate["p0"])
    return {
        "benchmarks": CITATION_BENCHES,
        "gold_source_hit_rate": rate,
        "n_scored_per_pattern": {p: len(v) for p, v in per_pat_hit.items()},
        "p1_p4_ge_p0_survives": survives,
        "metric": "report References block shares >=1 gold DOI or gold domain",
    }


# ── coverage / completeness guard ──────────────────────────────────────────────

def coverage(manifest, verdicts) -> dict:
    expected = {}
    have = {}
    for bench, rows in manifest.items():
        for pat in E12_GEN_PATTERNS:
            key = f"{bench}__{pat}"
            expected[key] = len(rows)
            have[key] = len(verdicts.get((bench, pat), {}))
    total_exp = sum(expected.values())
    total_have = sum(have.values())
    missing = {k: expected[k] - have[k] for k in expected if have[k] < expected[k]}
    return {
        "expected_verdicts": total_exp,
        "have_verdicts": total_have,
        "complete": total_have >= total_exp,
        "missing_by_arm": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Land canonical['external_validation_e12'].")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print would-be block; NEVER write canonical.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Write canonical even if E12 verdicts are incomplete (status=partial).")
    args = ap.parse_args()

    for req in (E12_MANIFEST, JUDGE_OUT, MAIN_LB_PARQUET):
        if not req.exists():
            print(f"[e12-extval] required input missing: {req}\n"
                  f"             run scripts/run_e12_extval.py --phase generate/judge first.")
            return 1

    manifest = load_manifest()
    verdicts = load_verdicts()
    gold = load_gold()
    main_lb = load_main_leaderboard()

    cov = coverage(manifest, verdicts)
    status = "complete" if cov["complete"] else "partial"

    block = {
        "status": status,
        "coverage": cov,
        "judge_model": "gpt-5.2",
        "generation_model": "gpt-4o",
        "generation_deployment": "sthree-ptu-02",
        "benchmarks": list(BENCH_FILES.keys()),
        "n_heldout_per_benchmark": {b: len(rows) for b, rows in manifest.items()},
        "test_1_rank_concordance": test_rank_concordance(verdicts, main_lb),
        "test_2_exact_match_tiering": test_exact_match_tiering(manifest, gold, build_e12_to_orig(manifest)),
        "test_3_gold_source_factuality": test_gold_source_factuality(verdicts),
        "test_4_citation_triangulation": test_citation_triangulation(manifest, gold),
    }

    print(json.dumps({"external_validation_e12": block}, indent=1))

    if args.dry_run:
        print("\n[DRY RUN] canonical_numbers.json NOT written.")
        return 0
    if status == "partial" and not args.allow_partial:
        print(f"\n[REFUSING] E12 verdicts incomplete "
              f"({cov['have_verdicts']}/{cov['expected_verdicts']}); "
              f"missing: {cov['missing_by_arm']}.\n"
              f"           Finish the judge phase, or pass --allow-partial to land "
              f"a status=partial block. Canonical NOT written.")
        return 2

    canon = json.loads(CANON.read_text()) if CANON.exists() else {}
    canon["external_validation_e12"] = block
    CANON.write_text(json.dumps(canon, indent=1))
    print(f"\nWrote canonical_numbers.json['external_validation_e12'] -> {CANON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
