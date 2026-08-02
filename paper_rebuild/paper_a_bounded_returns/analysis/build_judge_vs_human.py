#!/usr/bin/env python
"""build_judge_vs_human.py — generalise judge-vs-gold into a benchmark-family loop
over the PUBLIC human-label sets (E3 N_eff / E4 cite-causal / E12 layer-0 / E13' G4).

This is the human-anchored sibling of ``build_judge_vs_gold.py`` (which anchors our
panel against *mechanical* manifest gold, LitQA2+DeepSearchQA). Here the gold is
*human* expert labels loaded through the common-schema adapters in
``deep_research/benchmarks/gold_loaders.py`` (built earlier in this run; imported, not
re-implemented). The matching / answer-discrimination / seeded-cluster-bootstrap /
AUC machinery is reused from ``build_judge_vs_gold.py``.

What it computes
----------------
For each JUDGE FAMILY ∈ {gpt52, claude_opus, claude_sonnet, local} and each human-label
set ∈ {drb_race, healthbench, expertqa, deepfactbench, draco_full}, where verdicts for
that (family, set) exist:

  AGREEMENT
    * Macro-F1            (binarised verdict vs binarised human grade)
    * point-biserial r    (continuous verdict score vs binary human grade)
    * AUC                 (verdict score ranking the human grade)
    all with a SEEDED CLUSTER bootstrap 95% CI (clusters = the natural grouping of the
    set: pair_key / question / report_id / task_id) so dependence within a report or
    question is not under-counted.

  CARE CONFOUNDER (the N_eff input)
    * per-family inter-judge correlation CONDITIONED on the human grade. CARE asks:
      after we know the human label, do two judges in the SAME family still agree more
      than chance? The correlation that SURVIVES conditioning on the human grade is the
      family-level dependence that inflates a naive judge count — exactly the quantity
      N_eff must discount. Reported per human-label set and pooled.

  E4 DENSITY-RESIDUAL HOOK (ExpertQA)
    * regress each family's factual_accuracy verdict on the claim's CITATION COUNT
      (parsed from the inline ``[n]`` markers in the claim text), CONDITIONING on the
      expert factuality grade. A non-zero partial citation coefficient AFTER expert
      factuality is held fixed = the density artefact (the judge rewards citation
      markup independent of whether the claim is actually correct). ExpertQA is the
      clean instrument because factuality and attribution are labelled SEPARATELY on
      the same claim.

DRB-RACE PLUG-IN (our panel over the 400 released reports, judged LATER)
  drb_race gold rows carry meta[system]+meta[task_id]; this script builds the SAME join
  key on any verdict store, so when our 9-dim panel finishes scoring the 400 reports its
  verdicts slot straight into the per-family agreement / CARE tables with no code change.
  Until then drb_race appears in the schema with ``status="awaiting_panel"``.

SAFETY: writes ONLY ``canonical_numbers.json['judge_vs_human']`` under this analysis dir
(atomic tmp-then-replace, the build_judge_vs_gold lesson). Reads our verdict stores
read-only. Makes ZERO network / API calls. The self-test runs on the loaded real gold
plus a small synthetic verdict set, and on the one REAL on-disk machine verdict that
ships inside the gold (DeepFactBench ``agent_verdict``), so the full pipeline is
exercised end-to-end against real data without spending a cent.

Determinism: a single seeded numpy generator over SORTED inputs.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

# --- repo-root on path so the module imports cleanly however it is invoked ----
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_research.benchmarks.gold_loaders import (  # noqa: E402
    LOADERS,
    load_deepfactbench,
)

ROOT = str(_REPO_ROOT)
ANA = f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis"
SEED = 20260615

# Judge families and where their verdicts on OUR reports live (read-only). The
# external human sets are not yet panel-judged, so these stores are consulted via
# load_family_verdicts() and currently return nothing for the external sets; the
# wiring is in place so a later panel run plugs in with no code change.
PANEL_FAMILIES = ["gpt52", "claude_opus", "claude_sonnet", "local"]
FAMILY_VERDICT_DIRS: dict[str, list[str]] = {
    "gpt52": ["results/judge_gpt52"],
    "claude_opus": ["results/judge_claude_opus48", "results/judge_claude_opus"],
    "claude_sonnet": ["results/judge_claude_sonnet48", "results/judge_claude_sonnet"],
    "local": ["results/judge_local"],  # local 7B attribution/factuality judge (E13' build)
}

# Which human sets carry a usable binary/continuous grade for agreement stats, and the
# rubric dimension each one anchors. drb_race is multi-dimension and graded per system.
HUMAN_SETS = ["drb_race", "healthbench", "expertqa", "deepfactbench", "draco_full"]

# The natural cluster key per set (so the bootstrap resamples whole reports / questions,
# not individual rows) — read off the common-schema meta produced by gold_loaders.
CLUSTER_KEY: dict[str, Callable[[dict], str]] = {
    "drb_race": lambda r: str(r["meta"].get("task_id")),
    "healthbench": lambda r: str(r["meta"].get("pair_key")),
    "expertqa": lambda r: str(r["meta"].get("question")),
    "deepfactbench": lambda r: str(r["meta"].get("report_id")),
    "draco_full": lambda r: str(r["meta"].get("task_id")),
}

rng = np.random.default_rng(SEED)


def _stable_small_int(text: str, modulus: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus

# inline citation markers like [4] or [3, 7] -> distinct indices referenced
_CIT_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


# =============================================================================
# reused stats kernels (lifted from build_judge_vs_gold.py, generalised)
# =============================================================================
def auc(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """Tie-corrected AUC of score x ranking binary label y (Mann-Whitney U / n_pos n_neg)."""
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    vals = np.concatenate([pos, neg])
    s = np.argsort(vals, kind="mergesort")
    sv = vals[s]
    ranks = np.empty(len(vals), dtype=float)
    ranks[s] = np.arange(1, len(vals) + 1, dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[s[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return round(float(u / (len(pos) * len(neg))), 4)


def point_biserial(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(np.unique(y)) < 2 or np.std(x) == 0:
        return None
    return round(float(np.corrcoef(y, x)[0, 1]), 4)


def macro_f1(y: np.ndarray, yhat: np.ndarray) -> Optional[float]:
    """Macro-F1 over the {0,1} classes (binarised human grade vs binarised verdict)."""
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    if len(y) == 0:
        return None
    f1s = []
    for cls in (0, 1):
        tp = int(np.sum((yhat == cls) & (y == cls)))
        fp = int(np.sum((yhat == cls) & (y != cls)))
        fn = int(np.sum((yhat != cls) & (y == cls)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return round(float(np.mean(f1s)), 4)


def cluster_bootstrap(
    rows: list[dict], stat: Callable[[np.ndarray, np.ndarray], Optional[float]],
    y_key: str, x_key: str, cluster_ids: list[str], n_boot: int = 2000,
) -> Optional[dict]:
    """Seeded cluster bootstrap CI for stat(y, x). Resamples whole clusters (reports /
    questions) so within-cluster dependence is not under-counted. Deterministic: the
    module-level seeded generator iterates over SORTED unique clusters."""
    y = np.array([r[y_key] for r in rows], dtype=float)
    x = np.array([r[x_key] for r in rows], dtype=float)
    obs = stat(y, x)
    if obs is None:
        return None
    uniq = sorted(set(cluster_ids))
    if len(uniq) < 2:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    by_c: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        by_c[c].append(i)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx: list[int] = []
        for k in pick:
            idx.extend(by_c[uniq[k]])
        v = stat(y[idx], x[idx])
        if v is not None:
            boots.append(v)
    if not boots:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_clusters": len(uniq)}


# =============================================================================
# DRB-RACE correlation kernels (our per-dimension verdict score vs expert score)
# =============================================================================
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties shared) for a Spearman correlation."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # resolve ties to their average rank so Spearman is tie-corrected
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def _corr_bootstrap(
    xs: list[float], ys: list[float], clusters: list[str],
    stat: Callable[[np.ndarray, np.ndarray], Optional[float]], n_boot: int = 2000,
) -> Optional[dict]:
    """Seeded cluster-bootstrap CI for a correlation, resampling whole tasks (the
    natural DRB-RACE cluster) so within-task dependence across systems/dims is not
    under-counted. Deterministic via the module-level seeded generator over SORTED
    unique clusters."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    obs = stat(x, y)
    if obs is None:
        return None
    uniq = sorted(set(clusters))
    if len(uniq) < 2:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    by_c: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        by_c[c].append(i)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx: list[int] = []
        for k in pick:
            idx.extend(by_c[uniq[k]])
        v = stat(x[idx], y[idx])
        if v is not None:
            boots.append(v)
    if not boots:
        return {"point": obs, "ci95": None, "n_clusters": len(uniq)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_clusters": len(uniq)}


def load_drb_verdicts(verdicts_dir: str) -> dict[tuple[str, int, str], float]:
    """Read our panel verdicts on the DRB reports READ-ONLY and return
    {(system, task_id, dimension): score}.

    Each verdict file is results/drbrace/<judge>/drbrace_<system>/drb1_<task>.json
    carrying ``query_id="drb1_<task>"``, ``pattern="drbrace_<system>"`` and
    per-dimension ``dimensions{}[dim].score`` (our 9-dim panel). The (system,
    task_id, dim) key is exactly the gold join key.
    """
    vroot = Path(verdicts_dir)
    if not vroot.is_absolute():
        vroot = _REPO_ROOT / vroot
    out: dict[tuple[str, int, str], float] = {}
    if not vroot.is_dir():
        return out
    for sysdir in sorted(vroot.glob("drbrace_*")):
        if not sysdir.is_dir():
            continue
        system = sysdir.name[len("drbrace_"):]
        for fp in sorted(sysdir.glob("drb1_*.json")):
            try:
                rec = json.loads(fp.read_text())
            except Exception:
                continue
            qid = rec.get("query_id") or fp.stem
            try:
                task = int(str(qid).split("_")[-1])
            except (TypeError, ValueError):
                continue
            # prefer pattern field for the system, fall back to the dir name
            patt = rec.get("pattern", "")
            sys_name = patt[len("drbrace_"):] if patt.startswith("drbrace_") else system
            for dim, dd in (rec.get("dimensions") or {}).items():
                sc = dd.get("score") if isinstance(dd, dict) else None
                if sc is not None:
                    out[(sys_name, task, dim)] = float(sc)
    return out


def drb_human_by_cell() -> dict[tuple[str, int, str], float]:
    """Mean expert human score per (system, task_id, rubric_dimension) over the
    ~3 annotation replicates per cell (gold via load_drb_race)."""
    acc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for r in load_gold("drb_race"):
        key = (str(r["meta"]["system"]), int(r["meta"]["task_id"]), r["dimension"])
        acc[key].append(float(r["human_label"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_drb_race(verdicts_dir: str, judge: str = "gpt52", write: bool = False) -> dict:
    """Correlate OUR per-dimension verdict score against the EXPERT per-dimension
    human score across the annotated (system, task) cells, per dimension and overall.

    Join key: (system, task_id, dimension). Human gold has ~3 replicate annotations
    per cell (averaged); tasks 1..50 are annotated, so only those cells join. The
    overall row pools every joined (cell × dim) pair. Spearman + Pearson, each with
    a seeded task-clustered bootstrap 95% CI.
    """
    ours = load_drb_verdicts(verdicts_dir)
    human = drb_human_by_cell()
    gold_dims = sorted({d for (_s, _t, d) in human})
    systems_judged = sorted({s for (s, _t, _d) in ours})

    per_dim: dict[str, dict] = {}
    pooled_o: list[float] = []
    pooled_h: list[float] = []
    pooled_clusters: list[str] = []
    cells: set[tuple[str, int]] = set()
    for dim in gold_dims:
        xs: list[float] = []
        ys: list[float] = []
        clusters: list[str] = []
        for (system, task, d), hv in sorted(human.items()):
            if d != dim:
                continue
            ov = ours.get((system, task, d))
            if ov is None:
                continue
            xs.append(ov)
            ys.append(hv)
            clusters.append(str(task))     # cluster on task (whole-task resample)
            cells.add((system, task))
        pooled_o.extend(xs)
        pooled_h.extend(ys)
        pooled_clusters.extend(clusters)
        per_dim[dim] = {
            "n": len(xs),
            "n_tasks": len(set(clusters)),
            "spearman": _corr_bootstrap(xs, ys, clusters, spearman),
            "pearson": _corr_bootstrap(xs, ys, clusters, pearson),
        }

    overall = {
        "n": len(pooled_o),
        "n_cells": len(cells),
        "n_tasks": len(set(pooled_clusters)),
        "spearman": _corr_bootstrap(pooled_o, pooled_h, pooled_clusters, spearman),
        "pearson": _corr_bootstrap(pooled_o, pooled_h, pooled_clusters, pearson),
    }

    expected_systems = sorted({str(r["meta"]["system"]) for r in load_gold("drb_race")})
    result = {
        "status": "scored",
        "judge": judge,
        "verdicts_dir": verdicts_dir,
        "join_key": "(system, task_id, dimension)",
        "dimensions": gold_dims,
        "systems_judged": systems_judged,
        "systems_expected": expected_systems,
        "systems_pending": [s for s in expected_systems if s not in systems_judged],
        "n_verdict_cells_loaded": len(ours),
        "n_joined_system_task_cells": len(cells),
        "per_dimension": per_dim,
        "overall": overall,
        "note": "our verdict per-dim score vs expert per-dim score across annotated "
                "(system,task) cells; human = mean of ~3 replicate annotations; tasks "
                "1..50 are annotated so only those join; PARTIAL until all 4 systems "
                "finish judging.",
    }
    if write:
        _atomic_write_drb_race(result)
    return result


def _print_drb_race(res: dict) -> None:
    print("=" * 78)
    print(f"DRB-RACE correlation — our GPT-5.2 panel ({res['judge']}) vs expert RACE labels")
    print("=" * 78)
    print(f"  verdicts_dir : {res['verdicts_dir']}")
    print(f"  join_key     : {res['join_key']}")
    print(f"  systems judged   : {res['systems_judged']}")
    print(f"  systems pending  : {res['systems_pending']}")
    print(f"  verdict cells loaded   : {res['n_verdict_cells_loaded']}")
    print(f"  joined (system,task) cells : {res['n_joined_system_task_cells']}")
    print("-" * 78)
    hdr = f"  {'dimension':22s} {'n':>4s} {'tasks':>5s} {'Spearman':>9s} {'(95% CI)':>17s} {'Pearson':>9s}"
    print(hdr)
    print("-" * 78)

    def _fmt(b: Optional[dict]) -> tuple[str, str]:
        if not b or b.get("point") is None:
            return ("   n/a", "")
        ci = b.get("ci95")
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else ""
        return (f"{b['point']:+.4f}", cis)

    for dim in res["dimensions"]:
        d = res["per_dimension"][dim]
        sp, spci = _fmt(d["spearman"])
        pe, _ = _fmt(d["pearson"])
        print(f"  {dim:22s} {d['n']:>4d} {d['n_tasks']:>5d} {sp:>9s} {spci:>17s} {pe:>9s}")
    print("-" * 78)
    o = res["overall"]
    sp, spci = _fmt(o["spearman"])
    pe, _ = _fmt(o["pearson"])
    print(f"  {'OVERALL (pooled)':22s} {o['n']:>4d} {o['n_tasks']:>5d} {sp:>9s} {spci:>17s} {pe:>9s}")
    print("-" * 78)
    if res["systems_pending"]:
        print(f"  CAVEAT: PARTIAL — {len(res['systems_pending'])} system(s) still being "
              f"judged: {res['systems_pending']}")
    print("=" * 78)


def _atomic_write_drb_race(res: dict) -> None:
    """Write the DRB-RACE result into canonical_numbers.json['judge_vs_human']
    ['drb_race_correlation'] (atomic tmp-then-replace). Corpus-safe: writes only
    under this analysis dir."""
    cn_path = f"{ANA}/canonical_numbers.json"
    cn = json.load(open(cn_path))
    jvh = cn.get("judge_vs_human")
    if not isinstance(jvh, dict):
        jvh = {}
    jvh.setdefault("drb_race_correlation", {})[res["judge"]] = res
    cn["judge_vs_human"] = jvh
    txt = json.dumps(cn, indent=1)
    tmp = cn_path + ".tmp"
    open(tmp, "w").write(txt)
    os.replace(tmp, cn_path)


# =============================================================================
# gold loading + verdict matching
# =============================================================================
def load_gold(source: str) -> list[dict]:
    """Load one human-label set as common-schema rows, keeping only rows that carry a
    usable numeric human grade (None = unscoreable class, e.g. ExpertQA 'Unsure')."""
    rows = [r for r in LOADERS[source]() if r["human_label"] is not None]
    return rows


def citation_count(text: str) -> int:
    """Distinct inline-citation indices in a claim (ExpertQA density instrument)."""
    seen: set[int] = set()
    for m in _CIT_RE.findall(text or ""):
        for d in re.findall(r"\d+", m):
            seen.add(int(d))
    return len(seen)


def load_family_verdicts(family: str, source: str, gold_rows: list[dict]) -> dict[str, dict]:
    """Return {item_id: {"score": float in 0-1, "judge": str}} for this (family, set).

    Reads our on-disk verdict stores read-only. The external human sets are not yet
    judged by our panel, so for them this currently returns {} (the join is keyed on
    item_id / the DRB-RACE system+task_id key and slots in unchanged once a panel run
    lands verdicts in the store). The ONE exception is the real on-disk MACHINE verdict
    that ships INSIDE the gold: DeepFactBench's ``agent_verdict``, surfaced as a built-in
    "agent" family so the pipeline is exercised on real data with no API call.
    """
    # Real, on-disk, no-API machine verdicts bundled with the gold.
    if family == "agent" and source == "deepfactbench":
        out: dict[str, dict] = {}
        verdict_map = {"supported": 1.0, "inconclusive": 0.5, "contradictory": 0.0}
        for r in gold_rows:
            av = r["meta"].get("agent_verdict")
            if av in verdict_map:
                out[r["item_id"]] = {"score": verdict_map[av], "judge": "deepfact_agent"}
        return out

    # Our panel verdict stores: consulted, but the external sets are not present in them
    # yet. (Kept deliberately simple + read-only; a later panel run that writes
    # {item_id}.json into the family's dir will be picked up here.)
    for d in FAMILY_VERDICT_DIRS.get(family, []):
        store = _REPO_ROOT / d / f"_human_{source}"
        if store.is_dir():
            out = {}
            for fp in sorted(store.glob("*.json")):
                try:
                    rec = json.loads(fp.read_text())
                except Exception:
                    continue
                iid = rec.get("item_id")
                sc = rec.get("score")
                if iid is not None and sc is not None:
                    out[iid] = {"score": float(sc), "judge": rec.get("judge", family)}
            if out:
                return out
    return {}


def join(gold_rows: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    """Inner-join gold rows to verdict scores on item_id, attaching cluster id + density."""
    joined = []
    for r in gold_rows:
        v = verdicts.get(r["item_id"])
        if v is None:
            continue
        joined.append({
            "item_id": r["item_id"],
            "human": float(r["human_label"]),
            "human_bin": 1 if float(r["human_label"]) >= 0.5 else 0,
            "score": float(v["score"]),
            "score_bin": 1 if float(v["score"]) >= 0.5 else 0,
            "judge": v["judge"],
            "cluster": CLUSTER_KEY[r["source"]](r),
            "citations": citation_count(r["text"]) if r["source"] == "expertqa" else None,
            "dimension": r["dimension"],
        })
    return joined


# =============================================================================
# endpoints
# =============================================================================
def agreement_block(joined: list[dict]) -> Optional[dict]:
    if len(joined) < 2:
        return None
    clusters = [j["cluster"] for j in joined]
    return {
        "n": len(joined),
        "n_clusters": len(set(clusters)),
        "human_pos_rate": round(float(np.mean([j["human_bin"] for j in joined])), 4),
        "macro_f1": cluster_bootstrap(joined, macro_f1, "human_bin", "score_bin", clusters),
        "point_biserial_r": cluster_bootstrap(joined, point_biserial, "human_bin", "score", clusters),
        "auc": cluster_bootstrap(joined, auc, "human_bin", "score", clusters),
    }


def care_conditioned_corr(
    family_joined: dict[str, list[dict]],
) -> Optional[dict]:
    """Per-family inter-judge correlation CONDITIONED on the human grade (the CARE input
    to N_eff). For each pair of judges in the family, correlate their verdict scores on
    items they BOTH graded, AFTER subtracting the per-human-grade mean (partialling out
    the human label). The residual correlation is the family dependence N_eff discounts.
    Returns None when fewer than two judges in the family scored a shared item set.
    """
    # collect per-judge score by item, plus the human grade per item
    by_judge: dict[str, dict[str, float]] = defaultdict(dict)
    human: dict[str, float] = {}
    for rows in family_joined.values():
        for j in rows:
            by_judge[j["judge"]][j["item_id"]] = j["score"]
            human[j["item_id"]] = j["human"]
    judges = sorted(by_judge)
    if len(judges) < 2:
        return {"status": "single_judge_in_family", "judges": judges}
    pair_corrs = {}
    for a_i in range(len(judges)):
        for b_i in range(a_i + 1, len(judges)):
            a, b = judges[a_i], judges[b_i]
            shared = sorted(set(by_judge[a]) & set(by_judge[b]))
            if len(shared) < 5:
                continue
            xa = np.array([by_judge[a][i] for i in shared])
            xb = np.array([by_judge[b][i] for i in shared])
            h = np.array([human[i] for i in shared])
            # residualise each judge's score on the human grade (group-mean centering)
            ra = xa - _group_mean(xa, h)
            rb = xb - _group_mean(xb, h)
            raw = _safe_corr(xa, xb)
            cond = _safe_corr(ra, rb)
            pair_corrs[f"{a}|{b}"] = {
                "n_shared": len(shared),
                "raw_corr": raw,
                "conditioned_corr": cond,
            }
    if not pair_corrs:
        return {"status": "no_judge_pair_with_shared_items", "judges": judges}
    conds = [v["conditioned_corr"] for v in pair_corrs.values() if v["conditioned_corr"] is not None]
    return {
        "judges": judges,
        "pairs": pair_corrs,
        "mean_conditioned_corr": round(float(np.mean(conds)), 4) if conds else None,
        "note": "conditioned_corr = residual inter-judge correlation after partialling "
                "out the human grade; >0 is the family dependence that lowers N_eff.",
    }


def _group_mean(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Return the per-group mean of x aligned to each element (group = human grade)."""
    out = np.empty_like(x, dtype=float)
    for gv in np.unique(g):
        m = g == gv
        out[m] = x[m].mean()
    return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if np.std(a) == 0 or np.std(b) == 0 or len(a) < 3:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def expertqa_density_residual(joined: list[dict]) -> Optional[dict]:
    """E4 hook: regress factual_accuracy verdict ~ citation_count + expert_factuality.
    The partial coefficient on citation_count (expert factuality held fixed) is the
    density-artefact estimate. OLS via normal equations; ZERO external deps.
    """
    fa = [j for j in joined if j["dimension"] == "factual_accuracy" and j["citations"] is not None]
    if len(fa) < 10:
        return {"status": "insufficient_factual_rows", "n": len(fa)}
    y = np.array([j["score"] for j in fa], dtype=float)          # judge factual verdict
    cit = np.array([j["citations"] for j in fa], dtype=float)    # citation density
    expert = np.array([j["human"] for j in fa], dtype=float)     # expert factuality grade
    X = np.column_stack([np.ones(len(y)), cit, expert])
    # guard collinearity
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return {"status": "rank_deficient", "n": len(y)}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return {"status": "insufficient_dof", "n": len(y)}
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "n": len(y),
        "beta_citation_density": round(float(beta[1]), 5),
        "se_citation_density": round(float(se[1]), 5),
        "t_citation_density": round(float(beta[1] / se[1]), 3) if se[1] > 0 else None,
        "beta_expert_factuality": round(float(beta[2]), 5),
        "interpretation": "beta_citation_density > 0 (with t > ~2) => the family rewards "
                          "citation markup independent of expert-verified factuality (the "
                          "density artefact). ~0 => density-insensitive on this instrument.",
    }


# =============================================================================
# main builder
# =============================================================================
def build(write: bool = True, families: Optional[list[str]] = None) -> dict:
    families = families or PANEL_FAMILIES
    out: dict = {
        "_note": "build_judge_vs_human — human-anchored judge agreement over the public "
                 "human-label sets (gold via deep_research/benchmarks/gold_loaders.py). "
                 "Per (judge family × human set): Macro-F1 / point-biserial / AUC with "
                 "seeded cluster-bootstrap CIs; per-family inter-judge correlation "
                 "CONDITIONED on the human grade (the CARE confounder feeding N_eff); and "
                 "the ExpertQA density-residual regression (E4). Sets not yet panel-judged "
                 "appear with status='awaiting_panel' and slot in unchanged once verdicts "
                 "land. The DeepFactBench 'agent' row is a REAL on-disk machine verdict "
                 "(no API) so the pipeline is exercised end-to-end on real data.",
        "seed": SEED,
        "human_sets": {},
        "by_family": {},
        "care_conditioned": {},
        "expertqa_density_residual": {},
        "drb_race_plugin": {
            "status": "awaiting_panel",
            "join_key": "meta[system]+meta[task_id]",
            "n_gold_rows": None,
            "dimensions": sorted({r["dimension"] for r in load_gold("drb_race")}),
            "note": "our 9-dim panel over the 400 released DRB reports plugs in here: "
                    "match verdict.item_id to gold item_id (system+task_id+dimension); "
                    "agreement + CARE tables then populate with no code change.",
        },
    }

    # gold sizes (usable-grade rows) per set — recorded once.
    gold_cache: dict[str, list[dict]] = {}
    for s in HUMAN_SETS:
        g = load_gold(s)
        gold_cache[s] = g
        out["human_sets"][s] = {
            "n_gold_rows_scoreable": len(g),
            "dimensions": sorted({r["dimension"] for r in g}),
            "cluster_key": _cluster_key_name(s),
        }
    out["drb_race_plugin"]["n_gold_rows"] = len(gold_cache["drb_race"])

    # The families we actually evaluate now: the panel families (which currently have no
    # external verdicts on disk) PLUS the built-in 'agent' family on DeepFactBench so a
    # real agreement number is produced. Per-set per-family CARE is collected as we go.
    eval_families = list(families) + ["agent"]
    care_by_set: dict[str, dict[str, list[dict]]] = defaultdict(dict)  # set -> family -> joined

    for fam in eval_families:
        out["by_family"].setdefault(fam, {})
        for s in HUMAN_SETS:
            gold_rows = gold_cache[s]
            verdicts = load_family_verdicts(fam, s, gold_rows)
            if not verdicts:
                out["by_family"][fam][s] = {"status": "awaiting_panel"}
                continue
            joined = join(gold_rows, verdicts)
            if s == "draco_full":
                # B2 (world-class review) fix: the draco_full 'human_label' is the criterion's
                # EXPERT TARGET POLARITY (a constant per criterion = sign of its weight), NOT a
                # per-report human grade. The judge is shown only the criterion text and asked
                # to classify its polarity — which IS the label. So this macro-F1 measures
                # criterion-polarity DETECTION (a strong judge reads polarity off the wording →
                # 0.995; a weak 7B fails → 0.479), NOT judge-vs-human agreement on report quality.
                # Excluded from the human-anchor agreement; disclosed as a flagged sanity check.
                blk = agreement_block(joined)
                out["by_family"][fam][s] = {
                    "status": "excluded_label_leakage",
                    "NOT_judge_vs_human_agreement": True,
                    "reason": ("human_label == criterion target-polarity (constant per criterion), "
                               "not a per-report human grade; the score is polarity detection, "
                               "not report-grading agreement."),
                    "polarity_detection_macro_f1": blk.get("macro_f1"),
                    "n": blk.get("n"),
                }
                continue
            blk = agreement_block(joined)
            rec = {"status": "scored", "agreement": blk}
            if s == "expertqa":
                rec["density_residual"] = expertqa_density_residual(joined)
                out["expertqa_density_residual"][fam] = rec["density_residual"]
            out["by_family"][fam][s] = rec
            care_by_set[s][fam] = joined

    # CARE: per-family inter-judge conditioned correlation. With real multi-judge family
    # verdicts this populates per set; with the present single-judge-per-family stores it
    # records the structural status so N_eff knows the wiring is live.
    for s in HUMAN_SETS:
        # group the per-family joined rows by FAMILY then by judge-within-family
        fam_groups: dict[str, dict[str, list[dict]]] = defaultdict(dict)
        for fam, joined in care_by_set.get(s, {}).items():
            # bucket this family's rows by the concrete judge id inside them
            by_j: dict[str, list[dict]] = defaultdict(list)
            for j in joined:
                by_j[j["judge"]].append(j)
            fam_groups[fam] = by_j
        set_out = {}
        for fam, by_j in fam_groups.items():
            set_out[fam] = care_conditioned_corr(by_j)
        if set_out:
            out["care_conditioned"][s] = set_out

    if write:
        _atomic_write_canonical(out)
    return out


def _cluster_key_name(s: str) -> str:
    return {
        "drb_race": "meta.task_id",
        "healthbench": "meta.pair_key",
        "expertqa": "meta.question",
        "deepfactbench": "meta.report_id",
        "draco_full": "meta.task_id",
    }[s]


def _atomic_write_canonical(out: dict) -> None:
    cn_path = f"{ANA}/canonical_numbers.json"
    cn = json.load(open(cn_path))
    cn["judge_vs_human"] = out
    txt = json.dumps(cn, indent=1)
    tmp = cn_path + ".tmp"
    open(tmp, "w").write(txt)
    os.replace(tmp, cn_path)


# =============================================================================
# self-test (real gold + small SYNTHETIC verdict set; no API)
# =============================================================================
def _synthetic_verdicts(gold_rows: list[dict], noise: float, flip: float, seed: int) -> dict[str, dict]:
    """Make a deterministic synthetic verdict store: judge score = human grade nudged by
    Gaussian noise and a flip rate, clipped to [0,1]. Two synthetic judges per family so
    the CARE conditioned-correlation path is exercised on real gold structure."""
    g = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for r in gold_rows:
        h = float(r["human_label"])
        s = h + g.normal(0, noise)
        if g.random() < flip:
            s = 1.0 - h + g.normal(0, noise)
        out[r["item_id"]] = {"score": float(np.clip(s, 0, 1)), "judge": f"syn_{seed % 2}"}
    return out


def _print_schema_table(out: dict) -> None:
    print("\nOUTPUT TABLE SCHEMA (canonical_numbers.json['judge_vs_human'])")
    print("-" * 78)
    print(f"  seed                     : {out['seed']}")
    print("  human_sets[set]          : n_gold_rows_scoreable, dimensions[], cluster_key")
    print("  by_family[fam][set]      : status | {agreement:{n,n_clusters,human_pos_rate,")
    print("                             macro_f1{point,ci95,n_clusters}, point_biserial_r{...},")
    print("                             auc{...}}, [density_residual]}")
    print("  care_conditioned[set][fam]: {judges[], pairs{a|b:{n_shared,raw_corr,")
    print("                             conditioned_corr}}, mean_conditioned_corr}")
    print("  expertqa_density_residual[fam]: {n, beta_citation_density, se, t,")
    print("                             beta_expert_factuality, interpretation}")
    print("  drb_race_plugin          : {status, join_key, n_gold_rows, dimensions[]}")
    print("-" * 78)
    print("\n  human_sets (real on-disk gold):")
    for s, v in out["human_sets"].items():
        print(f"    {s:14s} n={v['n_gold_rows_scoreable']:>7d}  cluster={v['cluster_key']:14s}  "
              f"dims={v['dimensions']}")


def _selftest() -> int:
    print("=" * 78)
    print("build_judge_vs_human self-test (real human gold + synthetic verdicts, NO API)")
    print("=" * 78)
    ok = True

    # 1) base build (panel families -> awaiting_panel; agent family on deepfactbench REAL)
    out = build(write=False)

    # the real on-disk agent verdict must produce a scored deepfactbench agreement block
    dfb_agent = out["by_family"].get("agent", {}).get("deepfactbench", {})
    cond1 = dfb_agent.get("status") == "scored" and dfb_agent.get("agreement") is not None
    ok &= cond1
    print(f"\n[REAL on-disk verdict] deepfactbench agent agreement scored: "
          f"{'OK' if cond1 else 'FAIL'}")
    if cond1:
        ag = dfb_agent["agreement"]
        print(f"    n={ag['n']} clusters={ag['n_clusters']} human_pos_rate={ag['human_pos_rate']}")
        print(f"    macro_f1={ag['macro_f1']['point']} CI={ag['macro_f1']['ci95']}")
        print(f"    AUC    ={ag['auc']['point']} CI={ag['auc']['ci95']}")
        print(f"    point_biserial_r={ag['point_biserial_r']['point']} CI={ag['point_biserial_r']['ci95']}")
        # sanity vs the raw on-disk agreement (471/621)
        gold = load_gold("deepfactbench")
        raw_match = np.mean([
            (g["label_class"] == g["meta"]["agent_verdict"]) for g in gold
            if g["meta"].get("agent_verdict") is not None
        ])
        print(f"    [cross-check] raw agent-vs-human class agreement = {raw_match:.4f} (471/621)")

    # 2) synthetic verdicts on the SMALLEST real sets, two synthetic judges per family,
    #    to exercise agreement + CARE conditioned-correlation + (ExpertQA) density hook.
    print("\n[SYNTHETIC verdicts] exercising agreement / CARE / density on real gold structure")
    for s in ["deepfactbench", "draco_full"]:
        gold = load_gold(s)
        # subsample large sets deterministically for a fast self-test
        if len(gold) > 1500:
            gold = sorted(gold, key=lambda r: r["item_id"])[:1500]
        for fam, (noise, flip) in {"synA": (0.18, 0.05), "synB": (0.35, 0.20)}.items():
            # two synthetic judges in the family (seeds 10/11) for the CARE path
            vmap = {}
            for sd in (10, 11):
                vmap.update(_synthetic_verdicts(gold, noise, flip, seed=sd + _stable_small_int(fam, 7)))
            joined = join(gold, vmap)
            blk = agreement_block(joined)
            assert blk is not None and blk["auc"] is not None, f"agreement failed for {s}/{fam}"
            # CARE on two synthetic judges
            by_j = defaultdict(list)
            # regenerate per-judge so two distinct judge ids exist
            for sd, jid in ((10, "j10"), (11, "j11")):
                vj = _synthetic_verdicts(gold, noise, flip, seed=sd + _stable_small_int(fam, 7))
                for r in gold:
                    v = vj.get(r["item_id"])
                    if v:
                        by_j[jid].append({
                            "item_id": r["item_id"], "judge": jid,
                            "score": v["score"], "human": float(r["human_label"]),
                        })
            care = care_conditioned_corr(by_j)
            mc = care.get("mean_conditioned_corr") if isinstance(care, dict) else None
            print(f"    {s:13s} {fam:5s} noise={noise:<4} flip={flip:<4} -> "
                  f"AUC={blk['auc']['point']} F1={blk['macro_f1']['point']} "
                  f"r={blk['point_biserial_r']['point']}  CARE mean_cond_corr={mc}")
            # lower-noise family should agree better (monotonicity smoke test)
            if fam == "synA":
                auc_a = blk["auc"]["point"]
            else:
                ok &= (auc_a >= blk["auc"]["point"] - 0.05)

    # 3) ExpertQA density-residual hook on a deterministic subsample (real grades + real
    #    citation counts; synthetic verdict so no API). Verifies the regression runs and
    #    recovers an injected positive density coefficient.
    print("\n[E4 density hook] ExpertQA factual verdict ~ citation_count + expert_factuality")
    eqa = [r for r in load_gold("expertqa") if r["dimension"] == "factual_accuracy"]
    eqa = sorted(eqa, key=lambda r: r["item_id"])[:2000]
    g = np.random.default_rng(SEED)
    # inject a KNOWN positive density effect: verdict = expert + 0.05*citations + noise
    inj = {}
    for r in eqa:
        c = citation_count(r["text"])
        s = float(r["human_label"]) + 0.05 * c + g.normal(0, 0.1)
        inj[r["item_id"]] = {"score": float(np.clip(s, 0, 1)), "judge": "syn_density"}
    joined = join(eqa, inj)
    dr = expertqa_density_residual(joined)
    print(f"    n={dr.get('n')} beta_citation_density={dr.get('beta_citation_density')} "
          f"(t={dr.get('t_citation_density')}) beta_expert={dr.get('beta_expert_factuality')}")
    # control FIRST so we can compare against it: verdict = expert only (no density)
    inj0 = {r["item_id"]: {"score": float(np.clip(float(r["human_label"]) + g.normal(0, 0.1), 0, 1)),
                           "judge": "syn_nodensity"} for r in eqa}
    dr0 = expertqa_density_residual(join(eqa, inj0))
    print(f"    [control no-density] beta_citation_density={dr0.get('beta_citation_density')} "
          f"(t={dr0.get('t_citation_density')}) -> should be near 0")
    # Success criterion is the SIGN + SIGNIFICANCE of the partial coefficient, not its raw
    # magnitude: [0,1]-clipping attenuates beta and the citation/expert correlation absorbs
    # part of the injected effect, so an exact-magnitude check is wrong. The valid test is
    # that the injected case is a clearly significant positive (t > 2) while the control is
    # not (|t| < 2) — i.e. the instrument DISCRIMINATES density from no-density.
    t_inj = dr.get("t_citation_density")
    t_ctl = dr0.get("t_citation_density")
    cond3 = (
        isinstance(dr.get("beta_citation_density"), float) and dr["beta_citation_density"] > 0
        and t_inj is not None and t_inj > 2.0
        and (t_ctl is None or abs(t_ctl) < 2.0)
    )
    ok &= cond3
    print(f"    discriminates injected density (t={t_inj}) from control (t={t_ctl}): "
          f"{'OK' if cond3 else 'FAIL'}")

    # 4) drb_race plug-in advertised, gold rows counted, dims present
    plug = out["drb_race_plugin"]
    cond4 = plug["n_gold_rows"] and plug["dimensions"] and plug["status"] == "awaiting_panel"
    ok &= bool(cond4)
    print(f"\n[DRB-RACE plug-in] n_gold_rows={plug['n_gold_rows']} dims={plug['dimensions']} "
          f"status={plug['status']}: {'OK' if cond4 else 'FAIL'}")

    # 5) every panel family present in by_family with the external sets awaiting panel
    for fam in PANEL_FAMILIES:
        present = fam in out["by_family"]
        ok &= present
    print(f"\n[families wired] {sorted(out['by_family'])}")

    _print_schema_table(out)
    print("\n" + "=" * 78)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    print("=" * 78)
    return 0 if ok else 1


def _arg(flag: str, default: Optional[str] = None) -> Optional[str]:
    """Tiny dependency-free CLI value reader: --flag value  or  --flag=value."""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    if "--run-drb-race" in sys.argv:
        # CLI: correlate our DRB panel verdicts vs the expert RACE gold (read-only on
        # verdicts; writes canonical_numbers only with --write). NO API calls.
        vdir = _arg("--verdicts-dir", "results/drbrace/judge_gpt52")
        judge = _arg("--judge", "gpt52")
        do_write = "--write" in sys.argv
        res = run_drb_race(vdir, judge=judge, write=do_write)
        _print_drb_race(res)
        if do_write:
            print("\nWROTE canonical_numbers.json"
                  f"['judge_vs_human']['drb_race_correlation']['{judge}']")
        sys.exit(0 if res["overall"]["n"] > 0 else 1)
    elif "--write" in sys.argv:
        res = build(write=True)
        print("WROTE canonical_numbers.json['judge_vs_human'] "
              f"(human_sets={list(res['human_sets'])}, families={list(res['by_family'])})")
    else:
        sys.exit(_selftest())
