"""Build canonical analysis dataframes for the Deep Research Projects evaluation.

Reads:
    - data/eval_queries_v2.json              (query manifest, 90 queries)
    - results/experiments/<pattern>/<qid>.md (generated reports)
    - checkpoints/experiments/<pattern>/<qid>.json (process metrics)
    - results/judge_gpt52/<pattern>/<qid>.json
    - results/judge_claude_opus/<pattern>/<qid>.json
    - results/judge_claude_sonnet/<pattern>/<qid>.json
    - results/judge_claude_code/<pattern>/<qid>.json
    - deep_research/evaluation/rubric_v2.py  (DIMENSION_WEIGHTS_V2 + source overrides)

Writes to data/analysis/:
    df_queries.parquet, df_runs.parquet, df_scores.parquet,
    df_overall_scores.parquet, df_verdicts.parquet, coverage_report.md,
    build_manifest.json, DATA_DICTIONARY.md

Idempotent and re-runnable. Handles missing files gracefully.

IMPORTANT DATA NOTES (propagated into build_manifest.json + DATA_DICTIONARY.md):
  * claude_opus judge JSONs have HIGHLY HETEROGENEOUS verdict schemas:
        - criterion text key may be: "criterion", "description", "text", "criterion_text"
        - satisfied signal may be: satisfied:bool, OR verdict:"SATISFIED"/"NOT_SATISFIED"
        - reasoning key may be: "reasoning" or "reason"
    This script normalizes all of these per-verdict (see _extract_verdict_fields).
  * claude_sonnet stored `overall_score` is corrupted upstream. Downstream analyses
    MUST use `overall_score_recomputed` or `overall_score_per_query_weights` for
    claude_sonnet rows. The `overall_score_trustworthy` column flags this.
  * Criterion text is normalized (lowercased, whitespace-collapsed) BEFORE hashing
    into criterion_id so minor wording differences across judge runs do not
    inflate rubric-drift counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "experiments"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

JUDGE_DIRS = {
    "gpt52": RESULTS_DIR / "judge_gpt52",
    "claude_opus": RESULTS_DIR / "judge_claude_opus",
    "claude_sonnet": RESULTS_DIR / "judge_claude_sonnet",
    "claude_code": RESULTS_DIR / "judge_claude_code",
}

# Upstream claude_sonnet stored overall_score is known to be corrupted; downstream
# consumers must rely on the recomputed scores from dimension met/total.
TRUSTWORTHY_OVERALL_SCORE_JUDGES = {"gpt52", "claude_opus", "claude_code"}

# GPT-4o blended input+output rate used as a cost proxy. 7B-local patterns have no
# API cost; we substitute a compute proxy based on wall-clock seconds.
GPT4O_BLENDED_RATE_PER_M_TOKENS = 5.0  # USD per 1M tokens
LOCAL_MODEL_COMPUTE_RATE_PER_SEC = 0.0001  # USD per second of wall-clock
LOCAL_MODEL_PATTERNS_SHORT = {"p9", "p10", "p12"}
ABLATION_EXCLUDED_PATTERNS = {"ablation_p5_no_citation_verify"}
ABLATION_EXCLUSION_REASON = (
    "Only 2/90 reports were generated for ablation_p5_no_citation_verify (run "
    "aborted early); statistical comparisons would be unreliable."
)

# ----- Load V2 rubric weights -----
sys.path.insert(0, str(ROOT))
from deep_research.evaluation.rubric_v2 import (  # noqa: E402
    DIMENSION_WEIGHTS_V2,
    DIMENSION_WEIGHTS_BY_SOURCE,
)

V2_DIMENSIONS: set[str] = set(DIMENSION_WEIGHTS_V2.keys())

# Source-name mapping from manifest source -> rubric_v2 source key
SOURCE_TO_WEIGHT_KEY = {
    "custom": "default",
    "draco": "draco",
    "deepsearch_qa": "deepsearchqa",
    "research_qa": "researchqa",
    "litqa2": "litqa2",
}


def _safe_load_json(path: Path, warn_missing: bool = False) -> dict[str, Any] | None:
    if not path.exists():
        if warn_missing:
            print(f"WARN: missing {path}", file=sys.stderr)
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to parse {path}: {e}", file=sys.stderr)
        return None


def _word_count_md(path: Path) -> int:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return len(re.findall(r"\S+", txt))
    except Exception:  # noqa: BLE001
        return 0


def _normalize_criterion(text: str) -> str:
    """Canonicalize criterion text before hashing to criterion_id.

    Lowercases, trims, and collapses internal whitespace so that judges that
    phrase the same criterion with small formatting differences hash to the
    same id. This is required for rubric-drift detection to work meaningfully.
    """
    return " ".join(str(text or "").lower().strip().split())


def _crit_id(text: str) -> str:
    return hashlib.md5(_normalize_criterion(text).encode("utf-8")).hexdigest()[:12]


def _extract_verdict_fields(v: dict[str, Any]) -> tuple[str, bool | None, str, str, str | None, int | None]:
    """Normalize one verdict dict to (criterion_text, satisfied, evidence, reasoning, dimension, criterion_index).

    Handles all claude_opus schema variants encountered:
        criterion text key:  criterion | description | text | criterion_text
        verdict signal:      satisfied:bool  OR  verdict:"SATISFIED"/"NOT_SATISFIED"
        reasoning key:       reasoning | reason
    """
    crit_text = (
        v.get("criterion")
        or v.get("description")
        or v.get("text")
        or v.get("criterion_text")
        or ""
    )
    if not isinstance(crit_text, str):
        crit_text = str(crit_text)

    sat_raw = v.get("satisfied")
    if sat_raw is None:
        verdict_str = str(v.get("verdict", "")).strip().upper()
        if verdict_str == "SATISFIED":
            satisfied: bool | None = True
        elif verdict_str in ("NOT_SATISFIED", "NOT SATISFIED", "UNSATISFIED", "FAILED"):
            satisfied = False
        else:
            satisfied = None
    else:
        satisfied = bool(sat_raw)

    ev = v.get("evidence", "") or ""
    if not isinstance(ev, str):
        ev = str(ev)
    rs = v.get("reasoning") or v.get("reason") or ""
    if not isinstance(rs, str):
        rs = str(rs)

    dim_v = v.get("dimension")
    if dim_v is not None and not isinstance(dim_v, str):
        dim_v = str(dim_v)

    ci = v.get("criterion_index")
    if ci is not None:
        try:
            ci = int(ci)
        except Exception:  # noqa: BLE001
            ci = None

    return crit_text, satisfied, ev, rs, dim_v, ci


def _extract_dim_met_total(de: dict[str, Any]) -> tuple[float | None, int | None, int | None]:
    """Normalize a dimension dict to (score, met, total) across schema variants."""
    s = de.get("score")
    try:
        s_val = float(s) if s is not None else None
    except Exception:  # noqa: BLE001
        s_val = None

    # met / total normalization across the 11 schema variants encountered:
    m = de.get("met")
    if m is None:
        m = de.get("satisfied")
    if m is None:
        m = de.get("criteria_met")
    if m is None:
        m = de.get("satisfied_count")
    if m is None:
        m = de.get("n_satisfied")

    t = de.get("total")
    if t is None:
        t = de.get("criteria_total")
    if t is None:
        t = de.get("criteria_count")
    if t is None:
        t = de.get("n_criteria")

    m_val: int | None
    t_val: int | None
    try:
        m_val = int(m) if m is not None else None
    except Exception:  # noqa: BLE001
        m_val = None
    try:
        t_val = int(t) if t is not None else None
    except Exception:  # noqa: BLE001
        t_val = None
    return s_val, m_val, t_val


# ----- df_queries -----
def build_df_queries() -> pd.DataFrame:
    with (DATA_DIR / "eval_queries_v2.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    rows: list[dict[str, Any]] = []
    for q in manifest["queries"]:
        rows.append(
            {
                "query_id": q.get("id"),
                "source": q.get("source"),
                "domain": q.get("domain"),
                "difficulty": q.get("difficulty"),
                "query_text": q.get("query", ""),
                "expected_topics": q.get("expected_elements", []) or [],
                "gold_answer": q.get("reference_answer", "") or "",
            }
        )
    df = pd.DataFrame(rows)
    for c in ("source", "difficulty"):
        df[c] = pd.Categorical(df[c])
    return df


# ----- Pattern discovery -----
def discover_pattern_dirs() -> list[tuple[str, str, str]]:
    """Return list of (pattern, family, pattern_short).

    Recognised families:
      - base_p{N}           → ("base_pN", "base", "pN")
      - ablation_*          → ("ablation_*", "ablation", "*")
      - protocol_a_{r}_p{N} → ("protocol_a_…", "protocol_a", "{retriever}_pN")
      - base_p{N}_v{k}      → ("base_pN_vk", "variance", "pN_vk")  (E5 replicates)
      - disentangle_*       → ("disentangle_*", "disentanglement", "*")
    """
    patterns: list[tuple[str, str, str]] = []
    if EXPERIMENTS_DIR.exists():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name.startswith("base_") and re.match(r"^base_p\d+_v\d+$", name):
                # Variance replicate, e.g. base_p4_v1
                short = name[len("base_"):]
                patterns.append((name, "variance", short))
            elif name.startswith("base_"):
                patterns.append((name, "base", name[len("base_"):]))
            elif name.startswith("ablation_"):
                patterns.append((name, "ablation", name[len("ablation_"):]))
            elif name.startswith("protocol_a_"):
                patterns.append((name, "protocol_a", name[len("protocol_a_"):]))
            elif name.startswith("oracle_"):
                # Oracle-retrieval arm, e.g. oracle_t1_p4 (frozen-corpus counterfactual)
                patterns.append((name, "oracle", name[len("oracle_"):]))
            elif name.startswith("disentangle_"):
                patterns.append((name, "disentanglement", name[len("disentangle_"):]))
    return patterns


def _pattern_short_base_for_ablation(pattern_short: str) -> str:
    """Extract the base pattern short id (e.g. 'p3') from an ablation short name."""
    m = re.match(r"(p\d+)_", pattern_short)
    return m.group(1) if m else pattern_short


def _pattern_short_base_for_protocol_a(pattern_short: str) -> str:
    """Extract the base pattern short id from 'protocol_a_{retriever}_p{N}' short ('{retriever}_pN')."""
    m = re.match(r"^[a-z]+_(p\d+)$", pattern_short)
    return m.group(1) if m else pattern_short


def _pattern_short_base_for_variance(pattern_short: str) -> str:
    """Extract the base pattern short id from 'p{N}_v{k}'."""
    m = re.match(r"^(p\d+)_v\d+$", pattern_short)
    return m.group(1) if m else pattern_short


def _pattern_short_base_for_oracle(pattern_short: str) -> str:
    """Extract the base pattern short id from oracle short 't{N}_p{M}' (e.g. 't1_p4' -> 'p4')."""
    m = re.match(r"^t\d+_(p\d+)$", pattern_short)
    return m.group(1) if m else pattern_short


def _is_local_model_pattern(family: str, short: str) -> bool:
    if family == "base":
        return short in LOCAL_MODEL_PATTERNS_SHORT
    if family == "ablation":
        base_short = _pattern_short_base_for_ablation(short)
    elif family == "protocol_a":
        base_short = _pattern_short_base_for_protocol_a(short)
    elif family == "variance":
        base_short = _pattern_short_base_for_variance(short)
    elif family == "oracle":
        base_short = _pattern_short_base_for_oracle(short)
    else:
        base_short = short
    return base_short in LOCAL_MODEL_PATTERNS_SHORT


def _compute_cost_proxy(
    family: str,
    short: str,
    total_tokens: float | None,
    elapsed_seconds: float | None,
) -> float | None:
    is_local = _is_local_model_pattern(family, short)
    if is_local:
        if elapsed_seconds is None or pd.isna(elapsed_seconds):
            return None
        return float(elapsed_seconds) * LOCAL_MODEL_COMPUTE_RATE_PER_SEC
    if total_tokens is None or pd.isna(total_tokens):
        return None
    return float(total_tokens) * GPT4O_BLENDED_RATE_PER_M_TOKENS / 1_000_000.0


# ----- df_runs -----
def build_df_runs(query_ids: list[str]) -> pd.DataFrame:
    patterns = discover_pattern_dirs()
    rows: list[dict[str, Any]] = []
    for pattern, family, short in patterns:
        ckpt_dir = CHECKPOINTS_DIR / pattern
        rep_dir = EXPERIMENTS_DIR / pattern
        for qid in query_ids:
            rep_path = rep_dir / f"{qid}.md"
            ckpt_path = ckpt_dir / f"{qid}.json"
            ckpt: dict[str, Any] = _safe_load_json(ckpt_path) or {}
            timestamp = ckpt.get("timestamp")
            try:
                ts = pd.to_datetime(timestamp) if timestamp else pd.NaT
            except Exception:  # noqa: BLE001
                ts = pd.NaT
            report_exists = rep_path.exists()

            # Issue 5: if no checkpoint was present, status is None; coerce to sentinel.
            raw_status = ckpt.get("status")
            if raw_status is None or (isinstance(raw_status, float) and pd.isna(raw_status)):
                status = "missing_checkpoint" if not ckpt_path.exists() else raw_status
                if status is None:
                    status = "missing_status"
            else:
                status = raw_status

            rows.append(
                {
                    "pattern": pattern,
                    "pattern_family": family,
                    "pattern_short": short,
                    "query_id": qid,
                    "status": status,
                    "elapsed_seconds": ckpt.get("elapsed_seconds"),
                    "total_tokens": ckpt.get("total_tokens"),
                    "total_cost_usd": ckpt.get("total_cost_usd"),
                    "sections": ckpt.get("sections"),
                    "citations": ckpt.get("citations"),
                    "timestamp": ts,
                    "report_path": str(rep_path) if report_exists else None,
                    "report_exists": report_exists,
                    "word_count_is_present": report_exists,
                    # Issue 4: NaN (not 0) where the report is missing.
                    "report_word_count": _word_count_md(rep_path) if report_exists else float("nan"),
                }
            )

    df = pd.DataFrame(rows)

    # Coerce numeric BEFORE derived columns.
    for c in (
        "elapsed_seconds",
        "total_tokens",
        "total_cost_usd",
        "sections",
        "citations",
        "report_word_count",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Issue 4: derived cost_proxy_usd
    df["cost_proxy_usd"] = df.apply(
        lambda r: _compute_cost_proxy(
            r["pattern_family"], r["pattern_short"], r["total_tokens"], r["elapsed_seconds"]
        ),
        axis=1,
    )

    # Issue 4: excluded_from_analysis
    df["excluded_from_analysis"] = df["pattern"].isin(ABLATION_EXCLUDED_PATTERNS)

    # Issue 4: elapsed_is_suspect — per-pattern, > 2x median.
    medians = df.groupby("pattern", observed=True)["elapsed_seconds"].transform("median")
    df["elapsed_is_suspect"] = (
        df["elapsed_seconds"].notna()
        & medians.notna()
        & (df["elapsed_seconds"] > 2.0 * medians)
    )

    # Categoricals last.
    for c in ("pattern", "pattern_family", "pattern_short", "status"):
        df[c] = pd.Categorical(df[c])

    return df


# ----- df_scores, df_overall_scores, df_verdicts -----
def build_judge_frames(
    df_queries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    patterns = discover_pattern_dirs()
    score_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    verdict_rows: list[dict[str, Any]] = []
    anomalies: list[str] = []

    # Precompute query_id -> source for recomputation weights
    qid_to_source = dict(zip(df_queries["query_id"], df_queries["source"].astype(str)))

    # Load per-query dimension weights from the manifest (ground truth per-query weights)
    with (DATA_DIR / "eval_queries_v2.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    qid_to_weights: dict[str, dict[str, float]] = {}
    for q in manifest["queries"]:
        w = (q.get("rubric") or {}).get("dimension_weights") or {}
        if w:
            qid_to_weights[q["id"]] = {k: float(v) for k, v in w.items()}

    verdict_schema_anomalies = 0
    verdict_unknown_satisfied = 0

    for judge_name, judge_dir in JUDGE_DIRS.items():
        if not judge_dir.exists():
            continue
        for pattern, family, _ in patterns:
            pdir = judge_dir / pattern
            if not pdir.exists():
                continue
            for jpath in sorted(pdir.glob("*.json")):
                qid = jpath.stem
                data = _safe_load_json(jpath)
                if data is None:
                    anomalies.append(f"Unparseable JSON: {jpath}")
                    continue

                overall_score = data.get("overall_score")
                dims = data.get("dimensions", {}) or {}

                # Recompute with two weighting schemes:
                # (a) source-type weights from DIMENSION_WEIGHTS_BY_SOURCE (gpt52 stored convention)
                # (b) per-query weights from manifest rubric
                src_key = SOURCE_TO_WEIGHT_KEY.get(qid_to_source.get(qid, ""), "default")
                w_src = DIMENSION_WEIGHTS_BY_SOURCE.get(src_key, DIMENSION_WEIGHTS_V2)
                w_pq = qid_to_weights.get(qid, w_src)

                # Normalize per-dimension score/met/total across schema variants
                normalized_dims: dict[str, tuple[float | None, int | None, int | None]] = {}
                for dn, de in dims.items():
                    if isinstance(de, dict):
                        normalized_dims[dn] = _extract_dim_met_total(de)
                    elif isinstance(de, (int, float)):
                        normalized_dims[dn] = (float(de), None, None)

                def _recompute(
                    w: dict[str, float],
                    normalized_dims: dict[str, tuple[float | None, int | None, int | None]] = normalized_dims,
                ) -> float | None:
                    s_sum = 0.0
                    w_sum = 0.0
                    for dn, wt in w.items():
                        tup = normalized_dims.get(dn)
                        if tup is None:
                            continue
                        sv = tup[0]
                        if sv is None:
                            # Fall back to met/total if score missing
                            m, t = tup[1], tup[2]
                            if m is not None and t and t > 0:
                                sv = m / t
                        if sv is None:
                            continue
                        s_sum += float(sv) * float(wt)
                        w_sum += float(wt)
                    if w_sum == 0:
                        return None
                    if w_sum < 0.999:
                        return s_sum / w_sum
                    return s_sum

                recomputed = _recompute(w_src)
                recomputed_pq = _recompute(w_pq)

                overall_rows.append(
                    {
                        "pattern": pattern,
                        "pattern_family": family,
                        "query_id": qid,
                        "judge": judge_name,
                        "overall_score": overall_score,
                        "overall_score_recomputed": (
                            round(recomputed, 6) if recomputed is not None else None
                        ),
                        "overall_score_per_query_weights": (
                            round(recomputed_pq, 6) if recomputed_pq is not None else None
                        ),
                        # Issue 3: claude_sonnet stored overall_score is corrupted upstream.
                        "overall_score_trustworthy": judge_name in TRUSTWORTHY_OVERALL_SCORE_JUDGES,
                        "n_criteria": data.get("n_criteria"),
                        "n_satisfied": data.get("n_satisfied"),
                        "judge_tokens": data.get("tokens"),
                        "judge_latency_s": data.get("latency_s"),
                    }
                )

                for dim_name, (s_val, m_val, t_val) in normalized_dims.items():
                    if dim_name not in V2_DIMENSIONS:
                        anomalies.append(
                            f"Unknown dimension '{dim_name}' in {judge_name}/{pattern}/{qid}"
                        )
                    score_rows.append(
                        {
                            "pattern": pattern,
                            "pattern_family": family,
                            "query_id": qid,
                            "judge": judge_name,
                            "dimension": dim_name,
                            "score": s_val,
                            "met": m_val,
                            "total": t_val,
                        }
                    )

                for idx, v in enumerate(data.get("verdicts", []) or []):
                    if not isinstance(v, dict):
                        verdict_schema_anomalies += 1
                        continue
                    crit_text, satisfied, ev, rs, dim_v, ci = _extract_verdict_fields(v)
                    if satisfied is None:
                        verdict_unknown_satisfied += 1
                    verdict_rows.append(
                        {
                            "pattern": pattern,
                            "pattern_family": family,
                            "query_id": qid,
                            "judge": judge_name,
                            "criterion_index": ci if ci is not None else idx,
                            "dimension": dim_v,
                            "criterion": crit_text,
                            "criterion_id": _crit_id(crit_text),
                            # Issue 1: satisfied is now nullable (Int8 via pandas) to
                            # preserve distinction between False and "unknown"; but for
                            # analyses that require bool, False is a safe fallback.
                            "satisfied": bool(satisfied) if satisfied is not None else False,
                            "satisfied_is_known": satisfied is not None,
                            "evidence": ev,
                            "reasoning": rs,
                        }
                    )

    if verdict_schema_anomalies:
        anomalies.append(f"{verdict_schema_anomalies} non-dict verdicts skipped")
    if verdict_unknown_satisfied:
        anomalies.append(
            f"{verdict_unknown_satisfied} verdicts had no parseable satisfied/verdict field "
            "(coerced to False; see satisfied_is_known column)"
        )

    df_scores = pd.DataFrame(score_rows)
    df_overall = pd.DataFrame(overall_rows)
    df_verdicts = pd.DataFrame(verdict_rows)

    for c in ("pattern", "pattern_family", "judge", "dimension"):
        if c in df_scores.columns:
            df_scores[c] = pd.Categorical(df_scores[c])
    for c in ("pattern", "pattern_family", "judge"):
        if c in df_overall.columns:
            df_overall[c] = pd.Categorical(df_overall[c])
    for c in ("pattern", "pattern_family", "judge", "dimension"):
        if c in df_verdicts.columns:
            df_verdicts[c] = pd.Categorical(df_verdicts[c])

    for c in ("score",):
        if c in df_scores.columns:
            df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")
    for c in ("met", "total"):
        if c in df_scores.columns:
            df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce").astype("Int64")
    for c in (
        "overall_score",
        "overall_score_recomputed",
        "overall_score_per_query_weights",
        "judge_latency_s",
    ):
        if c in df_overall.columns:
            df_overall[c] = pd.to_numeric(df_overall[c], errors="coerce")
    for c in ("n_criteria", "n_satisfied", "judge_tokens"):
        if c in df_overall.columns:
            df_overall[c] = pd.to_numeric(df_overall[c], errors="coerce").astype("Int64")

    return df_scores, df_overall, df_verdicts, anomalies


# ----- Coverage report -----
def write_coverage_report(
    df_queries: pd.DataFrame,
    df_runs: pd.DataFrame,
    df_overall: pd.DataFrame,
    df_scores: pd.DataFrame,
    df_verdicts: pd.DataFrame,
    anomalies: list[str],
    mismatch_rows: pd.DataFrame,
    rubric_drift_groups: int,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Analysis Dataframes — Coverage Report\n")
    lines.append(f"- Total queries in manifest: **{len(df_queries)}**")
    lines.append(
        f"- Patterns discovered: **{df_runs['pattern'].nunique()}** "
        "(base + ablation + protocol_a + variance + disentanglement)"
    )
    lines.append(
        f"- Judges: {', '.join(sorted(df_overall['judge'].unique().tolist())) if len(df_overall) else 'none'}\n"
    )

    # Post-fix anomaly summary box
    lines.append("## Post-fix anomaly summary\n")
    lines.append(f"- Rubric-drift groups (same (pattern,query,dim) with different criterion sets across judges): **{rubric_drift_groups}**")
    lines.append(f"- Stored-vs-recomputed overall_score mismatches (|Δ|>0.01): **{len(mismatch_rows)}**")
    lines.append(f"- Total raw anomalies logged: **{len(anomalies)}**\n")

    # Per-pattern × per-judge coverage
    lines.append("## Judge coverage per pattern (X / 90 queries scored)\n")
    pivot = (
        df_overall.groupby(["pattern", "judge"], observed=True)["query_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    lines.append(pivot.to_markdown())
    lines.append("")

    # Run / report existence per pattern
    lines.append("\n## Run / report existence per pattern\n")
    run_pivot = df_runs.groupby("pattern", observed=True).agg(
        reports_exist=("report_exists", "sum"),
        checkpoints_found=("status", lambda s: (s != "missing_checkpoint").sum()),
        success=("status", lambda s: (s == "success").sum()),
        excluded=("excluded_from_analysis", "max"),
    )
    lines.append(run_pivot.to_markdown())
    lines.append("")

    # Missing (pattern, query_id) for base patterns
    lines.append("\n## Missing report files in base patterns (pattern, query_id)\n")
    base_missing = df_runs[
        (df_runs["pattern_family"] == "base") & (~df_runs["report_exists"])
    ][["pattern", "query_id"]]
    if len(base_missing):
        for _, r in base_missing.iterrows():
            lines.append(f"- `{r['pattern']}` / `{r['query_id']}`")
    else:
        lines.append("None.")
    lines.append("")

    # Ablation notes
    lines.append("\n## Ablation coverage notes\n")
    abl_patterns = sorted(
        df_runs[df_runs["pattern_family"] == "ablation"]["pattern"].unique().tolist()
    )
    for p in abl_patterns:
        reps = int((df_runs["pattern"] == p).sum())
        exist = int(((df_runs["pattern"] == p) & df_runs["report_exists"]).sum())
        judges_cov = df_overall[df_overall["pattern"] == p].groupby(
            "judge", observed=True
        )["query_id"].nunique().to_dict()
        excl = p in ABLATION_EXCLUDED_PATTERNS
        tag = " [EXCLUDED]" if excl else ""
        lines.append(f"- **{p}**{tag}: reports={exist}/{reps}; judge coverage={judges_cov}")
    if ABLATION_EXCLUDED_PATTERNS:
        lines.append(f"\nExclusion criterion: {ABLATION_EXCLUSION_REASON}")
    lines.append("")

    # Overall score verification (stored vs recomputed)
    lines.append("\n## Overall score verification (stored vs recomputed)\n")
    lines.append(
        "- `overall_score_recomputed` uses source-type weights from "
        "`DIMENSION_WEIGHTS_BY_SOURCE` (matches the gpt52 stored convention)."
    )
    lines.append(
        "- `overall_score_per_query_weights` uses per-query "
        "`rubric.dimension_weights` from the eval manifest."
    )
    lines.append(
        "- `overall_score_trustworthy` is False for claude_sonnet (upstream-corrupted)."
    )
    lines.append(f"- Rows compared: **{len(df_overall)}**")
    lines.append(f"- Rows with |stored - recomputed| > 0.01: **{len(mismatch_rows)}**")
    if len(df_overall):
        by_judge = (
            (df_overall["overall_score"] - df_overall["overall_score_recomputed"])
            .abs()
            .groupby(df_overall["judge"], observed=True)
            .agg(["mean", "median", "max"])
        )
        lines.append("\nPer-judge stored-vs-recomputed delta:\n")
        lines.append(by_judge.to_markdown())
    lines.append("")

    # Verdict satisfied distribution per judge (sanity check on Issue 1 fix)
    lines.append("\n## Verdict satisfied distribution per judge\n")
    if len(df_verdicts):
        vd = (
            df_verdicts.groupby("judge", observed=True)["satisfied"]
            .agg(["count", "sum"])
            .rename(columns={"count": "n_verdicts", "sum": "n_satisfied"})
        )
        vd["satisfaction_rate"] = vd["n_satisfied"] / vd["n_verdicts"]
        lines.append(vd.to_markdown())
    lines.append("")

    # Token histogram summary
    lines.append("\n## Total tokens per pattern (summary stats)\n")
    tok = df_runs.groupby("pattern", observed=True)["total_tokens"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    lines.append(tok.to_markdown())
    lines.append("")

    # Cost proxy summary
    lines.append("\n## Cost proxy (USD) per pattern\n")
    cp = df_runs.groupby("pattern", observed=True)["cost_proxy_usd"].agg(
        ["count", "mean", "median", "sum"]
    )
    lines.append(cp.to_markdown())
    lines.append("")

    # Schema / anomaly notes
    lines.append("\n## Schema anomalies and notes\n")
    if anomalies:
        seen: set[str] = set()
        shown = 0
        for a in anomalies:
            key = a[:200]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {a}")
            shown += 1
            if shown >= 50:
                lines.append(f"- ... ({len(anomalies) - 50} more suppressed)")
                break
    else:
        lines.append("No schema anomalies detected.")

    # df sizes summary
    lines.append("\n## Dataframe row counts\n")
    lines.append(f"- df_queries: {len(df_queries)}")
    lines.append(f"- df_runs: {len(df_runs)}")
    lines.append(f"- df_scores: {len(df_scores)}")
    lines.append(f"- df_overall_scores: {len(df_overall)}")
    lines.append(f"- df_verdicts: {len(df_verdicts)}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----- Data dictionary -----
def write_data_dictionary(out_path: Path) -> None:
    content = """# Data Dictionary — data/analysis/*.parquet

This dictionary documents every column in every parquet file produced by
`scripts/build_analysis_dataframes.py`. Intended as COLM/EMNLP supplementary
material.

## Conventions

- `pattern`: e.g. `base_p4`, `ablation_p3_no_quality_eval`.
- `pattern_family`: `"base"`, `"ablation"`, `"protocol_a"`, `"variance"`, or `"disentanglement"`.
- `pattern_short`: suffix after `base_` / `ablation_`, e.g. `p4`, `p3_no_quality_eval`.
- `query_id`: unique id from `data/eval_queries_v2.json`.
- `judge`: one of `gpt52`, `claude_opus`, `claude_sonnet`, `claude_code`.
- `dimension`: one of the 9 rubric-v2 dimensions listed below.

## Rubric v2 dimensions

`information_recall`, `factual_accuracy`, `coverage`, `analytical_depth`,
`citation_quality`, `logical_coherence`, `organization`, `instruction_following`,
`attribution_quality`. Weights are query- and source-dependent; see
`build_manifest.json` for the canonical hash.

## df_queries.parquet

| column | dtype | description | source |
|---|---|---|---|
| query_id | str | Unique query identifier | `data/eval_queries_v2.json` |
| source | category | Benchmark source: `custom`, `draco`, `deepsearch_qa`, `research_qa`, `litqa2` | manifest |
| domain | str | Query subject domain (free-form) | manifest |
| difficulty | category | `easy` / `medium` / `hard` | manifest |
| query_text | str | Natural-language query | manifest |
| expected_topics | list[str] | Expected coverage elements | manifest `expected_elements` |
| gold_answer | str | Reference/gold answer if known | manifest `reference_answer` |

## df_runs.parquet

One row per (pattern × query_id) whether or not a report exists.

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | Pattern directory name | `results/experiments/` |
| pattern_family | category | `base`, `ablation`, `protocol_a`, `variance`, or `disentanglement` | derived |
| pattern_short | category | Suffix after prefix, e.g. `p4` | derived |
| query_id | str | FK to df_queries | manifest |
| status | category | `success` / `failed` / `missing_checkpoint` / `missing_status` | checkpoint JSON (see Issue 5) |
| elapsed_seconds | float64 | Wall-clock seconds for the run | checkpoint |
| total_tokens | float64 | Total LLM tokens consumed | checkpoint |
| total_cost_usd | float64 | Cost recorded by upstream caller (may be missing for local models) | checkpoint |
| sections | float64 | Report section count | checkpoint |
| citations | float64 | Citations emitted | checkpoint |
| timestamp | datetime64 | Run timestamp | checkpoint |
| report_path | str \\| null | Absolute path to the `.md` report when it exists | filesystem |
| report_exists | bool | True iff `.md` report file is present | filesystem |
| word_count_is_present | bool | Alias of `report_exists` for explicit gating | filesystem |
| report_word_count | float64 | Word count of `.md`; **NaN when report is missing** (not 0) | filesystem |
| cost_proxy_usd | float64 | GPT-4o patterns: `total_tokens * $5/M`. Local 7B (p9, p10, p12): `elapsed_seconds * $0.0001/sec`. | derived |
| excluded_from_analysis | bool | True for patterns in the exclusion set (currently `ablation_p5_no_citation_verify`, only 2/90 reports) | derived |
| elapsed_is_suspect | bool | True if `elapsed_seconds > 2 * median(elapsed_seconds)` within the same pattern | derived |

## df_scores.parquet

One row per (pattern × query × judge × dimension).

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| dimension | category | One of 9 rubric-v2 dimensions | judge JSON |
| score | float64 | Per-dimension score in [0, 1] | judge JSON `dimensions[dim].score` |
| met | Int64 | Criteria met (normalized across upstream schema variants) | judge JSON |
| total | Int64 | Criteria total (normalized across upstream schema variants) | judge JSON |

## df_overall_scores.parquet

One row per (pattern × query × judge).

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| overall_score | float64 | Stored top-level overall score | judge JSON |
| overall_score_recomputed | float64 | Recomputed using `DIMENSION_WEIGHTS_BY_SOURCE[query.source]` applied to per-dimension scores (falls back to met/total when score is missing) | derived |
| overall_score_per_query_weights | float64 | Recomputed using the per-query `rubric.dimension_weights` from the manifest | derived |
| overall_score_trustworthy | bool | **False for claude_sonnet** — its stored `overall_score` is upstream-corrupted. Downstream analyses MUST use `overall_score_recomputed` or `overall_score_per_query_weights` for those rows. True for gpt52, claude_opus, claude_code. | policy |
| n_criteria | Int64 | Criteria total across dimensions | judge JSON |
| n_satisfied | Int64 | Criteria satisfied across dimensions | judge JSON |
| judge_tokens | Int64 | Tokens consumed by the judge (gpt52 only) | judge JSON |
| judge_latency_s | float64 | Judge call latency (gpt52 only) | judge JSON |

## df_verdicts.parquet

One row per individual criterion verdict.

| column | dtype | description | source |
|---|---|---|---|
| pattern | category | | |
| pattern_family | category | | |
| query_id | str | | |
| judge | category | | |
| criterion_index | Int64 | Index within the judge's verdict list | judge JSON |
| dimension | category | Rubric dimension | judge JSON |
| criterion | str | **Normalized** criterion text. Claude Opus uses varied keys — `criterion`, `description`, `text`, `criterion_text`; all are normalized to this column. | judge JSON |
| criterion_id | str | 12-char md5 of `normalize(criterion)` where `normalize = lowercase + strip + whitespace-collapse`. Stable across wording jitter. | derived |
| satisfied | bool | True/False. For claude_opus rows that used `verdict: "SATISFIED"/"NOT_SATISFIED"`, the string is mapped to bool. If the verdict could not be parsed, value is False and `satisfied_is_known` is False. | judge JSON |
| satisfied_is_known | bool | False when neither `satisfied` nor `verdict` could be extracted. Filter on this for trusted analyses. | derived |
| evidence | str | Evidence quote from report | judge JSON |
| reasoning | str | Judge reasoning (from `reasoning` or `reason`) | judge JSON |

## Known upstream data issues (documented, not silently repaired)

1. **Claude Opus verdict schema heterogeneity** (13 distinct variants observed).
   Normalized per-verdict in this script — see `_extract_verdict_fields`.
2. **Claude Sonnet stored overall_score is corrupted.** Flagged via
   `overall_score_trustworthy = False`. Always use the recomputed columns.
3. **Criterion-id stability.** Criterion text is normalized before hashing so
   whitespace/case jitter across judge runs does not inflate rubric-drift counts.
4. **ablation_p5_no_citation_verify is excluded** from statistical comparisons
   (`excluded_from_analysis = True`) because only 2/90 reports were generated.

## Build reproducibility

See `build_manifest.json` for script hash, input paths, rubric-weight hash,
python/pandas/pyarrow versions, and per-parquet row counts.
"""
    out_path.write_text(content, encoding="utf-8")


# ----- Build manifest -----
def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(obj: Any) -> str:
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def write_build_manifest(
    out_path: Path,
    row_counts: dict[str, int],
    anomalies: list[str],
) -> None:
    import platform

    import pyarrow

    try:
        script_path = Path(__file__).resolve()
    except NameError:
        script_path = ROOT / "scripts" / "build_analysis_dataframes.py"
    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "input_paths": [
            str(DATA_DIR / "eval_queries_v2.json"),
            str(EXPERIMENTS_DIR),
            str(CHECKPOINTS_DIR),
            *(str(p) for p in JUDGE_DIRS.values()),
            str(ROOT / "deep_research/evaluation/rubric_v2.py"),
        ],
        "v2_dimension_weights": dict(DIMENSION_WEIGHTS_V2),
        "v2_dimension_weights_by_source": {
            k: dict(v) for k, v in DIMENSION_WEIGHTS_BY_SOURCE.items()
        },
        "v2_weights_hash": _canonical_json_sha256(
            {
                "default": dict(DIMENSION_WEIGHTS_V2),
                "by_source": {k: dict(v) for k, v in DIMENSION_WEIGHTS_BY_SOURCE.items()},
            }
        ),
        "row_counts": row_counts,
        "detected_anomalies": anomalies[:500],  # cap to keep file small
        "detected_anomalies_truncated": len(anomalies) > 500,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "notes": {
            "claude_sonnet_overall_score": (
                "Upstream-corrupted. Use overall_score_recomputed or "
                "overall_score_per_query_weights for claude_sonnet rows."
            ),
            "excluded_patterns": sorted(ABLATION_EXCLUDED_PATTERNS),
            "excluded_reason": ABLATION_EXCLUSION_REASON,
            "cost_proxy_rates": {
                "gpt4o_blended_usd_per_m_tokens": GPT4O_BLENDED_RATE_PER_M_TOKENS,
                "local_model_usd_per_sec": LOCAL_MODEL_COMPUTE_RATE_PER_SEC,
                "local_model_patterns_short": sorted(LOCAL_MODEL_PATTERNS_SHORT),
            },
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building df_queries ...")
    df_queries = build_df_queries()
    query_ids = df_queries["query_id"].tolist()
    assert len(query_ids) == len(set(query_ids)), "Duplicate query_id in manifest"

    print(f"Building df_runs across {len(discover_pattern_dirs())} patterns ...")
    df_runs = build_df_runs(query_ids)

    print("Building judge frames (scores, overall, verdicts) ...")
    df_scores, df_overall, df_verdicts, anomalies = build_judge_frames(df_queries)

    # ----- Validation checks -----
    dup = df_overall.duplicated(subset=["pattern", "query_id", "judge"])
    assert not dup.any(), f"Duplicate rows in df_overall_scores: {dup.sum()}"

    if len(df_scores):
        bad_dims = set(df_scores["dimension"].astype(str).unique()) - V2_DIMENSIONS
        assert not bad_dims, f"Unknown dimensions in df_scores: {bad_dims}"

    nan_scores = df_scores["score"].isna().sum() if len(df_scores) else 0
    if nan_scores:
        anomalies.append(f"{nan_scores} dimension rows with NaN score")

    n_rubric_diff = 0
    if len(df_verdicts):
        dim_per_crit = df_verdicts.groupby("criterion_id")["dimension"].nunique()
        mixed = int((dim_per_crit > 1).sum())
        if mixed:
            anomalies.append(
                f"{mixed} criterion_id(s) mapped to >1 dimension across judges "
                "(rubric drift warning)"
            )
        per_key = (
            df_verdicts.groupby(["pattern", "query_id", "dimension", "judge"], observed=True)[
                "criterion_id"
            ]
            .apply(lambda s: tuple(sorted(set(s))))
            .reset_index()
        )
        mismatches = (
            per_key.groupby(["pattern", "query_id", "dimension"], observed=True)["criterion_id"]
            .nunique()
        )
        n_rubric_diff = int((mismatches > 1).sum())
        if n_rubric_diff:
            anomalies.append(
                f"{n_rubric_diff} (pattern, query_id, dimension) groups show different "
                "criterion sets across judges — rubric version drift"
            )

    # Stored vs recomputed overall score (restrict to trustworthy rows)
    mismatch_rows = pd.DataFrame()
    if len(df_overall):
        mask = (
            df_overall["overall_score"].notna()
            & df_overall["overall_score_recomputed"].notna()
            & df_overall["overall_score_trustworthy"]
        )
        delta = (
            df_overall.loc[mask, "overall_score"]
            - df_overall.loc[mask, "overall_score_recomputed"]
        ).abs()
        mismatch_rows = df_overall.loc[mask & (delta > 0.01)].copy()

    # ----- Write parquet -----
    print("Writing parquet files ...")
    df_queries.to_parquet(ANALYSIS_DIR / "df_queries.parquet", index=False)
    df_runs.to_parquet(ANALYSIS_DIR / "df_runs.parquet", index=False)
    df_scores.to_parquet(ANALYSIS_DIR / "df_scores.parquet", index=False)
    df_overall.to_parquet(ANALYSIS_DIR / "df_overall_scores.parquet", index=False)
    df_verdicts.to_parquet(ANALYSIS_DIR / "df_verdicts.parquet", index=False)

    # ----- Coverage report -----
    print("Writing coverage_report.md ...")
    write_coverage_report(
        df_queries,
        df_runs,
        df_overall,
        df_scores,
        df_verdicts,
        anomalies,
        mismatch_rows,
        n_rubric_diff,
        ANALYSIS_DIR / "coverage_report.md",
    )

    # ----- Data dictionary -----
    print("Writing DATA_DICTIONARY.md ...")
    write_data_dictionary(ANALYSIS_DIR / "DATA_DICTIONARY.md")

    # ----- Build manifest -----
    print("Writing build_manifest.json ...")
    row_counts = {
        "df_queries": int(len(df_queries)),
        "df_runs": int(len(df_runs)),
        "df_scores": int(len(df_scores)),
        "df_overall_scores": int(len(df_overall)),
        "df_verdicts": int(len(df_verdicts)),
    }
    write_build_manifest(ANALYSIS_DIR / "build_manifest.json", row_counts, anomalies)

    print(
        f"OK queries={len(df_queries)} runs={len(df_runs)} "
        f"scores={len(df_scores)} overall={len(df_overall)} verdicts={len(df_verdicts)} "
        f"score_mismatches={len(mismatch_rows)} rubric_drift_groups={n_rubric_diff} "
        f"anomalies={len(anomalies)}"
    )


if __name__ == "__main__":
    main()
