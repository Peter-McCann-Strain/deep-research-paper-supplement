#!/usr/bin/env python3
"""G2: Score the 5 trace-instrumented base patterns on the 4-dim process
trajectory rubric (E8) and compare to outcome rubric V2 overall_score.

Inputs:
  checkpoints/<pattern_dir>/<run_ts>/trace.json   (ProcessTrace.model_dump)
  results/experiments/<pattern>/<qid>.md          (final report)
  data/analysis/df_overall_scores.parquet         (V2 outcome scores)
  data/eval_queries_v2.json                       (query manifest)

Outputs:
  reports/phase8_trajectory/coverage_audit.md
  reports/phase8_trajectory/metadata_scores.csv
  reports/phase8_trajectory/llm_scores.csv
  reports/phase8_trajectory/per_pattern_summary.csv
  reports/phase8_trajectory/per_pattern_summary.md
  reports/phase8_trajectory/spearman_vs_outcome.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import DEFAULT_MODEL
from deep_research.tools import LLMCaller, CostTracker

# Pattern code -> base_p<N> experiment dir (for reading judged reports)
PATTERN_TO_EXP = {
    "p0_baseline": "base_p0",
    "p1_iterative_rag": "base_p1",
    "p4_perspective_storm": "base_p4",
    "p7_graph_decomposition": "base_p7",
    "p10_deep_researcher": "base_p10",
}

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "reports" / "phase8_trajectory"
DATA = ROOT / "data"


# ── Coverage audit ──────────────────────────────────────────────────────────

def audit_coverage(manifest_qmap: dict[str, str]) -> tuple[dict[str, dict], str]:
    """Walk checkpoints/p<N>_<pattern>/<run_ts>/trace.json, count traces per
    pattern and per-qid coverage. Returns (per_pattern_stats, audit_md)."""
    pattern_dirs = sorted(
        d for d in CKPT.iterdir()
        if d.is_dir() and d.name.startswith("p") and "_" in d.name
        and d.name != "p11_react"  # placeholder check — handled by glob below
    )
    # Reconstruct full list including p11
    pattern_dirs = sorted(
        d for d in CKPT.iterdir()
        if d.is_dir() and d.name.split("_")[0].startswith("p")
        and d.name.split("_")[0][1:].isdigit()
    )

    stats: dict[str, dict] = {}
    for pd_dir in pattern_dirs:
        name = pd_dir.name
        traces = sorted(pd_dir.glob("*/trace.json"))
        n_traces = len(traces)
        unique_qids: set[str] = set()
        unique_queries: set[str] = set()
        for tf in traces:
            try:
                d = json.loads(tf.read_text())
                q = d.get("data", {}).get("query", "")
                if q in manifest_qmap:
                    unique_qids.add(manifest_qmap[q])
                unique_queries.add(q)
            except Exception:
                pass
        stats[name] = {
            "n_trace_files": n_traces,
            "n_unique_queries": len(unique_queries),
            "n_unique_manifest_qids": len(unique_qids),
            "unique_qids": sorted(unique_qids),
        }

    instrumented = [n for n, s in stats.items() if s["n_trace_files"] >= 30]
    total = len(stats)

    md = ["# G2 — Process Trajectory Coverage Audit\n",
          f"\nTotal base patterns inspected: **{total}**",
          f"\nSubstantively instrumented (≥30 traces): **{len(instrumented)}** "
          f"({100 * len(instrumented) / total:.0f}%)\n",
          "## Per-pattern trace coverage\n",
          "| Pattern | Trace files | Unique queries | Unique manifest qids |",
          "|---|---:|---:|---:|"]
    for name in sorted(stats):
        s = stats[name]
        md.append(f"| {name} | {s['n_trace_files']} | {s['n_unique_queries']} | "
                  f"{s['n_unique_manifest_qids']} |")
    md.append("")
    md.append("**Interpretation.** Patterns with 0 trace files were not "
              "instrumented to call StateManager.save('trace', …) when their runs "
              "were executed. p12_rl_trained has a single trace (one-off probe, "
              "excluded from per-pattern analysis). The 5 fully-instrumented "
              "patterns (P0/P1/P4/P7/P10) cover 30 of the 90 v2 manifest queries, "
              "with 3 replicates each (~90 trace files per pattern; P0 has 53 "
              "across 27 qids). G2 scoring proceeds on this 5-pattern subset — "
              "downstream paper writeup is marked 'process audit on a 5-pattern "
              "subset (~42% of base patterns)'.\n")
    return stats, "\n".join(md)


# ── Metadata-only scoring ───────────────────────────────────────────────────

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def score_metadata(trace: dict) -> dict:
    """User-spec metadata-only scoring.

    retrieval_diversity: distinct retrieval atoms / total search invocations.
      We use the most informative signal available in trace metadata:
        primary    = n_unique_urls_visited    (when >0)
        fallback   = unique distinct query strings logged on search calls
        per-search = primary / max(1, n_search_invocations)
      Normalised to [0,1] via cap at 8 unique atoms per search invocation
      (effective signal saturates beyond ~8 distinct sources).

    tool_efficiency: total result yield / total tool calls.
      We use sum(n_results) over all calls / total tool_calls. Higher = more
      info gathered per agent step. Normalised by saturating at 8 results/call.
    """
    calls = trace.get("tool_calls", [])
    if not calls:
        return {
            "retrieval_diversity_raw": 0.0,
            "retrieval_diversity_score": 0.0,
            "tool_efficiency_raw": 0.0,
            "tool_efficiency_score": 0.0,
            "n_search_calls": 0, "n_tool_calls": 0,
            "n_unique_urls": 0, "n_total_results": 0,
        }

    search_calls = [c for c in calls
                    if c.get("tool") in ("search", "academic_search")]
    n_search = len(search_calls)
    n_search_invocations = max(1, n_search)

    # Diversity primary signal: n_unique_urls_visited from trace summary
    n_unique_urls = trace.get("n_unique_urls_visited", 0) or 0
    n_distinct_queries = 0
    n_search_results = sum((c.get("n_results") or 0) for c in search_calls)
    if n_unique_urls == 0:
        # Fallback 1: distinct query strings (works for P0, P10 where queries logged)
        distinct_queries = set()
        for c in search_calls:
            q = (c.get("input_args", {}) or {}).get("query", "")
            if q:
                distinct_queries.add(q.lower().strip())
        n_distinct_queries = len(distinct_queries)
        # Fallback 2: total search-result count as an extraction-diversity proxy
        # (P1 batches its sub-queries into one search call but logs n_results = #extractions).
        n_unique_urls = n_distinct_queries or n_search_results

    diversity_raw = n_unique_urls / n_search_invocations
    # Saturate at 8 distinct atoms per search call (treat that as fully diverse)
    diversity_score = min(1.0, diversity_raw / 8.0)

    # Efficiency: total result count / total tool calls
    n_total_results = sum((c.get("n_results") or 0) for c in calls)
    n_tool = len(calls)
    eff_raw = n_total_results / max(1, n_tool)
    eff_score = min(1.0, eff_raw / 8.0)

    return {
        "retrieval_diversity_raw": round(diversity_raw, 3),
        "retrieval_diversity_score": round(diversity_score, 3),
        "tool_efficiency_raw": round(eff_raw, 3),
        "tool_efficiency_score": round(eff_score, 3),
        "n_search_calls": n_search,
        "n_tool_calls": n_tool,
        "n_unique_urls": n_unique_urls,
        "n_total_results": n_total_results,
        "n_distinct_search_queries": n_distinct_queries,
    }


# ── LLM-judged scoring ──────────────────────────────────────────────────────

def trace_to_summary(trace: dict, max_calls: int = 30) -> str:
    calls = trace.get("tool_calls", [])[:max_calls]
    lines = []
    for c in calls:
        args = c.get("input_args") or {}
        arg = args.get("query") or args.get("url") or ""
        if not arg and isinstance(args, dict):
            # Fall back to a compact rendering of args
            arg = "; ".join(f"{k}={v}" for k, v in args.items() if not isinstance(v, (list, dict)))
        arg_short = str(arg)[:100]
        out_short = str(c.get("output_summary", ""))[:100]
        lines.append(f"  step {c.get('step_idx')}: {c.get('tool')}({arg_short!r}) -> "
                     f"{out_short} (n_results={c.get('n_results')}, tok={c.get('tokens_used')})")
    if len(trace.get("tool_calls", [])) > max_calls:
        lines.append(f"  ... ({len(trace['tool_calls']) - max_calls} more steps)")
    return "\n".join(lines)


REASONING_COHERENCE_PROMPT = """You are evaluating the reasoning coherence of a research-agent's tool-call trace. Score on a 0.0-1.0 scale.

A coherent trace:
  - Each search query builds on prior observations (not random)
  - Read calls are aimed at sources discovered by earlier searches
  - The final report integrates evidence retrieved during the trace
  - There are no obvious dead ends or unmotivated detours

An incoherent trace:
  - Search queries are unrelated to each other or the original query
  - Read calls fetch URLs that are never cited in the final report
  - Long detours that don't contribute to the final answer
  - Repeated near-identical operations without progress

Output JSON only: {{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}.

Original query: {query}

Tool-call trace:
{trace_summary}

Final report excerpt (first 2000 chars):
{report_excerpt}"""


ITERATIVE_REFINEMENT_PROMPT = """You are evaluating whether a research-agent's tool-call trace shows productive iterative refinement. Score on 0.0-1.0.

Productive refinement:
  - Later searches narrow or refine based on early findings
  - Reflection/critique steps lead to gap-filling searches
  - Multiple rounds with each round measurably improving evidence base
  - Quality-evaluator or self-critique calls trigger meaningful corrections

Unproductive (low score):
  - Just one search round with no refinement
  - "Refinement" loops that don't change behaviour
  - Iterations that re-fetch the same sources without adding new ones

Output JSON only: {{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}.

Original query: {query}

Tool-call trace:
{trace_summary}"""


async def llm_score(llm: LLMCaller, prompt: str, model: str) -> tuple[float, str]:
    raw = await llm.complete_json(prompt, model=model, temperature=0.1, max_tokens=300)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
    else:
        data = raw or {}
    try:
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))
    rationale = str(data.get("rationale", ""))[:300]
    return score, rationale


# ── Main pipeline ───────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-pattern", type=int, default=0,
                        help="Cap traces scored per pattern (0 = all)")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Only run metadata-only scoring + audit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Load manifest for query lookup
    manifest = json.loads((DATA / "eval_queries_v2.json").read_text())
    qmap = {q["query"]: q["id"] for q in manifest["queries"]}

    # ─── Step 2: coverage audit ────────────────────────────────────────────
    print("[1/5] Auditing trace coverage...")
    stats, audit_md = audit_coverage(qmap)
    (OUT / "coverage_audit.md").write_text(audit_md)
    print(f"  -> {OUT/'coverage_audit.md'}")

    # Patterns substantively instrumented (≥30 traces); skip one-off probes (e.g. p12 with n=1)
    instrumented = [n for n, s in stats.items() if s["n_trace_files"] >= 30]
    print(f"  Instrumented patterns (n>=30): {instrumented}")

    # ─── Step 3: collect traces and compute metadata scores ───────────────
    print("[2/5] Loading traces and computing metadata-only scores...")
    rows_meta = []
    rows_for_llm: list[dict] = []
    for pat in instrumented:
        traces = sorted((CKPT / pat).glob("*/trace.json"))
        if args.limit_per_pattern:
            traces = traces[:args.limit_per_pattern]
        for tf in traces:
            try:
                d = json.loads(tf.read_text())
                trace = d.get("data", {})
                query = trace.get("query", "")
                qid = qmap.get(query, "")
                if not qid:
                    continue  # query not in manifest
                ms = score_metadata(trace)
                rows_meta.append({
                    "pattern_code": pat,
                    "pattern_exp_dir": PATTERN_TO_EXP.get(pat, pat),
                    "run_ts": tf.parent.name,
                    "query_id": qid,
                    "query": query[:120],
                    "n_iterations": trace.get("n_iterations", 0),
                    **ms,
                })
                rows_for_llm.append({
                    "pattern_code": pat,
                    "run_ts": tf.parent.name,
                    "query_id": qid,
                    "query": query,
                    "trace": trace,
                })
            except Exception as e:
                print(f"  WARN: {tf} failed to parse: {e}")
    df_meta = pd.DataFrame(rows_meta)
    df_meta.to_csv(OUT / "metadata_scores.csv", index=False)
    print(f"  -> {OUT/'metadata_scores.csv'}  (n={len(df_meta)})")

    # ─── Step 4: LLM-judged scores ────────────────────────────────────────
    if args.skip_llm:
        print("[3/5] --skip-llm: skipping LLM scoring.")
        df_llm = pd.DataFrame()
    else:
        print(f"[3/5] LLM-judged scoring on {len(rows_for_llm)} traces with {args.concurrency} concurrency...")
        tracker = CostTracker(budget_usd=10.0)
        llm = LLMCaller(cost_tracker=tracker)
        sem = asyncio.Semaphore(args.concurrency)

        rows_llm: list[dict] = []
        completed = 0
        n_total = len(rows_for_llm)
        t0 = time.time()

        async def _score_one(rec: dict):
            nonlocal completed
            async with sem:
                pat = rec["pattern_code"]
                qid = rec["query_id"]
                trace = rec["trace"]
                query = rec["query"]
                exp_dir = PATTERN_TO_EXP.get(pat, pat)
                rep_path = EXP / exp_dir / f"{qid}.md"
                report_text = rep_path.read_text() if rep_path.exists() else ""

                t_summary = trace_to_summary(trace)
                rc_score = rc_rat = ir_score = ir_rat = None
                try:
                    rc_prompt = REASONING_COHERENCE_PROMPT.format(
                        query=query, trace_summary=t_summary,
                        report_excerpt=report_text[:2000],
                    )
                    rc_score, rc_rat = await llm_score(llm, rc_prompt, args.model)
                except Exception as e:
                    print(f"  WARN rc fail {pat}/{qid[:20]}: {str(e)[:80]}")
                    rc_score, rc_rat = 0.0, f"error: {str(e)[:80]}"

                try:
                    ir_prompt = ITERATIVE_REFINEMENT_PROMPT.format(
                        query=query, trace_summary=t_summary,
                    )
                    ir_score, ir_rat = await llm_score(llm, ir_prompt, args.model)
                except Exception as e:
                    print(f"  WARN ir fail {pat}/{qid[:20]}: {str(e)[:80]}")
                    ir_score, ir_rat = 0.0, f"error: {str(e)[:80]}"

                rows_llm.append({
                    "pattern_code": pat,
                    "run_ts": rec["run_ts"],
                    "query_id": qid,
                    "reasoning_coherence_score": rc_score,
                    "reasoning_coherence_rationale": rc_rat,
                    "iterative_refinement_score": ir_score,
                    "iterative_refinement_rationale": ir_rat,
                    "report_chars": len(report_text),
                })
                completed += 1
                if completed % 10 == 0 or completed == n_total:
                    rate = completed / max(1e-3, time.time() - t0)
                    print(f"  [{completed}/{n_total}] rate={rate:.2f}/s "
                          f"cost=${tracker.total_cost:.2f}", flush=True)

        await asyncio.gather(*(_score_one(r) for r in rows_for_llm))
        df_llm = pd.DataFrame(rows_llm)
        df_llm.to_csv(OUT / "llm_scores.csv", index=False)
        print(f"  -> {OUT/'llm_scores.csv'}  (n={len(df_llm)}, "
              f"cost=${tracker.total_cost:.2f})")

    # ─── Step 5: aggregate and Spearman vs outcome rubric ────────────────
    print("[4/5] Aggregating per-pattern means + 95% CIs...")
    if not df_meta.empty:
        joined = df_meta.copy()
        if not df_llm.empty:
            joined = df_meta.merge(
                df_llm[["pattern_code", "run_ts", "query_id",
                        "reasoning_coherence_score", "iterative_refinement_score"]],
                on=["pattern_code", "run_ts", "query_id"], how="left",
            )

        dim_cols = ["retrieval_diversity_score", "tool_efficiency_score"]
        if not df_llm.empty:
            dim_cols += ["reasoning_coherence_score", "iterative_refinement_score"]

        # Per-pattern aggregates
        rows_agg = []
        for pat, g in joined.groupby("pattern_code"):
            row = {"pattern_code": pat, "n": len(g)}
            for c in dim_cols:
                if c in g.columns:
                    s = g[c].dropna()
                    mean = s.mean() if len(s) else float("nan")
                    sd = s.std(ddof=1) if len(s) > 1 else 0.0
                    se = sd / math.sqrt(len(s)) if len(s) else 0.0
                    ci = 1.96 * se
                    row[f"{c}_mean"] = round(mean, 3)
                    row[f"{c}_ci95"] = round(ci, 3)
                    row[f"{c}_n"] = len(s)
            rows_agg.append(row)
        df_agg = pd.DataFrame(rows_agg)
        df_agg.to_csv(OUT / "per_pattern_summary.csv", index=False)

        # Markdown table
        md_lines = ["# G2 — Process Trajectory Per-Pattern Summary\n",
                    f"\nN traces scored: {len(joined)} across "
                    f"{joined['pattern_code'].nunique()} patterns "
                    f"(P0/P1/P4/P7/P10). Each pattern: ~30 unique "
                    f"queries × 3 replicates = ~90 traces.\n",
                    "## Per-pattern means (95% CI)\n"]
        header = "| Pattern | N | " + " | ".join(c.replace("_score", "") for c in dim_cols) + " |"
        sep = "|---|---:|" + "---:|" * len(dim_cols)
        md_lines.append(header)
        md_lines.append(sep)
        for _, r in df_agg.iterrows():
            cells = [r["pattern_code"], str(int(r["n"]))]
            for c in dim_cols:
                m = r.get(f"{c}_mean", float("nan"))
                ci = r.get(f"{c}_ci95", 0.0)
                cells.append(f"{m:.3f} ±{ci:.3f}" if pd.notna(m) else "—")
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")

        # Spearman vs outcome rubric
        print("[5/5] Computing Spearman vs outcome rubric (df_overall_scores)...")
        try:
            df_out = pd.read_parquet(DATA / "analysis" / "df_overall_scores.parquet")
            # We need to map our pattern_code to df_out['pattern']
            # df_out uses 'base_p0' etc.
            df_out = df_out[df_out["judge"] == "gpt52"].copy()
            df_out["pattern_code"] = df_out["pattern"].map(
                {v: k for k, v in PATTERN_TO_EXP.items()}
            )

            # Per-(pattern, qid) join: traj scores aggregated across 3 replicates
            traj_agg = joined.groupby(["pattern_code", "query_id"])[dim_cols].mean().reset_index()
            merged = traj_agg.merge(
                df_out[["pattern_code", "query_id", "overall_score"]],
                on=["pattern_code", "query_id"], how="inner",
            )

            from scipy.stats import spearmanr
            md_lines.append("## Spearman correlation: trajectory dim vs outcome rubric overall_score\n")
            md_lines.append("Per-(pattern, query) cells where both trajectory and outcome scores exist.\n")
            md_lines.append("### Pooled across all instrumented patterns\n")
            md_lines.append("| Trajectory dim | n | Spearman ρ | p-value |")
            md_lines.append("|---|---:|---:|---:|")
            for c in dim_cols:
                if c in merged.columns:
                    sub = merged[[c, "overall_score"]].dropna()
                    if len(sub) >= 4:
                        rho, p = spearmanr(sub[c], sub["overall_score"])
                        md_lines.append(f"| {c.replace('_score','')} | {len(sub)} | "
                                        f"{rho:+.3f} | {p:.3g} |")
                    else:
                        md_lines.append(f"| {c.replace('_score','')} | {len(sub)} | — | — |")
            md_lines.append("")

            md_lines.append("### Per-pattern breakdown (Spearman ρ across 30 query cells per pattern)\n")
            md_lines.append("| Pattern | dim | n | ρ | p |")
            md_lines.append("|---|---|---:|---:|---:|")
            for pat, g in merged.groupby("pattern_code"):
                for c in dim_cols:
                    sub = g[[c, "overall_score"]].dropna()
                    if len(sub) >= 4:
                        rho, p = spearmanr(sub[c], sub["overall_score"])
                        md_lines.append(f"| {pat} | {c.replace('_score','')} | "
                                        f"{len(sub)} | {rho:+.3f} | {p:.3g} |")
            md_lines.append("")

            # Also save merged df for downstream reuse
            merged.to_csv(OUT / "merged_traj_outcome.csv", index=False)

            # Pattern-level (n=5) Spearman: do trajectory means rank-correlate with outcome means?
            md_lines.append("### Pattern-level (n=5): mean trajectory vs mean outcome\n")
            md_lines.append("| Trajectory dim | ρ across 5 patterns | p |")
            md_lines.append("|---|---:|---:|")
            pat_means = merged.groupby("pattern_code").agg(
                **{c: (c, "mean") for c in dim_cols},
                outcome_mean=("overall_score", "mean"),
            ).reset_index()
            for c in dim_cols:
                if len(pat_means) >= 4:
                    rho, p = spearmanr(pat_means[c], pat_means["outcome_mean"])
                    md_lines.append(f"| {c.replace('_score','')} | {rho:+.3f} | {p:.3g} |")
            md_lines.append("")
            pat_means.to_csv(OUT / "pattern_level_means.csv", index=False)
        except Exception as e:
            md_lines.append(f"\n*Spearman computation failed: {e}*\n")
            print(f"  WARN Spearman: {e}")

        (OUT / "per_pattern_summary.md").write_text("\n".join(md_lines))
        # Also persist Spearman in its own file for clarity
        (OUT / "spearman_vs_outcome.md").write_text("\n".join(md_lines))
        print(f"  -> {OUT/'per_pattern_summary.md'}")
        print(f"  -> {OUT/'spearman_vs_outcome.md'}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
