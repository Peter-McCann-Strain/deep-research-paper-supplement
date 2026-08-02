"""Side-by-side comparison table generation."""

from __future__ import annotations

from typing import Dict, List

from deep_research.evaluation.metrics import EvalResult


def generate_comparison(results: List[EvalResult]) -> str:
    """Generate a markdown comparison table from evaluation results."""
    if not results:
        return "No results to compare."

    # Group by pattern
    by_pattern: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_pattern.setdefault(r.pattern_name, []).append(r)

    # Group by query
    by_query: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_query.setdefault(r.query_id, []).append(r)

    lines = ["# Deep Research Pattern Comparison\n"]

    # ── Summary table ────────────────────────────────────────────────────
    lines.append("## Summary by Pattern\n")
    lines.append("| Pattern | Avg Coverage | Avg Citations | Avg Words | Avg Cost | Avg Latency | Avg Overall |")
    lines.append("|---------|-------------|---------------|-----------|----------|-------------|-------------|")

    for pname, pres in sorted(by_pattern.items()):
        n = len(pres)
        avg_cov = sum(r.coverage_score for r in pres) / n
        avg_cit = sum(r.citation_count for r in pres) / n
        avg_words = sum(r.report_length_words for r in pres) / n
        avg_cost = sum(r.cost_usd for r in pres) / n
        avg_lat = sum(r.latency_seconds for r in pres) / n
        avg_overall = sum(r.overall_score for r in pres) / n
        lines.append(
            f"| {pname} | {avg_cov:.1%} | {avg_cit:.0f} | {avg_words:.0f} "
            f"| ${avg_cost:.3f} | {avg_lat:.0f}s | {avg_overall:.2f} |"
        )

    # ── Detail table per query ───────────────────────────────────────────
    lines.append("\n## Results by Query\n")
    for qid, qres in sorted(by_query.items()):
        lines.append(f"### {qid}\n")
        lines.append("| Pattern | Coverage | Citations | Words | Cost | Latency | Overall |")
        lines.append("|---------|----------|-----------|-------|------|---------|---------|")
        for r in sorted(qres, key=lambda x: x.overall_score, reverse=True):
            lines.append(
                f"| {r.pattern_name} | {r.coverage_score:.1%} | {r.citation_count} "
                f"| {r.report_length_words} | ${r.cost_usd:.4f} "
                f"| {r.latency_seconds:.0f}s | {r.overall_score:.2f} |"
            )
        lines.append("")

    # ── Coverage detail ──────────────────────────────────────────────────
    lines.append("\n## Coverage Detail\n")
    for r in results:
        if r.coverage_details:
            lines.append(f"**{r.pattern_name} × {r.query_id}** ({r.coverage_score:.1%})")
            for element, found in r.coverage_details.items():
                mark = "+" if found else "-"
                lines.append(f"  {mark} {element}")
            lines.append("")

    return "\n".join(lines)
