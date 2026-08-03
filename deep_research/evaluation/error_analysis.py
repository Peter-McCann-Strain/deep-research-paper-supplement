"""Error categorization and failure mode analysis.

Categorizes errors found in research reports to understand
where each pattern fails and why.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Any


# Valid categories and severities
VALID_CATEGORIES = frozenset({
    "hallucination",
    "citation_fabrication",
    "topic_drift",
    "factual_error",
    "missing_coverage",
    "synthesis_failure",
    "source_quality",
    "attribution_error",
})

VALID_SEVERITIES = frozenset({"minor", "moderate", "critical"})


@dataclass
class ErrorInstance:
    """A single categorized error in a report."""
    category: str       # "hallucination", "citation_fabrication", "topic_drift",
                        # "factual_error", "missing_coverage", "synthesis_failure",
                        # "source_quality", "attribution_error"
    severity: str       # "minor", "moderate", "critical"
    description: str
    section: str
    evidence: str = ""


@dataclass
class ErrorProfile:
    """Error profile for a single report."""
    pattern: str
    query_id: str
    errors: list[ErrorInstance]
    total_errors: int
    by_category: dict[str, int]
    by_severity: dict[str, int]
    critical_count: int


@dataclass
class PatternErrorProfile:
    """Aggregated error profile across all reports for a pattern."""
    pattern: str
    n_reports: int
    avg_errors_per_report: float
    category_distribution: dict[str, float]  # category -> proportion
    severity_distribution: dict[str, float]
    most_common_errors: list[tuple[str, int]]  # (category, count)
    failure_modes: list[str]  # narrative descriptions


async def categorize_errors(
    report_text: str,
    pattern: str,
    query_id: str,
    judge_verdicts: list | None = None,
    citation_verification: Any | None = None,
    llm_caller: Any | None = None,
) -> ErrorProfile:
    """Categorize all errors found in a report.

    Uses judge verdicts (failed criteria) and citation verification
    results to identify and categorize errors. Optionally uses LLM
    for deeper analysis.

    Args:
        report_text: the full text of the research report.
        pattern: pattern name (e.g. "p4_perspective_storm").
        query_id: identifier for the query that produced this report.
        judge_verdicts: list of dicts with keys "dimension", "verdict",
            "reasoning". Verdicts with verdict="NOT_SATISFIED" indicate failures.
        citation_verification: optional dict with keys "flagged_claims",
            "accuracy_rate", etc.
        llm_caller: optional LLM caller for deeper analysis (not used in
            basic mode).

    Returns:
        ErrorProfile with categorized errors.
    """
    errors: list[ErrorInstance] = []

    # 1. Errors from judge verdicts (failed criteria)
    if judge_verdicts:
        for verdict in judge_verdicts:
            if verdict.get("verdict") != "NOT_SATISFIED":
                continue

            dimension = verdict.get("dimension", "unknown")
            reasoning = verdict.get("reasoning", "")

            category, severity = _map_dimension_to_error(dimension, reasoning)
            errors.append(ErrorInstance(
                category=category,
                severity=severity,
                description=f"Judge: {dimension} NOT_SATISFIED -- {reasoning[:200]}",
                section=_guess_section(reasoning, report_text),
                evidence=reasoning[:300],
            ))

    # 2. Errors from citation verification
    if citation_verification:
        flagged = citation_verification.get("flagged_claims", [])
        accuracy_rate = citation_verification.get("accuracy_rate", 1.0)

        for claim in flagged:
            claim_text = claim if isinstance(claim, str) else claim.get("claim", str(claim))
            errors.append(ErrorInstance(
                category="citation_fabrication",
                severity="critical" if accuracy_rate < 0.5 else "moderate",
                description=f"Citation flagged: {claim_text[:200]}",
                section="References",
                evidence=claim_text[:300],
            ))

    # 3. Heuristic checks on report text
    heuristic_errors = _heuristic_error_detection(report_text)
    errors.extend(heuristic_errors)

    # Build profile
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    critical_count = 0

    for err in errors:
        by_category[err.category] = by_category.get(err.category, 0) + 1
        by_severity[err.severity] = by_severity.get(err.severity, 0) + 1
        if err.severity == "critical":
            critical_count += 1

    return ErrorProfile(
        pattern=pattern,
        query_id=query_id,
        errors=errors,
        total_errors=len(errors),
        by_category=by_category,
        by_severity=by_severity,
        critical_count=critical_count,
    )


def _map_dimension_to_error(dimension: str, reasoning: str) -> tuple[str, str]:
    """Map a judge dimension to an error category and severity."""
    dim_lower = dimension.lower()

    if "factual" in dim_lower or "accuracy" in dim_lower:
        return "factual_error", "critical"
    elif "citation" in dim_lower:
        return "citation_fabrication", "critical"
    elif "attribution" in dim_lower:
        return "attribution_error", "moderate"
    elif "information_recall" in dim_lower or "recall" in dim_lower:
        return "missing_coverage", "moderate"
    elif "coverage" in dim_lower:
        return "missing_coverage", "moderate"
    elif "logical" in dim_lower or "coherence" in dim_lower:
        return "synthesis_failure", "moderate"
    elif "analytical" in dim_lower or "depth" in dim_lower:
        return "synthesis_failure", "moderate"
    elif "organisation" in dim_lower or "organization" in dim_lower:
        return "synthesis_failure", "minor"
    elif "instruction" in dim_lower:
        return "topic_drift", "minor"
    else:
        return "factual_error", "moderate"


def _guess_section(reasoning: str, report_text: str) -> str:
    """Try to guess which section of the report an error relates to."""
    # Extract section headers from the report
    headers = re.findall(r"^##\s+(.+)$", report_text, re.MULTILINE)

    if not headers:
        return "Unknown"

    # Check if any section name appears in the reasoning
    reasoning_lower = reasoning.lower()
    for header in headers:
        if header.lower() in reasoning_lower:
            return header

    return headers[0] if headers else "Unknown"


def _heuristic_error_detection(report_text: str) -> list[ErrorInstance]:
    """Run simple heuristic checks for common error patterns."""
    errors: list[ErrorInstance] = []

    # Check for dangling citation references (e.g. [99] when only 5 sources)
    citation_refs = set(int(m) for m in re.findall(r"\[(\d+)\]", report_text))
    ref_section = report_text.split("## References")[-1] if "## References" in report_text else ""
    ref_entries = re.findall(r"\[(\d+)\]", ref_section)
    defined_refs = set(int(r) for r in ref_entries) if ref_entries else set()

    if citation_refs and defined_refs:
        dangling = citation_refs - defined_refs
        for ref_num in sorted(dangling):
            if ref_num > 0:  # Ignore [0] which might be formatting
                errors.append(ErrorInstance(
                    category="attribution_error",
                    severity="minor",
                    description=f"Citation [{ref_num}] referenced but not defined in References",
                    section="References",
                    evidence=f"[{ref_num}]",
                ))

    # Check for very short report (likely synthesis failure)
    word_count = len(report_text.split())
    if word_count < 500:
        errors.append(ErrorInstance(
            category="synthesis_failure",
            severity="critical",
            description=f"Report is very short ({word_count} words), suggesting synthesis failure",
            section="Overall",
            evidence=f"Word count: {word_count}",
        ))

    return errors


def aggregate_error_profiles(
    profiles: list[ErrorProfile],
) -> dict[str, PatternErrorProfile]:
    """Aggregate error profiles by pattern.

    Groups ErrorProfiles by pattern name and computes aggregate
    statistics including category distributions, severity distributions,
    most common errors, and narrative failure mode descriptions.

    Args:
        profiles: list of ErrorProfile objects from individual reports.

    Returns:
        dict mapping pattern name to PatternErrorProfile.
    """
    # Group by pattern
    by_pattern: dict[str, list[ErrorProfile]] = {}
    for profile in profiles:
        by_pattern.setdefault(profile.pattern, []).append(profile)

    result: dict[str, PatternErrorProfile] = {}

    for pattern, pattern_profiles in by_pattern.items():
        n_reports = len(pattern_profiles)
        total_errors = sum(p.total_errors for p in pattern_profiles)
        avg_errors = total_errors / n_reports if n_reports > 0 else 0.0

        # Aggregate category counts
        category_counts: Counter = Counter()
        severity_counts: Counter = Counter()
        for profile in pattern_profiles:
            for cat, count in profile.by_category.items():
                category_counts[cat] += count
            for sev, count in profile.by_severity.items():
                severity_counts[sev] += count

        # Compute distributions (proportions)
        category_distribution: dict[str, float] = {}
        if total_errors > 0:
            for cat, count in category_counts.items():
                category_distribution[cat] = count / total_errors

        severity_distribution: dict[str, float] = {}
        if total_errors > 0:
            for sev, count in severity_counts.items():
                severity_distribution[sev] = count / total_errors

        # Most common errors
        most_common = category_counts.most_common()

        # Generate narrative failure modes
        failure_modes = _generate_failure_modes(
            pattern, category_counts, severity_counts, avg_errors, n_reports
        )

        result[pattern] = PatternErrorProfile(
            pattern=pattern,
            n_reports=n_reports,
            avg_errors_per_report=avg_errors,
            category_distribution=category_distribution,
            severity_distribution=severity_distribution,
            most_common_errors=most_common,
            failure_modes=failure_modes,
        )

    return result


def _generate_failure_modes(
    pattern: str,
    category_counts: Counter,
    severity_counts: Counter,
    avg_errors: float,
    n_reports: int,
) -> list[str]:
    """Generate narrative descriptions of failure modes."""
    modes: list[str] = []

    total = sum(category_counts.values())
    if total == 0:
        modes.append(f"{pattern}: No errors detected across {n_reports} reports.")
        return modes

    # Identify dominant failure categories (>25% of errors)
    for cat, count in category_counts.most_common():
        proportion = count / total
        if proportion >= 0.25:
            modes.append(
                f"{pattern}: {cat} is a dominant failure mode "
                f"({count} instances, {proportion:.0%} of all errors)."
            )

    # Note critical error rate
    critical = severity_counts.get("critical", 0)
    if critical > 0:
        modes.append(
            f"{pattern}: {critical} critical errors across {n_reports} reports "
            f"({critical / n_reports:.1f} per report)."
        )

    # High error rate
    if avg_errors > 5:
        modes.append(
            f"{pattern}: High error rate ({avg_errors:.1f} errors per report)."
        )

    if not modes:
        modes.append(
            f"{pattern}: {total} errors across {n_reports} reports "
            f"({avg_errors:.1f} per report), no dominant failure mode."
        )

    return modes


def generate_error_report(
    pattern_profiles: dict[str, PatternErrorProfile],
) -> str:
    """Generate markdown error analysis report.

    Args:
        pattern_profiles: dict mapping pattern name to PatternErrorProfile,
            as returned by aggregate_error_profiles().

    Returns:
        Markdown-formatted error analysis report.
    """
    lines = ["# Error Analysis Report\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Pattern | Reports | Avg Errors | Most Common | Critical |")
    lines.append("|---|---|---|---|---|")

    for pattern in sorted(pattern_profiles.keys()):
        profile = pattern_profiles[pattern]
        most_common = profile.most_common_errors[0][0] if profile.most_common_errors else "N/A"
        # Count total criticals
        critical_total = 0
        for cat, count in profile.most_common_errors:
            # We need to go back to severity -- use the severity distribution
            pass
        # Use severity distribution
        total_errors_count = int(profile.avg_errors_per_report * profile.n_reports)
        critical_count = int(profile.severity_distribution.get("critical", 0) * total_errors_count)

        lines.append(
            f"| {pattern} | {profile.n_reports} | {profile.avg_errors_per_report:.1f} "
            f"| {most_common} | {critical_count} |"
        )
    lines.append("")

    # Per-pattern details
    for pattern in sorted(pattern_profiles.keys()):
        profile = pattern_profiles[pattern]
        lines.append(f"## {pattern}\n")

        # Category breakdown
        if profile.category_distribution:
            lines.append("### Error Categories\n")
            lines.append("| Category | Count | Proportion |")
            lines.append("|---|---|---|")
            total_errors_count = int(profile.avg_errors_per_report * profile.n_reports)
            for cat, count in profile.most_common_errors:
                proportion = profile.category_distribution.get(cat, 0)
                lines.append(f"| {cat} | {count} | {proportion:.0%} |")
            lines.append("")

        # Severity breakdown
        if profile.severity_distribution:
            lines.append("### Severity Distribution\n")
            lines.append("| Severity | Proportion |")
            lines.append("|---|---|")
            for sev in ["critical", "moderate", "minor"]:
                prop = profile.severity_distribution.get(sev, 0)
                if prop > 0:
                    lines.append(f"| {sev} | {prop:.0%} |")
            lines.append("")

        # Failure modes
        if profile.failure_modes:
            lines.append("### Failure Modes\n")
            for mode in profile.failure_modes:
                lines.append(f"- {mode}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failure clustering
# ---------------------------------------------------------------------------


def failure_clustering(
    error_profiles: list,  # list of PatternErrorProfile
    n_clusters: int = 4,
) -> dict:
    """Cluster failure patterns by error category co-occurrence.

    Builds a feature matrix from error category distributions and groups
    patterns using agglomerative clustering with a simple distance matrix
    (no sklearn dependency). Also computes pairwise co-occurrence rates
    between error categories across all patterns.

    Args:
        error_profiles: list of PatternErrorProfile objects. Each must have
            a ``category_distribution`` dict mapping category names to
            proportions and a ``pattern`` attribute.
        n_clusters: desired number of clusters (capped to number of profiles).

    Returns:
        dict with keys:
            - clusters: dict mapping cluster_id (int) to list of pattern names.
            - cross_pattern_failures: list of error categories that appear in
              every pattern's distribution.
            - pattern_specific_failures: dict mapping pattern name to list of
              categories unique to that pattern.
            - category_correlations: dict mapping (cat_a, cat_b) tuples to
              their co-occurrence rate across patterns.
    """
    if not error_profiles:
        return {
            "clusters": {},
            "cross_pattern_failures": [],
            "pattern_specific_failures": {},
            "category_correlations": {},
        }

    # Collect all category names across profiles
    all_categories: set[str] = set()
    for profile in error_profiles:
        all_categories.update(profile.category_distribution.keys())
    categories = sorted(all_categories)

    n = len(error_profiles)
    m = len(categories)

    # Build feature matrix: rows = patterns, columns = category proportions
    feature_matrix: list[list[float]] = []
    pattern_names: list[str] = []
    for profile in error_profiles:
        row = [profile.category_distribution.get(cat, 0.0) for cat in categories]
        feature_matrix.append(row)
        pattern_names.append(profile.pattern)

    # --- Agglomerative clustering (manual, single-linkage) ---
    # Compute pairwise Euclidean distance matrix
    def _euclidean(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    # Each element starts as its own cluster
    cluster_map: dict[int, list[int]] = {i: [i] for i in range(n)}
    active_clusters = set(range(n))

    effective_n_clusters = min(n_clusters, n)

    while len(active_clusters) > effective_n_clusters:
        # Find the two closest clusters (single-linkage: min distance between members)
        best_dist = float("inf")
        best_pair = (-1, -1)

        active_list = sorted(active_clusters)
        for idx_a in range(len(active_list)):
            for idx_b in range(idx_a + 1, len(active_list)):
                ca = active_list[idx_a]
                cb = active_list[idx_b]
                # Single-linkage: minimum distance between any pair of members
                min_dist = float("inf")
                for member_a in cluster_map[ca]:
                    for member_b in cluster_map[cb]:
                        d = _euclidean(feature_matrix[member_a], feature_matrix[member_b])
                        if d < min_dist:
                            min_dist = d
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_pair = (ca, cb)

        # Merge the two closest clusters
        ca, cb = best_pair
        cluster_map[ca] = cluster_map[ca] + cluster_map[cb]
        del cluster_map[cb]
        active_clusters.discard(cb)

    # Build clusters result: cluster_id -> list of pattern names
    clusters: dict[int, list[str]] = {}
    for cluster_id, (_, members) in enumerate(sorted(cluster_map.items())):
        clusters[cluster_id] = [pattern_names[i] for i in members]

    # --- Cross-pattern failures ---
    # Categories present in every pattern's distribution
    pattern_category_sets = [
        set(profile.category_distribution.keys())
        for profile in error_profiles
    ]
    if pattern_category_sets:
        cross_pattern = set.intersection(*pattern_category_sets)
    else:
        cross_pattern = set()
    cross_pattern_failures = sorted(cross_pattern)

    # --- Pattern-specific failures ---
    # Categories that appear only in one pattern
    all_pattern_cats: dict[str, set[str]] = {}
    for profile in error_profiles:
        all_pattern_cats[profile.pattern] = set(profile.category_distribution.keys())

    union_others: dict[str, set[str]] = {}
    for name in pattern_names:
        others = set()
        for other_name, cats in all_pattern_cats.items():
            if other_name != name:
                others |= cats
        union_others[name] = others

    pattern_specific_failures: dict[str, list[str]] = {}
    for name in pattern_names:
        unique = all_pattern_cats[name] - union_others[name]
        if unique:
            pattern_specific_failures[name] = sorted(unique)

    # --- Category correlations (co-occurrence) ---
    # For each pair of categories, how often do they co-occur across patterns?
    # Co-occurrence = fraction of patterns that have both categories present.
    category_correlations: dict[tuple[str, str], float] = {}
    for i_cat in range(len(categories)):
        for j_cat in range(i_cat + 1, len(categories)):
            cat_a = categories[i_cat]
            cat_b = categories[j_cat]
            co_count = 0
            for profile in error_profiles:
                has_a = profile.category_distribution.get(cat_a, 0.0) > 0
                has_b = profile.category_distribution.get(cat_b, 0.0) > 0
                if has_a and has_b:
                    co_count += 1
            rate = co_count / n if n > 0 else 0.0
            category_correlations[(cat_a, cat_b)] = rate

    return {
        "clusters": clusters,
        "cross_pattern_failures": cross_pattern_failures,
        "pattern_specific_failures": pattern_specific_failures,
        "category_correlations": category_correlations,
    }
