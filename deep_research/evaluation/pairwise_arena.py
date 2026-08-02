"""Arena-style pairwise comparison for report evaluation.

Complements dimension-based scoring with direct A-vs-B judgments.
Supports Elo rating and Bradley-Terry model for ranking.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import structlog

from deep_research.config import DEFAULT_MODEL, JUDGE
from deep_research.tools.llm_caller import LLMCaller

log = structlog.get_logger()


# ── Dimensions used for per-dimension comparison ────────────────────────────

_COMPARISON_DIMENSIONS = [
    "factual_accuracy",
    "coverage",
    "analytical_depth",
    "citation_quality",
    "organisation",
    "instruction_following",
]


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class PairwiseVerdict:
    """Result of a single A-vs-B pairwise comparison."""

    query_id: str
    system_a: str
    system_b: str
    winner: str  # "A", "B", or "tie"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    dimensions_won: dict[str, str]  # dimension -> "A", "B", or "tie"


@dataclass
class ArenaResult:
    """Aggregated arena evaluation result."""

    verdicts: list[PairwiseVerdict]
    elo_ratings: dict[str, float]
    bradley_terry_scores: dict[str, float]
    win_matrix: dict[str, dict[str, int]]
    head_to_head: dict[str, dict[str, dict[str, int]]]
    n_comparisons: int
    systems: list[str]


# ── LLM prompt ──────────────────────────────────────────────────────────────

_PAIRWISE_SYSTEM_PROMPT = """You are an expert research report evaluator. You will compare two research reports (Report A and Report B) written in response to the same research query.

Compare the reports across these 6 dimensions:
1. factual_accuracy — correctness and verifiability of claims
2. coverage — breadth and completeness of topic treatment
3. analytical_depth — depth of analysis, reasoning quality, and insight
4. citation_quality — quality and specificity of source attribution
5. organisation — structure, flow, and readability
6. instruction_following — adherence to the original query requirements

For each dimension, decide which report is better: "A", "B", or "tie".
Then provide an overall winner: "A", "B", or "tie".
Also provide a confidence score from 0.0 to 1.0 for your overall verdict.

Respond with valid JSON only in this exact format:
{
  "overall_winner": "A" | "B" | "tie",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the overall decision",
  "dimensions": {
    "factual_accuracy": "A" | "B" | "tie",
    "coverage": "A" | "B" | "tie",
    "analytical_depth": "A" | "B" | "tie",
    "citation_quality": "A" | "B" | "tie",
    "organisation": "A" | "B" | "tie",
    "instruction_following": "A" | "B" | "tie"
  }
}

Rules:
- Judge solely on the content quality, not length
- Be fair and consistent across dimensions
- A tie is valid when reports are genuinely comparable
- Provide a brief but specific reasoning referencing actual content differences"""


# ── Core comparison function ────────────────────────────────────────────────


async def pairwise_comparison(
    report_a: str,
    report_b: str,
    query_text: str,
    query_id: str,
    system_a: str,
    system_b: str,
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
) -> PairwiseVerdict:
    """Run a single pairwise comparison between two reports.

    Randomly swaps A/B order (deterministic based on inputs) to mitigate
    position bias.  The swap is reversed before returning the verdict.

    Args:
        report_a: Full text of report from system_a.
        report_b: Full text of report from system_b.
        query_text: The original research query.
        query_id: Identifier for the query.
        system_a: Name/label of the system that produced report_a.
        system_b: Name/label of the system that produced report_b.
        llm: LLMCaller instance for making API calls.
        model: Model deployment to use for judging.

    Returns:
        PairwiseVerdict with winner expressed in terms of original system_a/system_b.
    """
    # Deterministic swap based on hash of inputs
    swap_hash = hash(query_id + system_a + system_b) & 0x7FFFFFFF
    swapped = swap_hash % 2 == 1

    if swapped:
        presented_a, presented_b = report_b, report_a
    else:
        presented_a, presented_b = report_a, report_b

    user_msg = (
        f"## Research Query\n{query_text}\n\n"
        f"## Report A\n{presented_a}\n\n"
        f"## Report B\n{presented_b}"
    )

    try:
        result = await llm.complete_json(
            prompt=user_msg,
            model=model,
            system=_PAIRWISE_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=JUDGE.max_tokens,
        )
    except Exception as e:
        log.error(
            "pairwise_comparison_failed",
            query_id=query_id,
            system_a=system_a,
            system_b=system_b,
            error=str(e)[:200],
        )
        # Return a tie verdict on failure
        return PairwiseVerdict(
            query_id=query_id,
            system_a=system_a,
            system_b=system_b,
            winner="tie",
            confidence=0.0,
            reasoning=f"LLM call failed: {str(e)[:100]}",
            dimensions_won={d: "tie" for d in _COMPARISON_DIMENSIONS},
        )

    # Parse the response
    raw_winner = str(result.get("overall_winner", "tie")).upper()
    if raw_winner not in ("A", "B", "TIE"):
        raw_winner = "TIE"

    raw_confidence = result.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = str(result.get("reasoning", ""))

    raw_dims = result.get("dimensions", {})
    dimensions_won: dict[str, str] = {}
    for dim in _COMPARISON_DIMENSIONS:
        dim_val = str(raw_dims.get(dim, "tie")).upper()
        if dim_val not in ("A", "B", "TIE"):
            dim_val = "TIE"
        dimensions_won[dim] = dim_val

    # Un-swap if we swapped the presentation order
    if swapped:
        raw_winner = _swap_label(raw_winner)
        dimensions_won = {d: _swap_label(v) for d, v in dimensions_won.items()}

    # Map A/B back to system names for the winner field
    winner_map = {"A": "A", "B": "B", "TIE": "tie"}
    winner = winner_map.get(raw_winner, "tie")

    log.info(
        "pairwise_verdict",
        query_id=query_id,
        system_a=system_a,
        system_b=system_b,
        winner=winner,
        confidence=f"{confidence:.2f}",
        swapped=swapped,
    )

    return PairwiseVerdict(
        query_id=query_id,
        system_a=system_a,
        system_b=system_b,
        winner=winner,
        confidence=confidence,
        reasoning=reasoning,
        dimensions_won=dimensions_won,
    )


def _swap_label(label: str) -> str:
    """Swap A<->B, leave TIE unchanged."""
    if label == "A":
        return "B"
    elif label == "B":
        return "A"
    return label


# ── Arena runner ────────────────────────────────────────────────────────────


async def run_arena(
    reports_by_system: dict[str, dict[str, str]],
    queries: dict[str, str],
    llm: LLMCaller,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = JUDGE.max_concurrent,
    seed: int = 42,
) -> ArenaResult:
    """Run a full arena evaluation across all system pairs and queries.

    Args:
        reports_by_system: {system_name: {query_id: report_text}}.
        queries: {query_id: query_text}.
        llm: LLMCaller instance.
        model: Model deployment for judging.
        max_concurrent: Maximum concurrent comparisons.
        seed: Random seed for Elo computation order.

    Returns:
        ArenaResult with ratings, win matrix, and all verdicts.
    """
    systems = sorted(reports_by_system.keys())
    if len(systems) < 2:
        log.warning("arena_insufficient_systems", n_systems=len(systems))
        return ArenaResult(
            verdicts=[],
            elo_ratings={s: 1500.0 for s in systems},
            bradley_terry_scores={s: 1.0 / len(systems) for s in systems} if systems else {},
            win_matrix={},
            head_to_head={},
            n_comparisons=0,
            systems=systems,
        )

    # Generate all (system_pair, query) tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks: list[asyncio.Task] = []

    for sys_a, sys_b in combinations(systems, 2):
        for qid, qtext in queries.items():
            report_a = reports_by_system.get(sys_a, {}).get(qid)
            report_b = reports_by_system.get(sys_b, {}).get(qid)
            if report_a is None or report_b is None:
                continue

            async def _bounded_compare(
                ra: str = report_a,
                rb: str = report_b,
                qt: str = qtext,
                qi: str = qid,
                sa: str = sys_a,
                sb: str = sys_b,
            ) -> PairwiseVerdict:
                async with semaphore:
                    return await pairwise_comparison(
                        report_a=ra,
                        report_b=rb,
                        query_text=qt,
                        query_id=qi,
                        system_a=sa,
                        system_b=sb,
                        llm=llm,
                        model=model,
                    )

            tasks.append(asyncio.create_task(_bounded_compare()))

    verdicts: list[PairwiseVerdict] = await asyncio.gather(*tasks)

    # Compute aggregates
    win_mat = _build_win_matrix(verdicts, systems)
    h2h = _build_head_to_head(verdicts, systems)
    elo = compute_elo_ratings(verdicts, systems, seed=seed)
    bt = bradley_terry_scores(verdicts, systems)

    log.info(
        "arena_complete",
        n_systems=len(systems),
        n_comparisons=len(verdicts),
        elo_top=max(elo, key=elo.get) if elo else None,
    )

    return ArenaResult(
        verdicts=verdicts,
        elo_ratings=elo,
        bradley_terry_scores=bt,
        win_matrix=win_mat,
        head_to_head=h2h,
        n_comparisons=len(verdicts),
        systems=systems,
    )


def _build_win_matrix(
    verdicts: list[PairwiseVerdict], systems: list[str]
) -> dict[str, dict[str, int]]:
    """Build a win count matrix: win_matrix[winner][loser] = count."""
    matrix: dict[str, dict[str, int]] = {s: {t: 0 for t in systems} for s in systems}
    for v in verdicts:
        if v.winner == "A":
            matrix[v.system_a][v.system_b] += 1
        elif v.winner == "B":
            matrix[v.system_b][v.system_a] += 1
        # Ties are not counted in win matrix
    return matrix


def _build_head_to_head(
    verdicts: list[PairwiseVerdict], systems: list[str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Build detailed head-to-head records.

    h2h[sys_a][sys_b] = {"wins": n, "losses": n, "ties": n}
    """
    h2h: dict[str, dict[str, dict[str, int]]] = {}
    for s in systems:
        h2h[s] = {}
        for t in systems:
            if s != t:
                h2h[s][t] = {"wins": 0, "losses": 0, "ties": 0}

    for v in verdicts:
        a, b = v.system_a, v.system_b
        if a not in h2h or b not in h2h:
            continue
        if v.winner == "A":
            h2h[a][b]["wins"] += 1
            h2h[b][a]["losses"] += 1
        elif v.winner == "B":
            h2h[b][a]["wins"] += 1
            h2h[a][b]["losses"] += 1
        else:
            h2h[a][b]["ties"] += 1
            h2h[b][a]["ties"] += 1

    return h2h


# ── Elo rating computation ──────────────────────────────────────────────────


def compute_elo_ratings(
    verdicts: list[PairwiseVerdict],
    systems: list[str],
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings from pairwise verdicts.

    Standard Elo: E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Verdicts are processed in random order (seeded) to avoid order bias.

    Args:
        verdicts: List of pairwise comparison results.
        systems: List of all system names.
        initial_rating: Starting Elo for all systems (default 1500).
        k_factor: K-factor for rating updates.
        seed: Random seed for verdict ordering.

    Returns:
        Dict mapping system name to Elo rating.
    """
    if not verdicts:
        return {s: initial_rating for s in systems}

    ratings: dict[str, float] = {s: initial_rating for s in systems}

    # Shuffle verdicts with seed to avoid order bias
    rng = random.Random(seed)
    shuffled = list(verdicts)
    rng.shuffle(shuffled)

    for v in shuffled:
        r_a = ratings.get(v.system_a, initial_rating)
        r_b = ratings.get(v.system_b, initial_rating)

        # Expected scores
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        e_b = 1.0 - e_a

        # Actual scores
        if v.winner == "A":
            s_a, s_b = 1.0, 0.0
        elif v.winner == "B":
            s_a, s_b = 0.0, 1.0
        else:  # tie
            s_a, s_b = 0.5, 0.5

        # Update ratings
        ratings[v.system_a] = r_a + k_factor * (s_a - e_a)
        ratings[v.system_b] = r_b + k_factor * (s_b - e_b)

    return ratings


# ── Bradley-Terry model ─────────────────────────────────────────────────────


def bradley_terry_scores(
    verdicts: list[PairwiseVerdict],
    systems: list[str],
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Compute Bradley-Terry model scores via MLE iteration.

    p_i = wins_i / sum_j(n_ij / (p_i + p_j))

    Ties count as 0.5 win for each side.

    Args:
        verdicts: List of pairwise comparison results.
        systems: List of all system names.
        max_iterations: Maximum iterations for convergence.
        tolerance: Convergence tolerance.

    Returns:
        Dict mapping system name to probability score (sums to 1.0).
    """
    if not systems:
        return {}

    n = len(systems)
    if n == 1:
        return {systems[0]: 1.0}

    sys_idx = {s: i for i, s in enumerate(systems)}

    # Build win counts and match counts
    wins = [0.0] * n
    matches = [[0.0] * n for _ in range(n)]

    for v in verdicts:
        i = sys_idx.get(v.system_a)
        j = sys_idx.get(v.system_b)
        if i is None or j is None:
            continue

        matches[i][j] += 1.0
        matches[j][i] += 1.0

        if v.winner == "A":
            wins[i] += 1.0
        elif v.winner == "B":
            wins[j] += 1.0
        else:  # tie
            wins[i] += 0.5
            wins[j] += 0.5

    # Check if any system has matches
    total_matches = sum(sum(row) for row in matches)
    if total_matches == 0:
        # No matches at all — return uniform
        return {s: 1.0 / n for s in systems}

    # Initialize scores uniformly
    p = [1.0 / n] * n

    for iteration in range(max_iterations):
        p_new = [0.0] * n
        for i in range(n):
            if wins[i] == 0:
                p_new[i] = tolerance  # Avoid zero
                continue
            denominator = 0.0
            for j in range(n):
                if i == j or matches[i][j] == 0:
                    continue
                denominator += matches[i][j] / (p[i] + p[j])
            if denominator > 0:
                p_new[i] = wins[i] / denominator
            else:
                p_new[i] = p[i]

        # Normalize
        total = sum(p_new)
        if total > 0:
            p_new = [x / total for x in p_new]
        else:
            p_new = [1.0 / n] * n

        # Check convergence
        max_diff = max(abs(p_new[i] - p[i]) for i in range(n))
        p = p_new

        if max_diff < tolerance:
            log.debug(
                "bradley_terry_converged",
                iterations=iteration + 1,
                max_diff=f"{max_diff:.2e}",
            )
            break
    else:
        log.warning(
            "bradley_terry_max_iterations",
            max_iterations=max_iterations,
            max_diff=f"{max_diff:.2e}",
        )

    return {systems[i]: p[i] for i in range(n)}


# ── Transitivity check ──────────────────────────────────────────────────────


def transitivity_check(
    verdicts: list[PairwiseVerdict],
    systems: list[str],
) -> dict[str, Any]:
    """Check transitivity of pairwise preferences.

    For all triplets (A, B, C): if A > B and B > C, check whether A > C.
    A violation is when A > B and B > C but C >= A.

    Args:
        verdicts: List of pairwise comparison results.
        systems: List of all system names.

    Returns:
        Dict with keys: violation_rate, n_violations, n_testable_triplets,
        violations (list of violating triplet tuples).
    """
    if len(systems) < 3:
        return {
            "violation_rate": 0.0,
            "n_violations": 0,
            "n_testable_triplets": 0,
            "violations": [],
        }

    # Build aggregate preference: pref[a][b] > 0 means a beats b more often
    pref: dict[str, dict[str, int]] = {s: {t: 0 for t in systems} for s in systems}
    for v in verdicts:
        if v.winner == "A":
            pref[v.system_a][v.system_b] += 1
            pref[v.system_b][v.system_a] -= 1
        elif v.winner == "B":
            pref[v.system_b][v.system_a] += 1
            pref[v.system_a][v.system_b] -= 1
        # Ties don't affect preference direction

    n_violations = 0
    n_testable = 0
    violation_details: list[tuple[str, str, str]] = []

    for a, b, c in _ordered_triplets(systems):
        # Check: if a > b and b > c, is a > c?
        a_beats_b = pref[a][b] > 0
        b_beats_c = pref[b][c] > 0

        if a_beats_b and b_beats_c:
            n_testable += 1
            a_beats_c = pref[a][c] > 0
            if not a_beats_c:
                n_violations += 1
                violation_details.append((a, b, c))

    violation_rate = n_violations / n_testable if n_testable > 0 else 0.0

    log.info(
        "transitivity_check",
        n_violations=n_violations,
        n_testable=n_testable,
        violation_rate=f"{violation_rate:.3f}",
    )

    return {
        "violation_rate": violation_rate,
        "n_violations": n_violations,
        "n_testable_triplets": n_testable,
        "violations": violation_details,
    }


def _ordered_triplets(systems: list[str]):
    """Generate all ordered triplets (a, b, c) from systems.

    For each combination of 3, yields all 6 permutations so that all
    transitive chains are checked.
    """
    from itertools import permutations

    for trio in combinations(systems, 3):
        yield from permutations(trio)
