"""Ablation study framework for isolating component contributions.

Provides a registry of ablation configurations and a runner that
executes ablated pattern variants on a representative query subset.

Each ablation disables or modifies a single component of a pattern
to measure its contribution to overall quality.
"""

import asyncio
import json
import structlog
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Callable, Awaitable

from deep_research.types import ResearchReport

logger = structlog.get_logger()


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    id: str                    # e.g., "p4_no_conversations"
    base_pattern: str          # e.g., "p4_perspective_storm"
    description: str
    component_removed: str     # name of the disabled component
    modification: dict = field(default_factory=dict)  # kwargs overrides
    expected_effect: str = ""  # hypothesis

    @property
    def display_name(self) -> str:
        return f"{self.base_pattern} - {self.component_removed}"


@dataclass
class AblationResult:
    """Result of a single ablation run."""
    ablation_id: str
    query_id: str
    base_pattern: str
    component_removed: str
    status: str           # "success", "error"
    report_text: str = ""
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    error_message: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AblationComparison:
    """Comparison between base pattern and ablated variant."""
    ablation_id: str
    base_pattern: str
    component_removed: str
    description: str
    expected_effect: str
    # Scores (from judge)
    base_scores: list[float] = field(default_factory=list)
    ablated_scores: list[float] = field(default_factory=list)
    # Computed
    base_mean: float = 0.0
    ablated_mean: float = 0.0
    score_delta: float = 0.0       # base - ablated (positive = component helps)
    relative_change: float = 0.0    # (base - ablated) / base * 100
    is_significant: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    effect_label: str = "negligible"


# ===== ABLATION REGISTRY =====

ABLATION_REGISTRY: list[AblationConfig] = [
    # P4: Perspective STORM ablations
    AblationConfig(
        id="p4_no_conversations",
        base_pattern="p4_perspective_storm",
        description="Skip pairwise expert conversations, go directly from search to synthesis",
        component_removed="conversation_sim",
        modification={"skip_conversations": True},
        expected_effect="Lower analytical depth from loss of multi-perspective dialogue",
    ),
    AblationConfig(
        id="p4_no_triangulation",
        base_pattern="p4_perspective_storm",
        description="Skip claim triangulation, include all claims regardless of cross-perspective agreement",
        component_removed="triangulator",
        modification={"skip_triangulation": True},
        expected_effect="More factual errors from unverified claims, possibly broader coverage",
    ),
    AblationConfig(
        id="p4_fixed_perspectives",
        base_pattern="p4_perspective_storm",
        description="Use 3 fixed generic perspectives instead of LLM-discovered ones",
        component_removed="perspective_discovery",
        modification={"fixed_perspectives": True, "n_perspectives": 3},
        expected_effect="Less diverse source retrieval and analysis angles",
    ),

    # P3: MERIDIAN ablations
    AblationConfig(
        id="p3_no_quality_eval",
        base_pattern="p3_meridian",
        description="Skip quality evaluation and revision, take first draft",
        component_removed="quality_evaluator",
        modification={"skip_evaluation": True},
        expected_effect="Lower organization and potentially lower analytical depth",
    ),
    AblationConfig(
        id="p3_no_topic_mining",
        base_pattern="p3_meridian",
        description="Skip topic miner, pass extractions directly to writer",
        component_removed="topic_miner",
        modification={"skip_topic_mining": True},
        expected_effect="Less structured analysis, lower coherence",
    ),

    # P5: Hierarchical W&D ablations
    AblationConfig(
        id="p5_fixed_width",
        base_pattern="p5_hierarchical_wd",
        description="Fixed width=2 throughout instead of decaying W(t) schedule",
        component_removed="wd_schedule_decay",
        modification={"w_0": 2, "alpha": 1.0},
        expected_effect="Less initial breadth, more uniform resource allocation",
    ),
    AblationConfig(
        id="p5_no_meta_eval",
        base_pattern="p5_hierarchical_wd",
        description="Skip meta-evaluation and budget rebalancing",
        component_removed="meta_evaluator",
        modification={"skip_meta_eval": True},
        expected_effect="No adaptive gap-filling, possibly incomplete coverage",
    ),
    AblationConfig(
        id="p5_no_citation_verify",
        base_pattern="p5_hierarchical_wd",
        description="Skip internal citation spot-check verification",
        component_removed="citation_verifier",
        modification={"skip_citation_verify": True},
        expected_effect="Slightly lower citation quality, faster execution",
    ),

    # P2: Supervisor Parallel ablations
    AblationConfig(
        id="p2_sequential_workers",
        base_pattern="p2_supervisor_parallel",
        description="Process sub-topics sequentially instead of parallel dispatch",
        component_removed="parallel_dispatch",
        modification={"max_workers": 1},
        expected_effect="Slower but possibly more coherent (no aggregation conflicts)",
    ),
    AblationConfig(
        id="p2_no_quality_gate",
        base_pattern="p2_supervisor_parallel",
        description="Skip quality gate evaluation and gap-fill",
        component_removed="quality_gate",
        modification={"skip_quality_gate": True},
        expected_effect="Lower coverage from no gap-filling",
    ),

    # P1: Iterative RAG ablations
    AblationConfig(
        id="p1_single_iteration",
        base_pattern="p1_iterative_rag",
        description="Single retrieval pass with no reflection loop",
        component_removed="reflection_loop",
        modification={"max_iterations": 1},
        expected_effect="Should approximate P0 performance, demonstrating reflection value",
    ),

    # P6: Reactive Interleaved ablations
    AblationConfig(
        id="p6_no_reflect",
        base_pattern="p6_reactive_interleaved",
        description="Skip self-reflection action, agent only searches and drafts",
        component_removed="reflect_action",
        modification={"skip_reflect": True},
        expected_effect="Less targeted gap-filling, potentially lower coverage",
    ),
    AblationConfig(
        id="p6_limited_iterations",
        base_pattern="p6_reactive_interleaved",
        description="Cap at 5 iterations instead of default 15",
        component_removed="extended_iterations",
        modification={"max_iterations": 5},
        expected_effect="Less thorough research, testing if early iterations capture most value",
    ),

    # P7: Graph Decomposition ablations
    AblationConfig(
        id="p7_flat_only",
        base_pattern="p7_graph_decomposition",
        description="Max depth 1 — flat decomposition like P1, no dynamic expansion",
        component_removed="graph_expansion",
        modification={"max_depth": 1, "skip_expansion": True},
        expected_effect="Degrades to flat decomposition, tests value of adaptive depth",
    ),
    AblationConfig(
        id="p7_no_expansion",
        base_pattern="p7_graph_decomposition",
        description="Execute root nodes but don't spawn children from results",
        component_removed="dynamic_expansion",
        modification={"skip_expansion": True},
        expected_effect="No adaptive depth, tests whether dynamic spawning adds value",
    ),

    # P8: Beam Search ablations
    AblationConfig(
        id="p8_narrow_beam",
        base_pattern="p8_beam_search",
        description="Narrow beam width (keep only top 2 directions)",
        component_removed="beam_diversity",
        modification={"beam_width": 2},
        expected_effect="Less diverse exploration, tests optimal beam width",
    ),
    AblationConfig(
        id="p8_single_round",
        base_pattern="p8_beam_search",
        description="Single selection round instead of two",
        component_removed="second_selection",
        modification={"skip_second_selection": True},
        expected_effect="Less refined selection, tests value of iterative pruning",
    ),
    AblationConfig(
        id="p8_wide_beam",
        base_pattern="p8_beam_search",
        description="Wide initial exploration (10 hypotheses, beam width 5)",
        component_removed="narrow_exploration",
        modification={"n_hypotheses": 10, "beam_width": 5},
        expected_effect="More diverse but potentially less focused research",
    ),

    # ===== PARAMETER SWEEP CONFIGS =====
    # These test scaling UP parameters to find where more budget/capacity helps.

    # ── P1: Reflection loop sweep ──────────────────────────────────────────
    AblationConfig(
        id="p1_more_reflection",
        base_pattern="p1_iterative_rag",
        description="5 reflection loops instead of 3 — tests value of extended iteration",
        component_removed="reflection_cap",
        modification={"max_iterations": 5},
        expected_effect="More refinement cycles may improve quality on complex queries",
    ),

    # ── P4: Perspective count sweep ────────────────────────────────────────
    AblationConfig(
        id="p4_7_perspectives",
        base_pattern="p4_perspective_storm",
        description="7 perspectives instead of 5 — more diverse viewpoints",
        component_removed="perspective_cap",
        modification={"n_perspectives": 7},
        expected_effect="Broader coverage from more viewpoints, higher conversation cost",
    ),

    # ── P5: Width-depth schedule sweeps ────────────────────────────────────
    AblationConfig(
        id="p5_wide_start",
        base_pattern="p5_hierarchical_wd",
        description="w_0=6 with default decay — broader initial exploration",
        component_removed="default_width",
        modification={"w_0": 6},
        expected_effect="More parallel workers initially, better coverage at step 0",
    ),
    AblationConfig(
        id="p5_more_steps",
        base_pattern="p5_hierarchical_wd",
        description="max_steps=5 — more refinement iterations",
        component_removed="step_cap",
        modification={"max_steps": 5},
        expected_effect="Deeper iterative refinement, more budget consumption",
    ),
    AblationConfig(
        id="p5_slow_decay",
        base_pattern="p5_hierarchical_wd",
        description="alpha=0.7 — slower width decay, wider for longer",
        component_removed="fast_decay",
        modification={"alpha": 0.7},
        expected_effect="Maintains broader search longer before focusing",
    ),
    AblationConfig(
        id="p5_aggressive_decay",
        base_pattern="p5_hierarchical_wd",
        description="alpha=0.3 — fast decay to depth, quick focus",
        component_removed="gradual_decay",
        modification={"alpha": 0.3},
        expected_effect="Rapid transition to depth phase, may miss breadth",
    ),

    # ── P6: Iteration count sweep ──────────────────────────────────────────
    AblationConfig(
        id="p6_10_iterations",
        base_pattern="p6_reactive_interleaved",
        description="10 iterations — medium exploration",
        component_removed="default_iterations",
        modification={"max_iterations": 10},
        expected_effect="Tests middle ground between 5 (limited) and 15 (default)",
    ),
    AblationConfig(
        id="p6_extended",
        base_pattern="p6_reactive_interleaved",
        description="25 iterations — extended exploration cycle",
        component_removed="iteration_cap",
        modification={"max_iterations": 25},
        expected_effect="More search-reflect-draft cycles, potentially better coverage",
    ),

    # ── P7: Graph size and depth sweeps ────────────────────────────────────
    AblationConfig(
        id="p7_deep_graph",
        base_pattern="p7_graph_decomposition",
        description="max_depth=5, max_nodes=40 — deeper and wider graph",
        component_removed="graph_size_cap",
        modification={"max_depth": 5, "max_nodes": 40},
        expected_effect="Much deeper research rabbit holes, more thorough coverage",
    ),
    AblationConfig(
        id="p7_wide_roots",
        base_pattern="p7_graph_decomposition",
        description="n_roots=8 with max_nodes=30 — broader initial decomposition",
        component_removed="root_count",
        modification={"n_roots": 8, "max_nodes": 30},
        expected_effect="More initial sub-questions, wider but potentially shallower graph",
    ),
    AblationConfig(
        id="p7_medium_graph",
        base_pattern="p7_graph_decomposition",
        description="max_depth=4, max_nodes=30 — moderate scaling",
        component_removed="graph_limits",
        modification={"max_depth": 4, "max_nodes": 30},
        expected_effect="Middle ground between default (20/3) and deep (40/5)",
    ),

    # ── P8: Beam configuration sweeps ──────────────────────────────────────
    AblationConfig(
        id="p8_12_hypotheses",
        base_pattern="p8_beam_search",
        description="12 initial hypotheses, beam_width=4 — broader exploration space",
        component_removed="hypothesis_cap",
        modification={"n_hypotheses": 12, "beam_width": 4, "final_beam_width": 3},
        expected_effect="Double the exploration diversity, wider beam survival",
    ),
    AblationConfig(
        id="p8_deep_beams",
        base_pattern="p8_beam_search",
        description="Default hypotheses but keep all 3 beams through final selection",
        component_removed="final_pruning",
        modification={"final_beam_width": 3},
        expected_effect="No final pruning, synthesis uses all surviving beams",
    ),

    # ── P9: Evidence context sweep ─────────────────────────────────────────
    AblationConfig(
        id="p9_short_evidence",
        base_pattern="p9_local_baseline",
        description="3000 word evidence limit — test if quality drops with less context",
        component_removed="full_evidence",
        modification={"evidence_word_limit": 3000},
        expected_effect="Less evidence may reduce report quality but improve coherence",
    ),
    AblationConfig(
        id="p9_extended_evidence",
        base_pattern="p9_local_baseline",
        description="12000 word evidence limit — test if 7B model can use more context",
        component_removed="evidence_cap",
        modification={"evidence_word_limit": 12000},
        expected_effect="More evidence available but 7B may struggle with long context",
    ),

    # ── P10: Search depth and evidence sweeps ──────────────────────────────
    AblationConfig(
        id="p10_extended_search",
        base_pattern="p10_deep_researcher",
        description="15 search iterations — let RL agent search longer",
        component_removed="search_iteration_cap",
        modification={"max_search_iterations": 15},
        expected_effect="More autonomous search rounds, more evidence gathered",
    ),
    AblationConfig(
        id="p10_extended_evidence",
        base_pattern="p10_deep_researcher",
        description="12000 word evidence limit — test if 7B model can use more context",
        component_removed="evidence_cap",
        modification={"evidence_word_limit": 12000},
        expected_effect="More evidence available but 7B may struggle with long context",
    ),
    AblationConfig(
        id="p10_short_evidence",
        base_pattern="p10_deep_researcher",
        description="3000 word evidence limit — test if quality drops with less context",
        component_removed="full_evidence",
        modification={"evidence_word_limit": 3000},
        expected_effect="Less evidence may reduce report quality but improve coherence",
    ),
]


class AblationRunner:
    """Executes ablation experiments with checkpointing."""

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints/ablations"),
        budget_per_run: float = 2.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.budget_per_run = budget_per_run
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, ablation_id: str, query_id: str) -> Path:
        return self.checkpoint_dir / ablation_id / f"{query_id}.json"

    def is_completed(self, ablation_id: str, query_id: str) -> bool:
        cp = self._checkpoint_path(ablation_id, query_id)
        if not cp.exists():
            return False
        try:
            data = json.loads(cp.read_text())
            return data.get("status") == "success"
        except (json.JSONDecodeError, KeyError):
            return False

    async def run_ablation(
        self,
        config: AblationConfig,
        query_text: str,
        query_id: str,
    ) -> AblationResult:
        """Run a single ablated pattern variant.

        Dynamically imports the base pattern and passes modification
        kwargs to its run() function.
        """
        import time
        start = time.time()

        try:
            report = await self._execute_ablated_pattern(
                config.base_pattern, query_text, config.modification
            )
            elapsed = time.time() - start

            result = AblationResult(
                ablation_id=config.id,
                query_id=query_id,
                base_pattern=config.base_pattern,
                component_removed=config.component_removed,
                status="success",
                report_text=report.full_text() if report else "",
                elapsed_seconds=elapsed,
                total_tokens=report.total_tokens if report else 0,
            )
        except Exception as e:
            elapsed = time.time() - start
            result = AblationResult(
                ablation_id=config.id,
                query_id=query_id,
                base_pattern=config.base_pattern,
                component_removed=config.component_removed,
                status="error",
                elapsed_seconds=elapsed,
                error_message=str(e)[:500],
            )

        # Save checkpoint
        cp = self._checkpoint_path(config.id, query_id)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(asdict(result), indent=2))

        return result

    async def _execute_ablated_pattern(
        self,
        pattern: str,
        query_text: str,
        modifications: dict,
    ) -> ResearchReport:
        """Execute pattern with modifications applied.

        Passes modification kwargs to the pattern's run() function.
        Patterns should accept **kwargs and handle unknown keys gracefully.
        """
        if pattern == "p0_baseline":
            from deep_research.patterns.p0_baseline.pipeline import run
        elif pattern == "p1_iterative_rag":
            from deep_research.patterns.p1_iterative_rag.pipeline import run
        elif pattern == "p2_supervisor_parallel":
            from deep_research.patterns.p2_supervisor_parallel.pipeline import run
        elif pattern == "p3_meridian":
            from deep_research.patterns.p3_meridian.pipeline import run
        elif pattern == "p4_perspective_storm":
            from deep_research.patterns.p4_perspective_storm.pipeline import run
        elif pattern == "p5_hierarchical_wd":
            from deep_research.patterns.p5_hierarchical_wd.pipeline import run
        elif pattern == "p6_reactive_interleaved":
            from deep_research.patterns.p6_reactive_interleaved.pipeline import run
        elif pattern == "p7_graph_decomposition":
            from deep_research.patterns.p7_graph_decomposition.pipeline import run
        elif pattern == "p8_beam_search":
            from deep_research.patterns.p8_beam_search.pipeline import run
        elif pattern == "p9_local_baseline":
            from deep_research.patterns.p9_local_baseline.pipeline import run
        elif pattern == "p10_deep_researcher":
            from deep_research.patterns.p10_deep_researcher.pipeline import run
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return await run(query_text, budget_usd=self.budget_per_run, **modifications)

    async def run_all_ablations(
        self,
        queries: list,  # list[EvalQuery] - representative subset
        configs: list[AblationConfig] | None = None,
        resume: bool = True,
    ) -> list[AblationResult]:
        """Run all ablation configs on all queries."""
        configs = configs or ABLATION_REGISTRY

        results = []
        total = len(configs) * len(queries)
        completed = 0

        for config in configs:
            for query in queries:
                if resume and self.is_completed(config.id, query.id):
                    completed += 1
                    continue

                result = await self.run_ablation(config, query.query, query.id)
                results.append(result)
                completed += 1

                if completed % 5 == 0:
                    logger.info("ablation_progress",
                              completed=completed, total=total,
                              current=config.id)

        return results

    def compare_ablations(
        self,
        base_scores: dict[str, list[float]],  # pattern -> per-query scores
        ablation_scores: dict[str, list[float]],  # ablation_id -> per-query scores
        configs: list[AblationConfig] | None = None,
    ) -> list[AblationComparison]:
        """Compare base pattern scores with ablated variant scores.

        Uses Wilcoxon signed-rank test and Cliff's Delta for each comparison.
        """
        from deep_research.evaluation.statistical_analysis import (
            wilcoxon_signed_rank_pairwise, cliffs_delta
        )
        import numpy as np

        configs = configs or ABLATION_REGISTRY
        comparisons = []

        for config in configs:
            if config.id not in ablation_scores:
                continue
            if config.base_pattern not in base_scores:
                continue

            base = np.array(base_scores[config.base_pattern])
            ablated = np.array(ablation_scores[config.id])

            # Ensure same length (matched queries)
            n = min(len(base), len(ablated))
            base = base[:n]
            ablated = ablated[:n]

            base_mean = float(np.mean(base))
            ablated_mean = float(np.mean(ablated))
            delta = base_mean - ablated_mean

            # Statistical test
            if n >= 5:
                pair_result = wilcoxon_signed_rank_pairwise(
                    base, ablated, config.base_pattern, config.id
                )
                p_val = pair_result.p_value_raw
                is_sig = pair_result.is_significant
            else:
                p_val = 1.0
                is_sig = False

            # Effect size
            effect, label = cliffs_delta(base, ablated)

            comparisons.append(AblationComparison(
                ablation_id=config.id,
                base_pattern=config.base_pattern,
                component_removed=config.component_removed,
                description=config.description,
                expected_effect=config.expected_effect,
                base_scores=base.tolist(),
                ablated_scores=ablated.tolist(),
                base_mean=base_mean,
                ablated_mean=ablated_mean,
                score_delta=delta,
                relative_change=(delta / base_mean * 100) if base_mean > 0 else 0,
                is_significant=is_sig,
                p_value=p_val,
                effect_size=effect,
                effect_label=label,
            ))

        return comparisons

    def generate_ablation_report(
        self, comparisons: list[AblationComparison]
    ) -> str:
        """Generate markdown ablation study report."""
        lines = ["# Ablation Study Results\n"]

        # Group by base pattern
        by_pattern: dict[str, list[AblationComparison]] = {}
        for c in comparisons:
            by_pattern.setdefault(c.base_pattern, []).append(c)

        for pattern, comps in sorted(by_pattern.items()):
            lines.append(f"\n## {pattern}\n")
            lines.append("| Component Removed | Base Mean | Ablated Mean | D | D% | p-value | Effect | Sig? |")
            lines.append("|---|---|---|---|---|---|---|---|")

            for c in comps:
                sig = "Yes" if c.is_significant else "No"
                lines.append(
                    f"| {c.component_removed} | {c.base_mean:.3f} | "
                    f"{c.ablated_mean:.3f} | {c.score_delta:+.3f} | "
                    f"{c.relative_change:+.1f}% | {c.p_value:.4f} | "
                    f"{c.effect_label} ({c.effect_size:.3f}) | {sig} |"
                )

            lines.append("")
            for c in comps:
                lines.append(f"**{c.component_removed}**: {c.description}")
                lines.append(f"- Expected: {c.expected_effect}")
                lines.append(f"- Observed: D={c.score_delta:+.3f} ({c.relative_change:+.1f}%), "
                           f"effect={c.effect_label}")
                lines.append("")

        return "\n".join(lines)
