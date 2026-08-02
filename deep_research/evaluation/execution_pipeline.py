"""Execution pipeline for running patterns across evaluation queries.

Orchestrates 540+ pattern runs (6 patterns x 90 queries) with:
- Checkpoint/resume: skip completed runs
- Error handling: content filter failures, budget exceeded, timeouts
- Progress monitoring: periodic status reports
- Concurrent execution: configurable parallelism
"""

import asyncio
import json
import random
import time
import structlog
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

from deep_research.config import CHECKPOINTS_DIR, REPORTS_DIR, MAX_COST_PER_RUN, EVAL_PIPELINE
from deep_research.types import ResearchReport

logger = structlog.get_logger()


@dataclass
class RunResult:
    """Result of a single pattern x query execution."""
    pattern: str
    query_id: str
    status: str          # "success", "content_filter", "budget_exceeded", "error", "skipped"
    report_text: str = ""
    report_path: str = ""
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_call_count: int = 0
    cost_breakdown: dict = field(default_factory=dict)
    error_message: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)
    repeat_index: int = 0  # 0-based index for repeated runs
    word_count: int = 0  # report word count for length-as-covariate analysis

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass
class PipelineProgress:
    """Progress tracking for the execution pipeline."""
    total_runs: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = 0.0

    @property
    def remaining(self) -> int:
        return self.total_runs - self.completed

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60 if self.start_time else 0

    @property
    def eta_minutes(self) -> float:
        if self.completed == 0:
            return 0
        rate = self.elapsed_minutes / self.completed
        return rate * self.remaining

    def summary(self) -> str:
        return (
            f"Progress: {self.completed}/{self.total_runs} "
            f"({self.succeeded} ok, {self.failed} failed, {self.skipped} skipped) "
            f"[{self.elapsed_minutes:.1f}m elapsed, ~{self.eta_minutes:.1f}m remaining]"
        )


class ExecutionPipeline:
    """Orchestrates pattern runs with checkpoint/resume."""

    PATTERN_NAMES = [
        "p0_baseline",
        "p1_iterative_rag",
        "p2_supervisor_parallel",
        "p3_meridian",
        "p4_perspective_storm",
        "p5_hierarchical_wd",
        "p6_reactive_interleaved",
        "p7_graph_decomposition",
        "p8_beam_search",
        "p9_local_baseline",
        "p10_deep_researcher",
        "p13_vintage_qwen3_8b",
        "p14_vintage_deepseek_qwen7b",
        "p15_p9_repext",
        "p16_p10_repext",
        "p17_scale_qwen25_14b",
    ]

    def __init__(
        self,
        checkpoint_dir: Path = CHECKPOINTS_DIR / "eval_v2",
        results_dir: Path = REPORTS_DIR / "eval_v2",
        budget_per_run: float = MAX_COST_PER_RUN,
        max_concurrent: int = EVAL_PIPELINE.max_concurrent_runs,
        log_interval: int = 10,
        n_repeats: int = 1,
        token_budget: int = 0,
        randomize_order: bool = True,
        random_seed: int = 42,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.budget_per_run = budget_per_run
        self.max_concurrent = max_concurrent
        self.log_interval = log_interval
        self.n_repeats = n_repeats
        self.token_budget = token_budget  # 0 = unlimited
        self.randomize_order = randomize_order
        self.random_seed = random_seed
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, pattern: str, query_id: str, repeat: int = 0) -> Path:
        """Path for a run checkpoint."""
        if repeat > 0:
            return self.checkpoint_dir / pattern / f"{query_id}_r{repeat}.json"
        return self.checkpoint_dir / pattern / f"{query_id}.json"

    def _report_path(self, pattern: str, query_id: str, repeat: int = 0) -> Path:
        """Path for saved report markdown."""
        if repeat > 0:
            return self.results_dir / "reports" / pattern / f"{query_id}_r{repeat}.md"
        return self.results_dir / "reports" / pattern / f"{query_id}.md"

    def is_completed(self, pattern: str, query_id: str, repeat: int = 0) -> bool:
        """Check if a run has already completed successfully."""
        cp = self._checkpoint_path(pattern, query_id, repeat)
        if not cp.exists():
            return False
        try:
            data = json.loads(cp.read_text())
            return data.get("status") == "success"
        except (json.JSONDecodeError, KeyError):
            return False

    def save_checkpoint(self, result: RunResult) -> None:
        """Save run result as checkpoint."""
        cp = self._checkpoint_path(result.pattern, result.query_id, result.repeat_index)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(asdict(result), indent=2))

        # Also save report text as markdown (only repeat 0 for the primary report)
        if result.succeeded and result.report_text:
            rp = self._report_path(result.pattern, result.query_id, result.repeat_index)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(result.report_text)

    def load_checkpoint(self, pattern: str, query_id: str, repeat: int = 0) -> Optional[RunResult]:
        """Load a saved checkpoint.

        Handles backward compatibility with older checkpoints that may lack
        newer fields (e.g. word_count).
        """
        cp = self._checkpoint_path(pattern, query_id, repeat)
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
            # Backward compat: fill defaults for fields added after initial checkpoints
            data.setdefault("word_count", 0)
            data.setdefault("total_input_tokens", 0)
            data.setdefault("total_output_tokens", 0)
            data.setdefault("llm_call_count", 0)
            data.setdefault("cost_breakdown", {})
            data.setdefault("repeat_index", 0)
            return RunResult(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    async def run_single(
        self, pattern: str, query_text: str, query_id: str, repeat_index: int = 0,
    ) -> RunResult:
        """Run a single pattern on a single query with error handling.

        Imports the pattern module dynamically and calls its run() function.
        """
        start = time.time()
        timestamp = datetime.now().isoformat()

        try:
            # Dynamic import of pattern pipeline
            report = await self._execute_pattern(pattern, query_text)

            elapsed = time.time() - start
            cost_data = report.metadata.get("cost_breakdown", {}) if report else {}
            report_text = report.full_text() if report else ""
            return RunResult(
                pattern=pattern,
                query_id=query_id,
                status="success",
                report_text=report_text,
                elapsed_seconds=elapsed,
                total_tokens=report.total_tokens if report else 0,
                cost_usd=report.total_cost_usd if report else 0.0,
                total_input_tokens=cost_data.get("total_input_tokens", 0),
                total_output_tokens=cost_data.get("total_output_tokens", 0),
                llm_call_count=cost_data.get("llm_call_count", 0),
                cost_breakdown=cost_data,
                timestamp=timestamp,
                metadata={
                    "n_sections": len(report.sections) if report else 0,
                    "n_citations": len(report.citations) if report else 0,
                    "sub_queries": report.metadata.get("sub_queries", []) if report else [],
                    "search_queries_sent": report.metadata.get("search_queries_sent", []) if report else [],
                    "n_documents_retrieved": report.metadata.get("n_documents_retrieved", 0) if report else 0,
                    "n_extractions": report.metadata.get("n_extractions", 0) if report else 0,
                },
                repeat_index=repeat_index,
                word_count=len(report_text.split()) if report_text else 0,
            )
        except Exception as e:
            elapsed = time.time() - start
            error_type = type(e).__name__

            # Classify the error
            if "content_filter" in str(e).lower() or "content_management" in str(e).lower():
                status = "content_filter"
            elif "budget" in str(e).lower():
                status = "budget_exceeded"
            else:
                status = "error"

            logger.warning(
                "run_failed",
                pattern=pattern,
                query_id=query_id,
                status=status,
                error=str(e)[:200],
            )

            return RunResult(
                pattern=pattern,
                query_id=query_id,
                status=status,
                elapsed_seconds=elapsed,
                error_message=f"{error_type}: {str(e)[:500]}",
                timestamp=timestamp,
                repeat_index=repeat_index,
            )

    async def _execute_pattern(self, pattern: str, query_text: str) -> ResearchReport:
        """Dynamically import and execute a pattern pipeline."""
        # Import the appropriate pattern module
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
        elif pattern == "p13_vintage_qwen3_8b":
            from deep_research.patterns.p13_vintage_qwen3_8b.pipeline import run
        elif pattern == "p14_vintage_deepseek_qwen7b":
            from deep_research.patterns.p14_vintage_deepseek_qwen7b.pipeline import run
        elif pattern == "p15_p9_repext":
            from deep_research.patterns.p15_p9_repext.pipeline import run
        elif pattern == "p16_p10_repext":
            from deep_research.patterns.p16_p10_repext.pipeline import run
        elif pattern == "p17_scale_qwen25_14b":
            from deep_research.patterns.p17_scale_qwen25_14b.pipeline import run
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return await run(query_text, budget_usd=self.budget_per_run)

    async def run_all(
        self,
        queries: list,  # list[EvalQuery] — duck-typed: needs .id, .query attributes
        patterns: list[str] | None = None,
        resume: bool = True,
    ) -> list[RunResult]:
        """Execute all pattern x query combinations.

        Args:
            queries: List of query objects with .id and .query attributes
                (EvalQuery, TestQuery, or any duck-typed equivalent)
            patterns: Pattern names to run (default: all 6)
            resume: If True, skip already-completed runs

        Returns:
            List of all RunResult objects (including skipped)
        """
        patterns = patterns or self.PATTERN_NAMES

        # Save environment metadata for reproducibility
        from deep_research.config import get_environment_metadata
        env_meta = get_environment_metadata()
        env_path = self.results_dir / "environment.json"
        env_path.write_text(json.dumps(env_meta, indent=2))

        # Build run plan (supports repeated runs for variance estimation)
        plan: list[tuple[str, str, object, int]] = []  # (action, pattern, query, repeat)
        for repeat in range(self.n_repeats):
            for pattern in patterns:
                for query in queries:
                    if resume and self.is_completed(pattern, query.id, repeat):
                        plan.append(("skip", pattern, query, repeat))
                    else:
                        plan.append(("run", pattern, query, repeat))

        # Randomize run order to eliminate time-of-day confounds
        if self.randomize_order:
            rng = random.Random(self.random_seed)
            # Only shuffle the "run" items; keep "skip" items separate
            runs = [item for item in plan if item[0] == "run"]
            skips = [item for item in plan if item[0] == "skip"]
            rng.shuffle(runs)
            plan = skips + runs

        progress = PipelineProgress(
            total_runs=len(plan),
            skipped=sum(1 for action, _, _, _ in plan if action == "skip"),
            start_time=time.time(),
        )
        progress.completed = progress.skipped

        logger.info(
            "pipeline_start",
            total=progress.total_runs,
            to_run=progress.remaining,
            skipped=progress.skipped,
            patterns=patterns,
        )

        results: list[RunResult] = []

        # Process skipped first
        for action, pattern, query, repeat in plan:
            if action == "skip":
                cp = self.load_checkpoint(pattern, query.id)
                if cp:
                    results.append(cp)
                else:
                    results.append(
                        RunResult(pattern=pattern, query_id=query.id, status="skipped",
                                  repeat_index=repeat)
                    )

        # Run remaining with concurrency control
        async def _run_with_semaphore(
            pattern: str, query: object, repeat: int,
        ) -> RunResult:
            async with self._semaphore:
                return await self.run_single(pattern, query.query, query.id, repeat)

        to_run = [(p, q, r) for action, p, q, r in plan if action == "run"]

        if to_run:
            tasks = [_run_with_semaphore(p, q, r) for p, q, r in to_run]

            # Use asyncio.as_completed for progress tracking
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)

                # Save checkpoint
                self.save_checkpoint(result)

                # Update progress
                progress.completed += 1
                if result.succeeded:
                    progress.succeeded += 1
                else:
                    progress.failed += 1

                # Log progress periodically
                if (i + 1) % self.log_interval == 0 or (i + 1) == len(to_run):
                    logger.info(
                        "pipeline_progress",
                        completed=progress.completed,
                        total=progress.total_runs,
                        succeeded=progress.succeeded,
                        failed=progress.failed,
                        elapsed_min=f"{progress.elapsed_minutes:.1f}",
                        eta_min=f"{progress.eta_minutes:.1f}",
                    )

        # Save final summary
        self._save_summary(results)

        logger.info(
            "pipeline_complete",
            total=len(results),
            succeeded=sum(1 for r in results if r.succeeded),
            failed=sum(
                1
                for r in results
                if r.status in ("error", "content_filter", "budget_exceeded")
            ),
            elapsed_min=f"{progress.elapsed_minutes:.1f}",
        )

        return results

    def _save_summary(self, results: list[RunResult]) -> None:
        """Save pipeline summary to results directory."""
        summary: dict = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "by_status": {},
            "by_pattern": {},
        }

        for r in results:
            summary["by_status"][r.status] = summary["by_status"].get(r.status, 0) + 1
            if r.pattern not in summary["by_pattern"]:
                summary["by_pattern"][r.pattern] = {
                    "success": 0,
                    "failed": 0,
                    "skipped": 0,
                }
            if r.succeeded:
                summary["by_pattern"][r.pattern]["success"] += 1
            elif r.status == "skipped":
                summary["by_pattern"][r.pattern]["skipped"] += 1
            else:
                summary["by_pattern"][r.pattern]["failed"] += 1

        path = self.results_dir / "pipeline_summary.json"
        path.write_text(json.dumps(summary, indent=2))

    def get_all_results(self) -> list[RunResult]:
        """Load all checkpointed results."""
        results: list[RunResult] = []
        if not self.checkpoint_dir.exists():
            return results
        for pattern_dir in sorted(self.checkpoint_dir.iterdir()):
            if pattern_dir.is_dir():
                for cp_file in sorted(pattern_dir.glob("*.json")):
                    try:
                        data = json.loads(cp_file.read_text())
                        results.append(RunResult(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return results
