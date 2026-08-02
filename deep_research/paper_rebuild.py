"""Public Paper A rebuild helpers.

This module owns the supported public rebuild path for the paper artifacts:
tables, figures, and the final PDF source. It intentionally does not claim to
recreate the private historical raw run or regenerate every upstream verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PAPER_DIR = Path("paper_rebuild/paper_a_bounded_returns")
ANALYSIS_DIR = PAPER_DIR / "analysis"
FINAL_PDF = Path("papers/paper_a_bounded_returns/main.pdf")
PAPER_SOURCE_DATE_EPOCH = "1785283200"  # 2026-07-29 00:00:00 UTC

TABLE_SCRIPTS = [
    "make_tables.py",
    "make_b2_bestofn_tables.py",
    "make_paper2_tables.py",
]

FIGURE_SCRIPTS = [
    "make_money_figure.py",
    "make_judge_gold_figure.py",
    "make_cd_diagram.py",
    "make_stratification_figure.py",
    "make_vintage_figure.py",
    "make_cost_figure.py",
    "make_disentanglement_figure.py",
    "make_e5_dose_response.py",
    "make_oracle_figure.py",
]

REQUIRED_INPUTS = [
    PAPER_DIR / "main.tex",
    PAPER_DIR / "references.bib",
    PAPER_DIR / "references_new.bib",
    ANALYSIS_DIR / "canonical_numbers.json",
    "data/analysis/df_queries.parquet",
    "data/analysis/df_runs.parquet",
    "data/analysis/df_scores.parquet",
    "data/analysis/df_overall_scores.parquet",
    "data/analysis/df_verdicts.parquet",
]

REQUIRED_PDF_ASSETS = [
    PAPER_DIR / "figures/fig1_money.pdf",
    PAPER_DIR / "figures/fig_judge_gold.pdf",
    PAPER_DIR / "figures/fig_cd_clean.pdf",
    PAPER_DIR / "figures/fig_stratification.pdf",
    PAPER_DIR / "figures/fig_vintage.pdf",
    PAPER_DIR / "figures/fig_cost.pdf",
    PAPER_DIR / "figures/fig_disentanglement.pdf",
    PAPER_DIR / "figures/fig_e5_dose_response.pdf",
    PAPER_DIR / "figures/fig_oracle.pdf",
    PAPER_DIR / "tables/tab_p2_neff.tex",
    PAPER_DIR / "tables/tab_headline_means.tex",
    PAPER_DIR / "tables/tab_per_dimension.tex",
    PAPER_DIR / "tables/tab_irr.tex",
    PAPER_DIR / "tables/tab_verdicts.tex",
    PAPER_DIR / "tables/tab_citations.tex",
    PAPER_DIR / "tables/tab_per_source.tex",
    PAPER_DIR / "tables/tab_bing_tavily.tex",
    PAPER_DIR / "tables/tab_drjudge.tex",
    PAPER_DIR / "tables/tab_ablations.tex",
    PAPER_DIR / "tables/tab_single_judge.tex",
    PAPER_DIR / "tables/tab_b2.tex",
    PAPER_DIR / "tables/tab_bestofn_decoupled.tex",
]


@dataclass(frozen=True)
class ScriptResult:
    script: str
    returncode: int
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class PaperRebuildReport:
    status: str
    message: str
    project_root: str
    missing_inputs: list[str]
    missing_pdf_assets: list[str]
    table_scripts: list[ScriptResult]
    figure_scripts: list[ScriptResult]
    compile_result: ScriptResult | None
    compile_skipped_reason: str | None
    output_pdf: str
    canonical_store_unchanged: bool
    canonical_store_fingerprint_before: dict[str, str | int] | None
    canonical_store_fingerprint_after: dict[str, str | int] | None
    contract: str

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2)


def _rel(path: Path | str) -> str:
    return Path(path).as_posix()


def _tail(text: str, *, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _file_fingerprint(path: Path) -> dict[str, str | int] | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _run_python_script(project_root: Path, script_name: str) -> ScriptResult:
    script = ANALYSIS_DIR / script_name
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", PAPER_SOURCE_DATE_EPOCH)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    proc = subprocess.run(
        [sys.executable, _rel(script)],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return ScriptResult(
        script=_rel(script),
        returncode=proc.returncode,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )


def _compile_pdf(project_root: Path) -> tuple[ScriptResult | None, str | None]:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        return None, "tectonic is not installed"
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", PAPER_SOURCE_DATE_EPOCH)
    proc = subprocess.run(
        [tectonic, "main.tex"],
        cwd=project_root / PAPER_DIR,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    return (
        ScriptResult(
            script="tectonic main.tex",
            returncode=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        ),
        None,
    )


def run_paper_rebuild(
    project_root: Path,
    *,
    check_only: bool = False,
    tables: bool = True,
    figures: bool = True,
    compile_pdf: bool = True,
) -> PaperRebuildReport:
    project_root = project_root.resolve()
    missing_inputs = [
        _rel(path) for path in REQUIRED_INPUTS if not (project_root / path).exists()
    ]
    missing_pdf_assets = [
        _rel(path) for path in REQUIRED_PDF_ASSETS if not (project_root / path).exists()
    ]
    if missing_inputs or check_only:
        status = "success" if not missing_inputs and not missing_pdf_assets else "failed"
        return PaperRebuildReport(
            status=status,
            message="paper rebuild input check completed",
            project_root=str(project_root),
            missing_inputs=missing_inputs,
            missing_pdf_assets=missing_pdf_assets,
            table_scripts=[],
            figure_scripts=[],
            compile_result=None,
            compile_skipped_reason="check-only" if check_only else "missing inputs",
            output_pdf=_rel(PAPER_DIR / "main.pdf"),
            canonical_store_unchanged=True,
            canonical_store_fingerprint_before=_file_fingerprint(
                project_root / ANALYSIS_DIR / "canonical_numbers.json"
            ),
            canonical_store_fingerprint_after=_file_fingerprint(
                project_root / ANALYSIS_DIR / "canonical_numbers.json"
            ),
            contract=(
                "Rebuilds public paper artifacts from the included canonical store and compact "
                "derived analysis tables; it does not rerun the private historical raw corpus."
            ),
        )

    canonical_path = project_root / ANALYSIS_DIR / "canonical_numbers.json"
    canonical_before = _file_fingerprint(canonical_path)
    table_results = [_run_python_script(project_root, name) for name in TABLE_SCRIPTS] if tables else []
    figure_results = (
        [_run_python_script(project_root, name) for name in FIGURE_SCRIPTS] if figures else []
    )
    canonical_after = _file_fingerprint(canonical_path)
    canonical_store_unchanged = canonical_before == canonical_after

    compile_result = None
    compile_skipped_reason = None
    if compile_pdf:
        compile_result, compile_skipped_reason = _compile_pdf(project_root)
        if compile_result is not None and compile_result.ok:
            final_pdf = project_root / FINAL_PDF
            final_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project_root / PAPER_DIR / "main.pdf", final_pdf)
    else:
        compile_skipped_reason = "disabled"

    missing_pdf_assets_after = [
        _rel(path) for path in REQUIRED_PDF_ASSETS if not (project_root / path).exists()
    ]
    script_failures = [r for r in [*table_results, *figure_results] if not r.ok]
    compile_failed = compile_result is not None and not compile_result.ok

    if script_failures or compile_failed or missing_pdf_assets_after or not canonical_store_unchanged:
        status = "failed"
        if not canonical_store_unchanged:
            message = "paper artifact rebuild mutated canonical_numbers.json"
        else:
            message = "paper artifact rebuild failed"
    else:
        status = "success"
        if compile_skipped_reason:
            message = "paper artifacts rebuilt; PDF compile skipped"
        else:
            message = "paper artifacts and PDF rebuilt"

    return PaperRebuildReport(
        status=status,
        message=message,
        project_root=str(project_root),
        missing_inputs=missing_inputs,
        missing_pdf_assets=missing_pdf_assets_after,
        table_scripts=table_results,
        figure_scripts=figure_results,
        compile_result=compile_result,
        compile_skipped_reason=compile_skipped_reason,
        output_pdf=_rel(PAPER_DIR / "main.pdf"),
        canonical_store_unchanged=canonical_store_unchanged,
        canonical_store_fingerprint_before=canonical_before,
        canonical_store_fingerprint_after=canonical_after,
        contract=(
            "Rebuilds public paper artifacts from the included canonical store and compact "
            "derived analysis tables; it does not rerun the private historical raw corpus."
        ),
    )
