#!/usr/bin/env python3
"""Download external benchmark datasets and normalise them into our cache format.

Downloads from HuggingFace Hub and/or GitHub, then writes each benchmark to
``data/benchmarks/{name}/{name}_queries.json`` in the shared schema expected by
the evaluation pipeline.

Supported benchmarks (query-schema sets, -> data/benchmarks/):
    research_rubrics  -- ScaleAI/researchrubrics (HuggingFace)
    drb2              -- muset-ai/DeepResearch-Bench-II-Dataset (HuggingFace)
    drb1              -- muset-ai/DeepResearch-Bench-Dataset (HuggingFace)
    drbench           -- EVIGBYEN/DrBench (HuggingFace)

Human-label sets (native schema preserved, -> data/human_labels/):
    healthbench       -- OpenAI HealthBench meta_eval (physician binary met/not-met)
    expertqa          -- cmalaviya/expertqa (per-claim factuality + attribution)
    deepfactbench     -- kkkevinkkk/DeepFactBench (claim-level PhD verdicts)
    longjudgebench    -- cjj826/LongJudgeBench (gold pointwise/pairwise judging)
    draco_full        -- perplexity-ai/draco (100 tasks, 3,934 expert criteria)

Usage:
    python scripts/download_benchmarks.py --all
    python scripts/download_benchmarks.py --benchmark research_rubrics
    python scripts/download_benchmarks.py --benchmark drb1 --benchmark drb2 --force
    python scripts/download_benchmarks.py --benchmark healthbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

log = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_BENCHMARKS_DIR = _DATA_DIR / "benchmarks"
# Human-label sets live in their OWN dir, separate from the query-schema
# benchmarks above and from any results/* corpus dirs.  We never write into
# data/benchmarks/* from a human-label downloader.
_HUMAN_LABELS_DIR = _DATA_DIR / "human_labels"

AVAILABLE_BENCHMARKS = [
    "research_rubrics",
    "drb2",
    "drb1",
    "drbench",
    # Human-label sets (Phase-1 human-label validation programme).
    "healthbench",
    "expertqa",
    "deepfactbench",
    "longjudgebench",
    "draco_full",
]


# ── Shared helpers ───────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any, *, label: str = "") -> None:
    """Atomically write *data* as indented JSON."""
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("file_written", path=str(path), label=label,
             entries=len(data) if isinstance(data, list) else "n/a")


def _cache_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ── Human-label helpers ──────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: List[Dict[str, Any]], *, label: str = "") -> None:
    """Write *rows* one JSON object per line (native-schema raw/normalised view)."""
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("jsonl_written", path=str(path), label=label, rows=len(rows))


def _write_licence(cache_dir: Path, *, name: str, licence: str,
                   source: str, note: str = "") -> None:
    """Record the licence for a human-label set, per the asset register."""
    _ensure_dir(cache_dir)
    body = (
        f"Dataset: {name}\n"
        f"Source:  {source}\n"
        f"Licence: {licence}\n"
        f"Recorded from: reports/HUMAN_LABEL_ASSETS.md (v1, 2026-06-11)\n"
    )
    if note:
        body += f"\nNote: {note}\n"
    (cache_dir / "LICENCE.txt").write_text(body, encoding="utf-8")


def _urlopen_to_file(url: str, dest: Path, *, chunk: int = 1 << 20) -> int:
    """Stream a URL to *dest*. Returns bytes written. Raises on HTTP error."""
    import urllib.request
    _ensure_dir(dest.parent)
    req = urllib.request.Request(url, headers={"User-Agent": "deep-research-eval/1.0"})
    written = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            written += len(buf)
    return written


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, tolerating blank/bad lines."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _github_raw_first(urls: List[str], dest: Path) -> Optional[Path]:
    """Try each GitHub-raw URL in order; stream the first that works to *dest*."""
    for url in urls:
        try:
            n = _urlopen_to_file(url, dest)
            if n > 0:
                log.info("github_raw_ok", url=url, bytes=n)
                return dest
        except Exception as exc:  # noqa: BLE001
            log.warning("github_raw_failed", url=url, error=str(exc))
    return None


# ── 1. ResearchRubrics ───────────────────────────────────────────────────────

def _map_research_rubrics_difficulty(
    conceptual_breadth: str,
    logical_nesting: str,
    exploration: str,
) -> str:
    """Map the three complexity axes to our simple/moderate/complex scale.

    simple  — all three at their lowest level
    complex — all three at their highest level
    moderate — everything else
    """
    low = {"simple", "shallow", "low"}
    high = {"high", "deep"}

    vals = {conceptual_breadth.lower(), logical_nesting.lower(), exploration.lower()}
    if vals <= low:
        return "simple"
    if vals <= high:
        return "complex"
    return "moderate"


def download_research_rubrics(*, force: bool = False) -> int:
    """Download ScaleAI/researchrubrics and normalise.

    Returns the number of queries written.
    """
    cache_dir = _BENCHMARKS_DIR / "research_rubrics"
    cache_path = cache_dir / "research_rubrics_queries.json"
    raw_path = cache_dir / "research_rubrics_raw.jsonl"

    if not force and _cache_exists(cache_path):
        log.info("research_rubrics_skip", reason="cache exists, use --force to re-download")
        return 0

    log.info("research_rubrics_download_start", repo="ScaleAI/researchrubrics")
    _ensure_dir(cache_dir)

    # Try downloading the JSONL file from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        local_file = hf_hub_download(
            repo_id="ScaleAI/researchrubrics",
            filename="processed_data.jsonl",
            repo_type="dataset",
        )
    except Exception as exc:
        log.error("research_rubrics_download_failed", error=str(exc))
        # Fallback: try snapshot_download and find the file
        try:
            from huggingface_hub import snapshot_download
            snapshot_dir = snapshot_download(
                repo_id="ScaleAI/researchrubrics",
                repo_type="dataset",
            )
            candidates = list(Path(snapshot_dir).rglob("*.jsonl"))
            if not candidates:
                candidates = list(Path(snapshot_dir).rglob("*.json"))
            if not candidates:
                log.error("research_rubrics_no_files", snapshot=snapshot_dir)
                return 0
            local_file = str(candidates[0])
            log.info("research_rubrics_fallback_found", file=local_file)
        except Exception as exc2:
            log.error("research_rubrics_snapshot_failed", error=str(exc2))
            return 0

    # Read raw JSONL
    raw_lines: List[str] = []
    queries: List[Dict[str, Any]] = []

    with open(local_file, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            raw_lines.append(line)

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("research_rubrics_bad_line", line=line_num)
                continue

            prompt = row.get("prompt", "")
            sample_id = row.get("sample_id", f"rr_{line_num:04d}")
            domain = row.get("domain", "general").lower()
            cb = row.get("conceptual_breadth", "Moderate")
            ln = row.get("logical_nesting", "Intermediate")
            ex = row.get("exploration", "Medium")
            rubrics = row.get("rubrics", [])

            difficulty = _map_research_rubrics_difficulty(cb, ln, ex)

            queries.append({
                "id": sample_id,
                "query": prompt,
                "domain": domain,
                "difficulty": difficulty,
                "reference_answer": "",
                "expected_citations": [],
                "metadata": {
                    "sample_id": sample_id,
                    "conceptual_breadth": cb,
                    "logical_nesting": ln,
                    "exploration": ex,
                    "rubrics": rubrics,
                },
            })

    # Cache the raw JSONL alongside normalised output
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    log.info("research_rubrics_raw_cached", path=str(raw_path), lines=len(raw_lines))

    _write_json(cache_path, queries, label="research_rubrics")
    log.info("research_rubrics_done", queries=len(queries))
    return len(queries)


# ── 2. DRB-II ────────────────────────────────────────────────────────────────

def download_drb2(*, force: bool = False) -> int:
    """Download DRB-II (DeepResearch-Bench-II).

    Tries the HuggingFace dataset first (Parquet/JSONL), then falls back to
    the GitHub repository's ``tasks_and_rubrics.jsonl``.

    Returns the number of queries written.
    """
    cache_dir = _BENCHMARKS_DIR / "drb2"
    cache_path = cache_dir / "drb2_queries.json"

    if not force and _cache_exists(cache_path):
        log.info("drb2_skip", reason="cache exists, use --force to re-download")
        return 0

    log.info("drb2_download_start")
    _ensure_dir(cache_dir)

    rows: List[Dict[str, Any]] = []

    # Strategy A: HuggingFace dataset
    rows = _drb2_try_huggingface()

    # Strategy B: GitHub tasks_and_rubrics.jsonl
    if not rows:
        rows = _drb2_try_github()

    if not rows:
        log.error("drb2_download_failed", msg="All download strategies exhausted")
        return 0

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        qid = row.get("id", row.get("task_id", f"drb2_{i:04d}"))
        query_text = row.get("query", row.get("task", row.get("question", "")))
        domain = row.get("domain", row.get("field", "research")).lower()
        rubrics = row.get("rubrics", row.get("rubric", []))
        dimensions = row.get("dimensions", row.get("dimension_breakdown", {}))

        queries.append({
            "id": str(qid),
            "query": query_text,
            "domain": domain,
            "difficulty": "complex",  # All PhD-level
            "reference_answer": row.get("reference_answer", row.get("answer", "")),
            "expected_citations": [],
            "metadata": {
                "rubrics": rubrics,
                "dimension_breakdown": dimensions,
                "raw_keys": list(row.keys()),
            },
        })

    _write_json(cache_path, queries, label="drb2")
    log.info("drb2_done", queries=len(queries))
    return len(queries)


def _drb2_try_huggingface() -> List[Dict[str, Any]]:
    """Try loading DRB-II from HuggingFace datasets."""
    rows: List[Dict[str, Any]] = []
    try:
        from huggingface_hub import hf_hub_download
        # Try JSONL first
        for filename in ("tasks_and_rubrics.jsonl", "data.jsonl"):
            try:
                local = hf_hub_download(
                    repo_id="muset-ai/DeepResearch-Bench-II-Dataset",
                    filename=filename,
                    repo_type="dataset",
                )
                with open(local, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                if rows:
                    log.info("drb2_hf_jsonl", filename=filename, rows=len(rows))
                    return rows
            except Exception:
                continue

        # Try Parquet via datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "muset-ai/DeepResearch-Bench-II-Dataset",
                split="train",
            )
            for row in ds:
                rows.append(dict(row))
            if rows:
                log.info("drb2_hf_parquet", rows=len(rows))
                return rows
        except Exception as exc:
            log.warning("drb2_hf_parquet_failed", error=str(exc))

        # Try snapshot download as last HF strategy
        try:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(
                repo_id="muset-ai/DeepResearch-Bench-II-Dataset",
                repo_type="dataset",
            )
            for path in Path(snap).rglob("*.jsonl"):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                if rows:
                    log.info("drb2_snapshot_jsonl", path=str(path), rows=len(rows))
                    return rows
            for path in Path(snap).rglob("*.parquet"):
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    rows = df.to_dict(orient="records")
                    if rows:
                        log.info("drb2_snapshot_parquet", path=str(path), rows=len(rows))
                        return rows
                except Exception:
                    continue
        except Exception as exc:
            log.warning("drb2_snapshot_failed", error=str(exc))

    except ImportError:
        log.warning("drb2_no_huggingface_hub")

    return rows


def _drb2_try_github() -> List[Dict[str, Any]]:
    """Try downloading DRB-II tasks_and_rubrics.jsonl from GitHub."""
    rows: List[Dict[str, Any]] = []
    import urllib.request

    urls = [
        "https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/main/tasks_and_rubrics.jsonl",
        "https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/master/tasks_and_rubrics.jsonl",
    ]

    for url in urls:
        try:
            log.info("drb2_github_try", url=url)
            req = urllib.request.Request(url, headers={"User-Agent": "deep-research-eval/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp.read().decode("utf-8").splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if rows:
                log.info("drb2_github_ok", url=url, rows=len(rows))
                return rows
        except Exception as exc:
            log.warning("drb2_github_failed", url=url, error=str(exc))

    return rows


# ── 3. DRB-I ─────────────────────────────────────────────────────────────────

def download_drb1(*, force: bool = False) -> int:
    """Download DRB-I (DeepResearch-Bench).

    Saves:
        - drb1_queries.json           (normalised queries)
        - drb1_human_annotations.json (human RACE annotations)
        - drb1_system_reports.json    (system-generated reports by model)

    Returns the number of queries written.
    """
    cache_dir = _BENCHMARKS_DIR / "drb1"
    queries_path = cache_dir / "drb1_queries.json"
    annotations_path = cache_dir / "drb1_human_annotations.json"
    reports_path = cache_dir / "drb1_system_reports.json"

    if not force and _cache_exists(queries_path):
        log.info("drb1_skip", reason="cache exists, use --force to re-download")
        return 0

    log.info("drb1_download_start", repo="muset-ai/DeepResearch-Bench-Dataset")
    _ensure_dir(cache_dir)

    try:
        from huggingface_hub import snapshot_download
        snap_dir = snapshot_download(
            repo_id="muset-ai/DeepResearch-Bench-Dataset",
            repo_type="dataset",
        )
        snap = Path(snap_dir)
    except Exception as exc:
        log.error("drb1_download_failed", error=str(exc))
        return 0

    log.info("drb1_snapshot_downloaded", path=str(snap))

    # ── Human annotations ────────────────────────────────────────────────
    annotations: List[Dict[str, Any]] = []
    annot_file = _find_file(snap, "human_RACE_annotation.jsonl")
    if annot_file:
        with open(annot_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        annotations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        _write_json(annotations_path, annotations, label="drb1_human_annotations")
    else:
        log.warning("drb1_no_annotations_file")

    # ── System reports ───────────────────────────────────────────────────
    # Look for generated_reports/ directory or any JSONL files with article content
    system_reports: Dict[str, List[Dict[str, Any]]] = {}
    reports_dir = _find_dir(snap, "generated_reports")
    if reports_dir:
        for jsonl_file in sorted(reports_dir.glob("*.jsonl")):
            model_name = jsonl_file.stem
            entries: List[Dict[str, Any]] = []
            with open(jsonl_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if entries:
                system_reports[model_name] = entries
                log.info("drb1_system_reports_loaded", model=model_name, count=len(entries))
    else:
        # Fallback: look for any JSONL files with "prompt" and "article" keys
        for jsonl_file in sorted(snap.rglob("*.jsonl")):
            if "annotation" in jsonl_file.name.lower():
                continue
            entries = []
            with open(jsonl_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            if "prompt" in row and "article" in row:
                                entries.append(row)
                        except json.JSONDecodeError:
                            continue
            if entries:
                model_name = jsonl_file.stem
                system_reports[model_name] = entries
                log.info("drb1_fallback_reports", model=model_name, count=len(entries))

    if system_reports:
        _write_json(reports_path, system_reports, label="drb1_system_reports")

    # ── Queries ──────────────────────────────────────────────────────────
    # Extract unique queries from system reports (any model's file works)
    seen_ids: set = set()
    queries: List[Dict[str, Any]] = []

    for model_name, entries in system_reports.items():
        for entry in entries:
            qid = entry.get("id")
            if qid is None:
                continue
            str_id = f"drb1_{qid}"
            if str_id in seen_ids:
                continue
            seen_ids.add(str_id)

            queries.append({
                "id": str_id,
                "query": entry.get("prompt", ""),
                "domain": entry.get("domain", "research").lower(),
                "difficulty": "complex",  # PhD-level
                "reference_answer": "",
                "expected_citations": [],
                "metadata": {
                    "original_id": qid,
                    "source_model": model_name,
                },
            })

    # If we found no queries from system reports, try annotations
    if not queries and annotations:
        for annot in annotations:
            qid = annot.get("id")
            if qid is None:
                continue
            str_id = f"drb1_{qid}"
            if str_id in seen_ids:
                continue
            seen_ids.add(str_id)

            queries.append({
                "id": str_id,
                "query": "",  # Annotations may not have the prompt text
                "domain": "research",
                "difficulty": "complex",
                "reference_answer": "",
                "expected_citations": [],
                "metadata": {
                    "original_id": qid,
                    "annotation_id": annot.get("annotation_id", ""),
                    "dimension_scores": annot.get("dimension_scores", {}),
                    "overall_scores": annot.get("overall_scores", {}),
                },
            })

    # Sort by ID for reproducibility
    queries.sort(key=lambda q: q["id"])

    _write_json(queries_path, queries, label="drb1_queries")
    log.info("drb1_done", queries=len(queries),
             annotations=len(annotations),
             system_report_models=len(system_reports))
    return len(queries)


def _find_file(root: Path, name: str) -> Optional[Path]:
    """Recursively search for a file by name under *root*."""
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def _find_dir(root: Path, name: str) -> Optional[Path]:
    """Recursively search for a directory by name under *root*."""
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None


# ── 4. DR.BENCH ──────────────────────────────────────────────────────────────

def download_drbench(*, force: bool = False) -> int:
    """Download DR.BENCH (EVIGBYEN/DrBench).

    Returns the number of queries written.
    """
    cache_dir = _BENCHMARKS_DIR / "drbench"
    cache_path = cache_dir / "drbench_queries.json"

    if not force and _cache_exists(cache_path):
        log.info("drbench_skip", reason="cache exists, use --force to re-download")
        return 0

    log.info("drbench_download_start", repo="EVIGBYEN/DrBench")
    _ensure_dir(cache_dir)

    rows: List[Dict[str, Any]] = []

    # Strategy A: try specific file download
    try:
        from huggingface_hub import hf_hub_download
        for filename in ("DrBench.jsonl", "drbench.jsonl", "data.jsonl"):
            try:
                local = hf_hub_download(
                    repo_id="EVIGBYEN/DrBench",
                    filename=filename,
                    repo_type="dataset",
                )
                with open(local, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                if rows:
                    log.info("drbench_hf_file", filename=filename, rows=len(rows))
                    break
            except Exception:
                continue
    except ImportError:
        pass

    # Strategy B: datasets library
    if not rows:
        try:
            from datasets import load_dataset
            ds = load_dataset("EVIGBYEN/DrBench", split="train")
            for row in ds:
                rows.append(dict(row))
            if rows:
                log.info("drbench_hf_dataset", rows=len(rows))
        except Exception as exc:
            log.warning("drbench_dataset_failed", error=str(exc))

    # Strategy C: snapshot download
    if not rows:
        try:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(
                repo_id="EVIGBYEN/DrBench",
                repo_type="dataset",
            )
            snap_path = Path(snap)
            for path in snap_path.rglob("*.jsonl"):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                if rows:
                    log.info("drbench_snapshot_jsonl", path=str(path), rows=len(rows))
                    break
            if not rows:
                for path in snap_path.rglob("*.parquet"):
                    try:
                        import pandas as pd
                        df = pd.read_parquet(path)
                        rows = df.to_dict(orient="records")
                        if rows:
                            log.info("drbench_snapshot_parquet", path=str(path), rows=len(rows))
                            break
                    except Exception:
                        continue
            if not rows:
                for path in snap_path.rglob("*.json"):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            rows = data
                        elif isinstance(data, dict) and "data" in data:
                            rows = data["data"]
                        if rows:
                            log.info("drbench_snapshot_json", path=str(path), rows=len(rows))
                            break
                    except Exception:
                        continue
        except Exception as exc:
            log.warning("drbench_snapshot_failed", error=str(exc))

    if not rows:
        log.error("drbench_download_failed", msg="All download strategies exhausted")
        return 0

    # Normalise
    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        uid = row.get("uid", f"drbench_{i:04d}")
        query_text = row.get("query", row.get("question", ""))
        domain = _infer_domain(query_text)

        queries.append({
            "id": str(uid),
            "query": query_text,
            "domain": domain,
            "difficulty": "moderate",  # default; dataset has mixed difficulty
            "reference_answer": "",
            "expected_citations": [],
            "metadata": {
                "qsr": row.get("qsr", []),
                "tsl": row.get("tsl", []),
                "fak": row.get("fak", []),
                "fdk": row.get("fdk", []),
                "raw_keys": list(row.keys()),
            },
        })

    _write_json(cache_path, queries, label="drbench")
    log.info("drbench_done", queries=len(queries))
    return len(queries)


def _infer_domain(query: str) -> str:
    """Best-effort domain inference from query text using keyword matching."""
    q = query.lower()

    domain_keywords = {
        "medicine": ["disease", "treatment", "clinical", "patient", "drug",
                     "therapy", "cancer", "symptom", "diagnosis", "medical",
                     "health", "surgical"],
        "computer_science": ["algorithm", "neural", "machine learning",
                             "software", "programming", "database", "model",
                             "training", "deep learning", "llm", "language model"],
        "biology": ["gene", "protein", "cell", "species", "organism",
                    "evolution", "biological", "genome", "dna", "rna",
                    "molecular"],
        "physics": ["quantum", "particle", "energy", "gravity",
                    "electromagnetic", "relativity", "thermodynamic"],
        "chemistry": ["reaction", "compound", "catalyst", "synthesis",
                      "molecular", "chemical", "polymer"],
        "economics": ["market", "inflation", "gdp", "economic", "fiscal",
                      "monetary", "trade", "financial"],
        "history": ["century", "war", "empire", "civilization", "historical",
                    "ancient", "colonial"],
        "law": ["court", "legal", "statute", "regulation", "constitutional",
                "jurisdiction", "precedent"],
        "mathematics": ["theorem", "proof", "equation", "topology",
                        "algebra", "calculus", "combinatorial"],
        "environmental_science": ["climate", "emission", "pollution",
                                  "ecosystem", "biodiversity", "sustainability"],
    }

    best_domain = "general"
    best_count = 0

    for domain, keywords in domain_keywords.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count = count
            best_domain = domain

    return best_domain


# ══ HUMAN-LABEL SETS (data/human_labels/, native schema preserved) ════════════
#
# These are NOT query sets — they are pre-existing human labels (physician /
# expert / annotator verdicts) used as ground truth for judge validation.  We
# therefore preserve the native schema and DO NOT coerce into the query schema.
# Each set writes:  raw human labels  +  a small normalised view  +  LICENCE.txt.


# ── 5. HealthBench meta_eval (OpenAI) ─────────────────────────────────────────

_HEALTHBENCH_BLOB = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_meta_eval.jsonl"
)


def download_healthbench(*, force: bool = False) -> int:
    """OpenAI HealthBench meta_eval: physician binary met/not-met grades.

    Register (B): ~60,896 meta-examples; multiple physician binary met/not-met
    grades per (completion, rubric criterion) pair; meta_eval jsonl ~136 MB.
    Licence: MIT (repo); no separate data licence on blob.

    Raw schema per row: anonymized_physician_ids[], binary_labels[bool],
    category, completion, completion_id, prompt[], prompt_id, rubric, canary.

    Returns the number of raw meta-examples written.
    """
    cache_dir = _HUMAN_LABELS_DIR / "healthbench"
    raw_path = cache_dir / "healthbench_meta_eval.jsonl"
    norm_path = cache_dir / "healthbench_normalised.jsonl"

    if not force and _cache_exists(raw_path):
        log.info("healthbench_skip", reason="cache exists, use --force to re-download")
        return 0

    _ensure_dir(cache_dir)
    log.info("healthbench_download_start", url=_HEALTHBENCH_BLOB)

    # Stream the public Azure blob straight to disk (136 MB).
    try:
        n_bytes = _urlopen_to_file(_HEALTHBENCH_BLOB, raw_path)
        log.info("healthbench_blob_ok", bytes=n_bytes, path=str(raw_path))
    except Exception as exc:  # noqa: BLE001
        log.error("healthbench_blob_failed", url=_HEALTHBENCH_BLOB, error=str(exc))
        return 0

    # Build a small normalised view: one row per physician verdict.
    rows = _read_jsonl_file(raw_path)
    normalised: List[Dict[str, Any]] = []
    for ex in rows:
        labels = ex.get("binary_labels", []) or []
        pids = ex.get("anonymized_physician_ids", []) or []
        for i, met in enumerate(labels):
            normalised.append({
                "completion_id": ex.get("completion_id", ""),
                "prompt_id": ex.get("prompt_id", ""),
                "physician_id": pids[i] if i < len(pids) else f"anon_{i}",
                "rubric_criterion": ex.get("rubric", ""),
                "category": ex.get("category", ""),
                "human_met": bool(met),
            })
    _write_jsonl(norm_path, normalised, label="healthbench_normalised")

    _write_licence(
        cache_dir, name="HealthBench meta_eval (OpenAI)",
        source="github.com/openai/simple-evals (blob: openaipublic.blob.core.windows.net)",
        licence="MIT (repo); no separate data licence on blob",
    )

    log.info("healthbench_done", meta_examples=len(rows), verdicts=len(normalised))
    return len(rows)


# ── 6. ExpertQA ───────────────────────────────────────────────────────────────

def download_expertqa(*, force: bool = False) -> int:
    """ExpertQA: per-claim expert factuality (5-point) AND attribution labels.

    Register (B): 2,177 validated long-form answers; per-claim expert labels —
    factuality (correctness, 5-point) and attribution/support labelled
    SEPARATELY on the same claims; +informativeness, reliability. Licence: MIT.

    Raw row: question, annotator_id, answers{model:{answer_string, claims[],
    attribution, usefulness, ...}}, metadata{field, specific_field, ...}.
    Each claim: claim_string, evidence, support, correctness, informativeness,
    reliability, worthiness, ...

    Returns the number of raw answer records written.
    """
    cache_dir = _HUMAN_LABELS_DIR / "expertqa"
    raw_path = cache_dir / "expertqa_r2_compiled_anon.jsonl"
    claims_path = cache_dir / "expertqa_claims_normalised.jsonl"

    if not force and _cache_exists(raw_path):
        log.info("expertqa_skip", reason="cache exists, use --force to re-download")
        return 0

    _ensure_dir(cache_dir)
    log.info("expertqa_download_start", repo="cmalaviya/expertqa")

    local: Optional[str] = None
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id="cmalaviya/expertqa",
            filename="r2_compiled_anon.jsonl",
            repo_type="dataset",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("expertqa_hf_failed", error=str(exc))
        # GitHub-raw fallback.
        if _github_raw_first(
            [
                "https://raw.githubusercontent.com/chaitanyamalaviya/ExpertQA/main/data/r2_compiled_anon.jsonl",
                "https://raw.githubusercontent.com/chaitanyamalaviya/ExpertQA/master/data/r2_compiled_anon.jsonl",
            ],
            raw_path,
        ):
            local = str(raw_path)

    if not local:
        log.error("expertqa_download_failed", msg="HF and GitHub strategies exhausted")
        return 0

    rows = _read_jsonl_file(Path(local))
    if not rows:
        log.error("expertqa_empty")
        return 0

    # Preserve raw (copy HF cache into our dir if it was not already streamed).
    if Path(local) != raw_path:
        _write_jsonl(raw_path, rows, label="expertqa_raw")

    # Normalised per-claim view: factuality + attribution on the same claim.
    claims: List[Dict[str, Any]] = []
    for ri, row in enumerate(rows):
        meta = row.get("metadata", {}) or {}
        for model, ans in (row.get("answers", {}) or {}).items():
            for ci, claim in enumerate(ans.get("claims", []) or []):
                claims.append({
                    "question": row.get("question", ""),
                    "field": meta.get("field", ""),
                    "specific_field": meta.get("specific_field", ""),
                    "answer_model": model,
                    "claim_id": f"{ri}_{model}_{ci}",
                    "claim_string": claim.get("claim_string", ""),
                    "correctness": claim.get("correctness", ""),     # 5-point factuality
                    "support": claim.get("support", ""),             # attribution
                    "informativeness": claim.get("informativeness", ""),
                    "reliability": claim.get("reliability", ""),
                    "worthiness": claim.get("worthiness", ""),
                })
    _write_jsonl(claims_path, claims, label="expertqa_claims")

    _write_licence(
        cache_dir, name="ExpertQA",
        source="github.com/chaitanyamalaviya/ExpertQA (HF: cmalaviya/expertqa)",
        licence="MIT",
    )

    log.info("expertqa_done", answers=len(rows), claims=len(claims))
    return len(rows)


# ── 7. DeepFact-Bench (Amazon) ────────────────────────────────────────────────

def download_deepfactbench(*, force: bool = False) -> int:
    """DeepFact-Bench: claim-level SUPPORTED/CONTRADICTORY/INCONCLUSIVE verdicts.

    Register (A): PhD-domain-specialist verdicts on sentences of actual
    deep-research reports across 5 technical domains; claims carry inline
    citations. v1.0.0, 6.05 MB. Licence: MIT.

    Raw test.json: list of claim rows (sentence, human_verdict, agent_verdict,
    report_id, domain, relevance, ...).  test_reports.json: {report_id: report}.

    Returns the number of claim-level rows written.
    """
    cache_dir = _HUMAN_LABELS_DIR / "deepfactbench"
    raw_claims = cache_dir / "deepfactbench_test.json"
    raw_reports = cache_dir / "deepfactbench_test_reports.json"
    norm_path = cache_dir / "deepfactbench_normalised.jsonl"

    if not force and _cache_exists(raw_claims):
        log.info("deepfactbench_skip", reason="cache exists, use --force to re-download")
        return 0

    _ensure_dir(cache_dir)
    log.info("deepfactbench_download_start", repo="kkkevinkkk/DeepFactBench")

    claims_data: Optional[Any] = None
    reports_data: Optional[Any] = None
    try:
        from huggingface_hub import hf_hub_download
        cp = hf_hub_download("kkkevinkkk/DeepFactBench", "test.json", repo_type="dataset")
        claims_data = json.loads(Path(cp).read_text(encoding="utf-8"))
        try:
            rp = hf_hub_download(
                "kkkevinkkk/DeepFactBench", "test_reports.json", repo_type="dataset")
            reports_data = json.loads(Path(rp).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("deepfactbench_reports_missing", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.warning("deepfactbench_hf_failed", error=str(exc))
        # GitHub-raw fallback (code repo carries the data under data/).
        tmp = cache_dir / "_tmp_test.json"
        if _github_raw_first(
            [
                "https://raw.githubusercontent.com/kkkevinkkkkk/DeepFact/main/data/test.json",
                "https://raw.githubusercontent.com/kkkevinkkkkk/DeepFact/master/data/test.json",
            ],
            tmp,
        ):
            claims_data = json.loads(tmp.read_text(encoding="utf-8"))
            tmp.unlink(missing_ok=True)

    if claims_data is None:
        log.error("deepfactbench_download_failed", msg="HF and GitHub strategies exhausted")
        return 0

    # Persist raw blobs verbatim.
    raw_claims.write_text(json.dumps(claims_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if reports_data is not None:
        raw_reports.write_text(
            json.dumps(reports_data, ensure_ascii=False, indent=2), encoding="utf-8")

    claim_rows = claims_data if isinstance(claims_data, list) else claims_data.get("data", [])

    # Normalised view: the human verdict per claim sentence.
    normalised: List[Dict[str, Any]] = []
    for i, c in enumerate(claim_rows):
        normalised.append({
            "claim_id": f"dfb_{i:05d}",
            "report_id": c.get("report_id", ""),
            "domain": c.get("domain", ""),
            "sentence": c.get("sentence", ""),
            "human_verdict": c.get("human_verdict", ""),
            "agent_verdict": c.get("agent_verdict", ""),
            "relevance": c.get("relevance", ""),
            "split": c.get("split", ""),
        })
    _write_jsonl(norm_path, normalised, label="deepfactbench_normalised")

    _write_licence(
        cache_dir, name="DeepFact-Bench (Amazon) v1.0.0",
        source="huggingface.co/datasets/kkkevinkkk/DeepFactBench (code: github.com/kkkevinkkkkk/DeepFact)",
        licence="MIT",
    )

    log.info("deepfactbench_done", claims=len(normalised),
             reports=len(reports_data) if isinstance(reports_data, dict) else 0)
    return len(normalised)


# ── 8. LongJudgeBench ─────────────────────────────────────────────────────────

# Released gold-label files under ground_truth/ (verified live 2026-06-15).
# realdr_gt.jsonl is a 93-byte placeholder ("released after paper acceptance"),
# captured but flagged as not-yet-available.
_LJB_GT_FILES = [
    "deepresearch_bench_gt.jsonl",
    "realdr_gt.jsonl",
    "ma_gt.jsonl",
    "surge_gt.jsonl",
    "verify_bench_hard_gt.jsonl",
    "wp_bench_gt.jsonl",
]
_LJB_REPO = "cjj826/LongJudgeBench"
_LJB_RAW = "https://raw.githubusercontent.com/cjj826/LongJudgeBench/main/ground_truth/"


def download_longjudgebench(*, force: bool = False) -> int:
    """LongJudgeBench: gold pointwise/pairwise labels on long-form judging.

    Register (A): gold labels incl. deepresearch_bench_gt.jsonl and
    realdr_gt.jsonl deep-research subsets (filter to EN). Licence: MIT (repo);
    verify per-subset gold provenance in the paper.

    Strategy: GitHub HF-space mirror not present -> use HF snapshot if the repo
    is mirrored, else GitHub-raw per ground_truth file. Preserves each GT file
    verbatim and builds a normalised view of the deep-research GT subset.

    Returns the number of gold records across all fetched GT files.
    """
    cache_dir = _HUMAN_LABELS_DIR / "longjudgebench"
    gt_dir = cache_dir / "ground_truth"
    norm_path = cache_dir / "deepresearch_bench_gt_normalised.jsonl"
    marker = gt_dir / "deepresearch_bench_gt.jsonl"

    if not force and _cache_exists(marker):
        log.info("longjudgebench_skip", reason="cache exists, use --force to re-download")
        return 0

    _ensure_dir(gt_dir)
    log.info("longjudgebench_download_start", repo=_LJB_REPO)

    fetched: Dict[str, Path] = {}
    # GitHub-raw per file (the gold lives in the code repo, not on HF).
    for fname in _LJB_GT_FILES:
        dest = gt_dir / fname
        if _github_raw_first([_LJB_RAW + fname], dest):
            fetched[fname] = dest
        else:
            log.warning("longjudgebench_gt_404", file=fname)

    if not fetched:
        log.error("longjudgebench_download_failed", msg="no GT files fetched")
        return 0

    # Count gold records across all fetched GT files and detect placeholders.
    total = 0
    per_file: Dict[str, int] = {}
    placeholders: List[str] = []
    for fname, path in fetched.items():
        text = path.read_text(encoding="utf-8").strip()
        rows = _read_jsonl_file(path)
        if not rows:
            placeholders.append(fname)  # non-JSONL note (e.g. realdr placeholder)
        per_file[fname] = len(rows)
        total += len(rows)
    log.info("longjudgebench_gt_counts", per_file=per_file, placeholders=placeholders)

    # Normalised view of the in-genre deep-research GT (per-annotator dim scores).
    dr_gt = gt_dir / "deepresearch_bench_gt.jsonl"
    normalised: List[Dict[str, Any]] = []
    if dr_gt.exists():
        for rec in _read_jsonl_file(dr_gt):
            for ann in rec.get("raw_annotations", []) or []:
                for model, dims in (ann.get("dimension_scores", {}) or {}).items():
                    normalised.append({
                        "dataset": rec.get("dataset", "deepresearch_bench"),
                        "task_id": rec.get("id", ""),
                        "language": "zh",  # deepresearch_bench subset is zh
                        "annotation_id": ann.get("annotation_id", ""),
                        "model": model,
                        "dimension_scores": dims,
                        "overall_score": (ann.get("overall_scores", {}) or {}).get(model),
                    })
        _write_jsonl(norm_path, normalised, label="longjudgebench_dr_gt")

    note = (
        "Gold files under ground_truth/. deepresearch_bench_gt is zh; realdr_gt "
        "is a placeholder ('released after paper acceptance', 93 bytes) — EN "
        "deep-research gold not yet public. Verify per-subset provenance per "
        "the register before calling any subset 'human gold'."
    )
    _write_licence(
        cache_dir, name="LongJudgeBench",
        source="github.com/cjj826/LongJudgeBench",
        licence="MIT (repo); per-subset provenance varies",
        note=note,
    )

    log.info("longjudgebench_done", gt_files=len(fetched),
             gold_records=total, dr_normalised=len(normalised))
    return total


# ── 9. DRACO full release (Perplexity) ────────────────────────────────────────

def download_draco_full(*, force: bool = False) -> int:
    """DRACO full release: 100 tasks / 3,934 expert binary MET/UNMET criteria.

    Register (D): 100 real-user deep-research tasks with expert-curated binary
    rubrics by 26 domain experts; single test.jsonl. Licence: MIT.

    Written to data/human_labels/draco_full/ — distinct from the existing
    data/benchmarks/draco/ 40-task study subset, which is NOT touched.

    Raw row: id, problem, answer (JSON string with sections[].criteria[]), domain.

    Returns the number of tasks written.
    """
    cache_dir = _HUMAN_LABELS_DIR / "draco_full"
    raw_path = cache_dir / "draco_test.jsonl"
    crit_path = cache_dir / "draco_criteria_normalised.jsonl"

    if not force and _cache_exists(raw_path):
        log.info("draco_full_skip", reason="cache exists, use --force to re-download")
        return 0

    _ensure_dir(cache_dir)
    log.info("draco_full_download_start", repo="perplexity-ai/draco")

    local: Optional[str] = None
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download("perplexity-ai/draco", "test.jsonl", repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        log.warning("draco_full_hf_failed", error=str(exc))
        if _github_raw_first(
            [
                "https://raw.githubusercontent.com/perplexity-ai/draco/main/test.jsonl",
                "https://raw.githubusercontent.com/perplexity-ai/draco/master/test.jsonl",
            ],
            raw_path,
        ):
            local = str(raw_path)

    if not local:
        log.error("draco_full_download_failed", msg="HF and GitHub strategies exhausted")
        return 0

    rows = _read_jsonl_file(Path(local))
    if not rows:
        log.error("draco_full_empty")
        return 0

    if Path(local) != raw_path:
        _write_jsonl(raw_path, rows, label="draco_full_raw")

    # Normalised criteria view: explode the rubric embedded in `answer`.
    criteria: List[Dict[str, Any]] = []
    for task in rows:
        tid = task.get("id", "")
        domain = task.get("domain", "")
        try:
            rubric = json.loads(task.get("answer", "")) if task.get("answer") else {}
        except json.JSONDecodeError:
            rubric = {}
        for section in rubric.get("sections", []) or []:
            sec_id = section.get("id", section.get("title", ""))
            for crit in section.get("criteria", []) or []:
                criteria.append({
                    "task_id": tid,
                    "domain": domain,
                    "section": sec_id,
                    "criterion_id": crit.get("id", ""),
                    "weight": crit.get("weight"),
                    "requirement": crit.get("requirement", ""),
                })
    _write_jsonl(crit_path, criteria, label="draco_full_criteria")

    _write_licence(
        cache_dir, name="DRACO full release (Perplexity)",
        source="huggingface.co/datasets/perplexity-ai/draco",
        licence="MIT",
    )

    log.info("draco_full_done", tasks=len(rows), criteria=len(criteria))
    return len(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

_DOWNLOADERS = {
    "research_rubrics": download_research_rubrics,
    "drb2": download_drb2,
    "drb1": download_drb1,
    "drbench": download_drbench,
    "healthbench": download_healthbench,
    "expertqa": download_expertqa,
    "deepfactbench": download_deepfactbench,
    "longjudgebench": download_longjudgebench,
    "draco_full": download_draco_full,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download external benchmark datasets and normalise to cache format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_benchmarks.py --all\n"
            "  python scripts/download_benchmarks.py --benchmark research_rubrics\n"
            "  python scripts/download_benchmarks.py --benchmark drb1 --benchmark drb2 --force\n"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=AVAILABLE_BENCHMARKS,
        dest="benchmarks",
        help="Benchmark(s) to download. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available benchmarks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cache already exists.",
    )
    args = parser.parse_args()

    if not args.all and not args.benchmarks:
        parser.error("Specify --all or at least one --benchmark NAME")

    targets = AVAILABLE_BENCHMARKS if args.all else args.benchmarks

    print("=" * 70)
    print("  Benchmark Download Script")
    print("=" * 70)
    print(f"  Targets: {', '.join(targets)}")
    print(f"  Force:   {args.force}")
    print(f"  Output:  {_BENCHMARKS_DIR} (query sets)")
    print(f"           {_HUMAN_LABELS_DIR} (human-label sets)")
    print("=" * 70)
    print()

    summary: Dict[str, str] = {}

    for name in targets:
        downloader = _DOWNLOADERS[name]
        print(f"--- {name} ---")
        try:
            count = downloader(force=args.force)
            if count > 0:
                summary[name] = f"OK ({count} records)"
            else:
                summary[name] = "SKIPPED (cached or empty)"
        except Exception as exc:
            log.error("benchmark_download_error", benchmark=name, error=str(exc))
            summary[name] = f"FAILED: {exc}"
        print()

    # Print summary
    print("=" * 70)
    print("  Download Summary")
    print("=" * 70)
    for name, status in summary.items():
        print(f"  {name:25s} {status}")
    print("=" * 70)

    # Exit with error code if any failed
    if any(s.startswith("FAILED") for s in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
