#!/usr/bin/env python3
"""REGENERATE-ALL-ARMS — frozen-source vintage curve generation runner.

For each model arm (sequentially, one model resident in VRAM at a time), regenerate ALL 90
reports by reading the FROZEN evidence store built by scripts/freeze_vintage_sources.py. The
generator input (evidence_text) is byte-identical across every arm, so the ONLY variable is
the injected backbone (its release vintage / capacity). This is the generation half of the
frozen_vintage experiment; judging is a separate, paid, human-launched step (printed at the end).

Arms (curve = model RELEASE VINTAGE; 14B is a SAME-VINTAGE capacity point, not a date)
-------------------------------------------------------------------------------------
  base_p9                          Qwen2.5-7B-Instruct            2024-09  transformers 4-bit  (x=0 anchor)
  base_p14_vintage_deepseek_qwen7b DeepSeek-R1-Distill-Qwen-7B    2025-01  transformers 4-bit  (reasoning; <think> stripped)
  base_p13_vintage_qwen3_8b        Qwen3-8B                       2025-04  transformers 4-bit  (VERIFY load; may OOM)
  base_p17_scale_qwen25_14b        Qwen2.5-14B-Instruct           2024-09  GGUF/llama.cpp      (CAPACITY point, same vintage as P9)

Determinism
-----------
  * gen_temperature=0.0 -> LocalLLMCaller maps temperature<=0.01 to do_sample=False (greedy
    argmax); LlamaCppLLMCaller is ALWAYS strict-greedy (temp 0, top_k 1, top_p 1, seed 42).
  * torch.manual_seed(0) + transformers.set_seed(0) before each arm (greedy makes the seed
    moot but pins any tie-break).
  * max_tokens=4096 held constant (p9 hardcodes it) so truncation is identical across arms.
  * The FREEZE removed ALL retrieval/extraction nondeterminism; this leaves only CUDA matmul
    token-level nondeterminism (rare near-ties), which is accepted and documented.
  * The injected caller is reused across all 90 queries per arm; p9 rebinds llm.cost_tracker
    to a fresh per-run CostTracker each call, so per-query accounting stays correct and no
    generation state leaks across the loop.

Corpus safety (HARD)
--------------------
Writes ONLY under results/experiments_frozen_vintage/<out_pattern>/<query_id>.md (a NEW root).
NEVER results/experiments, results/judge_gpt52, data/analysis. Reads the frozen store
(data/frozen_corpus_vintage, read-only) — NO web/academic/extraction is ever performed here.
Resumable: an existing non-empty <query_id>.md is skipped. Per-query 600s timeout guard.

Usage:
    [ -f venv/bin/activate ] && source venv/bin/activate
    python scripts/run_frozen_vintage.py --dry-run                  # plan only, no model load
    python scripts/run_frozen_vintage.py --self-test               # offline wiring check (no model/net)
    python scripts/run_frozen_vintage.py --arm base_p9 --limit 1   # smoke one arm, one query
    python scripts/run_frozen_vintage.py --arm base_p9             # one arm, all 90 frozen queries
    python scripts/run_frozen_vintage.py                           # ALL arms, sequentially, 90 each
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── CUDA library path for the sm_120 llama.cpp build (set BEFORE native import) ──
def _ensure_cuda_ld_library_path() -> None:
    parts: list[str] = []
    cudatk = REPO_ROOT / ".cudatk" / "lib"
    if cudatk.is_dir():
        parts.append(str(cudatk))
    nvidia_root = REPO_ROOT / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        for lib in sorted(nvidia_root.glob("*/lib")):
            if lib.is_dir():
                parts.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = existing.split(":") if existing else []
    new_parts = [p for p in parts if p not in existing_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


_ensure_cuda_ld_library_path()

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_QUERIES = REPO_ROOT / "data" / "eval_queries_v2.json"
FROZEN_DIR = REPO_ROOT / "data" / "frozen_corpus_vintage"
OUTPUT_ROOT = REPO_ROOT / "results" / "experiments_frozen_vintage"
GGUF_PATH = REPO_ROOT / "models" / "gguf" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"

FORBIDDEN_PREFIXES = [
    REPO_ROOT / "results" / "judge_gpt52",
    REPO_ROOT / "results" / "experiments",      # the protected base study
    REPO_ROOT / "data" / "analysis",
    REPO_ROOT / "reports" / "eval_v2" / "verdicts",
]

DEFAULT_BUDGET_USD = 2.0
N_QUERIES = 90
GEN_TEMPERATURE = 0.0   # greedy / deterministic generation for every arm

# ── Arm registry ───────────────────────────────────────────────────────────────
# out_pattern is BOTH the on-disk subdir AND the --patterns-raw name the judge reads.
# backbone: "transformers" (LocalLLMCaller 4-bit) or "gguf" (LlamaCppLLMCaller).
# strip_think: enable <think>...</think> stripping for reasoning models.
ARMS = [
    {
        "out_pattern": "base_p9",
        "module": "deep_research.patterns.p9_local_baseline.pipeline",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "release_date": "2024-09",
        # Phase-3 de-confound: run via GGUF so ALL 4 arms share the llama.cpp greedy
        # backend, removing the decode-backend confound on the vintage axis.
        "backbone": "gguf",
        "gguf_path": REPO_ROOT / "models" / "gguf" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "strip_think": False,
        "axis": "vintage",
    },
    {
        "out_pattern": "base_p14_vintage_deepseek_qwen7b",
        "module": "deep_research.patterns.p14_vintage_deepseek_qwen7b.pipeline",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "release_date": "2025-01",
        # Phase-3 de-confound: GGUF backend (shared with all arms).
        "backbone": "gguf",
        "gguf_path": REPO_ROOT / "models" / "gguf" / "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "strip_think": True,   # reasoning model emits <think> chains
        "axis": "vintage",
    },
    {
        "out_pattern": "base_p13_vintage_qwen3_8b",
        "module": "deep_research.patterns.p13_vintage_qwen3_8b.pipeline",
        "model_id": "Qwen/Qwen3-8B",
        "release_date": "2025-04",
        # 4-bit transformers OOMs (loads to ~15.6GB of 15.47 usable, no room to generate);
        # the llama.cpp GGUF path that runs the 14B also runs the standard-arch Qwen3-8B.
        "backbone": "gguf",
        "gguf_path": REPO_ROOT / "models" / "gguf" / "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "strip_think": True,   # Qwen3 can think; harmless if none present
        "axis": "vintage",
    },
    {
        "out_pattern": "base_p17_scale_qwen25_14b",
        "module": "deep_research.patterns.p17_scale_qwen25_14b.pipeline",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "release_date": "2024-09",   # SAME vintage as P9 — capacity axis, x=0
        "backbone": "gguf",
        "gguf_path": REPO_ROOT / "models" / "gguf" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "strip_think": False,
        "axis": "capacity",
    },
]
ARMS_BY_NAME = {a["out_pattern"]: a for a in ARMS}


# ── Query loading ──────────────────────────────────────────────────────────────
def load_queries(limit: int) -> list[dict]:
    data = json.loads(EVAL_QUERIES.read_text())
    items = data["queries"] if isinstance(data, dict) else data
    items_sorted = sorted(items, key=lambda q: str(q["id"]))
    if limit and limit > 0:
        items_sorted = items_sorted[:limit]
    return items_sorted


# ── Output path / safety ───────────────────────────────────────────────────────
def _report_path(out_pattern: str, query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return OUTPUT_ROOT / out_pattern / f"{safe_id}.md"


def _assert_corpus_safe(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(
            f"CORPUS-SAFETY VIOLATION: refusing to write outside {OUTPUT_ROOT}: {resolved}"
        )
    for forbidden in FORBIDDEN_PREFIXES:
        fr = forbidden.resolve()
        if fr == resolved or fr in resolved.parents:
            raise RuntimeError(
                f"CORPUS-SAFETY VIOLATION: path under forbidden prefix {forbidden}: {resolved}"
            )


def is_done(out_pattern: str, query_id: str) -> bool:
    p = _report_path(out_pattern, query_id)
    return p.exists() and p.stat().st_size > 0


def _frozen_path(query_id: str) -> Path:
    safe_id = str(query_id).replace("/", "_").replace("\\", "_")
    return FROZEN_DIR / f"{safe_id}.json"


def is_frozen(query_id: str) -> bool:
    return _frozen_path(query_id).exists() and _frozen_path(query_id).stat().st_size > 0


# ── Determinism / VRAM helpers ─────────────────────────────────────────────────
def _seed_everything() -> None:
    try:
        import torch
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except Exception:
        pass
    try:
        import transformers
        transformers.set_seed(0)
    except Exception:
        pass


def _nvidia_smi_used_mb() -> float | None:
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def print_vram(tag: str) -> None:
    used = _nvidia_smi_used_mb()
    print(f"  VRAM ({tag}): {used:.0f} MiB used" if used is not None
          else f"  VRAM ({tag}): nvidia-smi unavailable")


def free_vram(backbone: str) -> None:
    """Evict whatever model is resident so the next arm starts from a clean card."""
    try:
        if backbone == "gguf":
            from deep_research.tools.llamacpp_llm_caller import unload_model
            unload_model()
        else:
            from deep_research.tools import local_llm_caller
            if getattr(local_llm_caller, "_loaded_model", None):
                m = local_llm_caller._loaded_model
                for k in ("model", "tokenizer"):
                    if k in m:
                        del m[k]
                local_llm_caller._loaded_model = None
    except Exception as e:
        print(f"  [warn] free_vram({backbone}) failed: {e}")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ── Caller construction (one per arm, reused across 90 queries) ─────────────────
def build_caller(arm: dict):
    """Construct the arm's LLM caller ONCE. CostTracker is rebound per-run inside p9."""
    if arm["backbone"] == "gguf":
        from deep_research.tools.llamacpp_llm_caller import LlamaCppLLMCaller
        GGUF = arm.get("gguf_path", GGUF_PATH)
        if not GGUF.exists():
            raise FileNotFoundError(f"GGUF model not found at {GGUF}")
        # n_ctx must hold the full frozen prompt (~7.9K tokens: REPORT_PROMPT + 6000-word
        # evidence) PLUS the 4096-token generation budget, so the 14B consumes the SAME context
        # as the transformers arms (32K-128K windows). Default 4096 would silently truncate the
        # 14B to ~half the evidence, breaking the byte-identical-context guarantee. 16384 leaves margin.
        return LlamaCppLLMCaller(model_path=str(GGUF), model_id=arm["model_id"], n_ctx=16384)
    from deep_research.tools.local_llm_caller import LocalLLMCaller
    return LocalLLMCaller(model_id=arm["model_id"], quantize_4bit=True)


# ── Single report on the frozen store ──────────────────────────────────────────
async def run_one(mod, arm: dict, caller, query: dict, budget: float) -> dict:
    query_id = query["id"]
    out_path = _report_path(arm["out_pattern"], query_id)
    _assert_corpus_safe(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the frozen evidence for this query and inject it. The pipeline's frozen
    # branch verifies corpus_sha256 before generating, so the input is provably the
    # frozen one. We pass the loaded dict directly (frozen_evidence) so each arm reads
    # byte-identical text; frozen_corpus_dir would also work but this avoids re-reads.
    fp = _frozen_path(query_id)
    if not (fp.exists() and fp.stat().st_size > 0):
        return {"query_id": query_id, "status": "no_frozen",
                "error": f"missing frozen corpus {fp.name}; run freeze_vintage_sources.py first"}
    frozen = json.loads(fp.read_text())

    t0 = time.time()
    try:
        report = await mod.run(
            query["query"],
            budget_usd=budget,
            query_id=query_id,
            llm=caller,
            frozen_evidence=frozen,
            gen_temperature=GEN_TEMPERATURE,
            strip_think=arm["strip_think"],
        )
        text = report.full_text()
        if not text.strip():
            text = f"# (empty report)\n\nQuery: {query['query']}\n"
        out_path.write_text(text)
        return {
            "query_id": query_id, "status": "success",
            "out_pattern": arm["out_pattern"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "chars": len(text),
            "tokens": getattr(report, "total_tokens", 0),
            "sections": len(getattr(report, "sections", [])),
            "citations": len(getattr(report, "citations", [])),
            "corpus_sha256": frozen.get("corpus_sha256"),
        }
    except Exception as e:
        return {"query_id": query_id, "status": "error",
                "out_pattern": arm["out_pattern"],
                "elapsed_seconds": round(time.time() - t0, 1),
                "error": str(e)[:300]}


# ── One arm, all queries ───────────────────────────────────────────────────────
async def run_arm(arm: dict, queries: list[dict], budget: float, resume: bool) -> dict:
    print(f"\n{'='*64}\nARM {arm['out_pattern']}  ({arm['model_id']}, {arm['release_date']}, "
          f"{arm['backbone']}, axis={arm['axis']})\n{'='*64}")
    _seed_everything()

    mod = importlib.import_module(arm["module"])

    work = [q for q in queries if is_frozen(q["id"]) and not (resume and is_done(arm["out_pattern"], q["id"]))]
    missing_frozen = [q["id"] for q in queries if not is_frozen(q["id"])]
    if missing_frozen:
        print(f"  [warn] {len(missing_frozen)} queries have NO frozen corpus and are skipped: "
              f"{missing_frozen[:5]}{' ...' if len(missing_frozen) > 5 else ''}")
    if not work:
        print("  nothing to generate (all done or none frozen)")
        return {"out_pattern": arm["out_pattern"], "status": "nothing_to_do", "results": []}

    # Build the caller ONCE; verify the model loads. For may_oom arms, a load failure
    # is recorded as oom_skipped (the curve then honestly has fewer points).
    try:
        caller = build_caller(arm)
    except Exception as e:
        print(f"  [FATAL] could not construct caller: {str(e)[:200]}")
        return {"out_pattern": arm["out_pattern"], "status": "caller_build_failed",
                "error": str(e)[:300], "results": []}

    results = []
    loaded_reported = False
    for i, q in enumerate(work, 1):
        print(f"  [{i}/{len(work)}] {q['id']} ...", flush=True)
        try:
            res = await asyncio.wait_for(run_one(mod, arm, caller, q, budget), timeout=600)
        except asyncio.TimeoutError:
            res = {"query_id": q["id"], "status": "timeout", "elapsed_seconds": 600.0,
                   "error": "per-query timeout (600s)"}
        except Exception as e:
            # First-query failure on a may_oom arm => treat as OOM-skip for the whole arm.
            msg = str(e).lower()
            if arm.get("may_oom") and i == 1 and ("out of memory" in msg or "cuda" in msg or "oom" in msg):
                print(f"  [oom] {arm['out_pattern']} failed to load/generate: {str(e)[:160]}")
                free_vram(arm["backbone"])
                return {"out_pattern": arm["out_pattern"], "status": "oom_skipped",
                        "error": str(e)[:300], "results": results}
            res = {"query_id": q["id"], "status": "error", "error": str(e)[:300]}
        results.append(res)
        if not loaded_reported:
            print_vram(f"after first {arm['model_id'].split('/')[-1]} load")
            loaded_reported = True
            # may_oom arm survived the first generation -> it fits; continue.
        st = res.get("status")
        if st == "success":
            print(f"    OK {res['elapsed_seconds']}s, {res['chars']} chars, "
                  f"{res['sections']} sections, {res['citations']} cites, sha={str(res.get('corpus_sha256',''))[:10]}")
        else:
            print(f"    {st.upper()} — {res.get('error','')[:140]}")

    print("  done — freeing VRAM before next arm ...")
    free_vram(arm["backbone"])
    print_vram("after unload")

    successes = sum(1 for r in results if r["status"] == "success")
    return {"out_pattern": arm["out_pattern"], "status": "completed",
            "successes": successes, "n": len(results), "results": results}


# ── Self-test (offline; no model load, no network) ─────────────────────────────
def self_test() -> int:
    """Validate runner wiring without loading any model or touching the network.

    Checks: arm registry well-formed; corpus-safety guard rejects forbidden paths and
    accepts the experiment dir; the pipeline frozen branch returns a parsed report from a
    fake frozen record using a stub greedy caller (proving evidence injection + sha guard +
    <think> stripping + gen_temperature plumbing all reach Stage 4 and skip Stages 1-3).
    """
    import hashlib

    # 1. Registry sanity.
    assert len(ARMS) == 4, "expected 4 arms"
    assert {a["axis"] for a in ARMS} == {"vintage", "capacity"}, "axis labels off"
    assert ARMS_BY_NAME["base_p17_scale_qwen25_14b"]["backbone"] == "gguf"

    # 2. Corpus-safety guard.
    _assert_corpus_safe(OUTPUT_ROOT / "base_p9" / "q.md")  # ok
    for bad in [REPO_ROOT / "results" / "experiments" / "base_p9" / "q.md",
                REPO_ROOT / "results" / "judge_gpt52" / "x.json"]:
        try:
            _assert_corpus_safe(bad)
            raise AssertionError(f"guard failed to reject {bad}")
        except RuntimeError:
            pass

    # 3. Frozen-branch round-trip through the REAL p9 pipeline with a stub caller.
    from deep_research.tools import format_extractions_as_evidence
    from deep_research.tools.source_extractor import SourceExtraction
    from deep_research.patterns.p9_local_baseline import pipeline as p9

    exts = [SourceExtraction(doc_id="d1", title="Src A", url="https://e.org/a",
                             summary="alpha " * 20, relevance_score=8,
                             key_findings=["k1"])]
    evidence_text = format_extractions_as_evidence(exts)
    sha = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    frozen = {
        "query_id": "selftest", "query": "Q?", "evidence_text": evidence_text,
        "extractions": [e.to_evidence_dict() for e in exts],
        "urls": ["https://e.org/a"], "corpus_sha256": sha, "frozen": True,
    }

    class _StubGreedyCaller:
        """Returns a fixed report; records that gen reached it with the frozen evidence."""
        model_id = "stub/greedy"
        cost_tracker = None
        seen = {}

        async def complete(self, prompt, model="", system="", temperature=0.3, max_tokens=4096):
            _StubGreedyCaller.seen = {"temperature": temperature,
                                      "evidence_in_prompt": evidence_text[:40] in prompt}
            # DeepSeek-style preamble to exercise <think> stripping.
            return ("<think>reasoning about the sources...</think>\n"
                    "# Title\n\n## Abstract\n\nFrozen-source report body [1].\n\n"
                    "## References\n[1] Src A — https://e.org/a\n")

    async def _drive():
        rep = await p9.run("Q?", budget_usd=2.0, query_id="selftest",
                           llm=_StubGreedyCaller(), frozen_evidence=frozen,
                           gen_temperature=0.0, strip_think=True)
        return rep

    report = asyncio.run(_drive())
    assert _StubGreedyCaller.seen.get("evidence_in_prompt"), "frozen evidence did NOT reach the generator prompt"
    assert _StubGreedyCaller.seen.get("temperature") == 0.0, "gen_temperature=0.0 did not reach llm.complete"
    full = report.full_text()
    assert "<think>" not in full, "<think> chain was NOT stripped before parse"
    assert "Title" in full or report.title, "report did not parse a title"

    # 4. sha mismatch must raise (integrity guard).
    bad_frozen = dict(frozen, corpus_sha256="deadbeef")
    try:
        asyncio.run(p9.run("Q?", budget_usd=2.0, query_id="selftest",
                           llm=_StubGreedyCaller(), frozen_evidence=bad_frozen,
                           gen_temperature=0.0, strip_think=True))
        raise AssertionError("sha integrity guard did NOT raise on a tampered corpus_sha256")
    except RuntimeError:
        pass

    print("[self-test] PASS: registry ok; corpus-safety guard active; frozen evidence reaches "
          "the generator (Stages 1-3 skipped); gen_temperature=0.0 plumbed; <think> stripped; "
          "sha-integrity guard raises on tamper.")
    return 0


# ── Main ───────────────────────────────────────────────────────────────────────
async def amain(args) -> int:
    queries = load_queries(args.limit)
    selected = ([ARMS_BY_NAME[args.arm]] if args.arm else ARMS)
    if args.arm and args.arm not in ARMS_BY_NAME:
        print(f"Unknown --arm {args.arm}; choices: {', '.join(ARMS_BY_NAME)}")
        return 2

    n_frozen = sum(1 for q in queries if is_frozen(q["id"]))
    print(f"{'='*64}\nFROZEN-SOURCE VINTAGE GENERATION PLAN\n{'='*64}")
    print(f"Queries:        {len(queries)} (sorted id); frozen on disk: {n_frozen}")
    print(f"Frozen store:   {FROZEN_DIR}")
    print(f"Output root:    {OUTPUT_ROOT}")
    print(f"Gen temperature:{GEN_TEMPERATURE} (greedy/deterministic)  max_tokens=4096")
    print(f"Arms:           {', '.join(a['out_pattern'] for a in selected)}")
    for a in selected:
        done = sum(1 for q in queries if is_done(a['out_pattern'], q['id']))
        print(f"  - {a['out_pattern']:34s} {a['model_id']:38s} {a['release_date']} "
              f"{a['backbone']:12s} done={done}")

    if args.dry_run:
        print("\nDRY RUN — no model loaded, nothing written.")
        print("\nJudging (separate, paid, human-launched) — one --patterns-raw per arm dir:")
        print("  JUDGE_RESULTS_BASE=results/experiments_frozen_vintage \\")
        print("    python scripts/run_gpt52_judge_namespaced.py \\")
        print("      --judge-out results/judge_gpt52_frozen_vintage \\")
        print(f"      --patterns-raw {','.join(a['out_pattern'] for a in ARMS)} --resume")
        return 0

    if n_frozen == 0:
        print("\n[abort] no frozen corpus found. Run scripts/freeze_vintage_sources.py first.")
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    arm_summaries = []
    for arm in selected:
        try:
            summary = await run_arm(arm, queries, args.budget, args.resume)
        except Exception as e:
            summary = {"out_pattern": arm["out_pattern"], "status": "arm_failed", "error": str(e)[:300]}
            free_vram(arm["backbone"])
        arm_summaries.append(summary)

    manifest = {
        "experiment": "frozen_vintage",
        "frozen_store": str(FROZEN_DIR),
        "output_root": str(OUTPUT_ROOT),
        "gen_temperature": GEN_TEMPERATURE,
        "max_tokens": 4096,
        "n_queries": len(queries),
        "elapsed_seconds": round(time.time() - overall_start, 1),
        "arms": arm_summaries,
    }
    mpath = OUTPUT_ROOT / "run_manifest_frozen_vintage.json"
    _assert_corpus_safe(mpath)
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n{'='*64}\nFROZEN-VINTAGE GENERATION COMPLETE\n{'='*64}")
    for s in arm_summaries:
        print(f"  {s['out_pattern']:34s} {s.get('status')}  "
              f"successes={s.get('successes','-')}/{s.get('n','-')}")
    print(f"Manifest: {mpath}")
    print("\nJudge next (paid, human-launched):")
    print("  JUDGE_RESULTS_BASE=results/experiments_frozen_vintage \\")
    print("    python scripts/run_gpt52_judge_namespaced.py \\")
    print("      --judge-out results/judge_gpt52_frozen_vintage \\")
    print(f"      --patterns-raw {','.join(a['out_pattern'] for a in ARMS)} --resume")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None,
                    help="Run a single arm by out_pattern (default: all arms sequentially).")
    ap.add_argument("--limit", type=int, default=N_QUERIES,
                    help=f"Queries (sorted-id slice). Default {N_QUERIES}.")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                    help="Per-run token budget ceiling (local inference is $0).")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="Re-generate reports even if a .md already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan. No model load, nothing written.")
    ap.add_argument("--self-test", action="store_true",
                    help="Offline wiring check (no model/network); validates frozen injection path.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
