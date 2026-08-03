#!/usr/bin/env python3
"""E10 noise-RL — HELD-OUT eval generation (per adapter), GPU-only, ZERO paid API.

This is the *generation* half of the E10 post-hoc held-out evaluation. It loads
each of the 8 trained QLoRA adapters in models/E10-arm*/ on top of the SAME
Qwen2.5-7B-Instruct 4-bit base used during training, then greedily generates one
research report per held-out eval query, writing each to a NEW report root
OUTSIDE the protected corpus so the namespaced GPT-5.2 judge can score it.

WHAT THIS DOES AND DOES NOT DO
------------------------------
  * GPU-only deterministic-as-possible generation (greedy, do_sample=False).
  * NO paid API. NO judge call. NO canonical write. The ONLY paid step is the
    SEPARATE GPT-5.2 judge invocation (scripts/run_gpt52_judge_namespaced.py).
  * Also computes the judge-FREE objective (anti-Goodhart) score per arm on the
    answer-checkable held-out slice (CPU, deterministic, no LLM).

PIPELINE-MATCHED GENERATION
---------------------------
Reuses the EXACT training scaffold from scripts/train_p12_rl_v2.py so the
generated report distribution matches what the policy was trained to produce:
  * same base "Qwen/Qwen2.5-7B-Instruct", same BitsAndBytesConfig (4bit nf4,
    bf16 compute, double-quant), same tokenizer (pad=eos),
  * p12.build_dataset's EXACT prompt string (imported, never duplicated).
The adapter is loaded as a SINGLE-adapter PeftModel (the trainable POLICY LoRA
saved at the arm-dir top level by trainer.save_model). The frozen DR-Judge LoRA
in the arm-dir judge/ subdir is NOT loaded — GPT-5.2 is the post-hoc judge, so
no multi-adapter set_adapter dance is needed for eval.

DETERMINISM (prereg: reproducibility unit = cross-seed variance, NOT bit-identity)
----------------------------------------------------------------------------------
Greedy decode + num_beams=1 + fixed max_new_tokens + a fixed GEN_SEED before the
loop makes runs near-deterministic per machine. Bit-identity across machines is
NOT claimed (GPU/4bit nondeterminism). The seeded noise layer was already baked
into the trained adapters; here the cross-arm/cross-seed differences reflect the
ADAPTER, not sampling.

ARMS (auto-discovered — the 8 dirs are NOT hardcoded)
-----------------------------------------------------
ARM_DIRS = sorted(models/E10-arm*/). Each has adapter_model.safetensors +
run_manifest.json. manifest['arm'] and manifest['noise_seed'] label the run and
map to a canonical arm-run id:
    A_clean        -> "A"
    B_struct s1/2/3 -> "B_s1"/"B_s2"/"B_s3"
    C_random s1/2/3 -> "C_s1"/"C_s2"/"C_s3"
    D_corrected    -> "D"

OUTPUT
------
  * results/experiments_e10/<pattern>/<query_id>.md   (judge READ root; pattern
    subdir = e10_A, e10_B_s1, ... e10_D). This is the JUDGE_RESULTS_BASE for the
    namespaced judge; treated READ-ONLY by the judge.
  * results/e10/<arm_run>/eval_manifest.json — full provenance for audit.
  * results/e10/<arm_run>/objective_eval.json — judge-free anti-Goodhart number.

82de3e92 QUARANTINE
-------------------
The quarantined query (id prefix 82de3e92) must NEVER reach GPT-5.2. We do not
write its .md into the per-arm judge dir at all, so the judge work-list never
sees it. The objective slice also drops it for consistency; the adjusted n is
disclosed in objective_eval.json.

IDEMPOTENT / RESUME
-------------------
Skip a (arm, query) if its .md already exists. Self-guards: assert every
adapter_dir/adapter_model.safetensors exists; assert split content_hash matches
the prereg before any GPU work.

USAGE
-----
    python scripts/run_e10_eval.py --self-test          # CPU-only wiring check, no GPU
    python scripts/run_e10_eval.py --dry-run            # list arms/queries, no GPU, no write
    python scripts/run_e10_eval.py                       # full GPU generation (all 8 arms)
    python scripts/run_e10_eval.py --arms A,B_s1         # subset of arm-runs
    python scripts/run_e10_eval.py --objective-only      # recompute objective from existing .md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# Memory-friendly env BEFORE torch import (mirrors train_p12_rl_v2.py)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Pinned prereg constants ──────────────────────────────────────────────────
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
GEN_SEED = 20260623
MAX_NEW_TOKENS = 512
SPLIT_PATH = _REPO_ROOT / "data" / "e10_split.json"
QUERIES_PATH = _REPO_ROOT / "data" / "eval_queries_v2.json"
SPLIT_CONTENT_HASH = "db4ae2affea3ea6f0b84113059f013ec090e5c3b6cd4fb56b5f4e11cc5586a04"
QUARANTINE_PREFIX = "82de3e92"

ARM_GLOB = "E10-arm*"
ARM_ROOT = _REPO_ROOT / "models"

# Report READ root the namespaced judge consumes (via JUDGE_RESULTS_BASE).
JUDGE_RESULTS_BASE = _REPO_ROOT / "results" / "experiments_e10"
# Per-arm provenance + objective (NOT read by the judge).
E10_RESULTS_ROOT = _REPO_ROOT / "results" / "e10"

# Protected corpus dirs run_e10_eval must NEVER write into.
PROTECTED = [
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis" / "canonical_numbers.json",
]


# ── Arm-run id mapping ───────────────────────────────────────────────────────
def arm_run_id(arm: str, noise_seed: int) -> str:
    """Map (manifest arm, noise_seed) -> canonical arm-run id.

    A_clean -> "A"; B_struct s{n} -> "B_s{n}"; C_random s{n} -> "C_s{n}";
    D_corrected -> "D". Unknown arms fall back to a sanitised label so a future
    arm is still handled rather than silently dropped.
    """
    if arm == "A_clean":
        return "A"
    if arm == "D_corrected":
        return "D"
    if arm == "B_struct":
        return f"B_s{int(noise_seed)}"
    if arm == "C_random":
        return f"C_s{int(noise_seed)}"
    # Generic fallback (keeps a novel arm visible instead of crashing).
    return f"{arm.replace('_', '')}_s{int(noise_seed)}"


def pattern_for(arm_run: str) -> str:
    """Judge pattern subdir name for an arm-run id."""
    return f"e10_{arm_run}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT)
        ).decode().strip()
    except Exception:
        return "unknown"


def assert_not_protected(path: Path) -> None:
    rp = path.resolve()
    for prot in PROTECTED:
        pr = prot.resolve()
        if rp == pr or pr in rp.parents:
            raise SystemExit(f"REFUSING: {path} is inside protected corpus path {prot}.")


# ── Discovery + guards (CPU only) ────────────────────────────────────────────
def discover_arms() -> list[dict]:
    """Return a sorted list of arm descriptors discovered on disk.

    Each: {arm_run, arm, noise_seed, num_generations, adapter_dir, base_model}.
    """
    arms = []
    for d in sorted(ARM_ROOT.glob(ARM_GLOB)):
        if not d.is_dir():
            continue
        man_path = d / "run_manifest.json"
        adapter = d / "adapter_model.safetensors"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        arm = man.get("arm", "")
        noise_seed = man.get("noise_seed", 1)
        arms.append({
            "arm_run": arm_run_id(arm, noise_seed),
            "arm": arm,
            "noise_seed": noise_seed,
            "num_generations": man.get("num_generations"),
            "adapter_dir": d,
            "adapter_safetensors": adapter,
            "base_model": man.get("base_model", DEFAULT_BASE),
            "manifest_hash": man.get("manifest_hash"),
        })
    arms.sort(key=lambda a: a["arm_run"])
    return arms


def assert_adapters_present(arms: list[dict]) -> None:
    missing = [str(a["adapter_safetensors"]) for a in arms
               if not a["adapter_safetensors"].exists()]
    if missing:
        raise SystemExit("Missing adapter_model.safetensors:\n  " + "\n  ".join(missing))


def load_split_and_assert() -> dict:
    split = json.loads(SPLIT_PATH.read_text())
    got = split.get("content_hash")
    if got != SPLIT_CONTENT_HASH:
        raise SystemExit(
            f"split content_hash mismatch: expected {SPLIT_CONTENT_HASH}, got {got}. "
            "Refusing to generate against a drifted split."
        )
    return split


def load_queries() -> dict:
    data = json.loads(QUERIES_PATH.read_text())
    return {q["id"]: q for q in data["queries"]}


def eval_query_ids(split: dict, queries: dict) -> list[str]:
    """Sorted held-out eval ids that exist in the query manifest (ALL 38,
    quarantine INCLUDED — generation of it is harmless; it is dropped only from
    the judge work-list)."""
    ids = [qid for qid in split["eval_ids"] if qid in queries]
    return sorted(ids)


def is_quarantined(qid: str) -> bool:
    return qid.startswith(QUARANTINE_PREFIX)


# ── Objective (judge-free, CPU) ──────────────────────────────────────────────
def compute_objective(arm_run: str, queries: dict, eval_ids: list[str]) -> dict:
    """Restrict the answer-checkable slice to the eval ids (minus quarantine),
    read the generated .md, and score with the prereg objective endpoint."""
    from deep_research.training import e10_objective_endpoint as obj

    gold = obj.load_answer_checkable(QUERIES_PATH)
    judged_ids = [q for q in eval_ids if not is_quarantined(q)]
    gold_slice = {qid: g for qid, g in gold.items() if qid in judged_ids}

    arm_report_dir = JUDGE_RESULTS_BASE / pattern_for(arm_run)
    reports_by_qid: dict[str, str] = {}
    for qid in gold_slice:
        md = arm_report_dir / f"{qid}.md"
        if md.exists():
            reports_by_qid[qid] = md.read_text()

    res = obj.evaluate_objective(reports_by_qid, gold_slice)
    return {
        "arm_run": arm_run,
        "mean_score": (res.mean_score if res.mean_score == res.mean_score else None),
        "n_queries": res.n_queries,
        "n_answer_checkable_eval": len(gold_slice),
        "quarantine_excluded": True,
        "token_overlap_threshold": res.token_overlap_threshold,
        "per_query": res.per_query,
        "per_query_source": res.per_query_source,
    }


def write_objective(arm_run: str, queries: dict, eval_ids: list[str]) -> Path:
    out = E10_RESULTS_ROOT / arm_run / "objective_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    block = compute_objective(arm_run, queries, eval_ids)
    out.write_text(json.dumps(block, indent=2, sort_keys=True))
    return out


# ── Generation (GPU) ─────────────────────────────────────────────────────────
def generate_for_arm(arm: dict, split: dict, queries: dict, eval_ids: list[str],
                     resume: bool = True) -> dict:
    """Load base+adapter for one arm, greedily generate held-out reports."""
    import torch  # local import: keep --self-test/--dry-run CPU-only
    import random as _random
    import numpy as _np
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed,
    )
    from peft import PeftModel
    import scripts.train_p12_rl_v2 as p12  # EXACT prompt + scaffold reuse

    arm_run = arm["arm_run"]
    adapter_dir = arm["adapter_dir"]
    pattern = pattern_for(arm_run)
    report_dir = JUDGE_RESULTS_BASE / pattern
    assert_not_protected(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build the per-query prompt via the EXACT training dataset builder.
    eval_queries = [queries[qid] for qid in eval_ids]
    ds = p12.build_dataset(eval_queries)  # rows: {"prompt", "query_id"}
    prompt_by_qid = {row["query_id"]: row["prompt"] for row in ds}

    # Determinism: seed everything before the loop, sorted query order.
    _random.seed(GEN_SEED)
    _np.random.seed(GEN_SEED)
    torch.manual_seed(GEN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GEN_SEED)
    set_seed(GEN_SEED)

    tokenizer = AutoTokenizer.from_pretrained(arm["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[{arm_run}] loading base {arm['base_model']} (4-bit NF4) + adapter {adapter_dir.name}",
          flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        arm["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Single-adapter load of the trainable POLICY LoRA saved at the arm-dir top
    # level. The judge/ subdir (frozen DR-Judge) is intentionally NOT loaded.
    model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="policy")
    model.eval()

    generated = []
    skipped = 0
    for qid in eval_ids:  # sorted
        if is_quarantined(qid):
            # prereg E10: the quarantined query (82de3e92) must NEVER reach GPT-5.2.
            # report_dir IS the judge-read root, so do not stage its .md here at all.
            continue
        md_path = report_dir / f"{qid}.md"
        if resume and md_path.exists():
            text = md_path.read_text()
            generated.append((qid, len(text.split()), sha256_text(text)))
            skipped += 1
            continue

        prompt = prompt_by_qid[qid]
        msgs = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True,
                           max_length=4096).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        report = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        md_path.write_text(report)
        generated.append((qid, len(report.split()), sha256_text(report)))
        print(f"[{arm_run}] {qid} words={len(report.split())}", flush=True)

    # Free VRAM before the next arm.
    del model, base
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Provenance manifest.
    man = {
        "arm_run": arm_run,
        "arm": arm["arm"],
        "noise_seed": arm["noise_seed"],
        "num_generations": arm["num_generations"],
        "adapter_dir": str(adapter_dir.relative_to(_REPO_ROOT)),
        "adapter_sha256": sha256_file(arm["adapter_safetensors"]),
        "base_model": arm["base_model"],
        "gen_params": {
            "do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS,
            "quantization": "4bit-nf4-bf16-double",
        },
        "gen_seed": GEN_SEED,
        "split_content_hash": SPLIT_CONTENT_HASH,
        "n_eval_generated": len(generated),
        "n_resumed_skip": skipped,
        "queries": [{"query_id": q, "n_words": w, "report_sha256": s}
                    for (q, w, s) in generated],
        "quarantine_id_generated_not_judged": [q for q in eval_ids if is_quarantined(q)],
        "bit_identity_claim": False,
        "bit_identity_note": (
            "Greedy + 4bit is near-deterministic per machine but NOT bit-identical "
            "across machines/GPUs; prereg reproducibility unit is cross-seed variance."
        ),
        "git_sha": git_sha(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = E10_RESULTS_ROOT / arm_run / "eval_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True))
    return man


# ── Self-test (CPU only, no GPU, no write to corpus) ─────────────────────────
def self_test() -> int:
    print("[self-test] E10 eval-gen wiring (CPU only, no GPU, no paid API)")

    arms = discover_arms()
    assert len(arms) >= 1, "no E10 arm dirs discovered"
    runs = [a["arm_run"] for a in arms]
    print(f"[self-test] discovered {len(arms)} arms: {runs}")

    # Expected 8-arm topology when all present.
    expect = {"A", "B_s1", "B_s2", "B_s3", "C_s1", "C_s2", "C_s3", "D"}
    if set(runs) == expect:
        print("[self-test] arm-run topology == expected 8-arm set")
    else:
        print(f"[self-test] NOTE: arm-run set {set(runs)} != canonical {expect} "
              "(ok for a subset run)")

    # arm_run_id mapping is correct.
    assert arm_run_id("A_clean", 1) == "A"
    assert arm_run_id("D_corrected", 1) == "D"
    assert arm_run_id("B_struct", 2) == "B_s2"
    assert arm_run_id("C_random", 3) == "C_s3"
    assert pattern_for("B_s1") == "e10_B_s1"
    print("[self-test] arm_run_id + pattern_for mapping OK")

    assert_adapters_present(arms)
    print("[self-test] all adapter_model.safetensors present")

    split = load_split_and_assert()
    print(f"[self-test] split content_hash matches prereg ({SPLIT_CONTENT_HASH[:12]}...)")

    queries = load_queries()
    eval_ids = eval_query_ids(split, queries)
    judged = [q for q in eval_ids if not is_quarantined(q)]
    quarantined = [q for q in eval_ids if is_quarantined(q)]
    print(f"[self-test] eval_ids in manifest: {len(eval_ids)} "
          f"(judged={len(judged)}, quarantined-excluded={len(quarantined)})")
    assert quarantined, "expected the 82de3e92 quarantined id in the eval split"
    assert len(judged) == len(eval_ids) - len(quarantined)

    # Prompt builder reuse (import scaffold WITHOUT importing torch-heavy paths).
    import importlib
    p12 = importlib.import_module("scripts.train_p12_rl_v2")
    ds = p12.build_dataset([queries[eval_ids[0]]])
    prompt = ds[0]["prompt"]
    assert prompt.startswith("Research query: "), "prompt must use the exact training template"
    assert "200-400 words" in prompt and "[1], [2]" in prompt
    print("[self-test] training prompt template reused exactly (build_dataset)")

    # Objective endpoint is importable and self-consistent on the eval slice.
    from deep_research.training import e10_objective_endpoint as obj
    gold = obj.load_answer_checkable(QUERIES_PATH)
    gold_eval = {qid: g for qid, g in gold.items()
                 if qid in judged}
    print(f"[self-test] answer-checkable held-out slice (minus quarantine): "
          f"n={len(gold_eval)}")
    if gold_eval:
        qid0 = sorted(gold_eval)[0]
        perfect = " ".join(gold_eval[qid0].key_facts)
        res = obj.evaluate_objective({qid0: perfect}, gold_eval)
        assert res.n_queries == 1 and res.mean_score >= 0.99
        print("[self-test] objective endpoint deterministic + in-range on eval slice")

    # Corpus-safety guard fires on protected paths.
    try:
        assert_not_protected(_REPO_ROOT / "results" / "experiments" / "e10_A")
        raise AssertionError("guard FAILED to refuse protected path")
    except SystemExit:
        pass
    # And allows the new dir.
    assert_not_protected(JUDGE_RESULTS_BASE / "e10_A")
    print("[self-test] corpus-safety guard refuses protected, allows results/experiments_e10")

    print("[self-test] PASS — generation wiring is sound (no GPU touched).")
    return 0


# ── Dry run (CPU only) ───────────────────────────────────────────────────────
def dry_run(selected: set[str] | None) -> int:
    arms = discover_arms()
    split = load_split_and_assert()
    queries = load_queries()
    eval_ids = eval_query_ids(split, queries)
    judged = [q for q in eval_ids if not is_quarantined(q)]
    print("E10 eval generation — DRY RUN (no GPU, no API, nothing written)")
    print(f"  arms discovered: {len(arms)}")
    print(f"  eval queries total: {len(eval_ids)}  judged: {len(judged)}  "
          f"quarantined(generated, NOT judged): {len(eval_ids) - len(judged)}")
    print(f"  report READ root (judge JUDGE_RESULTS_BASE): {JUDGE_RESULTS_BASE}")
    print(f"  per-arm provenance/objective root: {E10_RESULTS_ROOT}")
    total = 0
    for a in arms:
        if selected and a["arm_run"] not in selected:
            continue
        pat = pattern_for(a["arm_run"])
        report_dir = JUDGE_RESULTS_BASE / pat
        done = len(list(report_dir.glob("*.md"))) if report_dir.exists() else 0
        pending = len(eval_ids) - done
        total += max(0, pending)
        print(f"    {a['arm_run']:>6}  pattern={pat:<10} adapter={a['adapter_dir'].name} "
              f"done={done} pending={pending}")
    print(f"  total reports to generate: {total}")
    print(f"  judged-after = {len(judged)} per arm-run "
          "(quarantine .md is NOT staged into the judge dir)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only wiring check; no GPU, no API, no write.")
    ap.add_argument("--dry-run", action="store_true",
                    help="list arms/queries; no GPU, no API, no write.")
    ap.add_argument("--arms", type=str, default="",
                    help="comma-separated arm-run ids (e.g. A,B_s1). Default: all discovered.")
    ap.add_argument("--no-resume", action="store_true",
                    help="regenerate even if a .md already exists.")
    ap.add_argument("--objective-only", action="store_true",
                    help="recompute objective_eval.json from existing .md; no GPU.")
    args = ap.parse_args()

    selected = ({s.strip() for s in args.arms.split(",") if s.strip()}
                if args.arms else None)

    if args.self_test:
        return self_test()
    if args.dry_run:
        return dry_run(selected)

    arms = discover_arms()
    if selected:
        arms = [a for a in arms if a["arm_run"] in selected]
        if not arms:
            raise SystemExit(f"no arms matched --arms {sorted(selected)}")
    assert_adapters_present(arms)
    split = load_split_and_assert()
    queries = load_queries()
    eval_ids = eval_query_ids(split, queries)

    if args.objective_only:
        for a in arms:
            out = write_objective(a["arm_run"], queries, eval_ids)
            print(f"[{a['arm_run']}] objective -> {out}")
        return 0

    for a in arms:
        generate_for_arm(a, split, queries, eval_ids, resume=not args.no_resume)
        out = write_objective(a["arm_run"], queries, eval_ids)
        print(f"[{a['arm_run']}] objective -> {out}")
    print("Generation complete. NEXT: run the GPT-5.2 namespaced judge, then build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
