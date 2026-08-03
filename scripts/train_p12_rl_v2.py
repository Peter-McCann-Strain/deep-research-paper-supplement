#!/usr/bin/env python3
"""E10 P12 — GRPO RL training, 16 GB-friendly variant (v2).

Memory strategy (vs v1, which OOMs):
  * **One** Qwen2.5-7B-Instruct base in 4-bit NF4 (~7.7 GB).
  * **Two LoRA adapters** loaded on the same base:
      - `default`  = trainable policy LoRA (small, r=4 on q,v)
      - `judge`    = DR-Judge LoRA (frozen, from models/DR-Judge-7B-LoRA/)
    PEFT handles co-existence; we swap with `set_adapter("default" | "judge")`.
    Cost: ~7.7 GB (base) + ~50 MB (policy LoRA) + ~250 MB (judge LoRA).
  * `beta=0.0` to skip the reference model entirely
    (TRL 1.3.0: `grpo_trainer.py:651-653`). With PEFT and beta!=0, TRL
    *would* clone the LoRA adapter (cheap), but `beta=0` is simpler for
    the smoke test.
  * Aggressive: `max_completion_length=512`, `num_generations=2`,
    `per_device_train_batch_size=1`, `gradient_accumulation_steps=2`,
    LoRA r=4 / target=q_proj,v_proj only, gradient checkpointing.

Why not subprocess-isolated reward (the original plan)?
  Tried: subprocess can't get VRAM because the policy reserves ~15 GB
  even when only ~8 GB is allocated (PyTorch caching). `empty_cache()`
  can't reliably free the rest while the policy is mid-step. Multi-adapter
  reuses the same base weights, so total resident memory is dominated by
  ONE base — best case for 16 GB.

Usage:
    python scripts/train_p12_rl_v2.py --quick       # 10-step smoke
    python scripts/train_p12_rl_v2.py --max-steps N # full run
    python scripts/train_p12_rl_v2.py --quick --output-dir models/P12-RL-LoRA-v3 --num-generations 4
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# Memory-friendly env BEFORE torch import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DR_JUDGE_ADAPTER = REPO_ROOT / "models/DR-Judge-7B-LoRA"
P12_OUT_DIR = REPO_ROOT / "models/P12-RL-LoRA-v2"
QUERIES_PATH = REPO_ROOT / "data/eval_queries_v2.json"
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_ADAPTER_NAME = "judge"
POLICY_ADAPTER_NAME = "default"

JUDGE_SYSTEM = (
    "You are a strict research-report evaluator. Score the report from 0.0 to 1.0 on overall quality "
    "(coverage, factual grounding, citations, structure). Respond with ONLY a JSON object: "
    '{"score": <float in [0,1]>}. Do not include any other text.'
)


# ── Reward function (in-process, multi-adapter) ────────────────────────────
class MultiAdapterJudgeReward:
    """Reward callable that swaps the shared PEFT model to its `judge` adapter,
    runs scoring on each (prompt, completion) pair, then swaps back to
    `default` (the trainable policy adapter).

    TRL contract: `__call__(prompts, completions, completion_ids=None, **kw)`
    must return list[float] of length len(prompts). See
    `trl/trainer/grpo_trainer.py:_calculate_rewards` (~L1217) — synchronous
    callables are invoked exactly this way.
    """

    __name__ = "dr_judge_multi_adapter"

    def __init__(self, model_holder, tokenizer, max_new_tokens: int = 32,
                 max_input_chars: int = 6000, verbose: bool = True,
                 reward_log_path: Path | None = None,
                 raw_log_path: Path | None = None):
        # `model_holder` is a callable returning the live PEFT model — the
        # trainer wraps the model after `__init__`, so we resolve lazily.
        self.model_holder = model_holder
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_input_chars = max_input_chars
        self.verbose = verbose
        self.reward_log_path = reward_log_path
        self.raw_log_path = raw_log_path
        self._call_idx = 0
        if self.reward_log_path:
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.reward_log_path.exists():
                with self.reward_log_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "reward_call", "item_index", "reward",
                        "call_reward_mean", "call_reward_std", "call_zero_variance",
                        "completion_chars", "completion_words",
                    ])
        if self.raw_log_path:
            self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _judge_one(self, query_text: str, completion_text: str, model) -> tuple[float, str]:
        prompt = (
            f"Research query:\n{query_text}\n\n"
            f"Candidate report:\n{completion_text[:self.max_input_chars]}\n\n"
            'Return JSON only: {"score": <float in [0,1]>}.'
        )
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return parse_score(gen), gen

    def _write_logs(
        self,
        scores: list[float],
        raw_outputs: list[str],
        prompt_texts: list[str],
        completion_texts: list[str],
    ) -> None:
        if not scores:
            return
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        zero_variance = int(std == 0.0)
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.reward_log_path:
            with self.reward_log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i, (score, completion) in enumerate(zip(scores, completion_texts, strict=True)):
                    writer.writerow([
                        timestamp,
                        self._call_idx,
                        i,
                        f"{score:.6f}",
                        f"{mean:.6f}",
                        f"{std:.6f}",
                        zero_variance,
                        len(completion),
                        len(completion.split()),
                    ])

        if self.raw_log_path:
            with self.raw_log_path.open("a", encoding="utf-8") as f:
                for i, (score, raw, prompt, completion) in enumerate(
                    zip(scores, raw_outputs, prompt_texts, completion_texts, strict=True)
                ):
                    f.write(json.dumps({
                        "timestamp": timestamp,
                        "reward_call": self._call_idx,
                        "item_index": i,
                        "reward": score,
                        "raw_reward_model_output": raw,
                        "prompt_chars": len(prompt),
                        "completion_chars": len(completion),
                        "completion_words": len(completion.split()),
                    }, ensure_ascii=True) + "\n")

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):  # noqa: D401
        self._call_idx += 1
        n = len(prompts)
        # TRL passes prompts/completions as strings (when dataset has "prompt"
        # column with a string). Defensive flatten if list-of-messages slips in.
        prompt_texts: list[str] = []
        completion_texts: list[str] = []
        for p, c in zip(prompts, completions, strict=True):
            prompt_texts.append(
                p if isinstance(p, str)
                else " ".join(m.get("content", "") for m in p if isinstance(m, dict))
            )
            completion_texts.append(
                c if isinstance(c, str)
                else " ".join(m.get("content", "") for m in c if isinstance(m, dict))
            )

        model = self.model_holder()
        # Save current state so we can restore exactly. `active_adapters` may
        # be either a property (peft >=0.10) returning a list, OR a method on
        # older versions. Be defensive.
        attr = getattr(model, "active_adapters", None)
        if callable(attr):
            try:
                prior_active = list(attr())
            except TypeError:
                prior_active = list(attr) if attr else [POLICY_ADAPTER_NAME]
        elif isinstance(attr, (list, tuple)):
            prior_active = list(attr)
        elif isinstance(attr, str):
            prior_active = [attr]
        else:
            prior_active = [POLICY_ADAPTER_NAME]
        prior_training = model.training

        scores: list[float] = []
        raw_outputs: list[str] = []
        t0 = time.time()
        try:
            model.set_adapter(JUDGE_ADAPTER_NAME)
            model.eval()
            with torch.inference_mode():
                for q, c in zip(prompt_texts, completion_texts, strict=True):
                    try:
                        s, raw = self._judge_one(q, c, model)
                    except Exception as e:
                        print(f"[reward] _judge_one failed: {type(e).__name__}: {e}", flush=True)
                        s = 0.0
                        raw = f"_judge_one_failed: {type(e).__name__}: {e}"
                    scores.append(s)
                    raw_outputs.append(raw)
        finally:
            # Restore training state and adapter
            try:
                if prior_active:
                    model.set_adapter(prior_active if len(prior_active) > 1 else prior_active[0])
                else:
                    model.set_adapter(POLICY_ADAPTER_NAME)
            except Exception as e:
                print(f"[reward] WARNING: could not restore adapter: {e}", flush=True)
            if prior_training:
                model.train()

        elapsed = time.time() - t0
        self._write_logs(scores, raw_outputs, prompt_texts, completion_texts)
        if self.verbose:
            mean = sum(scores) / len(scores) if scores else 0.0
            print(
                f"[reward] call #{self._call_idx} n={n} elapsed={elapsed:.1f}s "
                f"mean={mean:.3f} min={min(scores):.3f} max={max(scores):.3f}",
                flush=True,
            )
        return scores


def parse_score(raw: str) -> float:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return float(max(0.0, min(1.0, obj.get("score", 0.0))))
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return float(max(0.0, min(1.0, obj.get("score", 0.0))))
        except Exception:
            pass
    m = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
    if m:
        try:
            return float(max(0.0, min(1.0, float(m.group(1)))))
        except Exception:
            pass
    return 0.0


# ── Dataset ────────────────────────────────────────────────────────────────
def stratified_subset(queries: list[dict], n: int = 6) -> list[dict]:
    by_diff: dict[str, list[dict]] = {}
    for q in queries:
        by_diff.setdefault(q.get("difficulty", "moderate"), []).append(q)
    buckets = sorted(by_diff)
    per = max(1, n // len(buckets))
    out: list[dict] = []
    for b in buckets:
        out.extend(by_diff[b][:per])
    return out[:n]


def build_dataset(queries: list[dict]) -> Dataset:
    rows = []
    for q in queries:
        prompt = (
            f"Research query: {q['query']}\n\n"
            "Write a concise research report (target 200-400 words) with a title, "
            "2-4 short sections, and inline numbered citations like [1], [2]."
        )
        rows.append({"prompt": prompt, "query_id": q["id"]})
    return Dataset.from_list(rows)


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--quick", action="store_true",
                    help="10-step smoke: 6 stratified queries, num_generations=2")
    ap.add_argument("--output-dir", type=Path, default=P12_OUT_DIR,
                    help="Adapter output directory. For v3 retry use models/P12-RL-LoRA-v3.")
    ap.add_argument("--reward-log", type=Path, default=None,
                    help="CSV reward diagnostics path. Defaults to <output-dir>/reward_trace.csv.")
    ap.add_argument("--raw-reward-log", type=Path, default=None,
                    help="JSONL raw reward-model output path. Defaults to <output-dir>/raw_reward_outputs.jsonl.")
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty. 0.0 skips reference model entirely.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    reward_log = args.reward_log or (out_dir / "reward_trace.csv")
    raw_reward_log = args.raw_reward_log or (out_dir / "raw_reward_outputs.jsonl")

    # Seed
    import random as _random
    import numpy as _np
    from transformers import set_seed as _hf_set_seed
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _hf_set_seed(args.seed)

    if not (DR_JUDGE_ADAPTER / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"DR-Judge adapter missing at {DR_JUDGE_ADAPTER}. "
            "Run finetune_dr_judge.py first."
        )

    print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    print(f"Output adapter: {out_dir}")
    print(f"Reward diagnostics: {reward_log}")
    print(f"Raw reward outputs: {raw_reward_log}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  free at start: {free_gb:.2f} GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model + policy adapter ───────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base: {args.base_model} (4-bit NF4)")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base = prepare_model_for_kbit_training(base)
    print(f"  base load: {time.time()-t0:.1f}s")

    # Add the trainable policy adapter as `default`
    policy_lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, policy_lora, adapter_name=POLICY_ADAPTER_NAME)
    print(f"After adding policy adapter (`{POLICY_ADAPTER_NAME}`):")
    model.print_trainable_parameters()

    # Add the DR-Judge adapter as a second, frozen adapter named `judge`
    print(f"Loading judge adapter from {DR_JUDGE_ADAPTER} as `{JUDGE_ADAPTER_NAME}`")
    model.load_adapter(
        str(DR_JUDGE_ADAPTER),
        adapter_name=JUDGE_ADAPTER_NAME,
        is_trainable=False,
    )
    # Ensure judge LoRA params don't get optimized
    for n, p in model.named_parameters():
        if f".{JUDGE_ADAPTER_NAME}." in n:
            p.requires_grad_(False)
    # Activate policy adapter for training
    model.set_adapter(POLICY_ADAPTER_NAME)
    _aa = getattr(model, "active_adapters", None)
    if callable(_aa):
        try:
            _aa_val = list(_aa())
        except TypeError:
            _aa_val = list(_aa) if _aa else None
    else:
        _aa_val = _aa
    print(f"Active adapter(s): {_aa_val}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  post-load alloc={alloc:.2f} GB reserved={reserved:.2f} GB")

    # ── Reward callable ────────────────────────────────────────────────────
    # Closure-based holder so we don't capture a stale reference if TRL
    # rewraps the model with accelerator.prepare_model.
    model_box = {"m": model}
    reward_fn = MultiAdapterJudgeReward(
        model_holder=lambda: model_box["m"],
        tokenizer=tokenizer,
        max_new_tokens=32,
        max_input_chars=4000,
        verbose=True,
        reward_log_path=reward_log,
        raw_log_path=raw_reward_log,
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    queries = json.loads(QUERIES_PATH.read_text())["queries"]
    if args.quick:
        sub = stratified_subset(queries, n=6)
        print(f"Quick mode: {len(sub)} stratified queries")
        for q in sub:
            print(f"  - [{q.get('difficulty')}] {q['id']}: {q['query'][:80]}")
        ds = build_dataset(sub)
    else:
        ds = build_dataset(queries)

    # ── GRPO config ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=1,
        save_strategy="no",
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        beta=args.beta,
        seed=args.seed,
        data_seed=args.seed,
        scale_rewards="group",
        temperature=0.9,
        top_p=0.95,
    )

    print("Starting GRPO training …")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    # Update the holder if accelerator wrapped/replaced the model
    model_box["m"] = trainer.model

    t_train = time.time()
    trainer.train()
    train_elapsed = time.time() - t_train

    # Save adapter
    trainer.save_model(str(out_dir))
    print(f"Saved adapter to {out_dir}")

    # Memory + timing summary
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print("\n=== SMOKE SUMMARY ===")
        print(f"  Train wall-clock: {train_elapsed:.1f}s for {args.max_steps} steps "
              f"({train_elapsed/max(1,args.max_steps):.1f}s/step)")
        print(f"  Peak GPU alloc:    {peak:.2f} GB")
        print(f"  Peak GPU reserved: {peak_reserved:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
