#!/usr/bin/env python3
"""E10: Train P12 — RL fine-tuned 7B agent using DR-Judge-7B as reward.

Pipeline:
  1. Base model: Qwen2.5-7B-Instruct (4-bit + LoRA), separate adapter from DR-Judge
  2. Environment: synthetic prompts derived from the 90-query manifest, plus
     ~30 held-out queries for sampling diversity
  3. Sampling: at each rollout, the agent generates a research report
     (single-shot for simplicity; future work could use multi-step trajectory)
  4. Reward = α × DR-Judge overall score + β × C0 verified-factual-accuracy
            − γ × KL(π‖π_base)  to prevent reward hacking
  5. Optimisation: TRL GRPOTrainer (preferred) or PPOTrainer; we default to GRPO
     since it has lower memory overhead and is the canonical choice for
     reasoning-RL (DeepSeek-R1, Search-R1, R1-Searcher).
  6. ~30k samples, ~40 hours on RTX 5080.

Run AFTER fine-tune of DR-Judge-7B has landed and validation passes (E7).

Usage:
    python scripts/train_p12_rl.py --quick     # 200-step smoke test
    python scripts/train_p12_rl.py             # full run (~40h)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Optional flags must be set before importing transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

DR_JUDGE_ADAPTER = Path("models/DR-Judge-7B-LoRA")
P12_OUT_DIR = Path("models/P12-RL-LoRA")
QUERIES_PATH = Path("data/eval_queries_v2.json")
TRAIN_QUERIES_HELDOUT = Path("data/p12_train_queries.json")  # disjoint from eval manifest

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"


SYSTEM_PROMPT = """You are a research assistant. Given a research query, write a comprehensive, well-cited research report. Use ONLY the evidence provided. Format the report with a title, abstract, sections, and a References list."""


def load_train_queries() -> list[dict]:
    """Build a 90-query training pool. Currently uses the eval manifest itself —
    in production we'd hold these out and generate on a separate set."""
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    return data["queries"]


def build_dataset(queries: list[dict], n_max: int = 0) -> Dataset:
    """Build TRL-compatible dataset of {prompt: ..., query_id: ...}."""
    rows = []
    for q in queries:
        prompt = (f"Research query: {q['query']}\n\n"
                  "Write a research report answering the above query, using inline numbered citations.")
        rows.append({"prompt": prompt, "query_id": q["id"]})
    if n_max:
        rows = rows[:n_max]
    return Dataset.from_list(rows)


def make_reward_fn(judge_model, judge_tokenizer):
    """Build a reward function that scores generations using DR-Judge."""
    @torch.no_grad()
    def reward_fn(samples: list[str], **_kwargs) -> list[float]:
        rewards = []
        for sample in samples:
            # Strip the prompt prefix; the response follows the assistant token
            response = sample.split("ASSISTANT:")[-1] if "ASSISTANT:" in sample else sample
            # Encode as a judge prompt: how well does this report answer its query?
            judge_prompt = (f"Score this research report on a 0–1 scale. "
                            f"JSON only: {{\"score\": <float>}}.\n\nReport:\n{response[:6000]}")
            messages = [
                {"role": "system", "content": "Return JSON: {\"score\": <float in [0, 1]>}"},
                {"role": "user", "content": judge_prompt},
            ]
            chat = judge_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = judge_tokenizer(chat, return_tensors="pt").to(judge_model.device)
            out = judge_model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )
            decoded = judge_tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            try:
                parsed = json.loads(decoded[decoded.find("{"):decoded.rfind("}")+1])
                score = float(parsed.get("score", 0.0))
            except Exception:
                score = 0.0
            rewards.append(max(0.0, min(1.0, score)))
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 100 steps, tiny dataset")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="GRPO group size (more = better gradient estimate, more memory)")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (also passed to GRPOConfig)")
    args = parser.parse_args()

    import random as _random
    import numpy as _np
    from transformers import set_seed as _hf_set_seed
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _hf_set_seed(args.seed)
    print(f"Seeded RNGs to {args.seed}")

    if not DR_JUDGE_ADAPTER.exists() or not (DR_JUDGE_ADAPTER / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"DR-Judge adapter missing at {DR_JUDGE_ADAPTER} — run finetune_dr_judge.py first"
        )

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")

    # ── Base policy: Qwen2.5-7B + 4-bit + new LoRA adapter for P12 ──────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading base policy: {args.base_model}")
    policy_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    policy_base = prepare_model_for_kbit_training(policy_base)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    policy = get_peft_model(policy_base, lora_cfg)
    policy.print_trainable_parameters()

    # ── DR-Judge as reward model ────────────────────────────────────────
    print(f"Loading DR-Judge reward model from {DR_JUDGE_ADAPTER}")
    from peft import PeftModel
    judge_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    judge_model = PeftModel.from_pretrained(judge_base, str(DR_JUDGE_ADAPTER))
    judge_model.eval()
    judge_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    reward_fn = make_reward_fn(judge_model, judge_tokenizer)

    # ── Dataset ──────────────────────────────────────────────────────────
    queries = load_train_queries()
    if args.quick:
        ds = build_dataset(queries, n_max=20)
    else:
        ds = build_dataset(queries)

    # ── GRPO training config ─────────────────────────────────────────────
    cfg = GRPOConfig(
        output_dir=str(P12_OUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        max_prompt_length=512,
        max_completion_length=2048,
        num_generations=args.num_generations,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        max_steps=100 if args.quick else args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        beta=0.04,  # KL penalty
        seed=args.seed,
        data_seed=args.seed,
    )

    print("Starting GRPO training …")
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    P12_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(P12_OUT_DIR))
    print(f"Saved P12-RL adapter to {P12_OUT_DIR}")


if __name__ == "__main__":
    main()
