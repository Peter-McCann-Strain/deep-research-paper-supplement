#!/usr/bin/env python3
"""E7 step 2: QLoRA fine-tune Qwen2.5-7B-Instruct → DR-Judge-7B.

Trains on the SFT examples produced by prep_dr_judge_data.py. Uses 4-bit
NF4 quantization + LoRA rank-16 adapter. Designed to fit within the
RTX 5080's 16.6 GB VRAM budget.

Hyperparams (sized for 16 GB):
  - per_device_train_batch_size = 1
  - gradient_accumulation_steps = 8 (effective batch = 8)
  - learning_rate = 1e-4
  - warmup_ratio = 0.03
  - num_train_epochs = 2
  - max_seq_length = 8192 (truncate longer)
  - bf16 = True (Ada Lovelace supports)
  - LoRA: r=16, alpha=32, dropout=0.05, targets=q/k/v/o + gate/up/down

Outputs:
  models/DR-Judge-7B-LoRA/  (adapter weights)
  reports/phase12_drjudge/training_metrics.json
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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

DATA_DIR = Path("data/dr_judge_training")
OUT_DIR = Path("models/DR-Judge-7B-LoRA")
METRICS_DIR = Path("reports/phase12_drjudge")

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 200 steps + 50 train examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (also passed to SFTConfig)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load model + tokenizer with 4-bit quantization ───────────────────
    print(f"\nLoading {args.base_model} with 4-bit NF4 quantization …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ────────────────────────────────────────────────────────
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Run prep_dr_judge_data.py first: {train_path}")
    ds = load_dataset("json", data_files={"train": str(train_path), "val": str(val_path)})
    if args.quick:
        ds["train"] = ds["train"].select(range(50))
        ds["val"] = ds["val"].select(range(20))

    # ── Trainer ───────────────────────────────────────────────────────────
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs if not args.quick else 0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
        eval_strategy="no",  # disabled — TRL/HF eval after step 3000 has been deadlocking on adamw_torch + bnb 0.49 + torch 2.10
        save_strategy="steps",
        save_steps=1500,
        save_total_limit=3,
        dataloader_num_workers=0,  # avoids forked-worker deadlock
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",  # 8-bit paged optimiser hit a CUDA illegal-memory-access on bnb 0.49 + torch 2.10
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=1.0,
        weight_decay=0.01,
        max_steps=200 if args.quick else -1,
        seed=args.seed,
        data_seed=args.seed,
    )

    metrics_log: list[dict] = []

    class _MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logs = {k: v for k, v in logs.items() if isinstance(v, (int, float, str))}
                logs["step"] = state.global_step
                metrics_log.append(logs)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        callbacks=[_MetricsCallback()],
    )

    # ── Train (auto-resume from latest checkpoint if present) ────────────
    print("\nStarting training …")
    # Detect latest checkpoint in output_dir
    out_path = Path(args.output_dir)
    ckpts = sorted(out_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        last_ckpt = ckpts[-1]
        print(f"Resuming from {last_ckpt}")
        train_result = trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        train_result = trainer.train()
    print(f"\nTraining complete. metrics: {train_result.metrics}")

    # ── Save adapter + metrics ────────────────────────────────────────────
    print(f"Saving adapter to {OUT_DIR}")
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(OUT_DIR)
    (METRICS_DIR / "training_metrics.json").write_text(json.dumps({
        "final": train_result.metrics,
        "log": metrics_log,
        "config": vars(args),
    }, indent=2, default=str))
    print(f"Metrics: {METRICS_DIR / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
