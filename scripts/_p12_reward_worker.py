#!/usr/bin/env python3
"""DR-Judge reward worker for P12 GRPO training.

Loads DR-Judge-7B (Qwen2.5-7B-Instruct + LoRA, 4-bit), scores N completions,
writes results as JSON, then exits — freeing all VRAM back to the OS so the
policy can use the GPU for the next rollout/update step.

Input JSON (stdin or --in-file):
    {"items": [{"query": "...", "completion": "..."}, ...]}

Output JSON (--out-file):
    {"scores": [0.xx, 0.xx, ...]}  # one float per input item, in [0, 1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Memory hygiene before importing torch
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ADAPTER_DIR_DEFAULT = "models/DR-Judge-7B-LoRA"
BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_SYSTEM = (
    "You are a strict research-report evaluator. Score the report from 0.0 to 1.0 on overall quality "
    "(coverage, factual grounding, citations, structure). Respond with ONLY a JSON object: "
    '{"score": <float in [0,1]>}. Do not include any other text.'
)


def build_judge_prompt(query: str, completion: str, max_chars: int = 6000) -> str:
    return (
        f"Research query:\n{query}\n\n"
        f"Candidate report:\n{completion[:max_chars]}\n\n"
        'Return JSON only: {"score": <float in [0,1]>}.'
    )


def parse_score(raw: str) -> float:
    raw = raw.strip()
    # Strict JSON
    try:
        obj = json.loads(raw)
        return float(max(0.0, min(1.0, obj.get("score", 0.0))))
    except Exception:
        pass
    # First {...} block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return float(max(0.0, min(1.0, obj.get("score", 0.0))))
        except Exception:
            pass
    # Loose float fallback: look for "score": 0.xx
    m = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
    if m:
        try:
            return float(max(0.0, min(1.0, float(m.group(1)))))
        except Exception:
            pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--adapter-dir", default=ADAPTER_DIR_DEFAULT)
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-prompt-chars", type=int, default=6000)
    args = ap.parse_args()

    t_start = time.time()
    payload = json.loads(Path(args.in_file).read_text())
    items = payload.get("items", [])
    print(f"[reward_worker] {len(items)} items to score", flush=True)

    if not items:
        Path(args.out_file).write_text(json.dumps({"scores": []}))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[reward_worker] loading base + adapter ({args.adapter_dir})", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[reward_worker] post-load: alloc={used:.2f} GB reserved={reserved:.2f} GB", flush=True)

    scores: list[float] = []
    for i, item in enumerate(items):
        query = item.get("query", "")
        completion = item.get("completion", "")
        user_prompt = build_judge_prompt(query, completion, max_chars=args.max_prompt_chars)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(chat_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        s = parse_score(gen)
        scores.append(s)
        if (i + 1) % 4 == 0 or i == len(items) - 1:
            print(f"[reward_worker] {i+1}/{len(items)} score={s:.3f}", flush=True)

    Path(args.out_file).write_text(json.dumps({"scores": scores}))
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[reward_worker] peak alloc={peak:.2f} GB", flush=True)
    print(f"[reward_worker] done in {time.time()-t_start:.1f}s; mean={sum(scores)/len(scores):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
