"""DR-Judge-7B inference wrapper.

Loads the QLoRA-fine-tuned Qwen2.5-7B-Instruct adapter (from
`models/DR-Judge-7B-LoRA/`) and exposes the same `LLMCaller`-like
interface as `LocalLLMCaller` for consumption by judge runners.

Usage:
    from deep_research.tools.dr_judge_caller import DRJudgeCaller
    judge = DRJudgeCaller()
    raw = await judge.complete_json(prompt, system=..., temperature=0.1, max_tokens=4096)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import structlog

# Defer torch/transformers imports to instance load — keeps script-import cheap
log = structlog.get_logger()

ADAPTER_DIR_DEFAULT = Path("models/DR-Judge-7B-LoRA")
BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"


_loaded: dict = {}
_load_lock = threading.Lock()


def _load_dr_judge(adapter_dir: Path = ADAPTER_DIR_DEFAULT,
                   base_model: str = BASE_MODEL_DEFAULT) -> dict:
    """Load DR-Judge-7B with adapter merged on the fly."""
    global _loaded
    with _load_lock:
        if _loaded.get("ready"):
            return _loaded
        os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        log.info("dr_judge_loading", adapter=str(adapter_dir), base=base_model)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        _loaded.update({
            "ready": True,
            "model": model,
            "tokenizer": tokenizer,
            "base_model": base_model,
            "adapter_dir": str(adapter_dir),
        })
        log.info("dr_judge_loaded")
        return _loaded


class DRJudgeCaller:
    """Same shape as `LocalLLMCaller` / `LLMCaller` for judge runners."""

    def __init__(self,
                 adapter_dir: Path | str = ADAPTER_DIR_DEFAULT,
                 base_model: str = BASE_MODEL_DEFAULT):
        self.adapter_dir = Path(adapter_dir)
        self.base_model = base_model
        if not self.adapter_dir.exists():
            log.warning("dr_judge_adapter_missing", path=str(self.adapter_dir))

    def _ensure_loaded(self) -> dict:
        return _load_dr_judge(self.adapter_dir, self.base_model)

    async def complete(self, prompt: str, system: str = "",
                       temperature: float = 0.1,
                       max_tokens: int = 2048) -> str:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_complete, prompt, system, temperature, max_tokens
        )

    def _sync_complete(self, prompt: str, system: str,
                       temperature: float, max_tokens: int) -> str:
        loaded = self._ensure_loaded()
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        import torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        return response

    async def complete_json(self, prompt: str, system: str = "",
                            temperature: float = 0.1,
                            max_tokens: int = 2048,
                            **_kwargs) -> Any:
        raw = await self.complete(prompt, system=system, temperature=temperature,
                                   max_tokens=max_tokens)
        # Try strict parse; fall back to first {...} block
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        log.warning("dr_judge_json_parse_failed", raw=raw[:200])
        return {"_parse_failure": raw[:500]}


def smoke_test() -> None:
    """Quick smoke: load adapter + run one completion."""
    j = DRJudgeCaller()
    j._ensure_loaded()
    res = asyncio.run(j.complete("Say 'hello world' in JSON: {\"greeting\": \"...\"}",
                                  system="Return JSON only.",
                                  temperature=0.1, max_tokens=64))
    print(f"DR-Judge smoke output: {res!r}")


if __name__ == "__main__":
    smoke_test()
