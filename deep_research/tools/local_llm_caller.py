"""Local LLM inference via transformers with 4-bit quantization.

Provides the same interface as LLMCaller (complete, complete_json, complete_messages)
but runs a local model on GPU instead of calling Azure OpenAI.

Used by P9 (Qwen2.5-7B-Instruct baseline) and P10 (DeepResearcher-7b RL agent)
to enable fair comparison at the same model scale.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from deep_research.tools.cost_tracker import CostTracker

log = structlog.get_logger()

# Reduce CUDA memory fragmentation (suggested by OOM error messages)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Singleton model cache ────────────────────────────────────────────────────

# Cached entry: {"cache_key": (model_id, lora_adapter_path or ""),
#                "model_id": str, "lora_adapter_path": Optional[str],
#                "model": ..., "tokenizer": ...}
_loaded_model: Optional[dict] = None


def _make_cache_key(model_id: str, lora_adapter_path: Optional[str]) -> tuple:
    return (model_id, lora_adapter_path or "")


def _load_model(
    model_id: str,
    quantize_4bit: bool = True,
    lora_adapter_path: Optional[str] = None,
) -> dict:
    """Load a model with optional 4-bit quantization + optional LoRA adapter.

    Caches by (model_id, lora_adapter_path) so callers with different adapters
    do not collide. Loading a new combination evicts the previous one.
    """
    global _loaded_model

    cache_key = _make_cache_key(model_id, lora_adapter_path)

    if _loaded_model and _loaded_model.get("cache_key") == cache_key:
        log.info("local_model_cached", model_id=model_id,
                 lora_adapter_path=lora_adapter_path)
        return _loaded_model

    # Unload previous model if different
    if _loaded_model:
        log.info("unloading_previous_model",
                 old=_loaded_model.get("cache_key"))
        del _loaded_model["model"]
        del _loaded_model["tokenizer"]
        _loaded_model = None
        torch.cuda.empty_cache()

    log.info("loading_local_model", model_id=model_id, quantize_4bit=quantize_4bit,
             lora_adapter_path=lora_adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    # Apply LoRA adapter if supplied
    if lora_adapter_path:
        from peft import PeftModel
        log.info("loading_lora_adapter", lora_adapter_path=lora_adapter_path)
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        log.info("lora_adapter_loaded", lora_adapter_path=lora_adapter_path)

    log.info("model_loaded", model_id=model_id,
             params=f"{model.num_parameters() / 1e9:.1f}B",
             dtype=str(model.dtype),
             lora=bool(lora_adapter_path))

    _loaded_model = {
        "cache_key": cache_key,
        "model_id": model_id,
        "lora_adapter_path": lora_adapter_path,
        "model": model,
        "tokenizer": tokenizer,
    }
    return _loaded_model


def unload_model():
    """Explicitly unload the cached model to free VRAM."""
    global _loaded_model
    if _loaded_model:
        del _loaded_model["model"]
        del _loaded_model["tokenizer"]
        _loaded_model = None
        torch.cuda.empty_cache()
        log.info("model_unloaded")


# ── LocalLLMCaller ───────────────────────────────────────────────────────────


class LocalLLMCaller:
    """Local model caller with the same interface as LLMCaller.

    Methods:
        complete(prompt, model, system, temperature, max_tokens) -> str
        complete_json(prompt, model, system, temperature, max_tokens) -> Any
        complete_messages(messages, model, temperature, max_tokens) -> str
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        cost_tracker: Optional[CostTracker] = None,
        quantize_4bit: bool = True,
        lora_adapter_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.cost_tracker = cost_tracker or CostTracker()
        self.quantize_4bit = quantize_4bit
        self.lora_adapter_path = lora_adapter_path
        self._loaded = False

    def _ensure_loaded(self) -> dict:
        """Lazy-load model on first use."""
        if not self._loaded:
            _load_model(
                self.model_id,
                self.quantize_4bit,
                lora_adapter_path=self.lora_adapter_path,
            )
            self._loaded = True
        return _loaded_model

    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
        tools: Optional[List[Dict]] = None,
    ) -> tuple[str, int, int]:
        """Generate text from messages. Returns (text, input_tokens, output_tokens)."""
        loaded = self._ensure_loaded()
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]

        # Apply chat template (with optional tools for Qwen2.5 native tool format)
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        # Qwen3 family defaults to long <think> generations that blow the 16GB KV cache in the
        # extraction stage (OOM observed 2026-06-11). Disable thinking for Qwen3 models: it both
        # fits and makes them COMPARABLE to the no-thinking Qwen2.5 baseline (a cleaner vintage
        # control). Guarded to Qwen3 so other models' templates are unaffected.
        if "qwen3" in str(getattr(self, "model_id", "")).lower():
            template_kwargs["enable_thinking"] = False

        text = tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),  # avoid 0.0
                do_sample=temperature > 0.01,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_len:]
        output_len = len(new_tokens)
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return result, input_len, output_len

    async def complete(
        self,
        prompt: str,
        model: str = "",  # ignored — always uses self.model_id
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Standard text completion (same interface as LLMCaller)."""
        self.cost_tracker.check_budget()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens, temperature=temperature,
        )

        # Record with zero cost (local inference)
        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="local_complete",
        )

        log.debug("local_complete", model=self.model_id.split("/")[-1],
                   input_tokens=input_tokens, output_tokens=output_tokens)
        return result

    async def complete_json(
        self,
        prompt: str,
        model: str = "",
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Any:
        """Completion that returns parsed JSON (same interface as LLMCaller)."""
        self.cost_tracker.check_budget()

        sys_msg = (system + "\n" if system else "") + "Respond with valid JSON only."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens, temperature=temperature,
        )

        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="local_complete_json",
        )

        # Try to extract JSON from response
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", result, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            # Last resort: try to find any JSON object
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            log.warning("local_json_parse_failed", response=result[:200])
            return {}

    async def complete_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """Completion with full message list (same interface as LLMCaller)."""
        self.cost_tracker.check_budget()

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens, temperature=temperature,
            tools=tools,
        )

        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="local_complete_messages",
        )

        return result
