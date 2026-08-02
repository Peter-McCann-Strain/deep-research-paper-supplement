"""Local LLM inference via llama.cpp (GGUF), GPU-accelerated for sm_120 (RTX 5080).

Provides the SAME async interface as LLMCaller / LocalLLMCaller
(``complete``, ``complete_json``, ``complete_messages``) but runs a GGUF-quantised
model through ``llama_cpp`` instead of transformers + bitsandbytes.

Why this exists
---------------
The transformers/bnb path (``local_llm_caller.LocalLLMCaller``) OOMs at the
weights-materialisation step for the 14B Qwen2.5 model even in 4-bit on the
single 16 GB RTX 5080 (documented in scripts/run_detector_panel.py and
scripts/build_e8_vintage.py). The llama.cpp Q4_K_M GGUF path full-offloads the
8.4 GB weights + a 4096-token KV cache comfortably (~10-11 GB peak), letting the
14B arm run on the frozen P9 scaffold without changing tools, prompts, or judge.

Determinism (June-2026 best practice)
-------------------------------------
``temperature=0`` alone is NOT strict-greedy on llama.cpp. Strict greedy here
sets ``temperature=0.0`` AND ``top_k=1`` AND ``top_p=1.0`` AND
``repeat_penalty=1.0`` AND a fixed ``seed=42``. CUDA kernels can still introduce
token-level drift across runs; pin the GGUF file + the llama-cpp-python 0.3.31
binary + the driver. (The transformers P9/P14 arms used ``do_sample`` with
temperature clamped to 0.01 — NOT identical greedy — so this GGUF arm's greedy
decode is a small, declared scaffold-frozen-ness imperfection, not hidden.)

Cost
----
Local inference is $0. CostTracker.record is called on the usage block's input /
output token counts with a model label NOT present in MODELS, so the tracker's
conservative fallback yields a negligible (effectively-zero) cost — byte-for-byte
the same accounting convention LocalLLMCaller uses for the 7B arms.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from deep_research.tools.cost_tracker import CostTracker

log = structlog.get_logger()

# ── CUDA library path for the sm_120 llama.cpp build ──────────────────────────
# The GPU-accelerated llama-cpp-python 0.3.31 binary in this venv links against
# the bundled CUDA toolkit under .cudatk/lib and the per-package nvidia/*/lib
# shared objects. We must ensure they are on LD_LIBRARY_PATH BEFORE the native
# extension is imported, otherwise the offload symbols fail to resolve and the
# model silently falls back to CPU (or fails to load). Setting it at import time
# is belt-and-braces; the runner also exports it in-process. Mirrors the manual
# `export LD_LIBRARY_PATH=...` instruction in the E8 build spec.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_cuda_ld_library_path() -> None:
    """Prepend the sm_120 CUDA lib dirs to LD_LIBRARY_PATH (idempotent)."""
    parts: List[str] = []
    cudatk = _REPO_ROOT / ".cudatk" / "lib"
    if cudatk.is_dir():
        parts.append(str(cudatk))
    nvidia_root = _REPO_ROOT / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        for lib in sorted(nvidia_root.glob("*/lib")):
            if lib.is_dir():
                parts.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = existing.split(":") if existing else []
    # Prepend only the parts not already present, preserving order.
    new_parts = [p for p in parts if p not in existing_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


_ensure_cuda_ld_library_path()

# Default GGUF path for the Qwen2.5-14B capacity/scale arm.
DEFAULT_GGUF_PATH = str(
    _REPO_ROOT / "models" / "gguf" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)

# Provenance label kept identical to the HF model id so the judged rows and
# canonical provenance read the same as the transformers arms would have.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Deterministic generation constants. Fixed seed + strict greedy.
SEED = 42
N_CTX = 4096
N_BATCH = 512


# ── Singleton model cache ─────────────────────────────────────────────────────
# Mirrors local_llm_caller._loaded_model: ONE GGUF model resident at a time on
# the 16 GB card. Loading a different GGUF evicts the previous one.
#
# Cached entry: {"cache_key": str (model_path), "model_path": str, "llama": Llama}
_loaded_model: Optional[dict] = None


def _load_model(model_path: str, n_ctx: int = N_CTX) -> dict:
    """Load (or return cached) Llama GGUF model with full GPU offload."""
    global _loaded_model

    cache_key = str(Path(model_path).resolve())

    if _loaded_model and _loaded_model.get("cache_key") == cache_key:
        log.info("llamacpp_model_cached", model_path=model_path)
        return _loaded_model

    if _loaded_model:
        log.info("llamacpp_unloading_previous", old=_loaded_model.get("cache_key"))
        del _loaded_model["llama"]
        _loaded_model = None

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"GGUF model not found at {model_path}. Download it first "
            f"(scripts/download_models.py or a manual GGUF fetch)."
        )

    # Import here so LD_LIBRARY_PATH is already set and modules that only want
    # the interface (e.g. tools/__init__) never pay the native import cost.
    from llama_cpp import Llama, llama_supports_gpu_offload

    if not llama_supports_gpu_offload():
        log.warning("llamacpp_no_gpu_offload",
                    note="llama_supports_gpu_offload() is False; this build will run on CPU")

    log.info("loading_llamacpp_model", model_path=model_path, n_ctx=n_ctx, seed=SEED)

    llama = Llama(
        model_path=model_path,
        n_gpu_layers=-1,      # full offload
        n_ctx=n_ctx,
        seed=SEED,
        n_batch=N_BATCH,
        logits_all=False,
        verbose=False,
    )

    log.info("llamacpp_model_loaded", model_path=model_path)

    _loaded_model = {
        "cache_key": cache_key,
        "model_path": model_path,
        "llama": llama,
    }
    return _loaded_model


def unload_model() -> None:
    """Explicitly unload the cached GGUF model to free VRAM."""
    global _loaded_model
    if _loaded_model:
        del _loaded_model["llama"]
        _loaded_model = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        log.info("llamacpp_model_unloaded")


# ── LlamaCppLLMCaller ──────────────────────────────────────────────────────────


class LlamaCppLLMCaller:
    """GGUF / llama.cpp caller with the same interface as LLMCaller / LocalLLMCaller.

    Methods:
        complete(prompt, model, system, temperature, max_tokens) -> str
        complete_json(prompt, model, system, temperature, max_tokens) -> Any
        complete_messages(messages, model, temperature, max_tokens) -> str

    The ``temperature`` arguments are accepted for interface parity but the
    backend ALWAYS decodes strict-greedy (temperature=0, top_k=1, top_p=1.0,
    repeat_penalty=1.0, fixed seed) for determinism. ``model`` is ignored — the
    model is fixed by ``model_path`` at construction, exactly like LocalLLMCaller
    ignores its ``model`` argument in favour of ``self.model_id``.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_GGUF_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        cost_tracker: Optional[CostTracker] = None,
        n_ctx: int = N_CTX,
    ):
        _ensure_cuda_ld_library_path()
        self.model_path = model_path
        self.model_id = model_id  # provenance label (e.g. "Qwen/Qwen2.5-14B-Instruct")
        self.cost_tracker = cost_tracker or CostTracker()
        self.n_ctx = n_ctx
        self._loaded = False

    def _ensure_loaded(self) -> dict:
        if not self._loaded:
            _load_model(self.model_path, n_ctx=self.n_ctx)
            self._loaded = True
        return _loaded_model

    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """Strict-greedy chat completion. Returns (text, input_tokens, output_tokens)."""
        loaded = self._ensure_loaded()
        llama = loaded["llama"]

        resp = llama.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            seed=SEED,
            repeat_penalty=1.0,
            max_tokens=max_new_tokens,
        )

        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return text, input_tokens, output_tokens

    async def complete(
        self,
        prompt: str,
        model: str = "",  # ignored — model fixed by model_path
        system: str = "",
        temperature: float = 0.3,  # accepted for parity; backend is strict-greedy
        max_tokens: int = 4096,
    ) -> str:
        """Standard text completion (same interface as LLMCaller)."""
        self.cost_tracker.check_budget()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens,
        )

        # Zero-cost local accounting: model label is not in MODELS, so the
        # CostTracker's conservative fallback yields a negligible cost — same
        # convention LocalLLMCaller uses for the 7B local arms.
        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="llamacpp_complete",
        )

        log.debug("llamacpp_complete", model=self.model_id.split("/")[-1],
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
        """Completion that returns parsed JSON (same interface as LLMCaller).

        JSON extraction fallback is copied VERBATIM from
        LocalLLMCaller.complete_json so the two local backends parse identically.
        """
        self.cost_tracker.check_budget()

        sys_msg = (system + "\n" if system else "") + "Respond with valid JSON only."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens,
        )

        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="llamacpp_complete_json",
        )

        # ── JSON extraction fallback (verbatim from LocalLLMCaller) ──────────
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
            log.warning("llamacpp_json_parse_failed", response=result[:200])
            return {}

    async def complete_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """Completion with full message list (same interface as LLMCaller).

        ``tools`` is accepted for interface parity. The frozen P9 scaffold does
        not pass native tools (it uses prompt-based extraction), so tools are not
        wired into the create_chat_completion call here.
        """
        self.cost_tracker.check_budget()

        result, input_tokens, output_tokens = self._generate(
            messages, max_new_tokens=max_tokens,
        )

        self.cost_tracker.record(
            model=self.model_id.split("/")[-1],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type="llamacpp_complete_messages",
        )

        return result
