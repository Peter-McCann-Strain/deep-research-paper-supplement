"""Azure OpenAI wrapper with PTU-optimized rate limiting, retry, cost tracking, and JSON mode.

Rate limiting strategy (Semaphore + Leaky Bucket):
- asyncio.Semaphore caps concurrent in-flight requests (prevents connection exhaustion)
- aiolimiter.AsyncLimiter provides leaky-bucket rate control matching PTU's algorithm
- PTU 429s are short-lived (100-2000ms) — retry-after-ms is honored directly
- Connection errors (BrokenPipe) get fast retry with short backoff
- SDK retries disabled (max_retries=0) so our centralised logic has full control
- Single shared httpx client with keepalive_expiry to prevent stale connections
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional

import httpx
import structlog
from aiolimiter import AsyncLimiter
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai import (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)

from deep_research.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    OPENAI_API_KEY,
    USE_AZURE_OPENAI,
    DEFAULT_MODEL,
    MODELS,
    RETRY,
    TIMEOUTS,
    POOL,
    RATE_LIMIT,
)
from deep_research.tools.cost_tracker import CostTracker

log = structlog.get_logger()

# ── Concurrency and rate defaults (from centralised config) ──────────────────
_MAX_CONCURRENT = RATE_LIMIT.max_concurrent
_RPM = RATE_LIMIT.rpm

# ── Retry defaults (from centralised config) ────────────────────────────────
_MAX_RETRIES = RETRY.max_retries
_PTU_BASE_DELAY = RETRY.base_delay
_CONN_BASE_DELAY = RETRY.conn_base_delay
_MAX_DELAY = RETRY.max_delay
_JITTER_MAX = RETRY.jitter_max

# Exception types that are safe to retry
_RETRYABLE = (
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    # Raw connection errors that may not be wrapped by the SDK
    ConnectionError,            # Covers BrokenPipeError, ConnectionResetError
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


# ── PTU-aware rate gate ──────────────────────────────────────────────────────

class _PTURateGate:
    """Dual-layer rate limiter matching Azure PTU's leaky-bucket algorithm.

    Layer 1: asyncio.Semaphore — hard concurrency cap (prevents connection exhaustion)
    Layer 2: aiolimiter.AsyncLimiter — leaky-bucket rate control (matches PTU server-side)

    Unlike AIMD, this uses fixed limits that don't cascade on transient 429s.
    PTU has known, fixed capacity — adaptive algorithms cause unnecessary oscillation.
    """

    def __init__(self, max_concurrent: int = _MAX_CONCURRENT, rpm: int = _RPM):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = AsyncLimiter(max_rate=rpm, time_period=60)
        self._max_concurrent = max_concurrent
        self._in_flight = 0
        self._total_success = 0
        self._total_rate_limit = 0
        self._total_conn_error = 0

    @property
    def current_limit(self) -> int:
        return self._max_concurrent

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self) -> None:
        """Acquire both semaphore slot and rate limiter token.

        Semaphore first so RPM tokens aren't wasted on queued requests.
        """
        await self._semaphore.acquire()
        await self._rate_limiter.acquire()
        self._in_flight += 1

    def release(self, success: bool = True, rate_limited: bool = False) -> None:
        """Release the semaphore slot and record stats."""
        self._in_flight -= 1
        self._semaphore.release()
        if success:
            self._total_success += 1
        elif rate_limited:
            self._total_rate_limit += 1
        else:
            self._total_conn_error += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "limit": self._max_concurrent,
            "in_flight": self._in_flight,
            "total_success": self._total_success,
            "total_rate_limit": self._total_rate_limit,
            "total_conn_error": self._total_conn_error,
        }


# Module-level singleton — created lazily on first use
_global_gate: Optional[_PTURateGate] = None


def _get_gate() -> _PTURateGate:
    """Get or create the global rate gate (singleton)."""
    global _global_gate
    if _global_gate is None:
        _global_gate = _PTURateGate()
    return _global_gate


def reset_limiter() -> None:
    """Reset the global rate gate (useful between test runs)."""
    global _global_gate
    _global_gate = None


# ── Retry helpers ─────────────────────────────────────────────────────────────

def _extract_retry_after(exc: BaseException) -> float:
    """Extract retry-after hint from a RateLimitError response.

    PTU returns retry-after-ms (milliseconds) — typically 100-2000ms.
    Standard deployments return retry-after (seconds).
    """
    if not isinstance(exc, RateLimitError):
        return 0.0

    response = getattr(exc, "response", None)
    if response is None:
        return 0.0

    headers = getattr(response, "headers", {})

    # PTU-specific: millisecond precision
    retry_ms = headers.get("retry-after-ms")
    if retry_ms:
        try:
            return int(retry_ms) / 1000.0
        except (ValueError, TypeError):
            pass

    # Standard: seconds
    retry_s = headers.get("retry-after")
    if retry_s:
        try:
            return float(retry_s)
        except (ValueError, TypeError):
            pass

    return 0.0


def _backoff_for_rate_limit(attempt: int, retry_after: float = 0.0) -> float:
    """Exponential backoff for 429 errors, always honouring retry-after-ms.

    The PTU tells us exactly when capacity will be free via retry-after-ms.
    We always trust that header and add exponential backoff on top so that
    concurrent retries from parallel sub-agents spread out naturally.
    """
    # Exponential backoff with jitter — capped at max_delay
    exp_delay = min(_MAX_DELAY, _PTU_BASE_DELAY * (2 ** min(attempt, 8)))
    jitter = random.uniform(0, _JITTER_MAX)
    backoff = exp_delay + jitter

    # Always honour retry-after if provided — use whichever is longer
    if retry_after > 0:
        return max(backoff, retry_after + random.uniform(0, 1.0))

    return backoff


def _backoff_for_connection(attempt: int) -> float:
    """Backoff for connection errors (BrokenPipe, timeout, etc).

    These are not rate limits — retry quickly.
    """
    exp_delay = min(_MAX_DELAY, _CONN_BASE_DELAY * (2 ** min(attempt, 4)))
    jitter = random.uniform(0, _JITTER_MAX)
    return exp_delay + jitter


# ── Shared client singleton ──────────────────────────────────────────────────

_shared_client: Optional[AsyncAzureOpenAI | AsyncOpenAI] = None


def _use_azure_client() -> bool:
    return USE_AZURE_OPENAI or bool(
        AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and not OPENAI_API_KEY
    )


def _get_client() -> AsyncAzureOpenAI | AsyncOpenAI:
    """Get or create the shared Azure OpenAI client (singleton).

    Sharing a single httpx client across all LLMCaller instances prevents
    connection pool fragmentation and reduces stale-connection BrokenPipe errors.
    """
    global _shared_client
    if _shared_client is None:
        timeout = httpx.Timeout(
            connect=TIMEOUTS.connect,
            read=TIMEOUTS.read,
            write=TIMEOUTS.write,
            pool=TIMEOUTS.pool,
        )
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=POOL.max_connections,
                max_keepalive_connections=POOL.max_keepalive_connections,
                keepalive_expiry=POOL.keepalive_expiry,
            ),
        )
        if _use_azure_client():
            _shared_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                max_retries=0,  # Disable SDK retries — we handle them centrally
                timeout=timeout,
                http_client=http_client,
            )
        else:
            _shared_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                max_retries=0,
                timeout=timeout,
                http_client=http_client,
            )
    return _shared_client


def _model_for_call(model: str) -> str:
    if _use_azure_client():
        spec = MODELS.get(model)
        return spec.deployment if spec else model
    return model


def _token_kwargs(model: str, max_tokens: int) -> dict:
    """Return the correct token-limit kwarg for the model."""
    spec = MODELS.get(model)
    if spec and spec.use_max_completion_tokens:
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


# ── LLMCaller ─────────────────────────────────────────────────────────────────

class LLMCaller:
    """Async Azure OpenAI caller with PTU-optimized rate limiting and cost tracking.

    All instances share:
    - A global rate gate (semaphore + leaky-bucket rate limiter)
    - A single shared httpx client (prevents connection pool fragmentation)

    Rate-limited requests are queued, not dropped.
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        self.client = _get_client()
        self.cost_tracker = cost_tracker or CostTracker()

    async def _call_with_rate_limit(self, coro_factory, call_type: str, model: str):
        """Execute an LLM call with rate limiting and retry.

        This is the single chokepoint for ALL LLM traffic. It:
        1. Acquires semaphore slot + rate limiter token
        2. Makes the API call
        3. On 429: waits (retry-after-ms + jitter), retries
        4. On connection error: short backoff, retries
        5. On success: returns result

        Uses try/finally to guarantee slot release — prevents semaphore leak
        on CancelledError or any unexpected BaseException.
        """
        gate = _get_gate()
        last_exc = None

        for attempt in range(_MAX_RETRIES):
            await gate.acquire()

            # Track release parameters; finally block uses these
            _release_success = True
            _release_rate_limited = False

            try:
                result = await coro_factory()
                # Success path — mark for release with success=True
                _release_success = True
                return result

            except RateLimitError as e:
                retry_after = _extract_retry_after(e)
                _release_success = False
                _release_rate_limited = True
                last_exc = e

                wait_time = _backoff_for_rate_limit(attempt, retry_after)
                log.warning(
                    "rate_limited",
                    model=model,
                    call_type=call_type,
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    retry_after_ms=f"{retry_after * 1000:.0f}",
                    wait_seconds=f"{wait_time:.1f}",
                    gate_in_flight=gate.in_flight,
                )
                # Sleep happens after finally releases the gate slot
                _sleep_time = wait_time

            except (APIConnectionError, ConnectionError,
                    httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                # Connection errors — retry quickly, don't penalize rate
                _release_success = False
                _release_rate_limited = False
                last_exc = e

                wait_time = _backoff_for_connection(attempt)
                log.warning(
                    "llm_connection_error",
                    model=model,
                    call_type=call_type,
                    error_type=type(e).__name__,
                    error=str(e)[:100],
                    attempt=attempt + 1,
                    wait_seconds=f"{wait_time:.1f}",
                )
                _sleep_time = wait_time

            except _RETRYABLE as e:
                # Other retryable errors (timeout, 5xx)
                _release_success = False
                _release_rate_limited = False
                last_exc = e

                wait_time = _backoff_for_connection(attempt)
                log.warning(
                    "llm_retryable_error",
                    model=model,
                    call_type=call_type,
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                    wait_seconds=f"{wait_time:.1f}",
                )
                _sleep_time = wait_time

            except Exception as e:
                # Non-retryable error — propagate immediately (finally releases gate)
                _release_success = True
                log.error(
                    "llm_non_retryable_error",
                    model=model,
                    call_type=call_type,
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )
                raise

            except BaseException:
                # CancelledError, KeyboardInterrupt, SystemExit, etc.
                # Mark as failure so stats reflect the cancellation.
                _release_success = False
                raise

            finally:
                gate.release(success=_release_success, rate_limited=_release_rate_limited)

            # Backoff sleep is OUTSIDE the try/finally so the gate slot is
            # already released while we wait — no slot held during sleep.
            await asyncio.sleep(_sleep_time)

        # All retries exhausted
        log.error(
            "llm_retries_exhausted",
            model=model,
            call_type=call_type,
            attempts=_MAX_RETRIES,
            gate_stats=gate.stats(),
        )
        raise last_exc  # type: ignore[misc]

    async def complete(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Standard text completion."""
        self.cost_tracker.check_budget()

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_model = _model_for_call(model)

        async def _do_call():
            return await self.client.chat.completions.create(
                model=call_model,
                messages=messages,
                temperature=temperature,
                **_token_kwargs(model, max_tokens),
            )

        resp = await self._call_with_rate_limit(_do_call, "complete", model)

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        if usage:
            self.cost_tracker.record(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                call_type="complete",
            )

        log.debug("llm_complete", model=model, tokens=usage.total_tokens if usage else 0)
        return content

    async def complete_json(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Any:
        """Completion that returns parsed JSON."""
        self.cost_tracker.check_budget()

        messages: List[Dict[str, str]] = []
        sys_msg = (system + "\n" if system else "") + "Respond with valid JSON only."
        messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})

        call_model = _model_for_call(model)

        async def _do_call():
            return await self.client.chat.completions.create(
                model=call_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **_token_kwargs(model, max_tokens),
            )

        resp = await self._call_with_rate_limit(_do_call, "complete_json", model)

        content = resp.choices[0].message.content or "{}"
        usage = resp.usage
        if usage:
            self.cost_tracker.record(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                call_type="complete_json",
            )

        log.debug("llm_json", model=model, tokens=usage.total_tokens if usage else 0)
        return json.loads(content)

    async def complete_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Completion with full message list (for multi-turn)."""
        self.cost_tracker.check_budget()

        call_model = _model_for_call(model)

        async def _do_call():
            return await self.client.chat.completions.create(
                model=call_model,
                messages=messages,
                temperature=temperature,
                **_token_kwargs(model, max_tokens),
            )

        resp = await self._call_with_rate_limit(_do_call, "complete_messages", model)

        content = resp.choices[0].message.content or ""
        usage = resp.usage
        if usage:
            self.cost_tracker.record(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                call_type="complete_messages",
            )

        return content
