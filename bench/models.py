"""The three systems under test, behind one uniform interface.

    LocalModel   - Gemma 4 E2B through LiteRT-LM (Metal GPU on Apple silicon)
    CloudModel   - Nemotron-3-Ultra through OpenRouter
    HybridModel  - Laya routes each prompt to one of the two above

All three expose `generate(prompt, max_tokens) -> GenResult` and record the
same hardware metrics so the comparison is apples-to-apples.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import litert_lm
from litert_lm.interfaces import CPU, GPU
from openai import OpenAI

from .quota import QuotaExhausted, RateLimiter
from .config import (
    CLOUD_API_KEY_ENV,
    CLOUD_BASE_URL,
    CLOUD_MAX_RETRIES,
    CLOUD_RETRY_BACKOFF_S,
    CLOUD_DIFFICULTY_THRESHOLD,
    CLOUD_MODEL_ID,
    CLOUD_SENSITIVE_THRESHOLD,
    CLOUD_TEMPERATURE,
    CLOUD_TOOLS_THRESHOLD,
    DISABLE_CLOUD_REASONING,
    LOCAL_MODEL_PATH,
)


@dataclass
class GenResult:
    text: str
    route: str
    ttft_ms: float
    total_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    decode_ms: float = 0.0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def output_tok_per_s(self) -> float:
        """End-to-end output rate (generation time only, excluding routing)."""
        if self.total_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_ms / 1000)

    @property
    def decode_tok_per_s(self) -> float:
        """Rate over the decode phase only (after the first token)."""
        if self.decode_ms <= 1.0:
            return 0.0
        return self.output_tokens / (self.decode_ms / 1000)


# ------------------------------------------------------------------- local
class LocalModel:
    name = "local"

    def __init__(self, model_path: str = LOCAL_MODEL_PATH, backend: str = "gpu") -> None:
        self.model_path = os.path.expanduser(model_path)
        self.backend_name = backend
        self._engine = None
        self.backend_desc = "unset"

    # -- lifecycle ---------------------------------------------------------
    def warm(self) -> None:
        """Load the model (takes a few seconds the first time)."""
        if self._engine is not None:
            return
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        try:
            self._engine = litert_lm.Engine(
                self.model_path, backend=GPU(gpu_decode_steps_per_sync=4)
            )
            self.backend_desc = "GPU (Metal)"
        except Exception:
            self._engine = litert_lm.Engine(self.model_path, backend=CPU())
            self.backend_desc = "CPU (XNNPACK)"

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

    @property
    def engine(self):
        self.warm()
        return self._engine

    # -- generation --------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int) -> GenResult:
        engine = self.engine
        prompt_tokens = len(engine.tokenize(prompt))
        parts: list[str] = []
        ttft = None

        started = time.perf_counter()
        with engine.create_conversation(max_output_tokens=max_tokens) as conv:
            for chunk in conv.send_message_async(prompt):
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                for item in chunk.get("content", []) or []:
                    if item.get("type") == "text":
                        parts.append(item["text"])
            total = (time.perf_counter() - started) * 1000
            used = conv.token_count

        text = "".join(parts)
        output_tokens = max(0, used - prompt_tokens)
        decode_ms = max(0.0, total - (ttft or 0.0))
        return GenResult(
            text=text,
            route="local",
            ttft_ms=round(ttft or total, 2),
            total_ms=round(total, 2),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            decode_ms=round(decode_ms, 2),
            extra={"backend": self.backend_desc},
        )


# ------------------------------------------------------------------- cloud
class CloudModel:
    name = "cloud"

    def __init__(
        self,
        model_id: str = CLOUD_MODEL_ID,
        limiter: "RateLimiter | None" = None,
    ) -> None:
        self.model_id = model_id
        self.client = OpenAI(
            base_url=CLOUD_BASE_URL,
            api_key=os.environ.get(CLOUD_API_KEY_ENV, ""),
        )
        self.limiter = limiter or RateLimiter()

    def _create_stream(self, prompt: str, max_tokens: int, extra_body: dict):
        """Open a streaming completion, retrying transient errors.

        OpenRouter returns 429 both for per-minute bursts and for the
        free-model daily quota. The former clears with a short wait, the
        latter does not - so we retry a few times and then give up with a
        message that says which of the two it was.
        """
        last_error: Exception | None = None
        for attempt in range(CLOUD_MAX_RETRIES):
            try:
                self.limiter.wait()
                return self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=CLOUD_TEMPERATURE,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra_body,
                )
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                message = str(exc)
                # OpenRouter's free tier has a hard daily cap that a retry
                # cannot clear. Other 429s (per-minute burst limits, provider
                # saturation) are transient and worth retrying.
                if status == 429 and "free-models-per-day" in message:
                    raise QuotaExhausted(
                        f"OpenRouter free-model daily quota exhausted for "
                        f"{self.model_id}: {message[:200]}"
                    ) from exc
                if attempt < CLOUD_MAX_RETRIES - 1:
                    time.sleep(CLOUD_RETRY_BACKOFF_S * (attempt + 1))
        raise last_error  # type: ignore[misc]

    def generate(self, prompt: str, max_tokens: int) -> GenResult:
        extra_body = {"reasoning": {"enabled": False}} if DISABLE_CLOUD_REASONING else {}
        started = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage = None

        stream = self._create_stream(prompt, max_tokens, extra_body)
        for chunk in stream:
            if ttft is None:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is not None and delta.content:
                    ttft = (time.perf_counter() - started) * 1000
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                parts.append(piece)

        total = (time.perf_counter() - started) * 1000
        text = "".join(parts)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = float(getattr(usage, "cost", 0.0) or 0.0)
        decode_ms = max(0.0, total - (ttft or 0.0))

        return GenResult(
            text=text,
            route="cloud",
            ttft_ms=round(ttft if ttft is not None else total, 2),
            total_ms=round(total, 2),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            decode_ms=round(decode_ms, 2),
            cost_usd=cost,
            extra={"model": self.model_id},
        )


# ------------------------------------------------------------------ hybrid
class HybridModel:
    name = "hybrid"

    def __init__(self, local: LocalModel, cloud: CloudModel, router=None) -> None:
        self.local = local
        self.cloud = cloud
        self._router = router

    @property
    def router(self):
        if self._router is None:
            import laya

            self._router = laya.Router()
        return self._router

    def decide(self, prompt: str) -> tuple[str, dict]:
        """Ask Laya which tier should handle this prompt."""
        import laya

        answers = self.router.predict(
            {"request": prompt}, laya.router_questions()
        )["answers"]
        difficulty = answers["difficulty"]["score"]
        tools_prob = answers["needs_tools"]["noul"]
        sensitive_prob = answers["is_sensitive"]["noul"]

        use_cloud = (
            difficulty >= CLOUD_DIFFICULTY_THRESHOLD
            or tools_prob >= CLOUD_TOOLS_THRESHOLD
            or sensitive_prob >= CLOUD_SENSITIVE_THRESHOLD
        )
        reason = {
            "difficulty": difficulty,
            "tools_prob": tools_prob,
            "sensitive_prob": sensitive_prob,
            "domain": answers["domain"]["choice"],
        }
        return ("cloud" if use_cloud else "local"), reason

    def generate(self, prompt: str, max_tokens: int) -> GenResult:
        started = time.perf_counter()
        route, reason = self.decide(prompt)
        decision_ms = (time.perf_counter() - started) * 1000

        backend = self.cloud if route == "cloud" else self.local
        result = backend.generate(prompt, max_tokens)
        result.route = route
        result.extra.update(reason)
        result.extra["decision_ms"] = round(decision_ms, 2)
        # Wall-clock as the caller experiences it: routing overhead + generation.
        result.total_ms = round(decision_ms + result.total_ms, 2)
        return result


def build_systems(include: list[str] | None = None) -> dict[str, Any]:
    """Instantiate the requested systems, sharing one local engine."""
    include = include or ["local", "cloud", "hybrid"]
    local = LocalModel()
    cloud = CloudModel()
    systems: dict[str, Any] = {}
    if "local" in include:
        systems["local"] = local
    if "cloud" in include:
        systems["cloud"] = cloud
    if "hybrid" in include:
        systems["hybrid"] = HybridModel(local, cloud)
    return systems


def result_to_dict(result: GenResult) -> dict:
    data = asdict(result)
    data["output_tok_per_s"] = round(result.output_tok_per_s, 2)
    return data
