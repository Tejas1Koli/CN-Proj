"""
Cloud-local hybrid chatbot (API only).

Laya -- a System 1 decision model -- decides which tier answers each query:

    easy query    ->  local  : Gemma 4 E2B, on-device, via LiteRT-LM
    complex query ->  cloud  : a big LLM, reached through OpenRouter

Run it:
    uvicorn app:app --reload --port 8000

Call it:
    curl -X POST localhost:8000/chat \
      -H 'Content-Type: application/json' \
      -d '{"message": "What is the capital of France?"}'
"""

import os
import threading
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import laya
import litert_lm
from fastapi import FastAPI
from litert_lm.interfaces import CPU, GPU
from openai import OpenAI
from pydantic import BaseModel


# --------------------------------------------------------------------- setup
def load_env(path=".env"):
    """Read KEY=value lines from .env into the environment (without overwriting)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

LOCAL_MODEL = os.environ.get(
    "LOCAL_MODEL", "~/.litert-lm/models/gemma4-e2b/model.litertlm"
)

# Cloud provider: "nvidia" (build.nvidia.com) or "openrouter".
CLOUD_PROVIDER = os.environ.get("CLOUD_PROVIDER", "nvidia").lower()
CLOUD_BASE_URL = os.environ.get(
    "CLOUD_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
    if CLOUD_PROVIDER == "nvidia"
    else "https://openrouter.ai/api/v1",
)
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
CLOUD_API_KEY = os.environ.get(
    "NVIDIA_API_KEY" if CLOUD_PROVIDER == "nvidia" else "OPENROUTER_API_KEY", ""
)

# A Laya difficulty score at or above this counts as "complex" -> send to the cloud.
# Measured on this checkpoint: easy queries score ~0.6-1.4, hard ones ~2.0+,
# so 1.5 cleanly splits the two groups.
CLOUD_THRESHOLD = 1.5

# "Needs tools" is a probability, and this checkpoint barely discriminates: real
# tool-requiring queries ("search the web...", "look up ... population") score
# 0.14-0.15, while plain factual lookups ("capital of France") score ~0.12.
# A threshold of 0.5 therefore never fires at all. 0.14 catches the live-data
# cases while leaving simple lookups local.
CLOUD_TOOLS_THRESHOLD = 0.14

app = FastAPI(title="Hybrid Chatbot API")

# Laya loads its checkpoints on the first predict() call, so this line is cheap.
router = laya.Router()
cloud = OpenAI(
    base_url=CLOUD_BASE_URL,
    api_key=CLOUD_API_KEY,
)

_local_engine = None
_local_lock = threading.Lock()  # the LiteRT-LM engine handles one conversation at a time


def local_engine():
    """The LiteRT-LM engine. Loading the model takes seconds, so keep one alive."""
    global _local_engine
    if _local_engine is None:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        path = os.path.expanduser(LOCAL_MODEL)
        try:
            # Metal GPU on Apple silicon. Measured on an M2: 51.8 tok/s vs 30.7 on CPU.
            _local_engine = litert_lm.Engine(
                path, backend=GPU(gpu_decode_steps_per_sync=4)
            )
        except Exception:
            # No usable GPU on this machine -- fall back to the CPU backend.
            _local_engine = litert_lm.Engine(path, backend=CPU())
    return _local_engine


# -------------------------------------------------------------------- models
def ask_local(message: str) -> str:
    """Answer on-device with Gemma 4 E2B."""
    with _local_lock:
        engine = local_engine()
        with engine.create_conversation() as conversation:
            reply = conversation.send_message(message)
    return reply["content"][0]["text"]


def ask_cloud(message: str) -> str:
    """Answer with a cloud LLM through OpenRouter."""
    reply = cloud.chat.completions.create(
        model=CLOUD_MODEL,
        messages=[{"role": "user", "content": message}],
    )
    return reply.choices[0].message.content or ""


# -------------------------------------------------------------------- router
def decide_route(message: str) -> tuple[str, dict]:
    """Ask Laya how hard the query is, then pick the tier."""
    answers = router.predict({"request": message}, laya.router_questions())["answers"]

    difficulty = answers["difficulty"]["score"]
    needs_tools_prob = answers["needs_tools"]["noul"]
    is_sensitive = answers["is_sensitive"]["noul"] > 0.5

    # Complex, tool-requiring or sensitive queries go to the cloud model.
    use_cloud = (
        difficulty >= CLOUD_THRESHOLD
        or needs_tools_prob >= CLOUD_TOOLS_THRESHOLD
        or is_sensitive
    )

    return ("cloud" if use_cloud else "local"), {
        "difficulty": difficulty,
        "domain": answers["domain"]["choice"],
        "needs_tools": needs_tools_prob >= CLOUD_TOOLS_THRESHOLD,
        "needs_tools_prob": needs_tools_prob,
        "is_sensitive": is_sensitive,
    }


# ----------------------------------------------------------------------- API
class ChatRequest(BaseModel):
    message: str
    force_route: str | None = None  # "local" or "cloud", to skip Laya while testing


class ChatResponse(BaseModel):
    reply: str
    route: str
    reason: dict
    timings_ms: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()

    if request.force_route in ("local", "cloud"):
        route, reason = request.force_route, {"forced": True}
    else:
        route, reason = decide_route(request.message)

    decided = round((time.perf_counter() - started) * 1000, 1)

    reply = ask_cloud(request.message) if route == "cloud" else ask_local(request.message)

    return ChatResponse(
        reply=reply,
        route=route,
        reason=reason,
        timings_ms={
            "decision": decided,
            "total": round((time.perf_counter() - started) * 1000, 1),
        },
    )
