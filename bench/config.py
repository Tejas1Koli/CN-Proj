"""Benchmark configuration: paths, models, sample sizes, run modes."""

import os
from pathlib import Path

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Pull KEY=value pairs out of the project .env into the environment."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
RESULTS_DIR = ROOT / "results"
RAW_PATH = RESULTS_DIR / "raw_paired.jsonl"
SYSTEM_RAW_PATH = RESULTS_DIR / "raw_systems.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------- models
LOCAL_MODEL_PATH = os.path.expanduser(
    "~/.litert-lm/models/gemma4-e2b/model.litertlm"
)
# Cloud provider. "nvidia" = NVIDIA's integrate.api endpoint (build.nvidia.com),
# "openrouter" = OpenRouter. NVIDIA has no per-day request cap, so the whole
# evaluation set fits in one run.
CLOUD_PROVIDER = os.environ.get("CLOUD_PROVIDER", "nvidia").lower()

CLOUD_MODEL_ID = os.environ.get(
    "CLOUD_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
)
CLOUD_BASE_URL = os.environ.get(
    "CLOUD_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
    if CLOUD_PROVIDER == "nvidia"
    else "https://openrouter.ai/api/v1",
)
CLOUD_API_KEY_ENV = (
    "NVIDIA_API_KEY" if CLOUD_PROVIDER == "nvidia" else "OPENROUTER_API_KEY"
)

# Laya routing thresholds (kept identical to app.py so the notebook measures
# the deployed system, not a different configuration).
CLOUD_DIFFICULTY_THRESHOLD = 1.5
CLOUD_TOOLS_THRESHOLD = 0.14
CLOUD_SENSITIVE_THRESHOLD = 0.5

# ------------------------------------------------------------------ run mode
# BENCH_MODE=quick -> small samples (default, ~15 min per model)
# BENCH_MODE=full  -> larger samples for publication-grade numbers
MODE = os.environ.get("BENCH_MODE", "quick").lower()

SAMPLE_SIZES = {
    # "quick" is a fast pilot: enough items per task to show real numbers
    # across all six benchmarks without a multi-hour run. Because the harness
    # resumes, re-running with BENCH_MODE=full accumulates the larger sample.
    "quick": {
        "mmlu": 6,
        "gsm8k": 6,
        "humaneval": 6,
        "mbpp": 6,
        "boolq": 6,
        "arc_challenge": 6,
    },
    "full": {
        "mmlu": 500,
        "gsm8k": 250,
        "humaneval": 164,
        "mbpp": 200,
        "boolq": 500,
        "arc_challenge": 250,
    },
    "smoke": {
        "mmlu": 6,
        "gsm8k": 4,
        "humaneval": 6,
        "mbpp": 4,
        "boolq": 6,
        "arc_challenge": 6,
    },
}

SAMPLES = SAMPLE_SIZES[MODE]

# ---------------------------------------------------------------- determinism
SEED = 1234
# Both models answer greedily. The LiteRT-LM checkpoint has sampler temp 0.0
# baked in; we pass temperature=0 to the cloud model to match.
CLOUD_TEMPERATURE = 0.0

# Retry policy for transient OpenRouter errors (per-minute burst limits).
CLOUD_MAX_RETRIES = 4
CLOUD_RETRY_BACKOFF_S = 2.0

# Reasoning is disabled on the cloud model so that neither side is allowed to
# spend hidden tokens on a reasoning trace (the local model has no thinking
# mode at all: `Supports Thinking: NO`).
DISABLE_CLOUD_REASONING = True

# --------------------------------------------------------------- exec sandbox
CODE_TIMEOUT_S = 15
CODE_MEMORY_LIMIT_BYTES = 2 * 1024**3  # 2 GiB

MODEL_LABELS = {
    "local": "Local only\nGemma 4 E2B (LiteRT-LM, Metal)",
    "cloud": "Cloud only\nNemotron-3-Ultra (OpenRouter)",
    "hybrid": "Hybrid\nLaya router -> local/cloud",
}
