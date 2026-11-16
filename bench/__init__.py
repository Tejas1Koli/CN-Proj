"""Benchmark harness for the cloud-local hybrid chatbot.

Systems under test:
    local   - Gemma 4 E2B via LiteRT-LM (Metal GPU)
    cloud   - Nemotron-3-Ultra via OpenRouter
    hybrid  - Laya routes each prompt to local or cloud
"""

from . import (  # noqa: F401
    config,
    data,
    hardware,
    models,
    protocol,
    quota,
    runner,
    stats,
)

__all__ = [
    "config",
    "data",
    "hardware",
    "models",
    "protocol",
    "quota",
    "runner",
    "stats",
]
