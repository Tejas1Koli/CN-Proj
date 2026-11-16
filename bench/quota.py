"""OpenRouter free-tier quota management.

The free tier allows 50 requests per UTC day per account (shared across every
`:free` model). A benchmark that runs each arm independently would spend two
requests per item -- one for the cloud arm and one for the hybrid arm -- and
run out of quota after a couple of dozen items.

The paired protocol in `bench/protocol.py` needs only ONE request per item, so
this module's job is to:

  * report how much of today's quota is left,
  * decide how many items are affordable,
  * pace requests to stay inside the 20 requests/minute burst limit,
  * stop cleanly when the daily cap is hit instead of burning more requests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import CLOUD_BASE_URL

FREE_DAILY_LIMIT = 50
FREE_PER_MINUTE = 20
MIN_REQUEST_INTERVAL_S = 60.0 / FREE_PER_MINUTE  # 3s

# NVIDIA's integrate endpoint allows a higher burst rate than OpenRouter's
# free tier, so the pacing is relaxed when that provider is in use.
NVIDIA_PER_MINUTE = 40


def min_request_interval() -> float:
    from .config import CLOUD_PROVIDER

    if CLOUD_PROVIDER == "nvidia":
        return 60.0 / NVIDIA_PER_MINUTE
    return MIN_REQUEST_INTERVAL_S


@dataclass
class Quota:
    used: int
    limit: int
    remaining: int
    resets_at_epoch_s: int
    available: bool
    note: str = ""

    @property
    def resets_at_utc(self) -> str:
        import datetime

        if not self.resets_at_epoch_s:
            return "n/a"
        return datetime.datetime.fromtimestamp(
            self.resets_at_epoch_s, datetime.UTC
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

    @property
    def hours_until_reset(self) -> float:
        if not self.resets_at_epoch_s:
            return 0.0
        return max(0.0, (self.resets_at_epoch_s - time.time()) / 3600.0)

    def summary(self) -> str:
        if self.note:
            return self.note
        if not self.available:
            return (
                f"quota unavailable ({self.used}/{self.limit} used)"
                if self.limit
                else "quota unavailable"
            )
        if self.remaining <= 0:
            return (
                f"daily free quota exhausted ({self.used}/{self.limit} used); "
                f"resets in {self.hours_until_reset:.1f}h at {self.resets_at_utc}"
            )
        return (
            f"free quota: {self.remaining}/{self.limit} requests left today "
            f"(resets at {self.resets_at_utc})"
        )


def check_quota(api_key: str | None = None) -> Quota:
    """Ask OpenRouter how many free-model requests are left today.

    NVIDIA's integrate endpoint has no per-day request cap, so for that
    provider we report an effectively unlimited budget and only rate-limit.
    """
    from .config import CLOUD_PROVIDER

    if CLOUD_PROVIDER != "openrouter":
        return Quota(
            used=0,
            limit=10**9,
            remaining=10**9,
            resets_at_epoch_s=0,
            available=True,
            note=f"{CLOUD_PROVIDER} provider: no daily request cap",
        )

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return Quota(0, FREE_DAILY_LIMIT, 0, 0, available=False)

    request = urllib.request.Request(
        f"{CLOUD_BASE_URL}/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return Quota(0, FREE_DAILY_LIMIT, 0, 0, available=False)

    daily = (payload.get("data") or {}).get("free_model_daily_requests") or {}
    used = int(daily.get("used", 0))
    limit = int(daily.get("limit", FREE_DAILY_LIMIT))
    remaining = int(daily.get("remaining", max(0, limit - used)))

    # The daily window is a UTC calendar day.
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    reset = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    return Quota(
        used=used,
        limit=limit,
        remaining=remaining,
        resets_at_epoch_s=int(reset.timestamp()),
        available=True,
    )


class RateLimiter:
    """Keeps request spacing above the per-minute burst limit."""

    def __init__(self, min_interval_s: float | None = None) -> None:
        self.min_interval_s = (
            min_interval_s if min_interval_s is not None else min_request_interval()
        )
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last = time.monotonic()


class QuotaExhausted(RuntimeError):
    """Raised when the daily free-model quota runs out mid-run."""
