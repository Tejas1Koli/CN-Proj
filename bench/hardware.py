"""Host-side hardware instrumentation: peak RSS and process sampling."""

from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class HostProfile:
    platform: str
    machine: str
    processor: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_gb: float
    python: str


def _cpu_brand() -> str:
    """A human-readable CPU name. platform.processor() is just "arm" on macOS."""
    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def profile_host() -> HostProfile:
    vm = psutil.virtual_memory()
    return HostProfile(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=_cpu_brand(),
        cpu_count_physical=psutil.cpu_count(logical=False) or 0,
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        total_ram_gb=round(vm.total / 1024**3, 2),
        python=platform.python_version(),
    )


class MemorySampler:
    """Polls this process's RSS in the background and keeps the peak."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / 1024**2
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            rss = self._proc.memory_info().rss / 1024**2
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
        except psutil.Error:
            pass


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def directory_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if not os.path.islink(full):
                try:
                    total += os.path.getsize(full)
                except OSError:
                    pass
    return round(total / 1024**2, 1)


class Stopwatch:
    """Monotonic timer used to separate prefill / decode phases."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def reset(self) -> None:
        self.t0 = time.perf_counter()
