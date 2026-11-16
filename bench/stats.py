"""Statistics helpers: Wilson intervals, McNemar's exact test, bootstrap CIs."""

from __future__ import annotations

import math
import random
from statistics import mean, median


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs.

    b = cases where system A is right and B is wrong
    c = cases where system A is wrong and B is right
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, seed: int = 1234, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def summarize(values: list[float]) -> dict:
    """Central-tendency summary for a list of measurements."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    ordered = sorted(values)
    n = len(ordered)
    stdev = 0.0
    if n > 1:
        mu = mean(ordered)
        stdev = math.sqrt(sum((v - mu) ** 2 for v in ordered) / (n - 1))
    return {
        "n": n,
        "mean": round(mean(ordered), 3),
        "median": round(median(ordered), 3),
        "p90": round(ordered[min(n - 1, int(0.9 * n))], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "stdev": round(stdev, 3),
    }
