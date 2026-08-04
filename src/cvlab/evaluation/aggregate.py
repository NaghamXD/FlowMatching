"""Aggregation of a metric across the seeds run per (combination, K, method) --
decision 7: mean and *sample* standard deviation (ddof=1)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Aggregate:
    mean: float
    std: float
    n: int


def aggregate(values: list[float]) -> Aggregate:
    """Sample std (ddof=1) over `values`. A single value gets std=0.0, not an
    error or NaN -- this is the expected shape for the full-split ImagePrototype
    "run" (deterministic, so recorded once rather than fabricating variance
    across 3 identical results; see the Run matrix note in the spec)."""
    if not values:
        raise ValueError("cannot aggregate an empty list of values")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return Aggregate(mean=mean, std=std, n=len(values))
