"""M7 test: aggregation over 3 synthetic runs reproduces a known mean and
ddof=1 std."""

from __future__ import annotations

import math

import pytest

from cvlab.evaluation.aggregate import aggregate


def test_aggregate_three_runs_known_mean_and_sample_std() -> None:
    # mean = 0.6; sample variance (ddof=1) = sum((x-mean)^2) / (n-1)
    #   deviations: -0.1, 0.0, 0.1 -> squared: 0.01, 0, 0.01 -> sum=0.02 -> /2 = 0.01 -> std=0.1
    values = [0.5, 0.6, 0.7]
    result = aggregate(values)
    assert math.isclose(result.mean, 0.6, rel_tol=1e-9)
    assert math.isclose(result.std, 0.1, rel_tol=1e-9)
    assert result.n == 3


def test_aggregate_single_value_has_zero_std_not_error() -> None:
    result = aggregate([0.42])
    assert result.mean == pytest.approx(0.42)
    assert result.std == 0.0
    assert result.n == 1


def test_aggregate_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        aggregate([])


def test_aggregate_matches_numpy_ddof1_on_random_values() -> None:
    import random

    random.seed(0)
    values = [random.random() for _ in range(10)]
    result = aggregate(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    assert math.isclose(result.mean, mean, rel_tol=1e-9)
    assert math.isclose(result.std, math.sqrt(variance), rel_tol=1e-9)
