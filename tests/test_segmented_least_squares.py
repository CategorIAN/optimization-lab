import numpy as np
import pytest

from dynamic_programming.segmented_least_squares import (
    segmented_least_squares,
    recover_segments,
)
from experiments.segmented_least_squares import generate_segmented_points


def test_generate_segmented_points():
    x, y, breaks = generate_segmented_points(
        size=30,
        break_prob=0.2,
        sigma=2,
        slope_size=1,
        seed=42,
    )

    assert len(x) == 30
    assert len(y) == 30
    assert np.array_equal(x, np.arange(30))
    assert all(1 <= i < 30 for i in breaks)


def test_reproducibility():
    result1 = generate_segmented_points(30, seed=42)
    result2 = generate_segmented_points(30, seed=42)

    assert np.array_equal(result1[0], result2[0])
    assert np.array_equal(result1[1], result2[1])
    assert result1[2] == result2[2]


def test_break_probabilities():
    _, _, no_breaks = generate_segmented_points(
        10, break_prob=0
    )

    _, _, all_breaks = generate_segmented_points(
        10, break_prob=1
    )

    assert no_breaks == []
    assert all_breaks == list(range(1, 10))


def test_recover_segments():
    segment_array = [0, 0, 0, 3, 3, 3]

    assert recover_segments(segment_array) == [
        (0, 2),
        (3, 5),
    ]


def test_perfect_linear_data():
    x = np.arange(6)
    y = 2 * x + 3

    segments, objective = segmented_least_squares(
        x, y, cost=10
    )

    assert segments == [(0, 5)]
    assert objective == pytest.approx(10)


def test_two_segments():
    x = np.arange(6)
    y = np.array([0, 1, 2, 10, 11, 12])

    segments, objective = segmented_least_squares(
        x, y, cost=1
    )

    assert segments == [(0, 2), (3, 5)]
    assert objective == pytest.approx(2)

