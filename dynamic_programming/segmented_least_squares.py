import numpy as np
from functools import reduce
from typing import NamedTuple
from tail_recursive import tail_recursive


class SegmentationResult(NamedTuple):
    segments: list[tuple[int, int]]
    error: float


def fit_line(x, a, b):
    return a * np.asarray(x) + b


def best_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)

    if n == 0:
        raise ValueError("Cannot fit a line to empty data.")

    if n == 1:
        return 0.0, float(y[0])

    denominator = n * (x @ x) - np.sum(x) ** 2

    if np.isclose(denominator, 0):
        return 0.0, float(np.mean(y))

    a = (n * (x @ y) - np.sum(x) * np.sum(y)) / denominator
    b = (np.sum(y) - a * np.sum(x)) / n

    return float(a), float(b)


def least_squares_error(x, y, i, j):
    if i >= j:
        return 0.0

    x_segment = np.asarray(x[i:j + 1])
    y_segment = np.asarray(y[i:j + 1])

    a, b = best_line(x_segment, y_segment)

    residuals = y_segment - fit_line(x_segment, a, b)

    return float(residuals @ residuals)


def error_matrix(x, y):
    n = len(x)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = least_squares_error(x, y, i, j)

    return matrix


def recover_segments(segment_array):
    @tail_recursive
    def go(end, segments):
        if end < 0:
            return segments

        start = segment_array[end]

        return go.tail_call(
            start - 1,
            [(start, end)] + segments
        )

    return go(len(segment_array) - 1, [])


def segmented_least_squares(x, y, cost):
    errors = error_matrix(x, y)

    def dynamic_update(array, j):
        index, error = min(
            (
                (i, errors[i, j] + cost + array[i][1])
                for i in range(j + 1)
            ),
            key=lambda item: item[1]
        )

        return array + [(index, error)]

    results = reduce(
        dynamic_update,
        range(len(x)),
        [(-1, 0.0)]
    )

    segment_array = [start for start, _ in results[1:]]
    optimal_cost = results[-1][1]

    return recover_segments(segment_array), optimal_cost