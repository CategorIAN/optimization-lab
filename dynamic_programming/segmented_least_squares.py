from typing import Any
import numpy as np
from functools import reduce
from typing import NamedTuple
from tail_recursive import tail_recursive
from math import inf


class SegmentationResult(NamedTuple):
    segments: list[tuple[int, int]]
    error: float


def fit_line(x: Any, a: Any, b: Any) -> Any:
    """Execute ``fit_line``.

    Args:
        x: Input accepted by this function.
        a: Input accepted by this function.
        b: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return a * np.asarray(x) + b


def best_line(x: Any, y: Any) -> Any:
    """Execute ``best_line``.

    Args:
        x: Input accepted by this function.
        y: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
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


def least_squares_error(x: Any, y: Any, i: Any, j: Any) -> Any:
    """Execute ``least_squares_error``.

    Args:
        x: Input accepted by this function.
        y: Input accepted by this function.
        i: Input accepted by this function.
        j: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    if i >= j:
        return 0.0

    x_segment = np.asarray(x[i:j + 1])
    y_segment = np.asarray(y[i:j + 1])

    a, b = best_line(x_segment, y_segment)

    residuals = y_segment - fit_line(x_segment, a, b)

    return float(residuals @ residuals)


def error_matrix(x: Any, y: Any) -> Any:
    """Execute ``error_matrix``.

    Args:
        x: Input accepted by this function.
        y: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    n = len(x)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = least_squares_error(x, y, i, j)

    return matrix


def recover_segments(segment_array: Any) -> Any:
    """Execute ``recover_segments``.

    Args:
        segment_array: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    @tail_recursive
    def go(end: Any, segments: Any) -> Any:
        """Execute ``go``.

        Args:
            end: Input accepted by this function.
            segments: Input accepted by this function.

        Returns:
            Result produced by this function.
        """
        if end < 0:
            return segments

        start = segment_array[end]

        return go.tail_call(
            start - 1,
            [(start, end)] + segments
        )

    return go(len(segment_array) - 1, [])


def segmented_least_squares(x: Any, y: Any, cost: Any) -> Any:
    """Execute ``segmented_least_squares``.

    Args:
        x: Input accepted by this function.
        y: Input accepted by this function.
        cost: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    errors = error_matrix(x, y)
    n = len(x)

    if n < 2:
        raise ValueError("At least two points are required.")

    def dynamic_update(array: Any, j: Any) -> Any:
        """Execute ``dynamic_update``.

        Args:
            array: Input accepted by this function.
            j: Input accepted by this function.

        Returns:
            Result produced by this function.
        """
        candidates = [
            (i, errors[i, j] + cost + array[i][1])
            for i in range(j)
        ]

        index, objective = min(
            candidates,
            key=lambda item: item[1]
        ) if candidates else (-1, inf)

        return array + [(index, objective)]

    results = reduce(
        dynamic_update,
        range(n),
        [(-1, 0.0)]
    )

    if results[-1][1] == inf:
        raise ValueError("No valid segmentation exists.")

    segment_array = [start for start, _ in results[1:]]

    return recover_segments(segment_array), results[-1][1]