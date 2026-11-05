import random
import numpy as np
from tail_recursive import tail_recursive
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def generate_segmented_points(
    size,
    break_prob=0,
    sigma=1,
    slope_size=1,
    seed=None,
):
    if size < 1:
        raise ValueError("size must be positive.")

    if not 0 <= break_prob <= 1:
        raise ValueError("break_prob must be between 0 and 1.")

    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")

    if not isinstance(slope_size, int) or slope_size < 0:
        raise ValueError("slope_size must be a nonnegative integer.")

    rng = random.Random(seed)

    x = np.arange(size)

    breaks = [
        i for i in range(1, size)
        if rng.random() < break_prob
    ]

    @tail_recursive
    def generate(start, remaining_breaks, y):
        slope = rng.randint(-slope_size, slope_size)

        end = (
            remaining_breaks[0]
            if remaining_breaks
            else size
        )

        values = [
            slope * xi + rng.normalvariate(0, sigma)
            for xi in x[start:end]
        ]

        new_y = y + values

        if not remaining_breaks:
            return new_y

        return generate.tail_call(
            end,
            remaining_breaks[1:],
            new_y,
        )

    y = np.asarray(generate(0, breaks, []), dtype=float)

    return x, y, breaks


import matplotlib.pyplot as plt

from dynamic_programming.segmented_least_squares import (
    segmented_least_squares,
    best_line,
    fit_line,
)


def plot_experiment():
    x, y, breaks = generate_segmented_points(
        size=30,
        break_prob=0.2,
        sigma=2,
        slope_size=1,
        seed=42,
    )

    print("Intended breaks:", breaks)

    costs = [0, 100, 1000000000]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for ax, cost in zip(axes, costs):
        segments, objective = segmented_least_squares(x, y, cost)

        ax.scatter(x, y, label="Original points")

        for start, end in segments:
            xs = x[start:end + 1]
            ys = y[start:end + 1]

            a, b = best_line(xs, ys)
            fitted = fit_line(xs, a, b)

            ax.plot(xs, fitted)

        ax.set_title(
            f"Cost = {cost}, Segments = {len(segments)}"
        )
        ax.legend()

        print(f"Cost: {cost}")
        print(f"Segments: {segments}")
        print(f"Objective: {objective}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_experiment()
