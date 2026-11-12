from typing import Any
import unittest

import numpy as np

from graphs.euclidean_graph_distribution import (
    EuclideanGraphDistribution,
)


class TestEuclideanGraphDistribution(unittest.TestCase):
    def test_generated_graph_respects_limits(self) -> Any:
        """Execute ``test_generated_graph_respects_limits``.

        Returns:
            Result produced by this function.
        """
        distribution = EuclideanGraphDistribution(
            pt_number_lim=(3, 5),
            xlim=(-2, 2),
            ylim=(10, 12),
            seed=42,
        )

        graph = distribution.randomGraph()

        self.assertTrue(3 <= graph.n <= 5)
        self.assertTrue(np.all(graph.points[:, 0] >= -2))
        self.assertTrue(np.all(graph.points[:, 0] <= 2))
        self.assertTrue(np.all(graph.points[:, 1] >= 10))
        self.assertTrue(np.all(graph.points[:, 1] <= 12))

    def test_same_seed_reproduces_graph_sequence(self) -> Any:
        """Execute ``test_same_seed_reproduces_graph_sequence``.

        Returns:
            Result produced by this function.
        """
        first = EuclideanGraphDistribution(seed=42)
        second = EuclideanGraphDistribution(seed=42)

        for _ in range(3):
            np.testing.assert_array_equal(
                first.randomGraph().points,
                second.randomGraph().points,
            )

    def test_rejects_reversed_limits(self) -> Any:
        """Execute ``test_rejects_reversed_limits``.

        Returns:
            Result produced by this function.
        """
        with self.assertRaises(ValueError):
            EuclideanGraphDistribution(xlim=(5, -5))

    def test_rejects_zero_points(self) -> Any:
        """Execute ``test_rejects_zero_points``.

        Returns:
            Result produced by this function.
        """
        with self.assertRaises(ValueError):
            EuclideanGraphDistribution(pt_number_lim=(0, 9))


if __name__ == "__main__":
    unittest.main()