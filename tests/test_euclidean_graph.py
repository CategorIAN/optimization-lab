from typing import Any
import unittest

import numpy as np

from graphs.euclidean_graph import EuclideanGraph


class TestEuclideanGraph(unittest.TestCase):
    def test_distance_between_points(self) -> Any:
        """Execute ``test_distance_between_points``.

        Returns:
            Result produced by this function.
        """
        graph = EuclideanGraph([(0, 0), (3, 4)])

        np.testing.assert_allclose(
            graph.w,
            [[0, 5], [5, 0]],
        )

    def test_tour_from_paper(self) -> Any:
        """Execute ``test_tour_from_paper``.

        Returns:
            Result produced by this function.
        """
        graph = EuclideanGraph([
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ])

        self.assertAlmostEqual(
            graph.tourDistance([0, 1, 2, 3]),
            4 * np.sqrt(2),
        )

    def test_rejects_nonfinite_coordinates(self) -> Any:
        """Execute ``test_rejects_nonfinite_coordinates``.

        Returns:
            Result produced by this function.
        """
        with self.assertRaises(ValueError):
            EuclideanGraph([(0, 0), (np.inf, 1)])


if __name__ == "__main__":
    unittest.main()