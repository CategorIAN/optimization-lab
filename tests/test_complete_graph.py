from typing import Any
import unittest

import numpy as np

from graphs.complete_graph import CompleteGraph


class TestCompleteGraph(unittest.TestCase):
    def setUp(self) -> Any:
        """Create a small graph for each test."""
        distances = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ])
        self.graph = CompleteGraph(distances)

    def test_walk_distance(self) -> Any:
        # 0 → 1 costs 1; 1 → 2 costs 3.
        """Execute ``test_walk_distance``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(self.graph.walkDistance([0, 1, 2]), 4)

    def test_tour_distance(self) -> Any:
        # The tour also returns from 2 → 0, which costs 2.
        """Execute ``test_tour_distance``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(self.graph.tourDistance([0, 1, 2]), 6)

    def test_neighbor_vector(self) -> Any:
        """Execute ``test_neighbor_vector``.

        Returns:
            Result produced by this function.
        """
        expected = np.array([[1], [0], [1]])

        np.testing.assert_array_equal(
            self.graph.neighbor_vec(1),
            expected,
        )

    def test_neighbor_square(self) -> Any:
        """Execute ``test_neighbor_square``.

        Returns:
            Result produced by this function.
        """
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ])

        np.testing.assert_array_equal(
            self.graph.neighbor_square(1),
            expected,
        )

    def test_edges(self) -> Any:
        """Execute ``test_edges``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(
            set(self.graph.edges),
            {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)},
        )

    def test_close_walk(self) -> Any:
        """Execute ``test_close_walk``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(self.graph.closeWalk([0, 1, 2]), [0, 1, 2, 0])

    def test_close_walk_already_closed(self) -> Any:
        """Execute ``test_close_walk_already_closed``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(self.graph.closeWalk([0, 1, 0]), [0, 1, 0])

    def test_non_square_matrix(self) -> Any:
        """Execute ``test_non_square_matrix``.

        Returns:
            Result produced by this function.
        """
        with self.assertRaises(ValueError):
            CompleteGraph([[0, 1, 2], [1, 0, 3]])


if __name__ == "__main__":
    unittest.main()