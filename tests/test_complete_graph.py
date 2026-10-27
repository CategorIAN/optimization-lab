import unittest

import numpy as np

from graphs.complete_graph import CompleteGraph


class TestCompleteGraph(unittest.TestCase):
    def setUp(self):
        """Create a small graph for each test."""
        distances = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ])
        self.graph = CompleteGraph(distances)

    def test_walk_distance(self):
        # 0 → 1 costs 1; 1 → 2 costs 3.
        self.assertEqual(self.graph.walkDistance([0, 1, 2]), 4)

    def test_tour_distance(self):
        # The tour also returns from 2 → 0, which costs 2.
        self.assertEqual(self.graph.tourDistance([0, 1, 2]), 6)

    def test_neighbor_vector(self):
        expected = np.array([[1], [0], [1]])

        np.testing.assert_array_equal(
            self.graph.neighbor_vec(1),
            expected,
        )

    def test_neighbor_square(self):
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ])

        np.testing.assert_array_equal(
            self.graph.neighbor_square(1),
            expected,
        )

    def test_edges(self):
        self.assertEqual(
            set(self.graph.edges),
            {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)},
        )

    def test_close_walk(self):
        self.assertEqual(self.graph.closeWalk([0, 1, 2]), [0, 1, 2, 0])

    def test_close_walk_already_closed(self):
        self.assertEqual(self.graph.closeWalk([0, 1, 0]), [0, 1, 0])

    def test_non_square_matrix(self):
        with self.assertRaises(ValueError):
            CompleteGraph([[0, 1, 2], [1, 0, 3]])


if __name__ == "__main__":
    unittest.main()