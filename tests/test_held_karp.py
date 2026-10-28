import unittest
from itertools import permutations

import numpy as np

from graphs.complete_graph import CompleteGraph
from graphs.euclidean_graph import EuclideanGraph
from traveling_salesman.held_karp import TSP_HK


class TestHeldKarp(unittest.TestCase):
    def test_four_city_example_from_paper(self):
        graph = EuclideanGraph([
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ])

        tour = TSP_HK().calculateWalk(graph)

        # Starts at 0 and visits every vertex exactly once.
        self.assertEqual(tour[0], 0)
        self.assertEqual(sorted(tour), graph.vertices)

        # Different optimal tours are possible, so check the distance.
        self.assertAlmostEqual(
            graph.tourDistance(tour),
            4 * np.sqrt(2),
        )

    def test_matches_brute_force(self):
        # Unequal, directional weights provide a less symmetric example.
        graph = CompleteGraph([
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ])

        # Independently check every tour starting at vertex 0.
        optimal_distance = min(
            graph.tourDistance([0, *order])
            for order in permutations(graph.vertices[1:])
        )

        tour = TSP_HK().calculateWalk(graph)

        self.assertEqual(tour[0], 0)
        self.assertEqual(sorted(tour), graph.vertices)
        self.assertAlmostEqual(
            graph.tourDistance(tour),
            optimal_distance,
        )

    def test_single_vertex(self):
        graph = CompleteGraph([[0]])

        tour = TSP_HK().calculateWalk(graph)

        self.assertEqual(tour, [0])
        self.assertEqual(graph.tourDistance(tour), 0)


if __name__ == "__main__":
    unittest.main()