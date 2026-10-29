import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from graphs.euclidean_graph_distribution import (
    EuclideanGraphDistribution,
)
from traveling_salesman.walked_graphs import WalkedGraphs


class TestWalkedGraphs(unittest.TestCase):
    def setUp(self):
        # Each test gets a temporary folder, removed automatically afterward.
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)

        self.file = Path(temporary_directory.name) / "graphs.json"

    def test_creates_solved_graphs_and_saves_file(self):
        distribution = EuclideanGraphDistribution(
            pt_number_lim=(4, 4),
            seed=42,
        )

        dataset = WalkedGraphs(
            self.file,
            graphDist=distribution,
            n=2,
        )

        self.assertTrue(self.file.exists())
        self.assertEqual(len(dataset.graphs), 2)

        for solved in dataset.graphs:
            # Every vertex appears exactly once.
            self.assertEqual(
                sorted(solved.shortest_cycle),
                solved.graph.vertices,
            )

            # The stored distance matches the stored tour.
            self.assertAlmostEqual(
                solved.distance,
                solved.graph.tourDistance(solved.shortest_cycle),
            )

    def test_save_and_load_preserves_data(self):
        distribution = EuclideanGraphDistribution(
            pt_number_lim=(4, 4),
            seed=42,
        )

        original = WalkedGraphs(
            self.file,
            graphDist=distribution,
            n=2,
        )

        loaded = WalkedGraphs(self.file)

        self.assertEqual(len(original.graphs), len(loaded.graphs))

        for before, after in zip(original.graphs, loaded.graphs):
            # Coordinates retain their original vertex order.
            np.testing.assert_array_equal(
                before.graph.points,
                after.graph.points,
            )

            self.assertEqual(
                before.shortest_cycle,
                after.shortest_cycle,
            )

            self.assertAlmostEqual(
                before.distance,
                after.distance,
            )

    def test_existing_file_does_not_generate_graphs(self):
        distribution = EuclideanGraphDistribution(
            pt_number_lim=(3, 3),
            seed=42,
        )
        WalkedGraphs(self.file, graphDist=distribution, n=1)

        unused_distribution = Mock()

        loaded = WalkedGraphs(
            self.file,
            graphDist=unused_distribution,
            n=5,
        )

        unused_distribution.randomGraph.assert_not_called()
        self.assertEqual(len(loaded.graphs), 1)

    def test_missing_file_requires_generation_arguments(self):
        with self.assertRaises(ValueError):
            WalkedGraphs(self.file)


if __name__ == "__main__":
    unittest.main()