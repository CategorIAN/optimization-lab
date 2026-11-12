from typing import Any
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.hyperparameter_search import HyperparameterSearch
from graphs.euclidean_graph import EuclideanGraph
from traveling_salesman.walked_graphs import SolvedGraph


class TestHyperparameterSearch(unittest.TestCase):
    def setUp(self) -> Any:
        """Execute ``setUp``.

        Returns:
            Result produced by this function.
        """
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)

        self.output_dir = Path(temporary_directory.name)

        self.hyperparameters = {
            "p": [2, 3],
            "T": [1],
            "eps": [0.05],
            "n": [2],
            "alpha": [0.001],
            "beta": [1],
            "gamma": [1],
        }

        graph = EuclideanGraph([
            (0, 0),
            (3, 0),
            (3, 4),
        ])

        solved_graph = SolvedGraph(
            graph=graph,
            shortest_cycle=[0, 1, 2],
            distance=12.0,
        )

        # Supply five small solved examples without creating a dataset.
        with patch(
            "experiments.hyperparameter_search.WalkedGraphs"
        ) as walked_graphs:
            walked_graphs.return_value.graphs = [
                solved_graph for _ in range(5)
            ]

            self.search = HyperparameterSearch(
                hyp_range_dict=self.hyperparameters,
                k=2,
                output_dir=self.output_dir,
            )

    def test_generates_hyperparameter_combinations(self) -> Any:
        """Execute ``test_generates_hyperparameter_combinations``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(len(self.search.hyp_dicts), 2)
        self.assertEqual(
            [config["p"] for config in self.search.hyp_dicts],
            [2, 3],
        )

    def test_partition_distributes_all_indices(self) -> Any:
        # Five examples split into folds of three and two.
        """Execute ``test_partition_distributes_all_indices``.

        Returns:
            Result produced by this function.
        """
        self.assertEqual(
            self.search.partition(2),
            [[0, 1, 2], [3, 4]],
        )

    def test_training_and_test_sets_are_separate(self) -> Any:
        """Execute ``test_training_and_test_sets_are_separate``.

        Returns:
            Result produced by this function.
        """
        all_indices = set(range(5))

        for train, test in self.search.train_test_dict.values():
            self.assertTrue(set(train).isdisjoint(test))
            self.assertEqual(set(train) | set(test), all_indices)

    def test_approximation_ratio_disables_exploration(self) -> Any:
        """Execute ``test_approximation_ratio_disables_exploration``.

        Returns:
            Result produced by this function.
        """
        solved_graph = self.search.graphs[0]

        with patch(
            "experiments.hyperparameter_search.TSP_RL"
        ) as model_class:
            model = model_class.return_value
            model.calculateWalk.return_value = [0, 1, 2]
            theta = object()

            evaluate = self.search.approx_ratio(model, theta)

            self.assertAlmostEqual(evaluate(solved_graph), 1.0)

            model.calculateWalk.assert_called_once_with(
                theta,
                solved_graph.graph,
                exploreOption=False,
            )

    def test_saves_and_resumes_results(self) -> Any:
        # Test persistence independently of neural-network training.
        """Execute ``test_saves_and_resumes_results``.

        Returns:
            Result produced by this function.
        """
        with patch.object(
            self.search,
            "hyp_error",
            side_effect=[1.2, 1.1],
        ) as calculate_error:
            first = self.search.getErrorDf()
            second = self.search.getErrorDf()

            self.assertEqual(list(first.index), [0])
            self.assertEqual(list(second.index), [0, 1])

            self.assertAlmostEqual(second.loc[0, "Error"], 1.2)
            self.assertAlmostEqual(second.loc[1, "Error"], 1.1)

            self.assertTrue(
                (self.output_dir / "Error.csv").exists()
            )
            self.assertTrue(
                (self.output_dir / "Error_1.csv").exists()
            )

            # Once complete, another call should not evaluate again.
            completed = self.search.getErrorDf()

            self.assertEqual(len(completed), 2)
            self.assertEqual(calculate_error.call_count, 2)

            self.assertEqual(
                calculate_error.call_args_list[0].args[0]["p"],
                2,
            )
            self.assertEqual(
                calculate_error.call_args_list[1].args[0]["p"],
                3,
            )


if __name__ == "__main__":
    unittest.main()