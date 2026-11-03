from functools import reduce
from itertools import product
from pathlib import Path
import time
import pandas as pd
from traveling_salesman.tsp_rl import TSP_RL
from traveling_salesman.walked_graphs import WalkedGraphs


class HyperparameterSearch:
    """Compare TSP reinforcement learning configurations using cross-validation."""

    def __init__(
        self,
        hyp_range_dict,
        k,
        graph_file=None,
        graphDist=None,
        n=None,
        output_dir="results/tsp",
    ):
        """
        Prepare hyperparameter configurations and cross-validation folds.

        hyp_range_dict: Possible values for each hyperparameter.
        k: Number of cross-validation folds.
        graph_file: JSON file containing graphs and optimal solutions.
        graphDist: Distribution used when generating a new dataset.
        n: Number of graphs to generate if the dataset does not exist.
        output_dir: Directory for experiment results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.error_file = self.output_dir / "Error.csv"

        graph_file = (
            self.output_dir / "graphs.json"
            if graph_file is None
            else Path(graph_file)
        )
        graph_file.parent.mkdir(parents=True, exist_ok=True)

        self.hyp_range_dict = hyp_range_dict
        self.hyp_names = [
            "p",
            "T",
            "eps",
            "n",
            "alpha",
            "beta",
            "gamma",
        ]

        # Generate every combination of the supplied parameter values.
        hyp_combos = product(
            *[hyp_range_dict[name] for name in self.hyp_names]
        )

        self.hyp_dicts = [
            dict(zip(self.hyp_names, values))
            for values in hyp_combos
        ]

        if not self.hyp_dicts:
            raise ValueError(
                "Each hyperparameter must have at least one value."
            )

        # Each item contains a graph, its optimal tour, and its distance.
        self.graphs = WalkedGraphs(
            graph_file,
            graphDist,
            n,
        ).graphs

        self.n = len(self.graphs)

        if type(k) is not int or not 2 <= k <= self.n:
            raise ValueError(
                "k must be an integer between 2 and the number of graphs."
            )

        if any(solved.distance <= 0 for solved in self.graphs):
            raise ValueError(
                "Approximation ratios require positive optimal distances."
            )

        self.train_test_dict = self.get_train_test_dict(k)

    def partition(self, k):
        """Partition graph indices into k approximately equal folds."""
        quotient, remainder = divmod(self.n, k)

        def build_partition(fold_index, start, folds):
            if fold_index == k:
                return folds

            # Distribute extra graphs among the first few folds.
            fold_size = quotient + int(fold_index < remainder)
            end = start + fold_size

            return build_partition(
                fold_index + 1,
                end,
                folds + [list(range(start, end))],
            )

        return build_partition(0, 0, [])

    def get_train_test_dict(self, k):
        """Map each fold to its training and test graph indices."""
        folds = self.partition(k)

        def training_indices(fold_index):
            # All folds except the current test fold form the training set.
            other_folds = (
                folds[:fold_index] + folds[fold_index + 1:]
            )

            return reduce(
                lambda accumulated, fold: accumulated + fold,
                other_folds,
                [],
            )

        return {
            fold_index: (
                training_indices(fold_index),
                folds[fold_index],
            )
            for fold_index in range(k)
        }

    def approx_ratio(self, model, Theta):
        """Return a function calculating a solved graph's approximation ratio."""

        def evaluate(solved_graph):
            graph = solved_graph.graph

            # Disable random exploration when evaluating a trained model.
            tour = model.calculateWalk(
                Theta,
                graph,
                exploreOption=False,
            )

            return (
                graph.tourDistance(tour)
                / solved_graph.distance
            )

        return evaluate

    def hyp_error(self, hyp_dict):
        """Return the mean approximation ratio across folds."""
        hyp_params = [
            hyp_dict[name]
            for name in self.hyp_names
        ]
        model = TSP_RL(*hyp_params)

        def fold_error(fold_index):
            train_indices, test_indices = (
                self.train_test_dict[fold_index]
            )

            training_graphs = [
                self.graphs[index].graph
                for index in train_indices
            ]

            # Each call initializes fresh weights and replay memory.
            _, Theta = model.QLearning(training_graphs)

            evaluate = self.approx_ratio(model, Theta)

            return sum(
                evaluate(self.graphs[index])
                for index in test_indices
            ) / len(test_indices)

        return sum(
            fold_error(fold_index)
            for fold_index in self.train_test_dict
        ) / len(self.train_test_dict)

    def getErrorDf(self, append=True):
        """
        Evaluate and save one hyperparameter configuration.

        With append=True, continue after the last saved configuration.
        If no results exist, start at configuration 0.

        Resume only with the same dataset, folds, and parameter grid.
        With append=False, restart the combined results table.
        """
        if append and self.error_file.exists():
            previous_results = pd.read_csv(
                self.error_file,
                index_col=0,
            )
        else:
            previous_results = pd.DataFrame(
                columns=self.hyp_names + ["Error"]
            )

        configuration_index = (
            0
            if previous_results.empty
            else int(previous_results.index.max()) + 1
        )

        if configuration_index >= len(self.hyp_dicts):
            print("All hyperparameter configurations are complete.")
            return previous_results

        start_time = time.perf_counter()

        hyp_dict = self.hyp_dicts[configuration_index]
        error = self.hyp_error(hyp_dict)

        new_row = pd.DataFrame(
            [{**hyp_dict, "Error": error}],
            index=[configuration_index],
        )

        # Keep both an individual result and the accumulated results.
        new_row.to_csv(
            self.output_dir / f"Error_{configuration_index}.csv"
        )

        results = (
            new_row
            if previous_results.empty
            else pd.concat([previous_results, new_row])
        )

        results.to_csv(self.error_file)

        elapsed_minutes = (
            time.perf_counter() - start_time
        ) / 60

        print(
            f"Configuration {configuration_index}: "
            f"mean approximation ratio = {error:.4f}"
        )
        print(f"Time elapsed: {elapsed_minutes:.2f} minutes")

        return results