import random
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import pandas as pd

from experiments.tour_comparison import TourComparison
from graphs.euclidean_graph_distribution import (
    EuclideanGraphDistribution,
)
from traveling_salesman.walked_graphs import WalkedGraphs


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# The configuration selected in your original paper.
PAPER_PARAMETERS = {
    "p": 4,
    "T": 2,
    "eps": 0.01,
    "n": 3,
    "alpha": 0.1,
    "beta": 5,
    "gamma": 0.9,
}


def plot_training_tours(vertex_counts=(9, 14, 17), episodes=3):
    """Run a separate training experiment for each graph size."""
    for vertex_count in vertex_counts:
        output_dir = (
            RESULTS_DIR
            / "paper_experiments"
            / f"vertices_{vertex_count}"
        )

        distribution = EuclideanGraphDistribution(
            pt_number_lim=(vertex_count, vertex_count),
            seed=42,
        )

        experiment = TourComparison(
            PAPER_PARAMETERS,
            output_dir=output_dir,
        )

        comparisons = experiment.run(
            graph_file=output_dir / "graphs.json",
            graphDist=distribution,
            n=episodes,
        )

        results = experiment.saveResults(comparisons)
        print(f"\nGraphs with {vertex_count} vertices:")
        print(results.to_string(index=False))

        experiment.drawGraphs(comparisons)


def print_graphs_as_latex(graph_file, start_row=0):
    """Display a JSON graph dataset as a flat LaTeX table."""
    solved_graphs = WalkedGraphs(graph_file).graphs

    records = [
        {
            "Graph": graph_index,
            "Vertex": vertex,
            "x": point[0],
            "y": point[1],
            "Optimal distance": solved.distance,
        }
        for graph_index, solved in enumerate(solved_graphs)
        for vertex, point in enumerate(solved.graph.points)
    ]

    dataframe = pd.DataFrame(records)

    print(
        dataframe.iloc[start_row:].to_latex(index=False)
    )


def print_best_results_as_latex(error_file):
    """Display saved best-result rows as a LaTeX table."""
    dataframe = pd.read_csv(error_file, index_col=0)
    print(dataframe.to_latex())


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Begin with a small run to check the complete workflow.
    plot_training_tours(vertex_counts=(9,14, 17), episodes=3)

    # After checking the small run, use the original sizes:
    # plot_training_tours()

    # Export a dataset table:
    # print_graphs_as_latex(
    #     RESULTS_DIR / "tsp_small" / "graphs.json",
    # )

    # Export the configurations saved by HyperparameterResults.bestRows():
    # print_best_results_as_latex(
    #     RESULTS_DIR / "tsp_small" / "summaries" / "Error_Best.csv"
    # )