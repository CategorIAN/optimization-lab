
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import numpy as np

from experiments.tour_comparison import TourComparison
from graphs.euclidean_graph import EuclideanGraph
from traveling_salesman.held_karp import TSP_HK


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "results" / "tour_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Original Analysis_3 hyperparameters.
    hyperparameters = {
        "p": 4,
        "T": 2,
        "eps": 0.01,
        "n": 3,
        "alpha": 0.1,
        "beta": 5,
        "gamma": 0.9,
    }

    # Exact graphs from the original experiments.
    # Vertex order and duplicate coordinates are preserved.
    graph_datasets = {
        "9_vertices": [
            [
                [-2, 3], [-4, 3], [-4, -2],
                [-5, -2], [-5, -5], [0, -5],
                [4, -4], [4, 3], [0, 2],
            ],
            [
                [-3, 5], [0, 5], [5, 5],
                [5, 4], [4, 2], [-1, -1],
                [-4, -5], [-2, -1], [-2, 3],
            ],
            [
                [-1, 3], [-2, 5], [-3, 0],
                [-1, -1], [0, -5], [4, -5],
                [2, -1], [4, 1], [0, 1],
            ],
        ],

        "14_vertices": [
            [
                [0, -4], [0, -5], [-2, -4],
                [-1, -2], [-1, -1], [-5, 1],
                [-3, 5], [-1, 3], [2, 5],
                [2, 5], [5, 1], [2, -3],
                [0, -2], [0, -3],
            ],
            [
                [1, -4], [1, -3], [4, 2],
                [5, 4], [2, 3], [2, 3],
                [0, 5], [-1, 5], [-3, 5],
                [-3, 4], [-3, 2], [-1, 2],
                [0, 0], [-2, -5],
            ],
            [
                [-3, 2], [-1, 5], [-1, 5],
                [5, 4], [5, 2], [2, 0],
                [3, -5], [-1, -3], [-2, -4],
                [-4, -1], [-3, 0], [-2, 0],
                [-3, 2], [-3, 2],
            ],
        ],
        "17_vertices": [
            [
                [4, 4], [1, 5], [-4, 5], [-5, 5],
                [-4, 4], [-4, 3], [-2, 2], [0, 3],
                [1, 2], [0, -1], [-1, -1], [-2, -4],
                [0, -5], [1, -4], [2, -5], [3, -3],
                [5, 0],
            ],
            [
                [-1, -2], [-2, -4], [-4, -5], [0, -5],
                [1, -3], [4, -3], [4, 0], [5, 0],
                [5, 3], [3, 2], [0, 4], [0, 4],
                [-1, 5], [-2, 5], [-3, 1], [-2, 1],
                [0, -2],
            ],
            [
                [-4, 2], [-4, 2], [-5, 3], [-2, 4],
                [2, 5], [4, 3], [1, 2], [2, 0],
                [2, -2], [5, -3], [5, -4], [3, -5],
                [1, -5], [-1, -3], [-2, -3], [-3, -4],
                [-2, 2],
            ],
        ],
    }

    solver = TSP_HK()

    def graph_record(points):
        graph = EuclideanGraph(points)
        optimal_tour = solver.calculateWalk(graph)

        return {
            "points": points,
            "optimal_tour": [
                int(vertex) for vertex in optimal_tour
            ],
            "optimal_distance": float(
                graph.tourDistance(optimal_tour)
            ),
        }

    # Run each dataset as an independent experiment.
    for dataset_name, graph_points in graph_datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Running dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Reset random generators for each experiment.
        random.seed(42)
        np.random.seed(42)

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Solve graphs and prepare JSON dataset.
        records = [
            graph_record(points)
            for points in graph_points
        ]

        graph_file = dataset_dir / "graphs.json"

        with open(graph_file, "w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                indent=2,
                allow_nan=False,
            )

        # Verify optimal distances.
        print("\nOptimal distances:")
        for index, record in enumerate(records, start=1):
            print(
                f"Graph {index}: "
                f"{record['optimal_distance']:.6f}"
            )

        # Run Q-learning on the fixed graphs.
        experiment = TourComparison(
            hyperparameters,
            output_dir=dataset_dir,
        )

        comparisons = experiment.run(
            graph_file=graph_file,
        )

        # Save and display results.
        results = experiment.saveResults(comparisons)

        print("\nTour comparison:")
        print(results.to_string(index=False))

        experiment.drawGraphs(comparisons)

        print(f"\nFiles saved in: {dataset_dir}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()

