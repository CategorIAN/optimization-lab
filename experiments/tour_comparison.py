from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from traveling_salesman.tsp_rl import TSP_RL
from traveling_salesman.walked_graphs import WalkedGraphs


class TourComparison:
    """Compare tours constructed during training with optimal tours."""

    hyp_names = [
        "p",
        "T",
        "eps",
        "n",
        "alpha",
        "beta",
        "gamma",
    ]

    def __init__(self, hyp_dict: Any, output_dir: Any="results/tour_comparison") -> Any:
        """Execute ``__init__``.

        Args:
            hyp_dict: Input accepted by this function.
            output_dir: Input accepted by this function.

        Returns:
            Result produced by this function.
        """
        self.hyp_dict = hyp_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, graph_file: Any, graphDist: Any=None, n: Any=None) -> Any:
        """
        Load or generate solved graphs, then train across them.

        Return a list of (SolvedGraph, learned_tour) pairs.
        """
        graph_file = Path(graph_file)
        graph_file.parent.mkdir(parents=True, exist_ok=True)

        solved_graphs = WalkedGraphs(
            graph_file,
            graphDist,
            n,
        ).graphs

        if not solved_graphs:
            raise ValueError("The dataset must contain at least one graph.")

        if any(solved.distance <= 0 for solved in solved_graphs):
            raise ValueError(
                "Approximation ratios require positive optimal distances."
            )

        model = TSP_RL(
            *[self.hyp_dict[name] for name in self.hyp_names]
        )

        # Each graph is one episode. Weights and memory carry forward.
        graph_tours, _ = model.QLearning([
            solved.graph
            for solved in solved_graphs
        ])

        return [
            (solved, tour)
            for solved, (_, tour)
            in zip(solved_graphs, graph_tours)
        ]

    def saveResults(self, comparisons: Any) -> Any:
        """Save one row per training episode."""

        def episode_record(index: Any, solved: Any, tour: Any) -> Any:
            """Execute ``episode_record``.

            Args:
                index: Input accepted by this function.
                solved: Input accepted by this function.
                tour: Input accepted by this function.

            Returns:
                Result produced by this function.
            """
            approximate_distance = solved.graph.tourDistance(tour)

            return {
                "episode": index + 1,
                "vertices": solved.graph.n,
                "optimal_distance": solved.distance,
                "approximate_distance": approximate_distance,
                "approximation_ratio": (
                    approximate_distance / solved.distance
                ),
                "optimal_tour": solved.shortest_cycle,
                "approximate_tour": tour,
            }

        results = pd.DataFrame([
            episode_record(index, solved, tour)
            for index, (solved, tour) in enumerate(comparisons)
        ])

        results.to_csv(
            self.output_dir / "tour_comparison.csv",
            index=False,
        )

        return results

    def drawGraphs(self, comparisons: Any) -> Any:
        """Save and display optimal and learned tours."""
        if not comparisons:
            raise ValueError("There are no tours to plot.")

        if any(
            solved.graph.points.shape[1] != 2
            for solved, _ in comparisons
        ):
            raise ValueError("Tour plotting requires 2D coordinates.")

        columns = min(3, len(comparisons))
        rows = (len(comparisons) + columns - 1) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 4 * rows),
            squeeze=False,
        )

        fig.suptitle("Tours Constructed During Training")

        def plot_tour(axis: Any, graph: Any, tour: Any, label: Any, style: Any) -> Any:
            # Explicitly close each route using its own tour order.
            """Execute ``plot_tour``.

            Args:
                axis: Input accepted by this function.
                graph: Input accepted by this function.
                tour: Input accepted by this function.
                label: Input accepted by this function.
                style: Input accepted by this function.

            Returns:
                Result produced by this function.
            """
            closed_tour = graph.closeWalk(list(tour))
            coordinates = graph.points[closed_tour]

            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                style,
                label=label,
            )

        for index, (solved, tour) in enumerate(comparisons):
            axis = axes.flat[index]
            graph = solved.graph

            approximate_distance = graph.tourDistance(tour)
            ratio = approximate_distance / solved.distance

            plot_tour(
                axis,
                graph,
                solved.shortest_cycle,
                "Optimal",
                "--",
            )
            plot_tour(
                axis,
                graph,
                tour,
                "Learned",
                "-",
            )

            axis.scatter(
                graph.points[:, 0],
                graph.points[:, 1],
                color="black",
                zorder=3,
            )

            for vertex, point in enumerate(graph.points):
                axis.annotate(
                    str(vertex),
                    point,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            axis.set_title(f"Episode {index + 1}")
            axis.set_xlabel(f"Approximation ratio: {ratio:.3f}")
            axis.set_aspect("equal", adjustable="datalim")
            axis.legend()

        for axis in list(axes.flat)[len(comparisons):]:
            axis.set_visible(False)

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            self.output_dir / "tour_comparison.png",
            dpi=150,
        )

        plt.show()
        plt.close(fig)