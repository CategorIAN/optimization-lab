from typing import Any
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class HyperparameterResults:
    """Summarize and plot saved TSP hyperparameter-search results."""

    def __init__(self, error_file: Any, output_dir: Any=None) -> Any:
        """
        error_file: CSV produced by HyperparameterSearch.
        output_dir: Directory for summaries and charts.
        """
        self.error_file = Path(error_file)

        self.output_dir = (
            self.error_file.parent / "summaries"
            if output_dir is None
            else Path(output_dir)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hyp_names = [
            "p",
            "T",
            "eps",
            "n",
            "alpha",
            "beta",
            "gamma",
        ]

    def groupedErrorDf(self) -> Any:
        """Return a function giving mean approximation ratio by parameter."""
        results = pd.read_csv(self.error_file, index_col=0)

        return lambda name: (
            results.groupby(name)["Error"].mean()
        )

    def groupedErrorDfs(self) -> Any:
        """Save and return a summary for each hyperparameter."""
        grouped_error = self.groupedErrorDf()

        def append_summary(summaries: Any, name: Any) -> Any:
            """Execute ``append_summary``.

            Args:
                summaries: Input accepted by this function.
                name: Input accepted by this function.

            Returns:
                Result produced by this function.
            """
            summary = grouped_error(name)

            summary.to_csv(
                self.output_dir / f"Error_By_{name}.csv"
            )

            return summaries | {name: summary}

        return reduce(
            append_summary,
            self.hyp_names,
            {},
        )

    def plots(self) -> Any:
        """Save and display bar charts of grouped approximation ratios."""
        summaries = self.groupedErrorDfs()

        def plot_group(names: Any, group_number: Any) -> Any:
            """Execute ``plot_group``.

            Args:
                names: Input accepted by this function.
                group_number: Input accepted by this function.

            Returns:
                Result produced by this function.
            """
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(10, 7),
                squeeze=False,
            )

            fig.suptitle(
                "Mean Approximation Ratio by Hyperparameter"
            )

            for axis, name in zip(axes.flat, names):
                summary = summaries[name]

                axis.bar(
                    [str(value) for value in summary.index],
                    summary.to_numpy(),
                )
                axis.set_xlabel(name)
                axis.set_ylabel("Mean approximation ratio")
                axis.axhline(
                    1,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )

            # The second group has three charts, so hide its unused axis.
            for axis in list(axes.flat)[len(names):]:
                axis.set_visible(False)

            fig.tight_layout(rect=(0, 0, 1, 0.94))

            fig.savefig(
                self.output_dir / f"Hyperparameters_{group_number}.png",
                dpi=150,
            )

            plt.show()
            plt.close(fig)

        plot_group(self.hyp_names[:4], 1)
        plot_group(self.hyp_names[4:], 2)

    def bestParams(self) -> Any:
        """
        Return the best value of each parameter by its grouped mean.

        This combination is not necessarily the best tested configuration.
        """
        grouped_error = self.groupedErrorDf()

        return {
            name: grouped_error(name).idxmin()
            for name in self.hyp_names
        }

    def bestConfiguration(self) -> Any:
        """Return the tested configuration with the lowest mean ratio."""
        results = pd.read_csv(self.error_file, index_col=0)

        if results.empty:
            raise ValueError("The results file contains no configurations.")

        best_index = results["Error"].idxmin()

        return results.loc[best_index].to_dict()

    def bestRows(self) -> Any:
        """Save and return configurations with ratios approximately 1."""
        results = pd.read_csv(self.error_file, index_col=0)

        optimal_rows = results.loc[
            np.isclose(
                results["Error"],
                1.0,
                rtol=0,
                atol=1e-9,
            )
        ]

        optimal_rows.to_csv(
            self.output_dir / "Error_Best.csv"
        )

        return optimal_rows