from typing import Any
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

from experiments.hyperparameter_results import (
    HyperparameterResults,
)


def main() -> Any:
    """Execute ``main``.

    Returns:
        Result produced by this function.
    """
    project_dir = Path(__file__).resolve().parent
    error_file = project_dir / "results" / "tsp_small" / "Error.csv"

    if not error_file.exists():
        print(f"No experiment results found at:\n{error_file}")
        print("Run run_tsp_experiment.py first.")
        return

    results = HyperparameterResults(error_file)

    print("Best tested configuration:")
    for name, value in results.bestConfiguration().items():
        print(f"  {name}: {value}")

    print("\nConfigurations with mean approximation ratios near 1:")
    best_rows = results.bestRows()

    if best_rows.empty:
        print("  None found.")
    else:
        print(best_rows.to_string())

    # Save grouped CSV summaries and save/display comparison charts.
    results.plots()

    print(f"\nSummaries and charts saved to:\n{results.output_dir}")


if __name__ == "__main__":
    main()