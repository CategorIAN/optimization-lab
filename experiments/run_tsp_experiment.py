import random
from pathlib import Path

import numpy as np

from experiments.hyperparameter_search import HyperparameterSearch
from graphs.euclidean_graph_distribution import (
    EuclideanGraphDistribution,
)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    output_dir = (
        Path(__file__).resolve().parent
        / "results"
        / "tsp_small"
    )

    hyperparameters = {
        "p": [2, 3],
        "T": [1],
        "eps": [0.05],
        "n": [2],
        "alpha": [0.001],
        "beta": [1],
        "gamma": [1],
    }

    distribution = EuclideanGraphDistribution(
        pt_number_lim=(4, 4),
        seed=42,
    )

    experiment = HyperparameterSearch(
        hyp_range_dict=hyperparameters,
        k=2,
        graphDist=distribution,
        n=8,
        output_dir=output_dir,
    )

    # Each call evaluates one configuration.
    for _ in experiment.hyp_dicts:
        results = experiment.getErrorDf()

    print("\nResults:")
    print(results.to_string())
    print(f"\nFiles saved in: {output_dir}")