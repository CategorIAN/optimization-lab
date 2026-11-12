from typing import Any
"""Reproducible random-search tuning for the existing TSP Q-learning model.

Smoke run (small, intended as a reproducibility check):
    python experiments/tune_tsp_q_learning.py --smoke

The initial search samples alpha log-uniformly from 1e-4 to 1e-1 and eps
uniformly from 0.0 to 0.30.  The fixed defaults (p=1, T=1, n=1, beta=1,
gamma=1) match the small-model configuration used by the repository tests.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphs.euclidean_graph import EuclideanGraph
from models.theta import RandomThetaObject, ThetaObject
from traveling_salesman.held_karp import TSP_HK
from traveling_salesman.tsp_rl import TSP_RL


HYPERPARAMETER_NAMES = ("p", "T", "eps", "n", "alpha", "beta", "gamma")
FIXED_HYPERPARAMETERS = {"p": 1, "T": 1, "n": 1, "beta": 1, "gamma": 1}


def closed_tour(tour: Any, size: Any) -> Any:
    """Execute ``closed_tour``.

    Args:
        tour: Input accepted by this function.
        size: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    tour = [int(vertex) for vertex in tour]
    if len(tour) != size or set(tour) != set(range(size)):
        raise ValueError("Q-learning returned a non-Hamiltonian tour.")
    return tour + [tour[0]]


def theta_payload(theta: Any) -> Any:
    """Execute ``theta_payload``.

    Args:
        theta: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return [array.tolist() for array in theta.thetas]


def generate_graph_records(seed: Any, count: Any, city_count: Any) -> Any:
    """Generate distinct-coordinate Euclidean graphs and cache Held-Karp optima."""
    rng = random.Random(seed)
    solver = TSP_HK()
    grid = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    records = []
    for _ in range(count):
        points = rng.sample(grid, city_count)
        graph = EuclideanGraph(points)
        optimal_tour = solver.calculateWalk(graph)
        records.append({
            "points": [list(point) for point in points],
            "optimal_tour": [int(vertex) for vertex in optimal_tour],
            "optimal_length": float(graph.tourDistance(optimal_tour)),
        })
    return records


def save_json(path: Any, value: Any) -> Any:
    """Execute ``save_json``.

    Args:
        path: Input accepted by this function.
        value: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_graphs(records: Any) -> Any:
    """Execute ``load_graphs``.

    Args:
        records: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return [(EuclideanGraph(record["points"]), record) for record in records]


def ratio(graph: Any, tour: Any, optimal_length: Any) -> Any:
    """Execute ``ratio``.

    Args:
        graph: Input accepted by this function.
        tour: Input accepted by this function.
        optimal_length: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    distance = float(graph.tourDistance(tour))
    return None if optimal_length == 0 else distance / optimal_length


def evaluate(model: Any, theta: Any, graphs: Any, baseline_seed: Any) -> Any:
    """Evaluation is greedy only; it neither changes theta nor consumes model RNG."""
    baseline_rng = random.Random(baseline_seed)
    learned, random_baseline = [], []
    for graph, record in graphs:
        optimal_length = float(record["optimal_length"])
        learned_tour = closed_tour(model.calculateWalk(theta, graph, exploreOption=False), graph.n)
        learned.append(ratio(graph, learned_tour, optimal_length))
        random_tour = list(range(graph.n))
        baseline_rng.shuffle(random_tour)
        random_baseline.append(ratio(graph, graph.closeWalk(random_tour), optimal_length))
    valid = [value for value in learned if value is not None]
    baseline_valid = [value for value in random_baseline if value is not None]
    return {
        "mean_ratio": float(np.mean(valid)) if valid else None,
        "median_ratio": float(np.median(valid)) if valid else None,
        "worst_ratio": float(np.max(valid)) if valid else None,
        "random_mean_ratio": float(np.mean(baseline_valid)) if baseline_valid else None,
        "ratios": learned,
    }


def train_and_diagnose(hyperparameters: Any, training_graphs: Any, validation_graphs: Any, seed: Any, checkpoints: Any) -> Any:
    """Run existing episode updates, retaining theta and replay memory within one run."""
    random.seed(seed)
    np.random.seed(seed)
    model = TSP_RL(**hyperparameters)
    theta = RandomThetaObject(hyperparameters["p"], seed=seed)
    memory = []
    diagnostics = []
    previous_theta = theta_payload(theta)

    def record_checkpoint(episode: Any) -> Any:
        """Execute ``record_checkpoint``.

        Args:
            episode: Input accepted by this function.

        Returns:
            Result produced by this function.
        """
        score = evaluate(model, theta, validation_graphs, baseline_seed=seed + 10_000)
        flat_before = np.concatenate([np.asarray(value).ravel() for value in previous_theta])
        flat_after = np.concatenate([array.ravel() for array in theta.thetas])
        diagnostics.append({
            "episode": episode,
            "validation": score,
            "theta": theta_payload(theta),
            "theta_update_l2": float(np.linalg.norm(flat_after - flat_before)),
        })

    record_checkpoint(0)
    checkpoint_set = set(checkpoints)
    start = time.perf_counter()
    for episode, (graph, _) in enumerate(training_graphs, start=1):
        old_theta = theta_payload(theta)
        theta, memory, _ = model.episode(theta, memory, graph)
        previous_theta = old_theta
        if episode in checkpoint_set:
            record_checkpoint(episode)
    elapsed = time.perf_counter() - start
    return theta, diagnostics, elapsed


def sample_candidates(seed: Any, count: Any) -> Any:
    """Execute ``sample_candidates``.

    Args:
        seed: Input accepted by this function.
        count: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    rng = random.Random(seed)
    candidates = []
    for index in range(count):
        candidates.append({
            **FIXED_HYPERPARAMETERS,
            "alpha": 10 ** rng.uniform(-4, -1),
            "eps": rng.uniform(0.0, 0.30),
            "candidate": index,
        })
    return candidates


def run_candidate(candidate: Any, train_graphs: Any, validation_graphs: Any, seed: Any, checkpoints: Any) -> Any:
    """Execute ``run_candidate``.

    Args:
        candidate: Input accepted by this function.
        train_graphs: Input accepted by this function.
        validation_graphs: Input accepted by this function.
        seed: Input accepted by this function.
        checkpoints: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    hyperparameters = {name: candidate[name] for name in HYPERPARAMETER_NAMES}
    theta, diagnostics, training_seconds = train_and_diagnose(
        hyperparameters, train_graphs, validation_graphs, seed, checkpoints
    )
    final = diagnostics[-1]["validation"]
    return {
        **candidate,
        "seed": seed,
        "mean_ratio": final["mean_ratio"],
        "median_ratio": final["median_ratio"],
        "worst_ratio": final["worst_ratio"],
        "random_mean_ratio": final["random_mean_ratio"],
        "training_seconds": training_seconds,
        "diagnostics": diagnostics,
        "theta": theta_payload(theta),
    }


def plot_diagnostics(results: Any, output_path: Any) -> Any:
    """Execute ``plot_diagnostics``.

    Args:
        results: Input accepted by this function.
        output_path: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    for result in results:
        episodes = [item["episode"] for item in result["diagnostics"]]
        values = [item["validation"]["mean_ratio"] for item in result["diagnostics"]]
        axis.plot(episodes, values, marker="o", label=f"candidate {result['candidate']}")
    axis.set_xlabel("Training episodes")
    axis.set_ylabel("Validation mean approximation ratio")
    axis.set_title("Q-learning validation performance")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> Any:
    """Execute ``main``.

    Returns:
        Result produced by this function.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/tsp_tuning")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--train-graphs", type=int, default=12)
    parser.add_argument("--validation-graphs", type=int, default=6)
    parser.add_argument("--test-graphs", type=int, default=6)
    parser.add_argument("--cities", type=int, default=7, help="2-9 recommended; Held-Karp is exponential.")
    parser.add_argument("--repeat-seeds", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Use a small reproducibility smoke run.")
    args = parser.parse_args()
    if args.smoke:
        args.candidates, args.train_graphs, args.validation_graphs, args.test_graphs, args.cities, args.repeat_seeds, args.top_candidates = 2, 3, 2, 2, 4, 2, 1
    if not 2 <= args.cities <= 9:
        raise ValueError("Use 2-9 cities for this practical Held-Karp tuning script.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = vars(args) | {
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "search_ranges": {"alpha": "log-uniform [1e-4, 1e-1]", "eps": "uniform [0.0, 0.30]"},
        "evaluation": "Greedy policy only (exploreOption=False); no theta updates.",
    }
    save_json(output_dir / "configuration.json", configuration)
    datasets = {
        "training": generate_graph_records(args.seed + 1, args.train_graphs, args.cities),
        "validation": generate_graph_records(args.seed + 2, args.validation_graphs, args.cities),
        "test": generate_graph_records(args.seed + 3, args.test_graphs, args.cities),
    }
    for name, records in datasets.items():
        save_json(output_dir / f"{name}_graphs.json", records)
    train_graphs, validation_graphs, test_graphs = (load_graphs(datasets[name]) for name in ("training", "validation", "test"))
    checkpoints = sorted({0, max(1, len(train_graphs) // 2), len(train_graphs)})

    initial_results = [run_candidate(candidate, train_graphs, validation_graphs, args.seed + 100, checkpoints) for candidate in sample_candidates(args.seed, args.candidates)]
    ranked = sorted(initial_results, key=lambda result: result["mean_ratio"] if result["mean_ratio"] is not None else float("inf"))
    pd.DataFrame([{key: value for key, value in result.items() if key not in {"diagnostics", "theta"}} for result in ranked]).to_csv(output_dir / "ranked_results.csv", index=False)
    save_json(output_dir / "candidate_diagnostics.json", initial_results)
    plot_diagnostics(initial_results, output_dir / "validation_curve.png")

    repeated = []
    for result in ranked[:args.top_candidates]:
        candidate = {name: result[name] for name in (*HYPERPARAMETER_NAMES, "candidate")}
        repeated.extend(run_candidate(candidate, train_graphs, validation_graphs, args.seed + 1_000 + repeat, checkpoints) for repeat in range(args.repeat_seeds))
    repeat_frame = pd.DataFrame([{key: value for key, value in result.items() if key not in {"diagnostics", "theta"}} for result in repeated])
    repeat_frame.to_csv(output_dir / "repeat_results.csv", index=False)
    winner_id = repeat_frame.groupby("candidate")["mean_ratio"].mean().idxmin()
    winner_runs = [result for result in repeated if result["candidate"] == winner_id]
    winner = min(winner_runs, key=lambda result: result["mean_ratio"])
    winner_hyperparameters = {name: winner[name] for name in HYPERPARAMETER_NAMES}
    test_theta, test_diagnostics, test_seconds = train_and_diagnose(winner_hyperparameters, train_graphs, validation_graphs, args.seed + 9_000, checkpoints)
    test_score = evaluate(TSP_RL(**winner_hyperparameters), test_theta, test_graphs, baseline_seed=args.seed + 20_000)
    winner_checkpoint = {"hyperparameters": winner_hyperparameters, "theta": theta_payload(test_theta), "validation_seed": winner["seed"], "test_score": test_score, "training_seconds": test_seconds}
    save_json(output_dir / "winner_checkpoint.json", winner_checkpoint)
    save_json(output_dir / "test_result.json", {"score": test_score, "diagnostics": test_diagnostics})
    print(pd.read_csv(output_dir / "ranked_results.csv").to_string(index=False))
    print(f"Winner: candidate {winner_id}; held-out mean ratio: {test_score['mean_ratio']}")


if __name__ == "__main__":
    main()
