import random
from greedy.set_cover import greedy_set_cover
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def generate_random_collection(n_elements, n_sets, probability=0.5, seed=None):
    rng = random.Random(seed)

    return {
        k: {
            i for i in range(n_elements)
            if rng.random() < probability
        }
        for k in range(n_sets)
    }


def run_experiment(n_elements, n_sets, probability=0.5, seed=None):
    collection = generate_random_collection(
        n_elements,
        n_sets,
        probability,
        seed
    )

    selected = greedy_set_cover(collection)

    universe = set().union(*collection.values())
    covered = set().union(
        *(collection[k] for k in selected)
    )

    return {
        "n_elements": n_elements,
        "n_sets": n_sets,
        "probability": probability,
        "universe_size": len(universe),
        "selected_count": len(selected),
        "coverage": len(covered) / len(universe) if universe else 1.0,
    }


def run_experiments():
    results = [
        run_experiment(
            n_elements=n,
            n_sets=2 * n,
            probability=0.3,
            seed=42
        )
        for n in range(5, 101, 5)
    ]

    for result in results:
        print(result)

    return results


def plot_experiments(results):
    x = [r["n_elements"] for r in results]
    y = [r["selected_count"] for r in results]

    plt.plot(x, y, marker="o")
    plt.xlabel("Number of possible elements")
    plt.ylabel("Number of sets selected")
    plt.title("Greedy Set Cover")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    results = run_experiments()
    plot_experiments(results)