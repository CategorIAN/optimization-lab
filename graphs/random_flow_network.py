import random
import pandas as pd


def random_flow_network(n, seed=None):
    if n < 2:
        raise ValueError("n must be at least 2.")

    rng = random.Random(seed)

    vertices = range(n)
    possible_edges = [
        (u, v)
        for u in vertices
        for v in vertices
        if u != v
    ]

    edge_count = rng.randint(
        max(1, n // 2),
        len(possible_edges),
    )

    edges = rng.sample(possible_edges, edge_count)

    return {
        edge: rng.randint(0, 5)
        for edge in edges
    }


def save_flow_network(capacity, file):
    df = pd.DataFrame(
        [
            {"Tail": u, "Head": v, "Capacity": c}
            for (u, v), c in capacity.items()
        ]
    )
    df.to_csv(file, index=False)


def load_flow_network(file):
    df = pd.read_csv(file)

    return {
        (int(row.Tail), int(row.Head)): row.Capacity
        for row in df.itertuples(index=False)
    }
