
from time import perf_counter
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from graphs.maximum_flow import ford_fulkerson
from graphs.random_flow_network import random_flow_network
from pathlib import Path


def analysis(first, last, step, graph=False, seed=42, trials=100):
    records = []

    for n in range(first, last + 1, step):
        results = {
            "DFS_Count": [],
            "DFS_Time": [],
            "BFS_Count": [],
            "BFS_Time": [],
        }

        for trial in range(trials):
            # Both algorithms use the same random network.
            capacity = random_flow_network(
                n,
                seed=seed + n * trials + trial,
            )

            values = {}

            for strategy in ("dfs", "bfs"):
                start = perf_counter()

                flow, value, count = ford_fulkerson(
                    capacity,
                    source=0,
                    sink=n - 1,
                    strategy=strategy,
                    count=True,
                )

                elapsed = perf_counter() - start
                prefix = strategy.upper()

                results[f"{prefix}_Count"].append(count)
                results[f"{prefix}_Time"].append(elapsed)
                values[strategy] = value

            assert values["dfs"] == values["bfs"]

        record = {"Size": n}

        for column, measurements in results.items():
            record[column] = sum(measurements) / trials

        records.append(record)

        print(
            f"Size {n}: "
            f"DFS = {record['DFS_Time']:.6f}s, "
            f"BFS = {record['BFS_Time']:.6f}s"
        )

    df = pd.DataFrame(records).set_index("Size")

    output_dir = Path(__file__).resolve().parent / "maximum_flow_analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"Analysis_from_{first}_to_{last}.csv"
    df.to_csv(filename)

    if graph:
        create_chart(df=df)

    return df


def create_chart(df=None, file=None):
    if df is None:
        df = pd.read_csv(file, index_col=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    axes[0].plot(
        df.index, df["DFS_Count"],
        marker="o", label="DFS",
    )
    axes[0].plot(
        df.index, df["BFS_Count"],
        marker="o", label="BFS",
    )
    axes[0].set_title("Number of Augmentations vs Network Size")
    axes[0].set_xlabel("Vertices")
    axes[0].set_ylabel("Augmentations")
    axes[0].legend()

    axes[1].plot(
        df.index, df["DFS_Time"],
        marker="o", label="DFS",
    )
    axes[1].plot(
        df.index, df["BFS_Time"],
        marker="o", label="BFS",
    )
    axes[1].set_title("Execution Time vs Network Size")
    axes[1].set_xlabel("Vertices")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = analysis(
        first=5,
        last=50,
        step=5,
        trials=100,
        graph=True,
    )

    print(df)

