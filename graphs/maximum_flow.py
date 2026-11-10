
from collections import deque
from functools import reduce
from math import inf

from tail_recursive import tail_recursive


def residual_network(capacity, flow):
    """
    Construct a residual network.

    capacity: {(u, v): capacity}
    flow: {(u, v): current_flow}

    Returns:
        {(u, v): residual_capacity}
    """
    def add_edge(residual, edge):
        u, v = edge
        forward = capacity[edge] - flow[edge]
        backward = flow[edge]

        if forward > 0:
            residual[(u, v)] = residual.get((u, v), 0) + forward

        if backward > 0:
            residual[(v, u)] = residual.get((v, u), 0) + backward

        return residual

    return reduce(add_edge, capacity, {})


def augmenting_path(residual, source, sink, strategy="bfs"):
    """
    Find an augmenting path using BFS or DFS.

    Returns:
        A list of vertices from source to sink,
        or None if no path exists.
    """
    if source == sink:
        raise ValueError("Source and sink must differ.")

    if strategy not in {"bfs", "dfs"}:
        raise ValueError("Strategy must be 'bfs' or 'dfs'.")

    neighbors = {}

    for (u, v), remaining in residual.items():
        if remaining > 0:
            neighbors.setdefault(u, []).append(v)

    @tail_recursive
    def search(frontier, visited, parent):
        if not frontier:
            return None

        current, *remaining = frontier

        if current == sink:
            @tail_recursive
            def recover(node, path):
                if node == source:
                    return [source] + path

                return recover.tail_call(
                    parent[node],
                    [node] + path,
                )

            return recover(sink, [])

        new = [
            node
            for node in neighbors.get(current, [])
            if node not in visited
        ]

        # Avoid discovering the same vertex twice.
        new = list(dict.fromkeys(new))

        new_parent = parent | {
            node: current for node in new
        }

        new_visited = visited | set(new)

        new_frontier = (
            remaining + new
            if strategy == "bfs"
            else new + remaining
        )

        return search.tail_call(
            new_frontier,
            new_visited,
            new_parent,
        )

    return search([source], {source}, {})


def augment_flow(capacity, flow, path):
    """
    Augment flow along a residual path.

    Handles forward and backward residual capacity,
    including networks containing antiparallel edges.

    Returns a new flow dictionary.
    """
    if path is None:
        return flow.copy()

    edges = list(zip(path, path[1:]))

    residual = residual_network(capacity, flow)

    bottleneck = min(
        residual[edge] for edge in edges
    )

    def update(current_flow, edge):
        u, v = edge
        remaining = bottleneck

        # First use available capacity on the forward edge.
        if edge in capacity:
            available = capacity[edge] - current_flow[edge]
            increase = min(remaining, available)

            current_flow[edge] += increase
            remaining -= increase

        # Then cancel flow on the reverse edge.
        reverse = (v, u)

        if remaining > 0 and reverse in capacity:
            current_flow[reverse] -= remaining

        return current_flow

    return reduce(update, edges, flow.copy())


def ford_fulkerson(capacity, source, sink, strategy="bfs", count=False):
    """
    Compute maximum flow using Ford-Fulkerson.

    strategy="bfs": Edmonds-Karp
    strategy="dfs": Ford-Fulkerson with DFS

    Returns:
        (flow_dictionary, maximum_flow_value)
    """
    if source == sink:
        raise ValueError("Source and sink must differ.")

    if strategy not in {"bfs", "dfs"}:
        raise ValueError("Strategy must be 'bfs' or 'dfs'.")

    if any(u == v or c < 0 for (u, v), c in capacity.items()):
        raise ValueError(
            "Capacities must be nonnegative and self-loops are unsupported."
        )

    initial_flow = {
        edge: 0 for edge in capacity
    }

    @tail_recursive
    def go(flow, iterations):
        residual = residual_network(capacity, flow)
        path = augmenting_path(
            residual, source, sink, strategy
        )

        if path is None:
            value = (
                    sum(f for (u, v), f in flow.items() if u == source)
                    - sum(f for (u, v), f in flow.items() if v == source)
            )

            return (
                (flow, value, iterations)
                if count
                else (flow, value)
            )

        return go.tail_call(
            augment_flow(capacity, flow, path),
            iterations + 1,
        )

    return go(initial_flow, 0)
