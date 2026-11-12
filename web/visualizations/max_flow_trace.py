from typing import Any
"""Serializable Ford-Fulkerson traces built from the project's flow helpers."""

from graphs.maximum_flow import (
    augment_flow,
    augmenting_path,
    ford_fulkerson,
    residual_network,
)


def edge_records(edges: Any) -> Any:
    """Convert a tuple-keyed edge mapping into JSON-safe records."""
    return [
        {"source": source, "target": target, "value": value}
        for (source, target), value in edges.items()
    ]


def flow_value(flow: Any, source: Any) -> Any:
    """Execute ``flow_value``.

    Args:
        flow: Input accepted by this function.
        source: Input accepted by this function.

    Returns:
        Result produced by this function.
    """
    return (
        sum(value for (start, _), value in flow.items() if start == source)
        - sum(value for (_, end), value in flow.items() if end == source)
    )


def augmentation_operations(capacity: Any, flow: Any, path: Any, bottleneck: Any) -> Any:
    """Describe augment_flow's forward-first, reverse-cancellation updates."""
    operations = []
    for source, target in zip(path, path[1:]):
        remaining = bottleneck
        edge = (source, target)
        if edge in capacity:
            increase = min(remaining, capacity[edge] - flow[edge])
            if increase:
                operations.append({
                    "type": "increase",
                    "source": source,
                    "target": target,
                    "value": increase,
                })
                remaining -= increase
        reverse = (target, source)
        if remaining and reverse in capacity:
            operations.append({
                "type": "cancel",
                "source": target,
                "target": source,
                "value": remaining,
            })
    return operations


def maximum_flow_trace(capacity: Any, source: Any, sink: Any, strategy: Any="bfs") -> Any:
    """Return a step-by-step trace while delegating all flow work to existing code."""
    flow = {edge: 0 for edge in capacity}
    initial = {
        "flow": edge_records(flow),
        "residual": edge_records(residual_network(capacity, flow)),
        "flow_value": 0,
    }
    steps = []

    while True:
        residual_before = residual_network(capacity, flow)
        path = augmenting_path(residual_before, source, sink, strategy)
        if path is None:
            break
        bottleneck = min(residual_before[edge] for edge in zip(path, path[1:]))
        flow_before = flow.copy()
        operations = augmentation_operations(capacity, flow_before, path, bottleneck)
        flow = augment_flow(capacity, flow_before, path)
        steps.append({
            "path": path,
            "bottleneck": bottleneck,
            "flow_before": edge_records(flow_before),
            "flow_after": edge_records(flow),
            "residual_before": edge_records(residual_before),
            "residual_after": edge_records(residual_network(capacity, flow)),
            "flow_value_before": flow_value(flow_before, source),
            "flow_value_after": flow_value(flow, source),
            "operations": operations,
        })

    final_flow, final_value = ford_fulkerson(capacity, source, sink, strategy)
    if flow != final_flow or flow_value(flow, source) != final_value:
        raise AssertionError("Trace diverged from ford_fulkerson().")

    return {
        "initial": initial,
        "steps": steps,
        "maximum_flow": final_value,
        "final_flow": edge_records(final_flow),
        "final_residual": edge_records(residual_network(capacity, final_flow)),
    }
