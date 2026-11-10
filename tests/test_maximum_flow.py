
import pytest

from graphs.maximum_flow import (
    residual_network,
    augmenting_path,
    augment_flow,
    ford_fulkerson,
)


def test_residual_network():
    capacity = {(0, 1): 10}
    flow = {(0, 1): 4}

    assert residual_network(capacity, flow) == {
        (0, 1): 6,
        (1, 0): 4,
    }


def test_residual_network_antiparallel():
    capacity = {(0, 1): 10, (1, 0): 7}
    flow = {(0, 1): 4, (1, 0): 3}

    assert residual_network(capacity, flow) == {
        (0, 1): 9,
        (1, 0): 8,
    }


@pytest.mark.parametrize("strategy", ["bfs", "dfs"])
def test_augmenting_path(strategy):
    residual = {
        (0, 1): 10,
        (0, 2): 5,
        (1, 3): 8,
        (2, 3): 4,
    }

    path = augmenting_path(residual, 0, 3, strategy)

    assert path in ([0, 1, 3], [0, 2, 3])


def test_augment_flow():
    capacity = {(0, 1): 10, (1, 2): 5}
    flow = {(0, 1): 0, (1, 2): 0}

    result = augment_flow(capacity, flow, [0, 1, 2])

    assert result == {(0, 1): 5, (1, 2): 5}


def test_augment_flow_backward():
    capacity = {(0, 1): 10}
    flow = {(0, 1): 4}

    result = augment_flow(capacity, flow, [1, 0])

    assert result == {(0, 1): 0}


@pytest.mark.parametrize("strategy", ["bfs", "dfs"])
def test_ford_fulkerson(strategy):
    capacity = {
        (0, 1): 10,
        (0, 2): 5,
        (1, 3): 8,
        (2, 3): 4,
    }

    flow, value = ford_fulkerson(
        capacity, 0, 3, strategy
    )

    assert value == 12
    assert all(0 <= flow[e] <= capacity[e] for e in capacity)


def test_no_path():
    residual = {(0, 1): 5}

    assert augmenting_path(residual, 0, 2) is None
