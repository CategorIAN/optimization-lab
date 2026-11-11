"""TSP comparison using the repository's Euclidean, RL, and Held-Karp code."""

from graphs.euclidean_graph import EuclideanGraph
from models.theta import ThetaObject
from traveling_salesman.held_karp import TSP_HK
from traveling_salesman.tsp_rl import TSP_RL


RL_SETTINGS = {
    "p": 4,
    "T": 2,
    "eps": 0.01,
    "n": 3,
    "alpha": 0.1,
    "beta": 5,
    "gamma": 0.9,
}


def closed_valid_tour(tour, city_count):
    """Validate a solver's permutation and return a closed JSON-safe tour."""
    open_tour = [int(city) for city in tour]
    if len(open_tour) != city_count or set(open_tour) != set(range(city_count)):
        raise ValueError("A solver returned an invalid tour.")
    return open_tour + [open_tour[0]]


def solve_tsp(points, checkpoint=None):
    """Run both existing solvers against one EuclideanGraph and compare lengths."""
    graph = EuclideanGraph(points)
    if checkpoint is None:
        hyperparameters = RL_SETTINGS.copy()
        model = TSP_RL(**hyperparameters)
        _, final_theta = model.QLearning([graph])
        q_learning_tour = model.calculateWalk(
            final_theta,
            graph,
            exploreOption=False,
        )
        training_description = "Greedy tour generated after training on this card's graph (exploration disabled for display)."
        evaluation_note = None
    else:
        hyperparameters = checkpoint.get("hyperparameters") if isinstance(checkpoint, dict) else None
        theta_values = checkpoint.get("theta") if isinstance(checkpoint, dict) else None
        if not isinstance(hyperparameters, dict) or set(hyperparameters) != set(RL_SETTINGS):
            raise ValueError("Checkpoint must contain the seven TSP hyperparameters.")
        model = TSP_RL(**hyperparameters)
        theta = ThetaObject(theta_values)
        if theta.p != hyperparameters["p"]:
            raise ValueError("Checkpoint theta dimension does not match p.")
        q_learning_tour = model.calculateWalk(theta, graph, exploreOption=False)
        training_description = "Loaded checkpoint: greedy evaluation only (exploration and theta updates disabled)."
        evaluation_note = "Saved eps is shown below; exploration was disabled during this evaluation."
    optimal_tour = TSP_HK().calculateWalk(graph)

    q_learning_closed = closed_valid_tour(q_learning_tour, graph.n)
    optimal_closed = closed_valid_tour(optimal_tour, graph.n)
    q_learning_length = float(graph.walkDistance(q_learning_closed))
    optimal_length = float(graph.walkDistance(optimal_closed))

    return {
        "q_learning_tour": q_learning_closed,
        "optimal_tour": optimal_closed,
        "q_learning_length": q_learning_length,
        "optimal_length": optimal_length,
        "approximation_ratio": (
            q_learning_length / optimal_length if optimal_length else None
        ),
        "training": training_description,
        "hyperparameters": hyperparameters,
        "evaluation_note": evaluation_note,
    }
