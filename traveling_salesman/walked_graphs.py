import json
from dataclasses import dataclass

from graphs.euclidean_graph import EuclideanGraph
from traveling_salesman.held_karp import TSP_HK


@dataclass
class SolvedGraph:
    """A graph paired with an optimal tour and its distance."""

    graph: EuclideanGraph
    shortest_cycle: list[int]
    distance: float


class WalkedGraphs:
    def __init__(self, file, graphDist=None, n=None):
        """
        Load solved graphs from JSON, or generate and save them.

        graphDist and n are required only when the file does not exist.
        """
        self.graphs = self.getGraphs(file, graphDist, n)

    def getGraphsFromJSON(self, file):
        """Read graphs and their optimal solutions from a JSON file."""
        with open(file, "r", encoding="utf-8") as json_file:
            records = json.load(json_file)

        def read_graph(record):
            return SolvedGraph(
                graph=EuclideanGraph(record["points"]),
                shortest_cycle=list(record["optimal_tour"]),
                distance=float(record["optimal_distance"]),
            )

        return [read_graph(record) for record in records]

    def createGraphs(self, graphDist, n):
        """Generate graphs and solve each one with Held–Karp."""
        solver = TSP_HK()

        def create_graph():
            graph = graphDist.randomGraph()
            tour = solver.calculateWalk(graph)

            return SolvedGraph(
                graph=graph,
                shortest_cycle=tour,
                distance=graph.tourDistance(tour),
            )

        return [create_graph() for _ in range(n)]

    def toJSON(self, graphs, file):
        """Save coordinates in original vertex order and store tours separately."""

        def graph_record(solved_graph):
            return {
                "points": solved_graph.graph.points.tolist(),
                "optimal_tour": [
                    int(vertex)
                    for vertex in solved_graph.shortest_cycle
                ],
                "optimal_distance": float(solved_graph.distance),
            }

        records = [
            graph_record(solved_graph)
            for solved_graph in graphs
        ]

        with open(file, "w", encoding="utf-8") as json_file:
            json.dump(
                records,
                json_file,
                indent=2,
                allow_nan=False,
            )

        return records

    def getGraphs(self, file, graphDist, n):
        """Load an existing dataset, or create it if the file is absent."""
        try:
            return self.getGraphsFromJSON(file)
        except FileNotFoundError:
            if graphDist is None or n is None:
                raise ValueError(
                    "graphDist and n are required to create a dataset."
                )

            if type(n) is not int or n < 1:
                raise ValueError("n must be a positive integer.")

            graphs = self.createGraphs(graphDist, n)
            self.toJSON(graphs, file)

            return graphs