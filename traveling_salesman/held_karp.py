from itertools import combinations
from functools import reduce


class TSP_HK:
    """
    Solve the traveling salesman problem using Held–Karp.

    Vertex 0 is the fixed start and final destination.

    cost_table[(vertices_to_visit, current_vertex)] stores the
    minimum distance needed to:
      1. Start at current_vertex.
      2. Visit all other vertices in vertices_to_visit.
      3. Return to vertex 0.

    vertices_to_visit includes current_vertex.
    """

    def policy(self, graph, cost_table):
        """Return a function selecting the best next vertex and cost."""

        def best_choice(vertices_to_visit, current_vertex):
            remaining_vertices = vertices_to_visit - {current_vertex}

            # Each choice combines one edge with a solved subproblem.
            candidate_costs = [
                (
                    next_vertex,
                    graph.w[current_vertex, next_vertex]
                    + cost_table[(remaining_vertices, next_vertex)],
                )
                for next_vertex in remaining_vertices
            ]

            def choose_cheaper(best, candidate):
                best_vertex, best_cost = best
                _, candidate_cost = candidate

                # Accept the first candidate, then keep cheaper choices.
                if best_vertex is None or candidate_cost < best_cost:
                    return candidate

                return best

            # If no vertices remain, the only step is returning home.
            return_to_start = (None, graph.w[current_vertex, 0])

            return reduce(
                choose_cheaper,
                candidate_costs,
                return_to_start,
            )

        return best_choice

    def vBest(self, graph, cost_table):
        """Return a function selecting only the best next vertex."""
        best_choice = self.policy(graph, cost_table)

        return lambda vertices_to_visit, current_vertex: best_choice(
            vertices_to_visit, current_vertex
        )[0]

    def QBest(self, graph, cost_table):
        """Return a function calculating only the minimum remaining cost."""
        best_choice = self.policy(graph, cost_table)

        return lambda vertices_to_visit, current_vertex: best_choice(
            vertices_to_visit, current_vertex
        )[1]

    def getQfromP(self, graph, cost_table):
        """Return a function calculating all cost entries for one subset."""
        minimum_cost = self.QBest(graph, cost_table)

        return lambda vertices_to_visit: {
            (vertices_to_visit, current_vertex): minimum_cost(
                vertices_to_visit, current_vertex
            )
            for current_vertex in vertices_to_visit
        }

    def createQdict(self, graph):
        """Build the cost table in increasing order of subset size."""
        non_start_vertices = [
            vertex
            for vertex in graph.vertices
            if vertex != 0
        ]

        def build_level(cost_table, subset_size):
            if subset_size == graph.n:
                return cost_table

            subsets = [
                frozenset(subset)
                for subset in combinations(
                    non_start_vertices, subset_size
                )
            ]

            # A subset of size k only needs costs from size k - 1.
            costs_for_subset = self.getQfromP(graph, cost_table)

            updated_cost_table = reduce(
                lambda accumulated, subset: (
                    accumulated | costs_for_subset(subset)
                ),
                subsets,
                cost_table,
            )

            return build_level(
                updated_cost_table,
                subset_size + 1,
            )

        return build_level({}, 1)

    def S_not(self, graph, tour):
        """Return the vertices not yet included in the tour."""
        return set(graph.vertices).difference(tour)

    def calculateWalkfromQ(self, graph, cost_table):
        """Reconstruct an optimal tour from the completed cost table."""

        def extend_tour(tour):
            if len(tour) == graph.n:
                return tour

            unvisited_vertices = self.S_not(graph, tour)
            current_vertex = tour[-1]

            # Include the current vertex to match the table's convention.
            vertices_to_visit = frozenset(
                unvisited_vertices
            ).union({current_vertex})

            next_vertex = self.vBest(graph, cost_table)(
                vertices_to_visit,
                current_vertex,
            )

            return extend_tour(tour + [next_vertex])

        return extend_tour([0])

    def calculateWalk(self, graph):
        """
        Return an optimal tour starting at vertex 0.

        Vertex 0 is not repeated at the end, but the optimization
        includes the cost of returning to it.
        """
        cost_table = self.createQdict(graph)

        return self.calculateWalkfromQ(graph, cost_table)