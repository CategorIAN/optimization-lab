from itertools import product
import numpy as np
from functools import reduce
import random

class CompleteGraph:
    def __init__(self, dist_matrix):
        """
        Create a weighted complete graph.

        dist_matrix[i, j] is the weight of the edge from vertex i
        to vertex j. Diagonal values must be zero.
        """
        dist_matrix = np.asarray(dist_matrix)

        if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
            raise ValueError("The distance matrix must be square.")

        if dist_matrix.shape[0] == 0:
            raise ValueError("The distance matrix must contain at least one vertex.")

        if not np.all(np.diag(dist_matrix) == 0):
            raise ValueError("The diagonal of the distance matrix must be zero.")

        self.n = dist_matrix.shape[0]
        self.vertices = list(range(self.n))

        # Store both directions: (0, 1) and (1, 0).
        self.edges = [
            (i, j)
            for i in self.vertices
            for j in self.vertices
            if i != j
        ]

        self.w = dist_matrix.copy()
        self.W = self.w.flatten()

        self.neighbors = {
            vertex: set(self.vertices) - {vertex}
            for vertex in self.vertices
        }

        # Matrices used by the original structure2vec implementation.
        self.N = self.neighbor_matrix()
        self.U = self.getU()

    def closeWalk(self, walk):
        """Return the walk with its starting vertex appended if needed."""
        if not walk or walk[-1] == walk[0]:
            return walk

        return walk + [walk[0]]

    def randomHamCycle(self):
        """Return a random Hamiltonian cycle."""
        walk = random.sample(self.vertices, k=self.n)
        return self.closeWalk(walk)

    def walkDistance(self, walk):
        """Return the total weight of the edges in a walk."""
        return sum(
            self.w[current, next_vertex]
            for current, next_vertex in zip(walk, walk[1:])
        )

    def tourDistance(self, tour):
        """Return the distance of a tour, including the return to its start."""
        return self.walkDistance(self.closeWalk(tour))

    def neighbor_vec(self, vertex):
        """Return a column vector indicating the neighbors of a vertex."""
        return np.array([
            [int(other in self.neighbors[vertex])]
            for other in self.vertices
        ])

    def neighbor_matrix(self):
        """Return a matrix whose columns are neighbor vectors."""
        return np.concatenate(
            [self.neighbor_vec(vertex) for vertex in self.vertices],
            axis=1,
        )

    def unit_vec(self, vertex):
        """Return a column unit vector for a vertex."""
        return np.array([
            [int(other == vertex)]
            for other in self.vertices
        ])

    def neighbor_square(self, vertex):
        """Return a matrix with the vertex's neighbor vector in its column."""
        return np.outer(
            self.neighbor_vec(vertex),
            self.unit_vec(vertex),
        )

    def getU(self):
        """Stack the neighbor-square matrices used by structure2vec."""
        return np.concatenate(
            [self.neighbor_square(vertex) for vertex in self.vertices],
            axis=0,
        )
