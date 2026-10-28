import numpy as np

from graphs.complete_graph import CompleteGraph


class EuclideanGraph(CompleteGraph):
    def __init__(self, points):
        """
        Create a complete graph from points in Euclidean space.

        Each row represents a point. Edge weights are the Euclidean
        distances between points.
        """
        points = np.array(points, dtype=float, copy=True)

        if points.ndim != 2:
            raise ValueError("Points must be a two-dimensional array.")

        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError(
                "Provide at least one point with at least one coordinate."
            )

        if not np.all(np.isfinite(points)):
            raise ValueError("Coordinates must be finite numbers.")

        self.points = points

        dist_matrix = np.array([
            [
                np.linalg.norm(first_point - second_point)
                for second_point in self.points
            ]
            for first_point in self.points
        ])

        super().__init__(dist_matrix)

    def __repr__(self):
        return f"EuclideanGraph(points={self.points.tolist()!r})"