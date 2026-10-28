import random

from graphs.euclidean_graph import EuclideanGraph


class EuclideanGraphDistribution:
    def __init__(
        self,
        pt_number_lim=(9, 9),
        xlim=(-5, 5),
        ylim=(-5, 5),
        seed=None,
    ):
        """
        Generate graphs with integer coordinates.

        All limits are inclusive. A seed makes the sequence of
        generated graphs reproducible.
        """
        self._validate_limits("pt_number_lim", pt_number_lim)
        self._validate_limits("xlim", xlim)
        self._validate_limits("ylim", ylim)

        if pt_number_lim[0] < 1:
            raise ValueError("Graphs must contain at least one point.")

        self.pt_number_lim = tuple(pt_number_lim)
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)

        self._rng = random.Random(seed)

    @staticmethod
    def _validate_limits(name, limits):
        if len(limits) != 2:
            raise ValueError(f"{name} must contain two limits.")

        if any(type(value) is not int for value in limits):
            raise ValueError(f"{name} limits must be integers.")

        if limits[0] > limits[1]:
            raise ValueError(
                f"{name} minimum must not exceed its maximum."
            )

    def randomGraph(self):
        """Return a random graph drawn from this distribution."""
        n = self._rng.randint(*self.pt_number_lim)

        points = [
            (
                self._rng.randint(*self.xlim),
                self._rng.randint(*self.ylim),
            )
            for _ in range(n)
        ]

        return EuclideanGraph(points)