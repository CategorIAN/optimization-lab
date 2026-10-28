import numpy as np


class ThetaObject:
    names = (
        "theta_1",
        "theta_2",
        "theta_3",
        "theta_4",
        "theta_5a",
        "theta_5b",
        "theta_6",
        "theta_7",
    )

    def __init__(self, thetas):
        """Store and validate the eight structure2vec parameter arrays."""
        if len(thetas) != len(self.names):
            raise ValueError("Exactly eight parameter arrays are required.")

        arrays = [
            np.array(theta, dtype=float, copy=True)
            for theta in thetas
        ]

        if arrays[0].ndim != 2 or arrays[0].shape[1] != 1:
            raise ValueError("theta_1 must be a column vector.")

        self.p = arrays[0].shape[0]

        if self.p < 1:
            raise ValueError("The embedding dimension must be positive.")

        expected_shapes = (
            (self.p, 1),
            (self.p, self.p),
            (self.p, self.p),
            (self.p, 1),
            (self.p, 1),
            (self.p, 1),
            (self.p, self.p),
            (self.p, self.p),
        )

        for name, array, shape in zip(
            self.names, arrays, expected_shapes
        ):
            if array.shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}; "
                    f"received {array.shape}."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite values.")

        self.thetas = arrays

        for name, array in zip(self.names, self.thetas):
            setattr(self, name, array)

    def __str__(self):
        return "\n".join(
            f"{'-' * 20}\n{name}\n{array}"
            for name, array in zip(self.names, self.thetas)
        )

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomThetaObject(ThetaObject):
    def __init__(self, p, seed=None):
        """
        Initialize parameters with uniform random values in [0, 1).

        A seed makes initialization reproducible.
        """
        if type(p) is not int or p < 1:
            raise ValueError("p must be a positive integer.")

        rng = np.random.default_rng(seed)

        shapes = (
            (p, 1),
            (p, p),
            (p, p),
            (p, 1),
            (p, 1),
            (p, 1),
            (p, p),
            (p, p),
        )

        super().__init__([
            rng.random(shape)
            for shape in shapes
        ])