import unittest

import numpy as np

from models.theta import RandomThetaObject, ThetaObject


class TestThetaObject(unittest.TestCase):
    def test_parameter_shapes(self):
        theta = RandomThetaObject(p=3, seed=42)

        for name in ("theta_1", "theta_4", "theta_5a", "theta_5b"):
            self.assertEqual(getattr(theta, name).shape, (3, 1))

        for name in ("theta_2", "theta_3", "theta_6", "theta_7"):
            self.assertEqual(getattr(theta, name).shape, (3, 3))

    def test_same_seed_reproduces_parameters(self):
        first = RandomThetaObject(p=3, seed=42)
        second = RandomThetaObject(p=3, seed=42)

        for first_array, second_array in zip(
            first.thetas, second.thetas
        ):
            np.testing.assert_array_equal(first_array, second_array)

    def test_rejects_incorrect_parameter_shape(self):
        arrays = RandomThetaObject(p=3, seed=42).thetas.copy()
        arrays[1] = np.zeros((2, 2))

        with self.assertRaises(ValueError):
            ThetaObject(arrays)

    def test_rejects_nonpositive_dimension(self):
        with self.assertRaises(ValueError):
            RandomThetaObject(p=0)


if __name__ == "__main__":
    unittest.main()