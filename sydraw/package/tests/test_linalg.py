from unittest import TestCase

import numpy as np

from sydraw import linalg, synth

circle_params = (3.0, (1.0, 1.0))
circle_points = synth.circles(
    ns=20, radius=3.0, center=(1.0, 1.0), n=10, noise_perc=0.0, homogeneous=True
)
circle_points_fixed = np.array(
    [[1, 4, 1], [4, 1, 1], [1, -2, 1], [-2, 1, 1]], dtype=float
)  # c = (1,1); r = 3


class Test(TestCase):
    def setUp(self) -> None:
        pass

    def test_conic_monomials(self):
        output = linalg.conic_monomials(circle_points_fixed)

        target = np.array(
            [
                [1, 16, 4, 1, 4, 1],
                [9, 1, 3, 3, 1, 1],
                [1, 4, -2, 1, -2, 1],
                [4, 1, -2, -2, 1, 1],
            ],
            dtype=float,
        )
        return

    def test_circle_monomials(self):
        output = linalg.circle_monomials(circle_points_fixed)
        target = np.array([[17, 1, 4, 1], [10, 3, 1, 1], [5, 1, -2, 1], [5, -2, 1, 1]])
        pass

    def test_circle_coefs_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=True)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, 1, 0, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        return True
        self.assertTrue((output_coefs == target_coefs).all())

    def test_circle_coefs_non_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=False)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        self.assertTrue((output_coefs == target_coefs).all())
