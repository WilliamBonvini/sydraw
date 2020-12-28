from unittest import TestCase
import syndalib.drawer as sydraw
import numpy as np


class SyndalibTest(TestCase):
    def setUp(self) -> None:
        self.circle_params = (3.0, (1.0, 1.0))
        self.circle_points = sydraw.circle_points(*self.circle_params, n=10, noise=0.0, homogeneous=True)
        self.circle_points_fixed = np.array([[1, 4, 1], [4, 1, 1], [1, -2, 1], [-2, 1, 1]], dtype=float)   # c = (1,1); r = 3
        self.circle_inliers_prob = np.ones(shape=10)

