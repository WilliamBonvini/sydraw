from unittest import TestCase
from tests import *
import numpy as np
from syndalib import linalg
import tensorflow as tf


class Test(TestCase):
    def setUp(self) -> None:
        pass


    def test_conic_monomials(self):
        output = linalg.conic_monomials(circle_points_fixed)

        target = np.array([[1, 16, 4, 1, 4, 1],
                           [9, 1, 3, 3, 1, 1],
                           [1, 4, -2, 1, -2, 1],
                           [4, 1, -2, -2, 1, 1]], dtype=float)

        self.assertTrue((output == target).all())

    def test_circle_monomials(self):
        output = linalg.circle_monomials(circle_points_fixed)
        target = np.array([[17, 1, 4, 1],
                           [10, 3, 1, 1],
                           [5, 1, -2, 1],
                           [5, -2, 1, 1]])
        self.assertTrue((output == target).all())

    def test_circle_coefs_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=True)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, 1, 0, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        self.assertTrue((output_coefs == target_coefs).all())

    def test_circle_coefs_non_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=False)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        self.assertTrue((output_coefs == target_coefs).all())


    def test_dlt_coefs_with_nparray(self):
        coefs = linalg.dlt_coefs(circle_vandermonde_as_conic, circle_inliers_prob)
        target = circle_coefs_as_conic
        print("coefs = {}".format(coefs))
        print("target = {}".format(target))
        self.assertTrue(np.allclose(coefs, target))



    def test_dlt_coefs_with_tf_tensors(self):
        coefs = tf.constant(linalg.dlt_coefs(circle_vandermonde_as_conic, circle_inliers_prob))
        target = tf.constant(circle_coefs_as_conic)
        print("coefs = {}".format(coefs))
        print("target = {}".format(target))
        self.assertTrue(np.allclose(coefs, target))








