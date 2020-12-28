import syndalib.drawer as sydraw
import syndalib.linalg as syla
import numpy as np
circle_params = (3.0, (1.0, 1.0))
circle_points = sydraw.circle_points(*circle_params, n=10, noise=0.0, homogeneous=True)
circle_points_fixed = np.array([[1, 4, 1], [4, 1, 1], [1, -2, 1], [-2, 1, 1]], dtype=float)  # c = (1,1); r = 3
circle_inliers_prob = np.ones(shape=10)
circle_vandermonde_as_conic = syla.conic_monomials(circle_points)
circle_coefs_as_circle = np.array([1, -2, -2, -7])
circle_coefs_as_conic = np.array([1, 0, 1, -2, -2, -7])
