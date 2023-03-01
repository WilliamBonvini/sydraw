import syndalib.src.synth as sydraw
import numpy as np
circle_params = (3.0, (1.0, 1.0))
circle_points = sydraw.circles(ns=20,
                               radius=3.0,
                               center=(1.0, 1.0),
                               n=10,
                               noise_perc=0.0,
                               homogeneous=True)
circle_points_fixed = np.array([[1, 4, 1], [4, 1, 1], [1, -2, 1], [-2, 1, 1]], dtype=float)  # c = (1,1); r = 3
circle_inliers_prob = np.ones(shape=10)
