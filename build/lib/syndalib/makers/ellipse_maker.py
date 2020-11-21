import math
from random import uniform
import numpy as np
import random as rand

from syndalib.makers.maker import Maker
from syndalib.utils.config import opts


class EllipseMaker(Maker):
    """
    Maker implementation for ellipses sampled from a 2D space.
    you can define the 2D space by calling the function syndalib.syn2d(...).
    """

    def __init__(self):
        Maker.MODEL = "ellipses"
        super(EllipseMaker, self).__init__()

    def generate_random_model(self, n_inliers):
        x_c_min, x_c_max = opts["ellipses"]["x_center_r"]
        y_c_min, y_c_max = opts["ellipses"]["y_center_r"]
        r_min, r_max = opts["ellipses"]["radius_r"]
        w_min, w_max = opts["ellipses"]["width_r"]
        h_min, h_max = opts["ellipses"]["height_r"]

        x_center = uniform(x_c_min, x_c_max)
        y_center = uniform(y_c_min, y_c_max)
        radius = uniform(r_min, r_max)
        width = uniform(w_min, w_max)
        height = uniform(h_min, h_max)
        xy = [
            EllipseMaker.point(
                y_center, x_center, radius, width, height, EllipseMaker.NOISE_PERC
            )
            for _ in range(n_inliers)
        ]
        return xy

    @staticmethod
    def point(h, k, r, width, height, noiseperc):
        """
        draws a random point belonging to an ellipse by scaling a circle with center (h,k) and radius r

        :param h:
        :param k:
        :param r:
        :param width:
        :param height:
        :param noiseperc:
        :return:
        """
        theta = rand.random() * 2 * math.pi
        x = h + math.cos(theta) * r
        noise = np.random.normal(0, noiseperc)
        x = x + noise
        y = k + math.sin(theta) * r
        noise = np.random.normal(0, noiseperc)
        y = y + noise

        return x * width, y * height
