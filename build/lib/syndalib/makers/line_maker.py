from random import uniform
from typing import List

from syndalib.makers.maker import Maker
import numpy as np
from syndalib.utils.config import opts


class LineMaker(Maker):
    """
    Maker implementation for lines sampled from a 2D space [x_min,x_max] x [y_min, y_max]
    you can define the 2D space by calling the function syndalib.syn2d(...)
    """
    def __init__(self):
        Maker.MODEL = "lines"
        super(LineMaker, self).__init__()

    @staticmethod
    def point(m, q, noise_perc, x_min, x_max, y_min, y_max):
        while True:
            x = uniform(x_min, x_max)
            y = m * x + q + np.random.normal(0, noise_perc)
            if y_min <= y <= y_max:
                break

        return x, y

    def generate_random_model(self, n_inliers: int) -> List:
        x_min, x_max = opts["lines"]["x_r"]
        y_min, y_max = opts["lines"]["y_r"]

        x1 = uniform(x_min, x_max)
        y1 = uniform(y_min, y_max)

        x2 = uniform(x_min, x_max)
        y2 = uniform(y_min, y_max)

        m = (y2 - y1) / (x2 - x1)
        q = y1 - m*x1

        xy = [LineMaker.point(m, q, LineMaker.NOISE_PERC, x_min, x_max, y_min, y_max) for _ in range(n_inliers)]
        return xy
