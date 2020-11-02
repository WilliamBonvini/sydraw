import math
from random import uniform
import numpy as np
import random as rand

from datafact.makers.maker import Maker
from datafact.utils.utils import checkExistenceAndCreate, convert_to_np_struct_and_shuffle, convert_to_mat_struct, \
    compute_num_of_inliers_for_each_model, plot_sample, \
    convert_to_np_struct


class EllipseMaker(Maker):

    def __init__(self):
        Maker.MODEL = 'ellipses'
        super(EllipseMaker, self).__init__()


    def generate_random_model(self, n_inliers=None):
        """
        :param n_inliers: number of inlier points for this specific model
        :return:
        """
        x_center = uniform(-1, 1)
        y_center = uniform(-1, 1)
        radius = uniform(0.5, 1.5)
        width = uniform(0.1, 1)
        height = uniform(0.1, 1)
        xy = [EllipseMaker.point(y_center, x_center, radius, width, height, EllipseMaker.NOISE_PERC) for _ in range(n_inliers)]
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

        return x*width, y*height
