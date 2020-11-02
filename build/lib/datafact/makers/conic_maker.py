import math
import os
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy.io

from datafact.makers.circle_maker import CircleMaker
from datafact.makers.ellipse_maker import EllipseMaker
from datafact.makers.line_maker import LineMaker
from datafact.makers.maker import Maker
from datafact.utils.utils import checkExistenceAndCreate, convert_to_np_struct_and_shuffle, convert_to_mat_struct, \
    compute_num_of_inliers_for_each_model, plot_sample, \
    convert_to_np_struct


class ConicMaker(Maker):

    def __init__(self):
        Maker.MODEL = 'conics'
        self.circle_maker = CircleMaker()
        self.line_maker = LineMaker()
        self.ellipse_maker = EllipseMaker()

        super(ConicMaker, self).__init__()


    def generate_random_model(self, n_inliers=None):
        """
        :param n_inliers: number of inlier points for this specific model
        :return:
        """
        randNum = np.random.randint(0, 3)

        if randNum == 0:
            return self.circle_maker.generate_random_model(n_inliers)
        if randNum == 1:
            return self.line_maker.generate_random_model(n_inliers)
        if randNum == 2:
            return self.ellipse_maker.generate_random_model(n_inliers)














