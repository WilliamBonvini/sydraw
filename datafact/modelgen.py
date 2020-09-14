import math
import string

import matplotlib.pyplot as plt
from math import pi, cos, sin
from random import random,uniform
import numpy as np
from random import randrange
import random as rand
import numpy as np
import os

import scipy.io

from datafact.coxswain import Coxwain, MyCoxwain
from datafact.line_maker import LineMaker
from datafact.utils import getRanges, checkExistenceAndCreate

from datafact.circle_maker import CircleMaker

coxwain = Coxwain(os.getcwd())






def set_train_dir(path: string):
    coxwain.setTrainDir(path)


def set_test_dir(path: string):
    coxwain.setTestDir(path)




def start():
    if coxwain.getModel() =='circle':
        circleMaker = CircleMaker(coxwain)
        circleMaker.start()
    elif coxwain.getModel()=='line':
        lineMaker = LineMaker(coxwain)
        lineMaker.start()






def generate_data(num_samples: int, num_points_per_sample: int, outliers_rate_range, model: string, noise_perc: float,dest: string = 'matlab',dirs = ['first_dataset','train','.']):
    """
    :param num_samples: total number of samples to be generated for each outlier rate
    :param num_points_per_sample: total number of points within each sample
    :param outliers_rate_range: list that contains outliers rates
    :param model: a string that identifies the model to be created. the possible models are: 'circle','line'
    :param noise_perc: the stddev of the gaussian noise to be added to the inliers
    :param dest: you want to read it with? 'matlab', 'numpy'
    :param dirs: list containing path to train data dir and to test data dir
    :return: a nice looking np array
    """
    coxwain.setNumSamples(num_samples)
    coxwain.setNumPointsPerSample(num_points_per_sample)
    coxwain.setOutliersRateRange(outliers_rate_range)
    coxwain.setModel(model)
    coxwain.setNoisePerc(noise_perc)
    coxwain.setDest(dest)
    coxwain.setBaseDir(dirs[0])
    coxwain.setTrainDir(dirs[1])
    coxwain.setTestDir(dirs[2])
    start()











