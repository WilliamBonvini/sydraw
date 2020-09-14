import math
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy.io

from datafact.maker import Maker
from datafact.pointgenerator import point, outliers_uniform
from datafact.utils import checkExistenceAndCreate, convert_to_np_struct_and_shuffle, convert_to_mat_struct


class CircleMaker(Maker):
    def __init__(self, coxwain):
        super(CircleMaker, self).__init__(coxwain)

    # creates a circle corrupted by noise with the parameters specified.
    def create_noisy_circle(self,x_center, y_center, radius, n_inliers=None, outliers_perc=None, plot=False):
        coxwain = self.getCoxwain()
        noiseperc = coxwain.getNoisePerc()
        xy = [point(y_center, x_center, radius,noiseperc) for _ in range(n_inliers)]
        n_outliers = coxwain.getNumPointsPerSample() - n_inliers
        xy_outliers = [outliers_uniform(x_center, y_center, radius) for _ in range(n_outliers)]

        if plot:
            plt.scatter(*zip(*xy), s=2)
            plt.scatter(*zip(*xy_outliers), s=2)
            plt.grid(color='k', linestyle=':', linewidth=1)
            plt.axes().set_aspect('equal', 'datalim')
            plt.gca().set_aspect('equal', adjustable='box')
            # save plot once in a while
            random_number = rand.randrange(10000)
            if random_number % 20 == 0:
                folder = './' + coxwain.getBaseDir() + '/' + coxwain.getCurDir() + '/' + 'imgs' + '/' + 'circles_no_' + str(
                    int(outliers_perc * 100)) + '/' + str(rand.randrange(10000)) + ".png"
                checkExistenceAndCreate(folder)
                plt.savefig(folder)

            # show it
            plt.show()

        return xy, xy_outliers




    def generate_circle(self,x_center: float, y_center: float, radius: float, n_inliers: int, outliers_perc: float = None):
        """
        generates a circle
        :param x_center: x of circle
        :param y_center: y of circle
        :param radius:  radius of circle
        :param n_inliers: number of points that are to be inliers
        :param outliers_perc: percentage of outliers
        :return:
        """
        coxwain = self.getCoxwain()
        inliers, outliers = self.create_noisy_circle(x_center, y_center, radius, n_inliers=n_inliers,
                                                outliers_perc=outliers_perc, plot=True)

        circle = convert_to_np_struct_and_shuffle(coxwain,inliers,outliers)

        circle_mat = convert_to_mat_struct(circle)

        return circle_mat





    def generate_dataset_given_or(self, outliers_perc):
        coxwain = self.getCoxwain()
        circle_dt = np.dtype([('x1p', 'O'), ('x2p', 'O'), ('labels', 'O')])
        numPoints = coxwain.getNumPointsPerSample()
        avg_num_inliers = numPoints - math.floor(float(numPoints) * outliers_perc)
        inliers_range = list(range(avg_num_inliers - 2, avg_num_inliers + 2))
        dir_list = [coxwain.getTrainDir(),coxwain.getTestDir()]


        # handle the possibility that one between the training and testing directory is empty. in such a case I won't create the data for such dir
        for curDir in dir_list:
            if curDir == '':
                continue
            coxwain.setCurDir(curDir)
            circles = []
            for _ in range(coxwain.getNumSamples()):
                x_center = uniform(-1, 1)
                y_center = uniform(-1, 1)
                radius = uniform(0.5, 1.5)
                n_inliers = inliers_range[rand.randint(0, len(inliers_range) - 1)]
                circle_mat = self.generate_circle(x_center, y_center, radius, n_inliers, outliers_perc=outliers_perc)
                circles.append(circle_mat)

            circle_arr = np.array(circles, dtype=circle_dt)
            dataset = np.array([circle_arr])
            # save it into a matlab file
            folder = './' + coxwain.getBaseDir() + '/' + curDir + '/circles_no_' + str(
                int(outliers_perc * 100)) + '.mat'
            scipy.io.savemat(folder, mdict={'dataset': dataset, 'outlierRate': outliers_perc})














