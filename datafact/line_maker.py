import math
from random import uniform
import random as rand

from datafact.maker import Maker
import numpy as np
import scipy.io

from datafact.utils import convert_to_np_struct_and_shuffle, convert_to_mat_struct, checkExistenceAndCreate
import matplotlib.pyplot as plt


class LineMaker(Maker):
    def __init__(self, coxwain):
        super(LineMaker, self).__init__(coxwain)






    def point(self,m,q,noise_perc):
        x = uniform(-8,8)
        y = m*x + q + np.random.normal(0,noise_perc)
        return x,y

    def outliers_uniform(self,m,q):
        x = uniform(-8,8)
        y = uniform(-8,8)
        return x,y




    def generate_dataset_given_or(self, outliers_perc):
        coxwain = self.getCoxwain()
        line_dt = np.dtype([('x1p', 'O'), ('x2p', 'O'), ('labels', 'O')])
        numPoints = coxwain.getNumPointsPerSample()
        avg_num_inliers = numPoints - math.floor(float(numPoints) * outliers_perc)
        inliers_range = list(range(avg_num_inliers - 2, avg_num_inliers + 2))
        dir_list = [coxwain.getTrainDir(), coxwain.getTestDir()]

        # handle the possibility that one between the training and testing directory is empty. in such a case I won't create the data for such dir
        for curDir in dir_list:
            if curDir == '':
                continue
            coxwain.setCurDir(curDir)
            lines = []
            for _ in range(coxwain.getNumSamples()):
                m = uniform(-3, 3)
                q = uniform(-1, 1)

                n_inliers = inliers_range[rand.randint(0, len(inliers_range) - 1)]
                line_mat = self.generate_line(m, q, n_inliers, outliers_perc=outliers_perc)
                lines.append(line_mat)

            line_arr = np.array(lines, dtype=line_dt)
            dataset = np.array([line_arr])
            # save it into a matlab file
            folder = './' + coxwain.getBaseDir() + '/' + curDir + '/lines_no_' + str(
                int(outliers_perc * 100)) + '.mat'
            scipy.io.savemat(folder, mdict={'dataset': dataset, 'outlierRate': outliers_perc})


    def generate_line(self, m, q, n_inliers, outliers_perc):
        coxwain = self.getCoxwain()
        inliers, outliers = self.create_noisy_line(m, q, n_inliers=n_inliers,
                                                     outliers_perc=outliers_perc, plot=True)

        line = convert_to_np_struct_and_shuffle(coxwain, inliers, outliers)

        line_mat = convert_to_mat_struct(line)
        return line_mat



    def create_noisy_line(self, m, q, n_inliers, outliers_perc, plot):
        coxwain = self.getCoxwain()
        noiseperc = coxwain.getNoisePerc()
        xy = [self.point(m,q, noiseperc) for _ in range(n_inliers)]
        n_outliers = coxwain.getNumPointsPerSample() - n_inliers
        xy_outliers = [self.outliers_uniform(m,q) for _ in range(n_outliers)]

        if plot:
            plt.scatter(*zip(*xy), s=2)
            plt.scatter(*zip(*xy_outliers), s=2)
            plt.grid(color='k', linestyle=':', linewidth=1)
            plt.axes().set_aspect('equal', 'datalim')
            plt.gca().set_aspect('equal', adjustable='box')
            # save plot once in a while
            random_number = rand.randrange(10000)
            if random_number % 20 == 0:
                folder = './' + coxwain.getBaseDir() + '/' + coxwain.getCurDir() + '/' + 'imgs' + '/' + 'lines_no_' + str(
                    int(outliers_perc * 100)) + '/' + str(rand.randrange(10000)) + ".png"
                checkExistenceAndCreate(folder)
                plt.savefig(folder)

            # show it
            plt.show()

        return xy, xy_outliers






