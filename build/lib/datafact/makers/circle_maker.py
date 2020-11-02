import math
import os
from random import uniform
import numpy as np
import random as rand
from datafact.makers.maker import Maker


class CircleMaker(Maker):

    def __init__(self):
        Maker.MODEL = 'circles'
        super(CircleMaker, self).__init__()


    def generate_random_model(self, n_inliers=None):
        """
        :param n_inliers: number of inlier points for this specific model
        :return:
        """
        x_center = uniform(-1, 1)
        y_center = uniform(-1, 1)
        radius = uniform(0.5, 1.5)
        xy = [CircleMaker.point(y_center, x_center, radius, CircleMaker.NOISE_PERC) for _ in range(n_inliers)]
        return xy

    @staticmethod
    def point(h, k, r, noiseperc):
        """
        draws a random point belonging to a circle with center (h,k) and radius r
        :param h:
        :param k:
        :param r:
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
        return x, y

"""
    @staticmethod
    def outliers_uniform(h, k, r):
        
        :param h: 
        :param k: 
        :param r: 
        :return: an outlier point 
        
        x = rand.uniform(h - 2 * r, h + 2 * r)
        y = rand.uniform(k - 2 * r, k + 2 * r)
        return x, y
"""


"""
    def generate_random_sample(self, plot=True):
      

        n_tot_inliers = rand.choice(CircleMaker.INLIERS_RANGE)
        n_inliers_for_each_model = compute_num_of_inliers_for_each_model(n_tot_inliers, CircleMaker.NUMBER_OF_MODELS)
        inliers = []

        for n_inliers in n_inliers_for_each_model:
            # model_inliers is a list of tuples (x,y)
            model_inliers = self.generate_random_model(n_inliers=n_inliers)
            inliers.append(model_inliers)

        # flatten inliers
        #inliers = [item for sublist in inliers for item in sublist]

        n_outliers = CircleMaker.NUM_POINTS_PER_SAMPLE - n_tot_inliers
        outliers = self.generate_outliers(n_outliers)

        if plot:
            imgdir = CircleMaker.IMG_DIR
            plot_sample(inliers, outliers, imgdir)

        sample = convert_to_np_struct(inliers, outliers)  # prima c'era convert_to_np_struct_and_shuffle e la transpose era dentro sta funzione, se voglio tornare indietro devo togliere la transpose da convert_to_mat_struct
        np.random.shuffle(sample)
        sample_mat = convert_to_mat_struct(sample)

        return sample_mat

"""

"""
    def generate_dataset_fixed_nr_and_or(self):
       

        avg_num_inliers = math.floor(CircleMaker.NUM_POINTS_PER_SAMPLE*(1.0 - CircleMaker.OUTLIERS_PERC))
        inliers_range = list(range(avg_num_inliers - 2, avg_num_inliers + 2))
        CircleMaker.INLIERS_RANGE = inliers_range

        # handle possibility that training or testing directory are empty
        for curDir in [CircleMaker.TRAIN_DIR, CircleMaker.TEST_DIR]:
            if curDir == '':
                continue
            CircleMaker.CUR_DIR = curDir
            CircleMaker.IMG_DIR = CircleMaker.IMG_BASE_DIR + '/' + CircleMaker.MODEL + '_no_' + str(
            int(CircleMaker.OUTLIERS_PERC * 100)) + '_noise_' + str(CircleMaker.NOISE_PERC)
            if not os.path.exists(CircleMaker.IMG_DIR):
                os.mkdir(CircleMaker.IMG_DIR)

            samples = []
            for _ in range(CircleMaker.NUM_SAMPLES):

                sample = self.generate_random_sample()
                samples.append(sample)

            samples = np.array(samples, dtype=CircleMaker.CIRCLE_DT)
            dataset = np.array([samples])
            # save it into a matlab file
            folder = './' + CircleMaker.BASE_DIR + '/' + curDir + '/circles_no_' + str(
                int(CircleMaker.OUTLIERS_PERC * 100)) + '_noise_' + str(CircleMaker.NOISE_PERC) + '.mat'
            scipy.io.savemat(folder, mdict={'dataset': dataset, 'outlierRate': CircleMaker.OUTLIERS_PERC})

"""