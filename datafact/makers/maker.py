import math
import os
from abc import ABC, abstractmethod
from random import uniform
import numpy as np
import scipy.io
from datafact.utils.utils import compute_num_of_inliers_for_each_model, plot_sample, convert_to_np_struct, \
    convert_to_mat_struct


class Maker(ABC):
    NUM_SAMPLES = None
    NUM_POINTS_PER_SAMPLE = None
    BASE_DIR = None
    TRAIN_DIR = None
    TEST_DIR = None
    CUR_DIR = None
    NUMBER_OF_MODELS = None
    OUTLIERS_PERC_RANGE = None
    NOISE_PERC_RANGE = None
    NOISE_PERC = None
    OUTLIERS_PERC = None
    INLIERS_RANGE = None
    IMG_BASE_DIR = None
    IMG_DIR = None
    MODEL = None
    DT = np.dtype([('x1p', 'O'), ('x2p', 'O'), ('labels', 'O')])

    def setInliersRange(self, inliersRange):
        self._inliersRange = inliersRange

    def getInliersRange(self):
        return self._inliersRange

    def setNoisePerc(self, noisePerc):
        self._noisePerc = noisePerc

    def getNoisePerc(self):
        return self._noisePerc


    @abstractmethod
    def generate_random_model(self,n_inliers):
        pass

    def generate_random_sample(self, plot=True):
        """
        generates a random sample of the desired type.
        a sample is defined as written in the return field of the doc
        :param: plot
        :return: np.array with NUM_POINTS_PER_SAMPLE points,
        where each point is or an outlier or belongs to any
        of the NUM_MODELS models.
        """
        n_tot_inliers = np.random.choice(Maker.INLIERS_RANGE)
        n_inliers_for_each_model = compute_num_of_inliers_for_each_model(n_tot_inliers, Maker.NUMBER_OF_MODELS)
        inliers = []
        for n_inliers in n_inliers_for_each_model:
            # model_inliers is a list of tuples (x,y)
            model_inliers = self.generate_random_model(n_inliers=n_inliers)
            inliers.append(model_inliers)

        # flatten inliers
        # inliers = [item for sublist in inliers for item in sublist]

        n_outliers = Maker.NUM_POINTS_PER_SAMPLE - n_tot_inliers
        outliers = self.generate_outliers(n_outliers)

        if plot:
            imgdir = Maker.IMG_DIR
            plot_sample(inliers, outliers, imgdir)

        sample = convert_to_np_struct(inliers,
                                      outliers)  # prima c'era convert_to_np_struct_and_shuffle e la transpose era dentro sta funzione, se voglio tornare indietro devo togliere la transpose da convert_to_mat_struct
        np.random.shuffle(sample)
        sample_mat = convert_to_mat_struct(sample)

        return sample_mat


    def generate_dataset_fixed_nr_and_or(self):
        """

        :return:
        """
        avg_num_inliers = math.floor(Maker.NUM_POINTS_PER_SAMPLE * (1.0 - Maker.OUTLIERS_PERC))
        inliers_range = list(range(avg_num_inliers - 2, avg_num_inliers + 2))
        Maker.INLIERS_RANGE = inliers_range

        # handle possibility that training or testing directory are empty
        for curDir in [Maker.TRAIN_DIR, Maker.TEST_DIR]:
            if curDir == '':
                continue
            Maker.CUR_DIR = curDir
            Maker.IMG_DIR = Maker.IMG_BASE_DIR + '/' + Maker.MODEL + '_no_' + str(
                int(Maker.OUTLIERS_PERC * 100)) + '_noise_' + str(Maker.NOISE_PERC)
            if not os.path.exists(Maker.IMG_DIR):
                os.mkdir(Maker.IMG_DIR)

            samples = []
            for _ in range(Maker.NUM_SAMPLES):
                sample = self.generate_random_sample()
                samples.append(sample)

            samples = np.array(samples, dtype=Maker.DT)
            dataset = np.array([samples])
            # save it into a matlab file
            folder = './' + Maker.BASE_DIR + '/' + curDir + '/' + Maker.MODEL + '_no_' + str(
                int(Maker.OUTLIERS_PERC * 100)) + '_noise_' + str(Maker.NOISE_PERC) + '.mat'
            scipy.io.savemat(folder, mdict={'dataset': dataset, 'outlierRate': Maker.OUTLIERS_PERC})

    def generate_outliers(self, n_outliers):
        outliers = [(uniform(-2.5, 2.5), uniform(-2.5, 2.5)) for _ in range(n_outliers)]
        return outliers

    def start(self):
        for nPerc in Maker.NOISE_PERC_RANGE:
            for oPerc in Maker.OUTLIERS_PERC_RANGE:
                self.setNoisePerc(nPerc)
                Maker.NOISE_PERC = nPerc
                Maker.OUTLIERS_PERC = oPerc
                self.generate_dataset_fixed_nr_and_or()
