import abc
import math
import os
from abc import ABC
from random import uniform
from typing import Tuple, List

import numpy as np
import scipy.io
from syndalib.utils.config import opts
from syndalib.utils.utils import (
    compute_num_inliers_per_model,
    plot_sample,
    convert_to_np_struct,
    convert_to_mat_struct,
)


class Maker(ABC):
    """
    abstract class. serves as reference to create all subclasses that
    will implement the generation of models of specific classes.
    """
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
    DT = np.dtype([("x1p", "O"), ("x2p", "O"), ("labels", "O")])

    @abc.abstractmethod
    def generate_random_model(self, n_inliers: int) -> List:
        """
        generate a random model. the generating process changes for each class.

        :param n_inliers: int, number of inliers to be generated
        :return: list, list of tuples (2d coordinates) of inliers
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def point(*args, **kwargs) -> Tuple:
        """
        generates a point of the random model of the specified class

        :param args: any parameter needed to generate model
        :param kwargs:
        :return: a tuple (x,y) representing a 2D point
        """
        pass

    def generate_random_sample(self, plot: bool = True) -> Tuple:
        """
        generates a random sample of the desired class.

        :param: bool, true to plot sample, false otherwise. default is True.
        :return: tuple, (x1,x2,labels) ---
        where:
        x1 is an np.ndarray (npps,);
        x2 is an np.ndarray (npps,);
        labels is an np.ndarray (npps, nm);
        """
        tot_inliers = np.random.choice(Maker.INLIERS_RANGE)
        inliers_per_model = compute_num_inliers_per_model(
            tot_inliers, Maker.NUMBER_OF_MODELS
        )
        inliers = []
        for n_inliers in inliers_per_model:
            # model_inliers is a list of tuples (x,y)
            model_inliers = self.generate_random_model(n_inliers=n_inliers)
            inliers.append(model_inliers)

        n_outliers = Maker.NUM_POINTS_PER_SAMPLE - tot_inliers
        outliers = self.generate_outliers(n_outliers)

        if plot:
            plot_sample(inliers, outliers, Maker.IMG_DIR)

        sample = convert_to_np_struct(inliers=inliers, outliers=outliers)
        np.random.shuffle(sample)
        sample_mat = convert_to_mat_struct(sample)

        return sample_mat

    def generate_dataset_fixed_nr_and_or(self) -> None:
        """
        saves a .mat file in a structured fashioned folder with a fixed gaussian noise and a fixed outliers rate

        :return:
        """

        avg_num_inliers = math.floor(
            Maker.NUM_POINTS_PER_SAMPLE * (1.0 - Maker.OUTLIERS_PERC)
        )
        Maker.INLIERS_RANGE = list(range(avg_num_inliers - 2, avg_num_inliers + 2))

        # handle possibility that training or testing directory are empty
        for curDir in [Maker.TRAIN_DIR, Maker.TEST_DIR]:
            if curDir == "":
                continue
            Maker.CUR_DIR = curDir
            Maker.IMG_DIR = (
                Maker.IMG_BASE_DIR
                + "/"
                + Maker.MODEL
                + "_no_"
                + str(int(Maker.OUTLIERS_PERC * 100))
                + "_noise_"
                + str(Maker.NOISE_PERC)
            )
            if not os.path.exists(Maker.IMG_DIR):
                os.mkdir(Maker.IMG_DIR)

            samples = []
            for _ in range(Maker.NUM_SAMPLES):
                sample = self.generate_random_sample()
                samples.append(sample)

            samples = np.array(samples, dtype=Maker.DT)
            dataset = np.array([samples])
            # save it into a matlab file
            folder = (
                "./"
                + Maker.BASE_DIR
                + "/"
                + curDir
                + "/"
                + Maker.MODEL
                + "_no_"
                + str(int(Maker.OUTLIERS_PERC * 100))
                + "_noise_"
                + str(Maker.NOISE_PERC)
                + ".mat"
            )
            scipy.io.savemat(
                folder, mdict={"dataset": dataset, "outlierRate": Maker.OUTLIERS_PERC}
            )

    def generate_outliers(self, n_outliers: int) -> List:
        """
        generates a list of randomly sampled outliers

        :param n_outliers: int, number of outliers to be generated
        :return: list of tuples, list of randomly sampled outliers
        """
        x_min, x_max = opts["outliers"]["x_r"]
        y_min, y_max = opts["outliers"]["y_r"]

        outliers = [
            (uniform(x_min, x_max), uniform(y_min, y_max)) for _ in range(n_outliers)
        ]
        return outliers

    def start(self):
        for noise_stddev in Maker.NOISE_PERC_RANGE:
            for out_rate in Maker.OUTLIERS_PERC_RANGE:
                Maker.NOISE_PERC = noise_stddev
                Maker.OUTLIERS_PERC = out_rate
                self.generate_dataset_fixed_nr_and_or()
