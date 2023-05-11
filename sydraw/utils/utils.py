import random as rand
from math import floor
from typing import List

import matplotlib.pyplot as plt


def compute_num_inliers_per_model(
    tot_num_inliers: int, num_of_models: int
) -> List[int]:
    """

    :param tot_num_inliers: total number of inliers in the whole sample
    :param num_of_models: the number of models to be drawn in each sample
    :return: a list that contains the number of inlier points for each model
    """
    n_inliers = []
    remaining = tot_num_inliers
    remaining_n_of_models = num_of_models
    while remaining_n_of_models > 0:
        cur_model_n_nliers = floor(remaining / remaining_n_of_models)
        n_inliers.append(cur_model_n_nliers)
        remaining -= cur_model_n_nliers
        remaining_n_of_models -= 1
    return n_inliers


def plot_sample(inliers, outliers, imgdir):
    for model in inliers:
        plt.scatter(*zip(*model), s=2)
    plt.scatter(*zip(*outliers), s=2)
    plt.grid(color="k", linestyle=":", linewidth=1)
    plt.axes().set_aspect("equal", "datalim")
    plt.gca().set_aspect("equal", adjustable="box")

    # save plot once in a while
    random_number = rand.randrange(10000)
    if random_number % 20 == 0:
        folder = imgdir + "/" + str(rand.randrange(10000)) + ".png"
        # checkExistenceAndCreate(folder)
        plt.savefig(folder)

    # show it
    plt.show()
