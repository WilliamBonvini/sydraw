from typing import Type
import matplotlib.pyplot as plt
import numpy as np


def plot_2d(points: Type[np.array]):
    """
    plots in 2D some points
    :param points: np.array, [(x_1,y_1,1), ..., (x_n,y_n,1)]
    :return:
    """
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.scatter(xs, ys, s=10)
