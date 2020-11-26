import math
import random
import numpy as np

from typing import Tuple


def circle_points(r: float,
                  c: Tuple[float, float],
                  n: int):
    """

    :param r: radius
    :param c: center (x,y)
    :param n: number of points
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)]
    """
    points = []
    for _ in range(n):
        alpha = 2 * math.pi * random.random()
        x = c[0] + r * math.cos(alpha)
        y = c[1] + r * math.sin(alpha)
        points.append((x, y, 1))

    return np.array(points)


def line_points(point_1: Tuple[float, float],
                point_2: Tuple[float, float],
                n: int):
    """
    samples point from a line obtained by connecting two points
    :param point_1: (float, float)
    :param point_2: (float, float)
    :param n: number of points to be drawn
    :return: np.array, [(x_1,y_1,1), ..., (x_n,y_n,1)]
    """
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    m = (y1-y2)/(x1-x2)
    q = y1-m*x1
    points = []
    for _ in range(n):
        x = random.uniform(point_1[0], point_2[0])
        y = m*x + q
        points.append((x, y, 1))
    return np.array(points)
