import math
import random
from typing import Optional, Tuple, Union

import numpy as np

from sydraw.utils.config import OPTS
from sydraw.utils.utils import compute_num_inliers_per_model


def outliers(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n: int,
    homogeneous: bool = True,
) -> np.ndarray:
    """
    generates outliers in a user specified 2D square

    :param x_range: floats tuple, interval of x values
    :param y_range: floats tuple, interval if y values
    :param n: number of outliers
    :param homogeneous: if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)] if homogeneous, else [(x_1,y_1),...,(x_np,y_np)]
    """
    n_coords = 3 if homogeneous else 2
    if n <= 0:
        return np.zeros((n, n_coords))
    elif x_range[0] >= x_range[1]:
        raise ValueError("invalid interval for x_range")
    elif y_range[0] >= y_range[1]:
        raise ValueError("invalid interval for y_range")

    outliers_points = np.ones((n, n_coords), dtype=float)
    for i in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        outlier = outliers_points[i]
        outlier[0] = x
        outlier[1] = y

    return outliers_points


# ---------------------------------- CIRCLES -----------------------------------


def circle(
    radius: float,
    center: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    homogeneous: bool = False,
) -> np.ndarray:
    """
    generates circle points given radius and center.

    :param radius: radius
    :param center: center (x,y)
    :param n: number of points
    :param noise_perc: gaussian noise standard deviation
    :param homogeneous: bool, if True returns homogeneous coordinates, otherwise euclidean coordinates, default is False
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)]
    """

    # setup
    n_coords = 3 if homogeneous else 2

    points = np.ones((n, n_coords))

    # generate circle's points
    for i in range(n):
        alpha = 2 * math.pi * random.random()
        x = center[0] + radius * math.cos(alpha) + np.random.normal(0, noise_perc)
        y = center[1] + radius * math.sin(alpha) + np.random.normal(0, noise_perc)
        points[i, 0] = x
        points[i, 1] = y

    return points


def circles_sample(
    nm: int,
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = False,
    radius: Optional[Union[float, Tuple[float, float], None]] = None,
) -> np.ndarray:
    """

    :param nm: number of models
    :param n: number of points in circle
    :param noise_perc:
    :param outliers_perc:
    :param homogeneous:
    :param radius
    :return:
    """

    # setup
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points
    n_coords = 3 if homogeneous else 2
    num_inl_per_model = compute_num_inliers_per_model(
        tot_num_inliers=n_points, num_of_models=nm
    )
    opt_circle = OPTS["circles"]
    if radius is None:
        min_r, max_r = opt_circle["radius_r"]
    elif isinstance(radius, float):
        min_r, max_r = radius, radius
    elif isinstance(radius, tuple) and all(isinstance(x, float) for x in radius):
        min_r, max_r = radius

    min_x_center, max_x_center = opt_circle["x_center_r"]
    min_center_y, max_center_y = opt_circle["y_center_r"]

    points = np.ones((n_points, n_coords + 1))
    i_tot = 0
    for i_model, npic in zip(range(nm), num_inl_per_model):
        # npic: number of points in circle
        radius = random.uniform(min_r, max_r)
        x_center = random.uniform(min_x_center, max_x_center)
        y_center = random.uniform(min_center_y, max_center_y)
        center = x_center, y_center

        c = circle(
            radius=radius,
            center=center,
            n=npic,
            noise_perc=noise_perc,
            homogeneous=homogeneous,
        )

        points[i_tot : i_tot + npic, 0:n_coords] = c
        points[i_tot : i_tot + npic, -1] = i_model + 1
        i_tot += npic

    # generate outliers
    x_range = OPTS["outliers"]["x_r"]
    y_range = OPTS["outliers"]["y_r"]
    outliers_points = outliers(
        x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
    )
    outlier_points_labelled = np.zeros((n_outliers, n_coords + 1))
    outlier_points_labelled[:, 0:n_coords] = outliers_points

    points = np.concatenate((points, outlier_points_labelled))
    np.random.shuffle(points)
    return points


def circles_dataset(
    nm: int,
    ns: int,
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = False,
    radius: Union[float, Tuple[float, float], None] = None,
) -> np.ndarray:
    """

    :param nm: number of models in each sample
    :param ns: number of samples
    :param n: number of points
    :param noise_perc: gaussian noise standard deviation
    :param outliers_perc: percentage of outliers out of n data points
    :param homogeneous: bool, if True returns homogeneous coordinates, otherwise euclidean coordinates, default is False
    :param radius:
    :return: np.array, shape (ns, n, num coordinates)
    """

    dim = 4 if homogeneous else 3
    samples = np.zeros((ns, n, dim))

    for i in range(ns):

        samples[i] = circles_sample(
            nm=nm,
            n=n,
            noise_perc=noise_perc,
            outliers_perc=outliers_perc,
            homogeneous=homogeneous,
            radius=radius,
        )
    return samples


# ---------------------------------- ELLIPSIS -----------------------------------


def ellipse(
    semi_x_axis: float,
    semi_y_axis: float,
    center: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = True,
    outliers_bounded: bool = True,
) -> np.ndarray:
    """
    samples point from an ellipse.
    Equation is (x-x_0)^2/a^2 + (y-y_0)^2/b^2=1.

    :param semi_x_axis: length of the semi-axis on the abscissa, commonly colled 'a'
    :param semi_y_axis: length of the semi-axis on the ordinate, commonly called 'b'
    :param center: center of the ellipse
    :param n: number of points to be sampled
    :param noise_perc: gaussian noise standard deviation, default is 0.0
    :param outliers_perc:
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :param outliers_bounded: bool,
    :return: np.array, shape = (n, n_coords)
    """

    # setup
    a = semi_x_axis
    b = semi_y_axis
    x_0, y_0 = center
    x_sampling_interval = (x_0 - a, x_0 + a)
    n_coords = 3 if homogeneous else 2
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points
    points = np.ones((n_points, n_coords))

    # generate ellipse's points
    for i in range(n_points):
        x = random.uniform(*x_sampling_interval)
        # equation for sampling y: y = sqrt((1-((x-x_0)**2)/a**2)*b**2) + y_0
        y = math.sqrt((1 - ((x - x_0) ** 2 / a**2)) * b**2)
        y = (
            y if random.random() < 0.5 else -y
        )  # choose positive or negative solution of the square root
        y = y + y_0 + np.random.normal(0, noise_perc)
        x = x + np.random.normal(0, noise_perc)
        points[i, 0] = x
        points[i, 1] = y

    # generate outliers
    x_range = (
        (center[0] - a, center[0] + a) if outliers_bounded else OPTS["outliers"]["x_r"]
    )
    y_range = (
        (center[1] - b, center[1] + b) if outliers_bounded else OPTS["outliers"]["y_r"]
    )
    outliers_points = outliers(
        x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
    )

    # inliers
    r, c = points.shape
    inliers = np.ones((r, c + 1))
    inliers[:, 0:n_coords] = points

    # outliers
    r, c = outliers_points.shape
    outliers_ = np.zeros((r, c + 1))
    outliers_[:, 0:n_coords] = outliers_points

    points = np.concatenate((inliers, outliers_))
    np.random.shuffle(points)

    return points


def ellipsis(
    ns: int,
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = True,
    outliers_bounded: bool = True,
) -> np.ndarray:
    """
    Return a numpy array of ns ellipsis

    :param ns: total number of samples
    :param n: number of points to be sampled
    :param noise_perc: gaussian noise standard deviation, default is 0.0
    :param outliers_perc:
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :param outliers_bounded: bool,
    :return: np.array, shape = (ns, n, 3) if homogeneous = False; (ns, n, 4) if homogeneous = True.
    """
    dim = 3 if not homogeneous else 4
    samples = np.zeros((ns, n, dim))
    opts_ellipse = OPTS["ellipses"]
    min_radius, max_radius = opts_ellipse["radius_r"]
    semi_x_axis = random.uniform(min_radius, max_radius)
    semi_y_axis = random.uniform(min_radius, max_radius)
    min_x_center, max_x_center = opts_ellipse["x_center_r"]
    x_center = random.uniform(min_x_center, max_x_center)
    min_y_center, max_y_center = opts_ellipse["y_center_r"]
    y_center = random.uniform(min_y_center, max_y_center)
    center = x_center, y_center

    for i in range(ns):
        samples[i] = ellipse(
            semi_x_axis=semi_x_axis,
            semi_y_axis=semi_y_axis,
            center=center,
            n=n,
            noise_perc=noise_perc,
            outliers_perc=outliers_perc,
            homogeneous=homogeneous,
            outliers_bounded=outliers_bounded,
        )
    return samples


def hyperbola(
    semi_x_axis: float,
    semi_y_axis: float,
    center: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = True,
    separate: bool = True,
    outliers_bounded: bool = True,
) -> np.ndarray:
    """
    samples point from an ellipse
    :param semi_x_axis: length of the semi-axis on the abscissa, commonly colled 'a'
    :param semi_y_axis: length of the semi-axis on the ordinate, commonly called 'b'
    :param center: center of the ellipse
    :param n: number of points to be sampled
    :param noise_perc: gaussian noise standard deviation, default is 0.0
    :param outliers_perc:
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :param separate: bool,
    :param outliers_bounded: bool,
    :return: np.array, shape = (n, n_coords)
    """

    # setup
    n_coords = 2 if not homogeneous else 3
    a = semi_x_axis
    b = semi_y_axis
    (x_0, y_0) = center

    n_points = int(n * (1 - outliers_perc))

    points = np.ones((n_points, n_coords))

    n_outliers = n - n_points

    # generate hyperbola's points
    for i in range(n_points):
        # sample points on the x-axis in such a way they belong in the space of one of the two compontents
        x = random.uniform(x_0 + a, x_0 + 2 * math.sqrt(a**2 + b**2))
        x = x if random.random() < 0.5 else -x
        # equation for sampling y:  y = sqrt((((x-x_0)**2 / a**2)  -  1) * b**2) + y_0
        sqrtarg = (((x - x_0) ** 2 / a**2) - 1) * b**2
        y = math.sqrt(sqrtarg)
        y = (
            y if random.random() < 0.5 else -y
        )  # choose positive or negative solution of the square root
        y = y + y_0 + np.random.normal(0, noise_perc)
        x = x + np.random.normal(0, noise_perc)
        points[i, 0] = x
        points[i, 1] = y

    # generate outliers
    if n_outliers > 0:
        x_outliers_range = (min(points[:, 0]), max(points[:, 0]))
        y_outliers_range = (min(points[:, 1]), max(points[:, 1]))
        x_range = x_outliers_range if outliers_bounded else OPTS["outliers"]["x_r"]
        y_range = y_outliers_range if outliers_bounded else OPTS["outliers"]["y_r"]
        outliers_points = outliers(
            x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
        )
        # process data for output
        if not separate:
            output = np.concatenate((points, outliers_points))
            np.random.shuffle(output)
        else:
            output = points, outliers_points
        return output

    return points


def parabola(
    a: float,
    b: float,
    c: float,
    n: int,
    theta: float = 0,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = True,
    separate: bool = True,
    outliers_bounded: bool = True,
) -> np.ndarray:
    """
    samples points from a parabola through equation y = ax^2 + bx + c. You can rotate the parabola specifying
    the rotation angle theta ( bounded between 0 and 2*pi).

    :param a:
    :param b:
    :param c:
    :param theta:
    :param n: number of points to be sampled
    :param noise_perc: gaussian noise standard deviation, default is 0.0
    :param outliers_perc:
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :param separate: bool,
    :param outliers_bounded: bool,
    :return: np.array, shape = (n, n_coords)
    """

    # setup
    n_points = int(n * (1 - outliers_perc))
    n_coords = 2 if not homogeneous else 3

    points = np.ones((n_points, n_coords))
    n_outliers = n - n_points
    x_vertex = -b / 2 * a
    amplitude = 2 / (0.1 + a)
    x_sampling_interval = (x_vertex - amplitude, x_vertex + amplitude)

    # generate parabola's points
    for i in range(n_points):
        x = random.uniform(*x_sampling_interval)
        y = a * x**2 + b * x + c
        xp = x * math.cos(theta) - y * math.sin(theta) + np.random.normal(0, noise_perc)
        yp = y * math.cos(theta) + x * math.sin(theta) + np.random.normal(0, noise_perc)
        points[i, 0] = xp
        points[i, 1] = yp

    # generate outliers
    if n_outliers > 0:
        x_outliers_range = (
            min(point[0] for point in points),
            max(point[0] for point in points),
        )
        y_outliers_range = (
            min(point[1] for point in points),
            max(point[1] for point in points),
        )
        x_range = x_outliers_range if outliers_bounded else OPTS["outliers"]["x_r"]
        y_range = y_outliers_range if outliers_bounded else OPTS["outliers"]["y_r"]
        outliers_points = outliers(
            x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
        )

        # process data for output
        if not separate:
            output = np.concatenate((points, outliers_points))
            np.random.shuffle(output)
        else:
            output = points, outliers_points
        return output
    return points


def line(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = True,
    separate: bool = True,
    outliers_bounded: bool = True,
):
    """
    samples point from a line obtained by connecting two points
    :param point_1: (float, float)
    :param point_2: (float, float)
    :param n: number of points to be drawn
    :param noise_perc: gaussian noise standard deviation, default is 0.0
    :param outliers_perc:
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :param separate: bool,
    :param outliers_bounded: bool,
    :return: np.array, shape = (n, n_coords)
    """

    # setup
    n_coords = 2 if not homogeneous else 3
    x1, y1 = point_1
    x2, y2 = point_2
    m = (y1 - y2) / (x1 - x2)
    q = y1 - m * x1
    n_points = int(n * (1 - outliers_perc))
    points = np.ones((n_points, n_coords))

    n_outliers = n - n_points

    # generate line's points
    for i in range(n_points):
        x = random.uniform(point_1[0], point_2[0])
        y = m * x + q + np.random.normal(0, noise_perc)
        points[i, 0] = x
        points[i, 1] = y

    # generate outliers
    if n_outliers > 0:
        x_range = (
            (min(points[:, 0]), max(points[:, 0]))
            if outliers_bounded
            else OPTS["outliers"]["x_r"]
        )
        y_range = (
            (min(points[:, 1]), max(points[:, 1]))
            if outliers_bounded
            else OPTS["outliers"]["y_r"]
        )
        outliers_points = outliers(
            x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
        )

        # process data for output
        if not separate:
            output = np.concatenate((points, outliers_points))
            np.random.shuffle(output)
        else:
            output = points, outliers_points
        return output
    return points
