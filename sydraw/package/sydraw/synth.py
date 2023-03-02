import math
import random
from typing import List, Tuple, Union

import numpy as np
from scipy.spatial.transform.rotation import Rotation
from sydraw.utils.cameraops import normalize, simCamProj, simCamTransform
from sydraw.utils.config import OPTS


def outliers(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n: int,
    homogeneous: bool = True,
):
    """
    generates outliers in a user specified 2D square

    :param x_range: floats tuple, interval of x values
    :param y_range: floats tuple, interval if y values
    :param n: number of outliers
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)] if homogeneous, else [(x_1,y_1),...,(x_np,y_np)]
    """

    if n <= 0:
        raise ValueError("parameter n should be greater than zero")
    elif x_range[0] >= x_range[1]:
        raise ValueError("invalid interval for x_range")
    elif y_range[0] >= y_range[1]:
        raise ValueError("invalid interval for y_range")

    outliers_points = []
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        if homogeneous:
            outliers_points.append((x, y, 1))
        else:
            outliers_points.append((x, y))

    return np.array(outliers_points, dtype=float)


def circle(
    radius: float,
    center: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = False,
    separate: bool = False,
    outliers_bounded: bool = True,
) -> np.ndarray:
    """
    generates circle points given radius and center.
    if outliers_perc = 0 it will return a np.array.
    otherwise:
     if shuffle=True -> will return a single shuffled array
     if shuffle=False -> will return a pair of np.arrays (circle points, outliers)

    :param radius: radius
    :param center: center (x,y)
    :param n: number of points
    :param noise_perc: gaussian noise standard deviation
    :param outliers_perc: percentage of outliers out of n data points
    :param homogeneous: bool, if True returns homogeneous coordinates, otherwise euclidean coordinates, default is False
    :param separate: bool, if True shuffles outliers with data, otherwise returns a tuple of np.arrays (inliers, outliers)
    :param outliers_bounded: bool, if True outliers are bounded within the borders of the curve, otherwise they assume
                             values defined in configuration, default is True.
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)] or pair of np.arrays
    """

    # setup
    points = []
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    # generate circle's points
    for _ in range(n_points):
        alpha = 2 * math.pi * random.random()
        x = center[0] + radius * math.cos(alpha) + np.random.normal(0, noise_perc)
        y = center[1] + radius * math.sin(alpha) + np.random.normal(0, noise_perc)
        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))
    points = np.array(points)

    # generate outliers
    if n_outliers > 0:
        x_range = (
            (center[0] - radius, center[0] + radius)
            if outliers_bounded
            else OPTS["outliers"]["x_r"]
        )
        y_range = (
            (center[1] - radius, center[1] + radius)
            if outliers_bounded
            else OPTS["outliers"]["y_r"]
        )
        outliers_points = outliers(
            x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=homogeneous
        )
        # process data for output
        if separate:
            points = points, outliers_points

        else:
            points = np.concatenate((points, outliers_points))
            np.random.shuffle(points)

    return points


def circles(
    ns: int,
    radius: float,
    center: Tuple[float, float],
    n: int,
    noise_perc: float = 0.0,
    outliers_perc: float = 0.0,
    homogeneous: bool = False,
    outliers_bounded: bool = True,
) -> np.array:
    """

    :param radius: radius
    :param center: center (x,y)
    :param n: number of points
    :param noise_perc: gaussian noise standard deviation
    :param outliers_perc: percentage of outliers out of n data points
    :param homogeneous: bool, if True returns homogeneous coordinates, otherwise euclidean coordinates, default is False
    :param outliers_bounded: bool, if True outliers are bounded within the borders of the curve, otherwise they assume
                             values defined in configuration, default is True.
    :return: np.array, shape (ns, n, num coordinates)
    """

    dim = 3 if homogeneous else 2
    samples = np.zeros((ns, n, dim))

    for i in range(ns):
        samples[i] = circle(
            radius=radius,
            center=center,
            n=n,
            noise_perc=noise_perc,
            outliers_perc=outliers_perc,
            homogeneous=homogeneous,
            separate=False,
            outliers_bounded=outliers_bounded,
        )
    return samples


def ellipse(
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
    samples point from an ellipse.
    Equation is (x-x_0)^2/a^2 + (y-y_0)^2/b^2=1
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
    a = semi_x_axis
    b = semi_y_axis
    (x_0, y_0) = center
    x_sampling_interval = (x_0 - a, x_0 + a)
    points = []
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    # generate ellipse's points
    for _ in range(n_points):
        x = random.uniform(*x_sampling_interval)
        # equation for sampling y: y = sqrt((1-((x-x_0)**2)/a**2)*b**2) + y_0
        y = math.sqrt((1 - ((x - x_0) ** 2 / a**2)) * b**2)
        y = (
            y if random.random() < 0.5 else -y
        )  # choose positive or negative solution of the square root
        y = y + y_0 + np.random.normal(0, noise_perc)
        x = x + np.random.normal(0, noise_perc)
        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))
    points = np.array(points)

    # generate outliers
    if n_outliers > 0:
        x_range = (
            (center[0] - a, center[0] + a)
            if outliers_bounded
            else OPTS["outliers"]["x_r"]
        )
        y_range = (
            (center[1] - b, center[1] + b)
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
    a = semi_x_axis
    b = semi_y_axis
    (x_0, y_0) = center
    points = []
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    # generate hyperbola's points
    for _ in range(n_points):
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
        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))
    points = np.array(points)

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
    points = []
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points
    x_vertex = -b / 2 * a
    amplitude = 2 / (0.1 + a)
    x_sampling_interval = (x_vertex - amplitude, x_vertex + amplitude)

    # generate parabola's points
    for _ in range(n_points):
        x = random.uniform(*x_sampling_interval)
        y = a * x**2 + b * x + c
        xp = x * math.cos(theta) - y * math.sin(theta) + np.random.normal(0, noise_perc)
        yp = y * math.cos(theta) + x * math.sin(theta) + np.random.normal(0, noise_perc)

        if homogeneous:
            points.append((xp, yp, 1))
        else:
            points.append((xp, yp))
    points = np.array(points)

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
    x1, y1 = point_1
    x2, y2 = point_2
    m = (y1 - y2) / (x1 - x2)
    q = y1 - m * x1
    points = []
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    # generate line's points
    for _ in range(n_points):
        x = random.uniform(point_1[0], point_2[0])
        y = m * x + q + np.random.normal(0, noise_perc)
        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))
    points = np.array(points)

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


def homography_v1(
    matrix: Union[List, np.array],
    n: int,
    src_range_x: Tuple[float, float] = (-5, 5),
    src_range_y: Tuple[float, float] = (-5, 5),
    src_range_z: Tuple[float, float] = (-5, 5),
    outliers_perc: float = 0.0,
    noise_perc: float = 0.0,
    homogeneous: bool = True,
    separate: bool = True,
):
    """
    generates points belonging to homography specified as input
    :param matrix: list 3x3 or np.array with shape (3,3)
    :param n: number of points to be drawn
    :param src_range_x:
    :param src_range_y:
    :param src_range_z:
    :param outliers_perc:
    :param noise_perc:
    :param homogeneous:
    :param separate:
    :return:
    """

    # setup
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    # generate homography points
    x1s = np.random.uniform(
        low=(src_range_x[0], src_range_y[0], src_range_z[0]),
        high=(src_range_x[1], src_range_y[1], src_range_z[1]),
        size=(n_points, 2),
    )  # todo why 2??
    x1s = x1s / x1s[:, 2]
    x2s = np.matmul(x1s, matrix)
    x2s = x2s / x2s[:, 2]
    x1s[:, 0:2] = x1s[:, 0:2] + np.random.normal(
        loc=0, scale=noise_perc, size=(n_points, 2)
    )  # add noise
    x2s[:, 0:2] = x2s[:, 0:2] + np.random.normal(
        loc=0, scale=noise_perc, size=(n_points, 2)
    )  # add noise
    if homogeneous:
        points = np.concatenate((x1s, x2s))
    else:
        points = np.concatenate((x1s[0:2], x2s[0:2]))

    # generate outliers
    if n_outliers > 0:
        outliers_points = np.random.uniform(
            low=(src_range_x[0], src_range_y[0], src_range_z[0]),
            high=(src_range_x[1], src_range_y[1], src_range_z[1]),
            size=(n_outliers, 2),
        )
        outliers_points = outliers_points / outliers_points[:, 2]

        # process data for output
        if separate:
            output = np.concatenate((points, outliers_points))
            np.random.shuffle(output)
        else:
            output = points, outliers_points
        return output
    return points


def homography(
    matrix: Union[List, np.array],
    n: int,
    src_range_x: Tuple[float, float] = (-5, 5),
    src_range_y: Tuple[float, float] = (-5, 5),
    src_range_z: Tuple[float, float] = (-5, 5),
    outliers_perc: float = 0.0,
    noise_perc: float = 0.0,
    homogeneous: bool = True,
    separate: bool = True,
):
    """
    generates points belonging to homography specified as input
    :param matrix: list 3x3 or np.array with shape (3,3)
    :param n: number of points to be drawn
    :param src_range_x:
    :param src_range_y:
    :param src_range_z:
    :param outliers_perc:
    :param noise_perc:
    :param homogeneous:
    :param separate:
    :return:
    """

    # setup
    n_points = int(n * (1 - outliers_perc))
    n_outliers = n - n_points

    random_point_cloud = np.random.uniform(
        low=(src_range_x[0], src_range_y[0], src_range_z[0]),
        high=(src_range_x[1], src_range_y[1], src_range_z[1]),
        size=(3, n_points),
    )

    # rotate point cloud
    angles = 0.8 * (0.5 - np.random.rand(1, 3))
    R = Rotation.from_euler("zyx", angles, degrees=False).as_matrix()[0]
    t = np.zeros((3, 1))

    # generate camera projection
    x1p_org, x2p_org, K = simCamProj("uncalibrated", random_point_cloud, R, t)


def homography_from_pointcloud(X: np.ndarray, matrix, rotation_threshold: float = 0.8):
    """
    starting from a 3D point clouds computes a pair of 2D projections explainable by a homography

    apply random rotation R1 to X,
    then generates 2 2D-projections of X:
    - the first  one is a projection of X rotated by Rglobal,
    - the second one is a projection obtained by further applying R and t

    :param X: 3D point cloud, np.ndarray, (3, npts)
    :return:
        x1ph: 1st projection in homogeneous coordinates, np.ndarray, (3, nps)
        x2ph: 2nd projection in homogeneous coordinates, np.ndarray, (3, nps)
        TN1:  de-normalization matrix for 1st projection supposedly, np.ndarray, (3,3)
        TN2:  de-normalization matrix for 2nd projection supposedly, np.ndarray, (3,3)
        Forg: identity, np.ndarray, (3,3)
        Fn:   identity, np.ndarray, (3,3)
        E:    identity, np.ndarray, (3,3)
        K,    intrinsic matrix, np.ndarray, (3,3)
        R,    motion matrix from view 1 to view 2, np.ndarray, (3,3)
        t,    translation vector from view 1 to view 2, np.ndarray, (3,3)
        H,    homography matrix, np.ndarray, (3,3)
        Hn,   homography matrix to be applied to normalized data, np.ndarray, (3,3)
    """
    motion = "rotation"
    camproj = "uncalibrated"
    npts = X.shape[-1]

    # randomly rotate X
    angles = 0.8 * (0.5 - np.random.rand(1, 3))
    Rglobal = Rotation.from_euler("zyx", angles, degrees=False)
    Rglobal = Rglobal.as_matrix()[0]
    X = np.matmul(Rglobal, X)

    # put in front of camera 1
    X = X + np.repeat(np.transpose(np.array([[0, 0, 2]])), npts, axis=1)
    # mix homographies
    # skipped

    # generate motion
    R, t = simCamTransform(motion, X, rotation_threshold=rotation_threshold)

    # generate camera projection
    x1p_org, x2p_org, K = simCamProj(camproj, X, R, t)

    x1p, TN1 = normalize(x1p_org, 1)
    x2p, TN2 = normalize(x2p_org, 1)

    x1ph = np.ones((3, npts))
    x1ph[0:2] = x1p
    x2ph = np.ones((3, npts))
    x2ph[0:2] = x2p

    # add noise
    noise = 10 / 512
    x1ph[0:2, :] = x1ph[0:2, :] + (2 * np.random.rand(2, npts) - 1) * noise
    x2ph[0:2, :] = x2ph[0:2, :] + (2 * np.random.rand(2, npts) - 1) * noise

    # compute homography from rotation
    H = np.matmul(K, np.matmul(R, np.linalg.inv(K)))
    Hn = np.matmul(TN2, np.matmul(np.linalg.inv(H), np.linalg.inv(TN1)))

    E = np.eye(3)
    Forg = np.eye(3)
    Fn = np.eye(3)

    return x1ph, x2ph, TN1, TN2, Forg, Fn, E, K, R, t, H, Hn
