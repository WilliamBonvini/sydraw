from syndalib.package.makers import Maker
from syndalib.package.utils.utils import convert_to_np_struct
import numpy as np


def test_convert_to_np_struct():
    Maker.NUM_POINTS_PER_SAMPLE = 10
    inliers_model_1 = [(3, 5), (2, 8), (3, 1)]
    inliers_model_2 = [(2, 9), (1, 1), (4, 8), (1, 0)]
    inliers_model_list = [inliers_model_1, inliers_model_2]
    outliers_list = [(4, 4), (2, 1), (3, 9)]

    target = np.array(
        [
            [3, 5, 0, 1],
            [2, 8, 0, 1],
            [3, 1, 0, 1],
            [2, 9, 1, 0],
            [1, 1, 1, 0],
            [4, 8, 1, 0],
            [1, 0, 1, 0],
            [4, 4, 1, 1],
            [2, 1, 1, 1],
            [3, 9, 1, 1],
        ]
    )
    """
    target = np.array([[3., 2., 3., 2., 1., 4., 1., 4., 2., 3.],
                       [5., 8., 1., 9., 1., 8., 0., 4., 1., 9.],
                       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 0., 0., 0., 0., 1., 1., 1.]])
    """
    result = convert_to_np_struct(inliers_model_list, outliers_list)
    assert (result == target).all()


def test_shuffler():
    Maker.NUM_POINTS_PER_SAMPLE = 10
    inliers_model_1 = [(3, 5), (2, 8), (3, 1)]
    inliers_model_2 = [(2, 9), (1, 1), (4, 8), (1, 0)]
    inliers_model_list = [inliers_model_1, inliers_model_2]
    outliers_list = [(4, 4), (2, 1), (3, 9)]

    result = convert_to_np_struct(inliers_model_list, outliers_list)

    print("before shuffling:")
    print(result)

    np.random.shuffle(result)
    print("after shuffling:")
    print(result)
    assert 1 == 1
