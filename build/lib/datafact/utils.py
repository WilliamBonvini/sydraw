import string
import os
import numpy as np


def getRanges(nSamplesPerOR: int):
    """

    :param nSamplesPerOR:
    :return:
    """
    xr = 1
    yr = 1
    rr = 1
    turn = 0
    while xr*yr*rr < nSamplesPerOR:
        if turn == 0:
            xr *= 2
        elif turn == 1:
            yr *= 2
        elif turn == 2:
            rr *= 2
            turn = -1
        turn += 1

    return [xr,yr,rr]









def checkExistenceAndCreate(path: string):
    """

    :param path: takes full path of mat file, included the name of the file
    :return:
    """
    folders = path.split('/')
    folders = folders[1:-1]  # dont' care about the '.' nor the name of the file
    path = folders[0]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/' + folders[1]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/' + folders[2]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/' + folders[3]
    if not os.path.exists(path):
        os.mkdir(path)




def convert_to_np_struct_and_shuffle(coxwain,inliers,outliers):
     """
     converts a sequence of inliers and outliers defined as pairs of points into an np structure
     in which inliers are coded as zeros and outliers are coded as ones.
     :param coxwain:
     :param inliers:
     :param outliers:
     :return:
     """
     inliers = np.array(inliers)
     rows = inliers.shape[0]
     columns = inliers.shape[1]
     n_outliers = coxwain.getNumPointsPerSample() - rows

     # set value of last column to 0 for inliers
     inliers_out = np.zeros((rows, columns + 1))
     inliers_out[:, 0:2] = inliers

     # set value of last column to 1 for outliers
     outliers_out = np.ones((n_outliers, columns + 1))
     outliers_out[:, 0:2] = outliers

     # concatenate arrays and shuffle
     model = np.concatenate((inliers_out, outliers_out), axis=0)
     np.random.shuffle(model)
     model = model.T
     return model




def convert_to_mat_struct(model):
    """

    :param model: is a numpy representation of a model coded as 2d points + their label
    :return: a triplet made of correspondences + labels, that is the kind of struct the matlab deals with
    """
    x1p = np.array([elem for elem in model[0, :]], dtype=np.float32)
    x2p = np.array([elem for elem in model[1, :]], dtype=np.float32)
    labels = np.array([elem for elem in model[2, :]], dtype=np.uint8)

    circle_mat = (x1p, x2p, labels)
    return circle_mat