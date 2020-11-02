import string
from math import floor
import matplotlib.pyplot as plt
import random as rand
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
    while xr * yr * rr < nSamplesPerOR:
        if turn == 0:
            xr *= 2
        elif turn == 1:
            yr *= 2
        elif turn == 2:
            rr *= 2
            turn = -1
        turn += 1

    return [xr, yr, rr]


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


# deprecated
def convert_to_np_struct_and_shuffle(inliers, outliers):
    """
     converts a sequence of inliers and outliers defined as pairs of points into an np structure
     in which inliers are coded as zeros and outliers are coded as ones.
     :param inliers:
     :param outliers:
     :return:
     """
    inliers = np.array(inliers)
    rows = inliers.shape[0]
    columns = inliers.shape[1]
    n_outliers = 256 - rows  #TODO: HARD CODED NUMBER OF POINTS PER SAMPLE

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


def convert_to_np_struct(inliers, outliers, npps):
    """
    #todo: I'm not considering whether there are intersections betweeen models! so same point would be repeated twice
    :param inliers: a list of lists of inliers. one list for each model. each list is made of tuples (coord1,coord2,...)
    :param outliers: a list of tuples (coord1,coord2,...)
    :param npps: number of points for each sample
    :return: shuffled sample, sample
    """

    # convert inliers to np array
    inliers = np.array([np.array(model_inliers) for model_inliers in inliers])

    # obtain number of models
    num_models = inliers.shape[0]

    # count all inliers by summing the inliers in each model
    num_inliers = 0
    for i_model in range(num_models):
        num_inliers += inliers[i_model].shape[0]

    # obtain number of coordinates (in the circle case it is: 2)
    n_coordinates = inliers[0].shape[1]

    # create an array of ones that I'll modify step by step
    # it is an array of ones because I'll have that some point will be inlirs for a model but outliers (ones) for others
    inliers_out = np.ones((num_inliers, n_coordinates+num_models))
    for i_model in range(num_models):
        # find startIndex row for this model
        startIndex = 0
        for i in range(i_model):
            startIndex += inliers[i].shape[0]
        num_inliers_in_model = inliers[i_model].shape[0]
        # find endIndex row for this model
        endIndex = startIndex + num_inliers_in_model
        inliers_out[startIndex:endIndex, 0:n_coordinates] = inliers[i_model]
        inliers_out[startIndex:endIndex, n_coordinates + i_model] = np.zeros(shape=(num_inliers_in_model,))

    # set value to 1 for outliers
    num_outliers = npps - num_inliers
    outliers_out = np.ones((num_outliers, n_coordinates + num_models))
    outliers_out[:, 0:n_coordinates] = outliers

    # concatenate arrays and shuffle
    sample = np.concatenate((inliers_out, outliers_out), axis=0)

    return sample


def convert_to_mat_struct(sample):
    """

    :param model: is a numpy representation of a sample coded as 2d points + their label
    :return: a triplet made of correspondences + labels, that is the kind of struct the matlab deals with
    """
    sample = sample.T
    x1p = np.array([elem for elem in sample[0, :]], dtype=np.float32)
    x2p = np.array([elem for elem in sample[1, :]], dtype=np.float32)
    num_models = sample.shape[0]-2

    num_points_per_sample = sample.shape[1]
    labels = np.zeros(shape=(num_points_per_sample, num_models))
    for i_model in range(num_models):
        labels[:, i_model] = np.array([elem for elem in sample[2 + i_model, :]], dtype=np)
    # se dovessi avere problemi con il single model fitting forse mi basta deglutire la seconda dimensione di labels
    # oltre al commento sopra dovrei evitare di fare la seguente transpose immagino
    labels = labels.T
    circle_mat = (x1p, x2p, labels)
    return circle_mat


def compute_num_of_inliers_for_each_model(tot_num_inliers, num_of_models):
    """

    :param tot_num_inliers: total number of inliers in the whole sample
    :param num_of_models: the number of models to be drawn in each sample
    :return: a list that contains the number of inlier points for each model
    """
    num_inliers = []
    remaining = tot_num_inliers
    remaining_num_of_models = num_of_models
    while remaining_num_of_models > 0:
        currentModelNumOfInliers = floor(remaining / remaining_num_of_models)
        num_inliers.append(currentModelNumOfInliers)
        remaining -= currentModelNumOfInliers
        remaining_num_of_models -= 1
    return num_inliers


def plot_sample(inliers, outliers, imgdir):
    for model in inliers:
        plt.scatter(*zip(*model), s=2)
    plt.scatter(*zip(*outliers), s=2)
    plt.grid(color='k', linestyle=':', linewidth=1)
    plt.axes().set_aspect('equal', 'datalim')
    plt.gca().set_aspect('equal', adjustable='box')

    # save plot once in a while
    random_number = rand.randrange(10000)
    if random_number % 20 == 0:
        folder = imgdir + '/' + str(rand.randrange(10000)) + ".png"
        checkExistenceAndCreate(folder)
        plt.savefig(folder)

    # show it
    plt.show()
