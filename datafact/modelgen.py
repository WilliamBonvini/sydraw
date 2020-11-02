import string
import os

from datafact.makers.circle_maker import CircleMaker
from datafact.makers.conic_maker import ConicMaker
from datafact.makers.ellipse_maker import EllipseMaker
from datafact.makers.line_maker import LineMaker
from datafact.makers.maker import Maker
from datafact.utils.coxswain import Coxwain

coxwain = Coxwain(os.getcwd())

def set_train_dir(path: string):
    coxwain.setTrainDir(path)


def set_test_dir(path: string):
    coxwain.setTestDir(path)


def start():
    Maker.NUM_SAMPLES = Coxwain.NUM_SAMPLES
    Maker.NUM_POINTS_PER_SAMPLE = Coxwain.NUM_POINTS_PER_SAMPLE
    Maker.OUTLIERS_PERC_RANGE = Coxwain.OUTLIERS_PERC_RANGE
    Maker.MODEL = Coxwain.MODEL
    Maker.NOISE_PERC_RANGE = Coxwain.NOISE_PERC_RANGE
    Maker.DEST = Coxwain.DEST
    Maker.BASE_DIR = Coxwain.BASE_DIR
    Maker.TRAIN_DIR = Coxwain.TRAIN_DIR
    Maker.TEST_DIR = Coxwain.TEST_DIR
    Maker.IMG_BASE_DIR = Coxwain.IMG_BASE_DIR
    Maker.NUMBER_OF_MODELS = Coxwain.NUMBER_OF_MODELS
    maker = None
    if Coxwain.MODEL == 'circles':
        maker = CircleMaker()
    elif Coxwain.MODEL == 'lines':
        maker = LineMaker()
    elif Coxwain.MODEL == 'ellipses':
        maker = EllipseMaker()
    elif Coxwain.MODEL == "conics":
        maker = ConicMaker()
    if maker is not None:
        maker.start()
    else:
        print("Incorrect name for \"model\"")


def generate_data(num_samples: int,
                  num_points_per_sample: int,
                  outliers_perc_range,
                  model: string,
                  noise_perc_range,
                  dest: string = 'matlab',
                  dirs=None,
                  number_of_models=1):
    """
    :param num_samples: total number of samples to be generated for each outlier rate
    :param num_points_per_sample: total number of points within each sample
    :param outliers_perc_range: list that contains outliers rates
    :param model: a string that identifies the kind of model to be created. the possible models are: 'circles','lines','ellipses','conics'
    :param noise_perc_range: a list with the stddevs of the gaussian noise to be added to the inliers
    :param dest: you want to read it with? 'matlab', 'numpy'
    :param dirs: list containing path to train data dir and to test data dir
    :param number_of_models: the number of models to be randomly sampled for each sample in the dataset, the default number is 1
    :return: a nice looking np array
    """
    Coxwain.NUM_SAMPLES = num_samples
    Coxwain.NUM_POINTS_PER_SAMPLE = num_points_per_sample
    Coxwain.OUTLIERS_PERC_RANGE = outliers_perc_range
    Coxwain.MODEL = model
    Coxwain.NOISE_PERC_RANGE = noise_perc_range
    Coxwain.DEST = dest
    Coxwain.TRAIN_DIR = dirs[1]
    Coxwain.TEST_DIR = dirs[2]
    Coxwain.NUMBER_OF_MODELS = number_of_models

    # create basedir
    basedir = model
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    basedir += '/' + str(number_of_models)
    if not (os.path.exists(basedir)):
        os.mkdir(basedir)
    basedir += '/' + dirs[0]
    if not (os.path.exists(basedir)):
        os.mkdir(basedir)
    basedir += '/' + str(num_samples)
    if not (os.path.exists(basedir)):
        os.mkdir(basedir)

    Coxwain.BASE_DIR = basedir

    # create img dirs
    imgbasedir = Coxwain.BASE_DIR
    if Coxwain.TRAIN_DIR != '':
        imgbasedir += '/'+'train'
    if Coxwain.TEST_DIR != '':
        imgbasedir += '/'+'test'
    if not os.path.exists(imgbasedir):
        os.mkdir(imgbasedir)
    imgbasedir += '/' + 'imgs'
    if not os.path.exists(imgbasedir):
        os.mkdir(imgbasedir)

    Coxwain.IMG_BASE_DIR = imgbasedir

    start()
