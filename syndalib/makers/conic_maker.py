import numpy as np
from syndalib.makers.circle_maker import CircleMaker
from syndalib.makers.ellipse_maker import EllipseMaker
from syndalib.makers.line_maker import LineMaker
from syndalib.makers.maker import Maker


class ConicMaker(Maker):
    """
    Maker implementation for conics sampled from a 2D space [x_min,x_max] x [y_min, y_max]
    you can define the 2D space by calling the function syndalib.syn2d(...).
    A conic so far is randomly sampled among circles, lines and ellipses.
    """
    def __init__(self):
        Maker.MODEL = "conics"
        self.circle_maker = CircleMaker()
        self.line_maker = LineMaker()
        self.ellipse_maker = EllipseMaker()

        super(ConicMaker, self).__init__()

    def generate_random_model(self, n_inliers):
        rand_num = np.random.randint(0, 3)

        if rand_num == 0:
            return self.circle_maker.generate_random_model(n_inliers)
        if rand_num == 1:
            return self.line_maker.generate_random_model(n_inliers)
        if rand_num == 2:
            return self.ellipse_maker.generate_random_model(n_inliers)

    def point(self):
        pass
