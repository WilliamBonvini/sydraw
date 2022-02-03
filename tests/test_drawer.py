import syndalib.package.synth as dr
from tests import SyndalibTest


class Test(SyndalibTest):
    def setUp(self) -> None:
        self.o_x_range = self.o_y_range = (-1.0, 1.0)
        self.circle_radius = 3.0
        self.circle_center = (1.0, 1.0)
        self.outliers_perc = 0.2
        self.n = 10

    def test_outliers_points_homogeneous(self):
        x_range = y_range = self.o_x_range
        outliers = dr.outliers_points(x_range, y_range, n=10)
        for i in range(outliers.shape[0]):
            assert x_range[0] <= outliers[i][0] < x_range[1] and y_range[0] <= outliers[i][1] < y_range[1] and \
                   outliers[i].shape[0] == 3

    def test_outliers_points_euclidean(self):
        x_range = y_range = self.o_x_range
        outliers = dr.outliers_points(x_range, y_range, n=10, homogeneous=False)
        for i in range(outliers.shape[0]):
            assert x_range[0] <= outliers[i][0] < x_range[1] and y_range[0] <= outliers[i][1] < y_range[1] and \
                   outliers[i].shape[0] == 2

    def test_corrupted_circle_points(self):
        """
        #todo: it's not testing whether the first part of the array belongs to the circles and the second doesn't.
        :return:
        """
        output = dr.corrupted_circle_points(radius=self.circle_radius,
                                            center=self.circle_center,
                                            n=self.n,
                                            outliers_range=(self.o_x_range, self.o_y_range),
                                            outliers_perc=self.outliers_perc,
                                            )
        n_in = int(self.n*(1.0 - self.outliers_perc))
        n_out = self.n - n_in
        inliers_mask = [True if i < n_in else False for i in range(self.n)]
        outliers_mask = [False if inliers_mask[i] is True else True for i in range(self.n)]
        inliers = output[inliers_mask]
        print(inliers.shape)
        outliers = output[outliers_mask]
        assert output.shape[0] == 10 and inliers.shape[0] == n_in and outliers.shape[0] == n_out



