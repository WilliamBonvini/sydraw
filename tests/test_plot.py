from syndalib import plot as sypl
import matplotlib.pyplot as plt
from tests import SyndalibTest


class Test(SyndalibTest):
    def setUp(self) -> None:
        self.circle_coefs = [1, 0, 1, 0, 0, -4]
        self.ellipse_coefs = [1, 0, 4, 0, 0, -4]
        self.parabola_y_coefs = [1, 0, 0, 0, -1, 0]  # y = x^2
        self.parabola_x_coefs = [0, 0, 1, -1, 0, 0]  # x = y^2
        self.parabola_xy_coefs = [0.05, 0.5, 0, 0, -1, 0]
        self.hyperbola_coefs = [0, 1, 0, 0, 0, 2]

    def test_plot_conic_curve_circle(self):
        sypl.plot_conic_curve(self.circle_coefs, resolution=1000)
        plt.show()
        assert True

    def test_plot_conic_curve_ellipse(self):
        sypl.plot_conic_curve(self.ellipse_coefs)
        assert True

    def test_plot_conic_curve_parabola_x(self):
        sypl.plot_conic_curve(self.parabola_x_coefs)
        assert True

    def test_plot_conic_curve_parabola_y(self):
        sypl.plot_conic_curve(self.parabola_y_coefs)
        assert True

    def test_plot_conic_curve_parabola_xy(self):
        sypl.plot_conic_curve(self.parabola_xy_coefs)
        # not working
        assert False

    def test_plot_conic_curve_hyperbola(self):
        sypl.plot_conic_curve(self.hyperbola_coefs)
        assert False





