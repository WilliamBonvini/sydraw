from unittest import TestCase

import numpy as np

from sydraw import synth


class Test(TestCase):
    def setUp(self) -> None:

        self.n = 20
        self.center = (2, 2)
        self.x_range = (-1, 1)
        self.y_range = (3, 5)
        self.outliers_perc = 0.20

        # circle specific arguments
        self.radius = 3

        # ellipse specific arguments
        self.semi_x_axis = 4.0
        self.semi_y_axis = 2.5

        # line specific arguments
        self.point_1 = (-3, -4)
        self.point_2 = (5, 8)

        # parabola specific arguments
        self.a = 2.0
        self.b = 1.0
        self.c = -2.0

        self.synth_foos = [
            synth.circle,
            synth.ellipse,
            synth.parabola,
            synth.line,
        ]

        # positional arguments
        self.circle_args = {"radius": self.radius, "center": self.center, "n": self.n}
        self.ellipse_args = {
            "semi_x_axis": self.semi_x_axis,
            "semi_y_axis": self.semi_y_axis,
            "center": self.center,
            "n": self.n,
        }

        self.line_args = {"point_1": self.point_1, "point_2": self.point_2, "n": self.n}
        self.outliers_args = {
            "x_range": self.x_range,
            "y_range": self.y_range,
            "n": self.n,
        }
        self.parabola_args = {"a": self.a, "b": self.b, "c": self.c, "n": self.n}

        # keyword arguments
        self.args_without_outliers = {"outliers_perc": 0}
        self.args_with_outliers_and_separate_true = {
            "outliers_perc": self.outliers_perc,
            "separate": True,
        }
        self.args_with_outliers_and_separate_false = {
            "outliers_perc": self.outliers_perc,
            "separate": False,
        }

    def _retrieve_positional_args(self, foo):
        if foo.__name__ == "circle":
            return self.circle_args
        elif foo.__name__ == "ellipse":
            return self.ellipse_args
        elif foo.__name__ == "outliers":
            return (self.outliers_args,)
        elif foo.__name__ == "parabola":
            return self.parabola_args
        elif foo.__name__ == "line":
            return self.line_args
        else:
            raise ValueError("Invalid class passed")

    def _output_shapes_are_correct(self, foo):
        """
        return True if shapes of the output of the passed function are coherent with function's expected behavior
        false otherwise
        """
        numpoints = self.n

        def _is_shape_single_output_valid(foo, **kwargs):
            for homogeneous in [False, True]:
                output = foo(**kwargs, homogeneous=homogeneous)
                n_coords = 3 if homogeneous else 2
                if output.shape != (numpoints, n_coords):
                    return False
            return True

        def _are_shapes_pair_output_valid(foo, **kwargs):
            for homogeneous in [False, True]:
                outputs = foo(**kwargs, homogeneous=homogeneous)
                n_coords = 3 if homogeneous else 2
                transformation_points = outputs[0]
                outliers_points = outputs[1]
                if (
                    transformation_points.shape[0] + outliers_points.shape[0]
                    != numpoints
                    or transformation_points.shape[1] != n_coords
                    or outliers_points.shape[1] != n_coords
                ):
                    return False
            return True

        single_output_scenarios = [
            self.args_with_outliers_and_separate_false,
            self.args_without_outliers,
        ]

        for scenario in single_output_scenarios:
            if not _is_shape_single_output_valid(
                foo, **self._retrieve_positional_args(foo), **scenario
            ):
                print(
                    "Shape inconsistency in function {}, scenario: {}".format(
                        foo.__name__, scenario
                    )
                )
                return False

        # Separate output scenario

        if not _are_shapes_pair_output_valid(
            foo,
            **self._retrieve_positional_args(foo),
            **self.args_with_outliers_and_separate_true,
        ):
            print(
                "Shape inconsistency in function {}, scenario: {}".format(
                    foo.__name__, self.args_with_outliers_and_separate_true
                )
            )
            return False

        return True

    """"***************************************************************************************************************
                                                        output shapes 
    ****************************************************************************************************************"""

    def test_output_shapes(self):
        for foo in self.synth_foos:
            try:
                print(f'\nTesting output shapes for function "{foo.__name__}"')
                if not self._output_shapes_are_correct(foo):
                    self.fail()
            except:
                raise Exception("Exception in function {}".format(foo.__name__))

    """"***************************************************************************************************************
                                                    OUTLIERS 
    ****************************************************************************************************************"""

    def test_outliers_behavior(self):
        """assert that all outliers are within the specified range"""
        output = synth.outliers(**self.outliers_args)
        xs = output[:, 0]
        ys = output[:, 1]
        self.assertTrue(
            all(xs >= self.x_range[0])
            and all(xs <= self.x_range[1])
            and all(ys >= self.y_range[0])
            and all(ys <= self.y_range[1])
        )

    def test_outliers_output_shape(self):
        """assert that the output shape of the outliers is correct"""
        for homogeneous in [False, True]:
            output = synth.outliers(
                x_range=self.x_range,
                y_range=self.y_range,
                n=self.n,
                homogeneous=homogeneous,
            )
            n_coords = 3 if homogeneous else 2
            if output.shape != (self.n, n_coords):
                return False
        return True

    def test_outliers_invalid_input_n(self):
        """assert that a value error is thrown if n <= 0"""
        args = self.outliers_args
        args["n"] = 0
        self.assertRaises(ValueError, synth.outliers, **args)

    def test_outliers_invalid_input_x_range(self):
        """assert that a value error is thrown if x_range interval is not valid"""
        args = self.outliers_args
        args["x_range"] = (2, 1)
        self.assertRaises(ValueError, synth.outliers, **args)

    def test_outliers_invalid_input_y_range(self):
        """assert that a value error is thrown if y_range interval is not valid"""
        args = self.outliers_args
        args["x_range"] = (2, 1)
        self.assertRaises(ValueError, synth.outliers, **args)

    """"***************************************************************************************************************
                                                    CIRCLE 
    ****************************************************************************************************************"""

    def _all_points_are_in_circle(self, points):
        """returns whether all points satisfy (x - h)^2 + (y - k)^2 = r^2"""
        xs = points[:, 0]
        ys = points[:, 1]
        res = np.allclose(
            (xs - self.center[0]) ** 2 + (ys - self.center[1]) ** 2,
            np.repeat(self.radius**2, xs.shape[0]),
        )
        return res

    def test_circle_separate_true_and_outliers(self):
        """assert that not all points are explained by the circle equation"""
        points = synth.circle(
            **self.circle_args, **self.args_with_outliers_and_separate_true
        )
        points, _ = points
        self.assertTrue(self._all_points_are_in_circle(points))

    def test_circle_separate_false(self):
        """assert that all points are explained by the circle equation"""
        points = synth.circle(**self.circle_args, separate=False)
        self.assertTrue(self._all_points_are_in_circle(points))

    """"***************************************************************************************************************
                                                    ELLIPSE 
    ****************************************************************************************************************"""

    def _all_points_are_in_ellipse(self, points):
        """return whether all points satisfy (x-x_0)^2/a^2 + (y-y_0)^2/b^2 = 1"""
        xs = points[:, 0]
        ys = points[:, 1]
        a = self.semi_x_axis
        b = self.semi_y_axis
        x_0, y_0 = self.center
        res = np.allclose(
            (xs - x_0) ** 2 / a**2 + (ys - y_0) ** 2 / b**2, np.ones(xs.shape[0])
        )
        return res

    def test_ellipse_separate_true_and_outliers(self):
        """assert that not all points are explained by the circle equation"""
        points, _ = synth.ellipse(
            **self.ellipse_args, **self.args_with_outliers_and_separate_true
        )
        self.assertTrue(self._all_points_are_in_ellipse(points))

    def test_ellipse_separate_false(self):
        """assert that all points are explained by the circle equation"""
        points = synth.ellipse(**self.ellipse_args, separate=False)
        self.assertTrue(self._all_points_are_in_ellipse(points))

    """"***************************************************************************************************************
                                                           LINE 
    ****************************************************************************************************************"""

    def _all_points_are_in_line(self, points):
        """return whether al points satisfy (x-x_0)^2/a^2 - (y-y_0)^2/b^2 = 1"""
        xs = points[:, 0]
        ys = points[:, 1]
        x1, y1 = self.point_1
        x2, y2 = self.point_2
        m = (y1 - y2) / (x1 - x2)
        q = y1 - m * x1
        res = np.allclose(m * xs + np.repeat(q, points.shape[0]), ys)
        return res

    def test_line_separate_true_and_outliers(self):
        """assert that not all points are explained by the circle equation"""
        points, _ = synth.line(
            **self.line_args, **self.args_with_outliers_and_separate_true
        )
        self.assertTrue(self._all_points_are_in_line(points))

    def test_line_separate_false(self):
        """assert that all points are explained by the circle equation"""
        points = synth.line(**self.line_args, separate=False)
        self.assertTrue(self._all_points_are_in_line(points))

    """"***************************************************************************************************************
                                                    PARABOLA 
    ****************************************************************************************************************"""

    def _all_points_are_in_parabola(self, points):
        """return whether al points satisfy (x-x_0)^2/a^2 - (y-y_0)^2/b^2 = 1"""
        xs = points[:, 0]
        ys = points[:, 1]
        res = np.allclose(
            self.a * xs**2 + self.b * xs + np.repeat(self.c, xs.shape[0]), ys
        )
        return res

    def test_parabola_separate_true_and_outliers(self):
        """assert that not all points are explained by the circle equation"""
        points, _ = synth.parabola(
            **self.parabola_args, **self.args_with_outliers_and_separate_true
        )
        self.assertTrue(self._all_points_are_in_parabola(points))

    def test_parabola_separate_false(self):
        """assert that all points are explained by the circle equation"""
        points = synth.parabola(**self.parabola_args, separate=False)
        self.assertTrue(self._all_points_are_in_parabola(points))
