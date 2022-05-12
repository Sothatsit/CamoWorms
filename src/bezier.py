"""
A re-implementation of the matplotlib BezierSegment to make it faster.
"""
import math
from typing import Final

import numpy as np


class FastBezierSegment:
    """
    A 2-dimensional Bezier segment with 3 control points.

    Parameters
    ----------
    control_points : (N, d) array
        Location of the *N* control points.
    """
    _orders: Final[np.ndarray] = np.arange(3)
    _reversed_orders: Final[np.ndarray] = np.arange(3)[::-1]
    _coeff: Final[np.ndarray] = np.array([
        math.factorial(3 - 1) // (math.factorial(i) * math.factorial(3 - 1 - i)) for i in range(3)
    ], dtype=np.float)

    def __init__(self, control_points: np.ndarray):
        point_count, dims = control_points.shape
        if point_count != 3:
            raise Exception("Expected exactly 3 points, not {}".format(point_count))
        if dims != 2:
            raise Exception("Expected exactly 2 dimensions, not {}".format(dims))

        self.control_points = control_points
        self._matrix = (control_points.T * FastBezierSegment._coeff).T

    def __call__(self, t: np.ndarray):
        """
        Evaluate the Bezier curve at point(s) t in [0, 1].

        Parameters
        ----------
        t : (k,) array-like
            Points at which to evaluate the curve.

        Returns
        -------
        (k, 2) arrays
            Value of the curve for each point in *t*.
        """
        return (np.power.outer(1 - t, FastBezierSegment._reversed_orders)
                * np.power.outer(t, FastBezierSegment._orders)) @ self._matrix
