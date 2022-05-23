"""
A re-implementation of the matplotlib BezierSegment to make it faster.
"""
import math
from typing import Final

import numba
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


@numba.njit
def create_consistency_map(points: np.ndarray) -> tuple[np.ndarray, float]:
    diffs = points[1:] - points[:-1]
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    c_map = np.empty((len(points),), dtype=dists.dtype)
    c_map[0] = 0
    c_map[1:] = np.cumsum(dists)
    length = c_map[-1]
    return c_map / length, length


class ConsistentBezierSegment:
    """
    Provides an approximate mapping of t-values
    to make them linear in arc length.
    """
    mapping_t_values: np.ndarray = np.linspace(0, 1, num=8)

    def __init__(self, control_points: np.ndarray, *, mapping_points=10):
        self.control_points = control_points
        self._bezier = FastBezierSegment(control_points)
        points = self._bezier(ConsistentBezierSegment.mapping_t_values)
        self._map, self.length = create_consistency_map(points)
        # diffs = np.diff(points, axis=0)
        # dists = np.hypot(diffs[:, 0], diffs[:, 1])
        # self._map = np.empty((mapping_points,), dtype=dists.dtype)
        # self._map[0] = 0
        # np.cumsum(dists, out=self._map[1:])
        # self.length = self._map[len(self._map) - 1]
        # np.divide(self._map, self.length, out=self._map)

    def __call__(self, t: np.ndarray):
        t = np.interp(t, self._map, ConsistentBezierSegment.mapping_t_values)
        return self._bezier(t)
