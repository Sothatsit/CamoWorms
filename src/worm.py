"""
This file contains the class we use to represent all worms.
The worms are represented by a simple bezier curve.
"""
from __future__ import annotations
import math
from typing import Tuple, TypeAlias, cast, Optional, Union

import numpy as np
import numpy.typing as npt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import euclidean_distances

from src import rng
from src.bezier import FastBezierSegment, ConsistentBezierSegment
from src.helpers import round_to


class CamoWorm:
    """
    A worm.
    """

    def __init__(
            self, x: float, y: float, r: float, theta: float, deviation_r: float,
            deviation_gamma: float, width: float, colour: float):

        self.x: int = round(x)
        self.y: int = round(y)
        self.r: float = round_to(r, 0.5)
        self.theta: float = round_to(theta, 0.005)
        self.dr: float = round_to(deviation_r, 0.5)
        self.dgamma: float = round_to(deviation_gamma, 0.005)
        self.width: float = round_to(width, 0.5)
        self._colour: float = 0.5 if colour is None else round_to(colour, 1/255)

        e_dr = abs(self.dr)
        e_theta = self.theta + (math.pi if self.dr < 0 else 0)

        # Calculate the point offsets.
        p0 = [-self.r * np.cos(e_theta), -self.r * np.sin(e_theta)]
        p2 = [self.r * np.cos(e_theta), self.r * np.sin(e_theta)]
        p1 = [e_dr * np.cos(e_theta + self.dgamma), e_dr * np.sin(e_theta + self.dgamma)]
        bezier = FastBezierSegment(np.array([p0, p1, p2]))

        # Center x/y half-way through the curve.
        dx, dy = -bezier(np.array([0.5]))[0]

        # Shift the curve to the x/y point.
        p0 = [self.x + dx + p0[0], self.y + dy + p0[1]]
        p2 = [self.x + dx + p2[0], self.y + dy + p2[1]]
        p1 = [self.x + dx + p1[0], self.y + dy + p1[1]]
        self.bezier: ConsistentBezierSegment = ConsistentBezierSegment(np.array([p0, p1, p2]))

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour: Optional[float]):
        """ colour was being set to None way too often by accident. """
        self._colour = 0.5 if colour is None else colour

    def copy(self, *,
             x: Optional[float] = None,
             y: Optional[float] = None,
             r: Optional[float] = None,
             theta: Optional[float] = None,
             dr: Optional[float] = None,
             dgamma: Optional[float] = None,
             width: Optional[float] = None,
             colour: Optional[float] = None) -> CamoWorm:
        """ Creates a copy of this worm with any defined properties overridden. """
        return CamoWorm(
            self.x if x is None else x,
            self.y if y is None else y,
            self.r if r is None else r,
            self.theta if theta is None else theta,
            self.dr if dr is None else dr,
            self.dgamma if dgamma is None else dgamma,
            self.width if width is None else width,
            self._colour if colour is None else colour
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CamoWorm):
            return False

        return self.x == other.x and \
            self.y == other.y and \
            self.r == other.r and \
            self.theta == other.theta and \
            self.dr == other.dr and \
            self.dgamma == other.dgamma and \
            self.width == other.width and \
            self._colour == self._colour

    @staticmethod
    def random(image_shape: Tuple[int, int]) -> CamoWorm:
        (ylim, xlim) = image_shape
        midx = xlim * rng.random()
        midy = ylim * rng.random()
        r = 10 + 10 * np.abs(rng.standard_normal())
        theta = 2 * math.pi * rng.random()
        dr = 5 * rng.standard_normal()
        colour = rng.random()
        width = 4 + 2.5 * rng.standard_gamma(1)
        return CamoWorm(midx, midy, r, theta, dr, math.pi/2, width, colour)

    def control_points(self) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], self.bezier.control_points)

    def path(self) -> mpath.Path:
        # The type hint thinks that mpath.Path requires an int as its first argument...
        control_points: Union[int, np.ndarray] = self.control_points()
        return mpath.Path(control_points, [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])

    def patch(self) -> mpatches.PathPatch:
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self._colour), lw=self.width/2, capstyle='round')

    def __str__(self) -> str:
        return "CamoWorm({:.0f}, {:.0f}, {:.0f}, {:.2f}, {:.0f}, {:.2f}, {:.1f}, {:.2f})".format(
            self.x, self.y,
            self.r, self.theta,
            self.dr, self.dgamma,
            self.width, self._colour
        )


Clew: TypeAlias = list[CamoWorm]
