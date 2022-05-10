"""
This file contains the class we use to represent all worms.
The worms are represented by a simple bezier curve.
"""
import math
from typing import Tuple, TypeAlias

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
from sklearn.metrics.pairwise import euclidean_distances

from src import rng


class CamoWorm:
    """
    A worm.
    """
    def __init__(
            self, x: float, y: float, r: float, theta: float, deviation_r: float,
            deviation_gamma: float, width: float, colour: float):

        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.colour = colour

        e_dr = abs(self.dr)
        e_theta = self.theta + (math.pi if self.dr < 0 else 0)

        dx = -0.1 * self.r * np.cos(e_theta)
        dy = -0.1 * self.r * np.sin(e_theta)

        p0 = [self.x + dx - self.r * np.cos(e_theta),
              self.y + dy - self.r * np.sin(e_theta)]
        p2 = [self.x + dx + self.r * np.cos(e_theta),
              self.y + dy + self.r * np.sin(e_theta)]
        p1 = [self.x + dx + e_dr * np.cos(e_theta+self.dgamma),
              self.y + dy + e_dr * np.sin(e_theta+self.dgamma)]

        self.bezier = mbezier.BezierSegment(np.array([p0, p1, p2]))

    def copy(self, *, x=None, y=None, r=None, theta=None, dr=None, dgamma=None, width=None, colour=None):
        """ Creates a copy of this worm with any defined properties overridden. """
        return CamoWorm(
            self.x if x is None else x,
            self.y if y is None else y,
            self.r if r is None else r,
            self.theta if theta is None else theta,
            self.dr if dr is None else dr,
            self.dgamma if dgamma is None else dgamma,
            self.width if width is None else width,
            self.colour if colour is None else colour
        )

    @staticmethod
    def random(image_shape: Tuple[int, int]) -> 'CamoWorm':
        (ylim, xlim) = image_shape
        midx = xlim * rng.random()
        midy = ylim * rng.random()
        r = 10 + 10 * np.abs(rng.standard_normal())
        theta = math.pi * rng.random()
        dr = 5 * rng.standard_normal()
        dgamma = rng.random() * np.pi
        colour = rng.random()
        width = 4 + 2 * rng.standard_gamma(1)
        return CamoWorm(midx, midy, r, theta, dr, dgamma, width, colour)

    def control_points(self):
        return self.bezier.control_points

    def path(self):
        return mpath.Path(self.control_points(), [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])

    def patch(self):
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.colour), lw=self.width/2, capstyle='round')

    def intermediate_points(self, intervals: int = None):
        if intervals is None:
            intervals = max(3, int(np.ceil(self.r/8)))
        return self.bezier.point_at_t(np.linspace(0, 1, intervals))

    def approx_length(self):
        intermediates = self.intermediate_points()
        eds = euclidean_distances(intermediates, intermediates)
        return np.sum(np.diag(eds, 1))

    def colour_at_t(self, t, image):
        intermediates = np.round(
            np.array(self.bezier.point_at_t(t)).reshape(-1, 2)).astype(np.int64)
        colours = [image[point[0], point[1]] for point in intermediates]
        return np.array(colours) / 255.0

    def __str__(self):
        return "CamoWorm({:.0f}, {:.0f}, {:.0f}, {:.2f}, {:.0f}, {:.2f}, {:.1f}, {:.2f})".format(
            self.x, self.y,
            self.r, self.theta,
            self.dr, self.dgamma,
            self.width, self.colour
        )


Clew: TypeAlias = list[CamoWorm]
