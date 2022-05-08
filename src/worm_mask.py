import math
from typing import Optional, Final

import numpy as np

from src import NPImage
from src.worm import CamoWorm


def filter_out_close_points(points, *, point_interval: float = 6):
    """
    Takes in a numpy array of points, and filters out
    all the points that are closer than point_interval
    together. The exception to this is that the first
    and last points are always kept.
    """
    if point_interval <= 0:
        return points

    # 1. Calculate the distances between adjacent points.
    n_points = points.shape[0]
    offsets = points[:-1] - points[1:]
    distances = np.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2)

    # 2. Adjust point_interval to evenly space the points.
    total_distance = np.sum(distances)
    point_interval = total_distance / math.ceil(total_distance / point_interval)

    # 3. Create an array to use as a filter for the points.
    point_filter = np.zeros(n_points, dtype=bool)  # An array of which points to keep.
    point_filter[0] = True  # The first and last points should always be kept.
    point_filter[n_points - 1] = True

    # 4. Fill in the point filter.
    curr_dist = 0.0
    for index in range(1, n_points):
        curr_dist += distances[index - 1]
        if curr_dist >= point_interval:
            point_filter[index] = True
            curr_dist = 0

    # 5. Apply the filter
    return points[np.where(point_filter)]


class CircleMask:
    """
    Contains a circular image of distances from a point
    that can be used to create a circular mask.
    """
    cache: list[Optional['CircleMask']] = []

    def __init__(self, radius: float):
        self.radius = radius

        # The radius may be fractional, but we need whole pixels.
        self.width = math.ceil(radius) * 2
        self.offset_x = -math.ceil(radius)
        self.offset_y = -math.ceil(radius)

        # Calculate a grid of distances to the center of the mask.
        xx, yy = np.mgrid[
            self.offset_x:(self.offset_x + self.width),
            self.offset_y:(self.offset_y + self.width)
        ]

        # Pixel-by-pixel distances.
        self.distances = xx**2 + yy**2

    def _get_overlapping_areas(self, target_distances: NPImage, x: float, y: float):
        """
        Returns a tuple of (target area, source area).
        """
        # We don't care that much about sub-pixel accuracy.
        x = round(x)
        y = round(y)

        # Calculate the bounds in the target mask to paste the circle.
        target_from_x = max(0, x + self.offset_x)
        target_from_y = max(0, y + self.offset_y)
        target_to_x = min(target_distances.shape[0] - 1, x + self.offset_x + self.width)
        target_to_y = min(target_distances.shape[1] - 1, y + self.offset_y + self.width)
        if target_from_x >= target_to_x or target_from_y >= target_to_y:
            return None, None

        # Calculate the bounds in the circle mask to paste.
        circle_from_x = target_from_x - x - self.offset_x
        circle_from_y = target_from_y - y - self.offset_y
        circle_to_x = target_to_x - x - self.offset_x
        circle_to_y = target_to_y - y - self.offset_y

        # Add the circle mask into the target mask.
        target_area = target_distances[target_from_x:target_to_x, target_from_y:target_to_y]
        source_area = self.distances[circle_from_x:circle_to_x, circle_from_y:circle_to_y]
        return target_area, source_area

    def apply(self, target_distances: NPImage, x: float, y: float):
        """
        Draws this circle into the target distance image with its center at (x, y).
        """
        target_area, source_area = self._get_overlapping_areas(target_distances, x, y)
        if target_area is None or source_area is None:
            return

        np.minimum(target_area, source_area, out=target_area)

    def remove_from_mask(self, target_mask: NPImage, x: float, y: float):
        """
        Erases this circle from the target mask with its center at (x, y).
        """
        target_area, source_area = self._get_overlapping_areas(target_mask, x, y)
        if target_area is None or source_area is None:
            return

        source_mask = source_area < self.radius**2
        np.logical_not(target_area, out=target_area)
        np.logical_or(target_area, source_mask, out=target_area)
        np.logical_not(target_area, out=target_area)

    @staticmethod
    def get_cached(radius: float) -> 'CircleMask':
        """
        Tries to find a cached circle that is close in size to radius.
        If one can not be found, it creates it.
        """
        # Quantize the radius into 0.25 pixel steps.
        quantized = round(radius * 4)

        # Find if we have cached a circle already.
        cache = CircleMask.cache
        if quantized < len(cache):
            cached = cache[quantized]
            if cached is not None:
                return cached

        # Add in None to fill out the cache size.
        if quantized >= len(cache):
            for i in range(len(cache), quantized + 1):
                cache.append(None)

        # Create the circle mask and cache it.
        mask = CircleMask(quantized / 4)
        cache[quantized] = mask
        return mask


class WormMask:
    """
    Contains a mask of a worm in an image.
    """
    FAR_DISTANCE: Final[float] = 1e12
    OUTER_MAX_DIST: Final[int] = 10

    def __init__(self, worm: CamoWorm, image: NPImage, *, copy: 'WormMask' = None):
        self.worm = worm
        self.image = image

        if copy is not None:
            self.min_x = copy.min_x
            self.min_y = copy.min_y
            self.max_x = copy.max_x
            self.max_y = copy.max_y
            self.width = copy.width
            self.height = copy.height
            self.distances = copy.distances
            self.mask = copy.mask
            self.area = copy.area
        else:
            # 1. Calculate some information about the worm
            img_width, img_height = image.shape
            radius = worm.width / 2 + 0.5
            n_points_estimate = math.ceil(2 * worm.r + 2 * abs(worm.dr))
            self.points = worm.bezier(np.linspace(0, 1, num=n_points_estimate))

            # The padding is needed for the radius of th circle and the outer mask calculation.
            padding = math.ceil(radius * 1.5) + WormMask.OUTER_MAX_DIST
            self.min_x = max(0, math.floor(np.amin(self.points[:, 1])) - padding)
            self.min_y = max(0, math.floor(np.amin(self.points[:, 0])) - padding)
            max_x = min(img_width, math.ceil(np.amax(self.points[:, 1])) + padding)
            max_y = min(img_height, math.ceil(np.amax(self.points[:, 0])) + padding)
            width = max_x - self.min_x
            height = max_y - self.min_y

            # If the worm is outside the image, then we just use a dummy empty 1x1 mask.
            if width <= 0 or height <= 0:
                self.min_x = 0
                self.min_y = 0
                self.distances = np.full((1, 1), WormMask.FAR_DISTANCE, dtype=np.float64)
            else:
                # 2. Some points are really close together, so we can filter some of them out for speed.
                self.points = filter_out_close_points(self.points, point_interval=2)

                # 3. Apply a circle mask at each point on the curve to the mask.
                self.distances = np.full((width, height), WormMask.FAR_DISTANCE, dtype=np.float64)
                circle_mask = CircleMask.get_cached(math.ceil(radius) + WormMask.OUTER_MAX_DIST)
                for y, x in self.points:
                    circle_mask.apply(self.distances, x - self.min_x, y - self.min_y)

            self.recalculate_mask()
            self.width = self.mask.shape[0]
            self.height = self.mask.shape[1]
            self.max_x = self.min_x + self.width
            self.max_y = self.min_y + self.height

    @property
    def radius(self) -> float:
        return self.worm.width / 2.0

    def copy(self) -> 'WormMask':
        """ Generates a copy of this worm. """
        return WormMask(self.worm, self.image, copy=self)

    def recalculate_mask(self):
        """ Re-calculates the mask based upon the worm associated with this mask. """
        self.mask = self.distances < self.radius**2
        self.area = float(np.sum(self.mask))

    def create_outer_mask(self, *, dest: 'WormMask' = None) -> 'WormMask':
        """
        Generates a mask encompassing the sides of this mask.
        """
        distance = min(float(WormMask.OUTER_MAX_DIST), max(1.0, self.worm.width / 2.0))

        # Create the new mask.
        if dest is None:
            dest = self.copy()
        dest.distances = self.distances.copy()

        # Update the mask to its outer edge.
        outer_radius = self.worm.width / 2 + distance
        dest.mask = (dest.distances < (outer_radius ** 2)) & (~self.mask)
        dest.area = np.sum(dest.mask)

        # Remove the outer mask around the ends of the worm.
        circle_mask = CircleMask.get_cached(2 * distance)
        for end, one_off_end in [(self.points[0], self.points[1]), (self.points[-1], self.points[-2])]:
            # Calculate the direction away from the end of the worm.
            vec = end - one_off_end
            unit_vec = vec / np.linalg.norm(vec)
            forward_vec = unit_vec * (distance / 2.0 + self.radius)
            left_vec = np.array([-unit_vec[1], unit_vec[0]]) * self.radius
            right_vec = -left_vec

            # Find three points off the end of the worm.
            pt_1 = end + forward_vec
            pt_2 = pt_1 + left_vec
            pt_3 = pt_1 + right_vec

            # Draw black circles around the points.
            for y, x in [pt_1, pt_2, pt_3]:
                circle_mask.remove_from_mask(dest.mask, x - self.min_x, y - self.min_y)

        return dest

    def draw_into(self, image, colour):
        """
        Draws this worm mask into the given image with the given colour.
        """
        # 1. Get the subsection that corresponds to the mask.
        image_subsection = image[self.min_x:self.max_x, self.min_y:self.max_y]

        # 2. Black-out the area under the mask.
        np.subtract(image_subsection, image_subsection * self.mask, image_subsection)

        # 3. Add in the colour for the masked area.
        np.add(image_subsection, colour * self.mask, image_subsection)

    def image_within_bounds(self, image=None):
        """
        Returns the image under the mask with the pixels
        not within the mask being marked as 0.
        """
        image = self.image if image is None else image

        # Get the subset of the image corresponding to this mask.
        return image[self.min_x:self.max_x, self.min_y:self.max_y] / 255.0

    def image_under_mask(self, image=None):
        """
        Returns the image under the mask with the pixels
        not within the mask being marked as 0.
        """
        # Apply the mask to the image.
        return self.image_within_bounds(image) * self.mask

    def difference_image(self, image=None):
        """
        This applies this mask to the image, to calculate the differences
        between the colour of the worm and all the pixels beneath the worm.
        """
        # 1. Calculate the difference of the colour of the worm to its background.
        diff_image = np.absolute(self.image_under_mask(image) - self.worm.colour)

        # 2. Apply the mask to the difference.
        return diff_image * self.mask

    def mean_colour(self, image=None):
        """
        Returns the mean colour of all pixels underneath this mask in the image.
        """
        if self.area == 0:
            return None

        return np.sum(self.image_under_mask(image)) / self.area

    def mode_colour(self, image=None):
        """
        Returns the most common colour of all pixels underneath this mask in the image.
        """
        if self.area == 0:
            return None

        # We want 0 to represent that a pixel is not under the worm.
        sub_image = self.image_within_bounds(image) + 1
        sub_image = sub_image * self.mask

        colours = np.sort(sub_image.flatten())
        first_index = np.flatnonzero(colours > 0)[0]
        mode_index = math.floor((first_index + len(colours) - 1) / 2.0)
        return colours[mode_index] - 1

    def subsection(self, min_x, min_y, max_x, max_y):
        """
        Gets the subsection of this mask that covers the given coordinates.
        """
        if max_x <= self.min_x or max_y <= self.min_y:
            return None
        if min_x > self.max_x or min_y > self.max_y:
            return None

        min_x = max(self.min_x, min_x)
        min_y = max(self.min_y, min_y)
        max_x = min(self.max_x, max_x)
        max_y = min(self.max_y, max_y)
        return self.mask[
            (min_x - self.min_x):(max_x - self.min_x),
            (min_y - self.min_y):(max_y - self.min_y)
        ]

    def intersection(self, other: 'WormMask') -> float:
        """
        Calculates the percentage of area that intersects between this mask and other.
        """
        sub_self = self.subsection(other.min_x, other.min_y, other.max_x, other.max_y)
        if sub_self is None:
            return 0

        sub_other = other.subsection(self.min_x, self.min_y, self.max_x, self.max_y)
        return np.sum(np.minimum(sub_self, sub_other)) / max(1.0, min(self.area, other.area))

    def midpoint_distance_squared(self, other: 'WormMask'):
        """
        Calculates the distance between the midpoints between this mask and other.
        """
        p1 = self.points[math.floor(len(self.points) / 2)]
        p2 = other.points[math.floor(len(other.points) / 2)]
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
