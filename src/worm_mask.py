import math
from typing import Optional, Final

import numpy as np

from src import NpImage, NpImageDType
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
    point_interval = total_distance / \
        math.ceil(total_distance / point_interval)

    # 3. Create an array to use as a filter for the points.
    # An array of which points to keep.
    point_filter = np.zeros(n_points, dtype=bool)
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
        self.offset_xy = np.array([self.offset_x, self.offset_y], dtype=np.int)

        # Calculate a grid of distances to the center of the mask.
        xx, yy = np.mgrid[
            self.offset_x:(self.offset_x + self.width),
            self.offset_y:(self.offset_y + self.width)
        ]

        # Pixel-by-pixel distances.
        self.distances = (xx**2 + yy**2).astype(NpImageDType)
        self.mask = self.distances < self.radius ** 2
        self.inverse_mask = np.logical_not(self.mask)

    def get_many_overlapping_areas(self, target: NpImage, points: NpImage) -> NpImage:
        """
        Returns an array of overlapping areas of shape (N, 4, 2).
        The 2 represents x/y.
        The 4 represents target_from, target_to, circle_to, circle_from.
        The N represents all the points where there is overlap.
        """
        points = points.astype(np.int)

        # Calculate the overlapping areas in target_distances.
        target_from = points + self.offset_xy
        target_to = target_from + self.width
        np.maximum(target_from, 0, out=target_from)
        np.minimum(target_to, np.array(
            target.shape, dtype=np.int), out=target_to)

        # Filter out areas with no overlap.
        overlap_condition = (target_from < target_to).all(
            axis=1, keepdims=False)
        if not np.all(overlap_condition):
            overlap_filter = np.asarray(overlap_condition).nonzero()
            points = points[overlap_filter]
            target_from = target_from[overlap_filter]
            target_to = target_to[overlap_filter]

        # Calculate the areas in this circle mask as well.
        out = np.empty((target_from.shape[0], 4, 2), dtype=np.int)
        out[:, 0] = target_from
        out[:, 1] = target_to

        out[:, 2] = target_from
        out[:, 2] -= points
        out[:, 2] -= self.offset_xy

        out[:, 3] = target_to
        out[:, 3] -= points
        out[:, 3] -= self.offset_xy
        return out

    def apply_many(self, target_distances: NpImage, points: np.ndarray):
        """ Applies this circle to all the x/y points within the points ndarray. """
        areas = self.get_many_overlapping_areas(target_distances, points)
        for (target_from_x, target_from_y), (target_to_x, target_to_y),  \
                (circle_from_x, circle_from_y), (circle_to_x, circle_to_y) in areas:

            target_area = target_distances[target_from_x:target_to_x,
                                           target_from_y:target_to_y]
            source_area = self.distances[circle_from_x:circle_to_x,
                                         circle_from_y:circle_to_y]
            np.minimum(target_area, source_area, out=target_area)

    def remove_many_from_mask(self, target_mask: np.ndarray, points: np.ndarray):
        """ Erases this circle from the target mask at all the given x/y points. """
        areas = self.get_many_overlapping_areas(target_mask, points)
        for (target_from_x, target_from_y), (target_to_x, target_to_y),  \
                (circle_from_x, circle_from_y), (circle_to_x, circle_to_y) in areas:

            target_area = target_mask[target_from_x:target_to_x,
                                      target_from_y:target_to_y]
            source_area = self.inverse_mask[circle_from_x:circle_to_x,
                                            circle_from_y:circle_to_y]
            np.logical_and(target_area, source_area, out=target_area)

    @staticmethod
    def get_cached(radius: float, *, quantize_accuracy: int = 2) -> 'CircleMask':
        """
        Tries to find a cached circle that is close in size to radius.
        If one can not be found, it creates it.
        """
        # Quantize the radius into 0.5 pixel steps.
        quantized = round(radius * quantize_accuracy)

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
        mask = CircleMask(quantized / quantize_accuracy)
        cache[quantized] = mask
        return mask


class WormMask:
    """
    Contains a mask of a worm in an image.
    """
    FAR_DISTANCE: Final[float] = 1e12
    OUTER_MAX_DIST: Final[int] = 10

    def __init__(self, worm: CamoWorm, image: NpImage,
                 *, copy: 'WormMask' = None, all_widths_accurate: bool = False):
        """
        :param all_widths_accurate: The mask will be generated so that its distance array
                                    will be accurate for all widths of worm. This is only
                                    useful if you are testing many widths of worm. Otherwise,
                                    for larger worms this really slows down mask generation.
        """
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
            radius = worm.width / 2 + 1.5
            n_points_estimate = math.ceil(1.5 * worm.r + 1.5 * abs(worm.dr))
            self.points = worm.bezier(np.linspace(
                0, 1, num=n_points_estimate))[:, ::-1]
            self.dense_points = self.points

            # The padding is needed for the radius of the circle and the outer mask calculation.
            padding = math.ceil(radius * 1.5) + WormMask.OUTER_MAX_DIST
            self.min_x = max(0, math.floor(
                np.amin(self.points[:, 0])) - padding)
            self.min_y = max(0, math.floor(
                np.amin(self.points[:, 1])) - padding)
            max_x = min(img_width, math.ceil(
                np.amax(self.points[:, 0])) + padding)
            max_y = min(img_height, math.ceil(
                np.amax(self.points[:, 1])) + padding)
            width = max_x - self.min_x
            height = max_y - self.min_y

            # If the worm is outside the image, then we just use a dummy empty 1x1 mask.
            if width <= 0 or height <= 0:
                self.min_x = 0
                self.min_y = 0
                self.points = self.points[0:1]
                self.distances = np.full(
                    (1, 1), WormMask.FAR_DISTANCE, dtype=NpImageDType)
            else:
                # 2. Some points are really close together, so we can filter some of them out for speed.
                if all_widths_accurate:
                    point_interval = 2
                else:
                    point_interval = max(2.0, radius / 2)

                self.points = filter_out_close_points(
                    self.points, point_interval=point_interval)

                # 3. Apply a circle mask at each point on the curve to the mask.
                self.distances = np.full(
                    (width, height), WormMask.FAR_DISTANCE, dtype=NpImageDType)
                circle_mask = CircleMask.get_cached(
                    math.ceil(radius) + WormMask.OUTER_MAX_DIST)
                mask_min_pt = np.array(
                    [self.min_x, self.min_y], dtype=self.points.dtype)
                circle_mask.apply_many(
                    self.distances, self.points - mask_min_pt)

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
        self.area = int(np.sum(self.mask))

    def create_outer_mask(self, *, dest: 'WormMask' = None) -> 'WormMask':
        """
        Generates a mask encompassing the sides of this mask.
        """
        distance = min(float(WormMask.OUTER_MAX_DIST),
                       max(1.0, self.worm.width / 2.0))

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
        remove_points = []
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
            remove_points.append(pt_1)
            remove_points.append(pt_2)
            remove_points.append(pt_3)

        # Draw black circles around the points.
        mask_min_pt = np.array([self.min_x, self.min_y],
                               dtype=self.points.dtype)
        circle_mask.remove_many_from_mask(
            dest.mask, np.array(remove_points) - mask_min_pt)
        return dest

    def draw_into(self, image, colour):
        """
        Draws this worm mask into the given image with the given colour.
        """
        # 1. Get the subsection that corresponds to the mask.
        image_subsection = image[self.min_x:self.max_x, self.min_y:self.max_y]

        # 2. Black-out the area under the mask.
        np.subtract(image_subsection, image_subsection *
                    self.mask, image_subsection)

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

    def difference_image(self, image: Optional[NpImage] = None) -> NpImage:
        """
        This applies this mask to the image, to calculate the differences
        between the colour of the worm and all the pixels beneath the worm.
        """
        # 1. Calculate the difference of the colour of the worm to its background.
        diff_image = np.absolute(
            self.image_under_mask(image) - self.worm.colour)

        # 2. Apply the mask to the difference.
        return diff_image * self.mask

    def mean_colour(self, image: Optional[NpImage] = None) -> Optional[float]:
        """
        Returns the mean colour of all pixels underneath this mask in the image.
        """
        if self.area == 0:
            return None

        return np.sum(self.image_under_mask(image)) / self.area

    def median_colour(self, image=None):
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
        median_index = math.floor((first_index + len(colours) - 1) / 2.0)
        return colours[median_index] - 1

    def subsection(self, min_x, min_y, max_x, max_y):
        """
        Gets the subsection of this mask that covers the given coordinates.
        """
        if max_x <= self.min_x or max_y <= self.min_y:
            return None
        if min_x >= self.max_x or min_y >= self.max_y:
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
        Calculates the percentage of this mask's area that intersects with the other mask.
        """
        sub_self = self.subsection(
            other.min_x, other.min_y, other.max_x, other.max_y)
        if sub_self is None:
            return 0

        sub_other = other.subsection(
            self.min_x, self.min_y, self.max_x, self.max_y)
        return np.sum(np.logical_and(sub_self, sub_other)) / max(1, self.area)

    def midpoint_distance_squared(self, other: 'WormMask'):
        """
        Calculates the distance between the midpoints between this mask and other.
        """
        p1 = self.points[math.floor(len(self.points) / 2)]
        p2 = other.points[math.floor(len(other.points) / 2)]
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
