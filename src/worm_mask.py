import math
from typing import Optional, Final

import numba
import numpy as np

from src import NpImage, NpImageDType
from src.bezier import FastBezierSegment
from src.helpers import clamp
from src.worm import CamoWorm


@numba.njit
def np_all_axis1(x: np.ndarray) -> np.ndarray:
    """ Numba compatible version of np.all(x, axis=1). """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


@numba.njit
def _get_many_overlapping_areas(
        offset_xy: np.ndarray, width: int, target: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Returns an array of overlapping areas of shape (N, 4, 2).
    The 2 represents x/y.
    The 4 represents target_from, target_to, circle_to, circle_from.
    The N represents all the points where there is overlap.
    """
    # Calculate the overlapping areas in target_distances.
    w, h = target.shape
    target_from = points + offset_xy
    target_to = np.minimum(target_from + width, np.array([w, h], dtype=np.int32))
    target_from = np.maximum(target_from, 0)

    # Filter out areas with no overlap.
    overlap_condition = np_all_axis1(np.less(target_from, target_to))
    if not np.all(overlap_condition):
        overlap_filter = np.asarray(overlap_condition).nonzero()
        points = points[overlap_filter]
        target_from = target_from[overlap_filter]
        target_to = target_to[overlap_filter]

    # Calculate the areas in this circle mask as well.
    out = np.empty((target_from.shape[0], 4, 2), dtype=np.int32)
    out[:, 0] = target_from
    out[:, 1] = target_to

    # circle_from
    out[:, 2] = target_from
    out[:, 2] -= points
    out[:, 2] -= offset_xy

    # circle_to
    out[:, 3] = target_to
    out[:, 3] -= points
    out[:, 3] -= offset_xy
    return out


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
        self.offset_xy = np.array([self.offset_x, self.offset_y], dtype=np.int32)

        # Calculate a grid of distances to the center of the mask.
        xx, yy = np.mgrid[
            self.offset_x:(self.offset_x + self.width),
            self.offset_y:(self.offset_y + self.width)
        ]

        # Pixel-by-pixel distances.
        self.distances = (xx**2 + yy**2).astype(NpImageDType)
        self.mask = self.distances < self.radius ** 2
        self.inverse_mask = np.logical_not(self.mask)

    def get_many_overlapping_areas(self, target: NpImage, points: np.ndarray) -> NpImage:
        """
        Returns an array of overlapping areas of shape (N, 4, 2).
        The 2 represents x/y.
        The 4 represents target_from, target_to, circle_to, circle_from.
        The N represents all the points where there is overlap.
        """
        if points.dtype != np.int32:
            raise Exception(f"Expected dtype=np.int32, not {points.dtype}")

        return _get_many_overlapping_areas(self.offset_xy, self.width, target, points)

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

    def draw_mask_many(self, target_mask: np.ndarray, points: np.ndarray):
        """ Applies this circle to all the x/y points within the mask. """
        areas = self.get_many_overlapping_areas(target_mask, points)
        for (target_from_x, target_from_y), (target_to_x, target_to_y),  \
                (circle_from_x, circle_from_y), (circle_to_x, circle_to_y) in areas:

            target_area = target_mask[target_from_x:target_to_x,
                                      target_from_y:target_to_y]
            source_area = self.mask[circle_from_x:circle_to_x,
                                    circle_from_y:circle_to_y]
            np.logical_or(target_area, source_area, out=target_area)

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


def filter_in_bounds_points(points: np.ndarray, max_x: int, max_y: int) -> np.ndarray:
    """
    Returns only the points that fall within the bounds [0, 0, max_x, max_y].
    """
    x = points[:, 0]
    y = points[:, 1]

    condition_component = np.greater_equal(x, 0)
    condition = condition_component.copy()
    np.greater_equal(y, 0, out=condition_component)
    condition &= condition_component
    np.less(x, max_x, out=condition_component)
    condition &= condition_component
    np.less(y, max_y, out=condition_component)
    condition &= condition_component

    if np.all(condition):
        return points
    if not np.any(condition):
        return np.empty((0, 2), dtype=NpImageDType)

    return points[np.where(condition)]


@numba.njit
def filter_out_close_points(points: np.ndarray, *, point_interval: float = 6) -> np.ndarray:
    """
    Takes in a numpy array of points, and filters out
    all the points that are closer than point_interval
    together. The exception to this is that the first
    and last points are always kept.
    """
    # No points should just return no points.
    if len(points) == 0:
        return points

    # 1. Calculate the distances between adjacent points.
    n_points = points.shape[0]
    offsets = points[:-1] - points[1:]
    distances = np.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2)

    # 2. Adjust point_interval to evenly space the points.
    total_distance = np.sum(distances)
    if total_distance == 0:
        return points[0:1]
    point_interval = total_distance / math.ceil(total_distance / point_interval)

    # 3. Create an array to use as a filter for the points.
    # An array of which points to keep.
    point_filter = np.zeros((n_points,), dtype=np.bool_)
    point_filter[0] = True  # The first and last points should always be kept.
    point_filter[n_points - 1] = True

    # 4. Fill in the point filter.
    curr_dist = 0.0
    for index in range(1, len(distances)):
        curr_dist += distances[index - 1]
        if curr_dist >= point_interval:
            point_filter[index] = True
            curr_dist = 0

    # 5. Apply the filter
    return points[np.where(point_filter)]


@numba.njit
def _midpoint_distance_squared(self_points: np.ndarray, other_points: np.ndarray) -> float:
    self_len = len(self_points)
    other_len = len(other_points)
    if self_len == 0 or other_len == 0:
        return 0

    p1 = self_points[self_len // 2]
    p2 = other_points[other_len // 2]
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def radius_from_worm_width(width: float) -> float:
    """ Based on empirical testing... """
    return width / 2 + 0.5


@numba.njit
def _colour_under_points(points: np.ndarray, image: NpImage, buckets: int) -> np.ndarray:
    """ Finds the mean colour in the image under the buckets sections of the given points. """
    point_groups = np.array_split(points, buckets)
    colours = np.empty((buckets,), dtype=NpImageDType)
    for group_index, point_group in enumerate(point_groups):
        total = 0
        for point in point_group:
            total += image[point[0], point[1]]
        colours[group_index] = (total / 255.0) / len(point_group)

    return colours


class WormMask:
    """
    Contains a mask of a worm in an image.
    """
    FAR_DISTANCE: Final[float] = 1e12
    OUTER_MAX_DIST: Final[int] = 5

    def __init__(
            self,
            image: NpImage,
            worm_width: float,
            worm_r: float,
            worm_dr: float,
            worm_bezier: FastBezierSegment,
            *,
            copy: 'WormMask' = None,
            high_qual_distances: bool = False):
        """
        :param high_qual_distances:
            The mask will be generated with a distance array
            that will be accurate for many widths of worm. This is only
            useful if you are testing many widths of worm. Otherwise,
            for larger worms this really slows down mask generation.
        """
        self.worm_width: float = worm_width
        self.worm_r: float = worm_r
        self.worm_dr: float = worm_dr
        self.worm_bezier: FastBezierSegment = worm_bezier
        self.image = image
        self.high_qual_distances = high_qual_distances
        self.distances: Optional[np.ndarray] = None

        if copy is not None:
            self.min_x = copy.min_x
            self.min_y = copy.min_y
            self.max_x = copy.max_x
            self.max_y = copy.max_y
            self.width = copy.width
            self.height = copy.height
            self.radius = copy.radius
            self.points = copy.points
            self.dense_points = copy.dense_points
            self.distances = copy.distances
            self.mask = copy.mask
            self.area = copy.area
            self.exists = copy.exists
        else:
            # 1. Calculate some information about the worm
            img_width, img_height = image.shape
            self.radius = radius_from_worm_width(worm_width)
            n_points_estimate = math.ceil(1.5 * worm_r + 1.5 * abs(worm_dr))

            raw_points = worm_bezier(
                np.linspace(0, 1, num=n_points_estimate)
            )[:, ::-1]
            raw_points = filter_in_bounds_points(raw_points, img_width, img_height)
            raw_points = raw_points.astype(np.int32)

            self.dense_points = filter_out_close_points(raw_points, point_interval=2)

            # Defaults in case we can't create the worm.
            self.distances = np.empty((0, 0), dtype=NpImageDType)
            self.points = np.empty((0, 2), dtype=NpImageDType)
            self.min_x = self.min_y = 0

            if len(self.dense_points) > 0:
                # The padding is needed for the radius of the circle and the outer mask calculation.
                padding = math.ceil(self.radius * 1.5) + WormMask.OUTER_MAX_DIST
                x, y = (self.dense_points[:, 0], self.dense_points[:, 1])
                self.min_x = max(0, math.floor(np.amin(x)) - padding)
                self.min_y = max(0, math.floor(np.amin(y)) - padding)
                max_x = min(img_width, math.ceil(np.amax(x)) + padding)
                max_y = min(img_height, math.ceil(np.amax(y)) + padding)
                width = max_x - self.min_x
                height = max_y - self.min_y

                # 2. Some points are really close together, so we can filter some of them out for speed.
                if high_qual_distances:
                    self.points = self.dense_points
                else:
                    point_interval = max(2.0, self.radius / 2)
                    self.points = filter_out_close_points(
                        self.dense_points, point_interval=point_interval
                    )

                # 3. Apply a circle mask at each point on the curve to the mask.
                circle_mask = CircleMask.get_cached(self.radius + WormMask.OUTER_MAX_DIST)
                mask_min_pt = np.array([self.min_x, self.min_y], dtype=np.int32)
                self.distances = np.full((width, height), WormMask.FAR_DISTANCE, dtype=NpImageDType)
                circle_mask.apply_many(self.distances, self.points - mask_min_pt)

            self.recalculate_mask()
            self.width = self.mask.shape[0]
            self.height = self.mask.shape[1]
            self.max_x = self.min_x + self.width
            self.max_y = self.min_y + self.height
            self.exists = self.width > 0 and self.height > 0

        # Cached results.
        self._outer_mask: Optional[WormMask] = None
        self._image_within_bounds: Optional[NpImage] = None
        self._image_under_mask: Optional[NpImage] = None

    @staticmethod
    def from_worm(worm: CamoWorm, image: NpImage, **kwargs):
        """ Creates a WormMask from the given worm for the given image. """
        return WormMask(
            image,
            worm.width,
            worm.r,
            worm.dr,
            worm.bezier,
            **kwargs
        )

    def copy(self) -> 'WormMask':
        """ Generates a copy of this worm. """
        return WormMask(
            self.image,
            self.worm_width,
            self.worm_r,
            self.worm_dr,
            self.worm_bezier,
            copy=self
        )

    def recalculate_mask(self, new_width: Optional[float] = None):
        """
        Re-calculates the mask based upon the worm associated with this mask.
        """
        if new_width is not None:
            self.worm_width = new_width
            self.radius = radius_from_worm_width(new_width)

        self.mask = self.distances < self.radius**2
        self.area = int(np.sum(self.mask))

        # Reset cached values that rely on the mask.
        self._image_under_mask = None

    def create_outer_mask(self, *, dest: 'WormMask' = None) -> 'WormMask':
        """
        Generates a mask encompassing the sides of this mask.
        """
        do_cache = dest is None
        if do_cache and self._outer_mask is not None:
            return self._outer_mask

        distance: float = min(self.radius, WormMask.OUTER_MAX_DIST)

        # Create the new mask.
        if dest is None:
            dest = self.copy()
        dest.distances = self.distances.copy()

        # Update the mask to its outer edge.
        outer_radius = self.worm_width / 2 + distance
        dest.mask = (dest.distances < (outer_radius ** 2)) & (~self.mask)
        dest.area = np.sum(dest.mask)

        # Calculate points to use to remove the ends off the worm.
        points = self.points
        if len(points) >= 4:
            remove_points = []
            i_offset = clamp(math.ceil(len(points) / 4), 1, 4)
            for end, one_off_end in [(points[0], points[i_offset]), (points[-1], points[-1 - i_offset])]:
                # Calculate the direction away from the end of the worm.
                vec = (end - one_off_end).astype(NpImageDType)
                vec_len = np.linalg.norm(vec)
                if vec_len == 0:
                    print("len = ", vec_len, "for", end, "-", one_off_end)
                    continue

                vec /= vec_len

                # Generate vectors to represent directions from the tip of the worm.
                forwards_vec = vec * (distance / 2.0 + self.radius)
                sideways_vec = np.array([-vec[1], vec[0]])
                sideways_vec *= self.radius

                # Find two points off the end of the worm.
                pt_1 = end + forwards_vec
                pt_2 = pt_1 + sideways_vec
                pt_1 -= sideways_vec
                remove_points.append(pt_2)
                remove_points.append(pt_1)

            # Draw black circles around the points.
            if len(remove_points) > 0:
                circle_mask = CircleMask.get_cached(self.radius + 2)
                mask_min_pt = np.array([self.min_x, self.min_y], dtype=self.points.dtype)
                remove_points = np.array(remove_points, dtype=np.int32)
                remove_points -= mask_min_pt
                circle_mask.remove_many_from_mask(dest.mask, remove_points)

        if do_cache:
            self._outer_mask = dest
        return dest

    def draw_into(self, image: NpImage, colour: float) -> None:
        """
        Draws this worm mask into the given image with the given colour.
        """
        if not self.exists:
            return

        # 1. Get the subsection that corresponds to the mask.
        image_subsection = image[self.min_x:self.max_x, self.min_y:self.max_y]

        # 2. Black-out the area under the mask.
        np.subtract(image_subsection, image_subsection *
                    self.mask, image_subsection)

        # 3. Add in the colour for the masked area.
        np.add(image_subsection, colour * self.mask, image_subsection)

    def image_within_bounds(self, image=None) -> NpImage:
        """
        Returns the image under the mask with the pixels
        not within the mask being marked as 0.
        """
        do_cache = image is None or image is self.image
        if do_cache and self._image_within_bounds is not None:
            return self._image_within_bounds

        image = self.image if image is None else image

        # Get the subset of the image corresponding to this mask.
        result = image[self.min_x:self.max_x, self.min_y:self.max_y] / 255.0
        if do_cache:
            self._image_within_bounds = result
        return result

    def colour_under_points(self, image=None, *, buckets=16) -> np.ndarray:
        """
        Returns a 1-d array of colours under the centre line of the worm,
        with the colours taken as the mean from sections of the worm.
        """
        if len(self.dense_points) < buckets:
            return np.empty((0,), dtype=NpImageDType)

        image = self.image if image is None else image
        return _colour_under_points(self.dense_points, image, buckets)

    def image_under_mask(self, image=None) -> NpImage:
        """
        Returns the image under the mask with the pixels
        not within the mask being marked as 0.
        """
        do_cache = image is None or image is self.image
        if do_cache and self._image_under_mask is not None:
            return self._image_under_mask

        # Apply the mask to the image.
        result = self.image_within_bounds(image) * self.mask
        if do_cache:
            self._image_under_mask = result
        return result

    def difference_image(self, worm_colour: float, image: Optional[NpImage] = None) -> NpImage:
        """
        This applies this mask to the image, to calculate the differences
        between the colour of the worm and all the pixels beneath the worm.
        """
        # 1. Calculate the difference of the colour of the worm to its background.
        diff_image = self.image_under_mask(image) - worm_colour
        np.absolute(diff_image, out=diff_image)

        # 2. Apply the mask to the difference.
        diff_image *= self.mask
        return diff_image

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
        if not self.exists:
            return None

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
        return _midpoint_distance_squared(self.points, other.points)
