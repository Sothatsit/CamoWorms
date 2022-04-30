import math
from typing import Optional

import numpy as np

from src.worm import Camo_Worm


def filter_out_close_points(points, *, point_interval=6):
    """
    Takes in a numpy array of points, and filters out
    all the points that are closer than point_interval
    together. The exception to this is that the first
    and last points are always kept.
    """
    # 1. Calculate the distances between adjacent points.
    n_points = points.shape[0]
    offsets = points[:-1] - points[1:]
    distances = np.sqrt(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)

    # 2. Adjust the point_interval so we can evenly space the points.
    total_distance = np.sum(distances)
    point_interval = total_distance / math.ceil(total_distance / point_interval)

    # 3. Create an array to use as a filter for the points.
    point_filter = np.zeros(n_points, dtype=bool)  # An array of which points to keep.
    point_filter[0] = True  # The first and last points should always be kept.
    point_filter[n_points - 1] = True

    # 4. Fill in the point filter.
    curr_dist = 0
    for index in range(1, n_points):
        curr_dist += distances[index - 1]
        if curr_dist >= point_interval:
            point_filter[index] = True
            curr_dist = 0

    # 5. Apply the filter
    return points[np.where(point_filter)]


class CircleMask:
    """
    Contains a circular mask.
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

        # Anti-aliasing.
        self.mask = np.zeros((self.width, self.width), dtype=np.float64)
        count = 0.0
        for offset_x in [-1 / 4, 0, 1 / 4]:
            for offset_y in [-1 / 4, 0, 1 / 4]:
                count += 1.0

                # Calculate all the pixels falling within the circle, and add them to the mask!
                dist_squared = (xx + offset_x) ** 2 + (yy + offset_y) ** 2
                self.mask += dist_squared < (radius ** 2)

        self.mask /= count

    def apply(self, target_mask, x: float, y: float):
        """
        Draws this circle into the target mask with its center at (x, y).
        """
        # We don't care that much about sub-pixel accuracy.
        x = round(x)
        y = round(y)

        # Calculate the bounds in the target mask to paste the circle.
        target_from_x = max(0, x + self.offset_x)
        target_from_y = max(0, y + self.offset_y)
        target_to_x = min(target_mask.shape[0] - 1, x + self.offset_x + self.width)
        target_to_y = min(target_mask.shape[1] - 1, y + self.offset_y + self.width)
        if target_from_x >= target_to_x or target_from_y >= target_to_y:
            return

        # Calculate the bounds in the circle mask to paste.
        circle_from_x = target_from_x - x - self.offset_x
        circle_from_y = target_from_y - y - self.offset_y
        circle_to_x = target_to_x - x - self.offset_x
        circle_to_y = target_to_y - y - self.offset_y

        # Add the circle mask into the target mask.
        target_area = target_mask[target_from_x:target_to_x, target_from_y:target_to_y]
        source_area = self.mask[circle_from_x:circle_to_x, circle_from_y:circle_to_y]
        np.maximum(target_area, source_area, out=target_area)

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
    def __init__(self, worm: Camo_Worm, image, *, copy: 'WormMask' = None):
        self.worm = worm
        self.image = image

        if copy is not None:
            self.min_x = copy.min_x
            self.min_y = copy.min_y
            self.max_x = copy.max_x
            self.max_y = copy.max_y
            self.width = copy.width
            self.height = copy.height
            self.mask = copy.mask
        else:
            # 1. Calculate some information about the worm
            img_width, img_height = image.shape
            radius = worm.width / 2
            self.points = worm.bezier(np.linspace(0, 1, num=500))
            self.min_x = max(0, math.floor(np.amin(self.points[:, 1])) - math.ceil(radius * 1.5))
            self.min_y = max(0, math.floor(np.amin(self.points[:, 0])) - math.ceil(radius * 1.5))
            max_x = min(img_width, math.ceil(np.amax(self.points[:, 1])) + math.ceil(radius * 1.5))
            max_y = min(img_height, math.ceil(np.amax(self.points[:, 0])) + math.ceil(radius * 1.5))
            width = max_x - self.min_x
            height = max_y - self.min_y

            # If the worm is outside of the image, then we just use a dummy empty 1x1 mask.
            if width <= 0 or height <= 0:
                self.min_x = 0
                self.min_y = 0
                self.mask = np.zeros((1, 1))
            else:
                # 2. Some points are really close together, so we can filter some of them out for speed.
                self.points = filter_out_close_points(self.points, point_interval=radius / 1.5)

                # 3. Apply a circle mask at each point on the curve to the mask.
                self.mask = np.zeros((width, height))
                circle_mask = CircleMask.get_cached(radius)
                for y, x in self.points:
                    circle_mask.apply(self.mask, x - self.min_x, y - self.min_y)

            self.area = np.sum(self.mask)
            self.width = self.mask.shape[0]
            self.height = self.mask.shape[1]
            self.max_x = self.min_x + self.width
            self.max_y = self.min_y + self.height

    def copy(self):
        """ Generates a copy of this worm. """
        return WormMask(self.worm, self.image, copy=self)

    def create_outer_mask(self, *, maximum_width=5) -> 'WormMask':
        """
        Generates a mask encompassing the sides of this mask.
        """
        distance = min(maximum_width, max(1, round(self.worm.width / 2.0)))

        # Create the new mask.
        outer_mask = self.copy()
        outer_mask.min_x -= distance
        outer_mask.min_y -= distance
        outer_mask.max_x += distance
        outer_mask.max_y += distance
        outer_mask.width += 2 * distance
        outer_mask.height += 2 * distance
        outer_mask.mask = np.zeros((outer_mask.width, outer_mask.height))

        # Paste the mask a few times.
        root2 = math.sqrt(2)
        for dx in [-distance, 0, distance]:
            for dy in [-distance, 0, distance]:
                if dx != 0 and dy != 0:
                    dx = round(dx / root2)
                    dy = round(dy / root2)
                if dx == 0 and dy == 0:
                    continue

                subsection = outer_mask.mask[
                             (distance + dx):(distance + dx + self.width),
                             (distance + dy):(distance + dy + self.height)
                             ]
                np.maximum(subsection, self.mask, subsection)

        # Subtract the original mask.
        original = outer_mask.mask[
                   distance:(distance + self.width),
                   distance:(distance + self.height)
                   ]
        np.minimum(original, 1 - self.mask, original)

        # Crop the mask to the bounds of the image.
        min_x = max(0, outer_mask.min_x)
        min_y = max(0, outer_mask.min_y)
        max_x = min(self.image.shape[0], outer_mask.max_x)
        max_y = min(self.image.shape[1], outer_mask.max_y)
        outer_mask.width = max_x - min_x
        outer_mask.height = max_y - min_y
        outer_mask.mask = outer_mask.mask[
                          (min_x - outer_mask.min_x):(min_x - outer_mask.min_x + outer_mask.width),
                          (min_y - outer_mask.min_y):(min_y - outer_mask.min_y + outer_mask.height)
                          ]
        outer_mask.min_x = min_x
        outer_mask.min_y = min_y
        outer_mask.max_x = max_x
        outer_mask.max_y = max_y

        # Update the area of the mask.
        outer_mask.area = np.sum(outer_mask.mask)
        return outer_mask

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

    @property
    def mean_colour(self):
        """
        Returns the mean colour of all pixels underneath this mask in the image.
        """
        if self.area == 0:
            return None

        return np.sum(self.image_under_mask()) / self.area

    @property
    def mode_colour(self):
        """
        Returns the most common colour of all pixels underneath this mask in the image.
        """
        if self.area == 0:
            return None

        # We want 0 to represent that a pixel is not under the worm.
        sub_image = self.image_within_bounds() + 1
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
        return 2 * np.sum(np.minimum(sub_self, sub_other)) / max(1, self.area + other.area)

    def midpoint_distance_squared(self, other: 'WormMask'):
        """
        Calculates the distance between the midpoints between this mask and other.
        """
        p1 = self.points[math.floor(len(self.points) / 2)]
        p2 = other.points[math.floor(len(other.points) / 2)]
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
