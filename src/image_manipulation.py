"""
This file contains methods to manipulate images.
"""
import math
from typing import Tuple

import numpy as np
from scipy import fftpack
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import convolve2d

from src import NpImage


def crop(image: NpImage, mask: Tuple[int, int, int, int]) -> NpImage:
    """ Crops the image to the given region described by (min_x, max_x, min_y, max_y). """
    h, w = np.shape(image)
    return image[
        max(mask[0], 0):min(mask[1], h),
        max(mask[2], 0):min(mask[3], w)
    ]


def blur(image: NpImage, *, sigma=2) -> NpImage:
    """ Blur the image using a Gaussian filter. """
    return gaussian_filter(image, sigma=sigma)


def mean_window(image: NpImage, *, window=3, times=1) -> NpImage:
    """ Run an averaging filter over the image. """
    for i in range(times):
        image = convolve2d(image, np.ones((window, window)) / (window ** 2), mode="same")
    return image


def median_window(image: NpImage, *, window=3, times=1) -> NpImage:
    """ Run an averaging filter over the image. """
    for i in range(times):
        image = median_filter(image, size=window)
    return image


def fft_denoise(image: NpImage, *, coeff=0.25) -> NpImage:
    """
    This uses a 2-dimensional fourier transform and a subsequent
    low-pass filter to reduce high-frequency noise.
    """
    # Compute the fourier transform of the image.
    image_fft = fftpack.fftshift(fftpack.fft2(image))

    # Discard high frequencies (which are further from the center of the image).
    w, h = image_fft.shape
    x, y = np.ogrid[0:w, 0:h]
    dist = ((2 * (x - w / 2.0) / w) ** 2 + (2 * (y - h / 2.0) / h) ** 2) / 2
    image_fft *= dist < coeff ** 2

    # Reconstruct the image.
    image = fftpack.ifft2(fftpack.ifftshift(image_fft)).real

    # Clamp the image to the range [0, 255]
    return np.minimum(np.maximum(image, 0), 255)


def scale_brighten(image: NpImage) -> NpImage:
    """ Scales the image so that the darkest pixel is 0, and the brightest is 255. """
    return np.interp(image, (image.min(), image.max()), (0.0, 255.0))


def scale_01(image: NpImage) -> NpImage:
    """ Scales the image so that the darkest pixel is 0, and the brightest is 1. """
    return np.interp(image, (image.min(), image.max()), (0.0, 1.0))


def extremify(image: NpImage, *, coeff=2.0) -> NpImage:
    """
    Makes dark pixels darker and bright pixels brighter.
    If coeff is larger, then this is more extreme.
    """
    # This is inspired by the sigmoid function.
    image_01 = 2.0 * (scale_01(image) - 0.5)
    shifted = 1.0 / (1.0 + math.e**((-coeff) * image_01)) - 0.5

    # Scale the image to the range [0, 255]
    return scale_brighten(shifted)


def local_brighten(image: NpImage, *, local_area_window=20, limit=0.2, scale_coefficient=3.0) -> NpImage:
    """ Brightens regions of the image based upon the brightness of their surroundings. """
    # Scale the image into the range [0, 1].
    image = scale_01(image)

    # Use a massive blur to approximate local brightness.
    local_brightness = gaussian_filter(image, sigma=local_area_window)

    # Apply limits so that the image cannot be changed too much.
    np.clip(local_brightness, limit, 1 - limit, out=local_brightness)

    # Shift the image towards its local brightness, away from 0.5.
    shift = 0.3 * (local_brightness - 0.5)
    image += shift

    # Apply the scaling relative to the local brightness.
    locally_brightened = extremify(image - local_brightness, coeff=scale_coefficient)

    # Combine the original image and the locally brightened image based on the limit.
    return 255 * limit * image + (1 - limit) * locally_brightened


def edge_enhance(image: NpImage, *, edge_weight=0.5) -> NpImage:
    """
    Attempts to enhance lines in the given image using a mixture
    of brightening and darkening different regions of the image,
    and the enhancement of edges.
    """
    # First, locally brighten the image.
    image = local_brighten(blur(image, sigma=1))

    # Mangle the image to help get stronger edges.
    median_image = local_brighten(median_window(image, times=2)) / 255.0

    # Detect its edges using a Sobel convolution.
    kernel_x = np.array(
        [[-2/8, -1/5,  0,  1/5,  2/8],
         [-2/5, -1/2,  0,  1/2,  2/5],
         [-2/4, -1/1,  0,  1/1,  2/4],
         [-2/5, -1/2,  0,  1/2,  2/5],
         [-2/8, -1/5,  0,  1/5,  2/8]]
    )
    kernel_y = np.swapaxes(kernel_x, 0, 1)

    edges_x = convolve2d(median_image, kernel_x, boundary='symm', mode='same')
    edges_y = convolve2d(median_image, kernel_y, boundary='symm', mode='same')
    edges = scale_01(np.sqrt(edges_x**2 + edges_y**2)) * edge_weight

    # Push regions under the edges closer to 0 or 1 using the sigmoid function.
    image = image * (1 - edges) + extremify(image, coeff=6) * edges
    image = np.minimum(np.maximum(image, 0), 255)

    # Brighten the resulting image.
    return image


def find_median_colour(image: NpImage) -> float:
    """
    Determines the median colour of the image.
    """
    colours = np.sort(image.flatten())
    median_index = math.floor((len(colours) - 1) / 2.0)
    return colours[median_index] - 1
