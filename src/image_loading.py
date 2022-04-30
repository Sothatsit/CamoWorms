"""
This file contains the logic to load images for CamoWorms.
"""

import os
from typing import Optional, Tuple

import imageio
import numpy as np

from src import NPImage
from src.image_manipulation import crop


def load_image(image_dir: str, image_name: str, mask: Optional[Tuple[int, int, int, int]] = None) -> NPImage:
    """ Loads the given image from the image directory. """
    image_path = os.path.join(image_dir, "{}.png".format(image_name))
    data = imageio.imread(image_path)
    if mask is not None:
        data = crop(data, mask)
    return np.flipud(data).astype(np.float64)
