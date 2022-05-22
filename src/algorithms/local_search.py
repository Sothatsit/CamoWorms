"""
This file contains logic to search for the best parameters to use for a worm.
"""
import math
from typing import Optional

import numpy as np
from src import NpImage
from src.helpers import clamp
from src.worm import CamoWorm
from src.worm_mask import WormMask


def fft_magnitudes(values: np.ndarray) -> Optional[np.ndarray]:
    if len(values) < 8:
        return None

    magnitudes = np.abs(np.fft.rfft(values)[1:])
    magnitudes /= 5
    return magnitudes


def score_worm_isolated(colour: float, mask: WormMask, outer_mask: WormMask) -> float:
    """
    Scores the given worm mask based on what is below and around it in the image.
    Does not consider other worms in any clew that the worm is a part of.
    """
    if mask.area <= 10:
        return -999999

    # Promotes the worms being similar colour to their background.
    body_score = -5 * np.sum(mask.difference_image(colour)**2) / max(1, mask.area)

    # Promotes larger good worms.
    # Works against larger bad worms.
    body_score += 0.1
    body_score *= clamp(mask.area, 1, 1000)**0.25
    body_score *= clamp(mask.area - 1000, 1, 1000)**0.2
    body_score -= 0.1

    # Promotes the regions outside the worm being dissimilar colour.
    edge_score = np.sum(outer_mask.difference_image(colour)) / max(1, outer_mask.area)
    edge_score *= max(1, outer_mask.area)**0.4

    # Promotes consistent colours below the worm.
    colour_fft = fft_magnitudes(mask.colour_under_points(buckets=16))
    if colour_fft is not None:
        colour_fft -= 0.06
        consistency_score = -np.sum(colour_fft[0:3])
        consistency_score *= max(1.0, 0.01 * mask.area)**0.15
        consistency_score *= max(1.0, 0.1 * mask.worm_r)**0.5
    else:
        # Either tiny or mostly off the image.
        return -999999

    # Promotes smaller worms slightly.
    area_score = -clamp(mask.area / 3000, 0.7, 1) - 0.1 * mask.width**0.5

    # We bias the result as we want a score of 0 to represent an okay worm.
    return 0.2 + area_score + 1 * body_score + 1.5 * edge_score + 1 * consistency_score


def locally_optimise_worm(
        image: NpImage, worm: CamoWorm,
        *, min_width: float = 4, max_width: float = 16):
    """
    Searches for the best-scoring width and colour for the given worm.
    Returns the inner and outer masks of the worm.
    """
    # Determine the best width based upon what's behind the worm.
    mask = WormMask(image, max_width, worm.r, worm.dr, worm.bezier, high_qual_distances=True)
    outer_mask = mask.copy()
    best_width = -1
    best_score = -1
    for width in np.linspace(min_width, max_width, 20):
        # Re-calculate the mask with the new width.
        mask.recalculate_mask(new_width=width)
        mask.create_outer_mask(dest=outer_mask)

        # Update the colour of the worm.
        new_colour = mask.mean_colour()

        # Score the width.
        score = score_worm_isolated(new_colour, mask, outer_mask)
        if best_width == -1 or score > best_score:
            best_width = width
            best_score = score

    # Apply the best width to the worm.
    worm.width = best_width
    mask.recalculate_mask(new_width=best_width)

    # Determine the colour based on what's behind the worm.
    new_colour = mask.mean_colour()
    if new_colour is not None:
        worm.colour = new_colour

    return mask


def locally_optimise_clew(image: NpImage, clew: list[CamoWorm]) -> list[WormMask]:
    """ Runs the local search optimisation on each worm in a clew. """
    output_masks = []
    for worm in clew:
        output_masks.append(locally_optimise_worm(image, worm))
    return output_masks
