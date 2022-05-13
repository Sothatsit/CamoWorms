"""
This file contains logic to search for the best parameters to use for a worm.
"""
import numpy as np
from src import NpImage
from src.helpers import clamp
from src.worm import CamoWorm
from src.worm_mask import WormMask


def score_worm_isolated(worm: CamoWorm, mask: WormMask, outer_mask: WormMask) -> float:
    """
    Scores the given worm mask based on what is below and around it in the image.
    Does not consider other worms in any clew that the worm is a part of.
    """
    if mask.area <= 10:
        return -1

    # Promotes the worms being similar colour to their background.
    body_score = -4 * np.sum(mask.difference_image()**2) / max(1, mask.area)

    # Promotes larger good worms.
    # Works against larger bad worms.
    body_score += 0.09
    body_score *= clamp(mask.area, 1, 1000)**0.25
    body_score *= clamp(mask.area - 1000, 1, 1000)**0.15
    body_score -= 0.09

    # Promotes the regions outside the worm being dissimilar colour.
    edge_score = np.sum(outer_mask.difference_image()) / max(1, outer_mask.area)

    edge_score -= 0.025
    edge_score *= max(1, outer_mask.area) ** 0.3
    edge_score += 0.025

    # We bias the result as we want a score of 0 to represent an okay worm.
    return -clamp(mask.area / 3000, 0.7, 1) + 1.2 * body_score + 1.3 * edge_score


def locally_optimise_worm(
        image: NpImage, worm: CamoWorm,
        *, min_width: float = 4, max_width: float = 16):
    """
    Searches for the best-scoring width and colour for the given worm.
    Returns the inner and outer masks of the worm.
    """
    # Determine the best width based upon what's behind the worm.
    worm.width = max_width
    mask = WormMask(worm, image, gen_distances=True)
    outer_mask = mask.copy()
    best_width = -1
    best_score = 0
    for width in np.linspace(min_width, max_width, 20):
        # Update the worm's width.
        worm.width = width

        # Re-calculate the mask with the new width.
        mask.recalculate_mask()
        mask.create_outer_mask(dest=outer_mask)

        # Update the colour of the worm.
        new_colour = mask.mean_colour()
        if new_colour is not None:
            worm.colour = new_colour

        # Score the width.
        score = score_worm_isolated(worm, mask, outer_mask)
        if best_width == -1 or (score > 0 and score > best_score):
            best_width = width
            best_score = score

    # Apply the best width to the worm.
    worm.width = best_width
    mask.recalculate_mask()
    mask.create_outer_mask(dest=outer_mask)

    # Determine the colour based on what's behind the worm.
    new_colour = mask.mean_colour()
    if new_colour is not None:
        worm.colour = new_colour

    return mask, outer_mask


def locally_optimise_clew(image: NpImage, clew: list[CamoWorm]) -> list[tuple[WormMask, WormMask]]:
    """ Runs the local search optimisation on each worm in a clew. """
    output_masks = []
    for worm in clew:
        output_masks.append(locally_optimise_worm(image, worm))
    return output_masks
