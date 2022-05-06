"""
This file implements a greedy genetic algorithm to evolve a clew of worms.
"""
import math
import numpy as np
from typing import List, Tuple

from src import rng
from src.algorithm import GeneticClewEvolution
from src.helpers import clamp
from src.image_manipulation import edge_enhance
from src.worm import Camo_Worm
from src.worm_mask import WormMask


class GreedyClewEvolution(GeneticClewEvolution):
    """
    A basic implementation of clew evolution to get the ball rolling.
    This implementation just makes random mutations and only keeps
    mutations that improve the score of the worm.
    """

    def __init__(self, image, clew: List[Camo_Worm]):
        super().__init__(image, clew, name="Greedy")
        self.edge_enhanced_image = edge_enhance(image)

    def score(self, worm: Camo_Worm, worm_mask: WormMask) -> float:
        """ A basic benchmark scoring function. """
        score = 0.0

        # Promotes the worms being similar colour to their background.
        score -= 100 * np.sum(worm_mask.difference_image()
                              ) / max(1, worm_mask.area)

        # Promotes the regions outside the worm being dissimilar colour.
        outer_mask = worm_mask.create_outer_mask()
        score += 50 * np.sum(outer_mask.difference_image()
                             ) / max(1, outer_mask.area)

        # Promotes bigger worms if the worms are already decent.
        score += (0.5 * worm.width + 0.2 * worm.r if score > -3 else 0)

        # Attempts to avoid overlapping and close worms.
        close_penalty = 0.0
        overlap_penalty = 0.0
        for other_index in range(len(self.clew)):
            other_worm = self.clew[other_index]
            # Give earlier worms in the clew priority
            if worm is other_worm:
                break

            other_worm_mask = self.clew_masks[other_index]
            close_penalty += max(0.0, 500 -
                                 worm_mask.midpoint_distance_squared(other_worm_mask))
            # Allow 10% overlap
            overlap_penalty += max(0.0,
                                   worm_mask.intersection(other_worm_mask) - 0.1)

        score -= 0.1 * close_penalty
        score -= 20 * overlap_penalty

        return score

    def random_mutate(self, worm: Camo_Worm, score: float) -> Tuple[Camo_Worm, WormMask]:

        # We make bigger mutations to worms with lower scores.
        temp = 5 * clamp(1 - score / 50, 0.01, 1.0)
        if rng.random() < 0.1:
            temp = 5

        x = worm.x + 4 * rng.standard_normal() * temp
        y = worm.y + 4 * rng.standard_normal() * temp
        r = worm.r + 1 * rng.standard_normal() * temp
        theta = worm.theta + math.pi / 2 * rng.standard_normal() * temp
        dr = worm.dr + rng.standard_normal() * temp
        dgamma = worm.dgamma + math.pi * rng.standard_normal() * temp
        width = worm.width + rng.standard_normal() * temp

        new_worm = Camo_Worm(
            clamp(x, 0, self.image.shape[1]),
            clamp(y, 0, self.image.shape[0]),
            clamp(r, 10, 100),
            (theta + 20 * math.pi) % (math.pi / 2),
            clamp(dr, -50, 50),
            (dgamma + 20 * math.pi) % math.pi,
            clamp(width, 2, 20),
            worm.colour
        )

        # Determine the colour based on what's behind the worm.
        new_worm_mask = WormMask(new_worm, self.image)
        new_colour = new_worm_mask.mean_colour
        if new_colour is not None:
            new_worm.colour = new_colour

        return new_worm, new_worm_mask

    def update(self, worm: Camo_Worm, worm_mask: WormMask) -> Tuple[Camo_Worm, WormMask]:
        """ A basic greedy update function based upon random mutations to the worm. """
        worm_score = self.score(worm, worm_mask)

        new_worm = worm
        new_worm_mask = worm_mask
        new_score = worm_score

        # Try some random mutations to improve the worm.
        for i in range(5):
            mutated_worm, mutated_worm_mask = self.random_mutate(
                worm, worm_score
            )
            mutated_score = self.score(mutated_worm, mutated_worm_mask)
            if mutated_score > new_score:
                new_worm = mutated_worm
                new_worm_mask = mutated_worm_mask
                new_score = mutated_score

        return new_worm, new_worm_mask
