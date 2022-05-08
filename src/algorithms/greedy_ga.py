"""
This file implements a greedy genetic algorithm to evolve a clew of worms.
"""
import math
import numpy as np
from typing import List, Tuple

from src import rng, NPImage
from src.algorithm import GeneticClewEvolution
from src.algorithms.local_search import score_worm_mask
from src.helpers import clamp
from src.image_manipulation import edge_enhance, local_brighten
from src.worm import CamoWorm, Clew
from src.worm_mask import WormMask


class GreedyClewEvolution(GeneticClewEvolution):
    """
    A basic implementation of clew evolution to get the ball rolling.
    This implementation just makes random mutations and only keeps
    mutations that improve the score of the worm.
    """
    def __init__(self, image, clew: Clew):
        super().__init__(local_brighten(image), clew, name="Greedy")

    def score(self, worm: CamoWorm, worm_mask: WormMask):
        """ A basic benchmark scoring function. """
        score = 100 * score_worm_mask(worm_mask, worm_mask.create_outer_mask())

        # Promotes bigger worms if the worms are already decent.
        score += 0.15 * (1 if score > 0 else -1) * (worm.r + 2 * worm.width)

        # Attempts to avoid overlapping and close worms.
        close_penalty = 0.0
        overlap_penalty = 0.0
        for other_index in range(len(self.clew)):
            other_worm = self.clew[other_index]
            if worm is other_worm:
                continue

            other_worm_mask = self.clew_masks[other_index]
            close_penalty += max(0.0, 400 - worm_mask.midpoint_distance_squared(other_worm_mask))
            overlap_penalty += max(0.0, worm_mask.intersection(other_worm_mask) - 0.2)  # Allow 20% overlap

        score -= 0.1 * close_penalty
        score -= 10 * overlap_penalty

        return score

    def random_mutate(self, worm: CamoWorm, score: float) -> CamoWorm:
        # We make bigger mutations to worms with lower scores.
        temp = 5 * clamp(1 - score / 100, 0.01, 1.0)
        if rng.random() < 0.1:
            temp = 5

        x = worm.x + 4 * rng.standard_normal() * temp
        y = worm.y + 4 * rng.standard_normal() * temp
        r = worm.r + 0.8 * rng.standard_normal() * temp
        theta = worm.theta + math.pi / 2 * rng.standard_normal() * temp
        dr = worm.dr + rng.standard_normal() * temp
        dgamma = worm.dgamma + math.pi * rng.standard_normal() * temp
        width = worm.width + rng.standard_normal() * temp
        colour = worm.colour + 0.05 * rng.standard_normal() * temp

        dr_max = r / 2
        new_worm = CamoWorm(
            clamp(x, 0, self.image.shape[1]),
            clamp(y, 0, self.image.shape[0]),
            clamp(r, 10, 200),
            (theta + 20 * math.pi) % (math.pi / 2),
            clamp(dr, -dr_max, dr_max),
            (dgamma + 20 * math.pi) % math.pi,
            clamp(width, 4, 20),
            clamp(colour, 0, 1)
        )
        return new_worm

    def update(self, worm: CamoWorm, worm_mask: WormMask) -> Tuple[CamoWorm, WormMask]:
        """ A basic greedy update function based upon random mutations to the worm. """
        worm_score = self.score(worm, worm_mask)

        new_worm = worm
        new_worm_mask = worm_mask
        new_score = worm_score

        # Try some random mutations to improve the worm.
        for i in range(10):
            mutated_worm = self.random_mutate(worm, worm_score)
            mutated_worm_mask = WormMask(mutated_worm, self.image)

            mutated_score = self.score(mutated_worm, mutated_worm_mask)
            if mutated_score > new_score:
                new_worm = mutated_worm
                new_worm_mask = mutated_worm_mask
                new_score = mutated_score
                break

        return new_worm, new_worm_mask
