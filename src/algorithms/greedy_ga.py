"""
This file implements a greedy genetic algorithm to evolve a clew of worms.
"""
import math
from typing import Tuple

from src import rng
from src.algorithm import GeneticClewEvolution
from src.algorithms.local_search import score_worm_mask
from src.helpers import clamp
from src.image_manipulation import edge_enhance
from src.worm import CamoWorm, Clew
from src.worm_mask import WormMask


class GreedyClewEvolution(GeneticClewEvolution):
    """
    A basic implementation of clew evolution to get the ball rolling.
    This implementation just makes random mutations and only keeps
    mutations that improve the score of the worm.
    """
    def __init__(self, image, clew_size: int):
        super().__init__(edge_enhance(image), clew_size, name="Greedy")

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
            close_penalty += max(0.0, 1000 - worm_mask.midpoint_distance_squared(other_worm_mask))
            overlap_penalty += max(0.0, worm_mask.intersection(other_worm_mask) - 0.1)  # Allow 10% overlap

        # Stop bad worms from all reaching the same bad solutions.
        if score < 0:
            score -= 0.1 * close_penalty

        score -= 40 * overlap_penalty

        return 0.5 * score

    def random_mutate(self, worm: CamoWorm, score: float) -> CamoWorm:
        # We make bigger mutations to worms with lower scores.
        temp = 5 * clamp(1 - score / 100, 0.02, 1.0)
        if rng.random() < 0.1:
            temp = 5

        # When the score is good, we grow the worm more often.
        # When the score is bad, we shrink the worm more often.
        below_ratio = 1.0 if score < 0 else 0.2
        above_ratio = 0.2 if score < 0 else 1.0
        change_r = 2 * rng.standard_normal() * temp**0.5
        change_width = rng.standard_normal() * temp**0.5
        change_r *= (below_ratio if change_r < 0 else above_ratio)
        change_width *= (below_ratio if change_width < 0 else above_ratio)

        x = worm.x + 5 * rng.standard_normal() * temp
        y = worm.y + 5 * rng.standard_normal() * temp
        r = worm.r + change_r
        theta = worm.theta + math.pi / 2 * rng.standard_normal() * temp
        dr = worm.dr + rng.standard_normal() * temp
        dgamma = worm.dgamma + math.pi * rng.standard_normal() * temp
        width = worm.width + change_width
        colour = worm.colour + 0.05 * rng.standard_normal() * temp

        dr_max = r / 2
        new_worm = CamoWorm(
            clamp(x, 0, self.image.shape[1]),
            clamp(y, 0, self.image.shape[0]),
            clamp(r, 10, 100 if score < 0 else 600),
            (theta + 20 * math.pi) % math.pi,
            clamp(dr, -dr_max, dr_max),
            # If the deviation is small, then the angle doesn't matter, so standardise it.
            math.pi / 2 if abs(dr) < r/10 else (dgamma + 20 * math.pi) % math.pi,
            clamp(width, 4, 8 if score < 0 else 20),
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
            mutated_worm.colour = mutated_worm_mask.median_colour()

            mutated_score = self.score(mutated_worm, mutated_worm_mask)
            if mutated_score > new_score:
                new_worm = mutated_worm
                new_worm_mask = mutated_worm_mask
                new_score = mutated_score
                break

        return new_worm, new_worm_mask
