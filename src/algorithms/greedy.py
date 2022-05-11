"""
This file implements a greedy genetic algorithm to evolve a clew of worms.
"""
import math
from typing import Tuple, Optional

from src import rng
from src.algorithm import GeneticClewEvolution
from src.algorithms.local_search import score_worm_isolated
from src.helpers import clamp
from src.image_manipulation import edge_enhance
from src.worm import CamoWorm, Clew
from src.worm_mask import WormMask


class GreedyClewEvolution(GeneticClewEvolution):
    """
    This greedy algorithm makes random mutations to the worms in the clew
    and only keeps mutations that improve the score of the worm.
    """
    def __init__(self, image, clew_size: int):
        super().__init__(edge_enhance(image), clew_size, name="Greedy")

    def score(self, worm: CamoWorm, worm_mask: WormMask, *, allowed_overlap=1):
        """ A basic benchmark scoring function. """
        score = 100 * score_worm_isolated(worm, worm_mask, worm_mask.create_outer_mask())

        # Promotes bigger worms if the worms are already decent.
        score += 0.15 * (1 if score > 0 else -1) * (worm.r + 2 * worm.width)

        # Attempts to avoid overlapping and close worms.
        avoid_close = score < 0
        avoid_overlap = allowed_overlap < 1
        if avoid_close and avoid_overlap:
            close_penalty = 0.0
            overlap_penalty = 0.0
            for other_index in range(len(self.clew)):
                other_worm = self.clew[other_index]
                if worm is other_worm:
                    continue

                other_worm_mask = self.clew_masks[other_index]
                if avoid_close:
                    close_penalty += max(0.0, 2000 - worm_mask.midpoint_distance_squared(other_worm_mask))

                if avoid_overlap:
                    intersection = worm_mask.intersection(other_worm_mask)
                    overlap_penalty += max(0.0, intersection - allowed_overlap) / (1 - allowed_overlap)

            score -= 0.1 * close_penalty
            score -= max(2 * score, 50) * overlap_penalty

        return 0.5 * score

    def _random_mutate(self, worm: CamoWorm, score: float, *, min_temp=0.2, drastic_chance=0.1) -> CamoWorm:
        # We make bigger mutations to worms with lower scores.
        temp = min_temp + (1 - min_temp) * clamp(1 - score / 100, 0, 1)
        if rng.random() < drastic_chance:
            temp = 1
        temp = min_temp + (1 - min_temp) * ((temp - min_temp) / (1 - min_temp))**2

        # When the score is good, we grow the worm more often.
        # When the score is bad, we shrink the worm more often.
        below_ratio = 1.0 if score < 0 else 0.2
        above_ratio = 0.2 if score < 0 else 1.0
        change_r = 4 * rng.standard_normal() * (1.1 - temp)
        change_width = 3 * rng.standard_normal() * (1.1 - temp)
        change_r *= (below_ratio if change_r < 0 else above_ratio)
        change_width *= (below_ratio if change_width < 0 else above_ratio)

        x = worm.x + 5 * rng.standard_normal() * temp
        y = worm.y + 5 * rng.standard_normal() * temp

        # 50% chance to move sideways if the length of the worm changed.
        if change_r > 0.5 and rng.random() < 0.5:
            # 50/50 move left or right.
            dir_angle = worm.theta + (0 if rng.random() < 0.5 else math.pi)
            x += change_r * math.cos(dir_angle)
            y += change_r * math.sin(dir_angle)

        r = worm.r + change_r
        theta = worm.theta + math.pi / 8 / (worm.r**0.3) * rng.standard_normal() * temp
        dr = worm.dr + 0.3 * worm.r * rng.standard_normal() * temp
        dgamma = worm.dgamma + math.pi / 4 * rng.standard_normal() * temp
        width = worm.width + change_width
        colour = worm.colour + 0.05 * rng.standard_normal() * temp

        # If the deviation is small, then the angle doesn't matter that much, so standardise it.
        # This helps, as otherwise when dr is small, the dgamma value is effectively random
        # as it has very little effect on the shape of the worm.
        if abs(dr) < min(r/20, 3):
            dgamma = math.pi/2

        dr_max = r
        new_worm = CamoWorm(
            clamp(x, 0, self.image.shape[1]),
            clamp(y, 0, self.image.shape[0]),
            clamp(r, 10, 100 if score < 0 else 600),
            (theta + 20 * math.pi) % math.pi,
            clamp(dr, -dr_max, dr_max),
            # If the deviation is small, then the angle doesn't matter that much, so standardise it.
            (dgamma + 20 * math.pi) % math.pi,
            clamp(width, 4, 8 if score < 0 else 20),
            clamp(colour, 0, 1)
        )
        return new_worm

    def random_mutate(self, worm: CamoWorm, score: float, **kwargs) -> Optional[CamoWorm]:
        """ Repeatedly mutates the given worm until a new worm is mutated. """
        for i in range(100):
            new_worm = self._random_mutate(worm, score, **kwargs)
            if new_worm != worm:
                return new_worm

        # We weren't able to mutate a new worm...
        return None

    def update(self, worm: CamoWorm, worm_mask: WormMask) -> Tuple[CamoWorm, WormMask]:
        """ A basic greedy update function based upon random mutations to the worm. """
        worm_score = self.score(worm, worm_mask)

        new_worm = worm
        new_worm_mask = worm_mask
        new_score = worm_score

        # Try some random mutations to improve the worm.
        for i in range(20):
            mutated_worm = self.random_mutate(worm, worm_score)
            if mutated_worm is None:
                continue

            mutated_worm_mask = WormMask(mutated_worm, self.image)
            median_colour = mutated_worm_mask.median_colour()
            if median_colour is not None:
                mutated_worm.colour = median_colour

            mutated_score = self.score(mutated_worm, mutated_worm_mask)
            if mutated_score > new_score:
                new_worm = mutated_worm
                new_worm_mask = mutated_worm_mask
                new_score = mutated_score

        return new_worm, new_worm_mask
