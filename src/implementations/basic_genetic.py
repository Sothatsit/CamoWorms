

from dataclasses import dataclass
from typing import cast

import numpy as np
from src import NpImage, rng
from src.algorithms.basic_ga import BasicGeneticAlgorithm
from src.helpers import clamp
from src.progress_image_generator import ProgressImageGenerator, build_gif
from src.worm import CamoWorm, Clew
from src.worm_mask import WormMask


GENERATIONS_PER_ITERATION = 20


@dataclass
class Underlying:
    clew: Clew
    worm_masks: list[WormMask]
    worm_costs: list[float]


def run_basic_genetic(image: NpImage, clew_size: int, number_of_clews: int, total_iterations: int) -> None:
    """ Runs the basic genetic algorithm. """

    def worm_cost(mask: WormMask) -> float:
        """ Cost of an individual worm. """

        return cast(float, np.sum(mask.difference_image()))

    def random_individual() -> Underlying:
        """ Generates a random clew. """

        shape = cast(tuple[int, int], image.shape)

        clew = [CamoWorm.random(shape) for _ in range(clew_size)]
        worm_masks = [WormMask(worm, image) for worm in clew]
        worm_costs = [worm_cost(mask) for mask in worm_masks]

        return Underlying(clew, worm_masks, worm_costs)

    def mutate_worm(worm: CamoWorm) -> CamoWorm:
        """ Mutates the given worm. """
        return CamoWorm(
            worm.x,
            worm.y,
            max(worm.r + rng.standard_normal() * 2, 5),
            worm.theta + np.pi * 0.1 * rng.standard_normal(),
            worm.dr + rng.standard_normal() * 2,
            worm.dgamma + np.pi * 0.1 * rng.standard_normal(),
            max(worm.width + rng.standard_normal() * 2, 5),
            clamp(worm.colour + rng.standard_normal() * 0.1, 0, 1)
        )

    def crossover_function(individual1: Underlying, individual2: Underlying) -> Underlying:
        """ Takes two individuals and crosses them over. """

        cross_index = rng.integers(0, clew_size)

        new_individual = Underlying([], [], [])

        if rng.random() > 0.5:
            new_individual.clew = individual1.clew[:cross_index] + \
                individual2.clew[cross_index:]
            new_individual.worm_masks = individual1.worm_masks[:cross_index] + \
                individual2.worm_masks[cross_index:]
            new_individual.worm_costs = individual1.worm_costs[:cross_index] + \
                individual2.worm_costs[cross_index:]
        else:
            new_individual.clew = individual2.clew[:cross_index] + \
                individual1.clew[cross_index:]
            new_individual.worm_masks = individual2.worm_masks[:cross_index] + \
                individual1.worm_masks[cross_index:]
            new_individual.worm_costs = individual2.worm_costs[:cross_index] + \
                individual1.worm_costs[cross_index:]

        # Do mutation here
        for index, worm in enumerate(new_individual.clew):
            if rng.random() < 0.01:
                new_worm = mutate_worm(worm)
                new_mask = WormMask(new_worm, image)
                new_individual.clew[index] = new_worm
                new_individual.worm_masks[index] = new_mask
                new_individual.worm_costs[index] = worm_cost(new_mask)

        return new_individual

    def match_background_cost(individual: Underlying) -> float:
        """ Calcluates difference between worm and background. """

        return sum(individual.worm_costs)

    algorithm_instance = BasicGeneticAlgorithm[Underlying](
        number_of_clews, match_background_cost, random_individual, crossover_function)

    progress_image_generator = ProgressImageGenerator(
        image, "./progress/match-bg/normal")
    progress_image_generator_white = ProgressImageGenerator(
        np.full(image.shape, 255.0), "./progress/match-bg/white")

    current_error_delta = 1.0
    last_cost = float('inf')

    for _ in range(total_iterations // GENERATIONS_PER_ITERATION):
        results = algorithm_instance.run_generations(GENERATIONS_PER_ITERATION)

        current_error_delta = (
            last_cost - results[-1].costs[0]) / results[-1].costs[0]

        last_cost = results[-1].costs[0]

        print(
            f"({results[-1].generation:04}) {results[-1].costs[0]:.2f} {current_error_delta:.4f} {results[-1].duration*GENERATIONS_PER_ITERATION:.3f}s")

        best_individual = algorithm_instance.population[0].underlying
        progress_image_generator.save_progress_image(
            best_individual.clew, best_individual.worm_masks, algorithm_instance.generation)
        progress_image_generator_white.save_progress_image(
            best_individual.clew, best_individual.worm_masks, algorithm_instance.generation)

    build_gif("./progress/match-bg/normal", "match-bg.gif")
    build_gif("./progress/match-bg/white", "match-bg-white.gif")
