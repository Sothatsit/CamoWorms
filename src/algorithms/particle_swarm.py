import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
import time
from typing import Callable, TypeAlias, TypeVar


IndividualType = TypeVar('IndividualType')
psoNDarray = npt.NDArray[np.float64]

RandomDirectionFunction: TypeAlias = Callable[[], psoNDarray]
RandomIndividualFunction: TypeAlias = Callable[[], psoNDarray]
MomentumFunction: TypeAlias = Callable[[int], float]
CostFunction: TypeAlias = Callable[[psoNDarray], float]


@dataclass
class GenerationResult():
    generation: int
    duration: float
    costs: list[float]
    population: psoNDarray


class ParticleSwarmOptimisation():
    """
    A generically typed particle swarm optimisation algorithm.
    """

    def __init__(
            self,
            population_size: int,
            individual_size: int,
            cost_function: CostFunction,
            random_individual_function: RandomIndividualFunction,
            random_direction_function: RandomDirectionFunction,
            w: MomentumFunction,
            c1: float,
            c2: float):
        # Generic functions
        self.cost_function = cost_function
        self.random_direction_function = random_direction_function

        # PSO parameters and weights
        self.population_size = population_size
        self.individual_size = individual_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.generation = 0

        # PSO population
        self.population: psoNDarray = np.zeros((self.individual_size, self.population_size), dtype=np.float64)
        self.direction: psoNDarray = np.zeros((self.individual_size, self.population_size), dtype=np.float64)
        self.local_best_position: psoNDarray = np.zeros((self.individual_size, self.population_size), dtype=np.float64)
        self.local_best_cost: psoNDarray = np.full((self.population_size), np.inf, dtype=np.float64)
        self.global_best_position: psoNDarray = np.zeros((self.individual_size, self.population_size), dtype=np.float64)
        self.global_best_cost = np.inf

        # Populate
        for index in range(population_size):
            self.population[:, index] = random_individual_function()
            self.local_best_position[:, index] = self.population[:, index].copy()
            self.local_best_cost[index] = self.cost_function(self.population[:, index])
            self.direction[:, index] = random_direction_function()

            if self.local_best_cost[index] < self.global_best_cost:
                # Fill the global_best matrix with this item
                self.global_best_position[:, :] = self.population[:, index].copy()[:, None]
                self.global_best_cost = self.local_best_cost[index]

    def run_generations(self, generations: int) -> list[GenerationResult]:
        """
        Runs the given number of generations of evolution.
        """

        return [self.run_generation() for _ in range(generations)]

    def run_generation(self) -> GenerationResult:
        """
        Evolves a single generation.
        """
        start_time = time.perf_counter()

        # ----- UPDATE STEP ----- #

        # 1 x 2 array of random variables [0, 1)
        r = np.random.rand(2)

        # individual_size x population_size array
        momentum_movement = self.w(self.generation) * self.direction
        individual_movement: psoNDarray = self.c1 * r[0] * (self.local_best_position - self.population)
        social_movement: psoNDarray = self.c2 * r[1] * (self.global_best_position - self.population)
        self.direction = momentum_movement + individual_movement + social_movement

        # individual_size x population_size array
        self.population = self.population + self.direction

        new_costs: psoNDarray = np.apply_along_axis(self.cost_function, 0, self.population)

        for index, new_cost in enumerate(new_costs):
            if new_cost < self.local_best_cost[index]:
                self.local_best_position[:, index] = self.population[:, index]
                self.local_best_cost[index] = new_cost

            if new_cost < self.global_best_cost:
                # I don't think I actually need to do the reshape here...
                # Would probably be more efficient do to somewhere else
                self.global_best_position[:, :] = self.population[:, index].copy()[:, None]
                self.global_best_cost = new_cost

        self.generation += 1

        duration = time.perf_counter() - start_time

        return GenerationResult(self.generation, duration, self.local_best_cost.tolist(), self.population.copy())
