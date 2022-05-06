from dataclasses import dataclass, field
import time
from typing import Callable, TypeAlias
from src.worm import Camo_Worm, Clew
import random

RandomClewFunction: TypeAlias = Callable[[], Clew]
CrossoverFunction: TypeAlias = Callable[[Clew, Clew], Clew]
ClewCostFunction: TypeAlias = Callable[[Clew], float]


@dataclass
class GenerationResult:
    generation: int
    duration: float
    costs: list[float]


@dataclass(order=True)
class Individual:
    cost: float
    clew: Clew = field(compare=False)


class ClewBasedGeneticAlgorithm:
    """
    Provides methods to encapsulate the evolution of a population of clews.

    i.e. each clew is one member of the population
    """

    def __init__(
            self,
            population_size: int,
            cost_function: ClewCostFunction,
            random_clew_function: RandomClewFunction,
            random_crossover_function: CrossoverFunction):

        self.population_size = population_size
        self.cost_function = cost_function
        self.random_clew_function = random_clew_function
        self.random_crossover_function = random_crossover_function

        self.population: list[Individual] = []
        self.generation: int = 0
        self.half_pop: int = self.population_size // 2

        for _ in range(population_size):
            clew = self.random_clew_function()
            cost = self.cost_function(clew)

            self.population.append(Individual(cost, clew))

        self.population.sort()

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

        for individual in self.population[self.half_pop:]:
            if random.random() > 0.05:
                individual.clew = self.random_crossover_function(
                    individual.clew,
                    random.choice(self.population[:self.half_pop]).clew)
            else:
                individual.clew = self.random_clew_function()
            individual.cost = self.cost_function(individual.clew)

        self.population.sort()

        costs = [individual.cost for individual in self.population]

        self.generation += 1

        duration = time.perf_counter() - start_time

        return GenerationResult(self.generation, duration, costs)
