from dataclasses import dataclass, field
import time
from typing import Callable, Generic, TypeAlias, TypeVar
import random

IndividualType = TypeVar('IndividualType')

RandomIndividualFunction: TypeAlias = Callable[[], IndividualType]
CrossoverFunction: TypeAlias = Callable[[
    IndividualType, IndividualType], IndividualType]
CostFunction: TypeAlias = Callable[[IndividualType], float]


@dataclass
class GenerationResult:
    generation: int
    duration: float
    costs: list[float]


@dataclass(order=True)
class Individual(Generic[IndividualType]):
    cost: float
    underlying: IndividualType = field(compare=False)


class BasicGeneticAlgorithm(Generic[IndividualType]):
    """
    A simple generically typed genetic algorithm.
    """

    def __init__(
            self,
            population_size: int,
            cost_function: CostFunction[IndividualType],
            random_individual_function: RandomIndividualFunction[IndividualType],
            crossover_function: CrossoverFunction[IndividualType]):

        self.population_size = population_size
        self.cost_function = cost_function
        self.random_individual_function = random_individual_function
        self.crossover_function = crossover_function

        self.population: list[Individual[IndividualType]] = []
        self.generation: int = 0
        self.half_pop: int = self.population_size // 2

        for _ in range(population_size):
            underlying_individual = self.random_individual_function()
            cost = self.cost_function(underlying_individual)

            self.population.append(Individual(cost, underlying_individual))

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
            individual.underlying = self.crossover_function(
                individual.underlying,
                random.choice(self.population[:self.half_pop]).underlying)
            individual.cost = self.cost_function(individual.underlying)

        self.population.sort()

        costs = [individual.cost for individual in self.population]

        self.generation += 1

        duration = time.perf_counter() - start_time

        return GenerationResult(self.generation, duration, costs)
