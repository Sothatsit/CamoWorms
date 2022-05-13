import numpy as np
from dataclasses import dataclass, field
import time
# from typing import Callable, Generic, TypeAlias, TypeVar
from typing import Callable, Generic, TypeVar
from copy import deepcopy

# np.random.seed(1)


IndividualType = TypeVar('IndividualType')

# RandomIndividualFunction: TypeAlias = Callable[[], IndividualType]
# CostFunction: TypeAlias = Callable[[IndividualType], float]
# IndividualToVector: TypeAlias = Callable[[IndividualType], list[float]]
# VectorToIndividual: TypeAlias = Callable[list[float], IndividualType]



# def clew_to_vector(clew):
#     vector=[]
#     for worm in clew:
#         vector.extend(worm.x, worm.y, worm.r, worm.theta, worm.dr, worm.dgamma, worm.width, worm.colour)
#     return vector
#
# def vector_to_clew(vector):
#     clew: Clew=[]
#     Clew = [CamoWorm(vector[i:i+8]) for i in range(0, len(vector), 8)]
#     return Clew



@dataclass
class GenerationResult:
    generation: int
    duration: float
    current_best: IndividualType = field(compare=False)
    costs: list[float]


@dataclass(order=True)
class Individual(Generic[IndividualType]):
    cost: float
    underlying: IndividualType = field(compare=False)
    best_cost: float
    best: IndividualType = field(compare=False)
    direction: IndividualType = field(compare=False)


class ParticleSwarmOptimisation(Generic[IndividualType]):
    """
    A generically typed particle swarm optimisation algorithm.
    """

    def __init__(
            self,
            population_size: int,
            # cost_function: CostFunction[IndividualType],
            # random_individual_function: RandomIndividualFunction[IndividualType],
            # random_direction_function: RandomIndividualFunction[IndividualType],
            # individual_to_vector: IndividualToVector,
            # vector_to_individual: VectorToIndividual,
            cost_function,
            random_individual_function,
            random_direction_function,
            individual_to_vector,
            vector_to_individual,
            w: float,
            c1: float,
            c2: float):
        self.population_size = population_size
        self.cost_function = cost_function
        self.random_individual_function = random_individual_function
        self.random_direction_function = random_direction_function
        self.individual_to_vector = individual_to_vector
        self.vector_to_individual = vector_to_individual
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.population: list[Individual[IndividualType]] = []
        self.bests: list[Individual[IndividualType]] = []
        self.generation: int = 0

        self.overall_best: Individual[IndividualType]

        mincost = np.inf
        for _ in range(population_size):
            underlying_individual = self.random_individual_function()
            cost = self.cost_function(underlying_individual)
            direction = self.random_direction_function()
            self.population.append(Individual(cost, underlying_individual, cost, deepcopy(underlying_individual), direction))
            if cost < mincost:
                self.overall_best = deepcopy(self.population[-1])
                mincost = cost

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
        x=[]
        v = []
        pbest=[]
        costs=[]

        for individual in self.population:
            x.append(self.individual_to_vector(individual.underlying))
            pbest.append(self.individual_to_vector(individual.best))
            v.append(self.individual_to_vector(individual.direction))

        x = np.transpose(x)
        v = np.transpose(v)
        pbest = np.transpose(pbest)
        gbest = np.array(self.individual_to_vector(self.overall_best.underlying))

        r = np.random.rand(2)
        v = self.w * v + self.c1 * r[0] * (pbest - x) + self.c2 * r[1] * (gbest[..., None] - x)
        x = x+v

        for count, individual in enumerate(self.population):
            individual.direction=self.vector_to_individual(v[:,count])
            individual.underlying = self.vector_to_individual(x[:,count])
            individual.cost = self.cost_function(individual.underlying)
            costs.append(individual.cost)
            if individual.cost < individual.best_cost:
                individual.best = deepcopy(individual.underlying)
                individual.best_cost = individual.cost
                if individual.cost < self.overall_best.cost:
                    self.overall_best = deepcopy(individual)

        self.population.sort()

        self.generation += 1

        duration = time.perf_counter() - start_time

        return GenerationResult(self.generation, duration, self.overall_best.underlying, [self.overall_best.cost])



