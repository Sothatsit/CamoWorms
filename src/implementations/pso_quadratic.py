"""
Implementation of the PSO algorithm finding the minimum of a quadratic
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.algorithms.particle_swarm import ParticleSwarmOptimisation, psoNDarray


def build_gif() -> None:
    frame_directory = Path("frames/")

    frame_files = [file for file in frame_directory.iterdir()
                   if file.is_file()]

    frame_files.sort()

    with imageio.get_writer("mygif.gif", mode="I", duration=1/10) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)

            writer.append_data(frame)


def run_quadratic_search_pos() -> None:
    """
    Each individual will be a point on the 2d plane.

    cost will be x^2 + y^2
    """

    def cost_function(individual: psoNDarray) -> float:
        return (individual.item(0) - 10)**2 + (individual.item(1) + 3)**2

    def random_individual_function() -> psoNDarray:
        return np.array([np.random.normal(), np.random.normal()], dtype=np.float64)

    def momentum_function(generation: int) -> float:
        return 0.9 - 0.5 * (generation / 100)

    algorithm_instance = ParticleSwarmOptimisation(
        population_size=50,
        individual_size=2,
        cost_function=cost_function,
        random_individual_function=random_individual_function,
        random_direction_function=random_individual_function,
        w=momentum_function,
        c1=2,
        c2=2
    )

    results = algorithm_instance.run_generations(50)

    plt.xlim([-15, 15])
    plt.ylim([-15, 15])

    for index, result in enumerate(results):
        plt.scatter(result.population[0, :], result.population[1, :])
        plt.savefig(f"frames/{index:05}.png")

    build_gif()
