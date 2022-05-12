"""
Runs a standard setup of the greedy algorithm.
"""

from src import NpImage
from src.algorithms.greedy import GreedyClewEvolution


def run_greedy(image: NpImage, clew_size: int, total_generations: int) -> None:
    greedy = GreedyClewEvolution(
        image, clew_size, profile_file=None)
    greedy.run_generations(total_generations)
    greedy.generate_progress_gif()
