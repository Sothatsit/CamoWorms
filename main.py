"""
This file represents the main entry point to the CamoWorms program.
"""
import sys
from src.image_loading import load_image
from src.image_manipulation import crop
from src.implementations.basic_genetic import run_basic_genetic
from src.implementations.greedy import run_greedy

import cProfile
import pstats


def main() -> None:
    total_generations = int(sys.argv[1])

    image = crop(load_image("images", "original"), (320, 560, 160, 880))

    with cProfile.Profile() as pr:
        # run_basic_genetic(image, clew_size=10,
        #                   number_of_clews=100, total_iterations=400)
        run_greedy(image, clew_size=10, total_generations=total_generations)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")


if __name__ == "__main__":
    main()
