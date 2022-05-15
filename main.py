"""
This file represents the main entry point to the CamoWorms program.
"""
import sys
import cProfile
import pstats
import imageio
from src.image_loading import load_image
from src.image_manipulation import crop
from src.implementations.basic_genetic import run_basic_genetic
from src.implementations.greedy import run_greedy
from src.implementations.pso_quadratic import run_quadratic_search_pos
from src.implementations.pso_worm import run_worm_search_pos


def main() -> None:
    total_generations = int(sys.argv[1])

    image = crop(load_image("images", "original"), (320, 560, 160, 880))

    with cProfile.Profile() as pr:
        # run_basic_genetic(image, clew_size=10,
        #                   number_of_clews=100, total_iterations=400)
        # run_greedy(image, clew_size=10, total_generations=total_generations)
        # run_quadratic_search_pos()
        run_worm_search_pos(image, 200)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")


if __name__ == "__main__":
    main()
