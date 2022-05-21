"""
This file represents the main entry point to the CamoWorms program.
"""
import sys
import cProfile
import pstats
from src.image_loading import load_image
from src.image_manipulation import crop
from src.implementations.greedy import run_greedy
from src.implementations.pso_quadratic import run_quadratic_search_pos
from src.implementations.pso_worm import run_worm_search_pos


def main() -> None:
    input_var_1 = int(sys.argv[1])

    image = crop(load_image("images", "original"), (320, 560, 160, 880))

    with cProfile.Profile() as pr:
        # run_greedy(image, clew_size=10, total_generations=input_var_1)
        # run_quadratic_search_pos()
        run_worm_search_pos(image, input_var_1, generations_per_worm=100, clew_size=100)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")


if __name__ == "__main__":
    main()
