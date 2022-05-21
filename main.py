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
    input_var_2 = float(sys.argv[2])

    with open("./progress/outline.txt", "w") as desc_file:
        desc_file.write(f"{input_var_1 = }\n")
        desc_file.write(f"{input_var_2 = }\n")

    image = crop(load_image("images", "original"), (320, 560, 160, 880))

    with cProfile.Profile() as pr:
        # run_greedy(image, clew_size=10, total_generations=input_var_1)
        # run_quadratic_search_pos()
        run_worm_search_pos(image, input_var_1, generations_per_worm=100,
                            clew_size=100, overlap_image_decay_rate=input_var_2)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")


if __name__ == "__main__":
    main()
