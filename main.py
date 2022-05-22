"""
This file represents the main entry point to the CamoWorms program.
"""
import os
import sys
import cProfile
import pstats
from typing import Callable

from src import NpImage
from src.algorithms.greedy import GreedyClewEvolution
from src.image_loading import read_image
from src.implementations.pso_worm import run_worm_search_pos


def maybe_profile(func: Callable, do_profile: bool):
    if not do_profile:
        func()
        return

    # Profile!
    with cProfile.Profile() as pr:
        func()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profile.prof")


def run_greedy(
        input_image: NpImage, output_dir: str, args: list[str], *,
        do_profile: bool = False) -> None:

    if len(args) < 2:
        perr("CamoWorms Greedy Usage:")
        perr("  python -m main greedy <input image> <output directory> <initial no. worms> <no. generations>")
        perr()
        perr("Example:")
        perr("  python -m main greedy ./images/original-cropped.png ./progress 50")
        perr()
        sys.exit(1)

    initial_no_worms = int(args[0])
    no_generations = int(args[1])

    with open(os.path.join(output_dir, "outline.txt"), "w") as desc_file:
        desc_file.write(f"{initial_no_worms = }\n")
        desc_file.write(f"{no_generations = }\n")

    def run():
        greedy = GreedyClewEvolution(input_image, initial_no_worms, progress_dir=output_dir, profile_file=None)
        greedy.run_generations(no_generations)
        greedy.generate_progress_gif()

    maybe_profile(run, do_profile)


def run_particle_swarm(
        input_image: NpImage, output_dir: str, args: list[str], *,
        do_profile: bool = False) -> None:

    if len(args) < 2:
        perr("CamoWorms Particle Swarm Usage:")
        perr("  python -m main swarm <input image> <output directory> <no. worms> <overlap decay rate>")
        perr()
        perr("Example:")
        perr("  python -m main swarm ./images/original-cropped.png ./progress 50 0.9")
        perr()
        sys.exit(1)

    no_worms = int(args[0])
    overlap_decay_rate = float(args[1])

    with open(os.path.join(output_dir, "outline.txt"), "w") as desc_file:
        desc_file.write(f"{no_worms = }\n")
        desc_file.write(f"{overlap_decay_rate = }\n")

    def run():
        run_worm_search_pos(input_image, no_worms, generations_per_worm=100,
                            clew_size=100, overlap_image_decay_rate=overlap_decay_rate)

    maybe_profile(run, do_profile)


def perr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def print_help():
    perr("CamoWorms Usage:")
    perr("  python -m main <algorithm> <input image> <output directory>")
    perr()
    perr("Options:")
    perr("  --profile - Generates a file profile.prof with profiling stats")
    perr()
    perr("Algorithms:")
    perr("  greedy - Greedy Evolution")
    perr("  swarm - Particle Swarm")
    perr()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_help()
        sys.exit(1)

    do_profile = False
    args = list(sys.argv[1:])
    for index in range(len(args)):
        if args[index].lower() == "--profile":
            do_profile = True
            del args[index]
            break

    algorithm = args[0].lower()
    input_image_path = args[1]
    output_dir = args[2]
    algo_args = args[3:]

    if not os.path.exists(input_image_path) or not os.path.isfile(input_image_path):
        perr("Input image path is not a valid file: {}".format(input_image_path))
        sys.exit(1)

    try:
        input_image = read_image(input_image_path)
    except Exception as e:
        perr("Unable to read input image from {}".format(input_image_path))
        perr()
        raise e

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        perr("{} is not a directory".format(output_dir))
        sys.exit(1)

    if algorithm == "swarm":
        run_particle_swarm(input_image, output_dir, algo_args, do_profile=do_profile)
    elif algorithm == "greedy":
        run_greedy(input_image, output_dir, algo_args, do_profile=do_profile)
    else:
        perr("Unknown algorithm: {}".format(algorithm))
        perr()
        print_help()
        sys.exit(1)
