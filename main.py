"""
This file represents the main entry point to the CamoWorms program.
"""
import os
import sys
import cProfile
import pstats
from typing import Callable

import imageio
import numpy as np

from src import NpImage
from src.algorithms.greedy import GreedyClewEvolution
from src.image_loading import read_image
from src.implementations.pso_worm import run_worm_search_pos
from src.image_manipulation import threhold_filter, median_window

from multiprocessing import Pool


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


def run_greedy(input_image: NpImage, output_dir: str, args: list[str], do_profile: bool) -> None:
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


def run_particle_swarm(input_image: NpImage, output_dir: str, args: list[str], do_profile: bool) -> None:
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
        run_worm_search_pos(
            input_image, output_dir, no_worms,
            generations_per_worm=100,
            clew_size=100,
            overlap_image_decay_rate=overlap_decay_rate
        )

    maybe_profile(run, do_profile)


def run_algo(
        im_name: str, algo_func: Callable,
        input_image: NpImage, output_dir: str, args: list[str], do_profile: bool) -> None:

    """ Runs the given algorithm function with the given arguments. """
    prefixed_name = ""
    stdout = sys.stdout
    if len(im_name) > 0:
        prefixed_name = " " + im_name
        log_file = open(os.path.join(output_dir, "log.txt"), "w", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    print("Processing{}...".format(prefixed_name), file=stdout)
    algo_func(input_image, output_dir, args, do_profile)
    print("Finished processing{}".format(prefixed_name), file=stdout)


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
    do_thres = False
    do_median = False
    args = list(sys.argv[1:])

    index = 0
    while index < len(args):
        arg = args[index].lower()
        if arg == "--profile":
            del args[index]
        elif arg == "--threshold":
            do_thres = True
            del args[index]
        elif arg == "--median":
            do_median = True
            del args[index]
        else:
            index += 1

    algorithm = args[0].lower()
    input_image_path = args[1]
    output_dir = args[2]
    algo_args = args[3:]

    if not os.path.exists(input_image_path):
        perr("Input image path is not a valid file: {}".format(input_image_path))
        sys.exit(1)

    if os.path.isdir(input_image_path):
        input_image_paths = []
        for im_file in os.listdir(input_image_path):
            im_name, ext = os.path.splitext(im_file)
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            im_path = os.path.join(input_image_path, im_file)
            input_image_paths.append((im_name, im_path))

        if len(input_image_paths) == 0:
            perr("Could find no images in {}".format(input_image_path))
            sys.exit(1)
        elif len(input_image_paths) > 0 and do_profile:
            perr("Cannot profile when more than one image is used as input")
            sys.exit(1)
    else:
        input_image_paths = [("", input_image_path)]

    input_images = []
    for im_name, im_path in input_image_paths:
        try:
            input_image = read_image(im_path)
            input_images.append((im_name, input_image))
        except Exception as e:
            perr("Unable to read input image from {}".format(im_path))
            perr()
            raise e

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        perr("{} is not a directory".format(output_dir))
        sys.exit(1)

    if algorithm == "swarm":
        algo_func = run_particle_swarm
    elif algorithm == "greedy":
        algo_func = run_greedy
    else:
        perr("Unknown algorithm: {}".format(algorithm))
        perr()
        print_help()
        sys.exit(1)

    # Use multiprocessing to run for all images.
    args = []
    for im_name, image in input_images:
        im_output_dir = os.path.join(output_dir, im_name)

        if not os.path.exists(im_output_dir):
            os.makedirs(im_output_dir)
        elif not os.path.isdir(im_output_dir):
            perr("{} is not a directory".format(im_output_dir))
            sys.exit(1)
        
        if(do_thres): 
            threhold_filter(image)
        if(do_median):
            image = median_window(image)

        imageio.imwrite(os.path.join(im_output_dir, "target.png"), np.flipud(image)[:, ::-1].astype(np.uint8))
        args.append((im_name, algo_func, image, im_output_dir, algo_args, do_profile))

    with Pool(len(args)) as pool:
        pool.starmap(run_algo, args)

    print()
    print("Done!")
