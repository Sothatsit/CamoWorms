import os
import imageio
from pathlib import Path
import numpy as np
import pygifsicle
from src import NpImage
from src.image_manipulation import find_median_colour
from src.plotting import Drawing
from src.worm import Clew
from src.worm_mask import WormMask


def build_gif(frame_directory_path: str, destination: str) -> None:
    """ Creates a gif from the frames in frame_directory. """

    frame_directory = Path(frame_directory_path)

    frame_files = [file for file in frame_directory.iterdir()
                   if file.is_file()]

    frame_files.sort()

    with imageio.get_writer(destination, mode="I") as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)

            # Crop to interesting part of frame
            frame = frame[120:-155, 81:-64]

            writer.append_data(frame)

    pygifsicle.optimize(destination)


def create_and_empty_directory(directory_path: str) -> None:
    """ Creates the given directory if it does not exist. Cleans the directory
    of any files matching progress image name format. """

    directory = Path(directory_path)

    # Make if it does not exist, otherwise no action
    directory.mkdir(parents=True, exist_ok=True)

    # Remove files from directory
    for child in directory.iterdir():
        if child.is_file():
            child.unlink()


class ProgressImageGenerator:
    """ Used to save progress images of a clew """

    def __init__(self, base_image: NpImage, run_title: str, progress_dir: str):
        self.__run_title = run_title
        self.__progress_dir = progress_dir

        self.__base_median_colour = find_median_colour(base_image)
        self.__base_image_shape = base_image.shape

        create_and_empty_directory(self.__progress_dir)

    def generate_gif(self):
        """ Generates a gif with the same name as the progress directory. """
        build_gif(self.__progress_dir, f"{self.__progress_dir}.gif")

    def save_progress_image(self, clew: Clew, worm_masks: list[WormMask], generation_num: int) -> None:
        """ Saves a progress image of the clew. """

        image = np.full(self.__base_image_shape, self.__base_median_colour)

        for worm, mask in zip(clew, worm_masks):
            mask.draw_into(image, worm.colour * 255.0)

        drawing = Drawing(image)

        file_path = os.path.join(
            self.__progress_dir, f"gen-{generation_num:06}.png"
        )
        drawing.plot(
            title=f"{self.__run_title} gen-{generation_num:06}",
            save=file_path,
            do_show=False
        )
