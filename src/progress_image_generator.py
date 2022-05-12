import os
import imageio
from pathlib import Path
import numpy as np
import pygifsicle
from src import NpImage
from src.worm import Clew
from src.worm_mask import WormMask


def build_gif(frame_directory_path: str, destination: str) -> None:
    """ Creates a gif from the frames in frame_directory. """

    frame_directory = Path(frame_directory_path)

    frame_files = [file for file in frame_directory.iterdir()
                   if file.is_file()]

    frame_files.sort()

    with imageio.get_writer(destination, mode="I", duration=1/30) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
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

    def __init__(self, base_image: NpImage, progress_dir: str):
        self.__progress_dir = progress_dir
        self.__base_image = base_image.copy()

        create_and_empty_directory(f"{self.__progress_dir}/frames")

    def generate_gif(self) -> None:
        """ Generates a gif with the same name as the progress directory. """
        build_gif(f"{self.__progress_dir}/frames",
                  f"{self.__progress_dir}/animation.gif")

    def save_progress_image(self, clew: Clew, worm_masks: list[WormMask], generation_num: int) -> None:
        """ Saves a progress image of the clew. """

        image = self.__base_image.copy()

        for worm, mask in zip(clew, worm_masks):
            mask.draw_into(image, worm.colour * 255.0)

        image = np.rot90(image, 2)

        imageio.imwrite(os.path.join(
            self.__progress_dir, "frames", f"gen-{generation_num:06}.png"
        ), image.astype(np.uint8))
