"""
This file contains a generic template that can be extended
to implement genetic evolution algorithms to evolve a clew
of camo worms.
"""
import cProfile
import pstats
import time
import numpy as np
from typing import Tuple, Optional, Iterable

from skimage.metrics import structural_similarity

from src import NpImage
from src.image_manipulation import find_median_colour
from src.plotting import Drawing
from src.progress_image_generator import ProgressImageGenerator
from src.worm import CamoWorm
from src.worm_mask import WormMask


class GeneticClewEvolution:
    """
    Provides methods to evolve a clew of camo worms
    to clean up an image using a basic genetic algorithm.
    """

    def __init__(
            self, image: NpImage, initial_clew_size: int,
            *,
            name: str = "Clew",
            evolve_clew_size: bool = True,
            progress_dir: Optional[str] = "progress",
            profile_file: Optional[str] = "profile.prof"):

        self.name = name
        self.initial_clew_size = initial_clew_size
        self.evolve_clew_size = evolve_clew_size
        self.image = image

        self.clew: list[CamoWorm] = []
        self.clew_masks: list[WormMask] = []
        for _ in range(initial_clew_size):
            self.add_new_random_worm()

        for index in range(len(self.clew)):
            colour = self.clew_masks[index].mean_colour()
            if colour is not None:
                self.clew[index].colour = colour

        self.generation = 0
        initial_ssim, _ = self.score_clew()
        self.generation_ssim_scores = [initial_ssim]

        self.progress_image_generator: Optional[ProgressImageGenerator] = None
        if progress_dir is not None:
            self.progress_image_generator = ProgressImageGenerator(
                np.full(image.shape, find_median_colour(image)), progress_dir)
            self.progress_image_generator.save_progress_image(
                self.clew, self.clew_masks, self.generation)

        self.profiler: Optional[cProfile.Profile] = None
        self.profile_file = profile_file
        if profile_file is not None:
            self.profiler = cProfile.Profile()

    def generate_progress_gif(self) -> None:
        """
        Generates a progress gif from the frames that were saved.
        """
        if self.progress_image_generator is None:
            raise Exception("No progress images were saved")
        self.progress_image_generator.generate_gif()

    def save_profiler_stats(self):
        """
        Saves the profiler stats that were recorded.
        """
        if self.profiler is None:
            raise Exception("No profiling was performed")

        stats = pstats.Stats(self.profiler).sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=self.profile_file)

    def score(self, worm: CamoWorm, worm_mask: WormMask, *, for_new_worm=False) -> float:
        """
        Calculates the score of the given worm.
        """
        raise Exception(
            "score is unimplemented for {}".format(type(self).__name__))

    def update(self, worm: CamoWorm, worm_mask: WormMask) -> Tuple[CamoWorm, WormMask]:
        """
        Updates the given camo worm.
        """
        raise Exception(
            "update is unimplemented for {}".format(type(self).__name__))

    def score_clew(self) -> tuple[float, Iterable[float]]:
        """
        Compares the image generated by drawing all worms in the clew
        to the original image. This uses the structural similarity
        index to compare the images. A value of 1 is the best possible
        comparison, and -1 is the worst possible comparison.

        Returns the structural similarity and a list of the scores given
        to each worm.
        """
        generated = self.draw_clew()
        ssim = structural_similarity(self.image, generated, data_range=255)
        scores = np.array([self.score(w, m)
                          for w, m in zip(self.clew, self.clew_masks)])
        return ssim, scores

    def draw_clew(self, image=None):
        """
        Draws the worm masks of the worms in the clew into image.
        """
        if image is None:
            image = np.zeros(self.image.shape)
            bg_colour = find_median_colour(self.image)
            image += bg_colour
        else:
            image = image.copy()

        for worm, mask in zip(self.clew, self.clew_masks):
            mask.draw_into(image, worm.colour * 255.0)

        return image

    def plot_clew(
            self, *,
            show_target_image: bool = False,
            exclude_background: bool = False):
        """
        Draws the current clew of worms on the image.
        """
        if show_target_image:
            drawing = Drawing(self.image)
            drawing.plot(title="Target Image")

        background = 0.25 * 255 + 0.5 * self.image if not exclude_background else None
        drawing = Drawing(self.draw_clew(background))
        drawing.plot(title="{} gen-{}".format(self.name, self.generation))

    def run_generation(self):
        """
        Runs a single generation of evolution.
        """
        self.run_generations(1)

    def run_generations(self, generations: int):
        """
        Runs the given number of generations of evolution.
        """
        if self.profiler is not None:
            self.profiler.enable()

        try:
            for _ in range(generations):
                self._run_generation()
        finally:
            if self.profiler is not None:
                self.profiler.disable()
                self.save_profiler_stats()

    def add_worm(self, worm: CamoWorm, worm_mask: Optional[WormMask] = None):
        """
        Adds a new worm to the clew that is being evolved.
        """
        worm_mask = WormMask.from_worm(worm, self.image) if worm_mask is None else worm_mask
        self.clew.append(worm)
        self.clew_masks.append(worm_mask)

    def add_new_random_worm(self, *, test_worms=100):
        """
        Adds a new random worm to the clew that is being evolved.
        """
        best_worm: Optional[CamoWorm] = None
        best_worm_mask: Optional[WormMask] = None
        best_worm_score: float = 0
        for i in range(test_worms):
            new_worm = CamoWorm.random(self.image.shape)
            new_worm_mask = WormMask.from_worm(new_worm, self.image)
            new_worm.colour = new_worm_mask.mean_colour()
            new_worm_score = self.score(
                new_worm, new_worm_mask, for_new_worm=True)
            if best_worm is None or new_worm_score > best_worm_score:
                best_worm = new_worm
                best_worm_mask = new_worm_mask
                best_worm_score = new_worm_score

        if best_worm is not None:
            self.add_worm(best_worm, best_worm_mask)

    def sort_clew_by_scores(self, worm_scores: Iterable[float]) -> list[float]:
        """
        Sorts the worms by their scores.
        """
        combined = list(zip(worm_scores, self.clew, self.clew_masks))
        combined.sort(key=lambda t: t[0])
        worm_scores = [s for s, _, _ in combined]
        self.clew = [w for _, w, _ in combined]
        self.clew_masks = [m for _, _, m in combined]
        return worm_scores

    def _run_generation(self):
        """
        Evolves a single generation.
        """
        start_time = time.perf_counter()
        self.generation += 1

        changed_worms = 0

        for index in range(len(self.clew)):
            worm = self.clew[index]
            worm_mask = self.clew_masks[index]
            new_worm, new_worm_mask = self.update(worm, worm_mask)
            self.clew[index] = new_worm
            self.clew_masks[index] = new_worm_mask
            if new_worm != worm:
                changed_worms += 1

        # Loop in case we add or remove a worm, so we can re-calculate the scores.
        while True:
            new_ssim, worm_scores = self.score_clew()
            self.generation_ssim_scores.append(new_ssim)

            # Sort the worms by their scores.
            self.sort_clew_by_scores(worm_scores)

            min_worm_score = np.min(worm_scores)
            max_worm_score = np.max(worm_scores)
            std_worm_score = np.std(worm_scores)
            mean_worm_score = np.mean(worm_scores)

            if changed_worms >= 1:
                break

            # If no worms were changed, then add or remove worms.
            if min_worm_score < 0 and len(self.clew) > (self.initial_clew_size + 1) // 2:
                del self.clew[0]
                del self.clew_masks[0]
                changed_worms += 1
                break  # Only remove 1 worm at a time.
            else:
                self.add_new_random_worm()
                changed_worms += 1

        # Save progress.
        if self.progress_image_generator is not None:
            self.progress_image_generator.save_progress_image(
                self.clew, self.clew_masks, self.generation
            )

        duration = time.perf_counter() - start_time
        print(
            "Generation {}: "
            "SSIM = {:.3f}, "
            "Scores = {:.0f} ± {:.0f}, [{:.0f}, {:.0f}], "
            "Worms = {} "
            " (took {:.2f} sec)".format(
                self.generation,
                new_ssim,
                mean_worm_score, std_worm_score, min_worm_score, max_worm_score,
                len(self.clew),
                duration
            )
        )
