
import imageio
import numpy as np
from src import NpImage
from src.algorithms.local_search import score_worm_isolated
from src.algorithms.particle_swarm import ParticleSwarmOptimisation, psoNDarray
from src.progress_image_generator import ProgressImageGenerator
from src.worm import CamoWorm
from src.plotting import Drawing
from src.worm_mask import WormMask


def logistic(input: psoNDarray) -> psoNDarray:
    """
    Implements the logistic function for a ndarray input
    """
    result: psoNDarray = 1.0 / (1.0 + np.exp(-input))
    return result


def map_vector_to_worm(image: NpImage, worm_vector: psoNDarray) -> CamoWorm:
    """
    Maps a vector of length 8 to a Camo Worm
    """

    x_lim = image.shape[1]
    y_lim = image.shape[0]

    mapped_vector = logistic(worm_vector)

    worm = CamoWorm(
        x=mapped_vector.item(0) * x_lim,
        y=mapped_vector.item(1) * y_lim,
        r=5 + mapped_vector.item(2) * 50,
        theta=2 * np.pi * mapped_vector.item(3),
        deviation_r=5 + mapped_vector.item(4) * 50,
        deviation_gamma=2 * np.pi * mapped_vector.item(5),
        width=3 + mapped_vector.item(6) * 30,
        colour=mapped_vector.item(7))

    return worm


def run_worm_search_pos(image: NpImage, total_worms: int, *, generations_per_worm: int = 100, clew_size: int = 50) -> None:
    """
    Tries to optimise a single worm.

    Each worm is a length 8 array.
    [x, y, r, theta, dr, dgamma, width, colour]
    """

    overlap_image: NpImage = np.zeros(image.shape, dtype=np.float64)

    def cost_function(individual: psoNDarray) -> float:
        """
        Maps -inf -> +inf of each dimension to appropriate values to construct worm.

        Runs a basic colour match check
        """

        worm = map_vector_to_worm(image, individual)
        mask = WormMask(worm, image)
        outer_mask = mask.create_outer_mask()

        main_cost = -score_worm_isolated(worm, mask, outer_mask)

        blank_image: NpImage = np.zeros(image.shape, dtype=np.float64)
        mask.draw_into(blank_image, 255)
        overlap_cost: float = np.sum(blank_image * overlap_image)

        return main_cost + overlap_cost

    def random_individual_function() -> psoNDarray:
        return np.array([
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal()
        ], dtype=np.float64)

    def momentum_function(generation: int) -> float:
        return 0.9 - 0.6 * (generation / generations_per_worm)

    progress_image_generator = ProgressImageGenerator(
        image, "./progress/")

    worms = []
    masks = []

    for index in range(total_worms):

        algorithm_instance = ParticleSwarmOptimisation(
            population_size=clew_size,
            individual_size=8,
            cost_function=cost_function,
            random_individual_function=random_individual_function,
            random_direction_function=random_individual_function,
            w=momentum_function,
            c1=2,
            c2=2
        )

        result = algorithm_instance.run_generations(generations_per_worm)[-1]

        min_index = np.argmin(result.costs)

        worm_vector = result.population[:, min_index]
        worm = map_vector_to_worm(image, worm_vector)
        worm_mask = WormMask(worm, image)

        worm_mask.draw_into(overlap_image, 255)
        imageio.imwrite(f"progress/mask/gen-{index:06}.png", np.rot90(overlap_image, 2).astype(np.uint8))

        worms.append(worm)
        masks.append(worm_mask)

        progress_image_generator.save_progress_image(worms, masks, index)

    progress_image_generator.generate_gif()
