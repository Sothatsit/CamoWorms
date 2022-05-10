"""
This file represents the main entry point to the CamoWorms program.
"""
from src.image_loading import load_image
from src.image_manipulation import crop
from src.algorithms.greedy_ga import GreedyClewEvolution
from src.progress_image_generator import build_gif

if __name__ == "__main__":
    image = crop(load_image("images", "original"), (320, 560, 160, 880))

    ga = GreedyClewEvolution(image, 10)
    ga.run_generations(100)

    build_gif("./progress", "progress.gif")
