"""
This file represents the main entry point to the CamoWorms program.
"""
from src import crop, load_image

if __name__ == "__main__":
    image = load_image("images", "original", (320, 560, 160, 880))
    print("This doesn't do anything yet")
