"""
Contains functions for plotting images, worms, and dots.
"""
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Drawing:
    """
    Makes it easier to plot images and worms.
    """
    def __init__(self, image):
        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.image = image
        self.im = self.ax.imshow(self.image, vmin=0, vmax=255, cmap='gray', origin='lower')

    def add_patches(self, patches):
        try:
            for patch in patches:
                self.ax.add_patch(patch)
        except TypeError:
            self.ax.add_patch(patches)

    def add_dots(self, points, radius=4, **kwargs):
        try:
            for point in points:
                self.ax.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
        except TypeError:
            self.ax.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))

    def add_worms(self, worms):
        try:
            self.add_patches([w.patch() for w in worms])
        except TypeError:
            self.add_patches([worms.patch()])

    def plot(self, *, save: Optional[str] = None, title: Optional[str] = None, do_show: bool = True):
        if title is not None:
            plt.title(title, fontdict={"fontsize": 20, "fontweight": 700})
        if save is not None:
            plt.savefig(save)
        if do_show:
            plt.show()
        plt.close(self.fig)
