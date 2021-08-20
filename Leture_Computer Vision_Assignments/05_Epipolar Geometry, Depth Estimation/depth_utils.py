import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot3D(depth, gridsize, gridpos, gridspan):
    xs = np.arange(0, depth.shape[1])
    ys = np.arange(depth.shape[0], 0, -1)
    xs, ys = np.meshgrid(xs, ys)

    ax = plt.subplot2grid(gridsize, gridpos, projection='3d', rowspan=gridspan[0], colspan=gridspan[1])
    ax.plot_surface(xs, ys, -depth, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    ax.axis("off")
    ax.set_title("3D from depth")

    ax.view_init(80, -90)
