# File:       PixelGrid.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np

eps = 1e-10


def make_pixel_grid(xlims, zlims, dx, dz):
    """ Generate a pixel grid based on input parameters. """
    x = np.arange(xlims[0], xlims[1] + eps, dx)
    z = np.arange(zlims[0], zlims[1] + eps, dz)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid


def make_foctx_grid(rlims, dr, oris, dirs):
    """ Generate a pixel grid based on input parameters. """
    # Get focusing positions in rho-theta coordinates
    r = np.arange(rlims[0], rlims[1] + eps, dr)  # Depth rho
    t = dirs[:, 0]  # Use azimuthal angle theta (ignore elevation angle phi)
    rr, tt = np.meshgrid(r, t, indexing="xy")

    # Convert the focusing grid to Cartesian coordinates
    xx = rr * np.sin(tt) + oris[:, [0]]
    zz = rr * np.cos(tt) + oris[:, [2]]
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid
