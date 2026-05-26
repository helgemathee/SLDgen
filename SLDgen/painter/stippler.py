#! /usr/bin/env python3
# -----------------------------------------------------------------------------
# Weighted Voronoi Stippler
# Copyright (2017) Nicolas P. Rougier - BSD license
#
# Implementation of:
#   Weighted Voronoi Stippling, Adrian Secord
#   Symposium on Non-Photorealistic Animation and Rendering (NPAR), 2002
# -----------------------------------------------------------------------------
# Some usage examples
#
# stippler.py boots.jpg --save --force --n_point 20000 --n_iter 50
#                       --pointsize 0.5 2.5 --figsize 8 --interactive
# stippler.py plant.png --save --force --n_point 20000 --n_iter 50
#                       --pointsize 0.5 1.5 --figsize 8
# stippler.py gradient.png --save --force --n_point 5000 --n_iter 50
#                          --pointsize 1.0 1.0 --figsize 6
# -----------------------------------------------------------------------------
# usage: stippler.py [-h] [--n_iter n] [--n_point n] [--epsilon n]
#                    [--pointsize min,max) (min,max] [--figsize w,h] [--force]
#                    [--save] [--display] [--interactive]
#                    image filename
#
# Weighted Vororonoi Stippler
#
# positional arguments:
#   image filename        Density image filename
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --n_iter n            Maximum number of iterations
#   --n_point n           Number of points
#   --epsilon n           Early stop criterion
#   --pointsize (min,max) (min,max)
#                         Point mix/max size for final display
#   --figsize w,h         Figure size
#   --force               Force recomputation
#   --save                Save computed points
#   --display             Display final result
#   --interactive         Display intermediate results (slower)
# -----------------------------------------------------------------------------
# Taken from https://github.com/ReScience-Archives/Rougier-2017
# -----------------------------------------------------------------------------

import numpy as np
import scipy.ndimage

from .voronoi import centroids


def normalize(D):
    Vmin, Vmax = D.min(), D.max()
    if Vmax - Vmin > 1e-5:
        D = (D - Vmin) / (Vmax - Vmin)
    else:
        D = np.zeros_like(D)
    return D


def initialization(n, D):
    """
    Return n points distributed over [xmin, xmax] x [ymin, ymax]
    according to (normalized) density distribution.

    with xmin, xmax = 0, density.shape[1]
         ymin, ymax = 0, density.shape[0]

    The algorithm here is a simple rejection sampling.
    """

    samples = []
    while len(samples) < n:
        # X = np.random.randint(0, D.shape[1], 10*n)
        # Y = np.random.randint(0, D.shape[0], 10*n)
        X = np.random.uniform(0, D.shape[1], 10 * n)
        Y = np.random.uniform(0, D.shape[0], 10 * n)
        P = np.random.uniform(0, 1, 10 * n)
        index = 0
        while index < len(X) and len(samples) < n:
            x, y = X[index], Y[index]
            x_, y_ = int(np.floor(x)), int(np.floor(y))
            if P[index] < D[y_, x_]:
                samples.append([x, y])
            index += 1
    return np.array(samples)


def stipple(density, n_point=5000, n_iter=50, threshold=255, reverse=False):
    if reverse:
        density = 1 - density

    # We want (approximately) 500 pixels per voronoi region
    zoom = (n_point * 500) / (density.shape[0] * density.shape[1])
    zoom = max(int(round(np.sqrt(zoom))), 1)
    density = scipy.ndimage.zoom(density, zoom, order=0)
    # Apply threshold onto image
    # Any color > threshold will be white
    density = np.minimum(density, threshold)

    density = 1.0 - normalize(density)
    # density = density[::-1, :]
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    # Initialization
    points = initialization(n_point, density)

    for i in range(n_iter):
        regions, points, vertices = centroids(points, density, density_P, density_Q)

    return regions, points, vertices
