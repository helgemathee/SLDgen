import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .stippler import stipple


def get_tsp_file_string_free(points):
    """Generate a TSP file format string for free endpoints."""
    base_str = f"""NAME : SLD_init
COMMENT : Initialization for SLD generation
TYPE : TSP
DIMENSION : {len(points)}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
"""
    for i, p in enumerate(points):
        base_str += f"{i} {p[0]} {p[1]}\n"

    base_str += "EOF"
    return base_str


def get_tsp_file_string_fixed(points):
    """Generate a TSP file format string for fixed endpoints."""
    base_str = f"""NAME : SLD_init
COMMENT : Initialization for SLD generation
TYPE : TSP
DIMENSION : {len(points)}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : LOWER_DIAG_ROW
EDGE_WEIGHT_SECTION
"""
    for i, p in enumerate(points):
        row_str = ""
        for j in range(i + 1):
            if j == 0 and i <= 2:
                dist = 0.0
            elif j == 0:
                dist = 1e6
            elif i == 3 and j == i:
                dist = 0.0
            elif i == 4 and j == 2:
                dist = 0.0
            else:
                dist = np.linalg.norm(p - points[j])
            row_str += f"{int(dist * 100)} "
        base_str += row_str + "\n"

    base_str += "EOF"
    return base_str


def get_tsp_file_string(points, fixed_endpoints=False):
    """Generate a TSP file format string."""
    if fixed_endpoints:
        return get_tsp_file_string_fixed(points)
    else:
        return get_tsp_file_string_free(points)


def read_tour(file):
    """Read a TSP tour file."""
    with open(file, "r") as f:
        lines = f.readlines()
    lines = lines[1:]  # Skip the first line

    order = [l.strip().split(" ") for l in lines]
    order = [int(x) for line in order for x in line]

    return order


def _zoom_factor(density, n_point):
    """The stipple() zoom (see stippler.py): ~500 px per Voronoi region.

    Reproduced here so injected points (origin, --init-points) land in the same
    zoomed pixel space that stipple() returns points in.
    """
    zoom = (n_point * 500) / (density.shape[0] * density.shape[1])
    return max(int(round(np.sqrt(zoom))), 1)


def load_init_points(svg_path, density, n_point):
    """Load TSP seed points from an SVG for the --init-points feature.

    Samples points along the SVG's paths in canvas pixel coordinates (reusing
    the avoidance loader), then returns them in stipple()'s zoomed pixel space.
    If the SVG yields more than ``n_point`` samples they are subsampled
    uniformly along the path; if fewer, they are used as-is (noted below). The
    order does not matter -- Concorde recomputes the tour.
    """
    # Reuse the exact avoidance/attraction SVG sampler (canvas-pixel frame).
    from ..avoidance import load_avoid_points

    pts = load_avoid_points([svg_path], sample_spacing_px=1.0, render_size=density.shape[1])
    if pts is None or len(pts) == 0:
        raise ValueError(f"--init-points: no points sampled from '{svg_path}'.")

    pts = np.asarray(pts, dtype=float)
    if len(pts) > n_point:
        # Uniform subsample along the arc-length-ordered samples.
        idx = np.linspace(0, len(pts) - 1, n_point).astype(int)
        pts = pts[idx]
        print(f"\t\t--init-points: seeded TSP from {n_point} points sampled along "
              f"'{svg_path}'.", flush=True)
    else:
        print(f"\t\t--init-points: '{svg_path}' yielded {len(pts)} points (fewer than "
              f"--n-control-points={n_point}); using all as-is.", flush=True)

    # Scale canvas pixels into stipple()'s zoomed frame so downstream (origin
    # injection, normalization by density.shape) is consistent with stippling.
    return pts * _zoom_factor(density, n_point)


def init_tsp_art(
    density,
    n_point=500,
    n_iter=25,
    reverse=True,
    output_dir=None,
    debug=False,
    fixed_endpoints=False,
    origin=None,
    init_points=None,
):
    """Initialize TSP art by stippling, generating a TSP problem, and solving it with Concorde."""
    if output_dir is None:
        # Get a temporary directory
        tmp_dir = tempfile.TemporaryDirectory()
        root_tmp_dir = Path(tmp_dir.name)
    else:
        root_tmp_dir = Path(output_dir)
        root_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Sample the seed points: from a provided SVG (--init-points, opt-in) or by
    # stippling the target density (upstream default). Either way Concorde solves
    # the tour over the resulting point set below.
    if init_points is not None:
        points = load_init_points(init_points, density, n_point)
    else:
        regions, points, vertices = stipple(density, n_point=n_point, n_iter=n_iter, reverse=reverse)

    # Add new points to fix the endpoints of the curve if fixed_endpoints is True
    if fixed_endpoints:
        # Find all the points with y coordinates between 0.6 and 0.8
        points = np.array(points)
        in_range_points = points[
            (points[:, 1] >= 0.6 * density.shape[0]) & (points[:, 1] <= 0.8 * density.shape[0])
        ]
        # Get the left most and right most of these points
        x_min = in_range_points[in_range_points[:, 0] == np.min(in_range_points[:, 0])]
        x_max = in_range_points[in_range_points[:, 0] == np.max(in_range_points[:, 0])]

        # Add 4 points to the list of points
        extrem_left = np.array([[0.05 * density.shape[1], 0.7 * density.shape[0]]])
        mid_left = np.array([[x_min[0][0], 0.7 * density.shape[0]]])
        mid_right = np.array([[x_max[0][0], 0.7 * density.shape[0]]])
        extrem_right = np.array([[0.95 * density.shape[1], 0.7 * density.shape[0]]])
        dumb_node = np.array([[0.5 * density.shape[1], 2 * density.shape[0]]])

        points = np.concatenate(
            [dumb_node, extrem_left, extrem_right, mid_left, mid_right, points], axis=0
        )

    # Inject the pinned origin as an extra node so Concorde is forced to route
    # through it; the tour is later rotated so this node becomes index 0.
    origin_index = None
    if origin is not None:
        # Mirror stipple()'s zoom (see stippler.py) so the origin lands in the same
        # (zoomed) pixel space as the points that stipple() returns.
        zoom = (n_point * 500) / (density.shape[0] * density.shape[1])
        zoom = max(int(round(np.sqrt(zoom))), 1)
        points = np.array(points)
        origin_index = len(points)
        origin_px = np.array(
            [[origin[0] * density.shape[1] * zoom, origin[1] * density.shape[0] * zoom]]
        )
        points = np.concatenate([points, origin_px], axis=0)

    # Save the points to a TSP file
    with open(root_tmp_dir / "point.tsp", "w") as f:
        f.write(get_tsp_file_string(points, fixed_endpoints=fixed_endpoints))

    # Run TSP concorde solver
    subprocess.run(
        [
            os.environ["CONCORDE_PATH"],
            "-o",
            str(root_tmp_dir / "tour.txt"),
            str(root_tmp_dir / "point.tsp"),
        ],
        check=True,
        cwd=root_tmp_dir,
        stdout=subprocess.DEVNULL if not debug else None,
        stderr=subprocess.DEVNULL if not debug else None,
    )

    # Load the tour and return the list or ordered points
    order = read_tour(root_tmp_dir / "tour.txt")
    if fixed_endpoints:
        index_of_0 = order.index(0)
        order = np.roll(order, -index_of_0 - 1, axis=0)[:-1]
    elif origin is not None:
        # Rotate the closed tour so the injected origin node becomes index 0.
        # Because the curve is rendered open, index 0 becomes the start endpoint.
        index_of_origin = order.index(origin_index)
        order = np.roll(order, -index_of_origin, axis=0)
    ordered_points = np.asarray(points[order], dtype=float)

    # stipple() (and the origin / init-points injectors, which deliberately mirror
    # it) work in a zoomed pixel space of ~500 px per Voronoi region: coordinates
    # span [0, shape*zoom], not [0, shape]. Scale back to the original (un-zoomed)
    # pixel frame so callers can normalize by density.shape alone. At zoom == 1
    # (which holds for the default point count) this is a no-op, so the default
    # behavior is byte-identical; it only corrects the overflow at higher point
    # counts, where zoom >= 2 previously pushed the curve off the bottom-right.
    zoom = _zoom_factor(density, n_point)
    if zoom != 1:
        ordered_points = ordered_points / zoom

    # Clean up temporary directory
    if output_dir is None:
        tmp_dir.cleanup()
    else:
        # If output_dir is specified, remove it as we won't need all the files anymore
        shutil.rmtree(root_tmp_dir)

    return ordered_points
