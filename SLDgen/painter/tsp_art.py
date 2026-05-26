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


def init_tsp_art(
    density,
    n_point=500,
    n_iter=25,
    reverse=True,
    output_dir=None,
    debug=False,
    fixed_endpoints=False,
):
    """Initialize TSP art by stippling, generating a TSP problem, and solving it with Concorde."""
    if output_dir is None:
        # Get a temporary directory
        tmp_dir = tempfile.TemporaryDirectory()
        root_tmp_dir = Path(tmp_dir.name)
    else:
        root_tmp_dir = Path(output_dir)
        root_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load image and sample points
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
    ordered_points = points[order]

    # Clean up temporary directory
    if output_dir is None:
        tmp_dir.cleanup()
    else:
        # If output_dir is specified, remove it as we won't need all the files anymore
        shutil.rmtree(root_tmp_dir)

    return ordered_points
