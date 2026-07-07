#!/usr/bin/env python
"""Partition a single high-detail SLDgen SVG into N spatial sub-curves.

SLDgen produces one long continuous stroke (a "single-line drawing"). For
plotting workflows it is sometimes useful to split that master curve into
several pieces that can each be plotted independently -- at a different time,
with a different pen -- and that still compose back into the original drawing
when overlaid. Because every partition is *literally a subset of the same
source curve*, the pieces are guaranteed to register perfectly by
construction; there is no re-optimization and no drift.

Typical workflow:
    1. Generate a high-detail master with SLDgen (~1000-1500 control points,
       strong ControlNet, careful tuning).
    2. Run this utility to split the master into N partitions.
    3. Plot each partition separately; they overlay into the master.

Strategies (``--strategy``)
    horizontal  Split by y-coordinate into equal horizontal bands
                (top / middle / bottom strips for N=3).
    vertical    Split by x-coordinate into equal vertical bands.
    radial      Split by angle from the canvas centre, like pie slices.
    sequence    Split by position along the master's traversal: first third,
                middle third, last third of the drawn curve. Each partition is
                one contiguous stroke.
    cluster     k-means the control points into N spatial groups, then order
                each group by nearest-neighbour traversal. Good for subjects
                with N natural blobs that don't fall on a clean axis.

For horizontal / vertical / radial the points of a partition come from
possibly non-contiguous segments of the master (the curve crosses a boundary
back and forth). Those strategies keep the master's original ordering and
simply break the kept points into separate sub-strokes wherever the master
wandered into another region -- the plotter lifts the pen between sub-strokes
naturally, so this is fine. This "preserve the master's ordering" approach
keeps each partition's local aesthetic identical to the master.

Origins & tails (optional)
    ``--origins`` gives each partition an anchor point in normalized [0,1]
    canvas space; the partition's first drawn point is connected to that
    anchor by a straight "tail". ``--connect-tails`` additionally returns to
    the anchor at the end (or, with no origins, tails to the nearest canvas
    edge at both ends). Tails give each plotted piece a clean lead-in/lead-out.

The tool only reads SLDgen-style output: a single ``<path>`` whose ``d`` is one
``M`` followed by ``L`` segments. It writes ``partition_<i>.svg`` for i in
``0..N-1`` into ``--output-dir`` (always exactly N files; empty partitions are
still written, carrying an explanatory comment).
"""

import argparse
import math
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# Master parsing
# --------------------------------------------------------------------------- #
def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _parse_length(value, fallback):
    """Parse an SVG length attribute like '512' or '512px' to a float."""
    if value is None:
        return fallback
    s = str(value).strip()
    for unit in ("px", "pt", "mm", "cm", "in"):
        if s.endswith(unit):
            s = s[: -len(unit)].strip()
            break
    try:
        return float(s)
    except ValueError:
        return fallback


def load_master(input_path):
    """Read the first path of a SLDgen SVG.

    Returns ``(path, style, canvas_w, canvas_h)`` where ``path`` is an
    svgpathtools ``Path``, ``style`` is a dict of the path's presentation
    attributes (minus ``d``), and the canvas dimensions are floats.
    """
    if not os.path.isfile(input_path):
        _die(f"input SVG does not exist: {input_path}")

    try:
        from svgpathtools import svg2paths2
    except ImportError:
        _die(
            "svgpathtools is required but not importable. It ships with "
            "SLDgen's dependency tree; activate the sldgen conda env."
        )

    try:
        paths, attributes, svg_attributes = svg2paths2(input_path)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure clearly
        _die(f"failed to parse '{input_path}' as SVG: {exc}")

    if not paths:
        _die(
            f"'{input_path}' contains no <path> elements. This tool expects "
            "SLDgen output: a single path of one M followed by L segments."
        )
    if len(paths) > 1:
        print(
            f"warning: '{input_path}' has {len(paths)} <path> elements; "
            "using only the first (SLDgen output has exactly one).",
            file=sys.stderr,
        )

    path = paths[0]
    if path.length() <= 0:
        _die("the master path has zero length; nothing to partition.")

    style = {k: v for k, v in attributes[0].items() if k != "d"}

    canvas_w = _parse_length(svg_attributes.get("width"), None)
    canvas_h = _parse_length(svg_attributes.get("height"), None)
    if canvas_w is None or canvas_h is None:
        # Fall back to the path's bounding box (xmin, xmax, ymin, ymax).
        xmin, xmax, ymin, ymax = path.bbox()
        canvas_w = canvas_w if canvas_w is not None else math.ceil(xmax)
        canvas_h = canvas_h if canvas_h is not None else math.ceil(ymax)
        print(
            "warning: master SVG has no width/height; inferred canvas "
            f"{canvas_w:g}x{canvas_h:g} from the path bounding box.",
            file=sys.stderr,
        )

    return path, style, float(canvas_w), float(canvas_h)


def sample_path(path, spacing_px):
    """Sample the master path into an ordered (M, 2) array of (x, y) points.

    Points are spaced ~``spacing_px`` of arc length apart, in traversal order,
    so partition boundaries have sub-pixel resolution regardless of how the
    master's own control points happen to be distributed.
    """
    length = path.length()
    n = max(2, int(round(length / spacing_px)) + 1)
    pts = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        if i == 0:
            t = 0.0
        elif i == n - 1:
            t = 1.0
        else:
            s = length * i / (n - 1)
            try:
                t = path.ilength(s)
            except ValueError:
                # Numerical edge case near an endpoint: fall back to
                # uniform-in-parameter sampling for this one point.
                t = i / (n - 1)
        p = path.point(t)
        pts[i, 0] = p.real
        pts[i, 1] = p.imag
    return pts


# --------------------------------------------------------------------------- #
# Partition assignment
# --------------------------------------------------------------------------- #
def assign_banded(points, n, canvas_w, canvas_h, strategy):
    """Assign each point to a partition by fixed geometric band.

    horizontal -> equal y bands, vertical -> equal x bands,
    radial -> equal angular slices about the canvas centre.
    Returns an int array of partition indices in ``[0, n-1]``.
    """
    if strategy == "horizontal":
        frac = points[:, 1] / max(canvas_h, 1e-9)
    elif strategy == "vertical":
        frac = points[:, 0] / max(canvas_w, 1e-9)
    elif strategy == "radial":
        cx, cy = canvas_w / 2.0, canvas_h / 2.0
        ang = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)  # (-pi, pi]
        frac = (ang + math.pi) / (2.0 * math.pi)  # -> [0, 1)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(strategy)

    labels = np.floor(frac * n).astype(int)
    return np.clip(labels, 0, n - 1)


def assign_sequence(m, n):
    """Assign M points to N contiguous arc-length slices (order preserved)."""
    idx = np.arange(m)
    labels = (idx * n) // m
    return np.clip(labels, 0, n - 1)


def runs_from_labels(points, labels, i):
    """Return the member points of partition ``i`` as a list of sub-strokes.

    Keeps the master's ordering; every maximal contiguous run of points that
    belong to partition ``i`` becomes one sub-stroke. A gap (the master left
    into another region) ends the current sub-stroke and starts a new one.
    """
    runs = []
    cur = []
    member = labels == i
    for idx in range(len(points)):
        if member[idx]:
            cur.append(points[idx])
        elif cur:
            runs.append(np.asarray(cur))
            cur = []
    if cur:
        runs.append(np.asarray(cur))
    return runs


# --------------------------------------------------------------------------- #
# Cluster strategy (pure-numpy k-means; sklearn is not a SLDgen dependency)
# --------------------------------------------------------------------------- #
def _kmeans(points, k, iters=100, seed=0):
    """Lloyd's algorithm with k-means++ seeding. Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    # k-means++ initialisation for stable, well-spread centres.
    centers = [points[rng.integers(n)]]
    for _ in range(1, k):
        cs = np.asarray(centers)
        d2 = np.min(
            ((points[:, None, :] - cs[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        total = d2.sum()
        if total <= 0:
            centers.append(points[rng.integers(n)])
        else:
            centers.append(points[rng.choice(n, p=d2 / total)])
    centers = np.asarray(centers, dtype=np.float64)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        new_centers = np.array(
            [
                points[new_labels == j].mean(axis=0)
                if np.any(new_labels == j)
                else centers[j]
                for j in range(k)
            ]
        )
        converged = np.array_equal(new_labels, labels) and np.allclose(
            new_centers, centers
        )
        labels, centers = new_labels, new_centers
        if converged:
            break
    return labels, centers


def _nearest_neighbour_order(points, start_idx):
    """Greedy nearest-neighbour ordering of ``points`` from ``start_idx``."""
    n = len(points)
    remaining = list(range(n))
    remaining.remove(start_idx)
    order = [start_idx]
    cur = start_idx
    while remaining:
        rem = np.asarray(remaining)
        d2 = ((points[rem] - points[cur]) ** 2).sum(axis=1)
        nxt = int(rem[np.argmin(d2)])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def cluster_partitions(points, n, origins_px, seed=0):
    """Split points into N k-means clusters, each ordered nearest-neighbour.

    Returns a list of length N; each entry is a list holding a single
    sub-stroke (the ordered cluster points), matching the runs_from_labels
    shape used by the banded strategies. Each cluster's traversal starts at
    the point nearest its assigned origin (if origins were given) or its
    centroid otherwise.
    """
    labels, centers = _kmeans(points, n, seed=seed)
    partitions = []
    for i in range(n):
        member_idx = np.where(labels == i)[0]
        if len(member_idx) == 0:
            partitions.append([])
            continue
        member_pts = points[member_idx]
        if origins_px is not None:
            anchor = np.asarray(origins_px[i])
        else:
            anchor = centers[i]
        start = int(np.argmin(((member_pts - anchor) ** 2).sum(axis=1)))
        order = _nearest_neighbour_order(member_pts, start)
        partitions.append([member_pts[order]])
    return partitions


# --------------------------------------------------------------------------- #
# Tails
# --------------------------------------------------------------------------- #
def _nearest_edge_point(x, y, w, h):
    """Point on the nearest canvas edge to (x, y)."""
    candidates = [
        (0.0, y, x),          # left
        (w, y, w - x),        # right
        (x, 0.0, y),          # top
        (x, h, h - y),        # bottom
    ]
    ex, ey, _ = min(candidates, key=lambda c: c[2])
    return np.array([ex, ey])


def apply_tails(runs, origin_px, connect_tails, canvas_w, canvas_h):
    """Prepend / append tail anchor points to a partition's sub-strokes.

    * ``origin_px`` given, no ``connect_tails``: a lead-in tail from the origin
      to the first point.
    * ``connect_tails`` set: lead-in *and* lead-out tails. The anchor is the
      origin if provided, else the nearest canvas edge to the relevant end.
    """
    if not runs:
        return runs

    start_tail = origin_px is not None or connect_tails
    end_tail = connect_tails
    if not start_tail and not end_tail:
        return runs

    runs = [np.array(r, dtype=np.float64) for r in runs]

    if start_tail:
        first = runs[0][0]
        anchor = (
            np.asarray(origin_px, dtype=np.float64)
            if origin_px is not None
            else _nearest_edge_point(first[0], first[1], canvas_w, canvas_h)
        )
        runs[0] = np.vstack([anchor[None, :], runs[0]])

    if end_tail:
        last = runs[-1][-1]
        anchor = (
            np.asarray(origin_px, dtype=np.float64)
            if origin_px is not None
            else _nearest_edge_point(last[0], last[1], canvas_w, canvas_h)
        )
        runs[-1] = np.vstack([runs[-1], anchor[None, :]])

    return runs


# --------------------------------------------------------------------------- #
# SVG writing
# --------------------------------------------------------------------------- #
DEFAULT_STYLE = {
    "stroke-width": "2.0",
    "fill": "none",
    "stroke": "rgb(0, 0, 0)",
    "stroke-opacity": "1.0",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
}


def _d_string(runs):
    """Build an SVG path 'd' from a list of sub-strokes (M..L per sub-stroke)."""
    parts = []
    for run in runs:
        if len(run) == 0:
            continue
        cmds = [f"M {run[0][0]:.4f} {run[0][1]:.4f}"]
        for p in run[1:]:
            cmds.append(f"L {p[0]:.4f} {p[1]:.4f}")
        parts.append(" ".join(cmds))
    return " ".join(parts)


def write_partition_svg(out_path, runs, style, canvas_w, canvas_h, empty_note):
    """Write one partition SVG matching SLDgen's output structure."""
    merged = dict(DEFAULT_STYLE)
    merged.update(style or {})
    attr_str = " ".join(f'{k}="{v}"' for k, v in merged.items())

    w = f"{canvas_w:g}"
    h = f"{canvas_h:g}"
    lines = [
        '<?xml version="1.0" ?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{w}" height="{h}">',
        "  <defs/>",
    ]
    if empty_note:
        lines.append(f"  <!-- {empty_note} -->")
    lines.append("  <g>")
    d = _d_string(runs)
    if d:
        lines.append(f'    <path d="{d}" {attr_str}/>')
    lines.append("  </g>")
    lines.append("</svg>")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sld_partition.py",
        description=(
            "Decompose a single SLDgen SVG output into N spatially partitioned "
            "sub-curves that overlay back into the original."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "strategies:\n"
            "  horizontal  equal y bands (top/middle/bottom for N=3)\n"
            "  vertical    equal x bands\n"
            "  radial      pie slices by angle about the canvas centre\n"
            "  sequence    contiguous thirds of the master's traversal\n"
            "  cluster     k-means groups, nearest-neighbour ordered\n"
        ),
    )
    parser.add_argument(
        "--input", required=True, help="path to the master SLDgen SVG"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for partition_<i>.svg outputs (created if absent)",
    )
    parser.add_argument(
        "--partitions",
        required=True,
        type=int,
        help="number of partitions N to produce (typically 2, 3, or 4)",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["horizontal", "vertical", "radial", "sequence", "cluster"],
        help="how to partition the master (see epilog)",
    )
    parser.add_argument(
        "--origins",
        type=float,
        nargs="+",
        default=None,
        metavar="X Y",
        help=(
            "2N floats: a normalized [0,1] anchor per partition. Each "
            "partition's start is tailed to its anchor. "
            "Example N=3: --origins 1.0 0.5 0.5 1.0 0.5 0.0"
        ),
    )
    parser.add_argument(
        "--connect-tails",
        action="store_true",
        help=(
            "also add a lead-out tail back to the origin (or, with no "
            "--origins, tail both ends to the nearest canvas edge)"
        ),
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=1.0,
        metavar="PX",
        help="arc-length spacing in px when sampling the master (default 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the cluster strategy's k-means (default 0)",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.partitions < 1:
        _die("--partitions must be >= 1")
    if args.sample_spacing <= 0:
        _die("--sample-spacing must be > 0")

    origins_norm = None
    if args.origins is not None:
        if len(args.origins) != 2 * args.partitions:
            _die(
                f"--origins expects 2*N = {2 * args.partitions} floats for "
                f"N={args.partitions}, got {len(args.origins)}"
            )
        origins_norm = [
            (args.origins[2 * i], args.origins[2 * i + 1])
            for i in range(args.partitions)
        ]
    return origins_norm


def main(argv=None):
    args = parse_args(argv)
    origins_norm = validate_args(args)

    path, style, canvas_w, canvas_h = load_master(args.input)
    points = sample_path(path, args.sample_spacing)
    m = len(points)

    origins_px = None
    if origins_norm is not None:
        origins_px = [(nx * canvas_w, ny * canvas_h) for nx, ny in origins_norm]

    n = args.partitions

    # Build, per partition, a list of sub-strokes (each an (k, 2) array).
    if args.strategy == "cluster":
        partitions = cluster_partitions(points, n, origins_px, seed=args.seed)
    elif args.strategy == "sequence":
        labels = assign_sequence(m, n)
        partitions = [runs_from_labels(points, labels, i) for i in range(n)]
    else:  # horizontal / vertical / radial
        labels = assign_banded(points, n, canvas_w, canvas_h, args.strategy)
        partitions = [runs_from_labels(points, labels, i) for i in range(n)]

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Report header ---
    print(
        f"master: {args.input}\n"
        f"  canvas {canvas_w:g}x{canvas_h:g}, sampled {m} points "
        f"(~{args.sample_spacing:g}px spacing)\n"
        f"  strategy '{args.strategy}' -> {n} partitions"
    )

    written = []
    warnings = []
    for i in range(n):
        origin_i = origins_px[i] if origins_px is not None else None
        runs = apply_tails(
            partitions[i], origin_i, args.connect_tails, canvas_w, canvas_h
        )
        n_pts = int(sum(len(r) for r in runs))
        n_strokes = len([r for r in runs if len(r) > 0])

        empty_note = None
        if n_pts == 0:
            empty_note = (
                f"partition {i} is empty: no master points fell in this "
                f"region under strategy '{args.strategy}'"
            )
            warnings.append(f"partition {i} is empty")

        out_path = os.path.join(args.output_dir, f"partition_{i}.svg")
        write_partition_svg(
            out_path, runs, style, canvas_w, canvas_h, empty_note
        )
        written.append((out_path, n_pts, n_strokes))

    # --- Report body ---
    print("\npartitions written:")
    for i, (out_path, n_pts, n_strokes) in enumerate(written):
        stroke_note = (
            f", {n_strokes} sub-strokes" if n_strokes > 1 else ""
        )
        tail_note = ""
        if origins_px is not None or args.connect_tails:
            tail_note = " [tailed]"
        print(f"  partition_{i}.svg: {n_pts} points{stroke_note}{tail_note}")
        print(f"    {out_path}")

    print(f"\n{n} files written to {args.output_dir} (~{m // n} points each).")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
