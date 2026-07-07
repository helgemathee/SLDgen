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
    labelmap    Split *semantically* by sampling a label PNG (``--labels``)
                under each master point. A continuous map (e.g. SLDgen's
                condition_depth.png) is quantile-binned into N depth layers
                (foreground / midground / background); a PNG with <= N flat
                gray regions is partitioned directly by region (hand-painted
                masks). Requires the label PNG to be in the master's canvas
                space -- SLDgen's saved condition image already is.

For horizontal / vertical / radial / labelmap the points of a partition come from
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

Preview (optional)
    ``--preview`` writes a ``partition_preview.png`` into ``--output-dir``: every
    partition's points scattered in a distinct colour over the label/depth image
    used for the split (or a blank canvas for the geometric strategies). It is a
    quick visual sanity check of where each piece landed. When a ``--labels`` PNG
    was used, a copy of it is also dropped into ``--output-dir`` so the folder is
    self-documenting. Needs matplotlib (part of SLDgen's env); if it is missing
    the preview is skipped with a warning and the SVGs are still written.

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


def assign_labelmap(points, n, labels_path, canvas_w, canvas_h, discrete_tol=6.0):
    """Assign points *semantically* by sampling a label PNG under each point.

    The master's points live in canvas coordinates aligned with the conditioned
    image, so partition assignment is a label lookup. Loads the label image as
    grayscale, samples the pixel under each master point (nearest-neighbour),
    then partitions in one of two modes:

    * **Discrete** (hand-painted regions / segmentation): if the sampled values
      form ``<= n`` distinct FLAT levels -- values grouped within ``discrete_tol``
      gray levels to absorb light anti-aliasing, each group spanning only a few
      levels -- the levels ARE the labels. Points are assigned by region, ordered
      dark-to-light. Any PNG with up to ``n`` flat regions works directly.
    * **Continuous** (depth): quantile binning -- points are sorted by sampled
      value and split into ``n`` equal-count groups. Real depth maps cluster, so
      equal-count quantiles stay balanced where equal-WIDTH bins would leave some
      near-empty; this also degrades gracefully as the value spread shrinks.

    Discrete vs continuous is decided by whether the value groups are *flat*: a
    hand-painted region is a tight cluster (small internal span), whereas a depth
    ramp chains through ``discrete_tol``-sized steps into one group that spans a
    wide range -- so a wide-span group forces the continuous path even though it
    is a single chained group. This is why a background-plus-gradient depth map
    (two chained groups) is still quantile-binned, not treated as 2 regions.

    The label PNG is assumed to share the master's canvas space; if its pixel
    dimensions differ (e.g. a 2x export) the sample coordinates are scaled to it.

    Returns an int label array in ``[0, n-1]``, one per master point.
    """
    from PIL import Image

    img = np.asarray(Image.open(labels_path).convert("L"), dtype=np.float64)  # (H, W)
    h_px, w_px = img.shape
    # Map master canvas coordinates -> label-image pixels (identity when sizes match).
    sx = (w_px - 1) / max(canvas_w - 1, 1e-9)
    sy = (h_px - 1) / max(canvas_h - 1, 1e-9)
    xi = np.clip(np.round(points[:, 0] * sx).astype(int), 0, w_px - 1)
    yi = np.clip(np.round(points[:, 1] * sy).astype(int), 0, h_px - 1)
    vals = img[yi, xi]  # (M,) sampled gray value per master point

    # Group sorted values into levels separated by more than discrete_tol.
    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    breaks = np.diff(sv) > discrete_tol
    group_sorted = np.concatenate([[0], np.cumsum(breaks)]) if len(sv) else np.array([], int)
    num_levels = int(group_sorted[-1]) + 1 if len(sv) else 0

    # A group is "flat" (a real painted region) only if its internal value span is
    # small; a chained depth ramp forms few groups but each spans a wide range.
    max_level_span = 3.0 * discrete_tol
    groups_flat = all(
        (seg.max() - seg.min()) <= max_level_span
        for seg in (sv[group_sorted == g] for g in range(num_levels))
    )

    if num_levels <= n and groups_flat:
        # Discrete map: the flat levels are the labels (0 = darkest region).
        labels = np.empty(len(vals), dtype=int)
        labels[order] = group_sorted
        return labels

    # Continuous map: quantile binning into n equal-count groups (0 = lowest values).
    labels = np.empty(len(vals), dtype=int)
    edges = np.linspace(0, len(vals), n + 1).astype(int)
    for i in range(n):
        labels[order[edges[i]:edges[i + 1]]] = i
    return labels


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
# Preview (optional matplotlib overlay of the partitioning)
# --------------------------------------------------------------------------- #
def _partition_colors(n):
    """N visually distinct hex colours: a hand palette first, then HSV fallback."""
    palette = [
        "#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#e9c46a",
        "#8d5b9e", "#bc6c25", "#606c38", "#d62828", "#118ab2",
    ]
    if n <= len(palette):
        return palette[:n]
    import colorsys

    return [
        "#%02x%02x%02x"
        % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / n, 0.65, 0.9))
        for i in range(n)
    ]


def write_preview(out_path, partitions_runs, bg_path, canvas_w, canvas_h):
    """Scatter each partition's points in its own colour over the label image.

    ``partitions_runs`` is the per-partition list of sub-strokes actually written
    (post-tails), so the preview matches the SVGs exactly. ``bg_path`` is the
    label/depth PNG used for the split (drawn as a grayscale backdrop, stretched
    into the master's canvas space so points register) or ``None`` for a blank
    canvas. Returns the written path, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "warning: --preview needs matplotlib but it is not importable; "
            "skipping partition_preview.png",
            file=sys.stderr,
        )
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    if bg_path is not None:
        from PIL import Image

        bg = np.asarray(Image.open(bg_path).convert("L"))
        # extent maps the image into canvas space so overlaid points line up even
        # if the label PNG was exported at a different pixel resolution.
        ax.imshow(
            bg, cmap="gray", extent=[0, canvas_w, canvas_h, 0], aspect="auto"
        )
    else:
        ax.set_facecolor("white")
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)  # SVG y grows downward

    colors = _partition_colors(len(partitions_runs))
    for i, runs in enumerate(partitions_runs):
        stroked = [np.asarray(r) for r in runs if len(r)]
        if not stroked:
            continue
        pts = np.vstack(stroked)
        ax.scatter(
            pts[:, 0], pts[:, 1], s=2, c=colors[i], label=f"partition_{i}"
        )
    ax.legend(loc="lower right", fontsize=8, markerscale=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
            "  labelmap    semantic split by a --labels PNG (depth quantiles\n"
            "              or flat painted regions)\n"
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
        choices=["horizontal", "vertical", "radial", "sequence", "cluster", "labelmap"],
        help="how to partition the master (see epilog)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        metavar="PNG",
        help=(
            "label map PNG for --strategy labelmap (required for it, ignored "
            "otherwise). A continuous map (e.g. SLDgen's condition_depth.png) is "
            "quantile-binned into N depth layers; a map with <= N flat gray "
            "regions is partitioned directly by region. Must be in the master's "
            "canvas space."
        ),
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
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "also write partition_preview.png: each partition's points scattered "
            "in a distinct colour over the --labels image (or a blank canvas). "
            "When --labels is used, a copy of it is saved to --output-dir too. "
            "Requires matplotlib."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.partitions < 1:
        _die("--partitions must be >= 1")
    if args.sample_spacing <= 0:
        _die("--sample-spacing must be > 0")

    if args.strategy == "labelmap":
        if args.labels is None:
            _die("--strategy labelmap requires --labels <PNG>")
        if not os.path.isfile(args.labels):
            _die(f"--labels PNG does not exist: {args.labels}")

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
    elif args.strategy == "labelmap":
        labels = assign_labelmap(points, n, args.labels, canvas_w, canvas_h)
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
    written_runs = []
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
        written_runs.append(runs)

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

    # --- Optional preview + self-documenting copy of the label image ---
    if args.preview:
        preview_bg = None
        if args.labels is not None:
            import shutil

            dst = os.path.join(args.output_dir, os.path.basename(args.labels))
            if os.path.abspath(dst) != os.path.abspath(args.labels):
                shutil.copy2(args.labels, dst)
                print(f"copied label image -> {dst}")
            preview_bg = dst if os.path.isfile(dst) else args.labels

        preview_path = write_preview(
            os.path.join(args.output_dir, "partition_preview.png"),
            written_runs,
            preview_bg,
            canvas_w,
            canvas_h,
        )
        if preview_path:
            print(f"preview -> {preview_path}")

    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
