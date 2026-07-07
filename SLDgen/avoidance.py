"""Avoidance constraint for SLDgen (opt-in via ``--avoid``).

This module implements a soft repulsion that pushes the newly-generated curve
away from a fixed set of obstacle points loaded from one or more SVG files. It
enables *sequential compositional* workflows: generate line A, then generate
line B with ``--avoid A.svg`` so B routes around A, and so on.

Two pieces live here:

* :func:`load_avoid_points` -- parse SVGs and sample points along their paths,
  uniformly by arc length, in canvas pixel coordinates.
* :func:`avoidance_loss` -- the repulsion energy between the curve's active
  control points and those fixed obstacle points.

The whole feature is strictly opt-in: with no ``--avoid`` argument,
``load_avoid_points`` returns ``None`` and the loss is never evaluated, so
behavior is byte-for-byte identical to upstream.
"""

import warnings

import numpy as np
import torch


def _warn_on_size_mismatch(svg_path, svg_attributes, render_size):
    """Warn if the SVG's declared canvas does not match ``render_size``.

    Avoid points are consumed in canvas pixel coordinates (0..render_size). If a
    supplied SVG was authored at a different size (e.g. a final SLDgen export is
    saved at 2x = 1024 for a render_size of 512), the obstacle geometry will be
    offset/scaled relative to the curve. We only warn -- per the design we assume
    the caller provides SVGs in the optimization coordinate frame.
    """

    def _first_number(value):
        if value is None:
            return None
        # Strip common unit suffixes ("px", "pt", ...) and parse the leading number.
        num = ""
        for ch in str(value).strip():
            if ch.isdigit() or ch in ".-+eE":
                num += ch
            else:
                break
        try:
            return float(num)
        except ValueError:
            return None

    w = h = None
    view_box = svg_attributes.get("viewBox") or svg_attributes.get("viewbox")
    if view_box is not None:
        parts = str(view_box).replace(",", " ").split()
        if len(parts) == 4:
            w, h = _first_number(parts[2]), _first_number(parts[3])
    if w is None:
        w = _first_number(svg_attributes.get("width"))
    if h is None:
        h = _first_number(svg_attributes.get("height"))

    for dim_name, dim in (("width", w), ("height", h)):
        if dim is not None and abs(dim - render_size) > 0.5:
            warnings.warn(
                f"--avoid: SVG '{svg_path}' declares {dim_name}={dim:g} which does "
                f"not match --render-size={render_size}. Avoid points are used as-is "
                f"in canvas pixel coordinates; the obstacle may be mis-registered "
                f"relative to the generated curve.",
                stacklevel=2,
            )
            break


def load_avoid_points(svg_paths, sample_spacing_px=2.0, render_size=None):
    """Sample obstacle points from SVG files for the avoidance constraint.

    Parses every ``<path>`` in each SVG with svgpathtools and samples points
    along it uniformly by arc length at roughly ``sample_spacing_px`` pixel
    intervals (``path.ilength`` maps an arc-length position to the path
    parameter, ``path.point`` evaluates the coordinate). All points from all
    paths in all files are concatenated into a single ``(N, 2)`` float32 array in
    canvas pixel coordinates.

    Args:
        svg_paths: list of SVG file paths, or ``None``/empty.
        sample_spacing_px: target spacing between samples, in canvas pixels. The
            default of 2.0 yields ~1000-2000 points for a typical full-canvas
            drawing.
        render_size: if given, emit a warning when an SVG's declared canvas size
            differs from it (the points are still used as-is).

    Returns:
        ``(N, 2)`` numpy float32 array, or ``None`` if ``svg_paths`` is empty/None
        or no path yielded a positive length.
    """
    if not svg_paths:
        return None

    # Local import so the dependency is only touched on the opt-in path.
    from svgpathtools import svg2paths2

    all_points = []
    for svg_path in svg_paths:
        paths, _attributes, svg_attributes = svg2paths2(svg_path)
        if render_size is not None:
            _warn_on_size_mismatch(svg_path, svg_attributes, render_size)

        for path in paths:
            length = path.length()
            if length <= 0:
                continue
            # Inclusive samples from s=0 to s=length so path endpoints are covered.
            n_samples = max(2, int(round(length / sample_spacing_px)))
            for i in range(n_samples):
                s = length * i / (n_samples - 1)
                t = path.ilength(s)
                point = path.point(t)
                all_points.append((point.real, point.imag))

    if len(all_points) == 0:
        return None
    return np.asarray(all_points, dtype=np.float32)


def avoidance_loss(active_control_points, avoid_points, d0):
    """Soft repulsion pushing control points away from fixed obstacle points.

    Mathematically, for each active control point ``p`` we take its distance to
    the nearest obstacle point and apply a one-sided quadratic hinge penalty::

        dist_i   = min_j || p_i - q_j ||
        penalty  = sum_i  max(0, d0 - dist_i)^2

    A control point contributes nothing once it is at least ``d0`` away from every
    obstacle; closer than ``d0`` it is pushed out, with a force that grows
    smoothly toward the obstacle. This is the same threshold-below-``d0`` shape as
    SLDgen's intra-curve ``wg.repulsion_loss`` (which repels the curve from
    itself); here the two point sets differ -- the curve's *active control points*
    versus a *fixed external* set sampled from the ``--avoid`` SVGs -- which is
    why this is a separate, differentiable-in-pure-torch implementation rather
    than a call into wiregrad.

    It exists to support sequential compositional workflows: pass a previously
    generated line as ``--avoid`` and the next line will route around it.

    Args:
        active_control_points: ``(M, 2)`` tensor of the currently optimized
            control points (pinned origin / fixed-endpoint / deactivated points
            already excluded by the caller). Carries gradient.
        avoid_points: ``(N, 2)`` fixed tensor of obstacle points, same coordinate
            frame (canvas pixels). No gradient.
        d0: distance threshold in canvas pixel units below which repulsion acts.

    Returns:
        Scalar tensor loss (0-d), on the same device/dtype as the inputs.
    """
    # Pairwise distances (M, N) via broadcasting, then nearest obstacle per point.
    diff = active_control_points.unsqueeze(1) - avoid_points.unsqueeze(0)  # (M, N, 2)
    dists = torch.linalg.vector_norm(diff, dim=-1)  # (M, N)
    min_dists, _ = dists.min(dim=1)  # (M,)
    penalty = torch.clamp(d0 - min_dists, min=0.0) ** 2
    return penalty.sum()
