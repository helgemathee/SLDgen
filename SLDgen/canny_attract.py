"""Canny edges as attraction targets (opt-in via ``--attract-canny``).

Depth conditioning tells SLDgen where the *volume* is; it has almost nothing to
say about an eyelid or a lip corner. This module derives an edge map from the
target and hands it to the geometric attraction constraint, so the curve is
pulled onto the features while SDS keeps running on depth. No second ControlNet,
no extra VRAM, no measurable time cost.

**Why this lives inside the run.** Attract points are consumed as raw canvas
pixel coordinates (:func:`avoidance.load_avoid_points` does no registration), so
the edge map has to come from an image that already went through ``targets.py``'s
mask, square-pad, resize and ``--object-size-ratio`` rescale. That image only
exists once ``get_target`` has run. Computing the edges anywhere else -- a
pre-pass, a web upload, a helper script pointed at the original photograph --
means reproducing that pipeline exactly or misregistering silently. Called from
:func:`run.run` between ``get_target`` and the painter, there is nothing to
reproduce: ``args.input_image`` *is* the canvas-space image.

**Why the point budget matters.** ``attraction_loss`` is two-sided, and its
coverage term sums ``max(0, dist - deadzone)^2`` over *every* attract point. A
raw Canny map is tens of thousands of pixels, which would ask a few hundred
control points to visit every edge in the picture and would bury the SDS
gradient under the constraint. The loss was written for "hundreds to low
thousands" of points per side, so :func:`thin_to_budget` enforces that: over
budget it first drops the shortest contours as speckle, then thins what remains
uniformly along its arc length. Keep ``max_points`` at or below
``--n-control-points``.

Thinning has to remove *length*, not vertices: SLDgen samples every path at a
fixed 2 px regardless of how the vertices are spaced, so decimating points would
change the file and not the result.

The module deliberately imports only cv2/numpy/PIL -- no torch, no diffusers --
so the whole thing is testable on a laptop with no GPU (``test_canny_attract_geom.py``).
"""

import math
from pathlib import Path

import cv2
import numpy as np

#: Must match ``load_avoid_points(sample_spacing_px=...)`` as called from
#: ``painter.py``, or every count this module reports is a fiction.
SAMPLE_SPACING_PX = 2.0

#: SLDgen's own Canny thresholds, from ``guidance/sd3_controlnet.py``. Defaulting
#: to them means "the edges the Canny ControlNet would have seen".
DEFAULT_LOW = 100.0
DEFAULT_HIGH = 200.0

DEFAULTS = {
    "low": DEFAULT_LOW,
    "high": DEFAULT_HIGH,
    "blur": 3,
    "simplify": 1.0,
    "min_length": 12.0,
    "max_points": 400,
    "roi": None,
}


def as_gray(image):
    """Coerce a PIL image or array to a 2-D uint8 array."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.dtype != np.uint8:
        scale = 255.0 if array.max() <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return array


def as_mask(mask, shape):
    """Coerce a mask (torch tensor, array or PIL image) to a boolean 'inside' map.

    Accepts both conventions in play here: ``args.mask`` is float 0..1 straight
    from RMBG, while a saved ``mask.png`` is 0/255. Thresholding at the halfway
    point is what ``targets.save_mask`` does, so the two agree.
    """
    if mask is None:
        return None
    array = np.asarray(mask.cpu() if hasattr(mask, "cpu") else mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    if array.max() > 1.5:
        array = array / 255.0
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return array > 0.5


def edge_map(image, mask=None, low=DEFAULT_LOW, high=DEFAULT_HIGH, blur=3, roi=None):
    """Binary (0/255) Canny edge map in canvas space.

    ``blur`` is not cosmetic on portraits: without it hair and fabric texture
    produce more edge pixels than the face does, and the budget then spends
    itself on them.
    """
    gray = as_gray(image)
    if gray.shape[0] != gray.shape[1]:
        raise ValueError(
            f"canny-attract needs a square canvas-space image, got {gray.shape[1]}x{gray.shape[0]}"
        )
    render_size = gray.shape[0]

    if blur and blur > 0:
        kernel = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(gray, float(low), float(high))

    inside = as_mask(mask, edges.shape)
    if inside is not None:
        edges[~inside] = 0

    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = np.zeros_like(edges)
        xs = slice(max(0, int(round(min(x0, x1)))), min(render_size, int(round(max(x0, x1)))))
        ys = slice(max(0, int(round(min(y0, y1)))), min(render_size, int(round(max(y0, y1)))))
        keep[ys, xs] = 1
        edges = edges * keep

    return edges


def polyline_length(points):
    """Arc length of an open polyline, in canvas pixels."""
    return float(sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)))


def estimate_samples(length):
    """How many attract points ``load_avoid_points`` will draw from one path."""
    return max(2, int(round(length / SAMPLE_SPACING_PX)))


def extract_polylines(edges, simplify=1.0, min_length=12.0):
    """Contours of the edge map: simplified, speckle-filtered, longest first.

    ``findContours`` traces a one-pixel-wide edge out and back, so a thin line
    contributes about twice its visual length. The duplicate samples land on the
    same coordinates and are harmless to a Chamfer loss, but they do count
    against the budget -- which is why every estimate here counts them.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = []
    for contour in contours:
        if simplify > 0:
            contour = cv2.approxPolyDP(contour, float(simplify), True)
        points = [(float(x), float(y)) for x, y in contour.reshape(-1, 2)]
        if len(points) < 2:
            continue
        length = polyline_length(points)
        if length < min_length:
            continue
        polylines.append((length, points))
    polylines.sort(key=lambda item: item[0], reverse=True)
    return polylines


def point_at(points, cumulative, distance):
    """Linearly interpolate a polyline at an arc-length position."""
    index = int(np.searchsorted(cumulative, distance, side="right")) - 1
    index = min(max(index, 0), len(points) - 2)
    span = cumulative[index + 1] - cumulative[index]
    t = 0.0 if span <= 0 else (distance - cumulative[index]) / span
    (x0, y0), (x1, y1) = points[index], points[index + 1]
    return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))


def dash(points, keep_px, period_px):
    """Cut a polyline into short dashes, one every ``period_px`` of arc length.

    Every contour keeps its full extent at a lower target density, so an eyebrow
    still gets targets -- it just gets fewer of them. Dropping whole contours
    instead would silently delete features.
    """
    cumulative = [0.0]
    for i in range(len(points) - 1):
        cumulative.append(cumulative[-1] + math.dist(points[i], points[i + 1]))
    total = cumulative[-1]
    if total <= keep_px:
        return [points]

    cumulative = np.asarray(cumulative)
    out = []
    start = 0.0
    while start + keep_px <= total:
        out.append(
            [point_at(points, cumulative, start), point_at(points, cumulative, start + keep_px)]
        )
        start += period_px
    return out


def thin_to_budget(polylines, max_points):
    """Fit the polylines into a point budget. Returns (paths, estimate, period, dropped).

    Two mechanisms, in this order, because they answer different problems. Every
    surviving contour emits at least one dash and every path yields at least 2
    samples, so *N contours cost 2N points no matter how hard they are thinned*:
    a speckly edge map has to lose contours before density thinning can do
    anything. Only then is the remaining length thinned uniformly.
    """
    estimate = sum(estimate_samples(length) for length, _ in polylines)
    if estimate <= max_points:
        return [points for _length, points in polylines], estimate, None, 0

    # Floor first: the budget affords max_points/2 contours, and the list is
    # already longest-first, so what goes is the speckle.
    dropped = 0
    max_contours = max(1, int(max_points) // 2)
    if len(polylines) > max_contours:
        dropped = len(polylines) - max_contours
        polylines = polylines[:max_contours]
        estimate = sum(estimate_samples(length) for length, _ in polylines)
        if estimate <= max_points:
            return [points for _length, points in polylines], estimate, None, dropped

    # Then density. The period cannot be solved for directly -- a contour shorter
    # than the period still emits one dash, so the affordable spacing depends on
    # how the length is distributed between few long contours and many short
    # ones. Bisect for the tightest period whose dash count fits the budget.
    total_length = sum(length for length, _ in polylines)
    keep_px = 2.0 * SAMPLE_SPACING_PX
    budget_dashes = max(1.0, max_points / 2.0)

    def dash_count(period):
        return sum(
            1 if length <= keep_px else int((length - keep_px) // period) + 1
            for length, _ in polylines
        )

    low, high = keep_px * 1.5, max(total_length, keep_px * 2.0)
    for _ in range(40):
        middle = 0.5 * (low + high)
        if dash_count(middle) > budget_dashes:
            low = middle
        else:
            high = middle
    period_px = high

    paths = []
    for _length, points in polylines:
        paths.extend(dash(points, keep_px, period_px))
    estimate = sum(estimate_samples(polyline_length(path)) for path in paths)
    return paths, estimate, period_px, dropped


def svg_document(paths, render_size):
    """One ``<path>`` per polyline, in canvas pixel coordinates.

    The declared width/height must equal ``--render-size``: ``load_avoid_points``
    warns on a mismatch, and that warning is the only thing standing between a
    mis-sized SVG and points that are silently in the wrong place.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{render_size}" '
        f'height="{render_size}" viewBox="0 0 {render_size} {render_size}">',
    ]
    for points in paths:
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        lines.append(f'  <path d="{d}" fill="none" stroke="black" stroke-width="1"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def generate(image, out_path, mask=None, **options):
    """Write the attract SVG and report what went into it.

    Returns a stats dict rather than printing: the run loop wants a log line, the
    API wants JSON, and the test wants numbers to assert on.
    """
    settings = {**DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
    edges = edge_map(
        image,
        mask=mask,
        low=settings["low"],
        high=settings["high"],
        blur=settings["blur"],
        roi=settings["roi"],
    )
    render_size = edges.shape[0]
    found = extract_polylines(
        edges, simplify=settings["simplify"], min_length=settings["min_length"]
    )
    paths, points, period, dropped = thin_to_budget(found, int(settings["max_points"]))

    Path(out_path).write_text(svg_document(paths, render_size))
    return {
        "path": str(out_path),
        "render_size": render_size,
        "edge_pixels": int((edges > 0).sum()),
        "contours": len(found),
        "dropped": dropped,
        "paths": len(paths),
        "points": points,
        "dash_period": period,
        "settings": settings,
    }


def describe(stats):
    """The one-line summary the run prints, and the API echoes."""
    thinning = (
        "full density"
        if stats["dash_period"] is None
        else f"thinned to one {2.0 * SAMPLE_SPACING_PX:.0f} px dash every "
        f"{stats['dash_period']:.0f} px"
    )
    return (
        f"{stats['edge_pixels']} edge px -> {stats['contours']} contours "
        f"({stats['dropped']} dropped as speckle, {thinning}) "
        f"-> ~{stats['points']} attract points"
    )
