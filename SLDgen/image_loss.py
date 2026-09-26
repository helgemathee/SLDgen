"""Image fidelity loss (opt-in via ``--image-loss``), blended with SDS per gradient.

SLDgen's optimiser never compares the drawing with the photograph: SDS asks "is
this a plausible single-line drawing of the caption", and the photograph only
reaches it as ControlNet conditioning and as the stipple mask. This module adds
a term that does compare, and a dial for how much it counts (Spec 6).

**The dial is a gradient blend, not a loss weight.** The SDS gradient carries
``sigma_t**2`` from a fresh random timestep and varies by an order of magnitude
between consecutive steps, so ``loss_sds + w * loss_img`` gives a ``w`` whose
influence wanders through the run. Instead, after the SDS backward has left
``g_sds`` in ``control_points.grad``, :func:`blend_gradients` rewrites it as

    g = (1 - alpha) * g_sds + alpha * (|g_sds| / |g_img|) * g_img

The image gradient is rescaled to the SDS gradient's own norm, so ``alpha`` means
the same thing at every step, and the result stays at SDS scale: the regularisers
that accumulate on top at their raw scale keep exactly their upstream balance.

**Everything is canvas space.** The curve lives in ``[0, render_size)`` pixels,
x right, y down -- the frame ``targets.get_target`` produces. The default edge
target is derived from ``args.input_image`` inside the run, reusing
:func:`canny_attract.edge_map`, for the same reason ``--attract-canny`` is: that
is the only place canvas space exists without reproducing the pipeline. A
supplied target must already be canvas space at ``--render-size``.

No diffusers and no pydiffvg here, so ``test_image_loss_geom.py`` runs on a CPU
in seconds.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import canny_attract
from .polylines import parse_polylines, polyline_points

#: Where decay/ramp start when ``--image-loss-schedule-start`` is not given.
DEFAULT_START = {"decay": 0.5, "ramp": 0.05}

#: Edge pixels kept for the chamfer term. Above this a fixed stride thins them.
EDGE_POINT_CAP = 20000

#: Raw edge-pixel counts outside this band usually mean the thresholds are off.
EDGE_PIXELS_SANE = (2000, 200000)

#: A supplied target with at least this share of pure 0/255 pixels is an edge map.
BINARY_SHARE = 0.99

TERMS = ("chamfer", "pyramid", "landmark")

LOG_COLUMNS = (
    "epoch",
    "alpha",
    "sds_norm",
    "img_norm",
    "cosine",
    "skipped",
    "loss_sds",
    "loss_img",
    "chamfer",
    "pyramid",
    "landmark",
)

#: ``skipped`` column values: the blend ran, the image gradient vanished (g_sds
#: kept), or the SDS gradient vanished (g_img used at its own scale).
SKIP_NONE, SKIP_IMG, SKIP_SDS = 0, 1, 2


# --------------------------------------------------------------------------- #
# Schedule and blend
# --------------------------------------------------------------------------- #


def alpha_at(epoch, num_iter, schedule, weight, start=None):
    """The blend weight at an absolute epoch.

    A pure function of the absolute epoch and the horizon, never of "iterations
    in this process", so a segmented run follows the same schedule as an
    uninterrupted one (Spec 1 SS3). ``weight`` is always the alpha the run ends on.
    """
    if schedule == "constant":
        return float(weight)
    lo = DEFAULT_START[schedule] if start is None else float(start)
    t = epoch / max(num_iter - 1, 1)
    return lo + (float(weight) - lo) * t


def blend_gradients(g_sds, g_img, alpha, eps=1e-7):
    """Blend two gradients of one tensor, anchored to the SDS gradient's norm.

    Global normalisation -- one scalar per gradient -- on purpose: per-point
    normalisation would flatten the magnitude structure inside each field.

    Returns ``(blended, stats)`` with ``stats`` holding ``sds_norm``,
    ``img_norm``, ``cosine`` (``None`` when either norm is zero) and ``skipped``.
    """
    n_sds = float(g_sds.norm())
    n_img = float(g_img.norm())
    cosine = None
    if n_sds > 0.0 and n_img > 0.0:
        cosine = float((g_sds * g_img).sum()) / (n_sds * n_img)
    stats = {"sds_norm": n_sds, "img_norm": n_img, "cosine": cosine, "skipped": SKIP_NONE}

    if n_img < eps:
        # The curve already sits on the evidence, or the term saturated: a
        # rescale would blow numerical noise up into a full-strength direction.
        stats["skipped"] = SKIP_IMG
        return g_sds.clone(), stats
    if n_sds < eps:
        stats["skipped"] = SKIP_SDS
        return g_img.clone(), stats
    if alpha == 0.0:
        # Exactly g_sds, not (1 - 0) * g_sds + 0 * ..., which may differ in the
        # sign of zero.
        return g_sds.clone(), stats

    blended = (1.0 - alpha) * g_sds + (alpha * n_sds / n_img) * g_img
    return blended, stats


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def curve_samples(renderer, n):
    """About ``n`` points along the current curve, differentiable.

    ``renderer.sampled_curve2d`` is recomputed by every ``get_image()`` at
    ``--sampling-rate`` fixed parameter values; a stride view of it needs no
    second basis evaluation and no second rasterisation.
    """
    points = renderer.sampled_curve2d
    return points[:: max(1, len(points) // int(n))]


def edge_points(edge_map_u8, cap=EDGE_POINT_CAP):
    """``(K, 2)`` float32 tensor of ``(x, y)`` for every edge pixel, capped.

    ``np.argwhere`` yields ``(row, col)``; the curve is ``(x, y) = (col, row)``.
    Over ``cap`` a *fixed* stride thins the points -- not a random draw, so a
    resumed segment rebuilds the identical tensor without touching the RNG.
    Returns ``(points, raw_count)``.
    """
    rows_cols = np.argwhere(np.asarray(edge_map_u8) > 0)
    raw = len(rows_cols)
    if raw > cap:
        rows_cols = rows_cols[:: math.ceil(raw / cap)]
    xy = rows_cols[:, ::-1].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(xy)), raw


def nearest_distances(sources, targets):
    """Distance from each ``sources`` point to its nearest ``targets`` point.

    The nearest neighbour is found without grad (a chunked ``cdist``), then the
    distance to it is recomputed differentiably. Same value and the same
    (sub)gradient as ``cdist(...).min(dim=1)``, without keeping an ``(S, K)``
    matrix alive for the backward pass.
    """
    with torch.no_grad():
        chunk = max(1, (1 << 24) // max(1, len(targets)))
        index = torch.cat(
            [
                torch.cdist(sources[i : i + chunk].detach(), targets).argmin(dim=1)
                for i in range(0, len(sources), chunk)
            ]
        )
    diff = sources - targets[index]
    # The epsilon keeps the gradient finite for a point exactly on its target.
    return torch.sqrt((diff * diff).sum(dim=1) + 1e-12)


def is_binary_map(gray):
    """True when ``gray`` is (almost) all 0/255 -- a finished edge map."""
    gray = np.asarray(gray)
    pure = np.count_nonzero((gray == 0) | (gray == 255))
    return pure >= BINARY_SHARE * gray.size


def load_supplied_target(path, render_size):
    """Load ``--image-loss-target`` as greyscale uint8, refusing a wrong size.

    Called from ``config.parse_arguments`` too, so a mis-sized file is refused
    before any model loads.
    """
    gray = np.asarray(Image.open(path).convert("L"))
    if gray.shape != (render_size, render_size):
        raise ValueError(
            f"--image-loss-target {path} is {gray.shape[1]}x{gray.shape[0]}, but "
            f"--render-size is {render_size}. A canvas-space PNG at the render size is "
            "expected: run sld_edge_target.py on a previous run's input.png."
        )
    return gray


def prepare_edges(
    image,
    mask=None,
    low=canny_attract.DEFAULT_LOW,
    high=canny_attract.DEFAULT_HIGH,
    blur=3,
    clahe_clip=0.0,
    clahe_grid=8,
    roi=None,
    preserve_silhouette=False,
):
    """Canny edge map of a canvas-space image, masked, 0/255.

    With only ``low``/``high``/``blur`` this is exactly what ``--image-loss``
    derives in the run, which is what lets the API preview say "no file needed".
    The extras are for ``sld_edge_target.py``:

    * ``clahe_clip > 0`` lifts shadow detail before Canny, so dark curly hair
      becomes edges instead of a silhouetted blob;
    * ``preserve_silhouette`` unions the mask's inner boundary back in, because
      aggressive CLAHE can raise a dark region toward the background and erase
      its outline;
    * ``roi`` keeps only a canvas-pixel box, applied last.
    """
    import cv2

    gray = canny_attract.as_gray(image)
    if clahe_clip and clahe_clip > 0:
        grid = max(1, int(clahe_grid))
        gray = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid)).apply(gray)
    edges = canny_attract.edge_map(gray, low=low, high=high, blur=blur)

    inside = canny_attract.as_mask(mask, edges.shape)
    if inside is not None:
        edges[~inside] = 0
        if preserve_silhouette:
            solid = inside.astype(np.uint8)
            boundary = solid - cv2.erode(solid, np.ones((3, 3), np.uint8))
            edges[boundary > 0] = 255

    if roi is not None:
        x0, y0, x1, y1 = roi
        size_y, size_x = edges.shape
        keep = np.zeros_like(edges)
        xs = slice(max(0, int(round(min(x0, x1)))), min(size_x, int(round(max(x0, x1)))))
        ys = slice(max(0, int(round(min(y0, y1)))), min(size_y, int(round(max(y0, y1)))))
        keep[ys, xs] = 1
        edges = edges * keep
    return edges


def build_edge_map(args, input_image, mask):
    """The effective chamfer target, 0/255, in canvas space. Returns ``(edges, source)``.

    A supplied binary map is used as is (inverted if its edges are the dark
    pixels); a supplied non-binary image gets Canny with the run's settings; no
    file means Canny over the canvas image, which is what ``sld_edge_target.py``
    reproduces for the preview. The mask is applied in every case, so the
    square-pad border never counts as an edge.
    """
    if args.image_loss_target:
        gray = load_supplied_target(args.image_loss_target, args.render_size)
        if is_binary_map(gray):
            edges = np.where(gray > 127, 255, 0).astype(np.uint8)
            if np.count_nonzero(edges) > edges.size // 2:
                # Black lines on white paper: the edges are the minority value.
                edges = 255 - edges
            inside = canny_attract.as_mask(mask, edges.shape)
            if inside is not None:
                edges[~inside] = 0
            return edges, f"supplied edge map {args.image_loss_target}"
        image, source = gray, f"Canny over supplied image {args.image_loss_target}"
    else:
        image, source = input_image, "Canny over the canvas image"

    edges = prepare_edges(
        image,
        mask=mask,
        low=args.image_loss_canny_low,
        high=args.image_loss_canny_high,
        blur=args.image_loss_canny_blur,
    )
    return edges, source


def load_landmarks(path, render_size):
    """``(xy (L, 2), weight (L,))`` from a ``sld_landmarks.py`` JSON.

    Polylines (Spec 7 addendum SS6) are densified here, each sharing its weight
    over its points, and appended: the landmark term sees one flat list.

    Refused unless it declares canvas space at the render size. There is no
    rescaling path on purpose: a coordinate mismatch here looks exactly like
    "the feature does not work", so the contract is the one every spatial input
    in this repo has. Called from ``config.parse_arguments`` too.
    """
    hint = "run sld_landmarks.py on a previous run's input.png at this --render-size"
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"--image-loss-landmarks {path} is not readable JSON ({exc}); {hint}.")
    if payload.get("space") != "canvas":
        raise ValueError(f"--image-loss-landmarks {path} is not in canvas space; {hint}.")
    if list(payload.get("image_size") or []) != [render_size, render_size]:
        raise ValueError(
            f"--image-loss-landmarks {path} was made at {payload.get('image_size')}, but "
            f"--render-size is {render_size}; {hint}."
        )
    entries = payload.get("landmarks") or []
    try:
        xy = [[float(e["xy"][0]), float(e["xy"][1])] for e in entries]
        weight = [float(e.get("weight", 1.0)) for e in entries]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"--image-loss-landmarks {path} has a malformed landmark ({exc}).")
    try:
        line_xy, line_weight = polyline_points(parse_polylines(payload.get("polylines")))
    except ValueError as exc:
        raise ValueError(f"--image-loss-landmarks {path}: {exc}.")
    xy += line_xy
    weight += line_weight
    if not xy or min(weight) < 0 or sum(weight) <= 0:
        raise ValueError(
            f"--image-loss-landmarks {path} needs at least one landmark with a positive weight."
        )
    return torch.tensor(xy, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)


def render_again(renderer):
    """The current drawing, rasterised by a *fresh* DiffVG call.

    Not the SDS raster, on purpose. DiffVG's backward accumulates into gradient
    buffers that belong to the forward call's scene and are zeroed only when
    that scene is built, so a second backward through the raster SDS already
    backpropagated returns g_sds + g_pyramid instead of g_pyramid (measured:
    cosine 0.99999 with g_sds, identical norms). A new forward over the same
    shapes, at the same fixed sampling seed, is the same image with its own
    buffers. Compositing mirrors ``SLDBSplinePainter.get_image``.
    """
    img = renderer.render_warp()
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=img.device
    ) * (1 - alpha)
    return img.unsqueeze(0).permute(0, 3, 1, 2)


#: Pyramid resolutions, capped at --render-size.
PYRAMID_LEVELS = (64, 128, 256, 512)


def ink_of(image):
    """``(1, 1, H, W)`` ink from an ``(N, 3, H, W)`` image in [0, 1]: 0 on white paper."""
    return 1.0 - image[:1].mean(dim=1, keepdim=True)


def ink_distribution(ink, level):
    """Ink pooled to ``level`` x ``level`` and normalised to unit sum."""
    pooled = F.adaptive_avg_pool2d(ink, level)
    return pooled / (pooled.sum() + 1e-8)


# --------------------------------------------------------------------------- #
# The loss
# --------------------------------------------------------------------------- #


class ImageFidelityLoss:
    """The image term: a weighted sum of chamfer / pyramid / landmark.

    Every target is built once here; a step does no disk I/O and no Canny.
    """

    def __init__(self, args, input_image, mask, canvas_tensor, device):
        self.device = device
        self.n_samples = int(args.image_loss_curve_samples)

        weights = {
            "chamfer": float(args.image_loss_chamfer),
            "pyramid": float(args.image_loss_pyramid),
            "landmark": float(args.image_loss_landmark),
        }
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items() if v > 0}

        self.edge_pts = None
        self.edge_stats = None
        if "chamfer" in self.weights:
            edges, source = build_edge_map(args, input_image, mask)
            out = Path(args.output_dir) / "image_loss_target.png"
            Image.fromarray(edges).save(out)
            points, raw = edge_points(edges)
            if len(points) == 0:
                raise ValueError(
                    f"the image-loss edge target has no edge pixels ({source}). Lower "
                    "--image-loss-canny-low/high, or supply a different --image-loss-target."
                )
            self.edge_pts = points.to(device)
            self.edge_stats = {"source": source, "raw": raw, "kept": len(points), "path": out}

        self.landmark_xy = None
        if "landmark" in self.weights:
            xy, weight = load_landmarks(args.image_loss_landmarks, args.render_size)
            self.landmark_xy = xy.to(device)
            self.landmark_w = weight.to(device)

        self.levels = []
        self.pyr_target = []
        if "pyramid" in self.weights:
            size = int(canvas_tensor.shape[-1])
            self.levels = [level for level in PYRAMID_LEVELS if level <= size] or [size]
            ink = ink_of(canvas_tensor.detach().to(device))
            self.pyr_target = [ink_distribution(ink, level) for level in self.levels]

    def describe(self):
        """The log line the run prints."""
        terms = ", ".join(f"{k} {v:.2f}" for k, v in self.weights.items())
        parts = [f"terms: {terms}", f"{self.n_samples} curve samples"]
        if self.levels:
            parts.append("pyramid levels " + "/".join(str(level) for level in self.levels))
        if self.landmark_xy is not None:
            parts.append(f"{len(self.landmark_xy)} landmarks")
        if self.edge_stats is not None:
            s = self.edge_stats
            parts.append(f"{s['source']}: {s['raw']} edge px -> {s['kept']} kept")
            lo, hi = EDGE_PIXELS_SANE
            if not lo <= s["raw"] <= hi:
                parts.append(
                    f"WARNING: {s['raw']} edge pixels is outside {lo}-{hi}; "
                    "the Canny thresholds are probably wrong"
                )
        return "; ".join(parts)

    def chamfer(self, points):
        """Mean distance from the curve to its nearest edge pixel, in pixels.

        One-directional by specification: the line may omit whatever it likes
        but should not invent structure. Coverage is ``--attract-canny``'s job.
        """
        return nearest_distances(points, self.edge_pts).mean()

    def landmark(self, points):
        """Weighted mean distance from each landmark to its nearest curve point, px.

        The opposite direction to chamfer: here the requirement *is* coverage --
        every anchor must have line near it. Right for twenty anchors, wrong for
        twenty thousand edge pixels.
        """
        distances = nearest_distances(self.landmark_xy, points)
        return (self.landmark_w * distances).sum() / self.landmark_w.sum()

    def pyramid(self, raster):
        """Mean L1 between the drawing's and the photograph's ink distributions.

        Each level is normalised to unit sum, so the term compares *where* the
        ink is, not how much: a one-pixel line can never match a photograph's
        darkness, and a raw L1 would only ask for more ink everywhere. After
        normalisation every level's L1 lies in [0, 2], so the levels are
        weighted equally. Backpropagates through DiffVG a second time.
        """
        ink = ink_of(raster)
        total = sum(
            (ink_distribution(ink, level) - target).abs().sum()
            for level, target in zip(self.levels, self.pyr_target)
        )
        return total / len(self.levels)

    def __call__(self, renderer):
        """``-> (scalar loss, {term: float or None})``.

        Takes the renderer, not the SDS raster: the pyramid term renders its own
        (see :func:`render_again`).
        """
        values = {}
        if "chamfer" in self.weights or "landmark" in self.weights:
            points = curve_samples(renderer, self.n_samples)
        if "chamfer" in self.weights:
            values["chamfer"] = self.chamfer(points)
        if "landmark" in self.weights:
            values["landmark"] = self.landmark(points)
        if "pyramid" in self.weights:
            values["pyramid"] = self.pyramid(render_again(renderer))
        loss = sum(self.weights[k] * values[k] for k in self.weights)
        parts = {k: (float(values[k]) if k in values else None) for k in TERMS}
        return loss, parts


# --------------------------------------------------------------------------- #
# Diagnostics log
# --------------------------------------------------------------------------- #


def _cell(value):
    """Floats at 7 significant digits.

    More would only record noise: DiffVG's backward is not bit-deterministic
    (two identical runs differ in the last digits of ``sds_norm``), so a
    full-precision column could never match between two runs anyway.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


class ImageLossLog:
    """``image_loss_log.csv``: one row per epoch, flushed on every write.

    Each segment of a resumed run is a new process in the same directory. On
    resume the rows past the checkpoint epoch -- left behind by a segment killed
    between its checkpoint and its exit -- are dropped before appending, so a
    segmented run produces the same rows as an uninterrupted one.
    """

    def __init__(self, output_dir, resume_epoch=None):
        self.path = Path(output_dir) / "image_loss_log.csv"
        kept = []
        if resume_epoch is not None and self.path.exists():
            with open(self.path, newline="") as f:
                rows = list(csv.reader(f))
            kept = [row for row in rows[1:] if row and int(row[0]) <= resume_epoch]
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file, lineterminator="\n")
        self.writer.writerow(LOG_COLUMNS)
        self.writer.writerows(kept)
        self.file.flush()

    def write(self, epoch, alpha, stats, loss_sds, loss_img, parts):
        self.writer.writerow(
            [
                _cell(int(epoch)),
                _cell(float(alpha)),
                _cell(stats["sds_norm"]),
                _cell(stats["img_norm"]),
                _cell(stats["cosine"]),
                _cell(int(stats["skipped"])),
                _cell(float(loss_sds)),
                _cell(float(loss_img)),
            ]
            + [_cell(parts[k]) for k in TERMS]
        )
        self.file.flush()

    def close(self):
        self.file.close()
