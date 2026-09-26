#!/usr/bin/env python
"""Extract face landmarks for ``--image-loss-landmarks`` (Spec 6 SS8.2, Spec 7 SS4).

Identity in a face sits in perhaps twenty points; a generic image loss gives an
eye corner and a hoodie fold the same pressure. This detects the face and writes
the identity-carrying points, weighted, as a canvas-space JSON the landmark term
reads.

Detection is view-aware (Spec 7 SS4):

1. MediaPipe Face Mesh. The head turn (yaw) is read off the mesh's own surface
   normals; far-side points are dropped past a per-point limit, because the mesh
   always fits a whole face and puts the hidden eye on the nose bridge. For
   strongly turned faces the profile points (nose tip, lips, chin, ...) are read
   off the subject's silhouette, which the saturated mesh cannot place.
2. Face Mesh finds nothing (a true profile): MediaPipe Pose supplies the facing
   direction, the visible eye and mouth corner, and the silhouette the rest.

Which points are written is ``--landmark-set`` (Spec 7 addendum): ``sparse``
(the portrait preset, byte-identical to before), ``standard`` (~50), ``dense``
(120 spread over the face) or ``pose-locked`` (dense, checked against a rigid
fit of MediaPipe's canonical face in ``assets/mediapipe``). Weights are shared
out so every region keeps its sparse share. ``--pose-report``,
``--include-hairline`` and ``--include-glasses`` add a pose diagnostic and
polylines.

It takes canvas-space input only -- a run's ``input.png`` (and ``mask.png``).
Pointing it at the original photograph produces landmarks in the wrong frame,
silently. The run refuses a file whose ``image_size`` is not ``--render-size``,
but it cannot detect one taken from the wrong image of the right size.

Needs MediaPipe in this interpreter: ``pip install mediapipe==0.10.21`` (the
0.10 line still bundles the Face Mesh and Pose models and works with numpy 1.x).
Exit codes: 0 written, 2 no face found / bad input, 3 MediaPipe missing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from SLDgen.polylines import parse_polylines

#: The ``portrait`` preset: (name, Face Mesh index, weight). Left/right are the
#: subject's. Eye corners and pupils carry identity most, then nostrils and
#: mouth corners, then brow arcs; the jaw only anchors the outline.
PORTRAIT = (
    ("right_eye_outer", 33, 3.0),
    ("right_eye_inner", 133, 3.0),
    ("left_eye_inner", 362, 3.0),
    ("left_eye_outer", 263, 3.0),
    ("right_pupil", 468, 3.0),
    ("left_pupil", 473, 3.0),
    ("right_nostril", 98, 2.0),
    ("left_nostril", 327, 2.0),
    ("mouth_right", 61, 2.0),
    ("mouth_left", 291, 2.0),
    ("right_brow_outer", 70, 1.5),
    ("right_brow_mid", 105, 1.5),
    ("right_brow_inner", 107, 1.5),
    ("left_brow_inner", 336, 1.5),
    ("left_brow_mid", 334, 1.5),
    ("left_brow_outer", 300, 1.5),
    ("chin", 152, 0.5),
    ("jaw_right", 172, 0.5),
    ("jaw_left", 397, 0.5),
    ("cheek_right", 234, 0.5),
    ("cheek_left", 454, 0.5),
)

#: How far (degrees) the face may turn away before a far-side point is hidden
#: (Spec 7 SS4.3). Keyed by the name with the side removed.
CULL_LIMIT = {
    "cheek": 20.0,
    "jaw": 20.0,
    "eye_outer": 25.0,
    "brow_outer": 25.0,
    "mouth": 35.0,
    "pupil": 38.0,
    "brow_mid": 38.0,
    "nostril": 38.0,
    "eye_inner": 40.0,
    "brow_inner": 45.0,
    # standard set (Spec 7 addendum SS3.1)
    "temple": 20.0,
    "jaw_high": 20.0,
    "jaw_low": 20.0,
    "chin_side": 30.0,
    "brow_outer_mid": 30.0,
    "eye_upper": 32.0,
    "eye_lower": 32.0,
    "ala": 38.0,
    "lip_peak": 38.0,
    "lower_lip": 38.0,
    "brow_inner_mid": 42.0,
}

#: The ``standard`` set adds these to the portrait points (Spec 7 addendum SS2):
#: (name, Face Mesh index). The midline names are shared with the silhouette
#: profile points, which replace them when the outline supplies them.
STANDARD_EXTRA = (
    ("right_brow_outer_mid", 63),
    ("right_brow_inner_mid", 66),
    ("left_brow_inner_mid", 296),
    ("left_brow_outer_mid", 293),
    ("right_eye_upper", 159),
    ("right_eye_lower", 145),
    ("left_eye_upper", 386),
    ("left_eye_lower", 374),
    ("nasion", 168),
    ("nose_tip", 1),
    ("subnasale", 2),
    ("right_ala", 64),
    ("left_ala", 294),
    ("upper_lip", 0),
    ("right_lip_peak", 37),
    ("left_lip_peak", 267),
    ("stomion", 13),
    ("lower_lip", 17),
    ("right_lower_lip", 84),
    ("left_lower_lip", 314),
    ("right_temple", 127),
    ("right_jaw_high", 132),
    ("right_jaw_low", 136),
    ("right_chin_side", 148),
    ("left_temple", 356),
    ("left_jaw_high", 361),
    ("left_jaw_low", 365),
    ("left_chin_side", 377),
)

LANDMARK_SETS = ("sparse", "standard", "dense", "pose-locked")
#: Points in the dense and pose-locked sets.
DENSE_COUNT = 120

#: MediaPipe's canonical face (Apache-2.0, see assets/mediapipe/NOTICE): the
#: dense layout, the weight budgets and the pose-locked rigid fit.
CANONICAL_MODEL = Path(__file__).resolve().parent / "assets" / "mediapipe" / "canonical_face_model.obj"
#: The canonical model has no irises; each refined-mesh iris point (468-472
#: right, 473-477 left) stands at the middle of its eye's corners and lids.
IRIS_FROM = {"right": (33, 133, 159, 145), "left": (362, 263, 386, 374)}

#: A point further than this (canonical cm) from every portrait anchor belongs
#: to the surface cell, whose budget is SURFACE_BUDGET (SS4).
SURFACE_RADIUS = 2.5
SURFACE_BUDGET = 1.0

#: Stable points for the rigid fit: eye corners, nostrils, nose base, mouth
#: corners, chin -- and their portrait names, for the far-side limits.
STABLE = {
    33: "right_eye_outer",
    133: "right_eye_inner",
    362: "left_eye_inner",
    263: "left_eye_outer",
    98: "right_nostril",
    327: "left_nostril",
    2: "subnasale",
    61: "mouth_right",
    291: "mouth_left",
    152: "chin",
}
#: A detection within this fraction of the inter-ocular distance (at least
#: 3 px) of the fitted model counts as consistent (SS3.3). Measured on real
#: portraits, a person's own shape already differs from the canonical face by
#: 5-10 % (median) and up to ~30 % (90th percentile, outline points): a tighter
#: bound would replace their identity with the average face's.
FIT_TOLERANCE = 0.25
#: Points this close to the canonical midline (cm) are never hidden: at the
#: turns the mesh can represent, the midline stays in view (as in Spec 7 SS4.3).
MIDLINE_HALF_WIDTH = 0.5
#: Depth margin (canonical cm) before a surface in front hides a point.
OCCLUSION_MARGIN = 0.05

#: Hairline (SS7.2): the upper face-outline points marched from (the temples
#: are left out: glasses arms and sideburns cross them), and the skin samples.
HAIRLINE_OUTLINE = (54, 103, 67, 109, 10, 338, 297, 332, 284)
SKIN_SAMPLES = (151, 108, 337, 50, 280)
BROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
HAIRLINE_WEIGHT = 1.5

#: From here on the mesh cannot represent the turn, and the outline carries the
#: face: add the silhouette's profile points.
PROFILE_YAW = 42.0
#: Below this |yaw| a face counts as frontal (reported, not acted on).
FRONTAL_YAW = 20.0

#: Profile points read off the silhouette (Spec 7 SS4.4), with their weights.
PROFILE_WEIGHTS = {
    "brow_ridge": 1.5,
    "nasion": 2.0,
    "nose_tip": 3.0,
    "subnasale": 1.5,
    "upper_lip": 2.0,
    "stomion": 1.5,
    "lower_lip": 2.0,
    "chin_front": 1.0,
}

#: Forehead-to-nose-tip midline of the mesh, whose mean normal gives the yaw.
MIDLINE = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4)
#: Mesh indices of each eye's corners, the mouth corners and the nose tip.
EYE_CORNERS = {"right": (33, 133), "left": (362, 263)}
MOUTH_CORNER = {"right": 61, "left": 291}
NOSE_TIP = 4

#: MediaPipe Pose indices (the subject's left/right).
POSE = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
}

#: Side of the upscaled face crop the mesh runs on when the face is small.
CROP_SIZE = 384
#: A detection region smaller than this is upscaled to it.
REGION_SIZE = 512

EXIT_NO_FACE = 2
EXIT_NO_MEDIAPIPE = 3


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Write canvas-space face landmarks for --image-loss-landmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python sld_landmarks.py --image work/jobs/<id>/target/run/input.png \\\n"
            "      --mask work/jobs/<id>/target/run/mask.png --out landmarks.json\n"
            "  python sldgen.py --target photo.png --image-loss --image-loss-landmark 1 \\\n"
            "      --image-loss-landmarks landmarks.json\n"
        ),
    )
    parser.add_argument(
        "--image", required=True, help="Canvas-space image: a run's input.png, never the photo."
    )
    parser.add_argument("--out", required=True, help="JSON to write.")
    parser.add_argument(
        "--preset",
        choices=["portrait", "all"],
        default="portrait",
        help="portrait: ~20 weighted identity points, view-aware. all: every mesh point at weight 1.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Canvas-space subject mask (a run's mask.png). Enables the profile points.",
    )
    parser.add_argument(
        "--landmark-set",
        choices=LANDMARK_SETS,
        default="sparse",
        help=(
            "sparse: the portrait preset (default). standard: ~50 points, adds brow arcs, lids, "
            "nose, lip contour, more outline. dense: 120 points spread over the face. "
            "pose-locked: dense, checked against a rigid fit of the canonical face."
        ),
    )
    parser.add_argument(
        "--pose-report",
        action="store_true",
        help="Print yaw/pitch/roll from a rigid fit and write them under 'pose'.",
    )
    parser.add_argument(
        "--include-glasses",
        default=None,
        metavar="JSON",
        help="Canvas-pixel polylines (glasses rims) to copy into the file.",
    )
    parser.add_argument(
        "--include-hairline",
        action="store_true",
        help="Trace the hairline above the forehead as a polyline (frontal and turned faces).",
    )
    parser.add_argument(
        "--hair-mask",
        default=None,
        help="Canvas-space hair mask for --include-hairline, instead of the skin-colour test.",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Detect only inside this canvas-pixel box (grown by a quarter for context).",
    )
    args = parser.parse_args(argv)
    if args.preset == "all" and args.landmark_set != "sparse":
        parser.error("--preset all is every mesh point; it cannot be combined with --landmark-set")
    if args.hair_mask and not args.include_hairline:
        parser.error("--hair-mask needs --include-hairline")
    return args


# -- geometry (pure numpy; tested without MediaPipe) --------------------------


def _tessellation():
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

    return FACEMESH_TESSELATION


def mesh_triangles(edges):
    """Triangles (T, 3) of a mesh given only as an edge set: its 3-cliques."""
    edges = {tuple(sorted(edge)) for edge in edges}
    neighbours = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    triangles = set()
    for a, b in edges:
        for c in neighbours[a] & neighbours[b]:
            triangles.add(tuple(sorted((a, b, c))))
    return np.array(sorted(triangles), dtype=np.int64), neighbours


def smoothed_normals(vertices, triangles, neighbours, rings=3):
    """Outward unit normals per vertex, averaged over a ``rings``-ring neighbourhood.

    Outward is away from the vertex centroid, which is right for a face-shaped
    shell. The smoothing is what makes the normal describe the head's surface
    rather than an eye socket's.
    """
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    face_normals = np.cross(b - a, c - a)
    centres = (a + b + c) / 3.0
    inward = np.sum(face_normals * (centres - vertices.mean(axis=0)), axis=1) < 0
    face_normals[inward] *= -1
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_normals)
    smoothed = normals.copy()
    for _ in range(rings):
        spread = smoothed.copy()
        for index, around in neighbours.items():
            if index < len(spread):
                spread[index] = smoothed[index] + smoothed[list(around)].sum(axis=0)
        smoothed = spread
    return smoothed / (np.linalg.norm(smoothed, axis=1, keepdims=True) + 1e-12)


def yaw_from_normals(normals):
    """Head turn in degrees from the midline normals; negative faces image-left.

    MediaPipe's z grows away from the camera, so a face looking at the camera
    has normals toward -z.
    """
    mean = normals[list(MIDLINE)].mean(axis=0)
    return float(np.degrees(np.arctan2(mean[0], -mean[2])))


def mesh_yaw(points3d):
    triangles, neighbours = mesh_triangles(_tessellation())
    vertices = np.asarray(points3d, dtype=np.float64)[:468]
    return yaw_from_normals(smoothed_normals(vertices, triangles, neighbours))


def split_side(name):
    """``(side, part)`` of a portrait name: ``left_eye_outer`` -> ('left', 'eye_outer')."""
    for side in ("left", "right"):
        if name.startswith(side + "_"):
            return side, name[len(side) + 1 :]
        if name.endswith("_" + side):
            return side, name[: -len(side) - 1]
    return "mid", name


def far_side(yaw):
    """The subject's side turned away: negative yaw faces image-left, hiding the right."""
    return "right" if yaw < 0 else "left"


def is_hidden(name, yaw):
    side, part = split_side(name)
    if side == "mid" or side != far_side(yaw):
        return False
    return abs(yaw) > CULL_LIMIT.get(part, 90.0)


def view_kind(yaw):
    if yaw is None or abs(yaw) >= PROFILE_YAW:
        return "profile"
    return "frontal" if abs(yaw) < FRONTAL_YAW else "turned"


def _point(name, xy, weight, source):
    return {
        "name": name,
        "xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
        "weight": float(weight),
        "source": source,
    }


def select_portrait(points, yaw):
    """The portrait preset, minus the points hidden at this yaw: ``(chosen, dropped)``."""
    chosen, dropped = [], []
    for name, index, weight in PORTRAIT:
        if index >= len(points):  # the pupils need the refined (478-point) mesh
            continue
        if yaw is not None and is_hidden(name, yaw):
            dropped.append(name)
            continue
        chosen.append(_point(name, points[index][:2], weight, "mesh"))
    return chosen, dropped


def head_component(mask, eye):
    """The connected part of the mask that holds the eye (or lies nearest to it).

    Stray specks, a hand or a second figure can sit further forward than the
    face; only the blob the face belongs to has the face's outline.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    x, y = int(round(eye[0])), int(round(eye[1]))
    height, width = mask.shape
    label = labels[min(max(y, 0), height - 1), min(max(x, 0), width - 1)]
    if label == 0:
        ys, xs = np.nonzero(labels)
        nearest = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        label = labels[ys[nearest], xs[nearest]]
    return labels == label


def profile_curve(mask, facing, eye, mouth):
    """``(rows, front_x, forwardness)`` of the head's front edge, per row.

    ``facing`` is 'left' or 'right' in the image. The front of a row is its most
    forward mask pixel within three eye-to-mouth distances of the eye, so
    shoulders and background blobs further away do not count.
    """
    mask = head_component(mask, eye)
    height, width = mask.shape
    scale = max(mouth[1] - eye[1], 8.0)
    top = int(max(0, eye[1] - 1.4 * scale))
    bottom = int(min(height - 1, mouth[1] + 1.6 * scale))
    reach = 3.0 * scale
    x_lo, x_hi = int(max(0, eye[0] - reach)), int(min(width - 1, eye[0] + reach))
    sign = -1.0 if facing == "left" else 1.0
    rows = np.arange(top, bottom + 1)
    front = np.full(len(rows), np.nan)
    for k, y in enumerate(rows):
        xs = np.nonzero(mask[y, x_lo : x_hi + 1])[0]
        if len(xs):
            front[k] = (xs.min() if facing == "left" else xs.max()) + x_lo
    forward = sign * front
    return rows, front, forward, scale


def zigzag(values, start, step, threshold, count, stop=None):
    """Alternating extrema walking from ``start`` (a maximum) in direction ``step``.

    The classic zigzag: track the running extreme and register it once the curve
    has reversed by ``threshold``; a plateau registers at its middle. Returns up
    to ``count`` indices, first a minimum, then a maximum, and so on. The walk
    ends at ``stop`` (exclusive), a NaN row, or the end of the curve; a pending
    extreme is registered there only if the walk was ended by ``stop`` (a hard
    edge such as the neck), never by simply running out of band.
    """
    found = []
    looking_for_min = True
    best = start
    best_end = start
    i = start + step
    while 0 <= i < len(values) and len(found) < count:
        if stop is not None and i == stop:
            if abs(values[best] - values[found[-1] if found else start]) >= threshold:
                found.append((best + best_end) // 2)
            break
        value = values[i]
        if np.isnan(value):
            break
        if (looking_for_min and value < values[best]) or (not looking_for_min and value > values[best]):
            best = best_end = i
        elif value == values[best]:
            best_end = i
        elif abs(value - values[best]) >= threshold:
            found.append((best + best_end) // 2)
            looking_for_min = not looking_for_min
            best = best_end = i
        i += step
    return found


def neck_row(rows, front, start, scale):
    """First row below ``start`` where the outline jumps back by a quarter face: the neck."""
    for k in range(start + 1, len(rows) - 2):
        if np.isnan(front[k]) or np.isnan(front[k + 2]):
            continue
        if abs(front[k + 2] - front[k]) > 0.25 * scale:
            return k + 1
    return None


def silhouette_points(mask, facing, eye, mouth):
    """Named profile points on the head's outline (Spec 7 SS4.4), as a dict name -> (x, y).

    The nose tip is the most forward point between eye and mouth. From there a
    zigzag walks down (subnasale, upper lip, stomion, lower lip, the fold under
    it, chin) until the neck, and up (nasion, brow ridge) to just above the eye.
    A feature that the outline does not show clearly is left out, not guessed.
    """
    rows, front, forward, s = profile_curve(mask, facing, eye, mouth)
    threshold = max(0.9, 0.015 * s)
    found = {}
    ey, my = eye[1], mouth[1]

    band = np.nonzero((rows >= ey + 0.1 * s) & (rows <= my - 0.1 * s) & ~np.isnan(forward))[0]
    if len(band) < 3:
        return found
    nose = int(band[np.argmax(forward[band])])
    # The nose must stand out from the face above it, or this is not a profile.
    above = forward[max(0, nose - int(0.5 * s)) : nose]
    above = above[~np.isnan(above)]
    if not len(above) or forward[nose] - above.min() < 0.1 * s:
        return found
    found["nose_tip"] = nose

    neck = neck_row(rows, front, nose, s)
    down = zigzag(forward, nose, +1, threshold, 6, stop=neck)
    for name, index in zip(("subnasale", "upper_lip", "stomion", "lower_lip", None, "chin_front"), down):
        if name:
            found[name] = index

    # Upward only to just above the eye: higher, the outline is forehead or hair.
    top = int(np.searchsorted(rows, ey - 1.2 * s))
    up = zigzag(forward[top:], nose - top, -1, threshold, 2)
    for name, index in zip(("nasion", "brow_ridge"), up):
        if rows[index + top] >= ey - 0.8 * s or name == "brow_ridge":
            found[name] = index + top

    return {name: (float(front[k]), float(rows[k])) for name, k in found.items()}


def profile_landmarks(mask, facing, eye, mouth, nose_hint=None):
    """Silhouette points as landmarks, or [] when the outline is not the face's front.

    ``nose_hint`` (the mesh's nose tip) is the consistency check: a silhouette
    nose tip far from it means the outline belongs to hair, a hand or a bad mask.
    """
    found = silhouette_points(mask, facing, eye, mouth)
    if "nose_tip" not in found:
        return []
    if nose_hint is not None:
        scale = max(mouth[1] - eye[1], 8.0)
        if np.hypot(*(np.asarray(found["nose_tip"]) - np.asarray(nose_hint[:2]))) > 0.5 * scale:
            return []
    return [_point(name, xy, PROFILE_WEIGHTS[name], "silhouette") for name, xy in found.items()]


def grow_box(box, width, height, grow=0.25):
    """Integer ``(left, top, right, bottom)`` of the box grown on every side, clipped."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * grow, (y1 - y0) * grow
    left, top = int(max(0, np.floor(x0 - dx))), int(max(0, np.floor(y0 - dy)))
    right, bottom = int(min(width, np.ceil(x1 + dx))), int(min(height, np.ceil(y1 + dy)))
    if right - left < 8 or bottom - top < 8:
        raise ValueError(f"--box {box} leaves no usable region inside the {width}x{height} canvas")
    return left, top, right, bottom


def region_to_canvas(points, left, top, scale):
    """Map region pixels (x, y[, z]) back to canvas pixels."""
    points = np.array(points, dtype=np.float64)
    points[:, :2] = points[:, :2] / scale + np.array([left, top])
    if points.shape[1] > 2:
        points[:, 2] = points[:, 2] / scale
    return points


def pose_view(pose):
    """``(facing, near, yaw_estimate)`` from Pose's nose, eyes, ears and mouth.

    Facing is where the nose points relative to the ears. The yaw estimate comes
    from the eyes' horizontal separation relative to the eye-to-mouth height
    (about 1.15 on a frontal face, near 0 in profile); it is coarse, and
    reported as such.
    """
    nose = pose["nose"]
    ears_x = (pose["left_ear"][0] + pose["right_ear"][0]) / 2
    facing = "left" if nose[0] < ears_x else "right"
    near = "left" if facing == "left" else "right"
    eyes_y = (pose["left_eye"][1] + pose["right_eye"][1]) / 2
    mouth_y = (pose["mouth_left"][1] + pose["mouth_right"][1]) / 2
    height = max(mouth_y - eyes_y, 1e-6)
    ratio = abs(pose["left_eye"][0] - pose["right_eye"][0]) / height
    yaw = float(np.degrees(np.arccos(min(1.0, ratio / 1.15))))
    return facing, near, (-yaw if facing == "left" else yaw)


def pose_landmarks(pose, yaw):
    """Portrait-named points from Pose, the far side culled by the same table."""
    names = {
        "eye_inner": "eye_inner",
        "eye": "pupil",
        "eye_outer": "eye_outer",
    }
    chosen, dropped = [], []
    for side in ("left", "right"):
        for pose_part, part in names.items():
            name = f"{side}_{part}"
            if is_hidden(name, yaw):
                dropped.append(name)
            else:
                chosen.append(_point(name, pose[f"{side}_{pose_part}"], 3.0, "pose"))
        name = f"mouth_{side}"
        if is_hidden(name, yaw):
            dropped.append(name)
        else:
            chosen.append(_point(name, pose[f"mouth_{side}"], 2.0, "pose"))
    return chosen, dropped


# -- landmark sets (Spec 7 addendum; pure numpy, tested without MediaPipe) ------

_CANONICAL = None


def canonical_model():
    """``(vertices (478, 3), triangles (T, 3))`` of the canonical face, in image axes.

    x right, y down, z away from the camera (MediaPipe's own convention), in cm.
    Indices 0-467 are the Face Mesh points; 468-477 the iris points, placed at
    the middle of each eye (the model has none).
    """
    global _CANONICAL
    if _CANONICAL is None:
        vertices, triangles = [], []
        for line in CANONICAL_MODEL.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "v":
                vertices.append([float(v) for v in parts[1:4]])
            elif parts and parts[0] == "f":
                triangles.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
        vertices = np.array(vertices, dtype=np.float64) * np.array([1.0, -1.0, -1.0])
        irises = []
        for side in ("right", "left"):
            centre = vertices[list(IRIS_FROM[side])].mean(axis=0)
            irises += [centre] * 5
        _CANONICAL = (np.vstack([vertices, irises]), np.array(triangles, dtype=np.int64))
    return _CANONICAL


def farthest_points(vertices, seeds, count):
    """``seeds`` extended by farthest-point sampling to ``count`` indices (deterministic)."""
    chosen = list(dict.fromkeys(seeds))
    nearest = np.full(len(vertices), np.inf)
    for index in chosen:
        nearest = np.minimum(nearest, np.linalg.norm(vertices - vertices[index], axis=1))
    while len(chosen) < count:
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(vertices - vertices[index], axis=1))
    return chosen


def set_entries(landmark_set):
    """``[(name, mesh index), ...]`` of a landmark set; sparse is the portrait preset."""
    entries = [(name, index) for name, index, _weight in PORTRAIT]
    if landmark_set == "sparse":
        return entries
    entries += list(STANDARD_EXTRA)
    if landmark_set == "standard":
        return entries
    vertices, _triangles = canonical_model()
    named = {index: name for name, index in entries}
    seeds = [index for _name, index in entries if index < 468]
    extra = DENSE_COUNT - sum(index >= 468 for _name, index in entries)
    order = farthest_points(vertices[:468], seeds, extra)
    irises = [(name, index) for name, index in entries if index >= 468]
    return [(named.get(index, f"m{index}"), index) for index in order] + irises


def anchor_weights(entries):
    """Weights that keep every region's share of the total as in the portrait set (SS4).

    Each point joins the cell of its nearest portrait anchor on the canonical
    face and the cell shares the anchor's weight equally; points further than
    SURFACE_RADIUS from every anchor share SURFACE_BUDGET.
    """
    vertices, _triangles = canonical_model()
    anchors = vertices[[index for _name, index, _weight in PORTRAIT]]
    budgets = [weight for _name, _index, weight in PORTRAIT] + [SURFACE_BUDGET]
    cells = []
    for _name, index in entries:
        distance = np.linalg.norm(anchors - vertices[index], axis=1)
        nearest = int(np.argmin(distance))
        cells.append(nearest if distance[nearest] <= SURFACE_RADIUS else len(PORTRAIT))
    counts = np.bincount(cells, minlength=len(budgets))
    return [round(budgets[cell] / counts[cell], 4) for cell in cells], cells


def lateral_hidden(index, yaw):
    """The dense rule (SS3.2): a far-side point at ``u = |x| / max|x|`` hides past 45 - 25 u degrees."""
    vertices, _triangles = canonical_model()
    x = vertices[index, 0]
    if abs(x) < 1e-6:
        return False
    side = "right" if x < 0 else "left"  # the subject's right is image-left
    if side != far_side(yaw):
        return False
    u = abs(x) / np.abs(vertices[:468, 0]).max()
    return abs(yaw) > 45.0 - 25.0 * u


def set_hidden(name, index, yaw):
    """Named points by the per-name table (SS3.1), ``m<index>`` points by the lateral rule."""
    if yaw is None:
        return False
    if name == f"m{index}":
        return lateral_hidden(index, yaw)
    return is_hidden(name, yaw)


def select_set(points, yaw, landmark_set):
    """``(chosen, dropped)`` for the sparse, standard or dense set (raw detections)."""
    if landmark_set == "sparse":
        return select_portrait(points, yaw)
    entries = set_entries(landmark_set)
    weights, _cells = anchor_weights(entries)
    chosen, dropped = [], []
    for (name, index), weight in zip(entries, weights):
        if index >= len(points):
            continue
        if set_hidden(name, index, yaw):
            dropped.append(name)
            continue
        chosen.append(_point(name, points[index][:2], weight, "mesh"))
    return chosen, dropped


def umeyama(source, target):
    """``(scale, R, t)`` minimising ``|scale R source + t - target|`` (no reflection)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - mu_s, target - mu_t
    u, singular, vt = np.linalg.svd(xt.T @ xs / len(source))
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1.0
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(singular) @ d) / ((xs ** 2).sum() / len(source)))
    return scale, rotation, mu_t - scale * rotation @ mu_s


def fit_rigid(points3d, yaw=None):
    """Similarity fit of the canonical face to the detected mesh (SS3.3).

    Stable points only, far-side ones only inside their limits; then the worst
    quarter by residual is dropped and the fit repeated. Returns a dict with
    ``scale``, ``R``, ``t``, the ``used`` indices and the 2D ``residual_px``.
    """
    vertices, _triangles = canonical_model()
    points3d = np.asarray(points3d, dtype=np.float64)
    used = [i for i, name in STABLE.items() if yaw is None or not is_hidden(name, yaw)]
    if len(used) < 4:
        used = list(STABLE)
    scale, rotation, shift = umeyama(vertices[used], points3d[used])
    if len(used) >= 6:
        residual = np.linalg.norm(scale * vertices[used] @ rotation.T + shift - points3d[used], axis=1)
        keep = np.argsort(residual)[: math.ceil(0.75 * len(used))]
        used = [used[k] for k in sorted(keep)]
        scale, rotation, shift = umeyama(vertices[used], points3d[used])
    projected = scale * vertices[used] @ rotation.T + shift
    residual = np.linalg.norm(projected[:, :2] - points3d[used, :2], axis=1)
    return {
        "scale": scale,
        "R": rotation,
        "t": shift,
        "used": used,
        "residual_px": float(np.sqrt((residual ** 2).mean())),
    }


def project(fit):
    """Canvas pixels (478, 2) of every canonical point under the fitted pose."""
    vertices, _triangles = canonical_model()
    return (fit["scale"] * vertices @ fit["R"].T + fit["t"])[:, :2]


def euler_rotation(yaw=0.0, pitch=0.0, roll=0.0):
    """Image-axes rotation turn(yaw) . tilt(pitch) . spin(roll), degrees (SS3.4 signs)."""
    y, p, r = np.radians([yaw, pitch, roll])
    turn = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    tilt = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    spin = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return turn @ tilt @ spin


def pose_angles(rotation):
    """``(yaw, pitch, roll)`` in degrees with ``euler_rotation(yaw, pitch, roll) == rotation``.

    Yaw negative faces image-left (as ``view.yaw``); pitch positive looks up;
    roll positive tilts the head clockwise in the image. Yaw and pitch are read
    off the facing direction; roll is what is left about it.
    """
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    yaw = float(np.degrees(np.arctan2(forward[0], -forward[2])))
    pitch = float(np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2]))))
    spin = euler_rotation(yaw, pitch, 0.0).T @ rotation
    roll = float(np.degrees(np.arctan2(spin[1, 0], spin[0, 0])))
    return yaw, pitch, roll


def occluded(vertices, triangles, margin=OCCLUSION_MARGIN):
    """Per vertex: does a triangle not containing it cover it, nearer the camera?

    Orthographic, looking down +z (smaller z is nearer).
    """
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    e0, e1 = c[:, :2] - a[:, :2], b[:, :2] - a[:, :2]
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    denominator = d00 * d11 - d01 * d01
    flat = np.abs(denominator) < 1e-12
    denominator[flat] = 1.0
    hidden = np.zeros(len(vertices), dtype=bool)
    for i, q in enumerate(vertices):
        e2 = q[:2] - a[:, :2]
        d20, d21 = (e2 * e0).sum(1), (e2 * e1).sum(1)
        u = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        inside = (u >= 0) & (w >= 0) & (u + w <= 1) & ~flat
        inside &= ~(triangles == i).any(axis=1)
        depth = a[:, 2] + u * (c[:, 2] - a[:, 2]) + w * (b[:, 2] - a[:, 2])
        hidden[i] = bool(np.any(inside & (depth < q[2] - margin)))
    return hidden


def visible_points(fit):
    """Per canonical point: is it seen under the fitted rotation?

    Occlusion on the canonical surface decides, except on the midline (always
    seen) and for the irises, which sit behind the eye's surface triangles in
    the model and follow their eye's corners and lids instead.
    """
    vertices, triangles = canonical_model()
    visible = ~occluded(vertices @ fit["R"].T, triangles)
    visible[np.abs(vertices[:, 0]) < MIDLINE_HALF_WIDTH] = True
    for side, first in (("right", 468), ("left", 473)):
        visible[first : first + 5] = visible[list(IRIS_FROM[side])].sum() >= 3
    return visible


def select_pose_locked(points, yaw):
    """``(chosen, dropped, fit)``: the dense set checked against the rigid fit (SS3.3).

    Hidden points are dropped. A visible point keeps its detection when that is
    consistent (inside its limit and near the fitted model), otherwise it is
    written where the model puts it, ``source: "model"``.
    """
    fit = fit_rigid(points, yaw)
    projected = project(fit)
    visible = visible_points(fit)
    iod = float(np.linalg.norm(projected[468] - projected[473]))
    tolerance = max(3.0, FIT_TOLERANCE * iod)
    entries = set_entries("pose-locked")
    weights, _cells = anchor_weights(entries)
    chosen, dropped = [], []
    for (name, index), weight in zip(entries, weights):
        if not visible[index]:
            dropped.append(name)
            continue
        detected = points[index][:2] if index < len(points) else None
        consistent = (
            detected is not None
            and not set_hidden(name, index, yaw)
            and np.linalg.norm(np.asarray(detected) - projected[index]) <= tolerance
        )
        if consistent:
            chosen.append(_point(name, detected, weight, "mesh"))
        else:
            chosen.append(_point(name, projected[index], weight, "model"))
    return chosen, dropped, fit


def pose_report(fit):
    yaw, pitch, roll = pose_angles(fit["R"])
    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "roll": round(roll, 1),
        "method": "rigid-fit",
        "residual_px": round(fit["residual_px"], 2),
    }


def with_silhouette(chosen, profile):
    """Mesh points plus silhouette points; a silhouette name replaces the mesh one."""
    names = {point["name"] for point in profile}
    return [point for point in chosen if point["name"] not in names] + profile


# -- polylines: glasses and hairline (SS7) -----------------------------------------


def polyline_json(name, xy, closed, weight, source):
    return {
        "name": name,
        "closed": bool(closed),
        "weight": float(weight),
        "source": source,
        "xy": [[round(float(x), 2), round(float(y), 2)] for x, y in xy],
    }


def load_glasses(path, size):
    """Polylines from an ``--include-glasses`` file, validated, in canvas pixels."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"--include-glasses {path} is not readable JSON ({exc})") from None
    entries = data.get("polylines") if isinstance(data, dict) else data
    lines = parse_polylines(entries, where=f"--include-glasses {path}")
    if not lines:
        raise ValueError(f"--include-glasses {path} holds no polylines")
    width, height = size
    for line in lines:
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in line["xy"]):
            raise ValueError(
                f"--include-glasses {path}: {line['name']} leaves the {width}x{height} canvas "
                "(coordinates are canvas pixels of the run's input.png)"
            )
    return [polyline_json(l["name"], l["xy"], l["closed"], l["weight"], "manual") for l in lines]


def rgb_to_lab(rgb):
    """CIE Lab (D65) of an sRGB uint8 array (..., 3)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1
    )


def skin_model(lab, points, mask=None):
    """``(mean Lab, threshold)`` from discs around the forehead and cheek points."""
    height, width = lab.shape[:2]
    eyes = np.linalg.norm(points[33, :2] - points[263, :2])
    radius = max(2.0, 0.08 * eyes)
    ys, xs = np.mgrid[0:height, 0:width]
    near = np.zeros((height, width), dtype=bool)
    for index in SKIN_SAMPLES:
        x, y = points[index][:2]
        near |= (xs - x) ** 2 + (ys - y) ** 2 <= radius ** 2
    if mask is not None and np.any(near & mask):
        near &= mask
    samples = lab[near]
    mean = samples.mean(axis=0)
    spread = float(np.sqrt(((samples - mean) ** 2).sum(axis=1).mean()))
    return mean, max(12.0, 3.0 * spread)


def march(start, direction, length, wanted, inside):
    """First of 3 consecutive pixels along the ray where ``wanted(x, y)``, or None.

    ``wanted`` and ``inside`` take integer pixels; leaving the subject (or the
    canvas) ends the march with nothing found.
    """
    run = 0
    first = None
    for step in range(int(math.ceil(length)) + 1):
        x = int(round(start[0] + direction[0] * step))
        y = int(round(start[1] + direction[1] * step))
        if not inside(x, y):
            return None
        if wanted(x, y):
            run += 1
            first = (x, y) if run == 1 else first
            if run == 3:
                return first
        else:
            run = 0
    return None


def hairline_polyline(rgb, points, mask=None, hair_mask=None):
    """The hairline as an open polyline dict, or None when fewer than 5 samples hit (SS7.2)."""
    height, width = rgb.shape[:2]
    points = np.asarray(points, dtype=np.float64)
    centre = points[:468, :2].mean(axis=0)
    brows = points[list(BROWS), :2]

    def inside(x, y):
        if not (0 <= x < width and 0 <= y < height):
            return False
        return mask is None or bool(mask[y, x])

    if hair_mask is not None:
        def is_hair(x, y):
            return bool(hair_mask[y, x])
    else:
        lab = rgb_to_lab(rgb)
        mean, threshold = skin_model(lab, points, mask)

        def is_hair(x, y):
            return float(np.linalg.norm(lab[y, x] - mean)) > threshold

    hits = []
    for index in HAIRLINE_OUTLINE:
        outline = points[index, :2]
        brow = brows[np.argmin(np.linalg.norm(brows - outline, axis=1))]
        direction = outline - centre
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm
        reach = float(np.linalg.norm(outline - brow))
        x, y = int(round(outline[0])), int(round(outline[1]))
        if not inside(x, y):
            continue
        if is_hair(x, y):
            # A fringe or low hairline covers the outline point: walk back down
            # toward the brow, at most halfway, to where the skin starts.
            hit = march(outline, -direction, reach / 2, lambda x, y: not is_hair(x, y), inside)
        else:
            hit = march(outline, direction, reach, is_hair, inside)
        if hit is not None:
            hits.append(hit)
    if len(hits) < 5:
        return None
    hits = np.array(hits, dtype=np.float64)
    smooth = hits.copy()
    for k in range(1, len(hits) - 1):
        smooth[k] = np.median(hits[k - 1 : k + 2], axis=0)
    return polyline_json("hairline", smooth, False, HAIRLINE_WEIGHT, "hairline")


# -- MediaPipe ---------------------------------------------------------------


def _import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError:
        print(
            "MediaPipe is not installed in this interpreter; run "
            "`pip install mediapipe==0.10.21` in the sldgen env.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NO_MEDIAPIPE)
    return mp


def detect(rgb):
    """Face Mesh points in pixels with depth, ``(N, 3)``, or None when no face is found."""
    mp = _import_mediapipe()
    points = _mesh(mp, rgb)
    if points is not None:
        return points
    # Face Mesh's own detector is short-range: on a full-figure canvas the face
    # is ~30 px wide and it finds nothing. The full-range detector does find
    # it; mesh an upscaled crop around it and map the points back.
    box = _find_face(mp, rgb)
    if box is None:
        return None
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * 2.0  # generous margin: the mesh wants context
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(max(0, cx - side / 2))
    top = int(max(0, cy - side / 2))
    right = int(min(width, cx + side / 2))
    bottom = int(min(height, cy + side / 2))
    crop = Image.fromarray(rgb[top:bottom, left:right])
    scale = CROP_SIZE / max(crop.width, crop.height)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    points = _mesh(mp, np.asarray(crop))
    if points is None:
        return None
    return region_to_canvas(points, left, top, scale)


def _mesh(mp, rgb):
    height, width = rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mesh:
        result = mesh.process(np.ascontiguousarray(rgb))
    if not result.multi_face_landmarks:
        return None
    points = result.multi_face_landmarks[0].landmark
    # z is in the same normalised units as x (MediaPipe), so scale it by width.
    return np.array([[p.x * width, p.y * height, p.z * width] for p in points], dtype=np.float64)


def _find_face(mp, rgb):
    """Pixel box ``(x0, y0, x1, y1)`` of the most confident face, full-range model."""
    height, width = rgb.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3
    ) as detector:
        result = detector.process(rgb)
    if not result.detections:
        return None
    best = max(result.detections, key=lambda d: d.score[0])
    box = best.location_data.relative_bounding_box
    return (
        box.xmin * width,
        box.ymin * height,
        (box.xmin + box.width) * width,
        (box.ymin + box.height) * height,
    )


def detect_pose(rgb):
    """Pose's head keypoints in pixels, ``{name: (x, y)}``, or None."""
    mp = _import_mediapipe()
    height, width = rgb.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        result = pose.process(np.ascontiguousarray(rgb))
    if not result.pose_landmarks:
        return None
    marks = result.pose_landmarks.landmark
    return {name: (marks[i].x * width, marks[i].y * height) for name, i in POSE.items()}


def _region(rgb, box):
    """``(region_rgb, left, top, scale)``: the box (or the canvas), upscaled if small."""
    height, width = rgb.shape[:2]
    left, top, right, bottom = (0, 0, width, height) if box is None else grow_box(box, width, height)
    crop = Image.fromarray(rgb[top:bottom, left:right])
    scale = max(1.0, REGION_SIZE / max(crop.width, crop.height))
    if scale > 1.0:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    return np.asarray(crop), left, top, scale


def landmarks_for(
    rgb,
    mask=None,
    box=None,
    preset="portrait",
    landmark_set="sparse",
    want_pose=False,
    hairline=False,
    hair_mask=None,
):
    """The whole pipeline (Spec 7 SS4 and its addendum), as a dict, or None (no face).

    Keys: ``landmarks``, ``view``, ``dropped``, ``points`` (the mesh or None),
    ``landmark_set`` (what was built: the Pose path is always sparse),
    ``pose`` (with ``want_pose``), ``polylines`` and ``notes`` (lines to print).
    """
    region, left, top, scale = _region(rgb, box)
    points = detect(region)
    result = {"pose": None, "polylines": [], "notes": [], "landmark_set": landmark_set}
    if points is not None:
        points = region_to_canvas(points, left, top, scale)
        result["points"] = points
        if preset == "all":
            chosen = [_point(f"p{i}", p, 1.0, "mesh") for i, p in enumerate(points)]
            view = {"kind": None, "yaw": None, "method": "mesh", "facing": None}
            return {**result, "landmarks": chosen, "view": view, "dropped": []}
        yaw = mesh_yaw(points)
        fit = None
        if landmark_set == "pose-locked":
            chosen, dropped, fit = select_pose_locked(points, yaw)
        else:
            chosen, dropped = select_set(points, yaw, landmark_set)
        if want_pose:
            result["pose"] = pose_report(fit or fit_rigid(points, yaw))
        facing = "left" if yaw < 0 else "right"
        view = {"kind": view_kind(yaw), "yaw": round(yaw, 1), "method": "mesh", "facing": facing}
        if mask is not None and abs(yaw) >= PROFILE_YAW:
            near = "left" if facing == "left" else "right"
            eye = points[list(EYE_CORNERS[near])][:, :2].mean(axis=0)
            mouth = points[MOUTH_CORNER[near]][:2]
            profile = profile_landmarks(mask, facing, eye, mouth, nose_hint=points[NOSE_TIP])
            chosen = with_silhouette(chosen, profile)
        if hairline:
            line = None
            if abs(yaw) < PROFILE_YAW:
                line = hairline_polyline(rgb, points, mask=mask, hair_mask=hair_mask)
            if line is None:
                result["notes"].append("hairline not found (needs a frontal or turned face and a visible hairline)")
            else:
                result["polylines"].append(line)
        return {**result, "landmarks": chosen, "view": view, "dropped": dropped}

    if preset == "all":
        return None
    pose = detect_pose(region)
    if pose is None:
        return None
    pose = {
        name: tuple(region_to_canvas([xy], left, top, scale)[0]) for name, xy in pose.items()
    }
    facing, near, yaw = pose_view(pose)
    chosen, dropped = pose_landmarks(pose, yaw)
    view = {"kind": view_kind(yaw), "yaw": round(yaw, 1), "method": "pose", "facing": facing}
    if mask is not None and abs(yaw) >= FRONTAL_YAW:
        eye = pose[f"{near}_eye"]
        mouth = pose[f"mouth_{near}"]
        chosen += profile_landmarks(mask, facing, eye, mouth)
    if landmark_set != "sparse":
        result["notes"].append(
            f"{landmark_set} needs Face Mesh; it found nothing, so these are the Pose + outline points"
        )
    if hairline:
        result["notes"].append("hairline not found (needs Face Mesh)")
    if want_pose:
        result["pose"] = {"yaw": round(yaw, 1), "pitch": None, "roll": None, "method": "pose"}
    return {
        **result,
        "landmark_set": "sparse",
        "points": None,
        "landmarks": chosen,
        "view": view,
        "dropped": dropped,
    }


def load_mask(path, size, flag="--mask"):
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise ValueError(f"{flag} is {mask.size[0]}x{mask.size[1]}, the image {size[0]}x{size[1]}")
    return np.asarray(mask) > 127


def main(argv=None):
    args = parse_arguments(argv)
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    try:
        mask = load_mask(args.mask, image.size) if args.mask else None
        hair_mask = load_mask(args.hair_mask, image.size, "--hair-mask") if args.hair_mask else None
        glasses = load_glasses(args.include_glasses, image.size) if args.include_glasses else []
        result = landmarks_for(
            rgb,
            mask=mask,
            box=args.box,
            preset=args.preset,
            landmark_set=args.landmark_set,
            want_pose=args.pose_report,
            hairline=args.include_hairline,
            hair_mask=hair_mask,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NO_FACE
    if result is None or not result["landmarks"]:
        where = " inside the box" if args.box else ""
        print(f"No face found in {args.image}{where}.", file=sys.stderr)
        return EXIT_NO_FACE

    landmarks, view, dropped, points = (
        result["landmarks"], result["view"], result["dropped"], result["points"]
    )
    polylines = glasses + result["polylines"]
    # Key order and content for the sparse set without extras are exactly the
    # pre-addendum --preset portrait file (Spec 7 addendum SS2, byte identity).
    preset = args.landmark_set if args.landmark_set != "sparse" else args.preset
    payload = {"space": "canvas", "image_size": [image.width, image.height], "preset": preset}
    if args.landmark_set != "sparse":
        payload["landmark_set"] = result["landmark_set"]
    payload["view"] = view
    if result["pose"] is not None:
        payload["pose"] = result["pose"]
    payload["dropped"] = dropped
    payload["landmarks"] = landmarks
    if polylines:
        payload["polylines"] = polylines
    payload["all_landmarks"] = (
        []
        if points is None
        else [[round(float(x), 2), round(float(y), 2)] for x, y in points[:, :2]]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    yaw = "?" if view["yaw"] is None else f"{view['yaw']:+.0f}"
    print(f"canvas   {image.width}x{image.height} px")
    print(f"view     {view['kind']}  yaw {yaw}  method {view['method']}  facing {view['facing']}")
    pose = result["pose"]
    if pose is not None and pose["method"] == "rigid-fit":
        print(
            f"pose     yaw {pose['yaw']:+.1f}  pitch {pose['pitch']:+.1f}  roll {pose['roll']:+.1f}"
            f"  (rigid fit, residual {pose['residual_px']:.1f} px)"
        )
    elif pose is not None:
        print(f"pose     yaw {pose['yaw']:+.1f} (coarse, from body pose)")
    label = args.preset if args.landmark_set == "sparse" else result["landmark_set"]
    print(f"face     {len(landmarks)} landmarks ({label})")
    modelled = sum(point["source"] == "model" for point in landmarks)
    if modelled:
        print(f"model    {modelled} points placed by the rigid fit")
    for line in polylines:
        print(f"line     {line['name']} ({len(line['xy'])} vertices, weight {line['weight']:g})")
    for note in result["notes"]:
        print(f"note     {note}")
    if dropped:
        print(f"dropped  {' '.join(dropped)}")
    print(f"wrote    {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
