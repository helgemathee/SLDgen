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
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
}

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
        "--box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Detect only inside this canvas-pixel box (grown by a quarter for context).",
    )
    return parser.parse_args(argv)


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


def landmarks_for(rgb, mask=None, box=None, preset="portrait"):
    """The whole Spec 7 pipeline: ``(landmarks, view, dropped, all_points)`` or None."""
    region, left, top, scale = _region(rgb, box)
    points = detect(region)
    if points is not None:
        points = region_to_canvas(points, left, top, scale)
        if preset == "all":
            chosen = [_point(f"p{i}", p, 1.0, "mesh") for i, p in enumerate(points)]
            return chosen, {"kind": None, "yaw": None, "method": "mesh", "facing": None}, [], points
        yaw = mesh_yaw(points)
        chosen, dropped = select_portrait(points, yaw)
        facing = "left" if yaw < 0 else "right"
        view = {"kind": view_kind(yaw), "yaw": round(yaw, 1), "method": "mesh", "facing": facing}
        if mask is not None and abs(yaw) >= PROFILE_YAW:
            near = "left" if facing == "left" else "right"
            eye = points[list(EYE_CORNERS[near])][:, :2].mean(axis=0)
            mouth = points[MOUTH_CORNER[near]][:2]
            chosen += profile_landmarks(mask, facing, eye, mouth, nose_hint=points[NOSE_TIP])
        return chosen, view, dropped, points

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
    return chosen, view, dropped, None


def load_mask(path, size):
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise ValueError(f"--mask is {mask.size[0]}x{mask.size[1]}, the image {size[0]}x{size[1]}")
    return np.asarray(mask) > 127


def main(argv=None):
    args = parse_arguments(argv)
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    try:
        mask = load_mask(args.mask, image.size) if args.mask else None
        result = landmarks_for(rgb, mask=mask, box=args.box, preset=args.preset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NO_FACE
    if result is None or not result[0]:
        where = " inside the box" if args.box else ""
        print(f"No face found in {args.image}{where}.", file=sys.stderr)
        return EXIT_NO_FACE

    landmarks, view, dropped, points = result
    payload = {
        "space": "canvas",
        "image_size": [image.width, image.height],
        "preset": args.preset,
        "view": view,
        "dropped": dropped,
        "landmarks": landmarks,
        "all_landmarks": []
        if points is None
        else [[round(float(x), 2), round(float(y), 2)] for x, y in points[:, :2]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    yaw = "?" if view["yaw"] is None else f"{view['yaw']:+.0f}"
    print(f"canvas   {image.width}x{image.height} px")
    print(f"view     {view['kind']}  yaw {yaw}  method {view['method']}  facing {view['facing']}")
    print(f"face     {len(landmarks)} landmarks ({args.preset})")
    if dropped:
        print(f"dropped  {' '.join(dropped)}")
    print(f"wrote    {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
