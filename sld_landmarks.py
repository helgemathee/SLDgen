#!/usr/bin/env python
"""Extract face landmarks for ``--image-loss-landmarks`` (Spec 6 SS8.2).

Identity in a face sits in perhaps twenty points; a generic image loss gives an
eye corner and a hoodie fold the same pressure. This detects the face with
MediaPipe Face Mesh and writes the identity-carrying points, weighted, as a
canvas-space JSON the landmark term reads.

It takes canvas-space input only -- a run's ``input.png``. Pointing it at the
original photograph produces landmarks in the wrong frame, silently. The run
refuses a file whose ``image_size`` is not ``--render-size``, but it cannot
detect one taken from the wrong image of the right size.

Needs MediaPipe in this interpreter: ``pip install mediapipe==0.10.21`` (the
0.10 line still bundles the Face Mesh model and works with numpy 1.x).
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

EXIT_NO_FACE = 2
EXIT_NO_MEDIAPIPE = 3


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Write canvas-space face landmarks for --image-loss-landmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python sld_landmarks.py --image work/jobs/<id>/target/run/input.png \\\n"
            "      --out landmarks.json\n"
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
        help="portrait: ~20 weighted identity points. all: every mesh point at weight 1.",
    )
    return parser.parse_args(argv)


def detect(rgb):
    """Face Mesh points in pixels, ``(N, 2)``, or None when no face is found."""
    try:
        import mediapipe as mp
    except ImportError:
        print(
            "MediaPipe is not installed in this interpreter; run "
            "`pip install mediapipe==0.10.21` in the sldgen env.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NO_MEDIAPIPE)

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
    return points / scale + np.array([left, top], dtype=np.float64)


#: Side of the upscaled face crop the mesh runs on when the face is small.
CROP_SIZE = 384


def _mesh(mp, rgb):
    height, width = rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mesh:
        result = mesh.process(np.ascontiguousarray(rgb))
    if not result.multi_face_landmarks:
        return None
    points = result.multi_face_landmarks[0].landmark
    return np.array([[p.x * width, p.y * height] for p in points], dtype=np.float64)


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


def select(points, preset):
    if preset == "all":
        return [
            {"name": f"p{i}", "xy": [round(float(x), 2), round(float(y), 2)], "weight": 1.0}
            for i, (x, y) in enumerate(points)
        ]
    chosen = []
    for name, index, weight in PORTRAIT:
        if index < len(points):  # the pupils need the refined (478-point) mesh
            x, y = points[index]
            chosen.append({"name": name, "xy": [round(float(x), 2), round(float(y), 2)], "weight": weight})
    return chosen


def main(argv=None):
    args = parse_arguments(argv)
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    points = detect(rgb)
    if points is None:
        print(f"No face found in {args.image}.", file=sys.stderr)
        return EXIT_NO_FACE

    landmarks = select(points, args.preset)
    payload = {
        "space": "canvas",
        "image_size": [image.width, image.height],
        "preset": args.preset,
        "landmarks": landmarks,
        "all_landmarks": [[round(float(x), 2), round(float(y), 2)] for x, y in points],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"canvas   {image.width}x{image.height} px")
    print(f"face     {len(points)} mesh points -> {len(landmarks)} landmarks ({args.preset})")
    print(f"wrote    {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
