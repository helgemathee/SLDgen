#!/usr/bin/env python
"""Standalone CLI for ``SLDgen/canny_attract.py``.

Inside a run, ``--attract-canny`` does all of this automatically and needs no
files: the edge map is derived from ``args.input_image`` the moment
``get_target`` has produced it, which is the only place canvas space exists
without reproducing it. **Prefer the flag.**

This script is for the cases the flag cannot serve:

* previewing the parameters against a finished run before committing GPU time
  (which is what the web UI's Canny panel drives, via ``sldgen_api/canny.py``);
* ``--roi``, which the flag does not expose;
* feeding the edges of one image to a run on a different one.

It takes canvas-space input only -- a run's ``input.png`` and ``mask.png``, or
its ``condition_canny.png`` with ``--already-edges``. Pointing it at the
original photograph produces points in the wrong frame, silently.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from SLDgen.canny_attract import DEFAULTS, describe, generate  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a Canny edge map into an --attract SVG in canvas space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python sld_canny_svg.py --image work/jobs/<id>/run/input.png \\\n"
            "      --mask work/jobs/<id>/run/mask.png --out attract.svg\n"
            "  python sldgen.py --target photo.png --attract attract.svg \\\n"
            "      --attraction-distance 10 --attraction-weight 0.004\n"
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Canvas-space image: a run's input.png, or condition_canny.png with --already-edges.",
    )
    parser.add_argument("--out", required=True, help="SVG to write, in canvas pixel coordinates.")
    parser.add_argument(
        "--mask",
        default=None,
        help=(
            "Optional run mask.png. Strongly recommended: without it the "
            "square-pad border is the strongest edge in the picture."
        ),
    )
    parser.add_argument(
        "--already-edges",
        action="store_true",
        help="Treat --image as a finished edge map instead of running Canny over it.",
    )
    parser.add_argument("--low", type=float, default=DEFAULTS["low"], help="Canny low threshold.")
    parser.add_argument(
        "--high", type=float, default=DEFAULTS["high"], help="Canny high threshold."
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=DEFAULTS["blur"],
        help="Gaussian kernel before Canny, odd, 0 to disable. Stops hair texture dominating.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Canvas-pixel box; edges outside it are dropped. 'The face, not the shoulders'.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULTS["simplify"],
        help="Douglas-Peucker tolerance in pixels (0 disables).",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=DEFAULTS["min_length"],
        help="Drop contours shorter than this many pixels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULTS["max_points"],
        help="Attract-point budget. Keep at or below --n-control-points.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    image = Image.open(args.image).convert("L")
    if args.already_edges:
        # Skip Canny by handing the module a pre-thresholded map and no blur:
        # a 0/255 image is its own edge map.
        image = Image.fromarray(((np.asarray(image) > 127) * 255).astype(np.uint8))
        args.blur, args.low, args.high = 0, 1, 2

    mask = Image.open(args.mask).convert("L") if args.mask else None
    stats = generate(
        image,
        args.out,
        mask=mask,
        low=args.low,
        high=args.high,
        blur=args.blur,
        roi=args.roi,
        simplify=args.simplify,
        min_length=args.min_length,
        max_points=args.max_points,
    )

    print(f"canvas   {stats['render_size']}x{stats['render_size']} px")
    print(f"edges    {describe(stats)}")
    print(f"wrote    {stats['path']}")
    print()
    print("Use it on a run with the SAME target, --render-size and --object-size-ratio:")
    print(f"  --attract {args.out} --attraction-distance 10 --attraction-weight 0.004")


if __name__ == "__main__":
    main()
