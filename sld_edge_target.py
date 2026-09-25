#!/usr/bin/env python
"""Prepare an ``--image-loss-target`` edge map in canvas space (Spec 6 SS8.1).

Inside a run, ``--image-loss`` derives its edge map from the canvas image on its
own and needs no file. With only ``--low/--high/--blur`` this script produces
exactly that map, which is what the web UI's preview relies on. The extras are
what the in-run derivation cannot do:

* ``--clahe-clip``: lift shadow detail before Canny (dark curly hair becomes
  edges instead of a silhouetted blob);
* ``--roi``: keep only a canvas-pixel box ("the face, not the shoulders");
* ``--preserve-silhouette``: put the subject's outline back when CLAHE erased it.

It takes canvas-space input only -- a run's ``input.png`` and ``mask.png``.
Pointing it at the original photograph produces a target in the wrong frame,
silently.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from SLDgen.image_loss import EDGE_PIXELS_SANE, prepare_edges  # noqa: E402


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a 0/255 canvas-space edge map for --image-loss-target.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python sld_edge_target.py --image work/jobs/<id>/target/run/input.png \\\n"
            "      --mask work/jobs/<id>/target/run/mask.png --out edges.png --clahe-clip 2\n"
            "  python sldgen.py --target photo.png --image-loss --image-loss-target edges.png\n"
        ),
    )
    parser.add_argument(
        "--image", required=True, help="Canvas-space image: a run's input.png, never the photo."
    )
    parser.add_argument("--out", required=True, help="PNG to write, same size as --image.")
    parser.add_argument(
        "--mask",
        default=None,
        help="The run's mask.png. Without it the square-pad border counts as an edge.",
    )
    parser.add_argument("--low", type=float, default=100.0, help="Canny low threshold.")
    parser.add_argument("--high", type=float, default=200.0, help="Canny high threshold.")
    parser.add_argument(
        "--blur", type=int, default=3, help="Gaussian kernel before Canny, odd, 0 disables."
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=0.0,
        help="CLAHE clip limit before Canny (0 disables; 2-4 lifts shadow detail).",
    )
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Canvas-pixel box; edges outside it are dropped.",
    )
    parser.add_argument(
        "--preserve-silhouette",
        action="store_true",
        help="Union the mask's outline back in (needs --mask).",
    )
    args = parser.parse_args(argv)
    if args.low >= args.high:
        parser.error(f"--low must be below --high; got {args.low} and {args.high}.")
    if args.blur < 0:
        parser.error("--blur must be 0 (disabled) or positive.")
    if args.clahe_clip < 0 or args.clahe_grid < 1:
        parser.error("--clahe-clip must be >= 0 and --clahe-grid >= 1.")
    if args.preserve_silhouette and args.mask is None:
        parser.error("--preserve-silhouette needs --mask.")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L") if args.mask else None
    edges = prepare_edges(
        image,
        mask=mask,
        low=args.low,
        high=args.high,
        blur=args.blur,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        roi=args.roi,
        preserve_silhouette=args.preserve_silhouette,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(edges.astype(np.uint8)).save(args.out)

    count = int(np.count_nonzero(edges))
    lo, hi = EDGE_PIXELS_SANE
    print(f"canvas   {edges.shape[1]}x{edges.shape[0]} px")
    print(f"edges    {count} edge px")
    if not lo <= count <= hi:
        print(f"warning  {count} edge pixels is outside {lo}-{hi}; check the thresholds")
    print(f"wrote    {args.out}")


if __name__ == "__main__":
    main()
