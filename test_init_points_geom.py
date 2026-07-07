"""Fast isolated test for the --init-points feature.

Seeds the TSP initializer from a provided SVG's points instead of stippling
the target image. No diffusion model, CPU only (needs Concorde for the tour).
It exercises:

  * load_init_points: canvas-pixel sampling reused from the avoidance loader,
    zoom-frame scaling, uniform subsample to --n-control-points, and the
    "fewer points than requested -> use as-is" branch
  * the painter initializing its control points ON the provided SVG's geometry
    (a circle) rather than from the firefighter target -- i.e. init really came
    from the SVG
  * the strict opt-in guarantee (no --init-points -> arg is None, normal init)
  * composition with --origin (origin pinned at index 0, rest on the SVG)

Run from the repo root:
    PYTHONPATH=. python test_init_points_geom.py
"""
import math
import sys

import numpy as np
import torch

from SLDgen import config
from SLDgen.painter.painter import SLDBSplinePainter
from SLDgen.painter.tsp_art import load_init_points, _zoom_factor

SCRATCH = "/tmp/claude-1000/-home-helge-SLDgen"


def write_circle_svg(path, size, cx, cy, r, n=256):
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
           for i in range(n + 1)]
    d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}"><path d="{d}" fill="none" stroke="black"/></svg>'
        )


def make_mask(render_size):
    m = torch.zeros((render_size, render_size), dtype=torch.float32)
    m[: render_size // 2, render_size // 4 : 3 * render_size // 4] = 1.0
    return m


def build_renderer(extra_args):
    args = config.parse_arguments(
        [
            "--target", "./data/firefighter.png",
            "--use-cpu",
            "--seed", "0",
            "--render-size", "512",
            "--experiment-name", "initpointstest",
            "--init-method", "tsp",
        ]
        + extra_args
    )
    mask = make_mask(512)
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer.init_image()
    return renderer


def check(name, cond):
    print(f"[{name}] {'PASS' if cond else 'FAIL'}")
    return cond


def dist_from_center(cp, cx=256.0, cy=256.0):
    cp = cp.detach().cpu().numpy()
    return np.sqrt((cp[:, 0] - cx) ** 2 + (cp[:, 1] - cy) ** 2)


def main():
    ok = True
    circle = f"{SCRATCH}/circle_init.svg"
    write_circle_svg(circle, 512, 256, 256, 100)
    density = np.zeros((512, 512), dtype=np.float32)

    # 1. Zoom factor for the standard canvas/point count is 1 (sanity).
    ok &= check("zoom-factor-1", _zoom_factor(density, 385) == 1)

    # 2. load_init_points: dense circle (~628px) subsampled uniformly to n_point,
    #    returned in the zoomed frame (== canvas pixels here since zoom == 1), all
    #    lying on the ring (dist ~ 100 from center).
    seed = load_init_points(circle, density, n_point=385)
    ok &= check("subsample-count", len(seed) == 385)
    d = np.sqrt((seed[:, 0] - 256) ** 2 + (seed[:, 1] - 256) ** 2)
    ok &= check("seed-on-ring", abs(d.mean() - 100) < 2.0 and d.std() < 2.0)

    # 3. "fewer than requested" branch: asking for more points than the SVG
    #    yields uses them all as-is.
    seed_few = load_init_points(circle, density, n_point=5000)
    ok &= check("fewer-as-is", 0 < len(seed_few) < 5000)

    # 4. Opt-in guarantee: no --init-points -> arg is None (normal stipple init).
    r_plain = build_renderer([])
    ok &= check("optin-none", getattr(r_plain.args, "init_points", "x") is None)

    # 5. --init-points: the painter's control points sit ON the circle, proving
    #    initialization came from the SVG (not from stippling the firefighter).
    r_ip = build_renderer(["--init-points", circle])
    di = dist_from_center(r_ip.control_points)
    ok &= check("init-on-svg",
                abs(di.mean() - 100) < 4.0 and di.std() < 4.0)
    ok &= check("init-count", abs(len(r_ip.control_points) - 385) <= 2)

    # 6. Compose with --origin: origin pinned at canvas (256, 256) = 0.5*512, and
    #    the remaining optimizable control points still lie on the circle.
    r_both = build_renderer(["--init-points", circle, "--origin", "0.5", "0.5"])
    ok &= check("compose-origin-present", hasattr(r_both, "first_origin_points"))
    op = r_both.first_origin_points.detach().cpu().numpy()[0]
    ok &= check("origin-at-center", abs(op[0] - 256) < 1.0 and abs(op[1] - 256) < 1.0)
    db = dist_from_center(r_both.control_points)
    ok &= check("origin-rest-on-svg", abs(db.mean() - 100) < 4.0 and db.std() < 4.0)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
