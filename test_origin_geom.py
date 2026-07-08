"""Fast isolated geometry test for the --origin feature.

No diffusion model, CPU only -> runs in a few seconds (only needs Concorde).
It builds the renderer with a synthetic mask, runs init_image (real TSP init +
Concorde + B-spline clamping), saves an SVG, and checks the first path command
(M) lands exactly at the requested origin in canvas pixels.

Run from the repo root:
    PYTHONPATH=. python test_origin_geom.py
"""
import re
import sys

import numpy as np
import torch

from SLDgen import config
from SLDgen.painter.painter import SLDBSplinePainter
from SLDgen.painter.tsp_art import init_tsp_art


def make_mask(render_size):
    # Object occupies the TOP HALF only -> origin 0.5,0.5 sits at the low-density
    # edge, exercising the forced origin injection into the TSP tour.
    m = torch.zeros((render_size, render_size), dtype=torch.float32)
    m[: render_size // 2, render_size // 4 : 3 * render_size // 4] = 1.0
    return m


def first_M(svg_path):
    with open(svg_path) as f:
        txt = f.read()
    m = re.search(r"[Mm]\s*([-\d.eE]+)[ ,]+([-\d.eE]+)", txt)
    assert m, f"no M command found in {svg_path}"
    return float(m.group(1)), float(m.group(2))


def run_case(origin, render_size=512, seed=0):
    args = config.parse_arguments(
        [
            "--target", "./data/firefighter.png",
            "--use-cpu",
            "--seed", str(seed),
            "--render-size", str(render_size),
            "--experiment-name", f"geomtest_{origin}",
            "--init-method", "tsp",
        ]
        + ([] if origin is None else ["--origin", str(origin[0]), str(origin[1])])
    )
    mask = make_mask(render_size)
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer.init_image()
    renderer.save_svg(str(args.output_dir), "geomcheck")
    mx, my = first_M(f"{args.output_dir}/geomcheck.svg")
    return (mx, my), hasattr(renderer, "first_origin_points")


def test_nearest_start(origin=(0.5, 0.0), render_size=512, n_point=200):
    """Acceptance test for nearest-start: with --origin, init_tsp_art must return
    the origin at index 0 and the stipple point NEAREST the origin at index 1.

    Calls init_tsp_art directly (real stipple + Concorde). ordered_points[1:] IS
    the full stipple point set (just reordered by the tour), so checking that
    ordered_points[1] minimizes distance-to-origin over ordered_points[1:] is a
    complete, self-contained correctness check independent of the stipple layout.
    """
    density = make_mask(render_size).numpy()
    ordered = init_tsp_art(
        density, n_point=n_point, n_iter=25, reverse=True,
        output_dir=None, debug=False, fixed_endpoints=False,
        origin=origin, init_points=None, verbose=True,
    )
    ordered = np.asarray(ordered, dtype=float)

    origin_px = np.array([origin[0] * render_size, origin[1] * render_size])
    # Index 0 is the origin itself (unzoomed pixel space).
    d0 = np.linalg.norm(ordered[0] - origin_px)

    real = ordered[1:]
    dists = np.linalg.norm(real - origin_px, axis=1)
    first_real_d = dists[0]
    min_d = dists.min()
    argmin = int(dists.argmin())

    passed = d0 < 0.5 and argmin == 0 and abs(first_real_d - min_d) < 1e-6
    print(
        f"[nearest-start] origin={origin} origin@idx0 dist={d0:.3f} | "
        f"first-real dist={first_real_d:.3f} min-over-all dist={min_d:.3f} "
        f"argmin={argmin} (expect 0) : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def main():
    ok = True
    ok = test_nearest_start() and ok
    for origin, name in [((0.5, 0.5), "b"), ((0.2, 0.8), "c")]:
        (mx, my), has_t = run_case(origin)
        want = (origin[0] * 512, origin[1] * 512)
        dx, dy = abs(mx - want[0]), abs(my - want[1])
        passed = has_t and dx < 0.5 and dy < 0.5
        ok = ok and passed
        print(
            f"[test {name}] origin={origin} -> first M=({mx:.3f},{my:.3f}) "
            f"want=({want[0]:.1f},{want[1]:.1f}) origin_tensor={has_t} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    (mx0, my0), has_t0 = run_case(None)
    passed0 = not has_t0
    ok = ok and passed0
    print(
        f"[no-origin] first M=({mx0:.3f},{my0:.3f}) origin_tensor={has_t0} "
        f"(expect False) : {'PASS' if passed0 else 'FAIL'}"
    )

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
