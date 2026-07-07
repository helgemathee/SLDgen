"""Fast isolated test for the --avoid feature.

No diffusion model, CPU only -> runs in a few seconds (only needs Concorde for
the TSP init). It exercises:

  * load_avoid_points sampling (count, coordinate frame, arc-length spacing)
  * avoidance_loss math + gradient direction (repels toward larger separation)
  * the painter picking up self.avoid_points and active_control_points
  * the strict opt-in guarantee (no --avoid  ->  avoid_points is None)
  * composition with --origin (both features active at once)
  * a real backward() through the painter's active control points

Run from the repo root:
    PYTHONPATH=. python test_avoidance_geom.py
"""
import math
import sys

import torch

from SLDgen import config
from SLDgen.avoidance import avoidance_loss, load_avoid_points
from SLDgen.painter.painter import SLDBSplinePainter

SCRATCH = "/tmp/claude-1000/-home-helge-SLDgen"


def write_circle_svg(path, size, cx, cy, r, n=128):
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
            "--experiment-name", "avoidtest",
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


def main():
    ok = True
    circle = f"{SCRATCH}/circle.svg"
    circle2 = f"{SCRATCH}/circle_b.svg"
    circle1024 = f"{SCRATCH}/circle1024.svg"
    write_circle_svg(circle, 512, 256, 256, 100)
    write_circle_svg(circle2, 512, 128, 400, 60)
    write_circle_svg(circle1024, 1024, 512, 512, 200)

    # 1. Sampling: count ~ circumference / spacing, all within the SVG bbox.
    pts = load_avoid_points([circle], sample_spacing_px=2.0, render_size=512)
    expected = round(2 * math.pi * 100 / 2.0)
    ok &= check("sample-count", abs(len(pts) - expected) <= 2)
    ok &= check("sample-frame", 155 <= pts[:, 0].min() and pts[:, 0].max() <= 357)

    # 2. Multiple SVGs -> union of both point sets.
    pts_multi = load_avoid_points([circle, circle2], sample_spacing_px=2.0)
    expected_multi = expected + round(2 * math.pi * 60 / 2.0)
    ok &= check("multi-svg-union", abs(len(pts_multi) - expected_multi) <= 4)

    # 3. avoidance_loss gradient: a point 10px inside the ring is repelled so that
    #    gradient descent (-grad) increases its distance from the nearest obstacle.
    av = torch.tensor(pts)
    p = torch.tensor([[346.0, 256.0]], requires_grad=True)  # nearest ring pt (356,256), dist 10
    loss = avoidance_loss(p, av, d0=25.0)
    loss.backward()
    descent = -p.grad[0]
    ok &= check("loss-value", abs(loss.item() - (25 - 10) ** 2) < 1e-3)
    ok &= check("repel-direction", descent[0].item() < 0)  # push toward -x, away from ring at +x

    # 4. Opt-in guarantee: no --avoid -> renderer.avoid_points is None.
    r_plain = build_renderer([])
    ok &= check("optin-none", r_plain.avoid_points is None)

    # 5. --avoid loads points onto the painter as a no-grad tensor.
    r_av = build_renderer(["--avoid", circle])
    has = r_av.avoid_points is not None
    ok &= check("avoid-loaded", has and not r_av.avoid_points.requires_grad)
    ok &= check("avoid-frame", has and float(r_av.avoid_points.max()) <= 512.0)

    # 6. active_control_points carries grad and a full step backprops to control_points.
    r_av.parameters()  # enables requires_grad on control_points
    acp = r_av.active_control_points
    l = avoidance_loss(acp, r_av.avoid_points, d0=r_av.args.avoidance_distance)
    l.backward()
    ok &= check("backprop-to-cp", r_av.control_points.grad is not None
                and torch.isfinite(r_av.control_points.grad).all().item())

    # 7. Compose with --origin: both features active; origin tensor present AND
    #    avoid points loaded; avoidance loss still differentiable.
    r_both = build_renderer(["--avoid", circle, "--origin", "0.5", "0.5"])
    ok &= check("compose-origin",
                r_both.avoid_points is not None and hasattr(r_both, "first_origin_points"))
    r_both.parameters()
    lb = avoidance_loss(r_both.active_control_points, r_both.avoid_points, d0=25.0)
    lb.backward()
    ok &= check("compose-origin-backprop",
                r_both.control_points.grad is not None
                and torch.isfinite(r_both.control_points.grad).all().item())

    # 8. viewBox/size mismatch emits a warning (1024 SVG with render_size 512).
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_avoid_points([circle1024], render_size=512)
        ok &= check("size-mismatch-warn", len(w) >= 1)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
