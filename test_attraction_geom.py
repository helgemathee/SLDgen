"""Fast isolated test for the --attract feature.

Structural mirror of test_avoidance_geom.py. No diffusion model, CPU only ->
runs in a few seconds (only needs Concorde for the TSP init). It exercises:

  * load_attract_points reuses the avoidance loader (count, coordinate frame)
  * attraction_loss math: the dead-zone hinge (inactive within the radius),
    the two-sided Chamfer (structure + coverage), and gradient direction
    (descent pulls a far point TOWARD the target)
  * the coverage term firing when the curve collapses onto a subset
  * the dead-zone sanity property (huge deadzone -> loss ~ 0)
  * the painter picking up self.attract_points as a no-grad tensor
  * the strict opt-in guarantee (no --attract -> attract_points is None)
  * composition with --origin and with --avoid (both loss modules coexist)
  * a real backward() through the painter's active control points

Run from the repo root:
    PYTHONPATH=. python test_attraction_geom.py
"""
import math
import sys

import torch

from SLDgen import config
from SLDgen.attraction import attraction_loss, load_attract_points
from SLDgen.avoidance import avoidance_loss
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
            "--experiment-name", "attracttest",
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
    write_circle_svg(circle, 512, 256, 256, 100)
    write_circle_svg(circle2, 512, 128, 400, 60)

    # 1. Loader is the exact avoidance loader (reused): count ~ circumference/spacing.
    pts = load_attract_points([circle], sample_spacing_px=2.0, render_size=512)
    expected = round(2 * math.pi * 100 / 2.0)
    ok &= check("sample-count", abs(len(pts) - expected) <= 2)
    ok &= check("sample-frame", 155 <= pts[:, 0].min() and pts[:, 0].max() <= 357)

    # 2. Dead-zone hinge + two-sided value. One curve point, one target point:
    #    both Chamfer terms reduce to the same pair, so loss = 2*(d-r)^2.
    p = torch.tensor([[400.0, 256.0]], requires_grad=True)
    q = torch.tensor([[356.0, 256.0]])  # d = 44
    loss = attraction_loss(p, q, deadzone=25.0)
    loss.backward()
    ok &= check("two-sided-value", abs(loss.item() - 2 * (44 - 25) ** 2) < 1e-3)
    # descent (-grad) must pull p toward q, i.e. in -x.
    ok &= check("attract-direction", (-p.grad[0])[0].item() < 0)

    # 3. Dead zone: a point INSIDE the radius contributes nothing (loss 0, no grad).
    p_in = torch.tensor([[366.0, 256.0]], requires_grad=True)  # d = 10 < 25
    loss_in = attraction_loss(p_in, q, deadzone=25.0)
    loss_in.backward()
    ok &= check("deadzone-silent",
                loss_in.item() == 0.0 and float(p_in.grad.abs().max()) == 0.0)

    # 4. Coverage term fires when the curve collapses onto a subset of targets.
    #    Two curve points sit exactly on target A (inside deadzone => structure
    #    term 0); target B is left uncovered, so only the attract->curve term
    #    contributes, proving coverage is active. loss = (dist_B - r)^2.
    curve = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0], [300.0, 0.0]])
    loss_cov = attraction_loss(curve, targets, deadzone=25.0)
    ok &= check("coverage-term", abs(loss_cov.item() - (300 - 25) ** 2) < 1e-3)

    # 5. Dead-zone sanity: a huge deadzone makes attraction ~inert everywhere.
    av = torch.tensor(pts)
    p_any = torch.tensor([[256.0, 256.0]], requires_grad=True)
    loss_big = attraction_loss(p_any, av, deadzone=10000.0)
    ok &= check("huge-deadzone-inert", loss_big.item() == 0.0)

    # 6. Opt-in guarantee: no --attract -> renderer.attract_points is None.
    r_plain = build_renderer([])
    ok &= check("optin-none", r_plain.attract_points is None)

    # 7. --attract loads points onto the painter as a no-grad tensor in-frame.
    r_at = build_renderer(["--attract", circle])
    has = r_at.attract_points is not None
    ok &= check("attract-loaded", has and not r_at.attract_points.requires_grad)
    ok &= check("attract-frame", has and float(r_at.attract_points.max()) <= 512.0)

    # 8. active_control_points carries grad and a full step backprops to control_points.
    r_at.parameters()
    acp = r_at.active_control_points
    l = attraction_loss(acp, r_at.attract_points, deadzone=r_at.args.attraction_distance)
    l.backward()
    ok &= check("backprop-to-cp", r_at.control_points.grad is not None
                and torch.isfinite(r_at.control_points.grad).all().item())

    # 9. Compose with --origin: origin tensor present AND attract points loaded;
    #    the pinned origin is excluded from active_control_points already.
    r_orig = build_renderer(["--attract", circle, "--origin", "0.5", "0.5"])
    ok &= check("compose-origin",
                r_orig.attract_points is not None and hasattr(r_orig, "first_origin_points"))
    r_orig.parameters()
    lo = attraction_loss(r_orig.active_control_points, r_orig.attract_points, deadzone=25.0)
    lo.backward()
    ok &= check("compose-origin-backprop",
                r_orig.control_points.grad is not None
                and torch.isfinite(r_orig.control_points.grad).all().item())

    # 10. Compose with --avoid in the same run: attract own partition, avoid others.
    #     Both no-grad tensors coexist and both losses are differentiable together.
    r_both = build_renderer(["--attract", circle, "--avoid", circle2])
    ok &= check("compose-avoid-loaded",
                r_both.attract_points is not None and r_both.avoid_points is not None)
    r_both.parameters()
    combined = (attraction_loss(r_both.active_control_points, r_both.attract_points, 25.0)
                + avoidance_loss(r_both.active_control_points, r_both.avoid_points, 25.0))
    combined.backward()
    ok &= check("compose-avoid-backprop",
                r_both.control_points.grad is not None
                and torch.isfinite(r_both.control_points.grad).all().item())

    # 11. Acceptance #2 in miniature (no SDS, deterministic): plain gradient
    #     descent on the attraction loss alone must pull scattered points toward
    #     the target ring, reducing their mean Chamfer distance to it.
    torch.manual_seed(0)
    moving = (torch.rand(40, 2) * 512.0).requires_grad_(True)
    ring = torch.tensor(pts)

    def mean_chamfer(a, b):
        d = torch.linalg.vector_norm(a.unsqueeze(1) - b.unsqueeze(0), dim=-1)
        return 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())

    before = mean_chamfer(moving.detach(), ring).item()
    # Mirror the real pipeline's scaling: loss * attraction_weight, optimizer lr.
    opt = torch.optim.SGD([moving], lr=0.8)
    for _ in range(300):
        opt.zero_grad()
        (0.004 * attraction_loss(moving, ring, deadzone=25.0)).backward()
        opt.step()
    after = mean_chamfer(moving.detach(), ring).item()
    ok &= check("descent-reduces-chamfer", after < before * 0.5)
    print(f"       (mean Chamfer {before:.1f} -> {after:.1f} px)")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
