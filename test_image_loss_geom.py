"""Fast isolated tests for the --image-loss feature (Spec 6 SS12.2).

No diffusion model, CPU only, a few seconds. Exercises the schedule, the
gradient blend, the edge-target construction and its coordinate frame, the
chamfer term, the diagnostics log, the flag validation, and a gradient through
the real painter.

Run from the repo root:
    PYTHONPATH=. python test_image_loss_geom.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from SLDgen import config
from SLDgen.image_loss import (
    SKIP_IMG,
    SKIP_NONE,
    SKIP_SDS,
    ImageFidelityLoss,
    ImageLossLog,
    alpha_at,
    blend_gradients,
    build_edge_map,
    curve_samples,
    edge_points,
    is_binary_map,
    nearest_distances,
    prepare_edges,
)
from SLDgen.painter.painter import SLDBSplinePainter

SCRATCH = Path("/tmp/claude-1000/-home-helge-SLDgen/image_loss_geom")
SCRATCH.mkdir(parents=True, exist_ok=True)


def check(name, cond, detail=""):
    print(f"[{name}] {'PASS' if cond else 'FAIL'}" + (f"  {detail}" if detail and not cond else ""))
    return bool(cond)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_alpha_at():
    ok = True
    ok &= check("alpha constant", all(alpha_at(e, 100, "constant", 0.3) == 0.3 for e in (0, 50, 99)))
    ok &= check(
        "alpha decay default start",
        close(alpha_at(0, 101, "decay", 0.1), 0.5)
        and close(alpha_at(50, 101, "decay", 0.1), 0.3)
        and close(alpha_at(100, 101, "decay", 0.1), 0.1),
    )
    ok &= check(
        "alpha ramp default start",
        close(alpha_at(0, 11, "ramp", 0.3), 0.05) and close(alpha_at(10, 11, "ramp", 0.3), 0.3),
    )
    ok &= check(
        "alpha explicit start",
        close(alpha_at(0, 11, "decay", 0.2, 0.8), 0.8) and close(alpha_at(5, 11, "decay", 0.2, 0.8), 0.5),
    )
    return ok


def test_blend():
    ok = True
    torch.manual_seed(0)
    g_sds, g_img = torch.randn(40, 2), torch.randn(40, 2) * 37.0
    out, stats = blend_gradients(g_sds, g_img, 0.0)
    ok &= check("blend alpha 0 is exactly g_sds", torch.equal(out, g_sds))

    out, stats = blend_gradients(g_sds, g_img, 1.0)
    cos = float((out * g_img).sum() / (out.norm() * g_img.norm()))
    ok &= check(
        "blend alpha 1 parallel to g_img at |g_sds|",
        close(cos, 1.0, 1e-5) and close(float(out.norm()), float(g_sds.norm()), 1e-4),
    )

    direct = float((g_sds * g_img).sum() / (g_sds.norm() * g_img.norm()))
    ok &= check("blend cosine", close(stats["cosine"], direct, 1e-6) and stats["skipped"] == SKIP_NONE)

    out, stats = blend_gradients(g_sds, g_img, 0.5)
    expect = 0.5 * g_sds + 0.5 * (g_sds.norm() / g_img.norm()) * g_img
    ok &= check("blend alpha 0.5", torch.allclose(out, expect, atol=1e-6))
    ok &= check("blend norm bounded by |g_sds|", float(out.norm()) <= float(g_sds.norm()) + 1e-5)

    out, stats = blend_gradients(g_sds, torch.zeros_like(g_sds), 0.5)
    ok &= check("zero-norm guard (img)", torch.equal(out, g_sds) and stats["skipped"] == SKIP_IMG)
    out, stats = blend_gradients(torch.zeros_like(g_img), g_img, 0.5)
    ok &= check("zero-norm guard (sds)", torch.equal(out, g_img) and stats["skipped"] == SKIP_SDS)
    ok &= check("undefined cosine is None", stats["cosine"] is None)
    return ok


def test_edge_points():
    ok = True
    edges = np.zeros((256, 256), np.uint8)
    edges[20:220, 100] = 255
    pts, raw = edge_points(edges)
    ok &= check("edge points are (x, y) = (col, row)", raw == 200 and torch.all(pts[:, 0] == 100))
    ok &= check("edge points y span rows", float(pts[:, 1].min()) == 20 and float(pts[:, 1].max()) == 219)

    dense = np.zeros((300, 300), np.uint8)
    dense[:, :150] = 255
    a, raw_a = edge_points(dense, cap=1000)
    b, _ = edge_points(dense, cap=1000)
    ok &= check(
        "edge cap: fixed stride, deterministic",
        raw_a == 45000 and len(a) <= 1000 and torch.equal(a, b),
        f"{raw_a} -> {len(a)}",
    )
    return ok


def test_chamfer():
    ok = True
    edges = np.zeros((256, 256), np.uint8)
    edges[:, 100] = 255
    edge_pts, _ = edge_points(edges)
    for d in (0.0, 3.0, 17.5):
        ys = torch.arange(10.0, 240.0, 5.0)  # integer rows: the edge is pixel-sampled
        curve = torch.stack([torch.full_like(ys, 100.0 + d), ys], dim=1)
        value = float(nearest_distances(curve, edge_pts).mean())
        ok &= check(f"chamfer offset {d}", close(value, d, 1e-3), f"got {value}")

    curve = torch.stack([torch.full((50,), 120.0), torch.linspace(10, 240, 50)], dim=1)
    curve.requires_grad_(True)
    nearest_distances(curve, edge_pts).mean().backward()
    ok &= check(
        "chamfer gradient points toward the line",
        bool(torch.all(curve.grad[:, 0] > 0)) and float(curve.grad[:, 1].abs().max()) < 1e-3,
    )

    on_line = torch.tensor([[100.0, 50.0]], requires_grad=True)
    nearest_distances(on_line, edge_pts).sum().backward()
    ok &= check("chamfer gradient finite on the edge", bool(torch.isfinite(on_line.grad).all()))
    return ok


def canvas_image(size=256):
    """A white canvas with a dark disc: the shape of what get_target produces."""
    yy, xx = np.mgrid[:size, :size]
    disc = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 4) ** 2
    rgb = np.full((size, size, 3), 255, np.uint8)
    rgb[disc] = 40
    mask = torch.zeros(size, size)
    mask[size // 8 : 7 * size // 8, size // 8 : 7 * size // 8] = 1.0
    return Image.fromarray(rgb), mask


def loss_args(**overrides):
    args = dict(
        image_loss_target=None,
        render_size=256,
        image_loss_canny_low=100.0,
        image_loss_canny_high=200.0,
        image_loss_canny_blur=3,
        image_loss_chamfer=1.0,
        image_loss_pyramid=0.0,
        image_loss_landmark=0.0,
        image_loss_curve_samples=200,
        output_dir=str(SCRATCH),
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_edge_target():
    ok = True
    image, mask = canvas_image()
    edges, source = build_edge_map(loss_args(), image, mask)
    ok &= check("derived target is binary", is_binary_map(edges) and edges.max() == 255)
    ys, xs = np.nonzero(edges)
    radius = np.hypot(xs - 128, ys - 128)
    ok &= check(
        "derived target traces the disc", len(xs) > 100 and abs(float(np.median(radius)) - 64) < 3
    )
    ok &= check("derived target == prepare_edges defaults", np.array_equal(edges, prepare_edges(image, mask)))

    # The mask zeroes everything outside the subject.
    border = np.zeros((256, 256, 3), np.uint8)
    border[:, 128:] = 255
    edges, _ = build_edge_map(loss_args(), Image.fromarray(border), torch.zeros(256, 256))
    ok &= check("mask removes edges outside the subject", edges.max() == 0)

    # A supplied black-on-white line drawing is inverted to edges-are-255.
    drawing = np.full((256, 256), 255, np.uint8)
    drawing[60:200, 90] = 0
    path = SCRATCH / "drawing.png"
    Image.fromarray(drawing).save(path)
    edges, source = build_edge_map(loss_args(image_loss_target=str(path)), None, None)
    ok &= check(
        "supplied black-on-white map used as edges",
        "supplied edge map" in source and np.count_nonzero(edges) == 140 and edges[100, 90] == 255,
    )

    # A supplied photograph-like image gets Canny.
    grey = np.asarray(image.convert("L"))
    Image.fromarray(grey).save(SCRATCH / "grey.png")
    edges, source = build_edge_map(loss_args(image_loss_target=str(SCRATCH / "grey.png")), None, mask)
    ok &= check("supplied non-binary image gets Canny", "Canny over supplied" in source and edges.max() == 255)

    # preserve_silhouette puts the mask outline in; roi drops what lies outside.
    edges = prepare_edges(image, mask, preserve_silhouette=True)
    ok &= check("preserve_silhouette adds the mask outline", edges[32, 128] == 255 and edges[128, 32] == 255)
    edges = prepare_edges(image, mask, roi=(0, 0, 128, 256))
    ok &= check("roi keeps only the box", np.count_nonzero(edges[:, 128:]) == 0 and edges.max() == 255)
    return ok


def test_loss_object():
    ok = True
    image, mask = canvas_image()
    loss = ImageFidelityLoss(loss_args(), image, mask, None, "cpu")
    ok &= check("target png written", (SCRATCH / "image_loss_target.png").exists())
    ok &= check("describe mentions edge px", "edge px" in loss.describe())

    t = torch.linspace(0, 2 * math.pi, 500)
    on_disc = torch.stack([128 + 64 * torch.cos(t), 128 + 64 * torch.sin(t)], dim=1)
    off_disc = torch.stack([128 + 90 * torch.cos(t), 128 + 90 * torch.sin(t)], dim=1)
    value_on, parts = loss(SimpleNamespace(sampled_curve2d=on_disc), None)
    value_off, _ = loss(SimpleNamespace(sampled_curve2d=off_disc), None)
    ok &= check(
        "loss lower on the edge than off it",
        float(value_on) < 2.0 and float(value_off) > 20.0,
        f"{float(value_on)} vs {float(value_off)}",
    )
    ok &= check(
        "parts: chamfer set, others None",
        parts["chamfer"] == float(value_on) and parts["pyramid"] is None and parts["landmark"] is None,
    )
    ok &= check("curve_samples strides", len(curve_samples(SimpleNamespace(sampled_curve2d=on_disc), 100)) == 100)
    return ok


def test_log_resume():
    ok = True
    stats = {"sds_norm": 1.5, "img_norm": 0.25, "cosine": None, "skipped": 0}
    parts = {"chamfer": 3.25, "pyramid": None, "landmark": None}
    log = ImageLossLog(SCRATCH)
    for epoch in range(6):
        log.write(epoch, 0.2, stats, 0.5, 3.25, parts)
    log.close()
    full = (SCRATCH / "image_loss_log.csv").read_text()
    ok &= check("log header + rows", full.splitlines()[0].startswith("epoch,alpha,") and len(full.splitlines()) == 7)
    ok &= check("log empty cells for None", full.splitlines()[1].endswith(",3.25,,"))

    log = ImageLossLog(SCRATCH, resume_epoch=3)
    for epoch in range(4, 6):
        log.write(epoch, 0.2, stats, 0.5, 3.25, parts)
    log.close()
    ok &= check("log resume drops rows past checkpoint, identical", (SCRATCH / "image_loss_log.csv").read_text() == full)
    return ok


def parse(extra):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            config.parse_arguments(
                ["--target", "./data/firefighter.png", "--use-cpu", "--experiment-name", "imglossgeom",
                 "--render-size", "256"] + extra
            )
            return True, out.getvalue() + err.getvalue()
        except SystemExit:
            return False, out.getvalue() + err.getvalue()


def test_validation():
    ok = True
    ok &= check("valid defaults", parse(["--image-loss"])[0])
    ok &= check("weight 0 refused", not parse(["--image-loss", "--image-loss-weight", "0"])[0])
    ok &= check("weight > 1 refused", not parse(["--image-loss", "--image-loss-weight", "1.5"])[0])
    ok &= check(
        "constant with start refused",
        not parse(["--image-loss", "--image-loss-schedule-start", "0.4"])[0],
    )
    ok &= check(
        "decay needs start > weight",
        not parse(["--image-loss", "--image-loss-schedule", "decay", "--image-loss-weight", "0.6"])[0]
        and parse(["--image-loss", "--image-loss-schedule", "decay", "--image-loss-weight", "0.1"])[0],
    )
    ok &= check(
        "ramp needs start < weight",
        not parse(["--image-loss", "--image-loss-schedule", "ramp", "--image-loss-weight", "0.3",
                   "--image-loss-schedule-start", "0.5"])[0],
    )
    ok &= check("all terms zero refused", not parse(["--image-loss", "--image-loss-chamfer", "0"])[0])
    ok &= check(
        "landmark without file refused",
        not parse(["--image-loss", "--image-loss-landmark", "1"])[0],
    )
    ok &= check(
        "canny low >= high refused",
        not parse(["--image-loss", "--image-loss-canny-low", "200", "--image-loss-canny-high", "100"])[0],
    )
    ok &= check(
        "curve samples > sampling rate refused",
        not parse(["--image-loss", "--image-loss-curve-samples", "6000"])[0],
    )
    Image.fromarray(np.zeros((100, 100), np.uint8)).save(SCRATCH / "small.png")
    passed, text = parse(["--image-loss", "--image-loss-target", str(SCRATCH / "small.png")])
    ok &= check("wrong-size target refused, names the script", not passed and "sld_edge_target.py" in text)
    ok &= check("missing target refused", not parse(["--image-loss", "--image-loss-target", "/nope.png"])[0])
    passed, text = parse(["--image-loss-weight", "0.7"])
    ok &= check("knob without gate warns, not refuses", passed and "ignored without --image-loss" in text)
    return ok


def test_real_painter():
    with contextlib.redirect_stdout(io.StringIO()):
        args = config.parse_arguments(
            ["--target", "./data/firefighter.png", "--use-cpu", "--seed", "0", "--render-size", "256",
             "--experiment-name", "imglossgeom", "--init-method", "trefoil", "--n-control-points", "30",
             "--sampling-rate", "1000", "--image-loss", "--image-loss-curve-samples", "200"]
        )
    image, mask = canvas_image()
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer.init_image()
    renderer.parameters()
    renderer.get_image()
    args.output_dir = str(SCRATCH)
    loss_fn = ImageFidelityLoss(args, image, mask, None, "cpu")
    loss, _ = loss_fn(renderer, None)
    (grad,) = torch.autograd.grad(loss, renderer.control_points)
    ok = check(
        "real painter: grad reaches control points, finite, nonzero",
        grad.shape == renderer.control_points.shape and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 0,
    )
    ok &= check("real painter: weights.grad untouched", renderer.weights.grad is None)
    return ok


def main():
    ok = True
    for test in (
        test_alpha_at,
        test_blend,
        test_edge_points,
        test_chamfer,
        test_edge_target,
        test_loss_object,
        test_log_resume,
        test_validation,
        test_real_painter,
    ):
        ok = test() and ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
