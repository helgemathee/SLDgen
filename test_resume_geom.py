"""Fast isolated test for the --stop-at / --resume checkpointing feature.

No diffusion model, CPU only -> runs in a few seconds. It drives the real
painter, the real optimizer and the real post_process_params with a *synthetic*
deterministic loss standing in for the SDS loss: a pull toward a circle, scaled
by a per-iteration random draw so that RNG restoration is genuinely exercised
(the real SDS loss samples a diffusion timestep every iteration -- get that wrong
and a resumed run silently diverges).

The invariant under test (Spec 1 SS8):

    running to epoch 40 uninterrupted == running to 20, checkpointing,
    and resuming to 40

On this CPU-only, diffusion-free path the equality is exact, so the assertions
are on bitwise-identical tensors rather than tolerances.

Run from the repo root:
    PYTHONPATH=. python test_resume_geom.py
"""
import random
import sys

import numpy as np
import torch

from SLDgen import config
from SLDgen.checkpoint import (
    CheckpointError,
    assert_fingerprint_matches,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    structural_fingerprint,
)
from SLDgen.painter.painter import SLDBSplinePainter
from SLDgen.painter.painter_optimizer import PainterOptimizer
from SLDgen.utils import get_sparse_loss_weight

HORIZON = 200  # --num-iter: the horizon every schedule normalises against
MIDPOINT = 20  # where the segmented run stops and checkpoints
END = 40  # where both runs finish

CAPTION = "a single line drawing of a test"


def make_args(name, horizon=HORIZON, extra=None):
    """Parse a real argument set, so validation and defaults are the real ones."""
    return config.parse_arguments(
        [
            "--target", "./data/firefighter.png",
            "--use-cpu",
            "--seed", "0",
            "--render-size", "256",
            "--n-control-points", "60",
            "--sampling-rate", "200",
            "--num-iter", str(horizon),
            "--init-method", "trefoil",
            "--experiment-name", f"resumetest_{name}",
        ]
        + (extra or [])
    )


def make_mask(render_size):
    m = torch.zeros((render_size, render_size), dtype=torch.float32)
    m[: render_size // 2, render_size // 4 : 3 * render_size // 4] = 1.0
    return m


def build(args, mask, checkpoint=None):
    """Fresh painter + optimizer, either initialised or restored from a checkpoint."""
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    if checkpoint is None:
        renderer.init_image()
    else:
        renderer.init_from_checkpoint(checkpoint)
    optimizer = PainterOptimizer(args, renderer)
    optimizer.init_optimizers()
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"], checkpoint["param_group_names"])
    return renderer, optimizer


def synthetic_loss(renderer, args, epoch):
    """Stand-in for the SDS loss: pull the curve toward a circle, with RNG jitter.

    The jitter draws from all three generators the real loop can touch, so a
    resumed segment that failed to restore RNG state would produce different
    numbers from the very first iteration.
    """
    curve = renderer.sampled_curve2d
    center = torch.tensor(
        [renderer.canvas_width / 2.0, renderer.canvas_height / 2.0], dtype=curve.dtype
    )
    radius = renderer.canvas_width * 0.3
    distance = torch.norm(curve - center, dim=-1)

    jitter = (
        1.0 + 0.2 * float(torch.rand(())) + 0.1 * float(np.random.rand()) + 0.1 * random.random()
    )
    return jitter * 1e-4 * ((distance - radius) ** 2).mean()


def step_segment(renderer, optimizer, args, start_epoch, stop_at):
    """The real loop's shape, with the diffusion term replaced.

    Keeps the real regularisation losses that do not need a GPU extension
    (sparse + length shortening) and the real post_process_params, so the
    monotone pruning of control points is part of what gets checkpointed.
    """
    for epoch in range(start_epoch + 1, stop_at + 1):
        optimizer.zero_grad_()
        renderer.get_polyline_2d()

        loss = synthetic_loss(renderer, args, epoch)

        if args.sparse_loss_weight > 0.0 and args.optimize_cp_weights:
            loss = loss + get_sparse_loss_weight(args, epoch) * torch.pow(
                torch.abs(renderer.weights) + 1e-8, args.sparse_loss_type
            ).mean()

        if args.length_shortening_loss_weight > 0.0:
            segments = renderer.sampled_curve2d[1:] - renderer.sampled_curve2d[:-1]
            loss = loss + torch.sum(torch.norm(segments, dim=-1)) * (
                args.length_shortening_loss_weight
            )

        loss.backward()
        optimizer.step_()
        with torch.no_grad():
            renderer.post_process_params()
    return stop_at


def snapshot(renderer, optimizer):
    return {
        "control_points": renderer.control_points.detach().clone(),
        "weights": renderer.weights.detach().clone(),
        "width": renderer.width.detach().clone(),
        "is_active_cp": renderer.is_active_cp.detach().clone(),
        "adam_step": float(optimizer.state_dict()["state"][0]["step"]),
    }


def compare(a, b, label):
    ok = True
    for key in ("control_points", "weights", "width", "is_active_cp"):
        equal = torch.equal(a[key], b[key])
        if not equal:
            delta = (a[key].float() - b[key].float()).abs().max().item()
            print(f"    {key}: DIFFERS (max |delta| = {delta:.3e})")
        ok = ok and equal
    step_equal = a["adam_step"] == b["adam_step"]
    if not step_equal:
        print(f"    adam_step: {a['adam_step']} vs {b['adam_step']}")
    ok = ok and step_equal
    print(f"[{label}] uninterrupted == segmented+resumed : {'PASS' if ok else 'FAIL'}")
    return ok


def test_resume_invariant():
    """Spec 1 SS8: a segmented run must land exactly where the uninterrupted one does."""
    mask = make_mask(256)

    args_a = make_args("uninterrupted")
    renderer_a, optimizer_a = build(args_a, mask)
    step_segment(renderer_a, optimizer_a, args_a, start_epoch=-1, stop_at=END)
    reference = snapshot(renderer_a, optimizer_a)

    args_b = make_args("segment1")
    renderer_b, optimizer_b = build(args_b, mask)
    step_segment(renderer_b, optimizer_b, args_b, start_epoch=-1, stop_at=MIDPOINT)
    ckpt_path = save_checkpoint(renderer_b, optimizer_b, args_b, MIDPOINT, CAPTION)

    # A genuinely fresh process would re-parse arguments and re-seed; do the same
    # so the test would catch a resume that only works because state lingered.
    args_c = make_args("segment2", extra=["--resume", str(ckpt_path)])
    checkpoint = load_checkpoint(ckpt_path)
    assert_fingerprint_matches(
        checkpoint["structural_fingerprint"], structural_fingerprint(args_c)
    )
    renderer_c, optimizer_c = build(args_c, mask, checkpoint=checkpoint)
    restore_rng(checkpoint["rng"])
    step_segment(renderer_c, optimizer_c, args_c, start_epoch=checkpoint["epoch"], stop_at=END)
    resumed = snapshot(renderer_c, optimizer_c)

    print(
        f"[resume] horizon={HORIZON} uninterrupted 0..{END} vs "
        f"0..{MIDPOINT} + resume {MIDPOINT + 1}..{END}"
    )
    return compare(reference, resumed, "resume")


def test_schedule_invariance():
    """The sparse-loss ramp must follow the horizon, never the segment length.

    This is the reason --stop-at exists as a separate flag: collapsing it back
    into --num-iter would make a 40-iteration preview apply sparsity five times
    faster than the first 40 iterations of the 200-iteration run it previews.
    """
    long_horizon = make_args("sched_long", horizon=HORIZON)
    short_horizon = make_args("sched_short", horizon=END)

    same = get_sparse_loss_weight(long_horizon, MIDPOINT)
    also_same = get_sparse_loss_weight(long_horizon, MIDPOINT)  # segment boundary is irrelevant
    different = get_sparse_loss_weight(short_horizon, MIDPOINT)

    passed = same == also_same and abs(different - same) > 1e-9
    print(
        f"[schedule] weight@epoch{MIDPOINT}: horizon {HORIZON} -> {same:.3f}, "
        f"horizon {END} -> {different:.3f} (must differ) : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_fingerprint_mismatch():
    """Resuming into different run-shaping arguments must be refused, not warned about."""
    mask = make_mask(256)
    args = make_args("fp_source")
    renderer, optimizer = build(args, mask)
    ckpt_path = save_checkpoint(renderer, optimizer, args, 0, CAPTION)
    checkpoint = load_checkpoint(ckpt_path)

    cases = {
        "lr": ["--lr", "0.4"],
        "num_iter": ["--num-iter", "4000"],
        "n_control_points": ["--n-control-points", "80"],
        "sparse_loss_weight": ["--sparse-loss-weight", "1000"],
        "caption": ["--caption", "something else"],
    }

    ok = True
    for field, extra in cases.items():
        other = make_args(f"fp_{field}", horizon=4000 if field == "num_iter" else HORIZON,
                          extra=extra)
        try:
            assert_fingerprint_matches(
                checkpoint["structural_fingerprint"], structural_fingerprint(other)
            )
            raised = False
            names = ""
        except CheckpointError as exc:
            raised = True
            names = str(exc)
        passed = raised and field in names
        ok = ok and passed
        print(f"[fingerprint] changing {field} refuses resume : {'PASS' if passed else 'FAIL'}")

    # ... while an operational-only difference must still resume.
    operational = make_args("fp_operational", extra=["--save-interval", "7", "--verbose"])
    try:
        assert_fingerprint_matches(
            checkpoint["structural_fingerprint"], structural_fingerprint(operational)
        )
        passed = True
    except CheckpointError as exc:
        passed = False
        print(f"    unexpected refusal: {exc}")
    ok = ok and passed
    print(
        "[fingerprint] operational-only differences still resume : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return ok


def test_pinned_geometry_roundtrip():
    """--origin's pinned tensors come from the TSP tour and must survive a checkpoint.

    Recomputing them would mean re-running Concorde and trusting it to return the
    same tour, so they are stored; this checks they come back byte for byte and
    that the resumed curve renders identically.
    """
    mask = make_mask(256)
    args = config.parse_arguments(
        [
            "--target", "./data/firefighter.png",
            "--use-cpu",
            "--seed", "0",
            "--render-size", "256",
            "--n-control-points", "60",
            "--sampling-rate", "200",
            "--num-iter", str(HORIZON),
            "--init-method", "tsp",
            "--origin", "0.5", "0.25",
            "--experiment-name", "resumetest_origin",
        ]
    )
    renderer, optimizer = build(args, mask)
    step_segment(renderer, optimizer, args, start_epoch=-1, stop_at=5)
    renderer.get_polyline_2d()
    curve_before = renderer.sampled_curve2d.detach().clone()

    ckpt_path = save_checkpoint(renderer, optimizer, args, 5, CAPTION)
    checkpoint = load_checkpoint(ckpt_path)
    restored, _ = build(args, mask, checkpoint=checkpoint)
    restored.get_polyline_2d()

    pinned_ok = all(
        torch.equal(getattr(renderer, name).cpu(), getattr(restored, name).cpu())
        for name in ("first_origin_points", "first_origin_weights", "first_origin_widths")
    )
    curve_ok = torch.equal(curve_before, restored.sampled_curve2d.detach())
    passed = pinned_ok and curve_ok and checkpoint["pinned_kind"] == "origin"
    print(
        f"[pinned] --origin tensors round-trip (pinned={pinned_ok}, curve={curve_ok}, "
        f"kind={checkpoint['pinned_kind']!r}) : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_prune_mask_is_state():
    """is_active_cp must be saved, not re-derived: pruning is monotone.

    A control point whose weight later rises back above the threshold must stay
    deactivated, which is impossible to reconstruct from the weights alone.
    """
    mask = make_mask(256)
    args = make_args("prune")
    renderer, _ = build(args, mask)

    with torch.no_grad():
        renderer.weights[:5] = 0.001  # push a few below the pruning threshold
        renderer.post_process_params()
        pruned = (~renderer.is_active_cp).sum().item()
        renderer.weights[:5] = 1.0  # ... and raise them again
        renderer.post_process_params()

    state = renderer.state_dict()
    restored = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    restored.init_image()
    restored.load_state_dict(state)

    stayed_pruned = (~restored.is_active_cp).sum().item() == pruned == 5
    derivable = (restored.weights <= 0.002).sum().item() == 0
    passed = stayed_pruned and derivable
    print(
        f"[prune] {pruned} pruned points survive a checkpoint although their weights "
        f"recovered : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def main():
    ok = True
    ok = test_schedule_invariance() and ok
    ok = test_resume_invariant() and ok
    ok = test_fingerprint_mismatch() and ok
    ok = test_pinned_geometry_roundtrip() and ok
    ok = test_prune_mask_is_state() and ok

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
