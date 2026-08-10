"""End-to-end test of run() itself, with the diffusion model stubbed out.

test_resume_geom.py proves the numerical invariant on a hand-driven loop and
test_checkpoint_ops.py proves the operational pieces in isolation. Neither
exercises ``SLDgen.run.run()`` -- the function that actually decides when to
checkpoint, what to skip on resume, and when a run counts as complete. This does,
by replacing only the four things that need a GPU, a network or a model:

    get_target          -> a synthetic image and mask (no RMBG-1.4 download)
    SD3GuidanceControl  -> a differentiable dummy loss that draws from the RNG,
                           and mutates args.caption exactly as create_caption does
    get_all_metrics     -> a fixed dict (no CLIP / DINO / aesthetic model loads)
    make_video          -> a touch (no ffmpeg)

Everything else is the real code path: the real painter, optimizer, diffvg
rasterisation, checkpoint writing, state.json, finalisation and completion gating.

The headline assertion is the same invariant as Spec 1 SS8, but black-box:
an uninterrupted run and a stopped-then-resumed run must produce a byte-identical
``final_sld.svg``.

CPU only, no network. Run from the repo root:
    PYTHONPATH=. python test_run_segments.py
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import torch

import SLDgen.run as run_module
from SLDgen import config
from SLDgen.run import run

HORIZON = 6  # tiny, but every branch of the loop is hit
RENDER_SIZE = 128
BLIP2_CAPTION = "a single line drawing of a firefighter"


class FakeGuidance:
    """Stand-in for SD3GuidanceControl.

    Reproduces the two behaviours the checkpointing code has to cope with: it
    consumes RNG every iteration (the real loss samples a diffusion timestep), and
    it fills in an empty caption by mutating args.caption in place.
    """

    def __init__(self, args, device):
        self.device = device
        if args.caption == "":
            args.caption = BLIP2_CAPTION

    def __call__(self, raster):
        jitter = 1.0 + 0.5 * torch.rand((), device=raster.device)
        return jitter * raster.mean()


def fake_get_target(args):
    args.original_target_path = args.target
    mask = torch.zeros((args.render_size, args.render_size), dtype=torch.float32)
    mask[args.render_size // 4 : 3 * args.render_size // 4, :] = 1.0
    inputs = torch.zeros(1, 3, args.render_size, args.render_size)
    return inputs, mask


def install_stubs():
    run_module.get_target = fake_get_target
    run_module.SD3GuidanceControl = FakeGuidance
    run_module.get_all_metrics = lambda *a, **k: {"stubbed": True}
    run_module.make_video = lambda args: Path(args.output_dir, "sketch.mp4").touch()


def make_args(name, extra=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return config.parse_arguments(
            [
                "--target", "./data/firefighter.png",
                "--use-cpu",
                "--seed", "0",
                "--render-size", str(RENDER_SIZE),
                "--n-control-points", "30",
                "--sampling-rate", "100",
                "--num-iter", str(HORIZON),
                "--save-interval", "2",
                "--init-method", "trefoil",
                "--experiment-name", name,
            ]
            + (extra or [])
        )


def fresh(name, extra=None):
    """Parse arguments into a run directory wiped of any previous attempt."""
    args = make_args(name, extra)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    return make_args(name, extra)


def quiet_run(args):
    with contextlib.redirect_stdout(io.StringIO()):
        run(args)


def read_state(args):
    with open(Path(args.output_dir) / "state.json") as f:
        return json.load(f)


def test_plain_run_is_unchanged():
    """With no checkpointing flag: same artifacts as before, and nothing new."""
    args = fresh("runseg_plain")
    quiet_run(args)
    out = Path(args.output_dir)

    produced = all(
        (out / name).exists()
        for name in ("final_sld.svg", "final_sld.png", "metrics.json", "sketch.mp4")
    )
    # Strict opt-in: no checkpoints/, no state.json.
    no_new_files = not (out / "checkpoints").exists() and not (out / "state.json").exists()
    # Frames at epoch 0 and every --save-interval, exactly as before.
    frames = sorted(p.name for p in (out / "svg_to_png").glob("iter_*.png"))
    frames_ok = frames == ["iter_0000.png", "iter_0002.png", "iter_0004.png", "iter_0006.png"]

    passed = produced and no_new_files and frames_ok
    print(
        f"[plain] finalised={produced} no-new-files={no_new_files} frames={frames} : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_segment_stops_without_finalising():
    """A segment that stops short leaves a checkpoint and nothing else."""
    args = fresh("runseg_segmented", ["--stop-at", "3"])
    quiet_run(args)
    out = Path(args.output_dir)

    checkpoint = out / "checkpoints" / "ckpt_00003.pt"
    checkpointed = checkpoint.exists() and (out / "checkpoints" / "latest.pt").exists()
    # No final export, no metrics, no video: the trajectory is not finished.
    not_finalised = not any(
        (out / name).exists()
        for name in ("final_sld.svg", "final_sld.png", "metrics.json", "sketch.mp4")
    )
    state = read_state(args)
    state_ok = (
        state["epoch"] == 3
        and state["stop_at"] == 3
        and state["num_iter"] == HORIZON
        and state["phase"] == "optimizing"
        and state["latest_checkpoint"] == "checkpoints/ckpt_00003.pt"
        and state["resolved_caption"] == BLIP2_CAPTION
        and state["iters_per_sec"] is not None
    )
    # config.json is written alongside the checkpoint, so a paused run is
    # self-documenting rather than only explaining itself once it finishes.
    config_ok = (out / "config.json").exists()

    passed = checkpointed and not_finalised and state_ok and config_ok
    print(
        f"[segment] checkpoint={checkpointed} not-finalised={not_finalised} "
        f"state={state_ok} config.json={config_ok} : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_resumed_run_matches_uninterrupted():
    """The headline invariant, black-box: 0..6 == (0..3, resume 4..6)."""
    reference_args = fresh("runseg_reference")
    quiet_run(reference_args)
    reference_svg = (Path(reference_args.output_dir) / "final_sld.svg").read_bytes()

    segment_args = fresh("runseg_resumed", ["--stop-at", "3"])
    quiet_run(segment_args)

    checkpoint = Path(segment_args.output_dir) / "checkpoints" / "ckpt_00003.pt"
    resumed_args = make_args("runseg_resumed", ["--resume", str(checkpoint)])
    quiet_run(resumed_args)
    out = Path(resumed_args.output_dir)
    resumed_svg = (out / "final_sld.svg").read_bytes()

    identical = reference_svg == resumed_svg
    finalised = all(
        (out / name).exists()
        for name in ("final_sld.svg", "final_sld.png", "metrics.json", "sketch.mp4")
    )
    # The frame at epoch 0 is not re-rendered on resume, so the video that spans
    # both segments has each frame exactly once.
    frames = sorted(p.name for p in (out / "svg_to_png").glob("iter_*.png"))
    frames_ok = frames == ["iter_0000.png", "iter_0002.png", "iter_0004.png", "iter_0006.png"]
    state = read_state(resumed_args)
    done = state["phase"] == "done" and state["epoch"] == HORIZON

    passed = identical and finalised and frames_ok and done
    print(
        f"[resume] final_sld.svg identical={identical} finalised={finalised} "
        f"frames={frames} phase={state['phase']!r} : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_periodic_checkpoints():
    """--checkpoint-interval writes crash-recovery checkpoints along the way."""
    args = fresh("runseg_interval", ["--checkpoint-interval", "2"])
    quiet_run(args)
    out = Path(args.output_dir)

    written = sorted(p.name for p in (out / "checkpoints").glob("ckpt_*.pt"))
    expected = ["ckpt_00002.pt", "ckpt_00004.pt", "ckpt_00006.pt"]
    # The run still finishes normally: periodic checkpointing is not a segment.
    finalised = (out / "final_sld.svg").exists() and read_state(args)["phase"] == "done"

    passed = written == expected and finalised
    print(
        f"[interval] checkpoints={written} (want {expected}) finalised={finalised} : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_no_video_skips_ffmpeg_only():
    """--no-video drops the mp4 and the ffmpeg call, and nothing else.

    The frames are what the mp4 is assembled from, so they must survive: the
    flag exists so a host without ffmpeg can still finish a run, not so a run
    produces less.
    """
    called = []
    original = run_module.make_video
    run_module.make_video = lambda args: called.append(args) or original(args)
    try:
        args = fresh("runseg_novideo", ["--no-video"])
        quiet_run(args)
    finally:
        run_module.make_video = original
    out = Path(args.output_dir)

    no_video = not (out / "sketch.mp4").exists() and not called
    # Everything else a completed run owes the caller is still there.
    finalised = all(
        (out / name).exists() for name in ("final_sld.svg", "final_sld.png", "metrics.json")
    )
    frames = sorted(p.name for p in (out / "svg_to_png").glob("iter_*.png"))
    frames_ok = frames == ["iter_0000.png", "iter_0002.png", "iter_0004.png", "iter_0006.png"]

    passed = no_video and finalised and frames_ok
    print(
        f"[no-video] mp4-absent={no_video} finalised={finalised} frames={len(frames)} : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_resume_refuses_redirected_trajectory():
    """Changing a run-shaping argument between segments must be a hard stop."""
    from SLDgen.checkpoint import CheckpointError

    args = fresh("runseg_redirect", ["--stop-at", "3"])
    quiet_run(args)
    checkpoint = Path(args.output_dir) / "checkpoints" / "ckpt_00003.pt"

    redirected = make_args(
        "runseg_redirect", ["--resume", str(checkpoint), "--sparse-loss-weight", "1234"]
    )
    try:
        quiet_run(redirected)
        raised, message = False, "run() accepted a redirected trajectory"
    except CheckpointError as exc:
        raised, message = True, str(exc)

    passed = raised and "sparse_loss_weight" in message
    print(f"[redirect] run() refuses a mismatched resume : {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"    {message[:200]}")
    return passed


def main():
    install_stubs()
    ok = True
    for test in (
        test_plain_run_is_unchanged,
        test_segment_stops_without_finalising,
        test_resumed_run_matches_uninterrupted,
        test_periodic_checkpoints,
        test_no_video_skips_ffmpeg_only,
        test_resume_refuses_redirected_trajectory,
    ):
        ok = test() and ok

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
