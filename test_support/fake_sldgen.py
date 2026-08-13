"""A stand-in for ``sldgen.py`` that needs no GPU, no models and no torch.

The worker's job is supervision: spawn a segment, poll its heartbeat, signal it,
classify how it ended, and decide what state the job lands in. All of that is
testable -- and worth testing -- without spending 15 minutes of GPU time per
case, provided the thing being supervised behaves like the real one. So this
script honours the parts of the contract the worker actually depends on:

  * the ``<output-dir>/<target-stem>/<experiment-name>/`` layout
  * ``state.json`` written at every save interval and every checkpoint
  * ``checkpoints/ckpt_NNNNN.pt`` plus a ``latest.pt`` copy
  * ``--resume`` continuing from the checkpoint's own epoch
  * graceful SIGTERM: finish the iteration, checkpoint, exit 0
  * a second SIGTERM exiting 143 immediately
  * finalisation only on reaching ``--num-iter``
  * SLDgen's exit codes, and tqdm-style ``\\r`` progress output

Checkpoints are JSON with a ``.pt`` name: the worker only ever reads their
*filenames*, and the fake reads its own back, so no tensor library is involved.

Fault injection, for the failure-path tests. Environment variables apply to every
segment, which is fine for a whole-run setting like iteration speed; per-job
faults instead come from a ``fault.json`` the test drops in the job directory
(the ``--output-dir``), because the worker gives every child the same environment
and a test needs to fail one specific job:

  {"exit_code": 3}                  exit with this code instead of running
  {"exit_code": 1, "fail_at": 50}   run to this epoch, then exit
  {"stall": true}                   ignore SIGTERM, to exercise the SIGKILL fallback
  {"iter_delay": 0.05}              slow the loop down

  FAKE_SLDGEN_ITER_DELAY  default seconds per iteration (default 0.002)
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BLIP2_CAPTION = "a single line drawing of a firefighter"

_stop_requested = False
_signal_count = 0
_fault = {}


def _handle_sigterm(signum, frame):
    global _stop_requested, _signal_count
    if _fault.get("stall"):
        print("ignoring SIGTERM (stall mode)", flush=True)
        return
    _signal_count += 1
    if _signal_count >= 2:
        print("second SIGTERM: aborting immediately", flush=True)
        os._exit(143)
    _stop_requested = True
    print("SIGTERM received: finishing the iteration, then checkpointing", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="run")
    parser.add_argument("--num-iter", type=int, default=4000)
    parser.add_argument("--stop-at", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--caption", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_false", dest="save_video")
    parser.add_argument("--render-size", type=int, default=512)
    # Canny attraction. The fake runs the *real* SLDgen/canny_attract.py here
    # rather than faking its output: that module needs only cv2/numpy, the SVG
    # it produces is what the service and UI consume, and a stub of it would
    # test the plumbing against a file the real run never writes.
    parser.add_argument("--attract-canny", action="store_true")
    parser.add_argument("--attract-canny-low", type=float, default=100.0)
    parser.add_argument("--attract-canny-high", type=float, default=200.0)
    parser.add_argument("--attract-canny-blur", type=int, default=3)
    parser.add_argument("--attract-canny-simplify", type=float, default=1.0)
    parser.add_argument("--attract-canny-min-length", type=float, default=12.0)
    parser.add_argument("--attract-canny-max-points", type=int, default=400)
    return parser.parse_known_args()[0]


def write_canvas_images(run_dir, target, render_size):
    """Stand in for ``targets.get_target``: the canvas-space image and its mask.

    The real run writes these before anything else, and three features read them
    back -- the labelmap partition default, the artwork pane, and the Canny
    preview -- so a fake that skips them cannot exercise any of those. Real
    pixels when PIL is available, placeholder bytes when it is not, because the
    supervision tests must keep running in an interpreter with no image stack.
    """
    try:
        from PIL import Image
    except ImportError:
        (run_dir / "input.png").write_bytes(b"\x89PNG\r\n\x1a\n(input)")
        (run_dir / "mask.png").write_bytes(b"\x89PNG\r\n\x1a\n(mask)")
        return False

    source = Image.open(target).convert("RGB") if Path(target).exists() else None
    canvas = Image.new("RGB", (render_size, render_size), (255, 255, 255))
    if source is not None:
        # Same shape as the real pipeline: fit the subject into the canvas
        # centred, which is what makes the coordinates canvas space.
        scaled = source.copy()
        scaled.thumbnail((int(render_size * 0.75), int(render_size * 0.75)))
        canvas.paste(
            scaled,
            ((render_size - scaled.width) // 2, (render_size - scaled.height) // 2),
        )
    canvas.save(run_dir / "input.png")

    mask = Image.new("L", (render_size, render_size), 0)
    inset = int(render_size * 0.1)
    mask.paste(255, (inset, inset, render_size - inset, render_size - inset))
    mask.save(run_dir / "mask.png")
    return True


def write_attract_canny(run_dir, args):
    """Run the real generator, exactly where ``run.run`` runs it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from PIL import Image

        from SLDgen.canny_attract import describe, generate
    except ImportError as error:
        print(f"fake sldgen: --attract-canny needs cv2/numpy/PIL ({error})", flush=True)
        return None

    stats = generate(
        Image.open(run_dir / "input.png"),
        run_dir / "attract_canny.svg",
        mask=Image.open(run_dir / "mask.png"),
        low=args.attract_canny_low,
        high=args.attract_canny_high,
        blur=args.attract_canny_blur,
        simplify=args.attract_canny_simplify,
        min_length=args.attract_canny_min_length,
        max_points=args.attract_canny_max_points,
    )
    print(f"\tCanny attraction: {describe(stats)}", flush=True)
    return stats


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_state(run_dir, **fields):
    """Atomic, exactly as the real heartbeat is: a poller must never see a partial file."""
    payload = {"updated_at": utcnow(), **fields}
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(payload, indent=4))
    os.replace(tmp, run_dir / "state.json")


def write_checkpoint(run_dir, epoch, num_iter, caption):
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ckpt_{epoch:05d}.pt"
    path.write_text(
        json.dumps({"format_version": 1, "epoch": epoch, "num_iter": num_iter,
                    "resolved_caption": caption})
    )
    shutil.copyfile(path, directory / "latest.pt")
    return path


def load_fault(output_dir):
    path = Path(output_dir) / "fault.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    global _fault
    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = parse_args()
    _fault = load_fault(args.output_dir)

    forced_exit = _fault.get("exit_code")
    fail_at = _fault.get("fail_at")
    if forced_exit is not None and fail_at is None:
        print(f"fake sldgen: failing immediately with exit code {forced_exit}", flush=True)
        return int(forced_exit)

    run_dir = Path(args.output_dir) / Path(args.target).stem / args.experiment_name
    for name in ("svg_to_png", "svg_logs", "checkpoints"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    caption = args.caption or BLIP2_CAPTION
    stop_at = args.stop_at if args.stop_at is not None else args.num_iter
    delay = float(_fault.get("iter_delay", os.environ.get("FAKE_SLDGEN_ITER_DELAY", "0.002")))

    start_epoch = -1
    if args.resume:
        resume_payload = json.loads(Path(args.resume).read_text())
        start_epoch = int(resume_payload["epoch"])
        print(f"Resuming {args.resume} at epoch {start_epoch} (horizon {args.num_iter}).",
              flush=True)

    print("Running SLDgen:", flush=True)
    write_state(run_dir, epoch=max(start_epoch, 0), num_iter=args.num_iter, stop_at=stop_at,
                phase="init", resolved_caption=caption, iters_per_sec=None,
                latest_checkpoint=None, latest_preview=None)

    # get_target runs on every segment, resumed or not, and so does the Canny
    # generation that follows it -- which is what makes the generated SVG
    # identical across segments rather than something only the first one has.
    real_images = write_canvas_images(run_dir, args.target, args.render_size)
    if args.attract_canny and real_images:
        write_attract_canny(run_dir, args)

    if start_epoch < 0:
        # save_current_step writes the SVG *and* the PNG, at epoch 0 as at every
        # save interval. Writing only the PNG here would make frame 0 look like
        # it had no SVG, which is not something the real run ever produces.
        (run_dir / "svg_to_png" / "iter_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n(frame 0)")
        (run_dir / "svg_logs" / "svg_iter0.svg").write_text("<svg/>")
        (run_dir / "config.json").write_text(json.dumps({"num_iter": args.num_iter}, indent=4))

    latest_preview = "svg_to_png/iter_0000.png"
    latest_checkpoint = None
    epoch = start_epoch
    started = time.time()

    for epoch in range(start_epoch + 1, stop_at + 1):
        time.sleep(delay)
        # tqdm-style repaint: the same line rewritten, which is what makes
        # carriage-return cooking worth having at serve time.
        sys.stdout.write(f"\r    {epoch}/{stop_at} iterations")
        sys.stdout.flush()

        if fail_at is not None and epoch >= int(fail_at):
            code = int(forced_exit if forced_exit is not None else 1)
            print(f"\nfake sldgen: failing at epoch {epoch} with exit code {code}", flush=True)
            return code

        rate = (epoch - start_epoch) / max(time.time() - started, 1e-9)
        if epoch % args.save_interval == 0 and epoch > 0:
            # Distinct per epoch, so a test can tell *which* frame was served --
            # "the preview follows the frame you parked on" is unprovable when
            # every frame is byte-identical.
            (run_dir / "svg_to_png" / f"iter_{epoch:04d}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n(frame %d)" % epoch
            )
            (run_dir / "svg_logs" / f"svg_iter{epoch}.svg").write_text(
                f"<svg><!-- epoch {epoch} --></svg>"
            )
            latest_preview = f"svg_to_png/iter_{epoch:04d}.png"
            write_state(run_dir, epoch=epoch, num_iter=args.num_iter, stop_at=stop_at,
                        phase="optimizing", iters_per_sec=rate, resolved_caption=caption,
                        latest_checkpoint=latest_checkpoint, latest_preview=latest_preview)

        due = args.checkpoint_interval > 0 and epoch > 0 and epoch % args.checkpoint_interval == 0
        if due and epoch < stop_at:
            path = write_checkpoint(run_dir, epoch, args.num_iter, caption)
            latest_checkpoint = f"checkpoints/{path.name}"
            write_state(run_dir, epoch=epoch, num_iter=args.num_iter, stop_at=stop_at,
                        phase="optimizing", iters_per_sec=rate, resolved_caption=caption,
                        latest_checkpoint=latest_checkpoint, latest_preview=latest_preview)

        if _stop_requested:
            break

    path = write_checkpoint(run_dir, epoch, args.num_iter, caption)
    latest_checkpoint = f"checkpoints/{path.name}"
    rate = (epoch - start_epoch) / max(time.time() - started, 1e-9)
    write_state(run_dir, epoch=epoch, num_iter=args.num_iter, stop_at=stop_at,
                phase="optimizing", iters_per_sec=rate, resolved_caption=caption,
                latest_checkpoint=latest_checkpoint, latest_preview=latest_preview)

    if epoch < args.num_iter:
        reason = "SIGTERM" if _stop_requested else f"--stop-at {stop_at}"
        print(f"\n\tStopped at epoch {epoch} of {args.num_iter} ({reason}).", flush=True)
        return 0

    write_state(run_dir, epoch=epoch, num_iter=args.num_iter, stop_at=stop_at,
                phase="finalizing", iters_per_sec=rate, resolved_caption=caption,
                latest_checkpoint=latest_checkpoint, latest_preview=latest_preview)
    (run_dir / "final_sld.svg").write_text("<svg><path d='M 0 0 L 1 1'/></svg>")
    (run_dir / "final_sld.png").write_bytes(b"\x89PNG\r\n\x1a\n(final)")
    (run_dir / "metrics.json").write_text(json.dumps({"clip": 0.5, "aesthetic": 5.0}, indent=4))
    if args.save_video:
        (run_dir / "sketch.mp4").write_bytes(b"(video)")
    write_state(run_dir, epoch=epoch, num_iter=args.num_iter, stop_at=stop_at,
                phase="done", iters_per_sec=rate, resolved_caption=caption,
                latest_checkpoint=latest_checkpoint, latest_preview=latest_preview)
    print("\nDone!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
