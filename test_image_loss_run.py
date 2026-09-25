"""run() with --image-loss, the diffusion model stubbed (Spec 6 SS12.1, SS12.3).

Same stubs as test_run_segments.py, except that the fake target is a canvas
with a disc in it, so the in-run Canny derivation has edges to find. Proves:

  * without --image-loss the class is never constructed and nothing new is written;
  * with it, image_loss_target.png and one CSV row per epoch are written, with
    the schedule's alphas;
  * a stopped-then-resumed run equals the uninterrupted one: final_sld.svg
    byte-identical, image_loss_log.csv row for row (see logs_match);
  * resuming with a changed --image-loss-weight is refused, naming the field;
  * a supplied target of the wrong size is refused at parse time.

CPU only, no network. Run from the repo root:
    PYTHONPATH=. python test_image_loss_run.py
"""
import contextlib
import csv
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import SLDgen.run as run_module
import test_run_segments as segments
from SLDgen import config
from SLDgen.image_loss import alpha_at
from SLDgen.run import run

HORIZON = segments.HORIZON
RENDER_SIZE = segments.RENDER_SIZE


def fake_get_target(args):
    """A white canvas with a dark disc, and the attributes get_target sets."""
    args.original_target_path = args.target
    size = args.render_size
    yy, xx = np.mgrid[:size, :size]
    disc = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 4) ** 2
    rgb = np.full((size, size, 3), 255, np.uint8)
    rgb[disc] = 40
    args.input_image = Image.fromarray(rgb)
    mask = torch.zeros((size, size), dtype=torch.float32)
    mask[size // 8 : 7 * size // 8, size // 8 : 7 * size // 8] = 1.0
    args.mask = mask
    args.input_image.save(Path(args.output_dir) / "input.png")
    inputs = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return inputs, mask


def install_stubs():
    segments.install_stubs()
    run_module.get_target = fake_get_target


def make_args(name, extra=None):
    return segments.make_args(name, extra)


def fresh(name, extra=None):
    return segments.fresh(name, extra)


def quiet_run(args):
    segments.quiet_run(args)


def read_log(args):
    with open(Path(args.output_dir) / "image_loss_log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_default_path_never_constructs():
    """The opt-in guarantee: no --image-loss, no class, no files."""
    original = run_module.ImageFidelityLoss

    def refuse(*_a, **_k):
        raise AssertionError("ImageFidelityLoss constructed without --image-loss")

    run_module.ImageFidelityLoss = refuse
    try:
        args = fresh("imgloss_plain")
        quiet_run(args)
        out = Path(args.output_dir)
        passed = (
            (out / "final_sld.svg").exists()
            and not (out / "image_loss_target.png").exists()
            and not (out / "image_loss_log.csv").exists()
        )
    except AssertionError as exc:
        passed = False
        print(f"    {exc}")
    finally:
        run_module.ImageFidelityLoss = original
    print(f"[default path] never constructed, nothing written : {'PASS' if passed else 'FAIL'}")
    return passed


FLAGS = ["--image-loss", "--image-loss-weight", "0.3", "--image-loss-schedule", "decay",
         "--image-loss-curve-samples", "50"]


def test_image_loss_run():
    args = fresh("imgloss_on", FLAGS)
    quiet_run(args)
    out = Path(args.output_dir)
    rows = read_log(args)
    epochs = [int(r["epoch"]) for r in rows]
    alphas_ok = all(
        abs(float(r["alpha"]) - alpha_at(int(r["epoch"]), HORIZON, "decay", 0.3)) < 1e-12 for r in rows
    )
    target = np.asarray(Image.open(out / "image_loss_target.png"))
    blended = all(r["skipped"] == "0" and float(r["img_norm"]) > 0 and r["chamfer"] for r in rows)
    passed = (
        (out / "final_sld.svg").exists()
        and target.shape == (RENDER_SIZE, RENDER_SIZE)
        and target.max() == 255
        and epochs == list(range(HORIZON + 1))
        and alphas_ok
        and blended
    )
    print(
        f"[image-loss run] target={target.shape} rows={len(rows)} alphas={alphas_ok} "
        f"blended={blended} : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_image_loss_changes_trajectory():
    """Sanity: the term does something, i.e. the blend reaches the optimiser."""
    plain = (Path(make_args("imgloss_plain").output_dir) / "final_sld.svg").read_bytes()
    on = (Path(make_args("imgloss_on", FLAGS).output_dir) / "final_sld.svg").read_bytes()
    passed = plain != on
    print(f"[trajectory] --image-loss changes the drawing : {'PASS' if passed else 'FAIL'}")
    return passed


EXACT_COLUMNS = ("epoch", "alpha", "skipped")


def logs_match(a, b, rtol=1e-4):
    """Same rows and schedule exactly; measured values to a relative tolerance.

    Not byte identity: DiffVG's backward is not bit-deterministic, so even two
    uninterrupted runs differ in the last digits of the gradient norms.
    """
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
        for key in ra:
            if key in EXACT_COLUMNS or ra[key] == rb[key]:
                if ra[key] != rb[key]:
                    return False
                continue
            if not ra[key] or not rb[key]:
                return False
            x, y = float(ra[key]), float(rb[key])
            if abs(x - y) > rtol * max(abs(x), abs(y), 1e-12):
                return False
    return True


def test_segmented_equals_uninterrupted():
    reference = fresh("imgloss_reference", FLAGS)
    quiet_run(reference)
    ref_out = Path(reference.output_dir)

    segment = fresh("imgloss_resumed", FLAGS + ["--stop-at", "3"])
    quiet_run(segment)
    checkpoint = Path(segment.output_dir) / "checkpoints" / "ckpt_00003.pt"

    # A segment killed between checkpoint and exit leaves rows past the
    # checkpoint behind; the resumed segment must drop them.
    log = Path(segment.output_dir) / "image_loss_log.csv"
    with open(log, "a") as f:
        f.write("4,0.1,1,1,0,0,1,1,1,,\n")

    resumed = make_args("imgloss_resumed", FLAGS + ["--resume", str(checkpoint)])
    quiet_run(resumed)
    out = Path(resumed.output_dir)

    svg_same = (ref_out / "final_sld.svg").read_bytes() == (out / "final_sld.svg").read_bytes()
    csv_same = logs_match(read_log(reference), read_log(resumed))
    passed = svg_same and csv_same
    print(
        f"[resume] final_sld.svg identical={svg_same} image_loss_log.csv matches={csv_same} : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_resume_refuses_changed_weight():
    from SLDgen.checkpoint import CheckpointError

    args = fresh("imgloss_redirect", FLAGS + ["--stop-at", "3"])
    quiet_run(args)
    checkpoint = Path(args.output_dir) / "checkpoints" / "ckpt_00003.pt"
    changed = [f if f != "0.3" else "0.25" for f in FLAGS]
    redirected = make_args("imgloss_redirect", changed + ["--resume", str(checkpoint)])
    try:
        quiet_run(redirected)
        raised, message = False, "accepted"
    except CheckpointError as exc:
        raised, message = True, str(exc)
    passed = raised and "image_loss_weight" in message
    print(f"[redirect] changed --image-loss-weight refused : {'PASS' if passed else 'FAIL'}")
    return passed


def test_wrong_size_target_refused_early():
    path = Path("/tmp/claude-1000/-home-helge-SLDgen/imgloss_small.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64), np.uint8)).save(path)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            make_args("imgloss_badsize", ["--image-loss", "--image-loss-target", str(path)])
        refused = False
    except SystemExit as exc:
        refused = exc.code == 2
    passed = refused and "64x64" in err.getvalue()
    print(f"[bad size] refused at parse time with exit 2 : {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    install_stubs()
    ok = True
    for test in (
        test_default_path_never_constructs,
        test_image_loss_run,
        test_image_loss_changes_trajectory,
        test_segmented_equals_uninterrupted,
        test_resume_refuses_changed_weight,
        test_wrong_size_target_refused_early,
    ):
        ok = test() and ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
