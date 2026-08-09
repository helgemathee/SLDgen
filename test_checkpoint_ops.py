"""Fast isolated tests for the operational half of the checkpointing feature.

Companion to ``test_resume_geom.py``, which covers the numerical invariant. This
file covers everything a supervising process depends on:

  * parse-time validation of --stop-at / --resume / --checkpoint-interval
  * strict opt-in: none of the machinery engages without one of those flags
  * the ``state.json`` heartbeat (contents, atomicity, relative paths)
  * distinct exit codes, so a failure can be classified without reading a log
  * graceful SIGTERM, including the second-signal abort
  * non-destructive finalisation: exporting must not disturb the renderer

CPU only, no diffusion model. Run from the repo root:
    PYTHONPATH=. python test_checkpoint_ops.py
"""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch

from SLDgen import config
from SLDgen.checkpoint import (
    EXIT_ABORTED,
    EXIT_ENVIRONMENT,
    EXIT_OOM,
    EXIT_VALIDATION,
    CheckpointError,
    GracefulStop,
    checkpointing_enabled,
    classify_exception,
    load_checkpoint,
    relative_to_output,
    save_checkpoint,
    structural_fingerprint,
    write_state,
)

BASE_ARGS = [
    "--target", "./data/firefighter.png",
    "--use-cpu",
    "--render-size", "128",
    "--n-control-points", "40",
    "--sampling-rate", "100",
    "--num-iter", "200",
    "--init-method", "trefoil",
]


def parse(extra, name="ckptops"):
    """parse_arguments with stdout muted, since it prints device chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        return config.parse_arguments(BASE_ARGS + ["--experiment-name", name] + extra)


def expect_rejected(extra, name):
    """Return (rejected, message) for an invocation that must fail validation.

    argparse exits 2 on error, which is exactly the validation exit code, so the
    CLI contract and the parser agree by construction.
    """
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            parse(extra, name=name)
    except SystemExit as exc:
        return exc.code == EXIT_VALIDATION, stderr.getvalue()
    return False, "accepted (expected rejection)"


class StubRenderer:
    """Minimal painter stand-in: save_checkpoint only needs these four things."""

    canvas_width = 128
    canvas_height = 128

    def state_dict(self):
        return {
            "control_points": torch.zeros(4, 2),
            "weights": torch.ones(4),
            "width": torch.ones(4),
            "is_active_cp": torch.ones(4, dtype=torch.bool),
            "pinned_kind": "none",
            "pinned": {},
        }


class StubOptimizer:
    def state_dict(self):
        return {"state": {}, "param_groups": []}

    def param_group_names(self):
        return ["control_points", "weights"]


def write_stub_checkpoint(args, epoch):
    return save_checkpoint(StubRenderer(), StubOptimizer(), args, epoch, "a caption")


def test_opt_in():
    """With no checkpointing flag, none of this machinery may engage."""
    plain = parse([], name="ckptops_plain")
    engaged = parse(["--stop-at", "50"], name="ckptops_stop")
    interval = parse(["--checkpoint-interval", "25"], name="ckptops_interval")

    passed = (
        not checkpointing_enabled(plain)
        and checkpointing_enabled(engaged)
        and checkpointing_enabled(interval)
        and plain.stop_at is None
        and plain.resume is None
        and plain.checkpoint_interval == 0
    )
    print(f"[opt-in] defaults leave checkpointing off : {'PASS' if passed else 'FAIL'}")
    return passed


def test_validation_rules():
    """Every Spec 1 SS4 rule must be caught at parse time, not mid-run."""
    args = parse([], name="ckptops_source")
    ckpt = write_stub_checkpoint(args, 100)

    not_a_checkpoint = Path(args.output_dir) / "not_a_checkpoint.pt"
    torch.save({"hello": "world"}, not_a_checkpoint)

    cases = [
        ("--stop-at 0", ["--stop-at", "0"], "must be > 0"),
        ("--stop-at beyond horizon", ["--stop-at", "500"], "must be <= --num-iter"),
        ("negative --checkpoint-interval", ["--checkpoint-interval", "-1"], "must be >= 0"),
        ("--resume missing file", ["--resume", "/nonexistent/ckpt.pt"], "does not exist"),
        ("--resume non-checkpoint", ["--resume", str(not_a_checkpoint)],
         "not an SLDgen checkpoint"),
        # These two carry --init-method tsp so the rejection is genuinely about
        # --resume rather than about the initializer they also require.
        (
            "--resume + --init-points",
            ["--init-method", "tsp", "--resume", str(ckpt),
             "--init-points", "./data/firefighter.png"],
            "incompatible with --init-points",
        ),
        (
            "--resume + --stipple-weight",
            ["--init-method", "tsp", "--resume", str(ckpt),
             "--stipple-weight", "./data/firefighter.png"],
            "incompatible with --stipple-weight",
        ),
        (
            "--resume with --stop-at at the checkpoint's epoch",
            ["--resume", str(ckpt), "--stop-at", "100"],
            "must be greater than the checkpoint",
        ),
        (
            "--resume with --stop-at before the checkpoint",
            ["--resume", str(ckpt), "--stop-at", "50"],
            "must be greater than the checkpoint",
        ),
    ]

    ok = True
    for label, extra, expected in cases:
        rejected, message = expect_rejected(extra, name="ckptops_reject")
        passed = rejected and expected in message
        ok = ok and passed
        print(f"[validate] {label} rejected : {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"    got: {message.strip()[:200]}")

    # A completed trajectory cannot be extended by resuming it.
    done = write_stub_checkpoint(args, 200)
    rejected, message = expect_rejected(["--resume", str(done)], name="ckptops_done")
    passed = rejected and "is complete" in message
    ok = ok and passed
    print(f"[validate] resuming a finished trajectory rejected : {'PASS' if passed else 'FAIL'}")

    # ... and the valid case must still be accepted.
    accepted = parse(["--resume", str(ckpt), "--stop-at", "150"], name="ckptops_ok")
    passed = accepted.stop_at == 150 and accepted.resume == str(ckpt)
    ok = ok and passed
    print(f"[validate] a well-formed resume is accepted : {'PASS' if passed else 'FAIL'}")
    return ok


def test_checkpoint_roundtrip():
    """A checkpoint must carry everything needed to identify its trajectory."""
    args = parse(["--stop-at", "10"], name="ckptops_roundtrip")
    path = write_stub_checkpoint(args, 10)
    ckpt = load_checkpoint(path)

    latest = Path(args.output_dir) / "checkpoints" / "latest.pt"
    required = {
        "format_version", "epoch", "num_iter", "canvas", "control_points", "weights",
        "width", "is_active_cp", "pinned_kind", "pinned", "optimizer",
        "param_group_names", "rng", "resolved_caption", "structural_fingerprint",
        "target_sha256", "target_space",
    }
    missing = required - set(ckpt)
    # latest.pt must be a real copy, not a symlink: the run directory gets moved.
    copy_ok = latest.exists() and not latest.is_symlink()
    copy_matches = copy_ok and load_checkpoint(latest)["epoch"] == 10
    rng_ok = set(ckpt["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}

    passed = not missing and copy_matches and rng_ok and ckpt["epoch"] == 10
    print(
        f"[checkpoint] round-trips (missing={sorted(missing)}, latest.pt copy={copy_matches}, "
        f"rng={rng_ok}) : {'PASS' if passed else 'FAIL'}"
    )

    # An unreadable or foreign file must raise CheckpointError, not something the
    # caller has to guess at.
    bad = Path(args.output_dir) / "truncated.pt"
    bad.write_bytes(b"not a torch file")
    try:
        load_checkpoint(bad)
        raised = False
    except CheckpointError:
        raised = True
    print(f"[checkpoint] unreadable file raises CheckpointError : {'PASS' if raised else 'FAIL'}")
    return passed and raised


def test_fingerprint_ignores_operational_fields():
    """Operational flags must never enter the fingerprint, or nothing would resume."""
    args = parse(["--stop-at", "10"], name="ckptops_fp")
    fingerprint = structural_fingerprint(args)
    operational = {"save_interval", "checkpoint_interval", "verbose", "debug",
                   "output_dir", "experiment_name", "stop_at", "resume"}
    leaked = operational & set(fingerprint)

    # Init-only flags are excluded on purpose: --resume forbids them, so including
    # them would make any run that used them permanently unresumable.
    init_only = {"init_points", "stipple_weight", "stipple_weight_mode"} & set(fingerprint)

    # The caption must be the one the caller passed, not one BLIP-2 filled in later.
    args.caption = "a single line drawing of a firefighter"  # what create_caption() does
    passed = (
        not leaked
        and not init_only
        and structural_fingerprint(args)["caption"] == fingerprint["caption"]
        and "target_sha256" in fingerprint
    )
    print(
        f"[fingerprint] operational leaked={sorted(leaked)}, init-only leaked={sorted(init_only)}, "
        f"caption survives BLIP-2 mutation : {'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_state_json():
    """The heartbeat must be complete, relative, atomic and machine-readable."""
    args = parse(["--stop-at", "150"], name="ckptops_state")
    path = write_stub_checkpoint(args, 130)

    write_state(
        args,
        epoch=130,
        phase="optimizing",
        iters_per_sec=4.87531,
        latest_checkpoint=relative_to_output(args, path),
        latest_preview="svg_to_png/iter_0130.png",
        resolved_caption="a single line drawing of a firefighter",
    )
    with open(Path(args.output_dir) / "state.json") as f:
        state = json.load(f)

    expected_keys = {"epoch", "num_iter", "stop_at", "phase", "iters_per_sec",
                     "latest_checkpoint", "latest_preview", "resolved_caption", "updated_at"}
    fields_ok = set(state) == expected_keys
    values_ok = (
        state["epoch"] == 130
        and state["num_iter"] == 200
        and state["stop_at"] == 150
        and state["phase"] == "optimizing"
        and state["iters_per_sec"] == 4.875
        and state["latest_checkpoint"] == "checkpoints/ckpt_00130.pt"
    )
    # Relative, so the whole run directory stays movable.
    relative_ok = not Path(state["latest_checkpoint"]).is_absolute()
    # No temp file left behind; os.replace means a poller never sees a partial file.
    no_temp = not (Path(args.output_dir) / "state.json.tmp").exists()

    passed = fields_ok and values_ok and relative_ok and no_temp
    print(
        f"[state.json] fields={fields_ok} values={values_ok} relative={relative_ok} "
        f"atomic={no_temp} : {'PASS' if passed else 'FAIL'}"
    )

    # Phase transitions a supervisor switches on.
    phases = [write_state(args, 130, phase=p)["phase"] for p in
              ("init", "optimizing", "finalizing", "done")]
    phases_ok = phases == ["init", "optimizing", "finalizing", "done"]
    print(f"[state.json] phase transitions recorded : {'PASS' if phases_ok else 'FAIL'}")
    return passed and phases_ok


def test_exit_code_classification():
    """A supervisor treats OOM, bad credentials and bad flags completely differently."""
    cases = [
        (CheckpointError("fingerprint mismatch"), EXIT_VALIDATION, "checkpoint error"),
        (FileNotFoundError("no such file"), EXIT_VALIDATION, "missing file"),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), EXIT_OOM, "cuda oom"),
        (
            RuntimeError(
                "401 Client Error: Unauthorized for url https://huggingface.co/"
                "stabilityai/stable-diffusion-3.5-medium"
            ),
            EXIT_ENVIRONMENT,
            "hf auth",
        ),
        (OSError("stabilityai/stable-diffusion-3.5-medium is not a local folder and is not a "
                 "valid model identifier"), EXIT_ENVIRONMENT, "gated model"),
        (AssertionError("Torch not compiled with CUDA enabled"), EXIT_ENVIRONMENT, "no cuda"),
        (ModuleNotFoundError("No module named 'pydiffvg'"), EXIT_ENVIRONMENT, "missing dep"),
        (ValueError("something unexpected"), 1, "unknown"),
    ]

    ok = True
    for exc, expected, label in cases:
        got = classify_exception(exc)
        passed = got == expected
        ok = ok and passed
        print(f"[exit code] {label} -> {got} (want {expected}) : {'PASS' if passed else 'FAIL'}")

    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None:
        got = classify_exception(oom_type("out of memory"))
        passed = got == EXIT_OOM
        ok = ok and passed
        print(f"[exit code] torch.cuda.OutOfMemoryError -> {got} : {'PASS' if passed else 'FAIL'}")
    return ok


def test_graceful_stop():
    """SIGTERM must be observed between iterations, never mid-step."""
    stop = GracefulStop().install()
    try:
        completed = []
        for epoch in range(10):
            completed.append(epoch)
            if epoch == 3:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.05)  # let the handler run before the flag is read
            if stop.requested:
                break
    finally:
        stop.uninstall()

    # The iteration during which the signal arrived still finished: a checkpoint
    # written afterwards therefore describes a completed epoch.
    passed = completed == [0, 1, 2, 3] and stop.requested
    print(
        f"[sigterm] stops after finishing epoch 3 (ran {completed}) : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    # The handler must restore the previous disposition, so a plain run (which
    # never installs it) keeps the default behaviour.
    restored = signal.getsignal(signal.SIGTERM) is not stop._handle
    print(f"[sigterm] handler uninstalled cleanly : {'PASS' if restored else 'FAIL'}")
    return passed and restored


def test_second_sigterm_aborts():
    """A second SIGTERM means stop now: immediate exit, distinguishable code."""
    script = textwrap.dedent(
        """
        import os, signal, sys, time
        sys.path.insert(0, %r)
        from SLDgen.checkpoint import GracefulStop
        stop = GracefulStop().install()
        os.kill(os.getpid(), signal.SIGTERM)   # first: request a graceful stop
        assert stop.requested
        os.kill(os.getpid(), signal.SIGTERM)   # second: must not return
        time.sleep(5)
        sys.exit(0)
        """
    ) % str(Path.cwd())

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    passed = result.returncode == EXIT_ABORTED
    print(
        f"[sigterm] second signal exits {result.returncode} (want {EXIT_ABORTED}) : "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def test_finalize_is_non_destructive():
    """Exporting must leave the renderer describing the trajectory, not the export.

    Finalisation used to double the canvas, control points and widths in place, so
    a finished run could not be continued from its own end state and a checkpoint
    written afterwards would have been wrong by a factor of two.
    """
    from SLDgen.painter.painter import SLDBSplinePainter
    from SLDgen.run import finalize

    args = parse(["--stop-at", "10"], name="ckptops_finalize")
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=None)
    with contextlib.redirect_stdout(io.StringIO()):
        renderer.init_image()

    before = {
        "canvas_width": renderer.canvas_width,
        "canvas_height": renderer.canvas_height,
        "control_points": renderer.control_points.detach().clone(),
        "width": renderer.width.detach().clone(),
    }
    control_points_object = renderer.control_points

    with contextlib.redirect_stdout(io.StringIO()):
        finalize(renderer, args)

    same_values = (
        renderer.canvas_width == before["canvas_width"]
        and renderer.canvas_height == before["canvas_height"]
        and torch.equal(renderer.control_points, before["control_points"])
        and torch.equal(renderer.width, before["width"])
    )
    # Identity matters as much as value: the optimizer holds a reference to this
    # exact tensor, and `control_points * 2` would have replaced it with a
    # non-leaf copy the optimizer no longer updates.
    same_object = renderer.control_points is control_points_object
    exported = (Path(args.output_dir) / "final_sld.svg").exists() and (
        Path(args.output_dir) / "final_sld.png"
    ).exists()

    passed = same_values and same_object and exported
    print(
        f"[finalize] state preserved (values={same_values}, identity={same_object}) and "
        f"exported={exported} : {'PASS' if passed else 'FAIL'}"
    )

    # A checkpoint taken after finalisation must still describe the trajectory.
    path = save_checkpoint(renderer, StubOptimizer(), args, 10, "a caption")
    reloaded = load_checkpoint(path)
    scale_ok = torch.allclose(reloaded["control_points"], before["control_points"].cpu())
    print(f"[finalize] post-export checkpoint is unscaled : {'PASS' if scale_ok else 'FAIL'}")
    return passed and scale_ok


def main():
    ok = True
    for test in (
        test_opt_in,
        test_validation_rules,
        test_checkpoint_roundtrip,
        test_fingerprint_ignores_operational_fields,
        test_state_json,
        test_exit_code_classification,
        test_graceful_stop,
        test_second_sigterm_aborts,
        test_finalize_is_non_destructive,
    ):
        ok = test() and ok

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
