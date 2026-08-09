"""Checkpointing, resume bookkeeping and process-control helpers (opt-in).

This module deliberately imports nothing beyond the standard library and torch.
The checkpoint machinery must be testable without loading diffusers, pydiffvg or
wiregrad, so ``test_checkpoint_ops.py`` runs in under a second.

Three concerns live here:

1. **Checkpoints** -- :func:`save_checkpoint` / :func:`load_checkpoint` plus the
   structural fingerprint that decides whether a checkpoint may be resumed at
   all (Spec 1 SS5, SS6).
2. **The ``state.json`` heartbeat** -- :func:`write_state`, a machine-readable
   progress file so a supervising process never has to scrape tqdm's stderr.
3. **Process control** -- :class:`GracefulStop` (checkpoint-on-SIGTERM) and
   :func:`classify_exception` (distinct exit codes so a supervisor can tell an
   OOM from a missing HuggingFace token without parsing logs).

Everything here is inert unless one of ``--stop-at`` / ``--resume`` /
``--checkpoint-interval`` was passed; see :func:`checkpointing_enabled`.
"""

import hashlib
import json
import os
import random
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

#: Bumped whenever the checkpoint layout changes incompatibly.
FORMAT_VERSION = 1

# Exit codes. A supervisor classifies a finished segment from these alone.
EXIT_OK = 0  # reached stop point, or stopped gracefully on SIGTERM
EXIT_VALIDATION = 2  # bad flags, missing file, unusable checkpoint
EXIT_ENVIRONMENT = 3  # HF auth / gated model / CUDA unavailable
EXIT_OOM = 4  # ran out of GPU (or host) memory
EXIT_ABORTED = 143  # second SIGTERM: killed mid-iteration, no checkpoint

#: Every run-shaping argument. Two runs that agree on all of these (and on the
#: target image bytes) are the same trajectory; any difference means a different
#: drawing, so resuming across one is refused rather than silently allowed.
#:
#: ``init_points``, ``stipple_weight`` and ``stipple_weight_mode`` are absent on
#: purpose: they only shape initialisation, initialisation does not re-run on
#: resume, and ``--resume`` refuses them at parse time. Including them would
#: make any run that used them unresumable.
STRUCTURAL_FIELDS = (
    "render_size",
    "n_control_points",
    "init_method",
    "seed",
    "num_iter",
    "optimize_cp_weights",
    "prune_low_weights",
    "width",
    "origin",
    "fixed_endpoints",
    "calligraphy",
    "object_size_ratio",
    "sampling_rate",
    "caption",
    "conditioning_scale",
    "condition",
    "lora_model",
    "lora_weight",
    "lr",
    "avoid",
    "attract",
    "avoidance_weight",
    "avoidance_distance",
    "attraction_weight",
    "attraction_distance",
    "repulsion_loss_weight",
    "sparse_loss_weight",
    "sparse_loss_type",
    "sparse_loss_progressive",
    "length_shortening_loss_weight",
)


class CheckpointError(ValueError):
    """Raised for an unusable checkpoint: bad version, or a fingerprint mismatch.

    Treated as a validation failure (``EXIT_VALIDATION``) because the caller
    asked for something impossible, not because the environment misbehaved.
    """


def checkpointing_enabled(args):
    """True when this invocation opted into the checkpointing machinery.

    With none of ``--stop-at`` / ``--resume`` / ``--checkpoint-interval`` set,
    nothing in this module runs and the output directory keeps exactly the file
    layout it had before the feature existed -- no ``checkpoints/``, no
    ``state.json``, no SIGTERM handler.
    """
    return (
        getattr(args, "stop_at", None) is not None
        or getattr(args, "resume", None) is not None
        or getattr(args, "checkpoint_interval", 0) > 0
    )


def target_sha256(path):
    """SHA-256 of a file's bytes, so a renamed or edited target is detected."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value):
    """Reduce an argparse value to something JSON- and equality-stable.

    Paths become strings, tuples become lists (argparse hands back lists, but a
    checkpoint round-trip through torch.save may not preserve the distinction),
    and anything exotic falls back to ``repr``.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def structural_fingerprint(args, target_hash=None):
    """Canonical dict of everything that shapes the trajectory.

    ``caption`` is read from ``args.raw_caption`` -- the caption *as passed on
    the command line*. ``SD3GuidanceControl.create_caption()`` mutates
    ``args.caption`` in place when it is empty, so fingerprinting the live
    attribute would compare a BLIP-2-derived caption against the ``""`` the
    caller passed on the next segment and fail every resume.
    """
    fp = {}
    for field in STRUCTURAL_FIELDS:
        if field == "caption":
            fp[field] = _canonical(getattr(args, "raw_caption", args.caption))
        else:
            fp[field] = _canonical(getattr(args, field, None))
    fp["target_sha256"] = target_hash if target_hash is not None else target_sha256(args.target)
    return fp


def fingerprint_differences(saved, current):
    """List of ``(field, saved_value, current_value)`` for every field that differs."""
    diffs = []
    for field in sorted(set(saved) | set(current)):
        old, new = saved.get(field, "<missing>"), current.get(field, "<missing>")
        if old != new:
            diffs.append((field, old, new))
    return diffs


def assert_fingerprint_matches(saved, current):
    """Raise :class:`CheckpointError` naming every mismatching field.

    Deliberately fatal rather than a warning: a trajectory silently redirected
    mid-flight produces a result its own recorded parameters cannot explain.
    """
    diffs = fingerprint_differences(saved, current)
    if not diffs:
        return
    lines = [
        "Cannot resume: the checkpoint was produced with different run-shaping "
        "arguments. Resume continues a trajectory; it cannot redirect one -- "
        "start a fresh run instead. Mismatching fields:"
    ]
    lines += [f"  {field}: checkpoint={old!r} current={new!r}" for field, old, new in diffs]
    raise CheckpointError("\n".join(lines))


def rng_state():
    """Snapshot every RNG that can influence a subsequent iteration.

    The SDS loss draws a diffusion timestep per iteration, so without this a
    resumed run diverges from an uninterrupted one even with identical geometry.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore what :func:`rng_state` captured.

    CUDA states are only restored when the device count matches; a checkpoint
    moved between machines keeps its CPU determinism rather than erroring.
    """
    random.setstate(_as_tuple(state["python"]))
    np.random.set_state(_as_tuple(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].to(torch.uint8).cpu())
    cuda_states = state.get("torch_cuda") or []
    if cuda_states and torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in cuda_states])


def _as_tuple(value):
    """torch.save round-trips ``random.getstate()``'s nested tuples as lists."""
    if isinstance(value, list):
        return tuple(_as_tuple(v) for v in value)
    return value


def checkpoint_dir(args):
    return Path(args.output_dir) / "checkpoints"


def save_checkpoint(renderer, optimizer, args, epoch, resolved_caption, target_hash=None):
    """Write ``checkpoints/ckpt_{epoch:05d}.pt`` and refresh ``latest.pt``.

    ``latest.pt`` is a copy rather than a symlink so the whole run directory can
    be moved or archived without dangling links.
    """
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "num_iter": int(args.num_iter),
        "canvas": {"width": renderer.canvas_width, "height": renderer.canvas_height},
        "optimizer": optimizer.state_dict(),
        "param_group_names": optimizer.param_group_names(),
        "rng": rng_state(),
        "resolved_caption": resolved_caption,
        "structural_fingerprint": structural_fingerprint(args, target_hash=target_hash),
        "target_sha256": (
            target_hash if target_hash is not None else target_sha256(args.target)
        ),
        # Recorded so downstream tooling can tell which coordinate space the
        # intermediate SVGs are in: when --object-size-ratio actually rescaled
        # the object, svg_logs/ lives in a different space than final_sld.svg.
        "target_space": {
            key: getattr(args, key, None)
            for key in ("scale_w", "scale_h", "original_center_x", "original_center_y")
        },
    }
    payload.update(renderer.state_dict())

    path = ckpt_dir / f"ckpt_{epoch:05d}.pt"
    torch.save(payload, path)
    shutil.copyfile(path, ckpt_dir / "latest.pt")
    return path


def load_checkpoint(path):
    """Load a checkpoint and reject anything this build cannot interpret."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"--resume checkpoint does not exist: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any load failure is a bad checkpoint
        raise CheckpointError(f"--resume checkpoint could not be read ({path}): {exc}") from exc
    if not isinstance(ckpt, dict) or "format_version" not in ckpt:
        raise CheckpointError(f"--resume file is not an SLDgen checkpoint: {path}")
    if ckpt["format_version"] != FORMAT_VERSION:
        raise CheckpointError(
            f"--resume checkpoint has format_version {ckpt['format_version']}, "
            f"this build understands {FORMAT_VERSION}: {path}"
        )
    return ckpt


def write_state(
    args,
    epoch,
    phase,
    iters_per_sec=None,
    latest_checkpoint=None,
    latest_preview=None,
    resolved_caption=None,
):
    """Write the ``state.json`` heartbeat atomically.

    Written at every ``--save-interval`` and every checkpoint. Atomic because a
    supervisor polls this file on a timer and must never read a half-written
    one. Paths are stored relative to the run directory so the directory stays
    relocatable.
    """
    state = {
        "epoch": int(epoch),
        "num_iter": int(args.num_iter),
        "stop_at": int(args.stop_at) if getattr(args, "stop_at", None) is not None else None,
        "phase": phase,
        "iters_per_sec": round(float(iters_per_sec), 3) if iters_per_sec else None,
        "latest_checkpoint": latest_checkpoint,
        "latest_preview": latest_preview,
        "resolved_caption": resolved_caption,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_dir = Path(args.output_dir)
    tmp = out_dir / "state.json.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=4)
    os.replace(tmp, out_dir / "state.json")
    return state


def relative_to_output(args, path):
    """Path relative to the run directory, for recording in ``state.json``."""
    try:
        return str(Path(path).relative_to(Path(args.output_dir)))
    except ValueError:
        return str(path)


class GracefulStop:
    """Checkpoint-on-SIGTERM: finish the iteration, then stop.

    A supervisor pauses a job by sending SIGTERM and waiting. The flag is only
    read between iterations, so the checkpoint written afterwards always
    describes a completed iteration rather than a half-applied optimizer step.

    A second SIGTERM means "stop arguing" and exits immediately with
    ``EXIT_ABORTED``; that path loses the current segment's uncheckpointed work,
    which is why it is distinguishable from the graceful exit code.
    """

    def __init__(self):
        self.requested = False
        self._count = 0
        self._previous = None

    def install(self):
        """Install the handler. A no-op off the main thread, where it is illegal."""
        try:
            self._previous = signal.signal(signal.SIGTERM, self._handle)
        except ValueError:
            self._previous = None
        return self

    def uninstall(self):
        if self._previous is not None:
            try:
                signal.signal(signal.SIGTERM, self._previous)
            except ValueError:
                pass
            self._previous = None

    def _handle(self, signum, frame):
        self._count += 1
        if self._count >= 2:
            print(
                "\nSecond SIGTERM received: aborting immediately without a checkpoint.",
                flush=True,
            )
            os._exit(EXIT_ABORTED)
        self.requested = True
        print(
            "\nSIGTERM received: finishing the current iteration, then writing a "
            "checkpoint and exiting. Send SIGTERM again to abort immediately.",
            flush=True,
        )


def classify_exception(exc):
    """Map an exception to an exit code a supervisor can act on.

    An OOM means "retry later or reduce the job"; an environment error means
    "the operator must fix credentials or drivers"; a validation error means
    "the request itself was wrong". Those need different handling, and none of
    them should require parsing a log to detect.
    """
    if isinstance(exc, CheckpointError):
        return EXIT_VALIDATION
    if isinstance(exc, FileNotFoundError):
        return EXIT_VALIDATION

    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return EXIT_OOM

    name = type(exc).__name__
    message = f"{name}: {exc}".lower()

    if "out of memory" in message or "cuda_error_out_of_memory" in message:
        return EXIT_OOM

    environment_markers = (
        "gatedrepoerror",
        "repositorynotfounderror",
        "hfhubhttperror",
        "localentrynotfounderror",
        "401 client error",
        "403 client error",
        "is not a local folder and is not a valid model identifier",
        "access to model",
        "awaiting a review",
        "you must be authenticated",
        "huggingface_hub.errors",
        "no cuda gpus are available",
        "cuda driver version",
        "found no nvidia driver",
        "torch not compiled with cuda",
    )
    if any(marker in message for marker in environment_markers):
        return EXIT_ENVIRONMENT
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return EXIT_ENVIRONMENT

    return 1
