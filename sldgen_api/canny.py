"""Canny-attraction preview (the parameters, before the GPU time).

``--attract-canny`` generates its SVG *inside* the run, which is the only place
canvas space exists (see ``SLDgen/canny_attract.py``). That is right for
correctness and wrong for iteration: you would not see what the thresholds did
until a job had been queued, claimed and started.

This module closes that loop by running the same code over a **previous run's**
``input.png`` and ``mask.png``, which are that run's canvas-space images and are
sitting on disk already. Same target and same ``--render-size`` /
``--object-size-ratio`` means the same canvas, so what the preview shows is what
the next run will generate.

Like ``partitions.py`` this is CPU-only, takes well under a second, and runs
synchronously in the API process; and like it, the work happens in a subprocess
under the conda interpreter, so cv2 and numpy stay out of the API venv.
"""

import subprocess
from pathlib import Path

#: Every knob the preview accepts, mapped to its CLI flag. Mirrors the
#: ``attract_canny_*`` parameters so the preview cannot drift from the run.
FLAGS = {
    "low": "--low",
    "high": "--high",
    "blur": "--blur",
    "simplify": "--simplify",
    "min_length": "--min-length",
    "max_points": "--max-points",
}


class CannyError(ValueError):
    """A preview that cannot be produced, with the script's own message."""


def preview_path(config, source_job_id):
    """Stable per-source path, so re-previewing overwrites in place.

    That is what makes the panel feel like a slider rather than a submission:
    the same URL updates as the thresholds move.
    """
    return config.tmp_dir / f"canny-{source_job_id}.svg"


def source_images(config, source_job_id):
    """The canvas-space image and mask a preview reads, or a reason it cannot."""
    run_dir = config.run_dir(source_job_id)
    image = run_dir / "input.png"
    mask = run_dir / "mask.png"
    if not image.exists():
        raise CannyError(
            f"job {source_job_id} has no input.png yet -- a preview needs a run that has "
            "reached target preprocessing, because that is what defines canvas space"
        )
    return image, (mask if mask.exists() else None)


def find_source_job(store, config, target_sha256):
    """The newest job on this target that has an input.png.

    Any run of the same target at the same render size produced the same canvas,
    so the newest usable one is as good as any and is the most likely to match
    what the user is about to submit.
    """
    for job in store.list_jobs(limit=200):
        if job["target_sha256"] != target_sha256:
            continue
        if (config.run_dir(job["id"]) / "input.png").exists():
            return job["id"]
    return None


def build_argv(config, image, mask, out_path, params):
    argv = [
        str(config.sldgen_python),
        str(config.canny_script),
        "--image",
        str(image),
        "--out",
        str(out_path),
    ]
    if mask is not None:
        argv += ["--mask", str(mask)]
    for name, flag in FLAGS.items():
        if params.get(name) is not None:
            argv += [flag, str(params[name])]
    if params.get("roi"):
        argv += ["--roi"] + [str(float(value)) for value in params["roi"]]
    return argv


def run_preview(config, source_job_id, params, timeout=60):
    """Generate the preview SVG and return where it is plus what it contains."""
    image, mask = source_images(config, source_job_id)
    out_path = preview_path(config, source_job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv = build_argv(config, image, mask, out_path, params)
    completed = subprocess.run(  # noqa: S603 - argv is built here, not by a caller
        argv, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise CannyError(
            (completed.stderr or completed.stdout or "").strip()
            or f"sld_canny_svg.py exited {completed.returncode}"
        )
    if not out_path.exists():
        raise CannyError("the preview script reported success but wrote no SVG")

    return {
        "source_job_id": source_job_id,
        "summary": _summary_line(completed.stdout),
        "points": _points(completed.stdout),
        "bytes": out_path.stat().st_size,
        "argv": argv,
        "stdout": completed.stdout,
    }


def _summary_line(stdout):
    """The script's own 'edges' line, which already reads as a sentence."""
    for line in stdout.splitlines():
        if line.startswith("edges"):
            return line.split(None, 1)[1].strip()
    return stdout.strip().splitlines()[0] if stdout.strip() else ""


def _points(stdout):
    """The point count, parsed back out so the UI can compare it to n_control_points."""
    summary = _summary_line(stdout)
    marker = "-> ~"
    if marker in summary:
        tail = summary.rsplit(marker, 1)[1]
        digits = ""
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)
    return None


def preview_svg(config, source_job_id):
    """The bytes of the last preview for a source job, for the file endpoint."""
    path = preview_path(config, source_job_id)
    if not path.exists():
        raise CannyError(f"no canny preview has been generated for {source_job_id}")
    return Path(path)
