"""Image-loss previews: the edge target and the landmarks (Spec 6 SS10).

Same shape as ``canny.py``: find a previous run of the same target, run a
root-level script over its canvas-space ``input.png`` under the conda
interpreter, write to ``work/tmp/``, serve the result. The API venv never
imports cv2, numpy or MediaPipe.

Unlike the Canny preview, both results are also **stored as uploads**
(content-addressed), so the client can attach one as an input role by its
``sha256`` without a second round trip.
"""

import json
import subprocess
from pathlib import Path

from sldgen_service import jobs as job_files

#: Knobs of the edge preview, mapped to ``sld_edge_target.py`` flags.
EDGE_FLAGS = {
    "low": "--low",
    "high": "--high",
    "blur": "--blur",
    "clahe_clip": "--clahe-clip",
    "clahe_grid": "--clahe-grid",
}

#: Knobs whose presence makes the map something the run cannot derive itself.
PREPARED_ONLY = ("clahe_clip", "roi", "preserve_silhouette")


class ImageLossError(ValueError):
    """A preview that cannot be produced, with the script's own message."""


def source_images(config, source_job_id):
    run_dir = config.run_dir(source_job_id)
    image = run_dir / "input.png"
    mask = run_dir / "mask.png"
    if not image.exists():
        raise ImageLossError(
            f"job {source_job_id} has no input.png yet -- a preview needs a run that has "
            "reached target preprocessing, because that is what defines canvas space"
        )
    return image, (mask if mask.exists() else None)


def edge_preview_path(config, source_job_id):
    """Stable per-source path, so a re-preview overwrites in place."""
    return config.tmp_dir / f"edge-{source_job_id}.png"


def derived_equivalent(params):
    """True when the run's own in-run derivation reproduces this map exactly."""
    for name in PREPARED_ONLY:
        value = params.get(name)
        if name == "clahe_clip":
            if value not in (None, "", 0, 0.0):
                return False
        elif value:
            return False
    return True


def build_edge_argv(config, image, mask, out_path, params):
    argv = [
        str(config.sldgen_python),
        str(config.edge_script),
        "--image",
        str(image),
        "--out",
        str(out_path),
    ]
    if mask is not None:
        argv += ["--mask", str(mask)]
    for name, flag in EDGE_FLAGS.items():
        if params.get(name) not in (None, ""):
            argv += [flag, str(params[name])]
    if params.get("roi"):
        argv += ["--roi"] + [str(float(value)) for value in params["roi"]]
    if params.get("preserve_silhouette"):
        argv.append("--preserve-silhouette")
    return argv


def _run(argv, timeout):
    completed = subprocess.run(  # noqa: S603 - argv is built here, not by a caller
        argv, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        # The script's last line is its own message (argparse error, or ours).
        lines = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise ImageLossError(
            lines[-1] if lines else f"{Path(argv[1]).name} exited {completed.returncode}"
        )
    return completed


def _edge_pixels(stdout):
    for line in stdout.splitlines():
        if line.startswith("edges"):
            digits = line.split()[1]
            if digits.isdigit():
                return int(digits)
    return None


def run_edge_preview(config, source_job_id, params, timeout=60):
    """Write the edge map, store it as an upload, and describe it."""
    image, mask = source_images(config, source_job_id)
    out_path = edge_preview_path(config, source_job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv = build_edge_argv(config, image, mask, out_path, params)
    completed = _run(argv, timeout)
    if not out_path.exists():
        raise ImageLossError("the edge script reported success but wrote no PNG")

    digest, _ = job_files.store_upload(config, out_path.read_bytes(), suffix=".png")
    return {
        "source_job_id": source_job_id,
        "sha256": digest,
        "edge_pixels": _edge_pixels(completed.stdout),
        "derived_equivalent": derived_equivalent(params),
        "argv": argv,
        "stdout": completed.stdout,
    }


def edge_preview_png(config, source_job_id):
    path = edge_preview_path(config, source_job_id)
    if not path.exists():
        raise ImageLossError(f"no edge preview has been generated for {source_job_id}")
    return Path(path)
