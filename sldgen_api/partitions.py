"""Partitioning (Spec 2 SS11).

Partitioning is CPU-only and takes seconds, so it runs **synchronously in the API
process** and never enters the GPU queue. ``sld_partition.py`` is a standalone
script with its own argparse, so it is invoked as a subprocess rather than
imported -- which also keeps numpy, scipy and matplotlib out of the API venv.

The interpreter used is the conda env's, by absolute path: matplotlib is needed
for ``--preview`` and lives there, while the API itself stays lightweight.
"""

import shutil
import subprocess
from pathlib import Path

STRATEGIES = ("horizontal", "vertical", "radial", "sequence", "cluster", "labelmap")


class PartitionError(ValueError):
    """A partition request that cannot be satisfied, with the script's own message."""


def preview_dir(config, source_job_id):
    """Stable per-source directory, so re-running overwrites in place.

    That is what makes strategy selection feel like scrubbing rather than
    submitting: the user changes N or the strategy and the same preview updates.
    """
    return config.partitions_dir / f"preview-{source_job_id}"


def build_argv(config, source_svg, output_dir, n, strategy, params, preview):
    argv = [
        str(config.sldgen_python),
        str(config.partition_script),
        "--input",
        str(source_svg),
        "--output-dir",
        str(output_dir),
        "--partitions",
        str(int(n)),
        "--strategy",
        strategy,
    ]
    if params.get("labels"):
        argv += ["--labels", str(params["labels"])]
    if params.get("origins"):
        origins = params["origins"]
        flat = [value for pair in origins for value in pair] if (
            origins and isinstance(origins[0], (list, tuple))
        ) else list(origins)
        if len(flat) != 2 * int(n):
            raise PartitionError(
                f"origins must supply 2*N = {2 * int(n)} numbers for N={n}, got {len(flat)}"
            )
        argv += ["--origins"] + [str(float(value)) for value in flat]
    if params.get("connect_tails"):
        argv.append("--connect-tails")
    if params.get("sample_spacing") is not None:
        argv += ["--sample-spacing", str(float(params["sample_spacing"]))]
    if params.get("seed") is not None:
        argv += ["--seed", str(int(params["seed"]))]
    if preview:
        argv.append("--preview")
    return argv


def default_labels(config, source_job_id, condition):
    """labelmap's natural default: the source run's own condition image.

    It is already persisted in canvas space, so it registers with the master SVG
    without any transformation.
    """
    candidate = config.run_dir(source_job_id) / f"condition_{condition}.png"
    return candidate if candidate.exists() else None


def run_partition(config, source_svg, output_dir, n, strategy, params, preview=True, timeout=300):
    if strategy not in STRATEGIES:
        raise PartitionError(f"unknown strategy {strategy!r}; expected one of {list(STRATEGIES)}")
    source_svg = Path(source_svg)
    if not source_svg.exists():
        raise PartitionError(f"source SVG does not exist: {source_svg}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        # Overwrite in place, but do not leave last run's extra partitions behind
        # when N shrinks -- the caller would silently get too many SVGs.
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = build_argv(config, source_svg, output_dir, n, strategy, params, preview)
    completed = subprocess.run(  # noqa: S603 - argv is built here, not by a caller
        argv, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise PartitionError(
            (completed.stderr or completed.stdout or "").strip()
            or f"sld_partition.py exited {completed.returncode}"
        )

    svgs = sorted(output_dir.glob("partition_*.svg"))
    preview_png = output_dir / "partition_preview.png"
    return {
        "output_dir": output_dir,
        "svgs": [path.name for path in svgs],
        "preview": preview_png.name if preview_png.exists() else None,
        "labels": "labels.png" if (output_dir / "labels.png").exists() else None,
        "argv": argv,
        "stdout": completed.stdout,
    }
