"""Disk accounting (Spec 2 SS10).

Deliberately cheap to call for the common case: per-job sizes are computed once
and cached in ``jobs.disk_bytes``, so the endpoint never walks the whole tree.
A recompute is explicit.
"""

import os
from pathlib import Path

#: Category -> subdirectory of a job's run directory. Everything else in a job
#: is counted as "other", which keeps the total honest when new artefacts appear.
RUN_CATEGORIES = ("checkpoints", "svg_logs", "svg_to_png", "weights_logs")


def directory_size(path):
    """Bytes used by a directory tree. Missing directories count as zero."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                # A file deleted mid-walk is normal while a job is running; the
                # figure is an estimate for a progress bar, not an audit.
                continue
    return total


def job_breakdown(config, job_id):
    """Per-category bytes for one job, plus its total."""
    job_dir = config.job_dir(job_id)
    run_dir = config.run_dir(job_id)
    categories = {name: directory_size(run_dir / name) for name in RUN_CATEGORIES}
    categories["inputs"] = directory_size(config.job_inputs_dir(job_id))
    categories["logs"] = directory_size(config.job_logs_dir(job_id))
    total = directory_size(job_dir)
    categories["other"] = max(0, total - sum(categories.values()))
    return {"job_id": job_id, "total_bytes": total, "by_category": categories}


def root_breakdown(config, store):
    """Whole-root accounting, using cached per-job sizes where available."""
    by_job = []
    jobs_total = 0
    for job in store.list_jobs(limit=10_000):
        size = job["disk_bytes"]
        if size is None:
            size = directory_size(config.job_dir(job["id"]))
        by_job.append({"job_id": job["id"], "title": job["title"], "bytes": size})
        jobs_total += size

    by_category = {
        "jobs": jobs_total,
        "uploads": directory_size(config.uploads_dir),
        "partitions": directory_size(config.partitions_dir),
        "tmp": directory_size(config.tmp_dir),
        "database": sum(
            directory_size(config.root / name)
            for name in ("sldgen.sqlite", "sldgen.sqlite-wal", "sldgen.sqlite-shm")
        ),
    }
    by_job.sort(key=lambda entry: entry["bytes"], reverse=True)
    return {
        "total_bytes": sum(by_category.values()),
        "by_category": by_category,
        "by_job": by_job,
    }
