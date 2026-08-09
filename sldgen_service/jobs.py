"""Job materialisation: the file work that accompanies a row in ``jobs``.

The rule this module exists to enforce is Spec 2 SS4.3: **inputs are copied, not
referenced.** When job B avoids job A's ``final_sld.svg``, the bytes are copied
into ``jobs/B/inputs/avoid_000.svg`` at submission time. B never reads from A's
directory, so deleting A cannot break or silently alter B, and B stays
reproducible. A few hundred KB per edge buys that.
"""

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from . import store as store_module
from .params import ParamError, canonical_params, validate_params

#: role -> (parameter it fills, whether the parameter is a list)
ROLE_PARAMS = {
    "avoid": ("avoid", True),
    "attract": ("attract", True),
    "init_points": ("init_points", False),
    "stipple_weight": ("stipple_weight", False),
    "labels": (None, False),  # partition input only; recorded for provenance
}

#: Roles whose meaning depends on the canvas coordinate space (Spec 1 SS7).
SPATIAL_ROLES = ("avoid", "attract", "init_points")


class JobError(ValueError):
    """A submission the service refuses, with a message meant for the user."""


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_upload(config, payload, suffix=".png"):
    """Content-address an upload. Identical bytes are stored once."""
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    path = config.uploads_dir / f"{digest}{suffix}"
    if not path.exists():
        # Write-then-rename so a concurrent reader never sees a partial upload.
        staging = config.uploads_dir / f".{digest}.{uuid.uuid4().hex}.part"
        staging.write_bytes(payload)
        os.replace(staging, path)
    return digest, path


def link_or_copy(source, destination):
    """Hardlink if the filesystem allows it, else copy.

    Uploads are immutable and content-addressed, so sharing an inode between the
    upload store and every job that uses it is free and safe.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    return destination


def resolve_input(config, store, spec):
    """Locate the bytes an input refers to, and refuse the ones that would mislead.

    ``spec`` is ``{role, source_kind, source_job_id|source_partition_id|sha256,
    path}``. ``path`` is relative to the source job's run directory (default
    ``final_sld.svg``) or to the partition directory.
    """
    role = spec.get("role")
    if role not in ROLE_PARAMS:
        raise JobError(f"unknown input role {role!r}; expected one of {sorted(ROLE_PARAMS)}")
    kind = spec.get("source_kind", "upload")

    if kind == "upload":
        digest = spec.get("sha256")
        if not digest:
            raise JobError(f"input role {role!r}: source_kind 'upload' needs a sha256")
        candidates = list(config.uploads_dir.glob(f"{digest}.*"))
        if not candidates:
            raise JobError(f"input role {role!r}: no upload with sha256 {digest}")
        return candidates[0], {"source_kind": kind}

    if kind == "job":
        source_job_id = spec.get("source_job_id")
        source_job = store.get_job(source_job_id) if source_job_id else None
        if source_job is None:
            raise JobError(f"input role {role!r}: no such source job {source_job_id!r}")
        relative = spec.get("path", "final_sld.svg")
        path = (config.run_dir(source_job_id) / relative).resolve()
        if not _within(path, config.run_dir(source_job_id)):
            raise JobError(f"input role {role!r}: path escapes the source job directory")
        if not path.exists():
            raise JobError(f"input role {role!r}: {relative} does not exist in job {source_job_id}")
        _guard_coordinate_space(config, source_job_id, role, relative)
        return path, {"source_kind": kind, "source_job_id": source_job_id}

    if kind == "partition":
        partition_id = spec.get("source_partition_id")
        partition = store.get_partition(partition_id) if partition_id else None
        if partition is None:
            raise JobError(f"input role {role!r}: no such partition {partition_id!r}")
        relative = spec.get("path")
        if not relative:
            raise JobError(f"input role {role!r}: a partition input needs a path")
        path = (config.partition_dir(partition_id) / relative).resolve()
        if not _within(path, config.partition_dir(partition_id)) or not path.exists():
            raise JobError(f"input role {role!r}: {relative} does not exist in {partition_id}")
        return path, {"source_kind": kind, "source_partition_id": partition_id}

    raise JobError(f"input role {role!r}: unknown source_kind {kind!r}")


def _within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _guard_coordinate_space(config, source_job_id, role, relative):
    """Spec 2 SS4.3: intermediate SVGs from a rescaled run are in a different space.

    ``increase_object_size`` applies only to the final export, so when
    ``--object-size-ratio`` actually rescaled the object, ``svg_logs/`` lives in a
    different coordinate space than ``final_sld.svg``. Feeding one to --avoid or
    --attract would misregister silently, which is far worse than a refusal.
    """
    if role not in SPATIAL_ROLES or Path(relative).name == "final_sld.svg":
        return
    config_path = config.run_dir(source_job_id) / "config.json"
    if not config_path.exists():
        return
    try:
        recorded = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return
    if recorded.get("scale_w") not in (None, "None") or recorded.get("scale_h") not in (
        None,
        "None",
    ):
        raise JobError(
            f"input role {role!r}: {relative} comes from a run whose object was rescaled "
            "(scale_w/scale_h are set in its config.json), so it is in a different "
            "coordinate space than final_sld.svg and would misregister. Use final_sld.svg."
        )


def create_job(
    store,
    config,
    target_sha256,
    params=None,
    target_epoch=None,
    title=None,
    inputs=(),
    parent_job_id=None,
    batch_id=None,
    priority=0,
):
    """Create a job: validate, lay out its directory, copy every input, insert the row.

    The row is written last. If anything fails, what is left behind is an
    orphaned directory under ``jobs/`` that no row points at -- harmless, and
    swept by maintenance -- rather than a job whose inputs are missing.
    """
    upload = config.upload_path(target_sha256)
    if not upload.exists():
        matches = list(config.uploads_dir.glob(f"{target_sha256}.*"))
        if not matches:
            raise JobError(f"no upload with sha256 {target_sha256}")
        upload = matches[0]

    params = validate_params(params or {})
    for name in ("avoid", "attract", "init_points", "stipple_weight"):
        if params[name] is not None:
            raise JobError(
                f"{name} may not be set directly in params; declare it as an input so the "
                "file is copied into the job and its provenance is recorded"
            )

    target_epoch = int(target_epoch if target_epoch is not None else params["num_iter"])
    job_id = store_module.new_ulid()

    inputs_dir = config.job_inputs_dir(job_id)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    config.job_logs_dir(job_id).mkdir(parents=True, exist_ok=True)
    link_or_copy(upload, inputs_dir / "target.png")

    resolved_inputs = []
    counters = {}
    for spec in inputs:
        source_path, provenance = resolve_input(config, store, spec)
        role = spec["role"]
        ordinal = counters.get(role, 0)
        counters[role] = ordinal + 1
        suffix = source_path.suffix or ".dat"
        param_name, is_list = ROLE_PARAMS[role]
        filename = f"{role}_{ordinal:03d}{suffix}" if is_list else f"{role}{suffix}"
        stored = inputs_dir / filename
        shutil.copyfile(source_path, stored)
        relative = str(stored.relative_to(config.root))
        resolved_inputs.append(
            {
                "role": role,
                "ordinal": ordinal,
                "stored_path": relative,
                "source_sha256": sha256_file(source_path),
                **provenance,
            }
        )
        if param_name:
            if is_list:
                params[param_name] = (params[param_name] or []) + [relative]
            else:
                params[param_name] = relative

    try:
        params = validate_params(params)
    except ParamError as exc:
        shutil.rmtree(config.job_dir(job_id), ignore_errors=True)
        raise JobError(str(exc)) from exc

    job = store.create_job(
        params=params,
        target_sha256=target_sha256,
        target_epoch=target_epoch,
        title=title,
        parent_job_id=parent_job_id,
        batch_id=batch_id,
        priority=priority,
        job_id=job_id,
    )
    for record in resolved_inputs:
        store.add_input(job_id, **record)
    return job


def run_again(store, config, source_job_id, variants=None, batch_id=None):
    """Fork a job's parameters into one or more fresh jobs (Spec 2 SS5).

    Permitted from **any** parent state, including running and failed, because it
    does not touch the parent: it re-uses the parent's target image and copies of
    its inputs, and starts from epoch 0 sharing no state.
    """
    source = store.require_job(source_job_id)
    variants = list(variants) if variants else [{}]
    batch_id = batch_id or store_module.new_ulid()

    source_inputs = [
        {
            "role": record["role"],
            "source_kind": "job",
            "source_job_id": source_job_id,
            "path": str(Path(record["stored_path"]).name),
        }
        for record in store.list_inputs(source_job_id)
    ]
    # The parent's inputs live in its inputs/ directory, not its run/ directory,
    # so they are copied directly rather than resolved through resolve_input.
    created = []
    for index, variant in enumerate(variants):
        overrides = dict(variant.get("params") or variant.get("params_overrides") or {})
        params = canonical_params({**source["params"], **overrides})
        for name in ("avoid", "attract", "init_points", "stipple_weight"):
            params[name] = None

        title = variant.get("title") or _variant_title(source, overrides, index)
        target_epoch = int(
            variant.get("target_epoch", min(source["target_epoch"], params["num_iter"]))
        )

        job = create_job(
            store,
            config,
            target_sha256=source["target_sha256"],
            params=params,
            target_epoch=target_epoch,
            title=title,
            inputs=(),
            parent_job_id=source_job_id,
            batch_id=batch_id,
            priority=variant.get("priority", source["priority"]),
        )
        _copy_parent_inputs(store, config, source_job_id, job["id"], source_inputs)
        created.append(store.get_job(job["id"]))
    return created


def _copy_parent_inputs(store, config, source_job_id, job_id, source_inputs):
    """Copy the parent's input files into the child and re-point its params."""
    if not source_inputs:
        return
    job = store.require_job(job_id)
    params = dict(job["params"])
    inputs_dir = config.job_inputs_dir(job_id)
    counters = {}
    for record in store.list_inputs(source_job_id):
        role = record["role"]
        param_name, is_list = ROLE_PARAMS[role]
        source_path = config.root / record["stored_path"]
        if not source_path.exists():
            continue
        ordinal = counters.get(role, 0)
        counters[role] = ordinal + 1
        suffix = source_path.suffix or ".dat"
        filename = f"{role}_{ordinal:03d}{suffix}" if is_list else f"{role}{suffix}"
        stored = inputs_dir / filename
        shutil.copyfile(source_path, stored)
        relative = str(stored.relative_to(config.root))
        store.add_input(
            job_id,
            role=role,
            ordinal=ordinal,
            stored_path=relative,
            source_sha256=record["source_sha256"],
            source_kind="job",
            source_job_id=source_job_id,
        )
        if param_name:
            if is_list:
                params[param_name] = (params[param_name] or []) + [relative]
            else:
                params[param_name] = relative
    store.set_params(job_id, params)


def _variant_title(source, overrides, index):
    base = source["title"] or "job"
    if overrides:
        changes = ", ".join(f"{name}={value}" for name, value in sorted(overrides.items()))
        return f"{base} ({changes})"
    return f"{base} (again {index + 1})"


def delete_job_files(config, job_id):
    """Move-then-unlink, so a half-deleted directory is never visible as a job."""
    job_dir = config.job_dir(job_id)
    if not job_dir.exists():
        return
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    staging = config.tmp_dir / f"delete-{job_id}-{uuid.uuid4().hex}"
    try:
        os.rename(job_dir, staging)
    except OSError:
        shutil.rmtree(job_dir, ignore_errors=True)
        return
    shutil.rmtree(staging, ignore_errors=True)


def sweep_tmp(config):
    """Clear staging directories left by an interrupted delete or download."""
    if not config.tmp_dir.exists():
        return 0
    removed = 0
    for entry in config.tmp_dir.iterdir():
        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(
            missing_ok=True
        )
        removed += 1
    return removed


KIND_BY_SUFFIX = {
    ".svg": "svg",
    ".png": "image",
    ".jpg": "image",
    ".mp4": "video",
    ".json": "json",
    ".log": "log",
    ".pt": "checkpoint",
    ".csv": "csv",
}


def artifacts(config, job_id):
    """Everything a job produced, newest-relevant first.

    Paths are relative to the job directory, which is exactly what the file
    endpoint takes, so a client never has to build one.
    """
    job_dir = config.job_dir(job_id)
    if not job_dir.exists():
        return []
    entries = []
    for path in sorted(job_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(job_dir)
        entries.append(
            {
                "name": path.name,
                "path": str(relative),
                "bytes": path.stat().st_size,
                "kind": KIND_BY_SUFFIX.get(path.suffix.lower(), "other"),
            }
        )
    priority = {"final_sld.svg": 0, "final_sld.png": 1, "sketch.mp4": 2, "metrics.json": 3}
    entries.sort(key=lambda entry: (priority.get(entry["name"], 9), entry["path"]))
    return entries


def prune_job(config, job_id, drop_frames=False):
    """Retention (Spec 2 SS10): keep the last checkpoint, optionally drop frames.

    Never touches ``final_sld.svg``, ``config.json``, ``metrics.json`` or
    ``state.json`` -- those are the reproducible record -- and never touches logs,
    which are exempt and kept for the life of the job.
    """
    run_dir = config.run_dir(job_id)
    freed = 0

    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    for path in checkpoints[:-1]:
        freed += path.stat().st_size
        path.unlink()

    if drop_frames and (run_dir / "sketch.mp4").exists():
        for path in (run_dir / "svg_to_png").glob("iter_*.png"):
            freed += path.stat().st_size
            path.unlink()
    return freed
