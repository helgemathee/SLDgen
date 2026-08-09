"""The API application (Spec 2 SS12).

Single user, no authentication, bound to the tailnet -- never 0.0.0.0 (see
``__main__``). Every endpoint is thin: it validates, calls into
``sldgen_service``, and serialises. The rules about what may change and when live
in ``sldgen_service.store`` and ``sldgen_service.params``, so the worker enforces
exactly the same ones.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from sldgen_service import disk as disk_utils
from sldgen_service import jobs as job_files
from sldgen_service import logs as log_utils
from sldgen_service import store as store_module
from sldgen_service.config import ServiceConfig
from sldgen_service.params import (
    OPERATIONAL_NAMES,
    ParamError,
    build_argv,
    canonical_params,
    split_by_group,
    structural_differences,
    validate_params,
)
from sldgen_service.store import Store, StoreError

from . import partitions as partition_utils
from .streaming import sse, stream_zip


def png_dimensions(payload):
    """Width and height from a PNG's IHDR, without importing an image library.

    The API venv stays lightweight on purpose; a 24-byte header read is a better
    trade than adding Pillow to it.
    """
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    if payload[12:16] != b"IHDR":
        return None, None
    return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")


def worker_alive(config):
    """True when something holds the worker's singleton lock.

    Asking the lock is better than looking for a process: it is the same
    mechanism the worker uses to guarantee singleton-ness, so it cannot disagree.
    """
    import fcntl

    if not config.lock_path.exists():
        return False
    try:
        with open(config.lock_path, "r+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def gpu_free_mb():
    """Diagnostic only (Spec 2 SS9): there is no VRAM guard, just this readout."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values = [int(line) for line in completed.stdout.split() if line.strip().isdigit()]
    return values[0] if values else None


def create_app(config=None):
    config = config or ServiceConfig.from_env()
    config.ensure_layout()
    store = Store(config)
    job_files.sweep_tmp(config)
    _reconcile_deleting(store, config)

    app = FastAPI(title="SLDgen", version="2.0", docs_url="/api/docs")
    app.state.config = config
    app.state.store = store

    # -- helpers ------------------------------------------------------------

    def require_job(job_id):
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"no such job: {job_id}")
        return job

    def job_summary(job):
        return {
            "id": job["id"],
            "title": job["title"],
            "state": job["state"],
            "desired_state": job["desired_state"],
            "num_iter": job["num_iter"],
            "target_epoch": job["target_epoch"],
            "current_epoch": job["current_epoch"],
            "progress": round(job["current_epoch"] / job["num_iter"], 4)
            if job["num_iter"]
            else 0.0,
            "resolved_caption": job["resolved_caption"],
            "target_sha256": job["target_sha256"],
            "parent_job_id": job["parent_job_id"],
            "batch_id": job["batch_id"],
            "priority": job["priority"],
            "error_class": job["error_class"],
            "error_message": job["error_message"],
            "disk_bytes": job["disk_bytes"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "preview_url": f"/api/jobs/{job['id']}/preview",
        }

    def job_detail(job):
        structural, operational = split_by_group(job["params"])
        state_file = config.state_path(job["id"])
        live_state = None
        if state_file.exists():
            try:
                live_state = json.loads(state_file.read_text())
            except (OSError, ValueError):
                live_state = None
        return {
            **job_summary(job),
            "params": job["params"],
            "structural_params": structural,
            "operational_params": operational,
            "segments": store.list_segments(job["id"]),
            "inputs": store.list_inputs(job["id"]),
            "artifacts": job_files.artifacts(config, job["id"]),
            "state_json": live_state,
            "command": " ".join(_command_for(job)),
        }

    def _command_for(job):
        current = job["current_epoch"]
        latest = config.latest_checkpoint(job["id"])
        return build_argv(
            config.sldgen_python,
            config.sldgen_script,
            job["params"],
            target=config.job_inputs_dir(job["id"]) / "target.png",
            output_dir=config.job_dir(job["id"]),
            stop_at=job["target_epoch"],
            resume=latest if (current > 0 and latest.exists()) else None,
            root=config.root,
        )

    def safe_job_file(job_id, relative):
        """Resolve a path inside a job directory, refusing anything that escapes it."""
        base = config.job_dir(job_id).resolve()
        candidate = (base / relative).resolve()
        if not str(candidate).startswith(str(base) + os.sep) and candidate != base:
            raise HTTPException(400, "path escapes the job directory")
        if not candidate.is_file():
            raise HTTPException(404, f"no such file: {relative}")
        return candidate

    def segment_log_path(job_id, segment_seq=None):
        segments = store.list_segments(job_id)
        if not segments:
            return None, None
        chosen = segments[-1]
        if segment_seq is not None:
            matches = [s for s in segments if s["seq"] == segment_seq]
            if not matches:
                raise HTTPException(404, f"job has no segment {segment_seq}")
            chosen = matches[0]
        path = (
            config.root / chosen["log_path"]
            if chosen["log_path"]
            else config.segment_log(job_id, chosen["seq"])
        )
        return chosen, path

    # -- health and settings ------------------------------------------------

    @app.get("/api/health")
    def health():
        try:
            store.connection.execute("SELECT 1").fetchone()
            db_ok = True
        except Exception:  # noqa: BLE001 - health must report, not raise
            db_ok = False
        return {
            "ok": db_ok,
            "worker_alive": worker_alive(config),
            "gpu_free_mb": gpu_free_mb(),
            "db_ok": db_ok,
            "root": str(config.root),
        }

    @app.get("/api/settings")
    def get_settings():
        return store.get_settings()

    @app.patch("/api/settings")
    def patch_settings(updates: dict = Body(...)):
        try:
            return store.update_settings(updates)
        except StoreError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- uploads ------------------------------------------------------------

    @app.post("/api/uploads")
    async def create_upload(file: UploadFile = File(...)):
        payload = await file.read()
        if not payload:
            raise HTTPException(400, "empty upload")
        suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
        digest, path = job_files.store_upload(config, payload, suffix=suffix)
        width, height = png_dimensions(payload)
        return {
            "sha256": digest,
            "width": width,
            "height": height,
            "bytes": len(payload),
            "filename": file.filename,
            "url": f"/api/uploads/{digest}",
        }

    @app.get("/api/uploads/{sha256}")
    def get_upload(sha256: str):
        matches = list(config.uploads_dir.glob(f"{sha256}.*"))
        if not matches:
            raise HTTPException(404, f"no upload with sha256 {sha256}")
        return FileResponse(matches[0])

    # -- jobs ---------------------------------------------------------------

    @app.get("/api/jobs")
    def list_jobs(
        state: str = Query(None),
        batch_id: str = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        cursor: str = Query(None),
    ):
        states = state.split(",") if state else None
        rows = store.list_jobs(state=states, batch_id=batch_id, limit=limit, cursor=cursor)
        return {
            "jobs": [job_summary(job) for job in rows],
            "next_cursor": rows[-1]["id"] if len(rows) == limit else None,
        }

    @app.post("/api/jobs", status_code=201)
    def create_job(body: dict = Body(...)):
        try:
            job = job_files.create_job(
                store,
                config,
                target_sha256=body["target_sha256"],
                params=body.get("params"),
                target_epoch=body.get("target_epoch"),
                title=body.get("title"),
                inputs=body.get("inputs") or (),
                parent_job_id=body.get("parent_job_id"),
                batch_id=body.get("batch_id"),
                priority=body.get("priority", 0),
            )
        except KeyError as exc:
            raise HTTPException(400, f"missing field: {exc.args[0]}") from exc
        except (job_files.JobError, ParamError, StoreError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return job_detail(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return job_detail(require_job(job_id))

    @app.patch("/api/jobs/{job_id}")
    def patch_job(job_id: str, body: dict = Body(...)):
        """Title, priority, target_epoch and the four operational settings only.

        A structural edit is a 409 pointing at /run-again, because a job's
        parameters and its result are one thing (Spec 2 SS4.2): a job that ran
        under two different structural sets could not be described by either.
        """
        job = require_job(job_id)
        fields = {}
        if "title" in body:
            fields["title"] = body["title"]
        if "priority" in body:
            fields["priority"] = int(body["priority"])
        if "target_epoch" in body:
            target = int(body["target_epoch"])
            if not 0 < target <= job["num_iter"]:
                raise HTTPException(
                    400, f"target_epoch must be in (0, num_iter={job['num_iter']}]"
                )
            fields["target_epoch"] = target

        params_update = body.get("params")
        if params_update:
            changed = structural_differences(job["params"], {**job["params"], **params_update})
            if changed:
                return JSONResponse(
                    status_code=409,
                    content={
                        "detail": (
                            "structural parameters are immutable once a job exists: "
                            f"{', '.join(changed)}. Use POST /api/jobs/{job_id}/run-again "
                            "to explore a variation."
                        ),
                        "structural_changes": changed,
                        "run_again": f"/api/jobs/{job_id}/run-again",
                    },
                )
            unknown = sorted(set(params_update) - OPERATIONAL_NAMES)
            if unknown:
                raise HTTPException(400, f"not operational parameters: {', '.join(unknown)}")
            try:
                merged = validate_params({**job["params"], **params_update})
            except ParamError as exc:
                raise HTTPException(400, str(exc)) from exc
            store.set_params(job_id, merged)

        if fields:
            store.update_job(job_id, **fields)
        return job_detail(require_job(job_id))

    @app.delete("/api/jobs/{job_id}", status_code=202)
    def delete_job(job_id: str):
        require_job(job_id)
        job = store.request_delete(job_id)
        if job["state"] == store_module.DELETING:
            # Not running: reclaim immediately rather than waiting for the worker
            # to notice. Move-then-unlink, so it is never half-visible.
            job_files.delete_job_files(config, job_id)
            store.delete_row(job_id)
            return {"id": job_id, "state": "deleted"}
        return job_summary(job)

    @app.post("/api/jobs/{job_id}/pause", status_code=202)
    def pause_job(job_id: str):
        require_job(job_id)
        try:
            return job_summary(store.request_pause(job_id))
        except StoreError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/resume", status_code=202)
    def resume_job(job_id: str):
        require_job(job_id)
        try:
            return job_summary(store.request_resume(job_id))
        except StoreError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/promote")
    def promote_job(job_id: str, body: dict = Body(...)):
        require_job(job_id)
        if "target_epoch" not in body:
            raise HTTPException(400, "promote requires target_epoch")
        try:
            return job_summary(store.promote(job_id, int(body["target_epoch"])))
        except StoreError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        require_job(job_id)
        try:
            return job_summary(store.retry(job_id))
        except StoreError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/run-again", status_code=201)
    def run_again(job_id: str, body: dict = Body(default={})):
        require_job(job_id)
        try:
            created = job_files.run_again(
                store, config, job_id, variants=body.get("variants"), batch_id=body.get("batch_id")
            )
        except (job_files.JobError, ParamError, StoreError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"jobs": [job_summary(job) for job in created]}

    # -- artifacts and files ------------------------------------------------

    @app.get("/api/jobs/{job_id}/artifacts")
    def list_artifacts(job_id: str):
        require_job(job_id)
        return {"artifacts": job_files.artifacts(config, job_id)}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str):
        require_job(job_id)
        return FileResponse(safe_job_file(job_id, path))

    @app.get("/api/jobs/{job_id}/preview")
    def get_preview(job_id: str):
        """The newest frame, whatever it is called.

        Reads the heartbeat first (it names the frame the segment just wrote) and
        falls back to scanning, so a preview is available even when a segment
        died before updating state.json.
        """
        require_job(job_id)
        state_path = config.state_path(job_id)
        if state_path.exists():
            try:
                latest = json.loads(state_path.read_text()).get("latest_preview")
            except (OSError, ValueError):
                latest = None
            if latest and (config.run_dir(job_id) / latest).exists():
                return FileResponse(config.run_dir(job_id) / latest)

        final = config.run_dir(job_id) / "final_sld.png"
        if final.exists():
            return FileResponse(final)
        frames = sorted((config.run_dir(job_id) / "svg_to_png").glob("iter_*.png"))
        if not frames:
            raise HTTPException(404, "job has no preview yet")
        return FileResponse(frames[-1])

    @app.get("/api/jobs/{job_id}/command", response_class=PlainTextResponse)
    def get_command(job_id: str):
        return " ".join(_command_for(require_job(job_id)))

    @app.get("/api/jobs/{job_id}/download.zip")
    def download_job(job_id: str, checkpoints: bool = Query(False)):
        """Streamed, generated on the fly.

        Checkpoints are excluded unless asked for: they are useless outside this
        service and would dominate the archive.
        """
        require_job(job_id)
        base = config.job_dir(job_id)
        entries = []
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            if not checkpoints and relative.parts[:1] == ("target",) and "checkpoints" in (
                relative.parts
            ):
                continue
            entries.append((f"{job_id}/{relative}", path))
        return StreamingResponse(
            stream_zip(entries),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
        )

    # -- logs ---------------------------------------------------------------

    @app.get("/api/jobs/{job_id}/log")
    def get_log(
        job_id: str,
        segment: int = Query(None),
        from_: int = Query(0, alias="from", ge=0),
        max_bytes: int = Query(log_utils.DEFAULT_MAX_BYTES, ge=1),
        raw: bool = Query(False),
    ):
        require_job(job_id)
        record, path = segment_log_path(job_id, segment)
        if record is None:
            return {"from": 0, "to": 0, "text": "", "eof": True, "size": 0, "running": False}
        payload = log_utils.read_range(path, start=from_, max_bytes=max_bytes, raw=raw)
        payload["running"] = record["finished_at"] is None
        payload["segment"] = record["seq"]
        return payload

    @app.get("/api/jobs/{job_id}/log/download", response_class=PlainTextResponse)
    def download_log(job_id: str, segment: int = Query(None)):
        require_job(job_id)
        record, path = segment_log_path(job_id, segment)
        if record is None or not Path(path).exists():
            raise HTTPException(404, "job has no log yet")
        return FileResponse(
            path,
            media_type="text/plain",
            filename=f"{job_id}_segment_{record['seq']:03d}.log",
        )

    @app.get("/api/jobs/{job_id}/log/stream")
    async def stream_log(
        job_id: str,
        request: Request,
        segment: int = Query(None),
        from_: int = Query(0, alias="from", ge=0),
        raw: bool = Query(False),
    ):
        require_job(job_id)
        record, path = segment_log_path(job_id, segment)
        if record is None:
            raise HTTPException(404, "job has no log yet")

        async def events():
            offset = from_
            while True:
                if await request.is_disconnected():
                    return
                payload = log_utils.read_range(path, start=offset, raw=raw)
                if payload["to"] > offset:
                    offset = payload["to"]
                    yield sse("append", payload)
                current = store.get_segment(record["id"])
                if current and current["finished_at"] is not None and payload["eof"]:
                    yield sse(
                        "end",
                        {"exit_code": current["exit_code"], "error_class": current["error_class"]},
                    )
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        """Push job progress. The detail endpoint is the polling fallback."""
        require_job(job_id)

        async def events():
            last = None
            while True:
                if await request.is_disconnected():
                    return
                job = store.get_job(job_id)
                if job is None:
                    yield sse("deleted", {"id": job_id})
                    return
                state_file = config.state_path(job_id)
                live = {}
                if state_file.exists():
                    try:
                        live = json.loads(state_file.read_text())
                    except (OSError, ValueError):
                        live = {}
                payload = {
                    "id": job_id,
                    "state": job["state"],
                    "desired_state": job["desired_state"],
                    "epoch": job["current_epoch"],
                    "target_epoch": job["target_epoch"],
                    "num_iter": job["num_iter"],
                    "phase": live.get("phase"),
                    "iters_per_sec": live.get("iters_per_sec"),
                    "resolved_caption": job["resolved_caption"],
                    "preview_url": f"/api/jobs/{job_id}/preview",
                    "error_class": job["error_class"],
                }
                if payload != last:
                    last = payload
                    yield sse("progress", payload)
                if job["state"] in (
                    store_module.COMPLETE,
                    store_module.FAILED,
                    store_module.WAITING,
                    store_module.PAUSED,
                ):
                    yield sse("settled", payload)
                    return
                await asyncio.sleep(1.0)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/logs/worker", response_class=PlainTextResponse)
    def worker_log(lines: int = Query(200, ge=1, le=10000)):
        """The worker's own journal: "why has nothing started?" without an SSH session."""
        try:
            completed = subprocess.run(
                ["journalctl", "-u", "sldgen-worker", "-n", str(lines), "--no-pager", "-o", "cat"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"journalctl is unavailable on this host ({exc})."
        if completed.returncode != 0:
            return (
                f"journalctl exited {completed.returncode}. The API user must be in the "
                f"systemd-journal group.\n{completed.stderr}"
            )
        return completed.stdout

    # -- partitions ---------------------------------------------------------

    @app.post("/api/partitions/preview")
    def partition_preview(body: dict = Body(...)):
        source_job_id = body.get("source_job_id")
        job = require_job(source_job_id) if source_job_id else None
        if job is None:
            raise HTTPException(400, "source_job_id is required")
        source_svg = config.run_dir(source_job_id) / body.get("source_svg", "final_sld.svg")
        params = dict(body.get("params") or {})
        strategy = body.get("strategy", "sequence")
        if strategy == "labelmap" and not params.get("labels"):
            default = partition_utils.default_labels(
                config, source_job_id, job["params"].get("condition", "depth")
            )
            if default is None:
                raise HTTPException(
                    400,
                    "strategy 'labelmap' needs a labels PNG, and this job has no "
                    "condition image to default to",
                )
            params["labels"] = str(default)

        output_dir = partition_utils.preview_dir(config, source_job_id)
        try:
            result = partition_utils.run_partition(
                config, source_svg, output_dir, int(body.get("n", 3)), strategy, params
            )
        except partition_utils.PartitionError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "source_job_id": source_job_id,
            "strategy": strategy,
            "n": int(body.get("n", 3)),
            "preview_url": f"/api/partitions/preview-{source_job_id}/files/{result['preview']}"
            if result["preview"]
            else None,
            "svg_urls": [
                f"/api/partitions/preview-{source_job_id}/files/{name}" for name in result["svgs"]
            ],
            "svgs": result["svgs"],
        }

    @app.post("/api/partitions", status_code=201)
    def commit_partition(body: dict = Body(...)):
        source_job_id = body.get("source_job_id")
        job = require_job(source_job_id) if source_job_id else None
        if job is None:
            raise HTTPException(400, "source_job_id is required")
        source_svg = config.run_dir(source_job_id) / body.get("source_svg", "final_sld.svg")
        params = dict(body.get("params") or {})
        strategy = body.get("strategy", "sequence")
        n = int(body.get("n", 3))
        if strategy == "labelmap" and not params.get("labels"):
            default = partition_utils.default_labels(
                config, source_job_id, job["params"].get("condition", "depth")
            )
            if default is not None:
                params["labels"] = str(default)

        partition = store.create_partition(
            source_job_id=source_job_id,
            source_svg=str(source_svg.relative_to(config.root))
            if str(source_svg).startswith(str(config.root))
            else str(source_svg),
            strategy=strategy,
            n=n,
            params=params,
            output_dir="",
        )
        output_dir = config.partition_dir(partition["id"])
        try:
            result = partition_utils.run_partition(
                config, source_svg, output_dir, n, strategy, params
            )
        except partition_utils.PartitionError as exc:
            store.connection.execute("DELETE FROM partitions WHERE id = ?", (partition["id"],))
            raise HTTPException(400, str(exc)) from exc
        store.connection.execute(
            "UPDATE partitions SET output_dir = ? WHERE id = ?",
            (str(output_dir.relative_to(config.root)), partition["id"]),
        )
        return {
            **store.get_partition(partition["id"]),
            "svgs": result["svgs"],
            "preview": result["preview"],
        }

    @app.get("/api/partitions")
    def list_partitions(source_job_id: str = Query(None)):
        return {"partitions": store.list_partitions(source_job_id)}

    @app.get("/api/partitions/{partition_id}/files/{name}")
    def partition_file(partition_id: str, name: str):
        base = config.partition_dir(partition_id).resolve()
        candidate = (base / name).resolve()
        if not str(candidate).startswith(str(base) + os.sep) or not candidate.is_file():
            raise HTTPException(404, f"no such partition file: {name}")
        return FileResponse(candidate)

    @app.get("/api/partitions/{partition_id}/download.zip")
    def download_partition(partition_id: str):
        base = config.partition_dir(partition_id)
        if not base.exists():
            raise HTTPException(404, f"no such partition: {partition_id}")
        entries = [
            (f"{partition_id}/{path.relative_to(base)}", path)
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]
        return StreamingResponse(
            stream_zip(entries),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{partition_id}.zip"'},
        )

    # -- disk and maintenance ------------------------------------------------

    @app.get("/api/disk")
    def get_disk(job_id: str = Query(None), recompute: bool = Query(False)):
        if job_id:
            require_job(job_id)
            breakdown = disk_utils.job_breakdown(config, job_id)
            store.update_job(job_id, disk_bytes=breakdown["total_bytes"])
            return breakdown
        if recompute:
            for job in store.list_jobs(limit=10_000):
                store.update_job(
                    job["id"], disk_bytes=disk_utils.directory_size(config.job_dir(job["id"]))
                )
        return disk_utils.root_breakdown(config, store)

    @app.post("/api/maintenance/prune")
    def prune(body: dict = Body(default={})):
        """Prune completed jobs (Spec 2 SS10). Never touches the reproducible record."""
        older_than_days = body.get("older_than_days")
        drop_frames = bool(body.get("drop_frames", False))
        cutoff = None
        if older_than_days is not None:
            cutoff = time.time() - float(older_than_days) * 86400

        pruned, freed = [], 0
        for job in store.list_jobs(state=store_module.COMPLETE, limit=10_000):
            if cutoff is not None:
                finished = job["finished_at"]
                if finished:
                    stamp = time.mktime(time.strptime(finished, "%Y-%m-%dT%H:%M:%SZ"))
                    if stamp > cutoff:
                        continue
            bytes_freed = job_files.prune_job(config, job["id"], drop_frames=drop_frames)
            if bytes_freed:
                pruned.append({"job_id": job["id"], "bytes_freed": bytes_freed})
                freed += bytes_freed
                store.update_job(
                    job["id"], disk_bytes=disk_utils.directory_size(config.job_dir(job["id"]))
                )
        return {"jobs_pruned": len(pruned), "bytes_freed": freed, "detail": pruned}

    @app.get("/api/maintenance/sweep")
    def sweep():
        removed = job_files.sweep_tmp(config)
        reclaimed = _reconcile_deleting(store, config)
        return {"tmp_entries_removed": removed, "jobs_reclaimed": reclaimed}

    return app


def _reconcile_deleting(store, config):
    """Finish deletions interrupted by a restart (Spec 2 SS8, API half)."""
    reclaimed = 0
    for job in store.list_jobs(state=store_module.DELETING, limit=1000):
        segments = store.open_segments(job["id"])
        if segments and segments[-1]["pid"]:
            continue  # the worker owns this one until its segment exits
        job_files.delete_job_files(config, job["id"])
        store.delete_row(job["id"])
        reclaimed += 1
    return reclaimed
