"""End-to-end test: a real API and a real worker, both as live subprocesses.

Nothing here is mocked except SLDgen itself, which is replaced by
``test_support/fake_sldgen.py`` -- a stub that honours the parts of the contract
the worker depends on (the run-directory layout, ``state.json``, checkpoints,
``--resume``, graceful SIGTERM, exit codes) but needs no GPU. Everything else is
the shipping code: ``python -m sldgen_api`` under uvicorn, ``python -m
sldgen_worker`` with its flock and its real supervision loop, talking to each
other only through SQLite and the filesystem, exactly as they will in
production.

What that buys is coverage of the things that only exist between processes:
FIFO ordering, pause mid-segment, resume from a checkpoint written by a previous
process, crash recovery after the worker is killed, adoption of an orphaned
segment, and the failure taxonomy.

Run it with the service venv (it needs fastapi, uvicorn and httpx):
    PYTHONPATH=. .venv-service/bin/python test_service_e2e.py
"""

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
FAKE_SLDGEN = REPO_ROOT / "test_support" / "fake_sldgen.py"


def make_png(width, height):
    """A genuinely valid PNG, so the API's header parser is tested against a real one."""

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


PNG_WIDTH, PNG_HEIGHT = 48, 32
PNG = make_png(PNG_WIDTH, PNG_HEIGHT)

RESULTS = []


def check(label, condition, detail=""):
    passed = bool(condition)
    RESULTS.append((label, passed))
    print(f"  [{label}] {'PASS' if passed else 'FAIL'}{(' -- ' + detail) if detail else ''}",
          flush=True)
    return passed


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(predicate, timeout=30.0, interval=0.1, what="condition"):
    """Poll until true. Returns the value, or raises with what it last saw."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}; last value: {last!r}")


class Harness:
    """Boots the two daemons against a throwaway root, and tears them down."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="sldgen-e2e-"))
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.api = None
        self.worker = None
        self.client = httpx.Client(base_url=self.base, timeout=30.0)

    def env(self):
        env = os.environ.copy()
        env.update(
            {
                "SLDGEN_WORK_ROOT": str(self.root),
                # The worker spawns "SLDgen" with this interpreter and script.
                "SLDGEN_PYTHON": sys.executable,
                "SLDGEN_SCRIPT": str(FAKE_SLDGEN),
                "SLDGEN_POLL_INTERVAL": "0.1",
                "SLDGEN_CLAIM_INTERVAL": "0.2",
                "SLDGEN_GRACE_SECONDS": "10",
                "SLDGEN_API_HOST": "127.0.0.1",
                "SLDGEN_API_PORT": str(self.port),
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    def start_api(self):
        self.api = subprocess.Popen(
            [sys.executable, "-m", "sldgen_api"],
            cwd=str(REPO_ROOT), env=self.env(),
            stdout=open(self.root / "api.out", "w"), stderr=subprocess.STDOUT,
        )
        wait_for(self._api_ready, timeout=30, what="the API to answer /api/health")

    def _api_ready(self):
        try:
            return self.client.get("/api/health").status_code == 200
        except httpx.HTTPError:
            return False

    def restart_api(self):
        """Stop and start the API. Never touches a running job -- that is the
        whole reason the two units share only SQLite and the filesystem."""
        if self.api and self.api.poll() is None:
            self.api.terminate()
            self.api.wait(timeout=15)
        self.start_api()

    def start_worker(self):
        self.worker = subprocess.Popen(
            [sys.executable, "-m", "sldgen_worker"],
            cwd=str(REPO_ROOT), env=self.env(),
            stdout=open(self.root / "worker.out", "a"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for(
            lambda: self.client.get("/api/health").json()["worker_alive"],
            timeout=20, what="the worker to take its lock",
        )

    def stop_worker(self, sig=signal.SIGTERM, timeout=20):
        if self.worker is None or self.worker.poll() is not None:
            return self.worker.returncode if self.worker else None
        self.worker.send_signal(sig)
        try:
            return self.worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            return self.worker.wait(timeout=10)

    def kill_worker_hard(self):
        """SIGKILL: no cleanup, no lock release -- what a crash actually looks like."""
        if self.worker and self.worker.poll() is None:
            self.worker.kill()
            self.worker.wait(timeout=10)

    def close(self):
        self.stop_worker()
        if self.api and self.api.poll() is None:
            self.api.terminate()
            try:
                self.api.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.api.kill()
        self.client.close()
        shutil.rmtree(self.root, ignore_errors=True)

    # -- convenience ------------------------------------------------------

    def upload(self, payload=PNG, name="target.png"):
        response = self.client.post("/api/uploads", files={"file": (name, payload, "image/png")})
        response.raise_for_status()
        return response.json()

    def create_job(self, sha256, target_epoch=200, num_iter=1000, title="job", **params):
        body = {
            "title": title,
            "target_sha256": sha256,
            "target_epoch": target_epoch,
            "params": {"num_iter": num_iter, "save_interval": 50,
                       "checkpoint_interval": 100, **params},
        }
        response = self.client.post("/api/jobs", json=body)
        response.raise_for_status()
        return response.json()

    def job(self, job_id):
        response = self.client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def await_state(self, job_id, *states, timeout=60):
        return wait_for(
            lambda: (lambda j: j if j["state"] in states else None)(self.job(job_id)),
            timeout=timeout, what=f"job {job_id[:8]} to reach {states}",
        )

    def set_paused(self, paused):
        self.client.patch("/api/settings", json={"worker_paused": paused}).raise_for_status()

    def write_fault(self, job_id, fault):
        (self.root / "jobs" / job_id / "fault.json").write_text(json.dumps(fault))


# -- the tests -------------------------------------------------------------


def test_health_and_upload(harness):
    print("\n--- health, settings and uploads")
    health = harness.client.get("/api/health").json()
    check("health/ok", health["ok"] and health["db_ok"])
    check("health/worker-alive-via-flock", health["worker_alive"] is True)
    check("health/root-is-the-temp-root", health["root"] == str(harness.root))

    upload = harness.upload()
    check("upload/hashes-content", len(upload["sha256"]) == 64)
    check("upload/reads-png-dimensions-without-pillow",
          (upload["width"], upload["height"]) == (PNG_WIDTH, PNG_HEIGHT),
          f"{upload['width']}x{upload['height']}")
    again = harness.upload()
    check("upload/deduplicates", again["sha256"] == upload["sha256"])
    stored = list((harness.root / "uploads").glob(f"{upload['sha256']}.*"))
    check("upload/stored-once", len(stored) == 1)
    check("upload/served-back", harness.client.get(upload["url"]).content == PNG)
    return upload["sha256"]


def test_preview_run_then_promote(harness, sha256):
    """The headline workflow: a short preview, then promote to the full horizon."""
    print("\n--- preview run, then promote (the compare-then-promote workflow)")
    job = harness.create_job(sha256, target_epoch=200, num_iter=1000, title="preview")
    job_id = job["id"]
    check("job/created-queued", job["state"] == "queued")

    settled = harness.await_state(job_id, "waiting", "failed", timeout=90)
    check("preview/reaches-waiting-not-complete", settled["state"] == "waiting",
          f"{settled['state']} {settled.get('error_message', '')}")
    check("preview/current-epoch-is-the-budget", settled["current_epoch"] == 200,
          str(settled["current_epoch"]))
    check("preview/caption-resolved-from-heartbeat",
          settled["resolved_caption"] == "a single line drawing of a firefighter",
          str(settled["resolved_caption"]))

    run_dir = harness.root / "jobs" / job_id / "target" / "run"
    check("preview/checkpoint-written", (run_dir / "checkpoints" / "ckpt_00200.pt").exists())
    check("preview/latest-pt-is-a-copy",
          (run_dir / "checkpoints" / "latest.pt").exists()
          and not (run_dir / "checkpoints" / "latest.pt").is_symlink())
    # A segment that stops short must not finalise (Spec 1 SS4).
    check("preview/not-finalised",
          not (run_dir / "final_sld.svg").exists() and not (run_dir / "metrics.json").exists())
    check("preview/preview-image-served",
          harness.client.get(f"/api/jobs/{job_id}/preview").status_code == 200)

    segments = settled["segments"]
    check("preview/one-segment", len(segments) == 1)
    check("preview/segment-recorded-exit-0", segments[0]["exit_code"] == 0)
    check("preview/segment-records-argv",
          "--stop-at" in json.loads(segments[0]["argv_json"]))
    check("preview/first-segment-has-no-resume", segments[0]["resume_from"] is None)

    # Promote: more iterations of exactly what is already there.
    promoted = harness.client.post(f"/api/jobs/{job_id}/promote", json={"target_epoch": 1000})
    check("promote/accepted", promoted.status_code == 200)
    check("promote/requeues", promoted.json()["state"] == "queued")

    done = harness.await_state(job_id, "complete", "failed", timeout=120)
    check("promote/reaches-complete", done["state"] == "complete", str(done["state"]))
    check("promote/final-epoch-is-the-horizon", done["current_epoch"] == 1000)
    check("promote/finalised",
          (run_dir / "final_sld.svg").exists() and (run_dir / "metrics.json").exists())

    segments = done["segments"]
    check("promote/second-segment-resumed", len(segments) == 2
          and segments[1]["resume_from"] is not None, str(len(segments)))
    check("promote/second-segment-starts-where-the-first-stopped",
          segments[1]["start_epoch"] == 200, str(segments[1]["start_epoch"]))
    check("promote/frames-accumulate-across-segments",
          len(list((run_dir / "svg_to_png").glob("iter_*.png"))) == 21,
          str(len(list((run_dir / "svg_to_png").glob("iter_*.png")))))

    # Promotion beyond the horizon is refused: the horizon defines the schedule.
    refused = harness.client.post(f"/api/jobs/{job_id}/promote", json={"target_epoch": 5000})
    check("promote/beyond-horizon-is-409", refused.status_code == 409,
          str(refused.status_code))
    return job_id


def test_logs(harness, job_id):
    print("\n--- logs: cooking, byte offsets, per-segment, download")
    response = harness.client.get(f"/api/jobs/{job_id}/log", params={"segment": 1})
    check("log/served", response.status_code == 200)
    payload = response.json()
    check("log/header-has-argv", "argv:" in payload["text"])
    check("log/footer-has-exit-code", "exit code" in payload["text"])
    check("log/cooked-collapses-repaints", payload["text"].count("iterations") <= 3,
          f"{payload['text'].count('iterations')} progress lines survived cooking")
    check("log/not-running", payload["running"] is False)

    raw = harness.client.get(
        f"/api/jobs/{job_id}/log", params={"segment": 1, "raw": "true"}
    ).json()
    check("log/raw-keeps-every-repaint", raw["text"].count("iterations") > 20,
          str(raw["text"].count("iterations")))
    check("log/raw-and-cooked-share-file-offsets", raw["to"] == payload["to"])

    head = harness.client.get(
        f"/api/jobs/{job_id}/log", params={"segment": 1, "from": 0, "max_bytes": 40}
    ).json()
    rest = harness.client.get(
        f"/api/jobs/{job_id}/log", params={"segment": 1, "from": head["to"], "raw": "true"}
    ).json()
    check("log/incremental-fetch", head["to"] == 40 and rest["from"] == 40 and rest["eof"])

    check("log/segments-are-separate",
          harness.client.get(f"/api/jobs/{job_id}/log", params={"segment": 2}).json()["segment"] == 2)
    check("log/unknown-segment-404",
          harness.client.get(f"/api/jobs/{job_id}/log", params={"segment": 99}).status_code == 404)

    download = harness.client.get(f"/api/jobs/{job_id}/log/download", params={"segment": 1})
    check("log/download-is-raw-text",
          download.status_code == 200 and "attachment" in download.headers["content-disposition"])


def test_artifacts_zip_and_command(harness, job_id):
    print("\n--- artifacts, zip and reproduction command")
    artifacts = harness.client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
    names = {entry["name"] for entry in artifacts}
    check("artifacts/lists-the-result", {"final_sld.svg", "metrics.json"} <= names)
    check("artifacts/final-first", artifacts[0]["name"] == "final_sld.svg")
    check("artifacts/paths-are-job-relative",
          all(not entry["path"].startswith("/") for entry in artifacts))

    first_svg = next(entry for entry in artifacts if entry["name"] == "final_sld.svg")
    file_response = harness.client.get(f"/api/jobs/{job_id}/files/{first_svg['path']}")
    check("files/served", file_response.status_code == 200 and b"svg" in file_response.content)
    check("files/traversal-refused",
          harness.client.get(f"/api/jobs/{job_id}/files/../../../etc/passwd").status_code in (400, 404))

    archive = harness.client.get(f"/api/jobs/{job_id}/download.zip")
    check("zip/streamed", archive.status_code == 200 and archive.content[:2] == b"PK")
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        entries = bundle.namelist()
        check("zip/readable", bundle.testzip() is None)
        check("zip/excludes-checkpoints-by-default",
              not any("checkpoints" in name for name in entries))
        check("zip/contains-the-result", any(name.endswith("final_sld.svg") for name in entries))

    with_checkpoints = harness.client.get(
        f"/api/jobs/{job_id}/download.zip", params={"checkpoints": "true"}
    )
    with zipfile.ZipFile(io.BytesIO(with_checkpoints.content)) as bundle:
        check("zip/includes-checkpoints-on-request",
              any("checkpoints" in name for name in bundle.namelist()))

    command = harness.client.get(f"/api/jobs/{job_id}/command").text
    check("command/reproducible",
          "--target" in command and "--num-iter" in command and "--stop-at" in command)


def test_patch_rules(harness, job_id):
    print("\n--- PATCH: operational is editable, structural is not")
    response = harness.client.patch(f"/api/jobs/{job_id}", json={"params": {"seed": 42}})
    check("patch/structural-is-409", response.status_code == 409, str(response.status_code))
    body = response.json()
    check("patch/409-names-the-fields", body.get("structural_changes") == ["seed"])
    check("patch/409-points-at-run-again", "run-again" in body.get("run_again", ""))

    response = harness.client.patch(
        f"/api/jobs/{job_id}", json={"params": {"save_interval": 25}, "title": "renamed"}
    )
    check("patch/operational-accepted", response.status_code == 200)
    check("patch/operational-applied",
          response.json()["params"]["save_interval"] == 25
          and response.json()["title"] == "renamed")

    response = harness.client.patch(f"/api/jobs/{job_id}", json={"target_epoch": 99999})
    check("patch/target-epoch-bounded", response.status_code == 400)


def test_pause_and_resume(harness, sha256):
    print("\n--- pause mid-segment, then resume from the checkpoint")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=4000, num_iter=4000, title="long",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)

    running = harness.await_state(job_id, "running", timeout=30)
    check("pause/job-starts", running["state"] == "running")
    # Wait until it has actually made progress, so the pause lands mid-segment.
    wait_for(lambda: harness.job(job_id)["current_epoch"] > 60, timeout=60,
             what="the segment to pass epoch 60")

    response = harness.client.post(f"/api/jobs/{job_id}/pause")
    check("pause/accepted-202", response.status_code == 202)
    check("pause/only-asks-while-running", response.json()["state"] == "running"
          and response.json()["desired_state"] == "pause")

    paused = harness.await_state(job_id, "paused", timeout=60)
    epoch_at_pause = paused["current_epoch"]
    check("pause/worker-applies-the-transition", paused["state"] == "paused")
    check("pause/checkpointed-before-stopping", epoch_at_pause > 0, str(epoch_at_pause))
    check("pause/stopped-short-of-the-budget", epoch_at_pause < 4000)

    run_dir = harness.root / "jobs" / job_id / "target" / "run"
    check("pause/not-finalised", not (run_dir / "final_sld.svg").exists())

    # A paused job must not be picked up again until asked.
    time.sleep(1.0)
    check("pause/stays-paused", harness.job(job_id)["state"] == "paused")

    harness.write_fault(job_id, {"iter_delay": 0.0005})
    response = harness.client.post(f"/api/jobs/{job_id}/resume")
    check("resume/accepted-202", response.status_code == 202)
    done = harness.await_state(job_id, "complete", "failed", timeout=120)
    check("resume/runs-to-completion", done["state"] == "complete", str(done["state"]))
    check("resume/continued-from-the-checkpoint",
          done["segments"][1]["start_epoch"] == epoch_at_pause
          or done["segments"][1]["start_epoch"] == (epoch_at_pause // 50) * 50,
          f"resumed at {done['segments'][1]['start_epoch']}, paused at {epoch_at_pause}")
    return job_id


def test_failure_taxonomy(harness, sha256):
    print("\n--- failure classification and retry")
    cases = [
        ("environment", 3, "environment"),
        ("oom", 4, "oom"),
        ("validation", 2, "validation"),
        ("unknown", 7, "unknown"),
    ]
    job_ids = {}
    for label, exit_code, expected in cases:
        harness.set_paused(True)
        job = harness.create_job(sha256, target_epoch=100, num_iter=100, title=f"fail-{label}")
        harness.write_fault(job["id"], {"exit_code": exit_code})
        harness.set_paused(False)
        failed = harness.await_state(job["id"], "failed", "complete", timeout=60)
        check(f"failure/{label}-classified",
              failed["state"] == "failed" and failed["error_class"] == expected,
              f"{failed['state']}/{failed['error_class']}")
        check(f"failure/{label}-keeps-a-message", bool(failed["error_message"]))
        job_ids[label] = job["id"]

    # Retry the OOM one after clearing the fault: retryable, unchanged.
    job_id = job_ids["oom"]
    harness.set_paused(True)
    (harness.root / "jobs" / job_id / "fault.json").unlink()
    response = harness.client.post(f"/api/jobs/{job_id}/retry")
    check("retry/accepted", response.status_code == 200 and response.json()["state"] == "queued")
    check("retry/clears-the-error", response.json()["error_class"] is None)
    harness.set_paused(False)
    recovered = harness.await_state(job_id, "complete", "failed", timeout=60)
    check("retry/succeeds-after-the-cause-is-fixed", recovered["state"] == "complete",
          str(recovered["state"]))

    # A job that never started cannot be retried into existence from `complete`.
    response = harness.client.post(f"/api/jobs/{job_id}/retry")
    check("retry/refused-when-not-failed", response.status_code == 409)


def test_fifo_order(harness, sha256):
    print("\n--- FIFO within priority, one job at a time")
    harness.set_paused(True)
    first = harness.create_job(sha256, target_epoch=20, num_iter=20, title="fifo-1")
    second = harness.create_job(sha256, target_epoch=20, num_iter=20, title="fifo-2")
    jumper = harness.create_job(sha256, target_epoch=20, num_iter=20, title="fifo-jumper")
    harness.client.patch(f"/api/jobs/{jumper['id']}", json={"priority": 10}).raise_for_status()
    harness.set_paused(False)

    for job_id in (first["id"], second["id"], jumper["id"]):
        harness.await_state(job_id, "complete", "failed", timeout=90)

    order = sorted(
        (
            (harness.job(job["id"])["started_at"], harness.job(job["id"])["title"])
            for job in (first, second, jumper)
        )
    )
    started = [title for _stamp, title in order]
    # started_at has second resolution, so assert on the segment rows instead when
    # the timestamps tie.
    segment_order = []
    for job in (first, second, jumper):
        segments = harness.job(job["id"])["segments"]
        segment_order.append((segments[0]["id"], harness.job(job["id"])["title"]))
    segment_order.sort()
    check("queue/priority-runs-first", segment_order[0][1] == "fifo-jumper",
          str([title for _id, title in segment_order]))
    check("queue/fifo-for-equal-priority",
          [title for _id, title in segment_order][1:] == ["fifo-1", "fifo-2"],
          str(started))


def test_run_again_and_delete(harness, sha256):
    print("\n--- run-again variants, then delete")
    harness.set_paused(True)
    parent = harness.create_job(sha256, target_epoch=20, num_iter=1000, title="parent")
    response = harness.client.post(
        f"/api/jobs/{parent['id']}/run-again",
        json={"variants": [{"params": {"seed": 1}}, {"params": {"seed": 2}}]},
    )
    check("run-again/created", response.status_code == 201)
    children = response.json()["jobs"]
    check("run-again/two-children", len(children) == 2)
    check("run-again/share-a-batch", children[0]["batch_id"] == children[1]["batch_id"])
    check("run-again/point-at-the-parent",
          all(child["parent_job_id"] == parent["id"] for child in children))

    listed = harness.client.get("/api/jobs", params={"batch_id": children[0]["batch_id"]}).json()
    check("run-again/queryable-as-a-batch", len(listed["jobs"]) == 2)
    harness.set_paused(False)

    for child in children:
        harness.await_state(child["id"], "waiting", "complete", "failed", timeout=90)

    # Delete an idle job: the API reclaims it immediately, row and directory.
    victim = children[0]["id"]
    response = harness.client.delete(f"/api/jobs/{victim}")
    check("delete/accepted", response.status_code == 202)
    check("delete/row-removed", harness.client.get(f"/api/jobs/{victim}").status_code == 404)
    check("delete/directory-removed", not (harness.root / "jobs" / victim).exists())
    check("delete/sibling-untouched",
          harness.client.get(f"/api/jobs/{children[1]['id']}").status_code == 200)


def test_delete_running_job(harness, sha256):
    """Deleting a running job is the two-phase path: the worker must stop it first."""
    print("\n--- delete a running job")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=4000, num_iter=4000, title="doomed",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)

    harness.await_state(job_id, "running", timeout=30)
    wait_for(lambda: harness.job(job_id)["current_epoch"] > 20, timeout=60,
             what="the segment to make progress")
    segment_pid = harness.job(job_id)["segments"][-1]["pid"]

    response = harness.client.delete(f"/api/jobs/{job_id}")
    check("delete-running/accepted-202", response.status_code == 202)
    check("delete-running/not-reclaimed-while-running",
          response.json().get("state") != "deleted", str(response.json().get("state")))

    wait_for(lambda: harness.client.get(f"/api/jobs/{job_id}").status_code == 404,
             timeout=60, what="the worker to stop and reclaim the job")
    check("delete-running/row-eventually-removed",
          harness.client.get(f"/api/jobs/{job_id}").status_code == 404)
    check("delete-running/directory-removed", not (harness.root / "jobs" / job_id).exists())
    check("delete-running/segment-was-stopped", not _pid_alive(segment_pid))


def test_crash_recovery(harness, sha256):
    """Spec 2 SS8: kill the worker and its segment; the job must resume, not restart."""
    print("\n--- crash recovery: worker and segment killed together")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=4000, num_iter=4000, title="crashed",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)

    harness.await_state(job_id, "running", timeout=30)
    wait_for(lambda: harness.job(job_id)["current_epoch"] > 100, timeout=60,
             what="progress past epoch 100")
    segment = harness.job(job_id)["segments"][-1]
    child_pid = segment["pid"]
    check("crash/segment-pid-recorded", bool(child_pid))
    check("crash/boot-id-recorded", bool(segment["boot_id"]))

    harness.kill_worker_hard()
    os.kill(child_pid, signal.SIGKILL)
    time.sleep(0.5)
    check("crash/job-left-running-in-the-db", harness.job(job_id)["state"] == "running")

    checkpoints = sorted(
        (harness.root / "jobs" / job_id / "target" / "run" / "checkpoints").glob("ckpt_*.pt")
    )
    last_checkpoint_epoch = int(checkpoints[-1].stem.split("_")[-1])
    check("crash/a-checkpoint-survived", last_checkpoint_epoch >= 50,
          str(last_checkpoint_epoch))

    harness.write_fault(job_id, {"iter_delay": 0.0005})
    harness.start_worker()
    recovered = harness.await_state(job_id, "complete", "failed", timeout=180)
    check("crash/job-recovers", recovered["state"] == "complete", str(recovered["state"]))

    segments = recovered["segments"]
    check("crash/dead-segment-marked", segments[0]["exit_code"] == -1
          and segments[0]["error_class"] == "interrupted",
          f"{segments[0]['exit_code']}/{segments[0]['error_class']}")
    check("crash/resumed-from-the-checkpoint-not-zero",
          segments[1]["start_epoch"] == last_checkpoint_epoch,
          f"restarted at {segments[1]['start_epoch']}, checkpoint was {last_checkpoint_epoch}")
    check("crash/resume-flag-was-passed", segments[1]["resume_from"] is not None)


def test_adoption(harness, sha256):
    """Spec 2 SS8 step 4: a segment that outlived its worker is adopted, not duplicated."""
    print("\n--- crash recovery: worker killed, segment left alive (adoption)")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=600, num_iter=600, title="orphaned",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.02})
    harness.set_paused(False)

    harness.await_state(job_id, "running", timeout=30)
    wait_for(lambda: harness.job(job_id)["current_epoch"] > 50, timeout=60,
             what="progress past epoch 50")
    orphan_pid = harness.job(job_id)["segments"][-1]["pid"]

    harness.kill_worker_hard()
    check("adopt/segment-outlived-its-worker", _pid_alive(orphan_pid))

    harness.start_worker()
    time.sleep(1.0)
    segments_now = harness.job(job_id)["segments"]
    check("adopt/no-duplicate-segment-spawned", len(segments_now) == 1,
          f"{len(segments_now)} segments")
    check("adopt/original-pid-still-the-one-running",
          segments_now[-1]["pid"] == orphan_pid and _pid_alive(orphan_pid))

    settled = harness.await_state(job_id, "complete", "waiting", "failed", timeout=120)
    check("adopt/job-settles-correctly", settled["state"] == "complete", str(settled["state"]))
    check("adopt/one-segment-total", len(settled["segments"]) == 1)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_worker_shutdown_requeues(harness, sha256):
    """systemd stop must not lose GPU time: checkpoint, requeue, resume on restart."""
    print("\n--- graceful worker shutdown mid-segment")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=4000, num_iter=4000, title="shutdown",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)

    harness.await_state(job_id, "running", timeout=30)
    wait_for(lambda: harness.job(job_id)["current_epoch"] > 60, timeout=60,
             what="progress past epoch 60")

    code = harness.stop_worker(signal.SIGTERM, timeout=30)
    check("shutdown/worker-exits-cleanly", code == 0, str(code))
    settled = harness.job(job_id)
    check("shutdown/job-returned-to-the-queue", settled["state"] == "queued",
          str(settled["state"]))
    check("shutdown/progress-preserved", settled["current_epoch"] > 0,
          str(settled["current_epoch"]))
    check("shutdown/not-marked-failed", settled["error_class"] is None)

    harness.write_fault(job_id, {"iter_delay": 0.0005})
    harness.start_worker()
    done = harness.await_state(job_id, "complete", "failed", timeout=180)
    check("shutdown/resumes-after-restart", done["state"] == "complete", str(done["state"]))


def test_sse(harness, sha256):
    print("\n--- server-sent events")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=300, num_iter=300, title="sse",
                             save_interval=50, checkpoint_interval=100)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)
    harness.await_state(job_id, "running", timeout=30)

    events = []
    with harness.client.stream("GET", f"/api/jobs/{job_id}/events", timeout=90) as response:
        check("sse/content-type", response.headers["content-type"].startswith("text/event-stream"))
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if events and events[-1] == "settled":
                break
    check("sse/pushes-progress", "progress" in events, str(events[:5]))
    check("sse/terminates-with-settled", events[-1] == "settled")

    # The log stream must end when the segment does, not hang forever.
    chunks = []
    with harness.client.stream(
        "GET", f"/api/jobs/{job_id}/log/stream", params={"from": 0}, timeout=60
    ) as response:
        for line in response.iter_lines():
            if line.startswith("event:"):
                chunks.append(line.split(":", 1)[1].strip())
            if chunks and chunks[-1] == "end":
                break
    check("sse/log-stream-appends-then-ends", "append" in chunks and chunks[-1] == "end",
          str(chunks[:4]))


def test_disk_and_prune(harness):
    print("\n--- disk accounting and retention")
    disk = harness.client.get("/api/disk").json()
    check("disk/reports-total", disk["total_bytes"] > 0)
    check("disk/breaks-down-by-category", "jobs" in disk["by_category"])
    check("disk/lists-jobs", len(disk["by_job"]) > 0)

    complete = harness.client.get("/api/jobs", params={"state": "complete"}).json()["jobs"]
    if complete:
        job_id = complete[0]["id"]
        per_job = harness.client.get("/api/disk", params={"job_id": job_id}).json()
        check("disk/per-job-breakdown", per_job["total_bytes"] > 0
              and "checkpoints" in per_job["by_category"])

        before = len(list(
            (harness.root / "jobs" / job_id / "target" / "run" / "checkpoints").glob("ckpt_*.pt")
        ))
        result = harness.client.post("/api/maintenance/prune", json={}).json()
        after = len(list(
            (harness.root / "jobs" / job_id / "target" / "run" / "checkpoints").glob("ckpt_*.pt")
        ))
        check("prune/keeps-one-checkpoint", after == 1 and before >= 1, f"{before} -> {after}")
        check("prune/reports-what-it-freed", result["bytes_freed"] >= 0)
        check("prune/keeps-the-record",
              (harness.root / "jobs" / job_id / "target" / "run" / "final_sld.svg").exists())


def test_api_restart_does_not_disturb_the_worker(harness, sha256):
    """Spec 2 SS1: restarting the API must never touch a running job."""
    print("\n--- API restart while a job runs")
    harness.set_paused(True)
    job = harness.create_job(sha256, target_epoch=1000, num_iter=1000, title="api-restart",
                             checkpoint_interval=50, save_interval=50)
    job_id = job["id"]
    harness.write_fault(job_id, {"iter_delay": 0.01})
    harness.set_paused(False)
    harness.await_state(job_id, "running", timeout=30)
    pid_before = harness.job(job_id)["segments"][-1]["pid"]

    harness.api.terminate()
    harness.api.wait(timeout=15)
    time.sleep(0.5)
    check("api-restart/segment-survived-the-api-going-away", _pid_alive(pid_before))
    started = time.time()
    harness.start_api()
    check("api-restart/comes-back-fast", time.time() - started < 10.0,
          f"{time.time() - started:.1f}s")

    after = harness.job(job_id)
    check("api-restart/same-segment-still-running",
          after["segments"][-1]["pid"] == pid_before)
    done = harness.await_state(job_id, "complete", "failed", timeout=120)
    check("api-restart/job-completed-normally", done["state"] == "complete", str(done["state"]))


def test_singleton_worker(harness):
    """The flock is what guarantees exactly one worker, and thus one GPU consumer."""
    print("\n--- worker singleton")
    second = subprocess.Popen(
        [sys.executable, "-m", "sldgen_worker"],
        cwd=str(REPO_ROOT), env=harness.env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output, _ = second.communicate(timeout=30)
    check("singleton/second-worker-refuses-to-start", second.returncode == 1,
          str(second.returncode))
    check("singleton/says-why", "another worker" in output.lower(), output.strip()[:120])


def main():
    harness = Harness()
    try:
        harness.start_api()
        harness.start_worker()

        sha256 = test_health_and_upload(harness)
        test_singleton_worker(harness)
        job_id = test_preview_run_then_promote(harness, sha256)
        test_logs(harness, job_id)
        test_artifacts_zip_and_command(harness, job_id)
        test_patch_rules(harness, job_id)
        test_fifo_order(harness, sha256)
        test_pause_and_resume(harness, sha256)
        test_failure_taxonomy(harness, sha256)
        test_run_again_and_delete(harness, sha256)
        test_delete_running_job(harness, sha256)
        test_sse(harness, sha256)
        test_api_restart_does_not_disturb_the_worker(harness, sha256)
        test_worker_shutdown_requeues(harness, sha256)
        test_crash_recovery(harness, sha256)
        test_adoption(harness, sha256)
        test_disk_and_prune(harness)
    finally:
        try:
            worker_log = (harness.root / "worker.out").read_text()
        except OSError:
            worker_log = ""
        failures = [label for label, passed in RESULTS if not passed]
        if failures and worker_log:
            print("\n--- worker output (tail)\n" + "\n".join(worker_log.splitlines()[-40:]))
        harness.close()

    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    print("\nRESULT:", "ALL PASS" if not failures else "FAILURE")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
