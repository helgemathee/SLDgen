"""The worker loop (Spec 2 SS7), crash recovery (SS8) and failure classification (SS15).

Scheduling policy is deliberately dull: **FIFO, one job at a time, no
preemption** (Spec 2 SS6). Four previews submitted together are delivered
back-to-back with zero context switches and zero model reloads; preemption would
*reduce* throughput here, because every switch costs a full pipeline load. When
a long run needs to yield, the answer is pause -- which puts that judgement with
the operator, where it belongs, instead of in a policy.
"""

import errno
import fcntl
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

from sldgen_service import jobs as job_files
from sldgen_service import logs as log_utils
from sldgen_service import store as store_module
from sldgen_service.config import REPO_ROOT
from sldgen_service.disk import directory_size
from sldgen_service.params import build_argv, split_by_group
from sldgen_service.store import Store

log = logging.getLogger("sldgen.worker")

#: Exit codes SLDgen promises (Spec 1 SS12 / Spec 2 SS2), mapped to the failure
#: taxonomy the UI switches on (Spec 2 SS15).
EXIT_CLASSES = {
    0: None,
    2: store_module.ERROR_VALIDATION,
    3: store_module.ERROR_ENVIRONMENT,
    4: store_module.ERROR_OOM,
    143: store_module.ERROR_INTERRUPTED,
}


def read_boot_id():
    """Identifies this boot, so "is that pid still mine?" survives a reboot."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # alive but owned by someone else
    return True


def is_sldgen_process(pid, script_name="sldgen.py"):
    """Guard against PID reuse: the pid must still be running our script."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return script_name in cmdline


def checkpoint_epoch(config, job_id):
    """The epoch a resume would actually start from.

    Read from checkpoint *filenames* rather than by loading a checkpoint, so the
    worker never imports torch -- it would cost seconds of start-up and hundreds
    of megabytes to learn a number that is already in the name.
    """
    directory = config.run_dir(job_id) / "checkpoints"
    if not directory.exists():
        return 0
    epochs = []
    for path in directory.glob("ckpt_*.pt"):
        try:
            epochs.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(epochs, default=0)


def read_state(config, job_id):
    """The segment's own heartbeat. Absent or half-written is normal; treat as unknown."""
    path = config.state_path(job_id)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Worker:
    def __init__(self, config):
        self.config = config
        self.store = Store(config)
        self.boot_id = read_boot_id()
        self._shutdown = False
        self._lock_handle = None

    # -- lifecycle --------------------------------------------------------

    def run(self):
        if not self._acquire_lock():
            log.error("another worker holds %s; exiting", self.config.lock_path)
            return 1
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        log.info("worker started (root=%s, boot=%s)", self.config.root, self.boot_id[:8])

        self.reconcile()
        try:
            while not self._shutdown:
                if self.store.get_settings().get("worker_paused"):
                    time.sleep(self.config.claim_interval)
                    continue
                job = self.store.claim_next_job()
                if job is None:
                    self._reap_deleting()
                    time.sleep(self.config.claim_interval)
                    continue
                log.info("claimed job %s (%s)", job["id"], job["title"] or "untitled")
                try:
                    self.run_job(job)
                except Exception:  # noqa: BLE001 - a bad job must not kill the queue
                    log.exception("job %s raised in the worker itself", job["id"])
                    self.store.finish_job(
                        job["id"],
                        store_module.FAILED,
                        store_module.ERROR_UNKNOWN,
                        "worker error; see the worker journal",
                    )
        finally:
            self._release_lock()
        log.info("worker stopped")
        return 0

    def _handle_shutdown(self, signum, frame):
        # Only sets a flag: the segment supervision loop reads it, signals the
        # child, and waits for its checkpoint. Exiting here would abandon the GPU
        # work this shutdown is supposed to preserve.
        self._shutdown = True
        log.info("shutdown requested (signal %s)", signum)

    def _acquire_lock(self):
        self.config.ensure_layout()
        self._lock_handle = open(self.config.lock_path, "w")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_handle.close()
            self._lock_handle = None
            return False
        self._lock_handle.write(f"{os.getpid()}\n")
        self._lock_handle.flush()
        return True

    def _release_lock(self):
        if self._lock_handle:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    # -- crash recovery (Spec 2 SS8) --------------------------------------

    def reconcile(self):
        """Decide what happened to every job that was `running` when we last stopped."""
        job_files.sweep_tmp(self.config)
        for job in self.store.list_jobs(state=store_module.RUNNING, limit=1000):
            segments = self.store.open_segments(job["id"])
            segment = segments[-1] if segments else None
            if segment is None:
                log.warning("job %s was running with no open segment; requeueing", job["id"])
                self._requeue_after_crash(job, None)
                continue

            same_boot = segment["boot_id"] == self.boot_id
            alive = same_boot and pid_alive(segment["pid"]) and is_sldgen_process(segment["pid"])
            if alive:
                log.info(
                    "adopting still-running segment %s of job %s (pid %s)",
                    segment["seq"],
                    job["id"],
                    segment["pid"],
                )
                self.supervise_adopted(job, segment)
            else:
                log.info(
                    "job %s died (boot %s, pid %s); resuming from its last checkpoint",
                    job["id"],
                    "same" if same_boot else "differs",
                    segment["pid"],
                )
                self._requeue_after_crash(job, segment)

    def _requeue_after_crash(self, job, segment):
        epoch = checkpoint_epoch(self.config, job["id"])
        if segment is not None:
            self.store.close_segment(
                segment["id"], -1, epoch, store_module.ERROR_INTERRUPTED
            )
        self.store.record_progress(job["id"], epoch)
        if job["desired_state"] == store_module.DESIRED_DELETE:
            self.store.update_job(job["id"], state=store_module.DELETING)
        else:
            self.store.update_job(
                job["id"], state=store_module.QUEUED, desired_state=store_module.DESIRED_RUN
            )

    # -- running one segment ----------------------------------------------

    def run_job(self, job):
        job_id = job["id"]
        start_epoch = checkpoint_epoch(self.config, job_id)
        target_epoch = int(job["target_epoch"])

        if start_epoch >= target_epoch:
            # Nothing to do: the job already reached its budget. Land it in the
            # right terminal state rather than spawning a segment SLDgen would
            # reject for having stop_at <= the checkpoint's epoch.
            self._settle(job_id, start_epoch, terminated_early=False)
            return

        resume = None
        if start_epoch > 0:
            latest = self.config.latest_checkpoint(job_id)
            resume = latest if latest.exists() else None
            if resume is None:
                log.warning("job %s has checkpoints but no latest.pt; restarting from 0", job_id)
                start_epoch = 0

        argv = build_argv(
            self.config.sldgen_python,
            self.config.sldgen_script,
            job["params"],
            target=self.config.job_inputs_dir(job_id) / "target.png",
            output_dir=self.config.job_dir(job_id),
            stop_at=target_epoch,
            resume=resume,
            root=self.config.root,
        )

        seq = len(self.store.list_segments(job_id)) + 1
        log_path = self.config.segment_log(job_id, seq)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        segment = self.store.open_segment(
            job_id,
            start_epoch=start_epoch,
            stop_at=target_epoch,
            argv=argv,
            resume_from=str(resume) if resume else None,
            operational_diff=self._operational_diff(job),
            log_path=str(log_path.relative_to(self.config.root)),
            boot_id=self.boot_id,
        )

        started = time.time()
        with open(log_path, "a", buffering=1, encoding="utf-8", errors="replace") as handle:
            log_utils.write_header(
                handle, seq, start_epoch, target_epoch, resume, argv, time.strftime("%FT%TZ")
            )
            process = subprocess.Popen(  # noqa: S603 - argv is built by params.build_argv
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=self._child_env(),
                start_new_session=True,  # own process group, so SIGTERM reaches the tree
            )
            self.store.update_segment(segment["id"], pid=process.pid)
            log.info("job %s segment %s pid %s: %s -> %s", job_id, seq, process.pid,
                     start_epoch, target_epoch)

            terminated_early = self._supervise(job_id, process)
            exit_code = process.wait()

            end_epoch = checkpoint_epoch(self.config, job_id)
            elapsed = time.time() - started
            iters = max(0, end_epoch - start_epoch)
            log_utils.write_footer(
                handle,
                exit_code,
                EXIT_CLASSES.get(exit_code, store_module.ERROR_UNKNOWN),
                end_epoch,
                elapsed,
                iters / elapsed if elapsed > 0 and iters else None,
            )

        error_class = self._classify(exit_code)
        self.store.close_segment(segment["id"], exit_code, end_epoch, error_class)
        self.store.record_progress(job_id, end_epoch)
        self.store.update_job(job_id, disk_bytes=directory_size(self.config.job_dir(job_id)))

        if exit_code == 0 or error_class == store_module.ERROR_INTERRUPTED:
            self._settle(job_id, end_epoch, terminated_early)
        else:
            message = self._failure_message(log_path, exit_code)
            log.warning("job %s failed (%s, exit %s)", job_id, error_class, exit_code)
            self.store.finish_job(job_id, store_module.FAILED, error_class, message)

    def _child_env(self):
        env = os.environ.copy()
        # Without this the log appears to hang: Python block-buffers stdout when
        # it is a pipe, and not every SLDgen print passes flush=True.
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TQDM_MININTERVAL", "1.0")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONPATH", str(REPO_ROOT))
        return env

    def _supervise(self, job_id, process):
        """Poll the heartbeat; signal the child when the API or systemd asks.

        Returns True if we terminated it rather than letting it reach its stop
        point. Polling ``state.json`` beats scraping tqdm: it survives the atomic
        replace, needs no inotify watch, and reports phase, which stderr cannot.
        """
        terminated = False
        while process.poll() is None:
            state = read_state(self.config, job_id)
            if state:
                self.store.record_progress(
                    job_id, state.get("epoch", 0), state.get("resolved_caption")
                )
            job = self.store.get_job(job_id)
            wants_stop = job is None or job["desired_state"] in (
                store_module.DESIRED_PAUSE,
                store_module.DESIRED_DELETE,
            )
            if wants_stop or self._shutdown:
                reason = "shutdown" if self._shutdown and not wants_stop else "request"
                log.info("stopping job %s segment gracefully (%s)", job_id, reason)
                self._terminate(process)
                terminated = True
                break
            time.sleep(self.config.poll_interval)
        return terminated

    def supervise_adopted(self, job, segment):
        """Watch a segment this worker did not spawn (Spec 2 SS8, step 4).

        It is not our child, so it cannot be reaped and there is no exit code to
        read; the outcome is taken from where the checkpoints ended up, which is
        the same thing the next segment would use.
        """
        job_id = job["id"]
        pid = segment["pid"]
        while pid_alive(pid) and is_sldgen_process(pid):
            state = read_state(self.config, job_id)
            if state:
                self.store.record_progress(
                    job_id, state.get("epoch", 0), state.get("resolved_caption")
                )
            current = self.store.get_job(job_id)
            if current is None or current["desired_state"] != store_module.DESIRED_RUN:
                self._signal_group(pid, signal.SIGTERM)
                break
            if self._shutdown:
                self._signal_group(pid, signal.SIGTERM)
                return
            time.sleep(self.config.poll_interval)

        deadline = time.time() + self.config.grace_seconds
        while pid_alive(pid) and time.time() < deadline:
            time.sleep(self.config.poll_interval)

        end_epoch = checkpoint_epoch(self.config, job_id)
        self.store.close_segment(segment["id"], None, end_epoch, None)
        self.store.record_progress(job_id, end_epoch)
        self._settle(job_id, end_epoch, terminated_early=False)

    def _terminate(self, process):
        """SIGTERM the group, wait out the grace period, then SIGKILL it.

        The grace period must cover an iteration plus a checkpoint write, and
        ideally a whole finalisation: interrupting metrics or ffmpeg mid-write
        leaves a corrupt artifact behind.
        """
        self._signal_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=self.config.grace_seconds)
            return
        except subprocess.TimeoutExpired:
            log.warning("segment pid %s ignored SIGTERM for %ss; killing",
                        process.pid, self.config.grace_seconds)
        self._signal_group(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.error("segment pid %s survived SIGKILL", process.pid)

    def _signal_group(self, pid, signum):
        try:
            os.killpg(os.getpgid(pid), signum)
        except OSError:
            try:
                os.kill(pid, signum)
            except OSError:
                pass

    def _classify(self, exit_code):
        if exit_code in EXIT_CLASSES:
            return EXIT_CLASSES[exit_code]
        if exit_code is not None and exit_code < 0:
            # Killed by a signal we sent: an interruption, not a failure.
            return store_module.ERROR_INTERRUPTED
        return store_module.ERROR_UNKNOWN

    def _settle(self, job_id, end_epoch, terminated_early):
        """Choose the terminal state for a segment that did not fail."""
        job = self.store.require_job(job_id)
        if job["desired_state"] == store_module.DESIRED_DELETE:
            self.store.update_job(job_id, state=store_module.DELETING)
            return
        if job["desired_state"] == store_module.DESIRED_PAUSE:
            self.store.update_job(
                job_id, state=store_module.PAUSED, finished_at=None
            )
            return
        if self._shutdown and terminated_early:
            # Not an error: the job returns to the queue and resumes from its
            # checkpoint when the worker comes back.
            self.store.update_job(job_id, state=store_module.QUEUED)
            return
        if end_epoch >= job["num_iter"]:
            self.store.finish_job(job_id, store_module.COMPLETE)
        elif end_epoch >= job["target_epoch"]:
            self.store.finish_job(job_id, store_module.WAITING)
        else:
            # Stopped short of its budget without being asked to: treat as
            # interrupted and let it pick up where it left off.
            self.store.update_job(job_id, state=store_module.QUEUED)

    def _failure_message(self, log_path, exit_code):
        """A short, quotable excerpt -- the UI shows this before the full log."""
        try:
            text = log_utils.tail(log_path, max_bytes=8192)["text"]
        except OSError:
            return f"segment exited {exit_code}"
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-8:]) or f"segment exited {exit_code}"

    def _operational_diff(self, job):
        """What changed operationally since the previous segment (Spec 2 SS4.4)."""
        segments = self.store.list_segments(job["id"])
        if not segments:
            return None
        from sldgen_service.params import argv_to_params

        try:
            previous = argv_to_params(json.loads(segments[-1]["argv_json"]))
        except (ValueError, KeyError):
            return None
        _, previous_operational = split_by_group(previous)
        _, current_operational = split_by_group(job["params"])
        diff = {
            name: {"from": previous_operational[name], "to": current_operational[name]}
            for name in current_operational
            if previous_operational.get(name) != current_operational[name]
        }
        return diff or None

    def _reap_deleting(self):
        """Reclaim directories for jobs the API marked `deleting` (Spec 2 SS10)."""
        for job in self.store.list_jobs(state=store_module.DELETING, limit=100):
            segments = self.store.open_segments(job["id"])
            if segments and pid_alive(segments[-1]["pid"]):
                continue
            log.info("reclaiming deleted job %s", job["id"])
            job_files.delete_job_files(self.config, job["id"])
            self.store.delete_row(job["id"])
