"""Database operations and the job state machine (Spec 2 SS5).

The concurrency rule that removes almost all the hazard between the two units:
**the API asks, the worker decides.** The API writes ``desired_state``; the
worker owns ``state`` for the job it is running.

With one refinement the spec already grants for deletion, and which is needed for
pause and resume to work when the worker is stopped: for a job that is *not*
running, the API may apply the state change itself, in the same transaction that
sets ``desired_state``. It can do that safely because "not running" is exactly
the condition under which no worker holds the job. Without this, pausing a queued
job would leave it queued-but-unclaimable with nothing to tell the user why.
"""

import json
import threading
from contextlib import contextmanager

from . import db
from .ids import new_ulid

# Job states (Spec 2 SS5).
QUEUED = "queued"
RUNNING = "running"
WAITING = "waiting"  # reached its budget, idle, pending a decision
PAUSED = "paused"  # the user intervened
COMPLETE = "complete"  # reached the horizon
FAILED = "failed"
DELETING = "deleting"

JOB_STATES = (QUEUED, RUNNING, WAITING, PAUSED, COMPLETE, FAILED, DELETING)
#: States in which no worker can be holding the job, so the API may transition it.
IDLE_STATES = (QUEUED, WAITING, PAUSED, COMPLETE, FAILED)

DESIRED_RUN = "run"
DESIRED_PAUSE = "pause"
DESIRED_DELETE = "delete"

ERROR_VALIDATION = "validation"
ERROR_ENVIRONMENT = "environment"
ERROR_OOM = "oom"
ERROR_INTERRUPTED = "interrupted"
ERROR_UNKNOWN = "unknown"


class StoreError(RuntimeError):
    """An operation the state machine does not permit."""


class Store:
    def __init__(self, config):
        self.config = config
        config.ensure_layout()
        db.initialize(config.db_path).close()
        self._local = threading.local()

    @property
    def connection(self):
        """One SQLite connection per thread.

        The API serves requests from a threadpool and sqlite3 forbids sharing a
        connection across threads. Per-thread connections are the honest fix --
        WAL already handles the concurrency between them, and between this
        process and the worker.
        """
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = db.connect(self.config.db_path)
            self._local.connection = connection
        return connection

    def close(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE, so two writers serialise instead of racing a read."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")

    # -- settings ---------------------------------------------------------

    def get_settings(self):
        rows = self.connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def update_settings(self, updates):
        known = set(db.DEFAULT_SETTINGS)
        unknown = sorted(set(updates) - known)
        if unknown:
            raise StoreError(f"unknown setting(s): {', '.join(unknown)}")
        with self.transaction() as connection:
            for key, value in updates.items():
                connection.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)),
                )
        return self.get_settings()

    # -- jobs -------------------------------------------------------------

    def create_job(
        self,
        params,
        target_sha256,
        target_epoch,
        title=None,
        parent_job_id=None,
        batch_id=None,
        priority=0,
        job_id=None,
    ):
        job_id = job_id or new_ulid()
        num_iter = int(params["num_iter"])
        if not 0 < target_epoch <= num_iter:
            raise StoreError(
                f"target_epoch must be in (0, num_iter={num_iter}]; got {target_epoch}"
            )
        now = db.utcnow()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO jobs (id, title, state, desired_state, params_json, target_sha256,
                                  num_iter, target_epoch, current_epoch, parent_job_id, batch_id,
                                  priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    title,
                    QUEUED,
                    DESIRED_RUN,
                    json.dumps(params, sort_keys=True),
                    target_sha256,
                    num_iter,
                    int(target_epoch),
                    parent_job_id,
                    batch_id,
                    int(priority),
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id):
        row = self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row)

    def require_job(self, job_id):
        job = self.get_job(job_id)
        if job is None:
            raise StoreError(f"no such job: {job_id}")
        return job

    def list_jobs(self, state=None, batch_id=None, limit=100, cursor=None):
        """Newest first. ``cursor`` is the last id of the previous page.

        Ids are ULIDs, so "created before" is just "sorts lower", and pagination
        needs no offset and no snapshot.
        """
        clauses, args = [], []
        if state:
            states = [state] if isinstance(state, str) else list(state)
            clauses.append(f"state IN ({','.join('?' * len(states))})")
            args += states
        if batch_id:
            clauses.append("batch_id = ?")
            args.append(batch_id)
        if cursor:
            clauses.append("id < ?")
            args.append(cursor)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM jobs {where} ORDER BY id DESC LIMIT ?", args + [int(limit)]
        ).fetchall()
        return [_job_from_row(row) for row in rows]

    def update_job(self, job_id, **fields):
        if not fields:
            return self.get_job(job_id)
        fields["updated_at"] = db.utcnow()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                list(fields.values()) + [job_id],
            )
        return self.get_job(job_id)

    def set_params(self, job_id, params):
        return self.update_job(job_id, params_json=json.dumps(params, sort_keys=True))

    # -- queue ------------------------------------------------------------

    def claim_next_job(self):
        """Atomically take the head of the queue, or return None.

        FIFO within priority (Spec 2 SS6). The SELECT and UPDATE share one
        IMMEDIATE transaction so a second worker -- which the flock should have
        prevented, but might exist during a restart overlap -- cannot claim the
        same job.
        """
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                 WHERE state = ? AND desired_state = ?
                 ORDER BY priority DESC, created_at ASC, id ASC
                 LIMIT 1
                """,
                (QUEUED, DESIRED_RUN),
            ).fetchone()
            if row is None:
                return None
            now = db.utcnow()
            connection.execute(
                "UPDATE jobs SET state = ?, started_at = COALESCE(started_at, ?), updated_at = ?, "
                "error_class = NULL, error_message = NULL WHERE id = ?",
                (RUNNING, now, now, row["id"]),
            )
        return self.get_job(row["id"])

    # -- transitions the API may request ----------------------------------

    def request_pause(self, job_id):
        job = self.require_job(job_id)
        if job["state"] in (COMPLETE, DELETING):
            raise StoreError(f"cannot pause a {job['state']} job")
        with self.transaction() as connection:
            if job["state"] in (QUEUED, WAITING):
                connection.execute(
                    "UPDATE jobs SET state = ?, desired_state = ?, updated_at = ? WHERE id = ?",
                    (PAUSED, DESIRED_PAUSE, db.utcnow(), job_id),
                )
            else:
                # Running: only ask. The worker signals the segment, waits for the
                # checkpoint, and sets `paused` itself.
                connection.execute(
                    "UPDATE jobs SET desired_state = ?, updated_at = ? WHERE id = ?",
                    (DESIRED_PAUSE, db.utcnow(), job_id),
                )
        return self.get_job(job_id)

    def request_resume(self, job_id):
        job = self.require_job(job_id)
        if job["state"] not in (PAUSED, WAITING, FAILED):
            raise StoreError(f"cannot resume a {job['state']} job")
        if job["current_epoch"] >= job["target_epoch"]:
            raise StoreError(
                f"job has already reached target_epoch {job['target_epoch']}; promote it instead"
            )
        return self._requeue(job_id)

    def _requeue(self, job_id):
        with self.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET state = ?, desired_state = ?, updated_at = ?, "
                "error_class = NULL, error_message = NULL, finished_at = NULL WHERE id = ?",
                (QUEUED, DESIRED_RUN, db.utcnow(), job_id),
            )
        return self.get_job(job_id)

    def promote(self, job_id, target_epoch):
        """Run more iterations of exactly what is already there.

        The only operation that extends a job. Everything else that changes what
        gets drawn is a new job, because every result-shaping parameter is
        structural (Spec 2 SS4.2).
        """
        job = self.require_job(job_id)
        target_epoch = int(target_epoch)
        if target_epoch > job["num_iter"]:
            raise StoreError(
                f"target_epoch {target_epoch} exceeds the job's horizon "
                f"num_iter={job['num_iter']}. The horizon defines the sparse-loss ramp "
                "for every iteration, so extending it means a new job, not a promotion."
            )
        if target_epoch <= job["current_epoch"]:
            raise StoreError(
                f"target_epoch {target_epoch} is not beyond the job's current epoch "
                f"{job['current_epoch']}"
            )
        if job["state"] not in (WAITING, PAUSED, COMPLETE, FAILED, QUEUED):
            raise StoreError(f"cannot promote a {job['state']} job")
        self.update_job(job_id, target_epoch=target_epoch)
        return self._requeue(job_id)

    def retry(self, job_id):
        job = self.require_job(job_id)
        if job["state"] != FAILED:
            raise StoreError(f"only failed jobs can be retried; this one is {job['state']}")
        return self._requeue(job_id)

    def request_delete(self, job_id):
        job = self.require_job(job_id)
        with self.transaction() as connection:
            state = DELETING if job["state"] in IDLE_STATES else job["state"]
            connection.execute(
                "UPDATE jobs SET state = ?, desired_state = ?, updated_at = ? WHERE id = ?",
                (state, DESIRED_DELETE, db.utcnow(), job_id),
            )
        return self.get_job(job_id)

    def delete_row(self, job_id):
        with self.transaction() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # -- transitions the worker owns --------------------------------------

    def finish_job(self, job_id, state, error_class=None, error_message=None):
        return self.update_job(
            job_id,
            state=state,
            error_class=error_class,
            error_message=error_message,
            finished_at=db.utcnow(),
        )

    def record_progress(self, job_id, current_epoch, resolved_caption=None):
        fields = {"current_epoch": int(current_epoch)}
        if resolved_caption:
            fields["resolved_caption"] = resolved_caption
        return self.update_job(job_id, **fields)

    # -- inputs -----------------------------------------------------------

    def add_input(
        self,
        job_id,
        role,
        stored_path,
        source_sha256,
        source_kind="upload",
        ordinal=0,
        source_job_id=None,
        source_partition_id=None,
    ):
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO job_inputs (job_id, role, ordinal, source_kind, source_job_id,
                                        source_partition_id, stored_path, source_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    role,
                    int(ordinal),
                    source_kind,
                    source_job_id,
                    source_partition_id,
                    str(stored_path),
                    source_sha256,
                ),
            )

    def list_inputs(self, job_id):
        rows = self.connection.execute(
            "SELECT * FROM job_inputs WHERE job_id = ? ORDER BY role, ordinal", (job_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- segments ---------------------------------------------------------

    def open_segment(
        self,
        job_id,
        start_epoch,
        stop_at,
        argv,
        resume_from=None,
        operational_diff=None,
        log_path=None,
        pid=None,
        boot_id=None,
    ):
        seq = (
            self.connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM segments WHERE job_id = ?", (job_id,)
            ).fetchone()["m"]
            + 1
        )
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO segments (job_id, seq, start_epoch, stop_at, resume_from, argv_json,
                                      operational_diff_json, pid, boot_id, log_path, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    seq,
                    int(start_epoch),
                    int(stop_at),
                    str(resume_from) if resume_from else None,
                    json.dumps(list(argv)),
                    json.dumps(operational_diff) if operational_diff else None,
                    pid,
                    boot_id,
                    str(log_path) if log_path else None,
                    db.utcnow(),
                ),
            )
            segment_id = cursor.lastrowid
        return self.get_segment(segment_id)

    def update_segment(self, segment_id, **fields):
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE segments SET {assignments} WHERE id = ?",
                list(fields.values()) + [segment_id],
            )
        return self.get_segment(segment_id)

    def close_segment(self, segment_id, exit_code, end_epoch, error_class=None):
        return self.update_segment(
            segment_id,
            exit_code=exit_code,
            end_epoch=end_epoch,
            error_class=error_class,
            finished_at=db.utcnow(),
        )

    def get_segment(self, segment_id):
        row = self.connection.execute(
            "SELECT * FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_segments(self, job_id):
        rows = self.connection.execute(
            "SELECT * FROM segments WHERE job_id = ? ORDER BY seq", (job_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def open_segments(self, job_id):
        rows = self.connection.execute(
            "SELECT * FROM segments WHERE job_id = ? AND finished_at IS NULL ORDER BY seq",
            (job_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # -- partitions -------------------------------------------------------

    def create_partition(self, source_job_id, source_svg, strategy, n, params, output_dir):
        partition_id = new_ulid()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO partitions (id, source_job_id, source_svg, strategy, n, params_json,
                                        output_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    partition_id,
                    source_job_id,
                    str(source_svg),
                    strategy,
                    int(n),
                    json.dumps(params, sort_keys=True),
                    str(output_dir),
                    db.utcnow(),
                ),
            )
        return self.get_partition(partition_id)

    def get_partition(self, partition_id):
        row = self.connection.execute(
            "SELECT * FROM partitions WHERE id = ?", (partition_id,)
        ).fetchone()
        if row is None:
            return None
        partition = dict(row)
        partition["params"] = json.loads(partition.pop("params_json"))
        return partition

    def list_partitions(self, source_job_id=None):
        if source_job_id:
            rows = self.connection.execute(
                "SELECT id FROM partitions WHERE source_job_id = ? ORDER BY id DESC",
                (source_job_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT id FROM partitions ORDER BY id DESC"
            ).fetchall()
        return [self.get_partition(row["id"]) for row in rows]


def _job_from_row(row):
    if row is None:
        return None
    job = dict(row)
    job["params"] = json.loads(job.pop("params_json"))
    return job
