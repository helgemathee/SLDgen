"""SQLite connection handling and schema (Spec 2 SS4).

WAL mode is what makes the two-process design work: the worker writes job state
while the API reads it, with no lock contention and no shared memory beyond the
database file itself. ``busy_timeout`` covers the brief writer overlap that WAL
does not eliminate.
"""

import json
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id                TEXT PRIMARY KEY,
  title             TEXT,
  state             TEXT NOT NULL,
  desired_state     TEXT NOT NULL,
  params_json       TEXT NOT NULL,
  target_sha256     TEXT NOT NULL,
  num_iter          INTEGER NOT NULL,
  target_epoch      INTEGER NOT NULL,
  current_epoch     INTEGER NOT NULL DEFAULT 0,
  resolved_caption  TEXT,
  parent_job_id     TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  batch_id          TEXT,
  priority          INTEGER NOT NULL DEFAULT 0,
  error_class       TEXT,
  error_message     TEXT,
  disk_bytes        INTEGER,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL,
  started_at        TEXT,
  finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(state, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);

CREATE TABLE IF NOT EXISTS partitions (
  id            TEXT PRIMARY KEY,
  source_job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  source_svg    TEXT NOT NULL,
  strategy      TEXT NOT NULL,
  n             INTEGER NOT NULL,
  params_json   TEXT NOT NULL,
  output_dir    TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_inputs (
  id             INTEGER PRIMARY KEY,
  job_id         TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  role           TEXT NOT NULL,
  ordinal        INTEGER NOT NULL DEFAULT 0,
  source_kind    TEXT NOT NULL,
  source_job_id  TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  source_partition_id TEXT REFERENCES partitions(id) ON DELETE SET NULL,
  stored_path    TEXT NOT NULL,
  source_sha256  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_inputs_job ON job_inputs(job_id);

CREATE TABLE IF NOT EXISTS segments (
  id              INTEGER PRIMARY KEY,
  job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  seq             INTEGER NOT NULL,
  start_epoch     INTEGER NOT NULL,
  stop_at         INTEGER NOT NULL,
  end_epoch       INTEGER,
  resume_from     TEXT,
  argv_json       TEXT NOT NULL,
  operational_diff_json TEXT,
  pid             INTEGER,
  boot_id         TEXT,
  exit_code       INTEGER,
  error_class     TEXT,
  log_path        TEXT,
  started_at      TEXT NOT NULL,
  finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_segments_job ON segments(job_id, seq);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Spec 3 SS9. The web UI's parameter state is stored server-side rather than in
-- localStorage, so "the next job starts from the last job's settings" survives a
-- browser change or a cache clear. The value is opaque JSON: it is the UI's
-- {enabled, value} form object, not a canonical parameter set, because both
-- halves of every optional flag must persist independently -- disabling the
-- origin must not discard its coordinates.
CREATE TABLE IF NOT EXISTS ui_state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  params_json TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {
    "default_num_iter": 4000,
    "default_preview_stop_at": 400,
    "retention_policy": None,
    "worker_paused": False,
}


def utcnow():
    """UTC ISO-8601, second resolution -- the timestamp format used everywhere."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path, timeout=10.0):
    connection = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def initialize(path):
    """Create the schema and seed defaults. Idempotent."""
    connection = connect(path)
    with connection:
        connection.executescript(SCHEMA)
        # Every version so far has been purely additive (new tables, created with
        # IF NOT EXISTS above), so an older database is upgraded by opening it.
        # Record the version we actually left it at rather than the one it had.
        connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        for key, value in DEFAULT_SETTINGS.items():
            connection.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
    return connection
