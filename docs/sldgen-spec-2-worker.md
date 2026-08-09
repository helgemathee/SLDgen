# Spec 2 — Worker daemon, job store, and API contract

**Status:** draft for review
**Depends on:** Spec 1 (SLDgen core checkpointing), including the amendments in §2.
**Scope:** everything server-side. No browser code — that is Spec 3.
**Target host:** `fractal` (Threadripper, RTX 5090 32 GB, Ubuntu Server), reached over Tailscale.

---

## 1. Process topology

Three processes, two of them long-lived systemd units:

| Process | Env | Owns |
|---|---|---|
| `sldgen-api` | lightweight venv (no torch) | HTTP API, SQLite writes for CRUD, uploads, zips, disk accounting, partition execution |
| `sldgen-worker` | conda env `sldgen` | The GPU queue. One job at a time. Spawns and supervises SLDgen segments. |
| SLDgen segment | conda env `sldgen` | A single `python sldgen.py …` invocation. Short-lived. |

**Why the API process must not live in the `sldgen` env:** importing FastAPI
alongside torch/diffusers/pydiffvg costs tens of seconds of start-up and holds
CUDA context memory the API never uses. The API must be restartable in under a
second — you will restart it often while building the UI, and restarting it must
never disturb a running job.

The two units communicate only through **SQLite (WAL mode) and the filesystem**.
No sockets, no message broker, no shared memory. This is what makes "close the
browser, restart the API, jobs keep running" true by construction rather than by
discipline.

## 2. Amendments required to Spec 1

Three additions to the core, all still opt-in:

1. **Graceful SIGTERM.** On `SIGTERM`, finish the current iteration, write a
   checkpoint, write `state.json`, exit `0`. This is what makes pause,
   `systemctl stop`, and clean reboots work without losing 20 minutes of GPU
   time. Without it the worker can only stop a run at its `stop_at`.
   Second `SIGTERM` within the grace period exits immediately.

2. **`state.json` heartbeat.** Written to the run directory at every
   `--save-interval` and every checkpoint:

   ```json
   {
     "epoch": 1300, "num_iter": 4000, "stop_at": 1500,
     "phase": "optimizing",          // init | optimizing | finalizing | done
     "iters_per_sec": 4.9,
     "latest_checkpoint": "checkpoints/ckpt_01300.pt",
     "latest_preview": "svg_to_png/iter_1300.png",
     "resolved_caption": "a single line drawing of a firefighter",
     "updated_at": "2026-08-09T12:41:07Z"
   }
   ```

   Progress must not be scraped from tqdm's stderr. tqdm writes carriage
   returns, changes format across versions, and tells you nothing about phase.

3. **Distinct exit codes.** `0` = segment reached its stop point or was
   gracefully terminated (check `state.json` for which). `2` = validation error
   (bad flags, missing file). `3` = environment error (HF auth, missing gated
   model access, CUDA unavailable). `4` = OOM. Anything else = unknown failure.
   This lets the worker classify failures without parsing logs, which matters
   because HF auth failure and OOM need completely different UI treatment.

## 3. Filesystem layout

A single configurable root, defaulting to `work/` **inside the repository
checkout**. It holds only generated data — no source — and is listed in
`.gitignore`, so it is never part of the repository even though it lives in the
tree. It can be deleted wholesale at any time; the service recreates it empty on
next start, and nothing in the repo depends on it.

Keeping it in-tree makes the whole system self-contained: clone the repo, run
the two units, and everything a job produces sits next to the code that produced
it. A production install can point the root elsewhere (`/srv/sldgen`, a
different disk) purely by configuration.

> **Caution:** because the root is inside the working tree, `git clean -xdf`
> will destroy every job, upload and checkpoint. That is the one command to
> avoid in this repo. If that risk is unacceptable, move the root outside the
> tree — nothing else in the design depends on its location.

`.gitignore` gains:

```
work/
```

The existing `output/` entry already covers ad-hoc CLI runs made outside the
service, and should stay.

Layout (paths shown relative to the root, with the repo-default root spelled out
literally where a full path is needed):

```
<repo>/work/
  sldgen.sqlite                    # WAL: also sldgen.sqlite-wal, sldgen.sqlite-shm
  worker.lock                      # flock; singleton guard
  uploads/
    <sha256>.png                   # content-addressed, immutable, deduplicated
  jobs/
    <job_id>/                      # job_id = ULID (sortable, opaque)
      inputs/
        target.png                 # hardlink to uploads/<sha256>.png
        stipple_weight.png         # copies of any SVG/PNG inputs, see §4.3
        avoid_000.svg
        attract_000.svg
      target/run/                  # SLDgen's own output dir, see note below
        input.png  mask.png  condition_depth.png
        config.json  state.json  metrics.json
        checkpoints/ckpt_00400.pt  latest.pt
        svg_logs/svg_iter400.svg …
        svg_to_png/iter_0400.png …
        final_sld.svg  final_sld.png  sketch.mp4
      logs/
        segment_001.log
  partitions/
    <partition_id>/
      partition_0.svg … partition_N-1.svg
      partition_preview.png
      labels.png
  tmp/
    zip-<uuid>/                    # staging for downloads, swept on start
```

**Note on `target/run/`.** `set_output_directories` inserts
`Path(args.target).stem` between `--output-dir` and `--experiment-name`. The
worker always names the input `target.png` and passes `--experiment-name run`,
so the path is deterministic: `jobs/<job_id>/target/run/`. Do not fight this;
just rely on it.

## 4. Data model

SQLite, WAL mode, `busy_timeout=5000`. All timestamps UTC ISO-8601 strings.

### 4.1 `jobs`

```sql
CREATE TABLE jobs (
  id                TEXT PRIMARY KEY,       -- ULID
  title             TEXT,                   -- user label, defaults to filename
  state             TEXT NOT NULL,          -- see §5
  desired_state     TEXT NOT NULL,          -- 'run' | 'pause' | 'delete'
  params_json       TEXT NOT NULL,          -- canonical param object, §4.2
  target_sha256     TEXT NOT NULL,
  num_iter          INTEGER NOT NULL,       -- horizon (default 4000)
  target_epoch      INTEGER NOT NULL,       -- where this job should stop next
  current_epoch     INTEGER NOT NULL DEFAULT 0,
  resolved_caption  TEXT,
  parent_job_id     TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  batch_id          TEXT,             -- shared by variants queued together
  priority          INTEGER NOT NULL DEFAULT 0,
  error_class       TEXT,                   -- 'validation'|'environment'|'oom'|'unknown'
  error_message     TEXT,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL,
  started_at        TEXT,
  finished_at       TEXT
);
CREATE INDEX idx_jobs_queue ON jobs(state, priority DESC, created_at);
CREATE INDEX idx_jobs_batch ON jobs(batch_id);
```

`current_epoch` is a **cache** of `state.json` for cheap list queries. The file
is authoritative.

### 4.2 Params

One canonical JSON object mirroring SLDgen's flags, stored verbatim so a job is
always reproducible and the exact command can be regenerated. Null means "flag
omitted". A single translator module (`params_to_argv`) is the *only* place
that knows the CLI, and it must round-trip: `argv_to_params(params_to_argv(p)) == p`.

Two groups, matching Spec 1 §6:

- **structural** — every parameter that shapes the result: `render_size`,
  `n_control_points`, `init_method`, `seed`, `num_iter`, `origin`,
  `fixed_endpoints`, `width`, `optimize_cp_weights`, `prune_low_weights`,
  `calligraphy`, `object_size_ratio`, `sampling_rate`, `caption`,
  `conditioning_scale`, `condition`, `lora_*`, `lr`, `avoid`, `attract`,
  `*_weight`, `*_distance`, and all loss weights. **Immutable once the job
  exists.** Changing any of them means a new job (§5, "run again").
- **operational** (`save_interval`, `checkpoint_interval`, `verbose`, `debug`) —
  freely editable at any time; they never affect the result.

There is deliberately no middle class. A job's parameters and its result are one
thing, so a job that ran under two different parameter sets could not be
described by either. The cost is that exploring a variation always spends a
fresh run; the benefit is that every job in the database is exactly what its
parameters say it is.

### 4.3 `job_inputs` — the DAG

```sql
CREATE TABLE job_inputs (
  id             INTEGER PRIMARY KEY,
  job_id         TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  role           TEXT NOT NULL,   -- 'avoid'|'attract'|'init_points'|'stipple_weight'|'labels'
  ordinal        INTEGER NOT NULL DEFAULT 0,
  source_kind    TEXT NOT NULL,   -- 'job'|'partition'|'upload'
  source_job_id  TEXT REFERENCES jobs(id) ON DELETE SET NULL,
  source_partition_id TEXT REFERENCES partitions(id) ON DELETE SET NULL,
  stored_path    TEXT NOT NULL,   -- the COPY under jobs/<id>/inputs/
  source_sha256  TEXT NOT NULL
);
```

**Inputs are copied, not referenced.** When job B is created with job A's
`final_sld.svg` as an `--avoid` input, the file is copied into
`jobs/B/inputs/avoid_000.svg` at submission. The `source_*` columns record
provenance for the graph view, but B never reads from A's directory. This means
deleting A can never break or silently alter B, and B stays reproducible. The
cost is a few hundred KB per edge — irrelevant.

**Coordinate-space guard.** Reject any `avoid`/`attract`/`init_points` input
taken from a source run whose `config.json` shows `scale_w`/`scale_h` present
*unless* it is `final_sld.svg` (see Spec 1 §7). Intermediate `svg_logs/` files
from a rescaled run are in a different space and would silently misregister.

### 4.4 `segments`

One row per SLDgen invocation. This is the audit trail and the source of the
throughput numbers the UI shows.

```sql
CREATE TABLE segments (
  id              INTEGER PRIMARY KEY,
  job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  seq             INTEGER NOT NULL,
  start_epoch     INTEGER NOT NULL,
  stop_at         INTEGER NOT NULL,
  end_epoch       INTEGER,
  resume_from     TEXT,              -- checkpoint path, null for first segment
  argv_json       TEXT NOT NULL,     -- exact argv, for reproduction
  operational_diff_json TEXT,        -- operational settings changed vs previous segment
  pid             INTEGER,
  boot_id         TEXT,              -- /proc/sys/kernel/random/boot_id
  exit_code       INTEGER,
  log_path        TEXT,
  started_at      TEXT NOT NULL,
  finished_at     TEXT
);
```

### 4.5 `partitions`

```sql
CREATE TABLE partitions (
  id            TEXT PRIMARY KEY,
  source_job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  source_svg    TEXT NOT NULL,       -- normally final_sld.svg
  strategy      TEXT NOT NULL,
  n             INTEGER NOT NULL,
  params_json   TEXT NOT NULL,       -- origins, connect_tails, sample_spacing, seed, labels
  output_dir    TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
```

### 4.6 `settings`

Single-row key/value: `default_num_iter`, `default_preview_stop_at`,
`retention_policy`, `worker_paused`.

## 5. Job lifecycle

```
                 ┌────────────┐
   create ──────►│  queued    │◄───────── promote / retry
                 └─────┬──────┘
                       │ worker claims
                 ┌─────▼──────┐
        ┌────────┤  running   ├────────┐
        │        └─────┬──────┘        │
        │              │               │
   pause requested   reached      segment failed
        │          target_epoch        │
   ┌────▼─────┐        │          ┌────▼─────┐
   │  paused  │        │          │  failed  │
   └────┬─────┘   ┌────▼────┐     └────┬─────┘
        │         │ waiting │          │ retry
        └────────►└────┬────┘◄─────────┘
                       │ target_epoch == num_iter
                  ┌────▼─────┐
                  │ complete │
                  └──────────┘
```

- **`waiting`** means "this job has reached its current budget and is idle,
  pending a decision". It is the state your four previews sit in. Distinct from
  `paused` (user intervened) and `complete` (reached the horizon).
- **`deleting`** is a terminal-ish state entered by the API; the worker sees it,
  aborts the job if running, and the API reclaims the directory afterwards.
- `desired_state` is how the API asks for a transition without racing the
  worker. The API only ever writes `desired_state`; the worker owns `state`.
  This single rule removes almost all the concurrency hazard.

**Promote** = set `target_epoch` higher and move `waiting` → `queued`.
**Run again** = create a new job with `parent_job_id` set to the source job,
reusing its source image, prepared `input.png`, mask, origin and parameters,
with any parameter edited. It starts from epoch 0 and shares no state with its
parent — only provenance. It is permitted from **any** parent state, including
`running` and `failed`, since it does not touch the parent.

This is the only derivation operation. There is no resume-from-a-parent's-
checkpoint path: because every result-shaping parameter is structural (§4.2),
such a job could only be identical to its parent, which is what `promote`
already does.

## 6. Scheduling

**Policy: FIFO, one job at a time, bounded by each job's `target_epoch`. No preemption.**

The four-previews-in-twenty-minutes workflow does not require a preemptive
scheduler. Submit four jobs with `target_epoch = 400` against a horizon of 4000
and FIFO delivers all four in roughly eight minutes with zero context switches
and zero model reloads. Preemption would *reduce* throughput here, because each
switch costs a full pipeline load.

The worker picks the next job by `priority DESC, created_at ASC` among
`queued`, and runs it as a single segment from `current_epoch` to
`target_epoch`.

**Long-run fairness needs no scheduler.** A job promoted to 4000 occupies the
GPU for ~15 minutes, and a queued preview waits behind it. When that is not what
you want, **pause** is the answer: the running job checkpoints and stops, the
preview runs, and the paused job resumes on request. This gives preemption's
benefit exactly when it is worth its cost — roughly 45 seconds of pipeline
reload per switch — and puts that judgement with the operator rather than in a
policy.

No preemption, no time slicing, no `max_segment_iters`. If FIFO ever proves
genuinely annoying in daily use, revisit it then, with evidence.

## 7. Worker loop

```
acquire flock on <root>/worker.lock, else exit         # singleton
sweep: reconcile any job in 'running' (§8)
loop:
  if settings.worker_paused: sleep, continue
  claim = SELECT … WHERE state='queued' ORDER BY priority DESC, created_at
          → UPDATE state='running' in one transaction
  if no claim: sleep 2s; continue
  argv = params_to_argv(job) + resume/stop-at flags
  spawn subprocess in conda env, new process group, stdout+stderr → segment log
  while running:
    poll state.json → update jobs.current_epoch, jobs.resolved_caption
    if desired_state in ('pause','delete') or worker shutting down:
        SIGTERM the process group; wait up to 120s; then SIGKILL
  classify exit code; write segments row
  set final state: waiting | complete | paused | failed
```

Three details that matter:

- **New process group** for the child, so SIGTERM reaches it and not just the
  shell wrapper, and so a `SIGKILL` fallback can take the whole tree.
- **The 120-second grace period** must exceed one iteration plus a checkpoint
  write. At ~5 it/s an iteration is 200 ms, so this is generous; keep it
  generous anyway, because `finalizing` (metrics + ffmpeg) can take longer and
  interrupting it mid-write corrupts outputs.
- **Poll `state.json`, don't watch inotify.** A 1-second poll of one small file
  is simpler, survives the file being replaced atomically, and has no watch-limit
  failure mode.

## 8. Crash and restart recovery

On worker start, for every job in `running`:

1. Read `segments` for the open segment: `pid` and `boot_id`.
2. If `boot_id` differs from the current boot, the machine rebooted — the
   process is definitely gone.
3. Else check whether `pid` is alive *and* is an SLDgen process (compare
   `/proc/<pid>/cmdline`), guarding against PID reuse.
4. If alive: adopt it — resume polling, do not spawn a duplicate.
5. If dead: read `checkpoints/latest.pt`, set `current_epoch` from it, mark the
   segment `exit_code = -1`, and return the job to `queued`. **Resume from the
   checkpoint; never restart from epoch 0.**

If `latest.pt` is missing (crash before the first checkpoint), restart from 0.
This is the argument for a non-zero default `--checkpoint-interval` — 200 is a
reasonable default, costing under a second per write.

On API start: sweep `tmp/`, and reconcile any job in `deleting`.

## 9. GPU exclusivity

**Within the service: exactly one job at a time.** The worker is a singleton
(flock, §7) and runs one segment at a time. SD3.5-medium plus the ControlNet
does not leave room for a second concurrent run on a 32 GB card, so this is not
a tuning parameter — it is the design.

**Outside the service: not the service's problem.** Other GPU consumers on the
host are managed by operator convention — simply don't start a long run while
something else is holding the card. Because each segment is a separate
short-lived process, SLDgen releases all VRAM when it exits, so back-to-back use
of the card by different tools needs no coordination.

Consequently there is **no VRAM guard, no polling for headroom, and no
`gpu_blocked` state**. The safety net is the `oom` error class (§15): if a
segment does hit an out-of-memory condition, it fails cleanly, is classified,
and can be retried once the card is free. That is a better trade than a
heuristic gate that delays every job to protect against a situation that only
arises when the operator has already made a choice.

`/api/health` still reports `gpu_free_mb` as a diagnostic, since it is useful
when a job does fail with OOM and you want to see what else is resident.

## 10. Deletion, retention, disk

**Deletion** is a two-phase operation, owned by the API but observed by the
worker:

1. API sets `desired_state = 'delete'`; if the job is not `running`, it
   transitions directly to `deleting`.
2. The worker, if running that job, SIGTERMs it and sets `deleting`.
3. The API (or worker, whoever sees `deleting` with no live PID) moves
   `jobs/<id>/` to `tmp/` and unlinks, then deletes the row.

Move-then-unlink so a partially deleted directory is never visible as a live
job. Because inputs are copied (§4.3), no dependency check is needed — deletion
can never orphan another job.

**Disk accounting.** A `disk` endpoint reporting total, per-job, and per-category
bytes. Compute per-job size on job completion and cache it in a
`jobs.disk_bytes` column; recompute lazily on demand. Do not `du` the whole tree
on every request.

Expected magnitudes, so the UI can set sensible thresholds: checkpoints are a
few hundred KB each (three float tensors over ~385 control points plus Adam
state); a 4000-iteration run at `--save-interval 100` produces ~40 SVGs and ~40
PNGs; `make_video` runs ffmpeg with `-vb 20M` over ~40 frames at 10 fps. Total
per run is tens of megabytes, not gigabytes. Disk pressure is a non-issue until
you have hundreds of runs, which is precisely why the accounting should exist
before it becomes one.

**Retention policy** (setting, default off): for `complete` jobs older than N
days, prune all checkpoints except the last, and optionally the `svg_to_png`
frames once `sketch.mp4` exists. Never prune `final_sld.svg`, `config.json`,
`metrics.json` or `state.json` — those are the reproducible record.

## 11. Partitions

Partitioning is CPU-only and takes seconds, so it runs **synchronously in the
API process**, never in the GPU queue. `sld_partition.py` is invoked as a
subprocess (it is a standalone script with its own argparse) and the API blocks
for the result.

The API must expose a "live preview" affordance: given a source job, a strategy,
N, and optional origins, run with `--preview` and return the
`partition_preview.png` plus the N SVGs. Re-running with different parameters
overwrites in place until the user commits, at which point a `partitions` row is
written. This is what makes strategy selection feel like scrubbing rather than
submitting.

For `--strategy labelmap`, the default `--labels` is the source job's
`condition_<condition>.png`, which is already persisted in canvas space.

Matplotlib is required for `--preview`; it is in the `sldgen` env but must also
be available to whichever env invokes the script. Simplest resolution: the API
invokes `sld_partition.py` using the **conda env's interpreter by absolute
path**, while the API itself remains lightweight.

## 12. API contract

Single user, no authentication, **bound to the Tailscale interface only** —
never `0.0.0.0`. JSON throughout except where noted.

```
GET    /api/health                      → {ok, worker_alive, gpu_free_mb, db_ok}
GET    /api/settings                    → settings object
PATCH  /api/settings                    → update

POST   /api/uploads                     multipart → {sha256, width, height, url}
GET    /api/uploads/{sha256}            → image bytes

GET    /api/jobs?state=&limit=&cursor=  → [job summary]   (grid view)
POST   /api/jobs                        {title, target_sha256, params, target_epoch,
                                         inputs:[{role, source_kind, source_id|sha256}]}
                                        → job
GET    /api/jobs/{id}                   → job detail incl. segments, inputs, artifacts
PATCH  /api/jobs/{id}                   {title?, operational params?, target_epoch?, priority?}
                                        → 409 if a structural param is edited,
                                          pointing at /run-again
DELETE /api/jobs/{id}                   → 202, sets desired_state=delete

POST   /api/jobs/{id}/pause             → 202
POST   /api/jobs/{id}/resume            → 202
POST   /api/jobs/{id}/promote           {target_epoch}     → job
POST   /api/jobs/{id}/run-again         {variants:[{params_overrides}], …} → [new jobs]
POST   /api/jobs/{id}/retry             → job

GET    /api/jobs/{id}/events            → SSE: {epoch, phase, iters_per_sec,
                                                preview_url, state}
GET    /api/jobs/{id}/preview           → latest iter PNG (302 to file route)
GET    /api/jobs/{id}/artifacts         → [{name, path, bytes, kind}]
GET    /api/jobs/{id}/files/{path}      → raw artifact (SVG/PNG/mp4/json)
GET    /api/jobs/{id}/log?segment=&from=&raw=
                                        → {from,to,text,eof,running}; Range honoured
GET    /api/jobs/{id}/log/stream?segment=&from=&raw=
                                        → SSE: appended chunks, terminal event on exit
GET    /api/jobs/{id}/log/download?segment=
                                        → text/plain attachment, always raw
GET    /api/jobs/{id}/download.zip      → streamed zip
GET    /api/jobs/{id}/command           → the exact argv, as text (reproduction)

POST   /api/partitions/preview          {source_job_id, strategy, n, params}
                                        → {preview_url, svg_urls}
POST   /api/partitions                  → commit → partition
GET    /api/partitions/{id}/download.zip

GET    /api/logs/worker?lines=          → recent sldgen-worker journal, text/plain

GET    /api/disk                        → {total_bytes, by_job:[…], by_category:{…}}
POST   /api/maintenance/prune           {policy} → summary
```

**SSE, not polling, for job detail.** The API tails `state.json` and pushes.
Falling back to polling the job detail endpoint must also work, because SSE
through some reverse proxies is unreliable and this needs to survive a bad
proxy config.

**Zips are streamed**, generated on the fly from the job directory with no
staging file, except when the caller requests a partition set (already small).
Exclude `checkpoints/` by default with an opt-in query flag — checkpoints are
useless outside this service and would dominate the archive.

## 13. Logging and console access

**Requirement: every job's full console output must be readable in the UI, live
while it runs and permanently afterwards.** This is a first-class feature, not a
debugging afterthought — when a run fails at 3800 iterations, the log is the only
thing that explains why.

### 13.1 Capture

The worker spawns each segment with stdout and stderr **merged into a single
file descriptor** writing to `<root>/jobs/<id>/logs/segment_<seq>.log`. Merging
matters: SLDgen prints status to stdout while tqdm and the HF libraries write to
stderr, and interleaving them in one stream is the only way the resulting log
reads in causal order.

Two environment settings are required on the child, or the log will appear to
hang for minutes at a time:

- `PYTHONUNBUFFERED=1`. Python block-buffers stdout when it is a pipe rather
  than a tty. SLDgen's own prints mostly pass `flush=True`, but not all of them
  (`print("Done!")` in `run.py` does not), and library output does not. Without
  this, output arrives in 8 KB bursts and live tailing is useless.
- `TQDM_MININTERVAL` left at its default, or raised to ~1.0. tqdm's default
  refresh produces a progress fragment several times per second; over a
  4000-iteration run that is a lot of noise for very little information.

Do **not** allocate a pty to make tqdm behave "normally". It introduces
platform-specific teardown behaviour on kill, and buys nothing the `\r` handling
below does not already give.

The worker writes a header block to the log before spawning: timestamp, segment
sequence, start and stop epochs, resume checkpoint, and the **exact argv**. The
log is then self-contained — it can be pasted into an issue and reproduced
without consulting the database.

On exit, the worker appends a footer: exit code, classified error class, end
epoch, wall time, and mean iterations per second.

### 13.2 Serving

Two access patterns, both reading the same file:

**Scrollback** — `GET /api/jobs/{id}/log?segment=N&from=<byte>` returns
`{from, to, text, eof, running}`. Byte-offset based rather than line based, so
a client can fetch incrementally, reconnect after a network blip, and resume
exactly where it left off without re-downloading megabytes. `Range` requests are
also honoured for raw download.

**Live tail** — `GET /api/jobs/{id}/log/stream?segment=N&from=<byte>` is an SSE
stream that emits appended chunks as they are written, and a terminal event when
the segment exits. Implemented by polling the file size (250 ms) and reading the
delta; no inotify, for the same reasons as §7.

Both accept `?raw=true`. Default (`raw=false`) applies **carriage-return
cooking**: a run of `\r`-separated fragments collapses to only its final state,
which turns thousands of tqdm repaints into one live-updating progress line —
exactly what a terminal shows. The file on disk is never modified; cooking
happens at serve time, so `raw=true` always yields the true bytes.

The UI should default to cooked, offer a raw toggle, and offer "download log".

### 13.3 Retention and size

Logs are per segment and never rotated mid-segment — a segment is bounded, so
its log is bounded. A 4000-iteration run at default tqdm settings produces on the
order of a few hundred KB; with `TQDM_MININTERVAL=1.0`, considerably less. This
is small against the run's own artefacts, so **logs are exempt from the retention
policy in §10** and are kept for the life of the job. A job's logs are deleted
only when the job is.

Because logs are per segment, a job promoted through several budgets accumulates
several files. The API returns the segment list with the job detail, and the UI
should present them as an ordered set — most recent first, with the ability to
read any earlier one — rather than concatenating them into a single stream,
since each has its own argv header and exit status.

### 13.4 Service-level logs

The worker and API themselves log to journald via systemd. These are separate
from job logs and contain scheduling decisions, claims, crash reconciliation and
deletion activity. Every worker line carries the `job_id` where one applies, so
`journalctl -u sldgen-worker | grep <job_id>` reconstructs a job's lifecycle from
the scheduler's point of view.

Expose the worker's recent journal through `GET /api/logs/worker?lines=` so the
UI can surface "why has nothing started?" without an SSH session. Read it by
invoking `journalctl -u sldgen-worker -n <lines> --no-pager -o cat` as a
subprocess; the API user must be in the `systemd-journal` group.

## 14. Deployment

Two systemd units, both `Restart=always`, `After=network-online.target`.
Per Helge's convention, all paths literal.

```ini
# /etc/systemd/system/sldgen-worker.service
[Service]
Type=simple
User=helge
WorkingDirectory=/home/helge/SLDgen
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=CONCORDE_PATH=/home/helge/src/concorde/TSP/concorde
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=/home/helge/miniforge3/envs/sldgen/bin/python -m sldgen_worker
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=180
```

`TimeoutStopSec` must exceed the worker's own child grace period, or systemd
will SIGKILL the worker before it can shepherd the segment to a checkpoint.

`WorkingDirectory` is the repo checkout, which is also where the default `work/`
root lives. Both units must agree on the root; declare it once as an environment
variable in each unit rather than defaulting independently.

The conda env is entered by **calling the env's interpreter directly**, not by
`conda activate` in a shell wrapper. Environment variables that
`conda env config vars set` would normally provide are declared explicitly in
the unit, because they are not applied when bypassing activation. This is a
common and confusing failure — `CONCORDE_PATH` missing means TSP init fails
several seconds into every job with an unhelpful error.

Repository layout, keeping everything in the SLDgen repo as intended:

```
SLDgen/
  SLDgen/                  # unchanged core package
  sldgen.py                # unchanged entry point
  sld_partition.py         # unchanged
  sldgen_worker/           # this spec
  sldgen_api/              # this spec
  sldgen_web/              # Spec 3
  requirements-service.txt # lightweight: fastapi, uvicorn, pydantic — no torch
  work/                    # GITIGNORED — all generated data, see §3
```

The `work/` tree is data, not source. It is safe to delete, is never committed,
and no code reads configuration from it.

## 15. Failure taxonomy

The UI must distinguish these; they need different actions.

| Class | Cause | Surface as | Retryable |
|---|---|---|---|
| `validation` | bad flag combination, missing input file | "fix the parameters" | after edit |
| `environment` | HF auth expired, gated model access revoked, CUDA missing, Concorde not found | "the machine needs attention", with the log excerpt | yes, unchanged |
| `oom` | VRAM exhausted, usually another process | "GPU busy", show current free VRAM | yes |
| `interrupted` | SIGTERM, reboot, worker crash | not an error; job returns to `waiting` | automatic |
| `unknown` | anything else | full log | manual |

Note that gated-model access is a genuine first-run failure mode on a fresh
machine: SD3.5-medium requires an approved HuggingFace account. It fails several
seconds in, deep in a stack trace. Detecting it and saying so plainly is worth
the special case.

## 16. Open questions

2. **Metrics on non-final segments.** Currently only computed at the horizon.
   The job grid would be more useful with CLIP/aesthetic scores on previews, but
   that means loading three more models per segment. Probably better as a
   separate opt-in "score this job" action that batches across jobs.
3. **Multi-GPU.** Specced as one worker, one card. If a second card ever
   appears, the worker becomes N workers each holding a device-scoped lock; the
   schema does not need to change.
4. **Remote access.** The API binds to the tailnet. Confirm that Postlab's
   GL.iNet subnet routing gives you the path you expect from the MacBook, or
   whether the API should also bind to the rack subnet.
