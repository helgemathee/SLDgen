# Deploying the SLDgen service

Two long-lived units and one short-lived process per segment (Spec 2 §1):

| Process | Env | Owns |
|---|---|---|
| `sldgen-api` | `.venv-service` (no torch) | HTTP API, uploads, zips, disk accounting, partitions |
| `sldgen-worker` | conda env `sldgen` | the GPU queue; one job at a time |
| SLDgen segment | conda env `sldgen` | one `python sldgen.py …` invocation |

They communicate **only** through SQLite (WAL) and the filesystem. That is what
makes "close the browser, restart the API, jobs keep running" true by
construction rather than by discipline.

## First install

```bash
cd /home/helge/SLDgen

# 1. The API's lightweight venv.
python -m venv .venv-service
.venv-service/bin/pip install -r requirements-service.txt

# 2. Verify the whole thing locally before involving systemd. This boots a real
#    API and a real worker against a throwaway root, with a stub SLDgen.
PYTHONPATH=. .venv-service/bin/python test_service_units.py
PYTHONPATH=. .venv-service/bin/python test_service_e2e.py
PYTHONPATH=. .venv-service/bin/python test_service_partitions.py

# 3. Install the units.
sudo cp deploy/sldgen-worker.service deploy/sldgen-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sldgen-worker sldgen-api

curl -s http://127.0.0.1:8765/api/health
```

Edit `SLDGEN_API_HOST` in `sldgen-api.service` to the tailnet address before
exposing it. The API has no authentication, so the bind address is the access
control; it refuses to start on `0.0.0.0`.

## Running it without systemd

Useful while developing, and the exact thing the units do:

```bash
SLDGEN_WORK_ROOT=$PWD/work PYTHONPATH=$PWD \
  /home/helge/miniforge3/envs/sldgen/bin/python -m sldgen_worker

SLDGEN_WORK_ROOT=$PWD/work PYTHONPATH=$PWD SLDGEN_API_PORT=8765 \
  .venv-service/bin/python -m sldgen_api
```

## Configuration

Every path and interval comes from the environment, and both units must agree on
the root.

| Variable | Default | Meaning |
|---|---|---|
| `SLDGEN_WORK_ROOT` | `<repo>/work` | everything the service generates |
| `SLDGEN_PYTHON` | `/home/helge/miniforge3/envs/sldgen/bin/python` | interpreter that runs a segment |
| `SLDGEN_SCRIPT` | `<repo>/sldgen.py` | the script a segment runs |
| `SLDGEN_PARTITION_SCRIPT` | `<repo>/sld_partition.py` | invoked synchronously by the API |
| `SLDGEN_POLL_INTERVAL` | `1.0` | how often the worker reads `state.json` |
| `SLDGEN_CLAIM_INTERVAL` | `2.0` | how long the worker sleeps on an empty queue |
| `SLDGEN_GRACE_SECONDS` | `120` | SIGTERM → SIGKILL grace for a segment |
| `SLDGEN_CHECKPOINT_INTERVAL` | `200` | default periodic checkpoint for new jobs |
| `SLDGEN_API_HOST` / `SLDGEN_API_PORT` | `127.0.0.1` / `8765` | API bind address |

`SLDGEN_PYTHON` and `SLDGEN_SCRIPT` exist so the worker's supervision logic can
be tested against a stub instead of a 15-minute GPU run; they are not something
a deployment normally sets.

## Operating notes

- **`work/` lives inside the checkout and is gitignored.** `git clean -xdf`
  would destroy every job, upload and checkpoint. That is the one command to
  avoid in this repo. Point `SLDGEN_WORK_ROOT` elsewhere if that risk is
  unacceptable — nothing in the design depends on its location.
- **One worker, enforced by `flock` on `work/worker.lock`.** A second worker
  exits 1 immediately. This is also how `/api/health` reports `worker_alive`.
- **Pause is the scheduler.** There is no preemption (Spec 2 §6). To let a
  preview jump ahead of a long run, pause the long one: it checkpoints, stops,
  and resumes on request.
- **Restarting the API never disturbs a job**, and restarting the worker
  mid-segment costs nothing: the segment checkpoints on SIGTERM, the job returns
  to `queued`, and the next worker resumes it from that checkpoint.
- **`journalctl -u sldgen-worker | grep <job_id>`** reconstructs a job's
  lifecycle from the scheduler's point of view; every worker line carries the
  job id where one applies. The same output is available at
  `GET /api/logs/worker`.
