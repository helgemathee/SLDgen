# Specs

Design specifications for adding checkpointing, a job service, and a web
interface to SLDgen. Written to be implemented in order.

| Spec | Scope | Depends on | Status |
|---|---|---|---|
| [1 — Core checkpointing](sldgen-spec-1-core.md) | `--stop-at`, `--resume`, `--checkpoint-interval`, non-destructive finalisation | nothing | ✅ complete (`4d30690`, 2026-08-09) |
| [2 — Worker and service](sldgen-spec-2-worker.md) | job queue daemon, SQLite store, HTTP API | Spec 1 | ✅ complete (2026-08-09) |
| [3 — Web UI](sldgen-spec-3-web-ui.md) | browser application | Spec 2 | not started |

Each completed spec ends with an **"As built"** section: what shipped, every
place the implementation departed from the design and why, and what was
deliberately left undone. Read that section before the body — where the two
disagree, it is the authority.

## Implementation order

Spec 1 first, **including the three amendments listed in Spec 2 §2** — graceful
SIGTERM, the `state.json` heartbeat, and distinct exit codes. They are written up
in Spec 2 because that is where their consumer lives, but they are changes to the
core and belong in the Spec 1 work. Without them the worker cannot pause a job or
classify a failure. *(Done — see Spec 1 §13 for what shipped and where it
departed from the design, including the one place those amendments had to be
gated to keep the core strictly opt-in.)*

Spec 2's schema is stable and can be built alongside, but cannot be completed
until the core lands. Both have landed, so Spec 3 is unblocked. Spec 1 §11's
recommendation was followed: the worker shells out to the CLI, one segment per
process, and the resident-worker refactor was not attempted — it should not be,
until a real workload shows the reload cost actually hurts.

## Running the service

```bash
python -m venv .venv-service
.venv-service/bin/pip install -r requirements-service.txt

SLDGEN_WORK_ROOT=$PWD/work PYTHONPATH=$PWD \
  /home/helge/miniforge3/envs/sldgen/bin/python -m sldgen_worker &
SLDGEN_WORK_ROOT=$PWD/work PYTHONPATH=$PWD \
  .venv-service/bin/python -m sldgen_api &

curl -s http://127.0.0.1:8765/api/health
```

`deploy/README.md` covers the systemd units (shipped, not installed), every
configuration variable, and the operating notes — including the one about
`git clean -xdf` destroying `work/`.

## Acceptance test for Spec 1

Spec 1 §8: running uninterrupted to epoch 800 must produce the same state as
running to 400, checkpointing, and resuming to 800. Nothing above the core works
if this does not hold. Get `test_resume_geom.py` passing before building anything
else.

**Holds.** `test_resume_geom.py` asserts it on exact tensor equality, and
`test_run_segments.py` asserts the black-box form of it: an uninterrupted run and
a stopped-then-resumed run produce a byte-identical `final_sld.svg`.

## Tests

| File | Needs | What it proves |
|---|---|---|
| `test_resume_geom.py` | conda `sldgen` | the Spec 1 §8 invariant, on the real painter and optimiser |
| `test_run_segments.py` | conda `sldgen` | `run()` end to end, diffusion stubbed: segmented == uninterrupted |
| `test_checkpoint_ops.py` | conda `sldgen` | validation, `state.json`, exit codes, SIGTERM, pure finalisation |
| `test_service_units.py` | `.venv-service` | params, the state machine, the input DAG, logs, disk |
| `test_service_e2e.py` | `.venv-service` | a **live** API and worker: the whole job lifecycle, with SLDgen stubbed |
| `test_service_partitions.py` | both | the real `sld_partition.py`, all six strategies, driven by the API |

None of them needs a GPU. The service tests boot real daemons against a
throwaway root; the only thing ever stubbed is SLDgen itself.

## The one idea to preserve

`--num-iter` is the **horizon** that all schedules normalise against;
`--stop-at` is where a given invocation stops. Collapsing them back into one flag
looks like a simplification and is not: the sparse-loss weight is
`weight * (epoch / num_iter)`, so a 400-iteration preview run with `--num-iter
400` applies sparsity ten times faster than the first 400 iterations of a 4000
run. Previews would stop being prefixes of the runs they preview, and the whole
compare-then-promote workflow would quietly stop meaning anything.

## The model in one sentence

A job is its parameters; **promote** runs more iterations of exactly what is
already there; everything else is a new run.