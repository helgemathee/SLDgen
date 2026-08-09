# Specs

Design specifications for adding checkpointing, a job service, and a web
interface to SLDgen. Written to be implemented in order.

| Spec | Scope | Depends on |
|---|---|---|
| [1 — Core checkpointing](spec-1-sldgen-core-checkpointing.md) | `--stop-at`, `--resume`, `--checkpoint-interval`, non-destructive finalisation | nothing |
| [2 — Worker and service](spec-2-worker-and-service.md) | job queue daemon, SQLite store, HTTP API | Spec 1 |
| [3 — Web UI](spec-3-web-ui.md) | browser application | Spec 2 |

## Implementation order

Spec 1 first, **including the three amendments listed in Spec 2 §2** — graceful
SIGTERM, the `state.json` heartbeat, and distinct exit codes. They are written up
in Spec 2 because that is where their consumer lives, but they are changes to the
core and belong in the Spec 1 work. Without them the worker cannot pause a job or
classify a failure.

Spec 2's schema is stable and can be built alongside, but cannot be completed
until the core lands.

## Acceptance test for Spec 1

Spec 1 §8: running uninterrupted to epoch 800 must produce the same state as
running to 400, checkpointing, and resuming to 800. Nothing above the core works
if this does not hold. Get `test_resume_geom.py` passing before building anything
else.

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