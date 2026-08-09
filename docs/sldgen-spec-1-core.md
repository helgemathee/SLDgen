# Spec 1 — SLDgen core: checkpointing, resume, and segmented runs

**Status:** draft for review
**Scope:** changes inside the SLDgen package only. No web service, no daemon, no UI.
**Repo:** `helgemathee/SLDgen` (the `sm120-cuda13` fork). Conda env `sldgen`.

---

## 1. Motivation

The intended workflow is *comparative*: submit four or five candidate images,
give each a short run (400–1000 iterations), look at the results side by side,
then promote the promising ones to a full run without throwing away the work
already done. This requires three things SLDgen does not currently have:

1. **Segmented execution** — run iterations *m..n* of a longer trajectory, then
   stop cleanly.
2. **Resume** — pick that trajectory back up exactly where it stopped.
3. **Schedule invariance** — a 400-iteration preview must be a genuine *prefix*
   of the 4000-iteration run it previews, not a different drawing.

Point 3 is the non-obvious one and is discussed in §3.

## 2. Design principles

These follow the discipline already established by `--origin`, `--avoid`,
`--attract`, `--init-points` and `--stipple-weight` in this fork:

- **Strictly opt-in.** With none of the new flags set, behaviour must be
  bit-identical to the current fork, including file layout and stdout.
- **Validation in `parse_arguments`.** Incompatible combinations error out at
  parse time with a clear message, not mid-run.
- **No new required dependencies.** `torch.save`/`torch.load` only.
- **A fast CPU-only geometry test**, in the style of `test_origin_geom.py`, that
  proves the invariant without loading a diffusion model.

## 3. The scheduling problem (must be solved, not worked around)

`SLDgen/run.py::get_sparse_loss_weight` is:

```python
return target_weight * (epoch / args.num_iter)
```

The sparsity pressure ramps linearly from zero to `--sparse-loss-weight`, and it
reaches full strength **at whatever `--num-iter` was passed**. Consequences:

- A run with `--num-iter 400` applies sparsity ten times faster than the first
  400 iterations of a `--num-iter 4000` run. The preview is *not* an early look
  at the long run; it is a different optimisation.
- Naively "continuing for 500 more" by launching a fresh `--num-iter 500` run
  from a checkpoint would restart the ramp from zero.

**Resolution:** `--num-iter` keeps its current meaning — the *horizon* that all
schedules normalise against, and the iteration at which the run is considered
complete. A new flag `--stop-at N` sets where *this invocation* stops. The
schedule is always a function of the absolute epoch and the horizon, so
segmenting is transparent.

This also means no `--start-iter` flag is needed: the starting epoch is a
property of the checkpoint, not something the caller asserts.

> If any future schedule is added (LR decay, progressive conditioning scale), it
> must likewise be a pure function of `(epoch, args.num_iter)` and never of
> "iterations executed so far in this process".

## 4. New CLI flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--stop-at` | int | `None` | Execute up to and including this absolute epoch, write a checkpoint, exit 0. When unset, run to `--num-iter` and finalise as today. |
| `--resume` | path | `None` | Restore state from this checkpoint and continue. The starting epoch comes from the file. |
| `--checkpoint-interval` | int | `0` | Additionally write a checkpoint every N epochs (crash recovery). `0` disables. |

### Validation rules

- `--stop-at` must be `> 0` and `<= --num-iter`.
- With `--resume`, `--stop-at` (if given) must be `>` the checkpoint's epoch.
  Equal is a no-op and should error rather than silently produce an identical
  checkpoint.
- `--resume` path must exist and carry a recognised `format_version`.
- `--resume` is incompatible with `--init-points` and `--stipple-weight`:
  both only affect initialisation, which does not re-run on resume. Error at
  parse time so the caller is not misled into thinking they took effect.
- `--checkpoint-interval` must be `>= 0`.

### Semantics of completion

A run is **complete** when the executed epoch reaches `--num-iter`. Only then:
`increase_object_size` runs, `final_sld.svg` / `final_sld.png` are written,
metrics are computed, and `make_video` is called. A segment that stops early
writes a checkpoint and nothing else. `svg_logs/` and `svg_to_png/` continue to
accumulate across segments exactly as within a single run, so the assembled
video spans all segments.

## 5. Checkpoint format

Written with `torch.save` to
`{output_dir}/checkpoints/ckpt_{epoch:05d}.pt`, plus a copy (not a symlink —
the service may move directories) at `{output_dir}/checkpoints/latest.pt`.

```
{
  "format_version": 1,
  "epoch": int,                      # last completed epoch
  "num_iter": int,                   # horizon this trajectory was scheduled against
  "canvas": {"width": int, "height": int},

  # optimised state
  "control_points": FloatTensor[N, 2],   # canvas pixel coords, CPU
  "weights":        FloatTensor[N],
  "width":          FloatTensor[N],
  "is_active_cp":   BoolTensor[N],

  # pinned, non-optimised geometry
  "pinned_kind": "none" | "origin" | "fixed_endpoints",
  "pinned": { ... tensors, see below ... },

  # optimiser
  "optimizer": optim.state_dict(),
  "param_group_names": ["control_points", "weights", "widths"],

  # reproducibility
  "rng": {
    "python": ..., "numpy": ..., "torch_cpu": ..., "torch_cuda": [...]
  },

  # resolved inputs
  "resolved_caption": str,
  "structural_fingerprint": { ... see §6 ... },
  "target_sha256": str
}
```

### Notes on each group

**Optimised state.** `control_points`, `weights` and `width` are the three
tensors registered in `SLDBSplinePainter.parameters()`. `is_active_cp` must be
saved because pruning in `post_process_params` is **monotone** — a control point
that drops below the weight threshold never reactivates — so it is genuine state,
not derivable from the weights alone.

Pruning never resizes the tensors (it masks in `get_polyline_2d`), so parameter
shapes are constant for the lifetime of a run. This is what makes the Adam state
trivially restorable.

**Pinned geometry.** Depending on the mode, save either
`first_origin_points/weights/widths` (origin mode) or
`first2points/first2weights/first2widths` + `last2points/last2weights/last2widths`
(fixed-endpoints mode). These are produced by `init_image` from the TSP tour and
cannot be cheaply recomputed — recomputation means re-running Concorde.

**Optimiser.** `Adam(betas=(0.9, 0.9))` — an unusually short second-moment
memory, so dropping the state would not be catastrophic, but the step counter
matters for bias correction and the payload is small. Save it.

**RNG.** Save all four states. The SDS loss samples a diffusion timestep per
iteration; without this, a resumed run diverges from an uninterrupted one even
with identical geometry. Restoring RNG is what makes the invariant in §8
testable.

**Resolved caption.** With `--caption ""`,
`SD3GuidanceControl.create_caption()` runs BLIP-2 and **mutates `args.caption`
in place**. On resume the caption must be taken from the checkpoint rather than
re-derived: it removes a nondeterminism risk and skips a model load. Implement
by assigning `args.caption` from the checkpoint before `SD3GuidanceControl` is
constructed.

## 6. Structural fingerprint

**Every run-shaping argument is structural.** A trajectory is defined by its
parameters; changing any of them means a different drawing, so the only honest
way to explore a variation is a fresh run from epoch 0. Resume exists to
*continue* a trajectory, never to redirect one.

**Structural — must match on resume, error if they do not:**
`render_size`, `n_control_points`, `init_method`, `seed`, `num_iter`,
`optimize_cp_weights`, `prune_low_weights`, `width` (mode), `origin`,
`fixed_endpoints`, `calligraphy`, `object_size_ratio`, `sampling_rate`,
`caption`, `conditioning_scale`, `condition`, `lora_model`, `lora_weight`,
`lr`, `avoid`, `attract`, `avoidance_*`, `attraction_*`, all regularisation
loss weights, and `target_sha256`.

**Operational — may differ between segments, never affects the result:**
`save_interval`, `checkpoint_interval`, `verbose`, `debug`, `output_dir`,
`experiment_name`, `stop_at`.

The comparison is a single equality check over a canonical dict. On mismatch,
error out naming every field that differs, with both values. Do not warn and
proceed: a silently redirected trajectory produces a result that cannot be
explained by its own recorded parameters, which is worse than a hard stop.

> **Note on `num_iter`.** It is structural, so a completed 4000-iteration run
> cannot be extended to 5000 by resuming — the horizon defines the sparse-loss
> ramp for every iteration, and raising it mid-trajectory would optimise the
> tail under a schedule the earlier iterations never saw. Extending means a new
> run at the higher horizon. This is a deliberate trade of flexibility for
> reproducibility.

## 7. Non-destructive finalisation

The tail of `run()` is currently destructive:

```python
increase_object_size(renderer, args)          # rewrites renderer.shapes
renderer.save_svg(args.output_dir, "final_sld")
renderer.canvas_width *= 2                    # in-place
renderer.control_points = renderer.control_points * 2
renderer.width = renderer.width * 2
renderer.first_origin_points = ... * 2
```

After this, the renderer's state no longer corresponds to the trajectory, so a
"finished" job cannot be continued from its own end state, and a checkpoint
written after finalisation would be wrong by a factor of two.

**Required change:** extract a `finalize(renderer, args)` that performs the 2×
export on *cloned* tensors and restores (or never mutates) the renderer's own
state. Finalisation must be a pure export step: the renderer's state after it
runs must be indistinguishable from its state before, so the last checkpoint of
a run always describes the trajectory rather than the export.

### Coordinate-space note (informational, affects downstream tooling)

`increase_object_size` is applied before `final_sld.svg` is written, but the
intermediates in `svg_logs/` are saved without it. When `--object-size-ratio`
actually triggered a rescale (`args.scale_w`/`scale_h` present), intermediate
SVGs live in a **different space** than `final_sld.svg`. They are safe as
previews but must not be fed to `sld_partition.py`, `--avoid`, `--attract` or
`--init-points`. Recommendation: record `scale_w`, `scale_h`,
`original_center_x`, `original_center_y` in the checkpoint and in the run's
`config.json`, so downstream tools can detect and normalise the case.

## 8. Correctness invariant

> Running `--num-iter 4000` uninterrupted to epoch 800 must produce the same
> control points, weights and widths as running `--num-iter 4000 --stop-at 400`
> followed by `--num-iter 4000 --stop-at 800 --resume ckpt_00400.pt`.

Exact bitwise equality is not required (CUDA non-determinism in the diffusion
backward pass makes it unattainable). The test asserts on a CPU-only,
diffusion-free path where equality *is* exact; see §10.

## 9. Implementation sketch by file

**`SLDgen/config.py`**
Add the three arguments and their validation. Add a `structural_fingerprint(args)`
helper and a `target_sha256(path)` helper.

**`SLDgen/painter/painter.py`**
- `state_dict()` / `load_state_dict(d)` on `SLDBSplinePainter`, covering the
  optimised tensors, `is_active_cp` and the pinned tensors.
- `init_from_checkpoint(ckpt)`: mirrors `init_image()` but restores tensors
  instead of calling `initialize_control_points`. It must still perform the
  non-geometry parts of `init_image` — building `shape_groups`, and loading
  `avoid_points` / `attract_points`, which are re-derived from the SVG paths
  each run and are therefore *not* stored in the checkpoint. Returns
  `self.get_image()` as `init_image` does.

**`SLDgen/run.py`**
- `save_checkpoint(renderer, optimizer, args, epoch, resolved_caption)`.
- `load_checkpoint(path)` + fingerprint comparison.
- Loop becomes `for epoch in range(start_epoch + 1, stop_at + 1)`, where
  `start_epoch` is 0 or the checkpoint's epoch, and `stop_at` is
  `args.stop_at or args.num_iter`.
- Skip `save_current_step(epoch=0)` when resuming (that frame already exists).
- Guard finalisation, metrics and `make_video` behind `epoch == args.num_iter`.
- Write a checkpoint at `stop_at`, and every `--checkpoint-interval` epochs.

**`SLDgen/painter/painter_optimizer.py`**
Expose `state_dict()` / `load_state_dict()` passthroughs to `self.optim`, and
assert that the restored param-group names match the current ones (they will
differ if `--optimize-cp-weights` or `--width optim` changed — caught by the
fingerprint, but a defence in depth).

**`sldgen.py`**
Extend the module docstring in the existing style. Write `config.json` after
every checkpoint, not only at the end, so a paused run is self-documenting.

## 10. Test: `test_resume_geom.py`

Style follows `test_origin_geom.py`: CPU only, no diffusion model, runs in
seconds, invoked as `PYTHONPATH=. python test_resume_geom.py`.

Approach: drive the painter and optimiser directly with a **synthetic
deterministic loss** (e.g. pull the curve toward a fixed circle) in place of the
SDS loss, keeping the real regularisation losses and the real
`post_process_params`. Then:

1. Run 40 steps uninterrupted with a horizon of 200 → snapshot tensors.
2. Run 20 steps, checkpoint, construct a fresh painter/optimiser, resume,
   run 20 more → snapshot tensors.
3. Assert exact equality of `control_points`, `weights`, `width`,
   `is_active_cp`, and the Adam step counter.
4. Assert the sparse loss weight at epoch 20 is identical in both, confirming
   the horizon — not the segment length — drives the schedule.
5. Assert that resuming with a mismatched structural fingerprint raises.

A second, slower opt-in check (not part of the fast test) can run the real
pipeline for 20 iterations in two segments and compare final SVGs within a
tolerance, to catch integration mistakes that the synthetic loss cannot.

## 11. Phase 2 (flagged now, decide later): library API for a resident worker

The service is intended to time-slice several jobs on one GPU. If each slice is
a fresh `python sldgen.py --resume ...`, every slice pays for loading
SD3.5-medium, the ControlNet, the LoRA fuse and TAESD3 — plausibly 30–60 s of
pure overhead per switch.

Avoiding that requires two refactors, neither of which belongs in Phase 1:

1. **Session object.** Split `run()` into an `SLDSession` exposing
   `step()`, `run_until(epoch)`, `save_checkpoint()`, `load_checkpoint()`, with
   `run(args)` reduced to a thin wrapper. Behaviour unchanged.
2. **Shared pipeline.** `SD3GuidanceControl` currently owns both the pipeline
   and the per-job conditioning (control image latents, prompt embeddings), and
   `fuse_lora` bakes the LoRA into the pipeline at load. To share one pipeline
   across jobs, the per-job conditioning must be separable from the pipeline —
   and all jobs sharing a pipeline must agree on `lora_model`, `lora_weight`
   and `condition`.

**Recommendation:** ship Phase 1 as a pure CLI feature and let the first version
of the worker shell out. Measure the real switch cost against a realistic slice
length before deciding whether Phase 2 is worth the coupling. If slices are
5 minutes long, a 45-second reload is ~15% overhead — annoying but survivable;
if slices are 200 iterations, it dominates.

## 12. Non-goals

- Metrics on intermediate checkpoints (CLIP/DINO/aesthetic model loads are
  expensive; the service can compute these out-of-band if it wants them).
- Changing the number of control points mid-run.
- Branching a trajectory into two divergent children *inside* the core — that is
  a service concern, and is achieved simply by resuming the same checkpoint
  twice into different output directories.
- Any database, queue, HTTP surface or file-watching. Spec 2.