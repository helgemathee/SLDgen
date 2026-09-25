# SLDgen — Image Fidelity Loss Spec

Sep 25, 2026 · @Helge

## 1. Motivation

SLDgen's optimizer never compares the rendered curve against the target photograph. That is the problem this feature fixes.

The loop is driven by Score Distillation Sampling: DiffVG renders the current URBS curve, noise is added, the diffusion model (SD 3.5 + ControlNet + fused style LoRA) denoises it toward the text prompt, and the residual becomes a gradient on the control points. The target image enters only indirectly, in two places:

1. As the source for the ControlNet conditioning image (depth or canny), which biases the denoising direction.
2. As the source for the RMBG-1.4 mask driving TSP stipple initialization, which sets the curve's starting configuration.

Both are hints inside a loss that asks "does this look like a single-line drawing of a bearded man", not "does this match this bearded man". At CFG 100 the text prompt dominates further, so raising `--conditioning-scale` gives diminishing returns.

Observed symptoms, all consistent with that diagnosis:

- Portraits are plausible but not recognizable as the specific sitter.
- Semantic bleed: a beard continues into the shirt below, because the depth map has no break at the collar and nothing in the loss objects.
- High-frequency structure (curly hair, feather texture) is ignored even when present in the conditioning image.

The goal is **not** maximum fidelity. The interpretive quality of SDS output is the point of the tool. The goal is a controllable abstraction-to-fidelity axis where a setting of 0.2 means "one fifth of the optimization pressure comes from the photograph", and where that setting behaves identically across images, prompts and runs.

## 2. Design principle: normalized gradient blending

The weight must be blended at the **gradient** level, not the loss level. This is the central design decision and the implementation must not deviate from it.

SDS gradient magnitudes are not normalized in any meaningful way. They depend on the sampled diffusion timestep, the CFG scale and the per-step noise draw, and they vary by an order of magnitude between consecutive steps. A naive `loss = loss_sds + w * loss_image` therefore does not give a controllable dial: `w` is a coefficient whose effective influence wanders unpredictably during the run and is not comparable across images or prompts.

Instead, compute both gradients with respect to the control points, normalize each to unit L2 norm, then blend:

```python
g_sds = torch.autograd.grad(loss_sds, points, retain_graph=True)[0]
g_img = torch.autograd.grad(loss_image, points, retain_graph=True)[0]

g_sds = g_sds / (g_sds.norm() + 1e-8)
g_img = g_img / (g_img.norm() + 1e-8)

points.grad = (1.0 - alpha) * g_sds + alpha * g_img
```

With this formulation `alpha` has a defined meaning:

| alpha | Behaviour |
| --- | --- |
| 0.0 | Stock SLDgen, bit-for-bit (see the opt-in guarantee in section 3) |
| 0.2 | One fifth of directional pull comes from the image term |
| 0.5 | Equal contribution |
| 1.0 | Pure image fitting, no SDS |

The value is comparable across images, prompts and runs, which matters because the author is producing a portrait series where consistent parameters are wanted across subjects.

**Costs and caveats the implementer should know:**

- Two backward passes per step instead of one. Expect 20–30% wall-clock increase. This is acceptable and is not to be optimized away by fusing the losses.
- Global normalization (one scalar norm over the whole control point tensor) is specified, not per-point normalization. Per-point normalization destroys the relative magnitude structure within each gradient field and produces uniform-speed motion everywhere.
- The optimizer (Adam or whatever `painter_optimizer.py` uses) still applies its own adaptive scaling on top. That is fine and expected; the blend controls direction, the optimizer controls step size.
- Because `points.grad` is assigned directly, any existing code path that calls `loss.backward()` on a combined loss must be bypassed when the feature is active. Check for gradient accumulation across multiple parameter groups (e.g. control points, widths, colors) and handle each group consistently.

## 3. CLI surface

All flags are opt-in and default to off. Follow the same pattern established by `--origin`, `--avoid` and `--stipple-weight`: when the feature is unused, the code path must be byte-identical to upstream.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--image-loss-weight` | float | `0.0` | The alpha dial. `0.0` disables the feature entirely. Valid range 0.0–1.0. |
| `--image-loss-type` | choice | `chamfer` | One of `chamfer`, `pyramid`, `landmark`. May be given multiple times to sum several terms. |
| `--image-loss-target` | path | `None` | Image the loss is computed against. Falls back to `--target` if omitted. |
| `--image-loss-schedule` | choice | `constant` | One of `constant`, `decay`, `ramp`. See section 6. |
| `--image-loss-schedule-range` | 2 floats | `None` | Start and end alpha for `decay` / `ramp`. Defaults per schedule in section 6. |
| `--image-loss-sub-weights` | floats | `None` | Relative weights when several `--image-loss-type` values are given. Normalized to sum to 1. Defaults to uniform. |
| `--landmark-file` | path | `None` | JSON of landmark coordinates for `landmark` type. Required if that type is used. |
| `--log-gradient-norms` | flag | `False` | Diagnostic; see section 8. |

**The opt-in guarantee.** Gate on `args.image_loss_weight > 0.0`. When false, no new tensor is allocated, no second backward pass is taken, no target image is loaded, and `points.grad` is produced exactly as upstream produces it. A regression test asserts this (section 8).

**Separating `--image-loss-target` from `--target` is deliberate and important.** It lets the author hand the fidelity term a purpose-built raster — a contrast-boosted edge map, a hand-painted hairline, a region-masked structure map — while SDS continues to see the original photograph through the ControlNet path. This gives per-region control over what counts as "the image", which is more useful than any single global weight. Do not collapse these into one argument.

**Example invocation:**

```bash
python sldgen.py \
    --target /home/helge/SLDgen/data/portrait.png \
    --caption "a head and shoulders portrait of a smiling man with curly dark hair" \
    --condition canny \
    --conditioning-scale 0.9 \
    --n-control-points 400 \
    --num-iter 8000 \
    --render-size 512 \
    --image-loss-weight 0.25 \
    --image-loss-type chamfer \
    --image-loss-target /home/helge/SLDgen/weights/portrait_edges.png \
    --image-loss-schedule decay \
    --log-gradient-norms \
    --experiment-name portrait_v04_chamfer_025
```

## 4. Core implementation

### 4.1 Investigate first

Before writing code, read and report on the following. Do not assume the structure below is correct — it is reconstructed from prior sessions and may be stale.

- `SLDgen/painter_optimizer.py` — locate the training step. Identify exactly where the SDS loss is computed, where `.backward()` is called, which parameter groups exist (control points, and possibly widths, colors, opacities), and how `optimizer.step()` is invoked.
- `SLDgen/sd3_sds_guidance_control.py` — confirm what the SDS function returns: a scalar loss, a pre-computed gradient, or a surrogate loss using the `(grad * x).sum()` trick. **This determines the whole implementation.** Many SDS implementations bypass autograd and inject gradients directly; if that is the case here, `torch.autograd.grad(loss_sds, points)` will not work as written in section 2 and the surrogate must be differentiated instead.
- `SLDgen/painter.py` — how control points are stored, their tensor shape, and how DiffVG rasterizes them. Note the coordinate convention (pixel space vs normalized, y-up vs y-down) — the chamfer term must match it.
- `SLDgen/config.py` and the argparse block — where new flags are registered and how they reach the optimizer.
- Whether a raster of the current curve is already produced each step (for the SDS render). If so, reuse it for the pyramid loss rather than rasterizing twice.

Write findings to `specs/notes_image_loss_investigation.md` before implementing.

### 4.2 New module: `SLDgen/losses/image_fidelity.py`

Create a package `SLDgen/losses/` with `__init__.py`. The new module holds:

```python
class ImageFidelityLoss(nn.Module):
    def __init__(self, loss_types, sub_weights, target_path,
                 landmark_file=None, render_size=512, device="cuda"):
        ...
    def forward(self, points, raster=None):
        """Returns a scalar loss. `raster` is the current DiffVG render if
        one is already available, else None and the term rasterizes itself."""
```

Each loss type is a separate method or small class inside this module. Targets are precomputed once in `__init__` and cached on the device — no per-step disk I/O, no per-step canny extraction.

### 4.3 Schedule helper

```python
def alpha_at(step, num_iter, schedule, base_weight, schedule_range=None):
    """Returns the blend weight for this step. See section 6."""
```

Keep it pure and free of global state so it can be unit-tested directly.

### 4.4 Optimizer loop changes

In `painter_optimizer.py`, replace the single backward pass with the blend when active. Sketch, to be adapted to the actual structure found in 4.1:

```python
if self.image_loss is not None:
    alpha = alpha_at(step, self.num_iter, self.schedule,
                     self.image_loss_weight, self.schedule_range)

    loss_sds = self.sds_guidance(...)          # or the surrogate
    g_sds = torch.autograd.grad(loss_sds, self.points, retain_graph=True)[0]

    loss_img = self.image_loss(self.points, raster=current_raster)
    g_img = torch.autograd.grad(loss_img, self.points)[0]

    n_sds = g_sds.norm()
    n_img = g_img.norm()
    g_sds = g_sds / (n_sds + 1e-8)
    g_img = g_img / (n_img + 1e-8)

    self.optimizer.zero_grad()
    self.points.grad = (1.0 - alpha) * g_sds + alpha * g_img
    self.optimizer.step()
else:
    # upstream path, untouched
    ...
```

Points to get right:

- `retain_graph=True` on the first `autograd.grad` call only. Both terms may share graph nodes if the image loss consumes the same raster.
- If other parameter groups exist (widths, colors), they still need gradients. Either take them from the SDS backward as upstream does and leave them out of the blend, or blend them too if the image loss touches them. Document which choice was made and why.
- Guard against a zero-norm gradient. If `n_img` is near zero (the curve already sits exactly on the target, or the loss saturated), the normalization will amplify numerical noise into a unit vector of garbage. If either norm is below `1e-7`, skip the blend for that step and fall through to the other term alone. Log it.
- Do not call `loss.backward()` anywhere in the active branch — it will accumulate into `.grad` and corrupt the assignment.

## 5. Image loss implementations

Three types, listed in build order. `chamfer` is the priority and must work well before the others are started.

### 5.1 `chamfer` — one-directional chamfer distance to edge points

The default and the best-behaved term at low alpha.

**Setup, once at construction:**

1. Load `--image-loss-target` as greyscale.
2. If it is not already a binary edge map, extract edges (canny with configurable thresholds, exposed as `--image-loss-canny-low` / `--image-loss-canny-high`, defaults 100 / 200). If the image is already predominantly binary, use it directly.
3. Collect the coordinates of all edge pixels into an `(N, 2)` float tensor, in the same coordinate convention as the control points (confirmed in 4.1). Normalize to `[0, 1]` if the points are normalized.
4. If `N` exceeds roughly 20,000, subsample uniformly at random to that cap. Record the count in the run log.

**Per step:**

```python
curve_pts = sample_along_curve(points, n=2000)   # differentiable
d = torch.cdist(curve_pts, self.edge_pts)         # (2000, N)
loss = d.min(dim=1).values.mean()
```

**One-directional is the specification, not an approximation.** The symmetric chamfer (adding `d.min(dim=0)`) forces the curve to *cover* every edge pixel, which demands completeness and destroys the abstraction. The one-directional form says only "wherever the line goes, be near something real" — the line stays free to omit whatever it likes but cannot invent structure. This is exactly the semantics wanted for a partial-weight fidelity term. Do not make it symmetric, and do not offer a flag for it in this phase.

**`sample_along_curve` must be differentiable** with respect to the control points. Evaluate the URBS/B-spline basis at a fixed set of parameter values and take the weighted sum of control points. Do not sample from the DiffVG raster, and do not use `torch.linspace` over a detached curve. Verify by checking that `curve_pts.requires_grad` is true and that gradients reach `points`.

**Memory:** `cdist` on (2000, 20000) is 40M floats, roughly 160 MB in fp32. Acceptable on the target hardware (dual RTX 5090). If it becomes a problem, chunk the curve points, not the edge points.

**Why chamfer over raster L2 at low weight:** raster L2 has near-zero gradient wherever the curve is not already close to target ink, so at alpha 0.2 it contributes almost nothing until the curve happens to land nearby, then snaps hard. Chamfer has a smooth, long-range gradient everywhere.

### 5.2 `pyramid` — multi-scale raster comparison

A blunter term that matches tonal mass rather than edges. Useful when the author wants overall value distribution to track the photograph.

1. Build the target pyramid once: load the target, convert to greyscale, invert so dark subject matter is high value, and downsample to 64, 128, 256 and 512 (capped at `--render-size`).
2. Each step, render the curve via DiffVG (reuse the existing raster if one is available — see 4.1), and build the same pyramid from it by successive average pooling.
3. Loss is the sum of per-level L1 or L2 differences, each level weighted by `1 / level_resolution` so coarse levels are not swamped by fine ones.

The multi-scale structure is the point: a single-resolution raster loss lets the optimizer chase fine detail while global composition drifts. Expose `--image-loss-pyramid-levels` with default `[64, 128, 256, 512]`.

### 5.3 `landmark` — facial landmark attraction

The most targeted term for the portrait series, and the one most likely to fix recognizability specifically.

Generic losses treat every pixel as equally important, so eye corners and hoodie folds receive the same optimization pressure. For faces, identity is concentrated in maybe twenty points.

**Landmark file format.** A separate preprocessing script (section 7) produces JSON; the loss only reads it. This keeps MediaPipe and dlib out of the main runtime dependencies.

```json
{
  "image_size": [1024, 1536],
  "landmarks": [
    {"name": "left_eye_outer",  "xy": [412, 605], "weight": 3.0},
    {"name": "left_eye_inner",  "xy": [498, 611], "weight": 3.0},
    {"name": "mouth_left",      "xy": [455, 902], "weight": 2.0},
    {"name": "jaw_mid",         "xy": [512, 1105], "weight": 0.5}
  ]
}
```

**Per step:** for each landmark, find the minimum distance from that landmark to any sampled curve point, and take the weighted mean:

```python
d = torch.cdist(self.landmark_xy, curve_pts)   # (L, 2000)
loss = (self.landmark_w * d.min(dim=1).values).sum() / self.landmark_w.sum()
```

Note the direction: **landmark to nearest curve point**, the opposite of the chamfer term. Here the requirement is coverage — every important landmark must have line near it. That is the correct semantics for a small set of high-value anchors, and the wrong one for a dense edge map.

**Suggested default weights** if the preprocessing script assigns them automatically: eye corners and pupil centres 3.0, nostrils and mouth corners 2.0, brow arcs 1.5, jaw and face outline 0.5. These are a starting point for tuning, not a claim about optimal values.

### 5.4 Combining types

When several `--image-loss-type` values are given, compute each, normalize `--image-loss-sub-weights` to sum to 1, and take the weighted sum **before** the single gradient is computed. The alpha blend in section 2 then applies to that combined term. Do not compute separate gradients per sub-type; that would multiply the backward passes without benefit.

## 6. Weight schedules

A constant alpha is probably not what produces the best results, so the schedule is part of the feature rather than a later addition.

| Schedule | Default range | Behaviour |
| --- | --- | --- |
| `constant` | — | `alpha = --image-loss-weight` for the whole run. The baseline. |
| `decay` | 0.5 → 0.1 | Linear from start to end value. Composition locks to the photograph early, SDS stylizes on a correct armature afterwards. |
| `ramp` | 0.05 → 0.3 | Linear, reversed. SDS finds its interpretation freely, then the image term pulls it back toward the evidence. |

When `--image-loss-schedule-range` is supplied it overrides the defaults. When `constant` is selected, `--image-loss-weight` is the value and the range argument is ignored (warn if both are given).

```python
def alpha_at(step, num_iter, schedule, base_weight, schedule_range=None):
    t = step / max(num_iter - 1, 1)
    if schedule == "constant":
        return base_weight
    lo, hi = schedule_range or DEFAULTS[schedule]
    return lo + (hi - lo) * t
```

Note that for `decay` the range is given as (start, end) with start > end, so the same linear interpolation covers both cases. Validate that both values are within 0.0–1.0 and fail early with a clear message if not.

The two schedules produce visibly different aesthetics, not merely different fidelity levels. `decay` should preserve likeness better; `ramp` should preserve the interpretive quality better. Both need to exist so the author can A/B them cheaply.

## 7. Target preprocessing

Two small standalone scripts, kept out of the training runtime so MediaPipe and OpenCV stay optional dependencies.

### 7.1 `tools/prep_edge_target.py`

Produces the edge raster for `--image-loss-target`. The author currently does this by hand in Photoshop; the script makes it repeatable across a portrait series.

```
python tools/prep_edge_target.py INPUT.png OUTPUT.png \
    [--clahe-clip 3.0] [--clahe-grid 8] \
    [--canny-low 100] [--canny-high 200] \
    [--blur 1.0] [--mask MASK.png] [--suppress-background]
```

- CLAHE first, to lift shadow detail. This is what makes dark curly hair produce usable edges instead of a silhouetted blob.
- Optional Gaussian blur before canny, to suppress skin and fabric noise.
- `--mask` restricts edge extraction to a region, so the author can produce a hair-only or face-only target.
- `--suppress-background` flattens everything outside the RMBG foreground mask, removing wall texture.
- Always write greyscale PNG at the source resolution. Print the edge pixel count to stdout — if it is under a few thousand or over a few hundred thousand, the parameters are wrong and the user should know before starting an 8,000-iteration run.

**Known failure mode to document in the script's help text:** aggressive CLAHE lifts the interior of a dark region but also raises its edge toward the background value, which can erase the silhouette. If the subject outline matters, composite: prepped interior, original edge. A `--preserve-silhouette` option that takes the outer contour from the unprepped image and unions it into the output would be a useful addition if it is cheap to implement.

### 7.2 `tools/extract_landmarks.py`

```
python tools/extract_landmarks.py INPUT.png OUTPUT.json [--preset portrait]
```

Uses MediaPipe Face Mesh (468 points) or dlib's 68-point model, whichever is easier to install on the target machine. The `--preset portrait` selection reduces the full set to the roughly 20 identity-carrying points with the weights suggested in 5.3. Emit the full set under a separate key as well, so weights can be hand-edited later without re-running detection.

Coordinates in the JSON are in source-image pixel space, with `image_size` recorded. The loss module rescales to the control point convention at load time — put the rescaling in one place and test it, because a silent coordinate mismatch here will look exactly like "the feature does not work".

## 8. Verification

### 8.1 Default-path identity

The most important test. With `--image-loss-weight 0.0` (the default), a run with a fixed seed must produce a bit-identical SVG to the same run on the pre-change commit. Automate this: run both, compare file hashes. If the hashes differ, the opt-in guarantee is broken and the feature is not done, however well it works when enabled.

### 8.2 Unit tests

- `alpha_at` returns the expected values at step 0, midpoint and final step for all three schedules.
- `sample_along_curve` is differentiable: gradients reach the control points, and a synthetic loss pulling one sampled point moves the expected control point.
- Chamfer loss on a synthetic case: a straight line target and a curve offset by a known distance returns approximately that distance.
- Landmark loss direction: adding a curve point near a landmark decreases the loss; adding one far away does not increase it (the min is unaffected).
- Coordinate round-trip: a landmark at a known pixel position maps to the expected normalized position and back.

### 8.3 Gradient diagnostics

With `--log-gradient-norms`, write a CSV alongside the experiment output with one row per step: `step, alpha, sds_norm_raw, img_norm_raw, loss_sds, loss_img, cosine_similarity`.

The cosine similarity between the two normalized gradients is the most informative column. If it hovers near zero the terms are working in unrelated directions, which is expected and fine. If it is persistently negative the terms are fighting, and the run will be unstable — worth surfacing in the log summary. If it is near 1.0 the image term is redundant.

The raw norm columns also answer a question the author has been guessing at: how much SDS gradient magnitude actually varies during a run, and therefore how badly a non-normalized weight would have behaved.

### 8.4 Visual smoke test

At `--image-loss-weight 1.0`, `--image-loss-type chamfer`, the output should be an obvious trace of the edge map, with no interpretive quality at all. If it is not, the image loss or its coordinate mapping is wrong, and no intermediate alpha will be meaningful. Run this before any aesthetic evaluation.

### 8.5 Checkpointing interaction

The repo already has `--stop-at` / `--resume` / `--checkpoint-interval` from the earlier checkpointing spec. Confirm that the schedule resumes at the correct step after a restart, and that the image loss targets are reconstructed identically. Add the new flags to whatever run-configuration record the checkpoint stores, and refuse to resume with mismatched image-loss settings rather than silently producing a hybrid run.

## 9. Phasing and scope

### Build order

1. **Investigation** (4.1) and the written findings note. Stop here and report before implementing — in particular, whether the SDS path is autograd-compatible. If it is not, the blend formulation needs adjusting and that should be agreed before code is written.
2. **Plumbing**: flags, the `losses/` package, the schedule helper, the gradient blend, the diagnostic log. Wire it up with a trivial placeholder loss (e.g. distance to image centre) and confirm the blend mechanism works and the default path is unchanged.
3. **Chamfer loss** plus `tools/prep_edge_target.py`. This is the deliverable that makes the feature useful; everything after is elaboration.
4. **Pyramid loss.**
5. **Landmark loss** plus `tools/extract_landmarks.py`.

Each phase ends with the tests from section 8 that apply to it, and a working run on a real portrait.

### Explicitly out of scope

- Changing the SDS implementation, CFG scale, timestep sampling range or style LoRA weight. These are separate levers with their own trade-offs and will be explored independently; conflating them with this change makes both impossible to evaluate.
- Symmetric chamfer, or any "cover the whole target" variant.
- Per-region alpha maps (different fidelity weight in different parts of the image). Interesting, but it should wait until the global dial is understood. Note it as a possible follow-up in the findings.
- Any change to the default behaviour of existing flags.

### Conventions to follow

- Match the existing repo style for flag naming, config handling and file layout.
- Literal absolute paths in any deployment or example scripts, not `~` or shell variables.
- New work under `specs/` follows the existing naming; this document becomes `specs/04_image_fidelity_loss.md` or the next number in sequence.
- Keep `work/` gitignored as before.

## 10. Validation sweep

Once phase 3 lands, this is the sweep that determines whether the feature is worth keeping and where the house setting sits. The agent should run it and save the outputs, but not judge the results — that is the author's call.

Hold every other parameter constant. Use the curly-haired portrait, which has been the hardest case.

| Run | alpha | Schedule | Everything else |
| --- | --- | --- | --- |
| A | 0.0 | — | Baseline, equals current output |
| B | 0.15 | constant | identical |
| C | 0.30 | constant | identical |
| D | 0.50 | constant | identical |
| E | 0.30 | decay (0.5→0.1) | identical |
| F | 0.30 | ramp (0.05→0.3) | identical |

Shared settings for all six: `--condition canny`, `--conditioning-scale 0.9`, `--n-control-points 400`, `--num-iter 8000`, `--render-size 512`, same seed, same caption, same edge target from `prep_edge_target.py`.

Save all six SVGs plus their gradient-norm CSVs into one directory with a manifest recording the exact command line per run.

The expectation is that a usable band exists somewhere around 0.2–0.35, and that whatever it turns out to be becomes a standing default for the portrait series rather than a per-image adjustment. Runs E and F answer a different question — whether scheduling beats a constant at the same nominal strength — and their difference should be aesthetic rather than merely one being better.
