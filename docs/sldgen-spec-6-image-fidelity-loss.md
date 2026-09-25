# Spec 6 — Image fidelity loss (`--image-loss`)

**Status:** ✅ **phases 1–5 implemented (2026-09-25); validation sweep (§14) not yet run.** See §15.
**Scope:** a new opt-in loss term in SLDgen core, blended with SDS at the gradient
level; its service parameters and two new input roles; two preview endpoints; a
panel in the new-job form and in Run again; an edge-target tab and a diagnostics
panel on the job page; two standalone preprocessing scripts.
**Depends on:** Specs 1–3 (checkpointing, service, web UI) and Spec 5 (the Canny
edge-map code this reuses). Independent of Spec 4.

> The first draft of this document was written from memory of earlier sessions
> and got several things about the code wrong. §0 lists every correction, with the
> reason, so the two can be compared. Everything below §0 is the corrected design.

---

## 0. Corrections to the first draft

Read this first if you have read the earlier version.

| # | First draft said | The code says | Consequence for the design |
|---|---|---|---|
| 1 | The training step lives in `SLDgen/painter_optimizer.py`; SDS in `SLDgen/sd3_sds_guidance_control.py`; flags in `SLDgen/config.py` plus "the argparse block". | The loop is `SLDgen/run.py::run` (`for epoch in epoch_range:`). `SLDgen/painter/painter_optimizer.py` only wraps Adam. SDS is `SLDgen/guidance/sd3_sds_guidance_control.py`. All flags and their validation are in `SLDgen/config.py::parse_arguments`. Specs live in `docs/`, not `specs/`. | File names throughout. No `specs/notes_…` note: the investigation is §2 of this document. |
| 2 | "Confirm whether the SDS path is autograd-compatible… this determines the whole implementation." | It is. `SD3GuidanceControl.forward` returns the standard surrogate `0.5·MSE(latents, (latents − w·(ε̂−ε)).detach(), sum)`, whose gradient w.r.t. `latents` is exactly the SDS gradient; autograd then flows through the TAESD3 encoder and DiffVG's `RenderFunction` to the control points. The loop calls `loss.backward(retain_graph=True)`. | `torch.autograd.grad(loss_sds, control_points)` works, but is not even needed: after the existing SDS backward, `control_points.grad` *is* `g_sds`. §3. |
| 3 | Blend two unit-norm gradients; "any existing code path that calls `loss.backward()` on a combined loss must be bypassed". | The SDS backward is followed by **five regularisers** that `.backward()` at their own raw scale on top of it: repulsion (wiregrad), avoidance, attraction, sparsity (on the weights), length shortening. Normalising SDS to unit length would silently change its balance against all of them. | The blend is anchored to the **raw SDS norm**, not to 1: `g = (1−α)·g_sds + α·(‖g_sds‖/‖g_img‖)·g_img`. Same direction semantics, same magnitude as upstream, regularisers untouched. §3. |
| 4 | "Check for gradient accumulation across multiple parameter groups (control points, widths, colors)." | Groups are `control_points` (always), `weights` (default on, `--no-optimize-cp-weights` off), `widths` (only `--width optim`). No colours. | Blend `control_points` only; `weights` and `widths` keep the upstream SDS gradient. §3.3. |
| 5 | Write `sample_along_curve`: evaluate the B-spline basis at fixed parameters. | `SLDBSplinePainter.get_polyline_2d` already does exactly that every step and stores the result as `renderer.sampled_curve2d` (`--sampling-rate` = 5000 points, differentiable, canvas pixels, pinned origin/endpoints included). | No new sampler. Stride-subsample `sampled_curve2d` to `--image-loss-curve-samples`. §5.2. |
| 6 | Control-point coordinate convention "pixel vs normalized, y-up vs y-down" to be confirmed. | Canvas pixel coordinates, origin top-left, x right, y down, range `[0, render_size)`. `np.argwhere` on an edge map gives `(row, col)`; the loss needs `(x, y) = (col, row)`. | Stated as a contract; a test asserts it. §2.3. |
| 7 | `--image-loss-target` is any image; landmarks are in "source-image pixel space" and rescaled at load time. | The curve lives in **canvas space**: `--target` goes through RMBG mask → white background → square pad → resize → `--object-size-ratio` crop-and-centre (`targets.py`). Nothing in the repo registers a source-space image into that frame; Spec 5 §2 hit the same wall and resolved it by deriving edges *inside the run* from `args.input_image`, which *is* canvas space. | Default: derive the edge map in the run from `args.input_image`, reusing `canny_attract.edge_map`. A supplied `--image-loss-target` or landmark file must already be canvas space at `--render-size`, the same contract as `--stipple-weight`/`--avoid`/`--attract`, and is validated for size. The preprocessing scripts and the API previews run on a previous run's `input.png`, exactly as the Canny preview does. §5.1, §8, §10. |
| 8 | Gate on `--image-loss-weight > 0`. | Every opt-in feature here is a boolean gate plus knobs that are inert without it (`--attract-canny`), and the service/UI pattern is built on that (a `true_flag` that is absent from argv when off, knobs always emitted). With schedules, the weight is not the alpha anyway. | Explicit `--image-loss` gate. §4. |
| 9 | `--image-loss-type` repeatable, plus `--image-loss-sub-weights`. | `params.py`/`params.ts` have no list-of-choices or list-of-floats kind; `RunAgainDialog` and `ParamFields` have no multi-select widget; `sameValue` compares arrays in order. | Three floats instead: `--image-loss-chamfer`, `--image-loss-pyramid`, `--image-loss-landmark` (relative weights, 0 disables a term). Same expressiveness, zero new kinds. §4. |
| 10 | `--image-loss-schedule-range` as two floats with per-schedule defaults; `--image-loss-weight` ignored for `decay`/`ramp`. | Two floats fit the `float_pair` kind, but the generic field widget cannot edit a pair, and a weight that is "ignored" for two of three schedules is a trap in the UI. | `--image-loss-weight` is always the alpha the run **ends** on; `--image-loss-schedule-start` is where it begins (defaults 0.5 for `decay`, 0.05 for `ramp`). §7. |
| 11 | `--log-gradient-norms` flag. | Two norms, one dot product and one CSV line per step cost nothing measurable; a flag means one more param, and an operational-vs-structural argument. | Dropped. `image_loss_log.csv` is always written when `--image-loss` is on. §5.4. |
| 12 | `--image-loss-pyramid-levels` flag. | Another list kind for a knob nobody will tune. | Fixed at 64/128/256/512 capped at `--render-size`. |
| 13 | Two backward passes → "20–30% wall-clock increase". | The image term's backward is a `cdist` plus one basis matmul (chamfer, landmark) or one extra DiffVG backward (pyramid). SDS's backward goes through the transformer-scale graph the forward already built. | Expect single-digit percent for chamfer/landmark; pyramid a little more. Not worth optimising either way. |
| 14 | "Normalize to `[0, 1]` if the points are normalized." | They are not. | Dropped. |
| 15 | `--image-loss-target` "falls back to `--target`". | `--target` is the original photograph, the wrong frame (correction 7). | Falls back to in-run derivation from the canvas image. |
| 16 | Pyramid compares "tonal mass". | A one-pixel line can never match a photograph's darkness; a raw L1 would ask for more ink everywhere. | Each level is normalised to unit sum before the L1, so the term compares *where* the ink is, not how much. §6.2. |
| 17 | New losses under `SLDgen/losses/` package; scripts under `tools/`. | Constraint modules are flat (`avoidance.py`, `attraction.py`, `canny_attract.py`); CLI scripts sit at the repo root (`sld_canny_svg.py`, `sld_partition.py`). | `SLDgen/image_loss.py`; `sld_edge_target.py`, `sld_landmarks.py`. |
| 18 | "dual RTX 5090". | One RTX 5090, 32 GB (`CLAUDE.md`). | Memory budget unchanged. |
| 19 | No relation to existing features mentioned. | `--attract-canny` (Spec 5) is already a Canny-edge pull on the curve: two-sided, hinge with a dead zone, on the *control points*, added at raw loss scale. | The chamfer term is its one-sided, sampled-curve, normalised cousin. Both can run together. §2.4 says when to use which. |
| 20 | No service, worker or web design. | The feature is unusable from the UI without one, and every earlier feature shipped all three. | §9–§11. |

---

## 1. Motivation

SLDgen's optimiser never compares the rendered curve against the target
photograph. That is what this feature fixes.

The loop is Score Distillation Sampling: DiffVG renders the current B-spline,
noise is added in latent space, SD 3.5 Medium + ControlNet + the fused style LoRA
denoise it toward the caption, and the residual becomes a gradient on the control
points. The photograph enters only indirectly:

1. as the source of the ControlNet conditioning image (depth or Canny), which
   biases the denoising direction at `--conditioning-scale` (default 0.5);
2. as the source of the RMBG-1.4 mask that drives TSP stipple initialisation;
3. since Spec 5, optionally as Canny edges the attraction constraint pulls the
   control points toward.

Items 1 and 2 are hints inside a loss that asks "does this look like a single-line
drawing of a bearded man", not "does this match *this* bearded man". At CFG 100
the caption dominates further, so raising `--conditioning-scale` gives diminishing
returns. Item 3 helps with features but is a loss-scale coefficient (§2.4).

Observed symptoms, all consistent with that diagnosis:

- Portraits are plausible but not recognisable as the sitter.
- Semantic bleed: a beard continues into the shirt, because the depth map has no
  break at the collar and nothing in the loss objects.
- High-frequency structure (curly hair, feather texture) is ignored even when
  the conditioning image shows it.

The goal is **not** maximum fidelity. The interpretive quality of SDS output is
the point of the tool. The goal is a controllable abstraction-to-fidelity dial,
where 0.2 means "one fifth of the optimisation pressure comes from the photograph"
and means the same thing across images, prompts and runs, because the author is
producing a portrait series with one house setting.

## 2. What the code actually does (investigation)

Everything in this section was read from the code on 2026-09-25 and is what the
rest of the document builds on.

### 2.1 The step

`SLDgen/run.py::run`, per epoch:

```python
optimizer.zero_grad_()
raster_sld = renderer.get_image().to(args.device)     # (1,3,H,W), white bg, black ink
loss = sds_loss(raster_sld)                             # SD3GuidanceControl.forward
loss.backward(retain_graph=True)                        # populates .grad on every param group
# then, each at its own raw scale, accumulating into .grad:
#   repulsion (wg.repulsion_loss on renderer.sampled_curve3d)
#   avoidance / attraction (on renderer.active_control_points)      -- opt-in
#   sparsity (on renderer.weights, ramped by epoch / num_iter)
#   length shortening (on renderer.sampled_curve2d)
loss.backward()                                          # the regularisers, if any
optimizer.step_()                                        # Adam(betas=(0.9,0.9), eps=1e-6)
renderer.post_process_params()                           # clamp weights/widths, prune
```

`retain_graph=True` on the SDS backward is what lets the regularisers reuse the
same rasterisation graph. The image term reuses it the same way.

### 2.2 The SDS loss is a surrogate, and autograd flows through it

`SD3GuidanceControl.forward`: encode the raster to latents (TAESD3 encoder, **with
grad**), sample `t ∈ [0.02, 0.98]·T`, run ControlNet + transformer under
`no_grad`, form `grad = clamp(σ_t²·(ε̂_cfg − ε), −1, 1)`, and return

```python
target = (latents - grad).detach()
sds_loss = 0.5 * F.mse_loss(latents, target, reduction="sum")
```

so `∂sds_loss/∂latents = grad` and the chain continues through the encoder and
DiffVG to `control_points`, `weights` and (with `--width optim`) `width`. The
magnitude of `grad` carries `σ_t²` from a fresh random `t` every step, which is
the reason a loss-level weight cannot be a stable dial (§3).

### 2.3 Geometry and coordinate frame

- `renderer.control_points`: `(N, 2)` float32 leaf, canvas pixels, x right, y
  down, `[0, render_size)`. With `--origin`/`--fixed-endpoints` the pinned points
  live in separate no-grad tensors and are concatenated in the forward pass.
- `renderer.sampled_curve2d`: `(sampling_rate, 2)`, the curve evaluated at 5000
  fixed parameter values through the weighted, normalised B-spline basis.
  Differentiable w.r.t. control points and weights. Recomputed by every
  `get_image()`, so it is fresh when the loop reaches the loss code.
- `renderer.active_control_points`: the optimised subset (pruned points removed).
- `get_image()` returns the DiffVG raster, `(1, 3, H, W)`, 1.0 = white paper,
  0.0 = ink.
- **Canvas space** is what `targets.get_target` produces: `args.input_image`
  (PIL RGB, white background, subject masked, square, `render_size`², object
  rescaled to `--object-size-ratio` and centred) and `args.mask` (float tensor,
  same frame). Both are saved to the run directory as `input.png` / `mask.png`.
  The loop's `inputs` tensor is `args.input_image` as `(1, 3, H, W)`.

### 2.4 The cousin: `--attract-canny`

Spec 5 already pulls the curve toward Canny edges. Differences, so the two are
not confused:

| | `--attract-canny` (Spec 5) | chamfer term (this spec) |
|---|---|---|
| What moves | control points | sampled curve (2000 points along the line) |
| Direction | two-sided: curve→edges *and* edges→curve (coverage) | one-sided: curve→edges only |
| Shape | hinge, inert inside a 25 px dead zone, quadratic beyond | mean nearest-edge distance, active everywhere |
| Strength | `--attraction-weight` × raw loss, added to SDS at whatever scale SDS has that step | a fraction of the SDS gradient's *own* norm, so "0.2" means 0.2 |
| Point budget | ~400 (coverage term sums over every target) | up to 20 000 edge pixels (no coverage term) |

Attraction is a leash: the curve may not wander far from the structure. The
chamfer term is a dial: wherever the line goes, it prefers to sit on evidence.
They compose; both are opt-in.

### 2.5 Plumbing the rest of the toolchain expects

- **Flags**: argparse in `config.py::parse_arguments`, validation with
  `parser.error` right after parsing, opt-in knobs inert without their gate.
- **Fingerprint**: `checkpoint.STRUCTURAL_FIELDS` names every run-shaping flag;
  resume refuses any mismatch (Spec 1 §6). Paths are fingerprinted as strings.
- **Service**: `sldgen_service/params.py::PARAM_SPECS` is the single CLI
  translator (`params_to_argv` ↔ `argv_to_params` must round-trip). Kinds: `int
  float str float_or_str true_flag false_flag float_pair path path_list`. File
  params are never set directly; they are **input roles** (`jobs.ROLE_PARAMS`)
  copied into `jobs/<id>/inputs/` at creation and inherited by Run again.
- **API venv** has no numpy/cv2/PIL/torch. Anything that computes runs under
  `config.sldgen_python` as a subprocess of a root-level script
  (`sld_canny_svg.py` → `sldgen_api/canny.py`).
- **Web**: `lib/params.ts` mirrors `PARAM_SPECS` name-for-name and
  default-for-default (`test_service_web.py` parses it); custom panels are
  stateless over `(params, onChange)` and hide their knobs from the generic
  field list; file inputs go through `INPUT_ROLES` → `toInputs()`.
- **Artifacts**: anything written under the run directory is listed by
  `jobs.artifacts()` (`.csv` and `.png` already have kinds) and served at
  `/api/jobs/{id}/files/target/run/<name>`. No endpoint work for new files.
- **Fake**: `test_support/fake_sldgen.py` stands in for `sldgen.py` in every
  service test and must accept the new flags and produce the new artifacts.

## 3. Design principle: normalised gradient blending

The weight is applied at the **gradient** level, not the loss level. This is the
central decision and the implementation must not deviate from it.

The SDS gradient's magnitude carries `σ_t²` from a random timestep and the
clamp, and varies by an order of magnitude between consecutive steps. A loss-level
`loss_sds + w·loss_img` therefore gives a `w` whose effective influence wanders
through the run and is not comparable across images or prompts.

### 3.1 The blend

With `g_sds = ∂loss_sds/∂control_points` and `g_img = ∂loss_img/∂control_points`:

```python
n_sds, n_img = g_sds.norm(), g_img.norm()          # one scalar each, over the whole tensor
g = (1 - alpha) * g_sds + alpha * (n_sds / n_img) * g_img
```

The image gradient is rescaled to the SDS gradient's norm and the two are mixed
by `alpha`. Direction-wise this is identical to the first draft's unit-norm
blend; magnitude-wise the result stays at SDS scale (`‖g‖ ≤ n_sds`, with equality
when the two agree), so:

- at `alpha = 0` the tensor is `g_sds` **exactly**, not a rescaled copy;
- the five regularisers (§2.1) keep precisely the balance against SDS they have
  upstream;
- Adam's own normalisation is untouched by the choice (Adam is scale-invariant
  up to `eps`; what matters is the *relative* balance, which this preserves).

| alpha | Behaviour |
|---|---|
| 0.0 | Not a valid setting — leave `--image-loss` off instead (§4). |
| 0.2 | One fifth of the directional pull on the control points comes from the image term. |
| 0.5 | Equal contribution. |
| 1.0 | Pure image fitting; SDS still shapes `weights` and `widths` (§3.3) and the regularisers still act. |

**Global normalisation** (one scalar per gradient) is specified, not per-point:
per-point normalisation flattens the magnitude structure inside each field and
produces uniform-speed motion everywhere.

### 3.2 Zero-norm guard

If `n_img < 1e-7` (the curve already sits on the evidence, or the term
saturated) the rescale would amplify numerical noise into a full-strength
direction. If `n_img` is below the threshold, leave `g_sds` untouched for that
step. If `n_sds` is below it, use `g_img` at its own scale. Either case is
logged (`skipped` column, §5.4).

### 3.3 Which parameter groups

Only `control_points` is blended. `weights` and `widths` receive the upstream
SDS gradient and nothing from the image term:

- The image terms are geometric; letting them push the B-spline weights would
  fight the sparsity schedule and the monotone pruning, both of which are tuned
  against SDS alone.
- `torch.autograd.grad(loss_img, [control_points])` is then also the cheapest
  possible call.

### 3.4 Where it sits in the step

The upstream SDS backward is kept as it is. Immediately after it, and before the
regularisers run, the control-point gradient is *rewritten*:

```python
loss = sds_loss(raster_sld)
loss.backward(retain_graph=True)                                   # unchanged

if image_loss is not None:                                         # --image-loss only
    alpha = alpha_at(epoch, args.num_iter, args.image_loss_schedule,
                     args.image_loss_weight, args.image_loss_schedule_start)
    loss_img, parts = image_loss(renderer, raster_sld)             # shares the graph
    g_sds = renderer.control_points.grad.detach().clone()
    (g_img,) = torch.autograd.grad(loss_img, renderer.control_points, retain_graph=True)
    blended, stats = blend_gradients(g_sds, g_img, alpha)
    renderer.control_points.grad.copy_(blended)
    image_log.write(epoch, alpha, stats, loss.item(), loss_img.item(), parts)

# regularisers accumulate on top, exactly as upstream
```

`retain_graph=True` on the `autograd.grad` is required: the regularisers still
need `sampled_curve2d`/`sampled_curve3d` afterwards. The final regulariser
backward frees the graph, as upstream. With `--keep-low-weights` off (the
default) pruned points have zero gradient from every term and contribute nothing
to either norm.

**The opt-in guarantee.** Without `--image-loss`: `image_loss is None`, no target
is loaded, no tensor is allocated, no extra backward runs, no file is written,
and `.grad` is produced exactly as upstream produces it. §12.1 verifies this.

## 4. Interface

```
--image-loss                          gate; everything below is inert without it
--image-loss-weight          0.2      alpha in (0, 1]. With constant: the alpha.
                                      With decay/ramp: the alpha the run ENDS on.
--image-loss-schedule        constant constant | decay | ramp                        (§7)
--image-loss-schedule-start  None     alpha at epoch 0 for decay/ramp; defaults 0.5 / 0.05
--image-loss-chamfer         1.0      relative weight of the chamfer term  (0 disables)
--image-loss-pyramid         0.0      relative weight of the pyramid term  (0 disables)
--image-loss-landmark        0.0      relative weight of the landmark term (0 disables)
--image-loss-target          None     canvas-space PNG at --render-size: an edge map,
                                      or an image to run Canny over. Omitted: derived
                                      in the run from the canvas image.           (§5.1)
--image-loss-canny-low       100.0    Canny thresholds and blur for the derived / non-binary
--image-loss-canny-high      200.0      target (same defaults as --attract-canny and the
--image-loss-canny-blur      3          Canny ControlNet)
--image-loss-curve-samples   2000     points taken along the curve for chamfer/landmark
--image-loss-landmarks       None     canvas-space landmark JSON (§6.3); required when
                                      --image-loss-landmark > 0
```

Validation in `parse_arguments`, all before any model loads:

- `--image-loss-weight` in `(0, 1]`; `--image-loss-schedule-start` in `[0, 1]`.
- `decay` needs `start > weight`; `ramp` needs `start < weight`; `constant`
  with a start given → error ("start only applies to decay/ramp").
- the three term weights ≥ 0 and at least one > 0.
- `--image-loss-landmark > 0` requires `--image-loss-landmarks`; the file must
  exist. `--image-loss-target` must exist if given.
- `low < high`; `blur ≥ 0`; `curve_samples ≥ 2` and `≤ --sampling-rate`.
- Knobs given **without** `--image-loss` are inert, including the two files
  (a warning is printed, not an error, because Run again inherits input files
  and a user may switch the gate off on a derived job).

**Separating the target from `--target` is deliberate.** It lets the author hand
the fidelity term a purpose-built raster (a CLAHE-lifted edge map, a hand-painted
hairline, a face-only region) while SDS keeps seeing the photograph through the
ControlNet. That is per-region control over what counts as "the image". Do not
collapse the two. What the first draft missed is that the raster must be in
canvas space; §8 gives the tool that produces one.

**Example:**

```bash
python sldgen.py \
    --target /home/helge/SLDgen/data/portrait.png \
    --caption "a head and shoulders portrait of a smiling man with curly dark hair" \
    --condition canny --conditioning-scale 0.9 \
    --n-control-points 400 --num-iter 8000 --render-size 512 \
    --image-loss --image-loss-weight 0.25 \
    --image-loss-schedule decay --image-loss-schedule-start 0.5 \
    --image-loss-target /home/helge/SLDgen/work/portrait_edges.png \
    --experiment-name portrait_v04_chamfer_025
```

## 5. Core implementation

### 5.1 `SLDgen/image_loss.py`

Flat module beside `attraction.py`. Imports torch, numpy, and
`canny_attract.edge_map` (cv2). No diffusers, no pydiffvg, so
`test_image_loss_geom.py` runs on a CPU in seconds.

```python
class ImageFidelityLoss:
    def __init__(self, args, input_image, mask, canvas_tensor, device):
        """Precompute every target once. No per-step disk I/O or Canny."""
    def __call__(self, renderer, raster):
        """-> (scalar loss, {"chamfer": float, "pyramid": float, "landmark": float})"""

def blend_gradients(g_sds, g_img, alpha, eps=1e-7):
    """-> (blended tensor, {"sds_norm", "img_norm", "cosine", "skipped"})"""

def alpha_at(epoch, num_iter, schedule, weight, start=None):
    """Pure function of the absolute epoch and the horizon (Spec 1 §3)."""

def curve_samples(renderer, n):
    """Stride-subsample renderer.sampled_curve2d to about n points. Differentiable."""

def edge_points(edge_map_u8, cap=20000):
    """(K, 2) float tensor of (x, y) for every edge pixel, stride-subsampled to cap."""
```

**Building the edge target** (chamfer), once, in `__init__`:

1. If `--image-loss-target` is given: load greyscale, refuse unless it is
   `render_size × render_size` ("canvas-space PNG expected; run
   `sld_edge_target.py` on a previous run's `input.png`"). If ≥ 99 % of its
   pixels are 0 or 255 it is an edge map and is used as is; otherwise Canny runs
   over it with the `--image-loss-canny-*` settings.
2. Otherwise: `canny_attract.edge_map(input_image, mask=mask, low, high, blur)`.
3. In both cases `mask` is applied (`edges[~inside] = 0`), so the square-pad
   border never counts as an edge. The silhouette survives on the inside of the
   mask boundary, as with `--attract-canny`.
4. Write the effective map to `<output_dir>/image_loss_target.png` so the job
   page can show it (§11) and the author can see what the term saw.
5. `edge_points`: `np.argwhere(edges > 0)` → swap to `(x, y)` → float32 → if `K
   > 20000`, keep every `ceil(K/20000)`-th point (a **fixed stride**, not a random
   draw, so a resumed segment rebuilds the identical tensor without touching the
   RNG). Print `K` before and after; under ~2000 or over ~200 000 raw pixels the
   thresholds are probably wrong and the log should say so.

**Pyramid target**: from `canvas_tensor` (the loop's `inputs`), greyscale,
inverted (`ink = 1 − grey`, which is 0 on the white background), pooled to each
level with `F.adaptive_avg_pool2d`, each level divided by its sum.

**Landmark target**: §6.3.

### 5.2 Curve samples

`renderer.sampled_curve2d` has `--sampling-rate` points (5000). Chamfer and
landmark use `curve_samples(renderer, args.image_loss_curve_samples)`:
`pts[::max(1, len(pts) // n)]`. That is a view of a differentiable tensor; no new
basis evaluation, no second rasterisation. `cdist(2000, 20000)` is 160 MB in
fp32, fine on 32 GB; if it ever is not, chunk the curve points, never the edges.

### 5.3 Wiring in `run.py`

- Construct right after the `--attract-canny` block (canvas space exists, the
  painter is not yet built, nothing here draws from the RNG):
  ```python
  image_loss = None
  if args.image_loss:
      image_loss = ImageFidelityLoss(args, args.input_image, args.mask, inputs, args.device)
      print(f"\tImage fidelity loss: {image_loss.describe()}", flush=True)
  ```
- The loop change is §3.4, verbatim.
- `image_loss_log.csv` handling: §5.4.

### 5.4 `image_loss_log.csv`

Written to `<output_dir>/image_loss_log.csv` whenever `--image-loss` is on, one
row per epoch, flushed each write:

```
epoch,alpha,sds_norm,img_norm,cosine,skipped,loss_sds,loss_img,chamfer,pyramid,landmark
```

`cosine` is between the raw `g_sds` and `g_img`. Near 0: the terms pull in
unrelated directions (expected). Persistently negative: they fight and the run
will be unstable. Near 1: the image term is redundant. `sds_norm` answers a
question the author has been guessing at: how much the SDS magnitude actually
varies, and therefore how badly a loss-level weight would have behaved.

**Resume.** Each segment is a new process in the same directory. On `--resume`
the file is opened, rows with `epoch > checkpoint epoch` are dropped (a segment
killed between checkpoint and SIGTERM leaves them behind), and the segment
appends. `test_image_loss_run.py` asserts a segmented run produces a CSV
byte-identical to the uninterrupted one.

### 5.5 Fingerprint and checkpointing

Add to `checkpoint.STRUCTURAL_FIELDS`, in this order, after the
`attract_canny_*` block:

```
image_loss, image_loss_weight, image_loss_schedule, image_loss_schedule_start,
image_loss_chamfer, image_loss_pyramid, image_loss_landmark,
image_loss_target, image_loss_canny_low, image_loss_canny_high, image_loss_canny_blur,
image_loss_curve_samples, image_loss_landmarks
```

Paths fingerprint as strings (precedent: `avoid`, `attract`). The two files are
**not** init-only: they are used every step, so resume segments must keep
passing them (the service already resolves them to the same absolute
`jobs/<id>/inputs/...` path on every segment). Targets are rebuilt from the same
bytes with deterministic code, so a resumed segment sees identical tensors; the
schedule is a function of the absolute epoch, so it resumes at the right alpha.
Nothing new goes in the checkpoint payload.

## 6. Loss terms

Build order: chamfer, then pyramid, then landmark. Chamfer must work well before
the others are started.

### 6.1 `chamfer` — one-directional distance from the curve to the edges

```python
pts = curve_samples(renderer, n)                 # (S, 2), differentiable
d = torch.cdist(pts, self.edge_pts)              # (S, K)
chamfer = d.min(dim=1).values.mean()             # mean nearest-edge distance, in px
```

**One-directional is the specification, not an approximation.** Adding
`d.min(dim=0)` would make the curve *cover* every edge pixel, which demands
completeness and destroys the abstraction. One-directional says only "wherever
the line goes, be near something real": the line may omit whatever it likes but
cannot invent structure. Do not make it symmetric, and do not add a flag for it.
(Coverage is what `--attract-canny` is for; §2.4.)

The value is in pixels and is a legible metric on its own ("the line sits on
average 3.1 px from an edge"), which is why the CSV records it.

**Why chamfer rather than raster L2 at low alpha:** raster L2 has near-zero
gradient wherever the curve is not already close to target ink, so at 0.2 it
contributes nothing until the curve happens to land nearby, then snaps. Chamfer
has a smooth, long-range gradient everywhere.

### 6.2 `pyramid` — multi-scale ink distribution

A blunter term for overall composition. Levels 64/128/256/512, capped at
`--render-size`.

```python
ink = 1.0 - raster.mean(dim=1, keepdim=True)     # (1,1,H,W), reuses the SDS raster
for size, target in zip(self.levels, self.pyr_target):
    p = F.adaptive_avg_pool2d(ink, size)
    p = p / (p.sum() + 1e-8)                     # a distribution, not a mass
    loss += (p - target).abs().sum()
loss /= len(self.levels)
```

Normalising each level to unit sum is the correction from §0 #16: without it the
term asks for more ink everywhere, since a line's total ink is a small fraction
of a photograph's darkness. With it, the term compares *where* the ink is. Equal
level weights: after normalisation every level's L1 is in `[0, 2]` and
comparable, so the first draft's `1/resolution` weighting is not needed.

This term backpropagates through DiffVG a second time (the raster is the input),
which is why it costs more than the other two.

### 6.3 `landmark` — coverage of a few high-value anchors

Identity in a face is concentrated in perhaps twenty points; generic losses give
an eye corner and a hoodie fold the same pressure.

**File format** (`--image-loss-landmarks`), produced by `sld_landmarks.py` (§8.2):

```json
{
  "space": "canvas",
  "image_size": [512, 512],
  "preset": "portrait",
  "landmarks": [
    {"name": "left_eye_outer", "xy": [212.4, 230.1], "weight": 3.0},
    {"name": "mouth_left",     "xy": [231.0, 331.8], "weight": 2.0},
    {"name": "jaw_mid",        "xy": [256.2, 402.5], "weight": 0.5}
  ],
  "all_landmarks": [[x, y], ...]
}
```

The loss refuses the file unless `space == "canvas"` and `image_size ==
[render_size, render_size]`. There is no rescaling path: a coordinate mismatch
here looks exactly like "the feature does not work", so the contract is the same
canvas-space-only contract every other spatial input in this repo has.
`all_landmarks` is informational (the full detector output, so weights can be
hand-edited without re-running detection).

```python
d = torch.cdist(self.landmark_xy, pts)                       # (L, S)
landmark = (self.landmark_w * d.min(dim=1).values).sum() / self.landmark_w.sum()
```

Direction: **landmark → nearest curve point**, the opposite of chamfer. Here the
requirement *is* coverage: every anchor must have line near it. Correct for
twenty anchors, wrong for twenty thousand edge pixels.

### 6.4 Combining

```python
w = {chamfer: args.image_loss_chamfer, pyramid: ..., landmark: ...}   # zeros drop the term
w = {k: v / sum(w.values()) for k, v in w.items() if v > 0}
loss_img = sum(w[k] * term[k] for k in w)
```

One weighted scalar, **one** `autograd.grad`, then the blend of §3. Do not take
separate gradients per term; that multiplies the backward passes without
benefit. The per-term values are logged (§5.4) so the author can see which term
is doing the work.

## 7. Schedules

A constant alpha is probably not what produces the best results, so the schedule
ships with the feature.

| Schedule | alpha over the run | Default start | Intent |
|---|---|---|---|
| `constant` | `weight` throughout | — | the baseline |
| `decay` | linear `start → weight`, `start > weight` | 0.5 | lock composition to the photograph early, let SDS stylise on a correct armature |
| `ramp` | linear `start → weight`, `start < weight` | 0.05 | let SDS find its interpretation freely, then pull it back toward the evidence |

```python
DEFAULT_START = {"decay": 0.5, "ramp": 0.05}

def alpha_at(epoch, num_iter, schedule, weight, start=None):
    if schedule == "constant":
        return weight
    lo = DEFAULT_START[schedule] if start is None else start
    t = epoch / max(num_iter - 1, 1)
    return lo + (weight - lo) * t
```

`epoch` is the absolute epoch and `num_iter` the horizon, never "iterations in
this process", so a segmented run is identical to an uninterrupted one (Spec 1
§3). The two schedules should produce visibly different *aesthetics*, not only
different fidelity: `decay` should preserve likeness better, `ramp` the
interpretive quality. Both exist so they can be A/B'd cheaply (§14).

## 8. Preprocessing scripts

Two root-level scripts in the style of `sld_canny_svg.py`. They take
**canvas-space input only**: a run's `input.png` (and `mask.png`). Pointing them
at the original photograph produces a target in the wrong frame, silently, and
their help text says so. They run under the `sldgen` conda env (cv2 is already
there; MediaPipe is an optional extra) and are what the API previews shell out to.

### 8.1 `sld_edge_target.py`

```
python sld_edge_target.py --image work/jobs/<id>/target/run/input.png \
    --mask  work/jobs/<id>/target/run/mask.png --out edges.png \
    [--low 100] [--high 200] [--blur 3] \
    [--clahe-clip 0] [--clahe-grid 8] [--roi X0 Y0 X1 Y1] [--preserve-silhouette]
```

- With only `--low/--high/--blur` it calls `canny_attract.edge_map` with the
  same arguments the run uses, so its output **is** what `--image-loss` derives
  in-run. The API preview relies on this (§10.1).
- `--clahe-clip > 0` applies CLAHE before Canny to lift shadow detail: this is
  what turns dark curly hair into usable edges instead of a silhouetted blob.
- `--roi` keeps edges inside a canvas-pixel box ("the face, not the shoulders").
- `--preserve-silhouette` unions the mask's outer contour back in: aggressive
  CLAHE raises a dark region's edge toward the background value and can erase
  the outline.
- Always writes a 0/255 greyscale PNG at the input's size and prints the edge
  pixel count with a warning outside ~2 000–200 000.

The first draft's `--suppress-background` is the mask: the run's `input.png` is
already background-suppressed, and `--mask` zeroes anything outside the subject.

### 8.2 `sld_landmarks.py`

```
python sld_landmarks.py --image work/jobs/<id>/target/run/input.png --out landmarks.json \
    [--preset portrait|all]
```

MediaPipe Face Mesh (468 points) via `pip install mediapipe` in the `sldgen`
env; a clear error naming that command if it is missing. `--preset portrait`
selects ~20 identity-carrying points with the weights from §6.3 (eye corners and
pupils 3.0, nostrils and mouth corners 2.0, brow arcs 1.5, jaw/outline 0.5); the
full set goes under `all_landmarks`. Coordinates are canvas pixels; `image_size`
is the input's size. Exit code 2 with a message when no face is found.

## 9. Service and worker

### 9.1 Parameters (`sldgen_service/params.py`)

Thirteen `ParamSpec`s, all `STRUCTURAL`, in the order of §4, appended after the
`attract_canny_*` block:

| name | flag | kind | default |
|---|---|---|---|
| `image_loss` | `--image-loss` | `true_flag` | `False` |
| `image_loss_weight` | `--image-loss-weight` | `float` | `0.2` |
| `image_loss_schedule` | `--image-loss-schedule` | `str` | `"constant"` |
| `image_loss_schedule_start` | `--image-loss-schedule-start` | `float` | `None` |
| `image_loss_chamfer` | `--image-loss-chamfer` | `float` | `1.0` |
| `image_loss_pyramid` | `--image-loss-pyramid` | `float` | `0.0` |
| `image_loss_landmark` | `--image-loss-landmark` | `float` | `0.0` |
| `image_loss_target` | `--image-loss-target` | `path` | `None` |
| `image_loss_canny_low` | `--image-loss-canny-low` | `float` | `100.0` |
| `image_loss_canny_high` | `--image-loss-canny-high` | `float` | `200.0` |
| `image_loss_canny_blur` | `--image-loss-canny-blur` | `int` | `3` |
| `image_loss_curve_samples` | `--image-loss-curve-samples` | `int` | `2000` |
| `image_loss_landmarks` | `--image-loss-landmarks` | `path` | `None` |

No new kind is needed (§0 #9, #10). `validate_params` mirrors §4's rules,
gated on `params["image_loss"]` like the canny block. A `None` default means the
flag is absent from argv; every other knob is emitted at its default as all
scalar params are, which is harmless because the core gate is the flag.

### 9.2 Input roles (`sldgen_service/jobs.py`)

Two new roles, single-valued: `image_loss_target` → param `image_loss_target`,
`image_loss_landmarks` → param `image_loss_landmarks`. They go in `ROLE_PARAMS`,
the refuse-direct list in `create_job`, and the nulling list in `run_again`, so
Run again inherits them automatically through `_copy_parent_inputs`. They are
**not** in `SPATIAL_ROLES`: that guard compares SVG viewBoxes, and these are a
PNG and a JSON; the run validates their size against `--render-size` instead
(§5.1, §6.3), which the service surfaces as an ordinary validation failure
(exit 2).

Sources accepted, as for the other roles: an upload (`source_kind: upload`,
content-addressed sha256; a `.json` suffix already works through the
`{sha}.*` fallback), or a file from another job (`source_kind: job`, e.g.
`target/run/image_loss_target.png` or `target/run/condition_canny.png` of a
run on the same target).

### 9.3 Worker and fake

No worker change: `build_argv` emits the new params and resolves the two paths
against the work root like every other path param.

`test_support/fake_sldgen.py` gains the flags, and when `--image-loss` is set:
runs the **real** `canny_attract.edge_map` over its `input.png` (or copies the
supplied target) to write `image_loss_target.png`, and writes one
`image_loss_log.csv` row per epoch with the resume truncation-and-append of
§5.4. As with Spec 5, the fake runs the real derivation so the file the UI
previews is the file SLDgen writes.

### 9.4 Config

`ServiceConfig` gains `edge_script` (`SLDGEN_EDGE_SCRIPT`, default
`REPO_ROOT/sld_edge_target.py`) and `landmark_script`
(`SLDGEN_LANDMARK_SCRIPT`, default `REPO_ROOT/sld_landmarks.py`), beside
`canny_script`. `deploy/README.md` lists both.

## 10. API

Both endpoints follow `sldgen_api/canny.py`: find a source run on the same
target, shell out under `config.sldgen_python`, write to `work/tmp/`, serve the
result. New module `sldgen_api/image_loss.py`; routes in `app.py` next to the
canny ones.

### 10.1 Edge-target preview

```
POST /api/image-loss/preview
  {source_job_id? , target_sha256?, params: {low, high, blur, clahe_clip?, clahe_grid?, roi?, preserve_silhouette?}}
→ {source_job_id, sha256, edge_pixels, derived_equivalent, edge_url, image_url, stdout}
GET  /api/image-loss/preview/{source_job_id}.png
```

- Source-run lookup is `canny.find_source_job` (newest job on the target whose
  `input.png` exists); 404 with the same "run this image once first" message
  when there is none.
- Runs `sld_edge_target.py --image input.png --mask mask.png --out
  work/tmp/edge-{source_job_id}.png ...`.
- **Stores the result as an upload** (`job_files.store_upload`, content-addressed)
  and returns its `sha256`, so the client can attach it as the
  `image_loss_target` input without a second round trip.
- `derived_equivalent` is `true` when only `low/high/blur` were used: the run's
  in-run derivation reproduces that map exactly (§8.1), so the panel need not
  attach a file. It is `false` once CLAHE, ROI or silhouette preservation is in
  play, and then the panel must attach the `sha256`.

### 10.2 Landmark extraction

```
POST /api/image-loss/landmarks
  {source_job_id?, target_sha256?, preset: "portrait"}
→ {source_job_id, sha256, count, landmarks: [{name, xy, weight}], image_url}
```

Runs `sld_landmarks.py`, stores the JSON as an upload, returns its `sha256` and
the weighted points so the panel can overlay them. A missing MediaPipe or "no
face found" comes back as 422 with the script's message.

### 10.3 Nothing else

`image_loss_target.png` and `image_loss_log.csv` are served by the existing
`/api/jobs/{id}/files/target/run/...` route and listed in `artifacts` (kinds
`png`, `csv`). Params, command, run-again and lineage need no change.

## 11. Web UI

### 11.1 Schema mirror and plumbing

- `lib/params.ts`: the thirteen specs (defaults before labels, so
  `test_service_web.py` can parse them), section `losses`, the two paths with
  `optional: true, viaInput: true`; `validateParams` mirrors §4; export
  `IMAGE_LOSS_PARAMS`.
- `api/types.ts`: `JobInput.role` gains the two roles; `ImageLossPreview` and
  `LandmarkExtract` response types. `api/client.ts`: `imageLossPreview`,
  `extractLandmarks`.
- The four hard-coded input-role lists (`formstate.ts` `INPUT_ROLES`,
  `NewJobPage.tsx`, `paramdiff.ts` `IDENTITY_EXCLUDED`, `params.ts`
  `withoutInputPaths`) gain the two roles. `IDENTITY_EXCLUDED` matters: every
  job stores its own copy path, so without it duplicate detection breaks.
- `params.test.ts`'s exact optional-names set is updated.

### 11.2 `components/ImageLossPanel.tsx`

Stateless over `(params, optional, targetSha256, onChange, onOptional)`, like
`CannyPanel`. Mounted in `NewJobPage` as its own `<details class="group">`
"Image fidelity" between Guidance and Losses, with `hide={IMAGE_LOSS_PARAMS}`
passed to the generic `losses` fields. Contents:

1. **Gate row**: checkbox bound to `image_loss`, label and hint from the spec.
   Everything below renders only when on.
2. **Strength**: `image_loss_weight` as a slider 0.05–1.0 (step 0.05) with the
   number beside it; `image_loss_schedule` select; when not `constant`, a
   `image_loss_schedule_start` number field whose placeholder shows the default
   for that schedule, and a one-line sparkline of alpha over the run.
3. **Terms**: three number fields (`chamfer`, `pyramid`, `landmark`); the
   landmark section (5) appears when its weight is > 0.
4. **Edge map**, three-way source:
   - *Derived in the run* (default): `low/high/blur` knobs and the preview:
     `input.png` of the source run with the edge map overlaid (same
     `<img>`-over-`<img>` construction as `CannyPanel`, cache-busted by the knob
     values, 350 ms debounce, generation ticket). Leaves `image_loss_target`
     empty.
   - *Prepared here*: the same plus `clahe_clip`, `clahe_grid`, `roi` (typed
     for now) and `preserve_silhouette`. On submit the latest preview's `sha256`
     is pushed as `{role: 'image_loss_target', source_kind: 'upload', sha256}`.
     The panel refuses to submit while a preview is pending or failed.
   - *From a file*: a `ConstraintPicker`-style picker: upload a PNG, or pick
     `image_loss_target.png` / `condition_canny.png` from a job on the same
     target.
   No previous run → the preview area says so, the derived knobs stay editable.
5. **Landmarks**: "Extract from the source run" button → `extractLandmarks`;
   the returned points are drawn as weighted dots over `input.png` and the
   `sha256` pushed as the `image_loss_landmarks` input. Or upload a JSON.
6. **Budget note**: the edge pixel count from the preview, with a warning under
   2 000 or over 200 000.

### 11.3 Run again

`RunAgainDialog` mounts the panel over `base` (Spec 3 §6.5 promised parity with
the new-job panel; `CannyPanel` never got it, and that gap is closed at the same
time by mounting it too). File inputs are inherited from the parent and shown
read-only, as for every other role today. The structural scalars
(`image_loss_weight`, `image_loss_schedule`, …) are already variant-column
candidates, which is how the sweep in §14 is queued.

### 11.4 Job page

- `ArtworkPane`: a tab **Edge target** when `target/run/image_loss_target.png`
  is in `job.artifacts`, the same way the Stipple weight tab finds its file.
- `components/ImageLossDiagnostics.tsx` + `lib/imageloss.ts`: when
  `image_loss_log.csv` is listed, fetch it, parse it (pure, unit-tested),
  and show: mean and share-negative `cosine`, `sds_norm` min/median/max,
  current `alpha`, latest `chamfer`/`pyramid`/`landmark`, and three small inline
  sparklines (alpha, cosine, sds_norm). Refetch when the job's state changes
  (the existing `useJob.refresh`) and on a manual refresh while running.
  Mounted in the right column under `ParamTable`.
- `ActionsPanel`: a download link for the CSV in the Downloads row.
- `ParamTable` and Compare captions pick the new params up from the registry
  with no change.

## 12. Verification

### 12.1 Default-path identity

The most important test. Without `--image-loss` the code path is upstream's:

- `test_image_loss_run.py` monkeypatches `run_module.ImageFidelityLoss` to raise
  and runs a plain stubbed run to completion: the class is never constructed,
  no `image_loss_target.png`, no `image_loss_log.csv`.
- Byte identity across the change is recorded once, by hand, in §15 "As built":
  run `PYTHONPATH=. python test_run_segments.py` at the parent commit and at the
  new one and record the sha256 of `runseg_plain/final_sld.svg` from both. They
  must match. (Golden hashes are not committed: DiffVG's CPU output is not
  guaranteed identical across machines.)

### 12.2 Unit tests — `test_image_loss_geom.py` (CPU, seconds)

- `alpha_at`: epoch 0, midpoint, last epoch for all three schedules, default
  and explicit starts.
- `edge_points`: a vertical line at column 100 yields points with `x == 100`;
  the cap keeps a fixed stride and is deterministic across two calls.
- Chamfer: a straight-line target and a curve offset by `d` returns ≈ `d`;
  gradient on the curve points towards the line.
- Landmark: adding a curve point near a landmark lowers the loss; adding one far
  away leaves it unchanged; weights scale as expected.
- Pyramid: identical inputs give 0; a translated blob gives a positive value
  that decreases as the translation shrinks.
- `blend_gradients`: `alpha = 0` returns `g_sds` exactly (`torch.equal`);
  `alpha = 1` returns a tensor parallel to `g_img` with norm `‖g_sds‖`; the
  zero-norm guard returns `g_sds` and sets `skipped`; `cosine` matches a direct
  computation.
- Through the real painter: build `SLDBSplinePainter` as `test_attraction_geom.py`
  does, call `parameters()`, evaluate chamfer on `curve_samples(renderer, 200)`,
  `autograd.grad` reaches `control_points` with finite values and leaves
  `weights.grad` untouched.
- Landmark file validation: wrong `image_size` or `space` is refused with a
  message that names `sld_landmarks.py`.

### 12.3 Run-level — `test_image_loss_run.py` (stubbed diffusion, like `test_run_segments.py`)

- `--image-loss` run: `image_loss_target.png` and `image_loss_log.csv` exist;
  the CSV has `num_iter + 1` rows with the schedule's alphas.
- Segmented (`--stop-at`, then `--resume`) equals uninterrupted: `final_sld.svg`
  **and** `image_loss_log.csv` byte-identical.
- Resume with a changed `--image-loss-weight` is refused and the message names
  `image_loss_weight`.
- A supplied target of the wrong size is refused before any model would load.

### 12.4 Service — `test_service_image_loss.py` (mirror of `test_service_canny.py`)

Round trip of every new param; validation cases; the two roles copied into
`inputs/` (PNG upload, JSON upload, PNG from another job's run dir) and their
paths in params; Run again inherits both; the fake writes both artifacts and
`artifacts()` lists them; two segments leave one CSV with no duplicate epochs;
both preview endpoints (404 without a source run, 200 with, `derived_equivalent`
true/false, `sha256` resolvable as an upload). Skips cleanly when cv2 or
MediaPipe is missing, as the canny test does.

### 12.5 Web

`lib/imageloss.test.ts` for the CSV parser and summaries; `params.test.ts`
updated for the optional set and the new validation cases;
`test_service_web.py`'s mirror check passes.

### 12.6 Visual smoke test (GPU)

`--image-loss --image-loss-weight 1.0` with chamfer only must produce an obvious
trace of the edge map with no interpretive quality. If it does not, the loss or
its coordinate handling is wrong and no intermediate alpha means anything. Run
this before any aesthetic judgement.

## 13. Phasing

**How to start, for whoever picks this up.** Environment and rebuild recipe:
`CLAUDE.md`. Which interpreter runs which suite: `docs/readme.md` "Tests".
In short, from the repo root:

```bash
# core and geometry tests: the conda env (torch, pydiffvg, cv2)
PYTHONPATH=. python test_image_loss_geom.py
PYTHONPATH=. python test_image_loss_run.py
PYTHONPATH=. python test_run_segments.py          # must still pass unchanged
# service tests: the API venv, pointed at the conda interpreter for subprocesses
PYTHONPATH=. SLDGEN_CANNY_PYTHON=$CONDA_PREFIX/bin/python .venv-service/bin/python test_service_image_loss.py
PYTHONPATH=. .venv-service/bin/python test_service_units.py
PYTHONPATH=. .venv-service/bin/python test_service_web.py   # params.ts <-> params.py mirror
# web
cd sldgen_web && npm test && npm run typecheck
```

For the identity check in §12.1, check the parent commit out into a worktree
(`git worktree add ../sldgen-parent HEAD~1`), run `test_run_segments.py` there
and in the working tree, and compare `sha256sum` of the two
`output/firefighter/runseg_plain/final_sld.svg` files. Only §12.6 and §14 need
the GPU host; every other step runs on a CPU. Fill §15 as you go.

Each phase ends with its tests green and, from phase 1 on, a working stubbed
run; GPU checks happen at the end of phases 1, 4 and 5.

1. **Core + chamfer**: `image_loss.py` (schedule, blend, edge target, chamfer,
   CSV), `config.py` flags and validation, `run.py` wiring,
   `STRUCTURAL_FIELDS`, `sld_edge_target.py`, `docs.md` flag table,
   `test_image_loss_geom.py`, `test_image_loss_run.py`, identity check (§12.1).
   Chamfer is included here rather than after a placeholder because, given
   `sampled_curve2d`, it is thirty lines.
2. **Service**: `params.py`, `jobs.py` roles, `config.py` scripts, fake,
   `sldgen_api/image_loss.py` + routes, `test_service_image_loss.py`.
3. **Web**: schema mirror and plumbing, `ImageLossPanel` in the new-job form
   and Run again (with `CannyPanel` mounted there too), Edge target tab,
   diagnostics panel, tests.
4. **Pyramid** (core + tests; the UI already has the field).
5. **Landmark**: term, `sld_landmarks.py`, extraction endpoint, overlay in the
   panel, tests.
6. **Validation sweep** (§14), queued from the UI.

### Out of scope

- Changing the SDS implementation, CFG scale, timestep range or LoRA weight.
  Separate levers; conflating them makes both impossible to evaluate.
- Symmetric chamfer or any "cover the whole target" variant (that is
  `--attract-canny`).
- Per-region alpha maps. Worth a follow-up once the global dial is understood.
- Blending `weights`/`widths` (§3.3).
- ROI drawn on the prep canvas (typed box for now; same gap Spec 5 left).
- Any change to existing flags' defaults.

## 14. Validation sweep

Once phase 3 lands, queue this from Run again as one batch on the curly-haired
portrait (the hardest case), with `image_loss_weight` / `image_loss_schedule` /
`image_loss_schedule_start` as the variant columns. Everything else identical:
`--condition canny --conditioning-scale 0.9 --n-control-points 400 --num-iter
8000 --render-size 512`, same seed, same caption, edge map derived in-run with
the same knobs (or one prepared map attached to the parent and inherited).

| Run | `--image-loss` | weight | schedule | start | mean alpha |
|---|---|---|---|---|---|
| A | off | — | — | — | 0 (baseline, equals current output) |
| B | on | 0.15 | constant | — | 0.15 |
| C | on | 0.30 | constant | — | 0.30 |
| D | on | 0.50 | constant | — | 0.50 |
| E | on | 0.10 | decay | 0.50 | 0.30 |
| F | on | 0.30 | ramp | 0.05 | 0.175 |

E compares with C at the same mean strength and answers "does scheduling beat a
constant"; F answers "does a late pull keep more of the interpretation". Each
job's stored params and `/api/jobs/{id}/command` are the manifest; the six
`final_sld.svg` and `image_loss_log.csv` files are the outputs. The agent runs
the sweep and saves the results; judging them is the author's call. The
expectation is a usable band around 0.2–0.35 that becomes the standing default
for the series.

## 15. As built

Implemented 2026-09-25 in five commits, one per phase (`db922cd`, `483c52a`,
`108f09e`, `2b608c3`, and the phase 5 commit). Phase 6, the sweep in §14, has
not been run.

### Identity (§12.1)

`output/firefighter/runseg_plain/final_sld.svg` from `test_run_segments.py`:
`0f8f7b8ba98b0336427b281e5223298c44d8a7a799e2b37b6f6ca4d14a904db7` at the parent
commit `b6acb7e` and again after phase 1, the same hash. Every suite that
existed before still passes.

### GPU checks (§12.6), firefighter, 300 iterations, 512 px

| Run | Result |
|---|---|
| chamfer, alpha 1.0 | the curve lands on the edges: chamfer 2.08 → 0.65 px. 5.4 it/s, the same as without the term. Mean cosine 0.0003. **SDS norm 0.27–47.6 across the 300 steps**, the spread that motivated the gradient blend. |
| pyramid, alpha 1.0 | pyramid 1.18 → 0.69, ink moves onto the figure's dark masses. 4.6 it/s (the second DiffVG pass). Mean cosine 0.03. |
| chamfer + landmark, alpha 0.5 | 21 portrait landmarks from `sld_landmarks.py` on the run's own canvas. Landmark distance 2.67 → 0.64 px: the line passes through the eye/brow row, nose, mouth and chin anchors. Chamfer 2.08 → 1.19 px at half weight. Mean cosine −0.0005. |

A 300-iteration run is not the full "obvious trace" of §12.6: the one-sided
chamfer lets the line sit on *any* edge, so at 300 steps the result is
edge-hugging line segments rather than a complete trace. Registration (no
offset, flip or scale error) is unambiguous.

### Departures from the design

1. **The pyramid term renders its own raster** (`image_loss.render_again`),
   instead of reusing the SDS raster as §6.2 has it. DiffVG's backward
   accumulates into gradient buffers owned by the forward call's scene, so a
   second backward through `raster_sld` returned `g_sds + g_pyramid`. The
   first GPU pyramid run showed it as a cosine of 0.99999 with equal norms. A
   fresh forward over the same shapes at the same fixed seed is the same image
   with its own buffers. `test_pyramid_is_independent_of_the_sds_backward`
   fails on the old behaviour. Consequently `ImageFidelityLoss.__call__` takes
   `(renderer)`, not `(renderer, raster)`.
2. **The CSV is not byte-identical across a resume, and cannot be.** DiffVG's
   backward is not bit-deterministic: two *uninterrupted* runs of the same
   config differ in the last digits of `sds_norm` and `cosine`. `final_sld.svg`
   stays identical only because the SVG writer rounds. Floats are written at 7
   significant digits, and the resume test compares the log row for row:
   epochs, alphas and `skipped` exact, measured values to 1e-4 relative. The
   SVG is still compared byte for byte.
3. **Nearest neighbours without a live `cdist`.** Chamfer and landmark find
   the nearest point under `no_grad` (a chunked `cdist`), then recompute that
   one distance differentiably. The value and the (sub)gradient are the same as
   `cdist(...).min()`, without keeping an `(S, K)` matrix alive for the
   backward.
4. **`skipped` is 0/1/2**: blended / image gradient vanished (g_sds kept) /
   SDS gradient vanished (g_img used). An undefined cosine is an empty cell.
5. **A supplied binary map with dark edges on white is inverted**, so a
   hand-drawn black-on-white hairline works. The edges are taken to be the
   minority value.
6. **`sld_landmarks.py` falls back to a full-range detector.** Face Mesh's
   built-in detector is short-range and found nothing on a full-figure canvas,
   where the face is about 30 px wide. MediaPipe's full-range face detector
   finds it; the mesh runs on an upscaled crop and is mapped back. Both paths
   produce canvas pixels.
7. **MediaPipe is pinned to 0.10.21** in the `sldgen` env. The latest (1.0.x)
   needs numpy 2, which this env must not have, and drops the bundled Face
   Mesh model. 0.10.21 kept numpy at 1.26.4 but replaced protobuf 7.35 with
   4.25 (nothing in the env declares a protobuf requirement; `pip check` is
   clean, and imports and GPU runs are unaffected) and added
   `opencv-contrib-python` 4.11, the same version as the installed OpenCV.
8. **Service**: the rule "landmark weight needs a landmarks file" is enforced
   in `create_job` after inputs resolve, not in `validate_params` (which runs
   before inputs are attached), and it knows which roles Run again is about to
   inherit.
9. **Web**: `NewJobPage`'s own `INPUT_ROLES` still lists only the four SVG
   roles, because it drives `ConstraintPicker` and the SVG overlays. The two
   new roles are owned by `ImageLossPanel` and handled through the shared
   `INPUT_PARAMS` everywhere else. Image-loss inputs are only sent while the
   gate is on. The Edge target tab is hidden, not disabled, on jobs without
   the file. Submission is blocked while a prepared map is pending, or when
   the landmark weight is set with no landmarks attached.
10. **`test_service_canny.py` always skipped** before this work: it requires
    cv2 in its *own* interpreter, and neither the service venv nor the conda
    env has both fastapi and cv2. `test_service_image_loss.py` probes the
    interpreter it spawns instead, so it runs.

### Left undone

- The validation sweep (§14). It needs a portrait target and about six
  8000-iteration GPU runs, and queuing it is the author's call.
- No browser check of the panels: the machine is headless. The logic is
  covered by vitest (`imageloss.test.ts`, `params.test.ts`,
  `formstate.test.ts`), typecheck and build. The first real use of the panel
  is its UI test.
- The API must be restarted to serve the new endpoints (`./stop.sh &&
  ./start.sh`). `dist/` is already rebuilt.
