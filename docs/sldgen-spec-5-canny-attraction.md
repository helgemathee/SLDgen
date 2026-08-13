# Spec 5 — Canny-derived attraction (`--attract-canny`)

**Status:** ✅ **implemented**, tested on CPU and simulated end to end through the
service. Not yet run on the GPU host.
**Scope:** a new opt-in constraint in SLDgen core, its service parameters, a
preview endpoint, and a panel in the new-job form.
**Depends on:** Specs 1–3. Independent of [Spec 4](sldgen-spec-4-dual-conditioning.md) —
this is the cheap alternative that Spec 4 §5 recommends trying first.

---

## 1. Motivation

Depth conditioning tells SLDgen where the *volume* is. It has almost nothing to
say about an eyelid, a nostril or a lip corner, which is why portraits come out
with convincing mass and vague features. The obvious fix is a second ControlNet
on Canny edges (Spec 4); the cheap one is to keep depth driving SDS and pull the
curve onto the edges **geometrically**, through the attraction constraint that
already exists.

No second model, no extra VRAM, no measurable time cost. The whole feature is a
few hundred lines and an edge detector.

## 2. The constraint that shapes the design

Attract points are consumed as **raw canvas pixel coordinates**:
`avoidance.load_avoid_points` samples an SVG's paths and hands the numbers
straight to the loss, with no registration step. So the edge map must come from
an image that has already been through `targets.py`'s RMBG mask, square pad,
resize and `--object-size-ratio` rescale.

That image exists in exactly one place: `args.input_image`, after `get_target`
and before the painter is constructed. Every other option means reproducing that
pipeline and hoping it matches —

| Where the edges could be computed | Why not |
|---|---|
| A helper script on the original photo | Wrong frame. Misregisters silently, which is the worst failure mode available. |
| A web upload at submission time | Same problem: the browser has the photo, not the canvas. |
| A service pre-pass | Would have to run RMBG — a model load, on the API host, to reproduce something the run does anyway. |
| **Inside the run, after `get_target`** | Nothing to reproduce. The image *is* canvas space. |

So the generation lives in `run.run`, and the flag is all the user supplies.
`SLDgen/canny_attract.py` holds the implementation and imports only cv2/numpy,
which is what makes it testable without a GPU.

## 3. Interface

```
--attract-canny                      off by default; everything below is inert without it
--attract-canny-low        100.0     SLDgen's own ControlNet thresholds, so the default is
--attract-canny-high       200.0       "the edges the Canny ControlNet would have seen"
--attract-canny-blur       3         odd kernel, 0 disables
--attract-canny-simplify   1.0       Douglas-Peucker tolerance, canvas px
--attract-canny-min-length 12.0      drop contours shorter than this (speckle)
--attract-canny-max-points 400       budget for the generated points
```

Pull strength is **not** duplicated here: the generated SVG is one more attract
source, so `--attraction-weight` and `--attraction-distance` govern it exactly as
they govern a partition SVG. It composes with `--attract` rather than replacing
it.

`--roi` (restrict to a box — "the face, not the shoulders") is deliberately
**not** exposed on the flag. It needs a rectangle drawn on a canvas to be worth
anything, and the standalone script has it in the meantime.

## 4. The point budget is the whole safety story

`attraction_loss` is two-sided, and its coverage term is a **sum over every
attract point** of `max(0, dist - deadzone)²`. A raw Canny map is tens of
thousands of pixels; asking ~385 control points to visit all of them produces a
constraint that buries the SDS gradient. The loss's own docstring scopes it to
"hundreds to low thousands of points per side".

`thin_to_budget` enforces that in two stages, because they solve different
problems:

1. **Drop speckle.** Every surviving contour emits at least one dash and every
   path yields at least 2 samples, so *N contours cost 2N points no matter how
   hard they are thinned*. A speckly map has to lose contours first; the list is
   sorted longest-first, so what goes is the noise.
2. **Thin what remains**, uniformly along arc length, by dashing. Every contour
   keeps its full extent at a lower density — an eyebrow still gets targets, it
   just gets fewer. Dropping whole contours instead would silently delete
   features.

Thinning removes *length*, not vertices: SLDgen resamples every path at a fixed
2 px regardless of vertex spacing, so decimating points would change the file and
not the result. The dash period cannot be solved for directly (a contour shorter
than the period still emits one dash), so it is bisected.

The UI warns when the budget exceeds `--n-control-points`, which is the point at
which the curve is being asked for coverage it cannot deliver.

## 5. Fingerprinting and resume

The generated path is **not** appended to `args.attract`. That field is part of
`STRUCTURAL_FIELDS`, and the fingerprint is computed from the parsed arguments
before `get_target` and again when a checkpoint is written — mutating it in
between would make a resumed segment disagree with the segment that saved the
checkpoint.

Instead `--attract-canny` and its six knobs are fingerprinted, and the painter
picks the generated SVG up through a separate `args.attract_canny_svg`. Canny is
deterministic, so settings that match produce identical points; every segment
regenerates the same file, and `test_service_canny.py` asserts exactly that on
two segments of one job.

## 6. The preview, and why it is a separate path

Generating inside the run is right for correctness and wrong for iteration: you
would not see what a threshold did until a job had been queued, claimed and
started. `POST /api/canny/preview` closes the loop by running the same code over
a **previous run of the same image** — whose `input.png` is that canvas, sitting
on disk already. Same target and same render size means the same canvas, so the
preview is what the next run will generate.

It shells out to `sld_canny_svg.py` under the conda interpreter, exactly as
partitioning does, keeping cv2 and numpy out of the API venv. No previous run
means no preview, and the panel says so rather than hiding the controls.

## 7. What shipped

| File | Change |
|---|---|
| `SLDgen/canny_attract.py` | new — edge map, contour tracing, budget, SVG. cv2/numpy only |
| `sld_canny_svg.py` | new — thin CLI over the module; adds `--roi` and `--already-edges` |
| `SLDgen/config.py` | the seven flags and their validation |
| `SLDgen/run.py` | generate between `get_target` and the painter; log a summary |
| `SLDgen/painter/painter.py` | the generated SVG joins `--attract`'s sources |
| `SLDgen/checkpoint.py` | seven names added to `STRUCTURAL_FIELDS` |
| `sldgen_service/params.py` | seven `ParamSpec`s + validation mirror |
| `sldgen_service/config.py` | `canny_script` |
| `sldgen_api/canny.py` | new — preview subprocess, source-run lookup |
| `sldgen_api/app.py` | `POST /api/canny/preview`, `GET /api/canny/preview/{id}.svg` |
| `sldgen_web/.../CannyPanel.tsx` | new — the knobs next to the trace they produce |
| `sldgen_web/.../params.ts` | specs + client-side validation mirror |
| `sldgen_web/.../NewJobPage.tsx` | panel wired in; knobs hidden from the generic field list |
| `test_support/fake_sldgen.py` | writes `input.png`/`mask.png`; runs the **real** generator |

The fake running the real generator is deliberate. A stub would prove the
plumbing moves a file around; running the actual module proves the file the UI
reads is the file SLDgen writes.

## 8. Tests

| File | Needs | What it proves |
|---|---|---|
| `test_canny_attract_geom.py` | cv2/numpy | points land in canvas space, budgets hold, the mask is honoured, output is deterministic, and — via svgpathtools — the reported count is the count the **real sampler** produces |
| `test_service_canny.py` | fastapi + cv2 | CLI round trip, validation, the run writing `attract_canny.svg` as an artifact, two segments generating identical points, and every preview-endpoint path |

Neither needs a GPU. 32 checks in the service test, 19 in the geometry test.

## 9. Not done

- **No GPU run yet.** Everything above is CPU-verified and simulated through the
  fake; the real `run.py` path has not executed on the 5090.
- **No ROI in the UI.** It wants a rectangle drawn on the prep canvas.
- **Tuning is unmeasured.** `--attraction-distance` defaults to 25 px, which is
  5 % of a 512 canvas — a dead zone that wide barely pulls. Around 8–12 is the
  place to start for landing the curve on features, with the weight at its 0.004
  default and raised to 0.01–0.02 if nothing moves. Since the coverage term sums
  over N, weight and point count trade off: doubling the budget wants roughly
  half the weight.
