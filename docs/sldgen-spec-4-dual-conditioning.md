# Spec 4 — Dual conditioning: depth *and* canny in one run

**Status:** 📝 **design only** — nothing implemented. This document is the plan.
**Scope:** `SLDgen/guidance/*`, the `--condition` family of flags, and the
service/UI plumbing that assumes exactly one condition image per run.
**Depends on:** Specs 1–3 (checkpointing, the worker, the web UI) — all landed.
**Motivating case:** portraits. Depth alone puts the line where the *volume* is;
the features (eye line, nostril, lip corner, hair boundary) arrive late or not
at all.

---

## 1. What exists today

`--condition` picks exactly one of `depth` | `canny`
(`SLDgen/config.py:220-225`). Everything downstream is keyed off that single
string:

| Where | What it does |
|---|---|
| `SLDgen/guidance/sd3_controlnet.py:9` | loads one `tensorart/SD3.5M-Controlnet-{Depth,Canny}` |
| `sd3_controlnet.py:25` | builds one conditioning image from `args.input_image` |
| `sd3_sds_guidance_control.py:94` | encodes it through the VAE **once**, caches the latent |
| `sd3_sds_guidance_control.py:164` | saves it as `condition_<name>.png`, canvas-space |
| `sd3_sds_guidance_control.py:285` | one ControlNet call per SDS step |
| `SLDgen/checkpoint.py:53` | `condition` is a `STRUCTURAL_FIELDS` member |
| `sldgen_service/params.py:68` | one `ParamSpec`, validated to the two names |
| `sldgen_api/partitions.py:68` | `labelmap` labels default to `condition_<name>.png` |
| `sldgen_web/.../ArtworkPane.tsx:21` | finds *the* condition artefact by regex |

The important structural fact: **`forward()` bypasses the diffusers pipeline**
and calls `self.pipe.controlnet(...)` itself, then passes the per-block residuals
to the transformer as `block_controlnet_hidden_states`
(`sd3_sds_guidance_control.py:285-301`). Adding a second ControlNet therefore
does not require `StableDiffusion3ControlNetPipeline` to cooperate, or
`SD3MultiControlNetModel` to be adopted — it requires running a second
ControlNet and **summing the residuals**, which is what the multi-ControlNet
wrapper does internally anyway. The change at the tensor level is about ten
lines.

## 2. Design principles

Unchanged from Spec 1 §2, and they decide most of the open questions here:

- **Strictly opt-in.** With `--condition-2` unset, behaviour is bit-identical to
  today, including stdout and the set of files written.
- **Validation in `parse_arguments`**, not mid-run.
- **Schedules are pure functions of the absolute epoch.** `SLDgen/utils.py:6`
  states the rule and says why: a segmented run must equal an uninterrupted one,
  and a short preview must be a genuine prefix of the long run. Any
  epoch-dependent condition switching obeys it or it breaks resume.
- **No new dependencies.** Both ControlNets already exist on the Hub; the second
  is one more `from_pretrained`.

## 3. Three combination modes

`--condition-mode` selects how the two conditions combine. They are *not*
equivalent, and only one of them costs anything.

| Mode | Cost per iteration | Behaviour |
|---|---|---|
| `sum` | **+1 full ControlNet forward** | both residuals every step, summed. Standard multi-ControlNet. |
| `alternate` | free | condition A on even epochs, B on odd. The SDS gradient is the average of the two in expectation. |
| `sequential` | free | A for the first `--condition-switch` fraction of `--num-iter`, B afterwards. |

`alternate` and `sequential` only ever run **one** ControlNet per step, so they
cost nothing but the VRAM of the second resident model. They are the cheap way
to get "both", and `alternate` is the one to try first: it needs no switch point
to be tuned and produces no seam.

Both must select on the **absolute epoch** (`epoch % 2`, `epoch / num_iter`),
never on a counter of iterations executed in this process, and never on RNG.
`get_sparse_loss_weight` (`SLDgen/utils.py:6`) is the template.

### The cost of `sum`, honestly

One extra ControlNet forward per SDS step, against a transformer forward that
runs on the CFG-doubled batch. Somewhere in the region of 15–30% slower per
iteration — **unmeasured**, and it must be measured on the 5090 before the
number is quoted anywhere. At 4000 iterations that is not nothing.

## 4. The part that is not plumbing

Two ControlNets is easy. Two ControlNets that *help* is not, and the risk is
specific to what SLD draws:

**They compete for the same ink.** An SLD run has a fixed line budget —
`--n-control-points`, tightened over time by the progressive sparse loss. Depth
wants that length spent on tonal mass (shading a cheek, a forehead); canny wants
it spent on contours. Summed at the current `--conditioning-scale 0.5` each, the
effective conditioning strength roughly doubles, which is the classic
over-constrained ControlNet failure: both objectives partially served, neither
convincing. **Expect to retune to ≈0.3 / 0.25 before judging the idea at all.**
A first experiment run at 0.5/0.5 will look bad for reasons that have nothing to
do with whether dual conditioning works.

**Canny on a masked portrait is fragile.** The thresholds are hardcoded at
100/200 (`sd3_controlnet.py:38-39`) and Canny runs *after* the rescale and mask.
On soft studio lighting the edge map can come back nearly empty, and
`create_masked_condition` then computes `im_np / im_np.max()`
(`sd3_sds_guidance_control.py:193`) — a divide by zero on an all-black map,
producing a NaN conditioning image. This is a latent bug **today**; it becomes
much likelier once canny is routinely in play. Guarding it (and probably
exposing the two thresholds as flags) belongs in this work.

**`sequential` has a seam.** Adam carries momentum across the switch from a
different objective. Usually harmless, sometimes a visible jolt in the frames
around the switch epoch.

## 5. Cheaper alternative worth trying first

This fork already has `--attract` (attraction points) and `--stipple-weight` (an
image). Feeding **the canny edge map as attraction points** while depth keeps
driving SDS pulls the line onto the facial features *geometrically*, at zero GPU
cost, with no conflicting gradients and no second model resident. It reuses
machinery that is built, tested and shipped.

For the portrait case specifically this is the better first bet, and it is an
afternoon of experimenting rather than a day of implementation. Spec 4 should
not be built until it has been tried and found insufficient.

## 6. Interface

```
--condition            depth | canny                    (unchanged, default depth)
--conditioning-scale   float                            (unchanged, default 0.5)
--condition-2          depth | canny | None             (new, default None = off)
--conditioning-scale-2 float                            (new, default 0.5)
--condition-mode       sum | alternate | sequential     (new, default sum)
--condition-switch     float in (0, 1)                  (new, default 0.5)
```

Validation at parse time: `--condition-2` may not equal `--condition`;
`--condition-switch` is only meaningful for `sequential`; the `-2` flags are
inert (and warned about) when `--condition-2` is unset.

`--condition-2` defaults to `None` rather than the string `"none"` so that
`params_to_argv` emits nothing for it — `sldgen_service/params.py` canonicalises
parameters and emits every **non-null** value, so a null default is how a new
flag stays invisible in the recorded command of a single-condition job.
`--condition-mode` and `--condition-switch` have non-null defaults and will be
emitted always, which is the existing convention (`--conditioning-scale` behaves
the same way).

## 7. File-by-file plan

1. **`SLDgen/config.py`** — the four flags above, plus their validation.
2. **`SLDgen/utils.py`** — `active_conditions(args, epoch)` returning the list of
   `(name, scale)` active at that epoch. Pure, epoch-absolute, importable
   without diffusers, so it is testable on a laptop with no GPU. Same reason
   `get_sparse_loss_weight` lives there.
3. **`SLDgen/guidance/sd3_sds_guidance_control.py`** —
   `configure()` builds a list of `(name, controlnet, encoded_control_image,
   scale)` instead of the three scalar attributes it caches now
   (`self.control_image`, `self.conditioning_scale`). `forward(x, epoch)` asks
   `active_conditions`, runs each ControlNet, and sums the residual lists
   element-wise. Assert at load time that both ControlNets emit the same number
   of blocks, with a clear error rather than a shape mismatch deep in the
   transformer — tensorart's SD3.5M set is expected to match, but verify rather
   than assume. The existing `conditioning_scale == 0.0` skip path generalises
   to "no active conditions this epoch → `control_block_samples = None`".
4. **`SLDgen/run.py:198`** — `sds_loss(raster_sld, epoch)`.
5. **`SLDgen/checkpoint.py:53`** — add the four names to `STRUCTURAL_FIELDS`.
   Consequence, and it is the correct one: an existing single-condition job
   cannot be resumed into a dual-condition one. A/B comparison means new jobs.
6. **`sldgen_service/params.py`** — four `ParamSpec`s, all `STRUCTURAL`, plus
   the validation mirror at `params.py:188`. All four are plain scalars, so the
   `argv_to_params(params_to_argv(p)) == p` round-trip needs no new machinery.
7. **`sldgen_api/partitions.py:68`** — `default_labels` must prefer the **depth**
   image when a run wrote two. A binary canny edge map is useless as a
   quantile-binned label map, so "whichever matched first" is not acceptable.
   Two call sites in `sldgen_api/app.py` (941, 983).

## 8. Web UI

A dual-condition run writes **both** `condition_depth.png` and
`condition_canny.png`. They are already canvas-space aligned and already named
by condition, so nothing collides — but `availableArtwork`
(`ArtworkPane.tsx:21`) does `.find()` on `/^target\/run\/condition_\w+\.png$/`
and would silently display whichever artefact happened to sort first.

**Decision: sub-chips inside the existing Condition tab.** `Available.condition`
becomes a list of `{ name, url }`; when it holds more than one entry the
Condition tab renders a small segmented control naming each. The `ArtworkTab`
union, the tab labels and `defaultTab` are untouched, and — the reason this
beats a second tab — the shared pan/zoom state (`ArtworkPane.tsx:49-55`) keeps
working, so flipping depth↔canny at a fixed zoom stays the A/B comparison the
tabs exist for.

Also:

- `sldgen_web/src/lib/params.ts:93` — four new specs in the `guidance` section,
  with `choices`, and the validation mirror at `params.ts:216`.
- `PartitionPanel.tsx:155` — the copy says labels default to "this job's own
  condition image"; it must say *which one* when there are two.

## 9. Tests

| File | Adds |
|---|---|
| new `test_condition_schedule.py` | `active_conditions` across all three modes and the switch boundary; asserts the epoch-absolute invariant (the value at epoch *n* does not depend on where a segment started) |
| `test_service_units.py` | the four params round-trip; validation rejects `condition_2 == condition` |
| `test_service_web.py` | a dual-condition job exposes two condition artefacts |
| `sldgen_web/src/lib/params.test.ts`, `formstate.test.ts` | the new specs and their defaults |

None needs a GPU. The tensor-level change in `sd3_sds_guidance_control.py` is
**not** covered by any of these and needs a manual smoke run on the CUDA box —
which is also where the `sum` slowdown gets measured.

## 10. Effort and sequencing

Roughly half a day for §7–§9. The split matters for where it can be done: steps
1, 2, 5, 6, 7 and all of §8–§9 are editable and testable on the Mac checkout;
step 3 needs the Linux/CUDA box to be exercised at all.

Suggested order:

1. §5 first — canny as `--attract` points, no code changes. If that solves the
   portrait problem, stop here.
2. The NaN guard in `create_masked_condition` and the Canny thresholds as flags.
   Worth doing regardless of whether dual conditioning is built.
3. `alternate` and `sequential` — free at runtime, and they answer "do the two
   conditions help each other at all?" before anything is paid for.
4. `sum` last, once there is a reason to believe the answer is yes.

## 11. Open questions

- Do `tensorart/SD3.5M-Controlnet-Depth` and `-Canny` produce identical block
  counts? Assumed, unverified. If not, `sum` is dead and only `alternate` /
  `sequential` survive.
- Should the two conditions be allowed to be the *same* type at different scales
  or thresholds (e.g. tight canny + loose canny)? Rejected for now — the
  condition image filename is keyed by type and two runs would overwrite each
  other's `condition_canny.png`.
- Is a third condition ever wanted? The list-of-tuples shape in §7.3 admits it
  for free; the flag naming (`--condition-2`) does not. Left as-is deliberately:
  two is the case that has a motivation.
