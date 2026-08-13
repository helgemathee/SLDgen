# Spec 3 — Web UI

**Status:** ✅ **complete** — implemented, tested, and run against a real GPU job
(2026-08-09).
**Depends on:** Spec 2 (API contract, job lifecycle), Spec 1 (core semantics).
**Scope:** browser application only. Single user, no authentication, desktop-first.
**Served by:** `sldgen-api` as a static build; `sldgen_web/` in the repo.

> The body below is the design as written. §18 records what was actually built,
> where it departed from this text and why, and what is deliberately not done.
> §17's open questions are answered there. Along the way this work produced the
> **first real GPU run the service has ever performed** — Spec 2 §17 listed that
> as not done — including a segmented resume from a checkpoint.

---

## 1. What this is

A control surface for a queue of long-running, expensive, visually-judged jobs.
Three things dominate every design decision:

1. **A job takes 20 minutes and the interesting result is often not the final
   one.** Intermediate states must be as reachable as final ones.
2. **The workflow is comparative.** Four candidates get short runs; you look at
   them together; one gets promoted. The UI's job is to make that comparison and
   that promotion fast.
3. **Settings are hard-won.** An origin point placed carefully on one image
   should still be there for the next run. Losing parameter state between
   submissions is the single most annoying thing this UI could do.
4. **Runs come in families.** A result is rarely the end — it is the thing you
   vary. Re-running an existing job with an edited caption or a different seed,
   without re-preparing the image, is a primary action rather than an
   afterthought (§6.5).

## 2. Stack

React + TypeScript + Vite. No component library — the two hardest surfaces (the
image prep canvas and the iteration scrubber) are custom, and a kit would
contribute weight without helping either. Plain CSS with a token layer.

Built to `sldgen_web/dist/`, served by `sldgen-api` at `/`. Dev runs Vite with a
proxy to the API so SSE works unchanged.

No client-side routing library needed beyond a small hash router: the app has
four routes — `/jobs`, `/jobs/:id`, `/compare`, `/new`.

## 3. Design direction

### Grounding

The subject is an instrument that drives a pen across paper. Its output is
literally one black stroke on white. The interface should feel like the
instrument's control panel, not like a SaaS dashboard: dense, technical, quiet,
with the artwork as the only thing allowed to be beautiful.

### The one risk

**Colour is reserved entirely for machine state.** No decorative colour
anywhere — not in headers, not in buttons, not in the logo. The only saturated
pixels in the entire application indicate what the queue is doing. This is
defensible rather than merely austere: in a UI whose content is monochrome line
art, any decorative colour competes with the work, and a status colour that
appears nowhere else is legible from across the room. Primary actions get weight
and position, not hue.

### Tokens

```
--paper        #FBFAF7   surface, the "page"
--ink          #14140F   text, strokes, borders at full weight
--graphite     #6B6A63   secondary text, axis labels
--rule         #DDDBD3   hairlines, dividers, canvas grid
--slate        #2E3138   panel chrome, the rail

state colours — used ONLY for job state, nowhere else
--st-queued    #8A8F98   waiting its turn
--st-running   #C2410C   actively on the GPU
--st-waiting   #4F46E5   budget reached, decision needed  ← the important one
--st-complete  #15803D   reached the horizon
--st-failed    #B91C1C
```

`--st-waiting` gets the most assertive hue on purpose. "Reached its budget and
is waiting for you" is the state the entire workflow pivots on, and it must
never read as a passive intermediate.

### Type

- **Archivo** for interface text; **Archivo Expanded** at small sizes for
  section labels and eyebrows, which gives structure without rules or boxes.
- **Martian Mono** for all machine data: epochs, iteration counts, coordinates,
  it/s, byte sizes, seeds, hashes, log output. Every number that came from the
  machine is monospaced; every word written for a human is not. The distinction
  is informational, not decorative — you can tell at a glance what is a
  measurement.

Type scale: 11 / 13 / 15 / 19 / 27, tight leading, generous letter-spacing on
the expanded labels only.

### Signature element

**The contact sheet.** The iteration history is presented as a horizontal
filmstrip of the actual frames — the drawing emerging from noise into structure,
scrubable. It is the most characteristic thing this system produces and it is
also exactly the tool needed for the "intermediate is often better" problem. It
appears on the job detail page and, in miniature, in the compare view.

Everything else stays quiet so the filmstrip and the artwork carry the page.

### Motion

Almost none. Two exceptions: the running-state indicator advances continuously
(it encodes real progress), and the filmstrip scrubs at frame rate. No page
transitions, no reveal animations. `prefers-reduced-motion` freezes the running
indicator to a static pulse-free marker.

## 4. Shell layout

```
┌────────────────────────────────────────────────────────────────────────┐
│ ▍SLDgen            [ New job ]              [ Jobs ] [ Compare ]       │  header, 48px
├──────────────┬─────────────────────────────────────────────────────────┤
│              │                                                         │
│  JOB RAIL    │   MAIN                                                  │
│              │                                                         │
│  ▣ filter    │   (job detail | compare grid | new job flow)            │
│              │                                                         │
│  ◐ car_03    │                                                         │
│    1500/4000 │                                                         │
│  ● bridge_1  │                                                         │
│  ◑ portrait  │                                                         │
│  ○ tree_a    │                                                         │
│              │                                                         │
│  280px       │                                                         │
├──────────────┴─────────────────────────────────────────────────────────┤
│ ● car_03 · 1512/4000 · 4.9 it/s · ~8m left │ queue 3 │ GPU 26.1 GB free │ disk 4.2 GB │
└────────────────────────────────────────────────────────────────────────┘  status bar, 32px
```

The rail persists across all routes. The status bar persists across all routes.
Neither ever unmounts, so the running job is always visible no matter where you
are.

## 5. Job rail

### Status iconography

State is encoded by a **progress ring** plus colour, not by a separate badge.
The ring does double duty:

- **Filled arc** = `current_epoch / num_iter` — how far through the horizon.
- **Tick mark on the ring** = `target_epoch` — where this job is currently
  headed.

So a job at 400 of 4000 with a budget of 400 shows a small filled arc with the
tick sitting exactly at its leading edge: *complete against its budget,
incomplete against the horizon*. That is precisely the "partially completed"
distinction, and it needs no words.

| State | Ring | Colour |
|---|---|---|
| queued | empty ring, hairline | `--st-queued` |
| running | arc advancing, leading edge marker | `--st-running` |
| waiting | arc static, tick at arc edge, ring emphasised | `--st-waiting` |
| paused | arc static, ring dashed | `--st-queued` |
| complete | ring fully filled | `--st-complete` |
| failed | ring broken at the failure point | `--st-failed` |
| deleting | ring at 30% opacity | `--graphite` |

### Row contents

Thumbnail (latest preview frame, 44px), title, ring, and one line of monospace
data: `1500/4000` for running or waiting, elapsed for queued, error class for
failed. Nothing else — the rail is for scanning, not reading.

Thumbnails matter more than they look: with four candidates in flight, the rail
becomes the first-pass comparison surface.

### Controls

Filter chips by state, a text filter over title and caption, and a sort toggle
(newest / longest running). Multi-select via checkbox or shift-click feeds the
compare view and bulk delete.

## 6. Job detail

Three columns inside the main area: **artwork** (largest), **history**, and
**actions and data**.

### 6.1 Artwork

Tabbed, defaulting to the most advanced artefact available:

- **Result** — `final_sld.svg` rendered inline as SVG (not `<img>`) so it can be
  zoomed and panned, with a white/paper background toggle and a stroke-weight
  override for legibility at small zoom. Shows total path length and an
  estimated pen travel distance, which is the number that actually predicts how
  long this takes to draw.
- **Preview** — the latest `svg_to_png/iter_*.png` while running.
- **Input** — the prepared `input.png` as SLDgen received it.
- **Mask** — `mask.png`, the RMBG result.
- **Condition** — `condition_depth.png` or `condition_canny.png`.
- **Stipple weight** — `inputs/stipple_weight.png` when the job has one, shown
  raw as painted: white is full ink, black is none. Last in the strip because it
  is the only tab that is not a run artefact and is usually absent. In multiply
  mode the field that actually seeds the stipple is this times the RMBG mask,
  and that product is never written to disk — the Mask tab is the other half.

Toggling between Input, Mask and Result at the same zoom is how you diagnose a
bad run, so all tabs share one pan/zoom state.

What they share is the **framing**, not the raw scale. The tabs do not agree on
a pixel size — the run's artefacts are rendered at `render_size`, while a
painted weight map is the full resolution of whatever was uploaded — so holding
`scale` fixed would show the weight map several times larger than the mask it
modulates, which is exactly the comparison the shared view exists to make.
Carried across a tab switch instead: the zoom *relative to fit*, and the
normalised point of the picture under the middle of the viewport. Fitted stays
fitted; twice-fit on an ear stays twice-fit on that ear.

### 6.2 History — the contact sheet

A horizontal filmstrip of every frame in `svg_to_png/`, plus:

- An integer slider over frame indices, with the epoch shown in monospace.
- Arrow keys step one frame; shift-arrow steps ten; `home`/`end` jump.
- A play control that runs the sequence at 10 fps, matching the mp4.
- Clicking a frame loads it into the artwork pane at full size.
- For any frame: **open the matching SVG** from `svg_logs/`, and download it.

**Two warnings the UI must carry**, both from Spec 1 §7:

1. Intermediate SVGs are written *before* `increase_object_size` runs. If the
   job's `config.json` shows `scale_w`/`scale_h`, intermediates are in a
   different coordinate space than `final_sld.svg`. Label them clearly, and
   **disable** using them as `avoid` / `attract` / `init_points` inputs or as a
   partition source. Only `final_sld.svg` may feed the graph.
2. Frames exist only at `save_interval` granularity. Say so once, rather than
   implying the slider is continuous.

The mp4 is offered as a download but is not the scrubbing mechanism — frames
seek instantly and the video does not.

### 6.3 Actions

Grouped by consequence, with destructive actions separated by a rule:

**While running:** `Pause` (checkpoints, keeps position), `Cancel` (stops and
leaves the job at `waiting` — for when you have already seen enough).

**While waiting:** `Promote to…` with an iteration input pre-filled to the
horizon, plus quick buttons (`+500`, `+1000`, `to 4000`). This is the primary
action on the page and should look like it.

**Always, from any state:** `Run again with changes…` (see §6.5), `Retry` on
failure, `Delete`.

**Downloads:** final SVG, final PNG, the mp4, and `Download everything (.zip)`
with a checkbox for including checkpoints — off by default, since they are large
and useless outside this service.

### 6.4 Data and log

A monospace parameter table showing exactly what was run, with the reproduction
command (from `/api/jobs/{id}/command`) available to copy. Parameters are shown
read-only: they define this job and cannot be edited once it exists. The panel's
one action is `Run again with changes…`, which carries them into a new job.

A segments list — one row per SLDgen invocation with epochs, exit status, wall
time and mean it/s — where each row opens **that segment's console log**.

The log viewer is a full panel, not a modal: monospace, byte-offset incremental
loading, live tail via SSE when the segment is running, auto-scroll that
disengages the moment you scroll up and re-engages at the bottom, a search
filter, a `raw` toggle (Spec 2 §13.2), and download. Default view is cooked, so
tqdm renders as one advancing progress line rather than thousands of fragments.

When a job fails, the detail page opens on the log, scrolled to the end, with
the classified error class stated above it in plain words — "The GPU ran out of
memory", "Hugging Face rejected the model request" — and the retry action beside
it.

### 6.5 Run again with changes

The most common action after looking at a result is *this, but with a different
caption*. It must be available from **every** job state — running, waiting,
paused, complete, or failed — and it must not require retyping anything.

`Run again with changes…` opens the full parameter panel from §8.3,
**pre-filled with this job's exact parameters**, using the same source image
(already uploaded, so no re-upload and no re-preparation), the same prepared
`input.png`, the same mask, and the same origin pin. Edit whatever you like,
including structural parameters, and queue it. The new job starts from epoch 0.

The panel highlights every field you have changed against the parent, and the
submit button names the count: `Queue job (3 changes)`. After submission, both
jobs show a lineage line — "derived from car_03" / "1 variant" — and selecting a
job and its variants opens the compare view directly, which is the natural next
step.

**There is no fork.** Resuming a job with different parameters is not offered,
because every result-shaping parameter is structural (Spec 1 §6) — a job that
changed its caption partway through could not be described by either caption.
Continuing an existing job means only one thing, `Promote`, which runs more
iterations of exactly what is already there. Everything else is a new run.

**Lineage** is recorded with `parent_job_id`, so a job shows what it was derived
from and how many variants came out of it, and selecting a job with its variants
opens the compare view.

See §6.6 for queueing several variants at once, which is the more common case.

### 6.6 Batch variants

The dominant exploration pattern is not one edited copy but **a handful of
variants queued together**: five runs differing only by seed, then later another
five differing by seed *and* caption. The run-again panel therefore has a
variant table rather than a simple multiplier.

```
Variants                                            [ + Add ] [ Regenerate seeds ]
┌──┬────────┬──────────────────────────────────────────────────┬────┐
│ ✓│ seed   │ caption                                          │    │
├──┼────────┼──────────────────────────────────────────────────┼────┤
│ ✓│ 1041   │ a single line drawing of a vintage racing car    │ ⧉ ✕│
│ ✓│ 1042   │ a single line drawing of a vintage racing car    │ ⧉ ✕│
│ ✓│ 1043   │ a single line drawing of a racing car, side view │ ⧉ ✕│
│ ✓│ 1044   │ a single line drawing of a vintage racing car    │ ⧉ ✕│
│ ✓│ 1099   │ a minimal line drawing of a car                  │ ⧉ ✕│
└──┴────────┴──────────────────────────────────────────────────┴────┘
5 variants · 400 iterations each · ~11 min queued      [ Queue 5 jobs ]
```

**Behaviour**

- Setting the count to N fills the table with N rows. Every field is prefilled
  from the parent and every field is editable per row.
- **Seeds are always visible before submission** and individually editable.
  Default generation is sequential from the parent's seed (`parent+1 … parent+N`)
  rather than random: it is reproducible, you can tell at a glance which family a
  run belongs to, and re-running the batch tomorrow does not silently collide.
  `Regenerate seeds` offers a random block instead, for when the sequential range
  has been used up.
- A seed that already exists on another job with otherwise identical parameters
  is flagged inline as a duplicate run — not blocked, since re-running a seed is
  occasionally deliberate, but it should never happen by accident.
- **Captions default to the parent's caption and are per-row editable.** If the
  parent's caption was auto-generated, the row is prefilled with the parent's
  *resolved* caption, not an empty field. Leaving a row's caption empty means
  BLIP-2 will caption it again, which is a real option but should be a choice —
  an empty-by-default table would silently make five variants that differ in
  ways you did not ask for.
- Rows can be disabled (unchecked), duplicated, and removed. Duplicating a row
  and editing its caption is how the "same seed, two captions" comparison gets
  built.
- **Any other parameter can be promoted to a column** via `+ Add column`,
  choosing from the parameter list — conditioning scale and control-point count
  are the usual next candidates. Seed and caption are simply the two that are
  columns by default.
- The footer states variant count, per-run budget, and estimated total queue
  time from the running average it/s, so the cost of the batch is visible before
  committing to it.

**Submission** creates N jobs in one transaction, each with `parent_job_id` set
and a **shared `batch_id`**. Titles are derived
automatically and legibly — `car_03 · s1042` — with the caption difference shown
in the rail where one exists.

`batch_id` is the one schema addition this requires (Spec 2 §4.1):

```sql
ALTER TABLE jobs ADD COLUMN batch_id TEXT;
CREATE INDEX idx_jobs_batch ON jobs(batch_id);
```

It earns its place downstream — lineage shows a job's batch siblings, and
`GET /api/jobs?batch_id=…` filters a submission out of the list. It is
deliberately *not* a grouping in the rail: an early version collapsed each batch
under a "batch · N variants" heading, and in practice that heading only got in
the way of scanning a flat, newest-first list. The rail treats every job as its
own row; a batch is a fact about where jobs came from, not a container.

**Compare is where a batch is consumed.** When a batch's jobs reach `waiting`,
the compare view shows them with only their differing fields — which, for a seed
batch, means the grid is captioned purely by seed number, and for a mixed batch
by seed and caption. Promote acts per cell. Running a second batch from a
promoted variant continues the family, and lineage keeps the chain readable.

## 7. Compare view

Selecting two or more jobs in the rail opens a grid: 2×2 for four, sized to
fill.
Each cell shows the artwork, the title, the ring, and the parameter deltas
*relative to the others* — only the fields that differ, so the caption or seed or
conditioning scale that actually distinguishes them is immediately visible.

Each cell carries `Promote` directly. This view is where the core workflow
completes, and it should be possible to go from four finished previews to one
promoted job in two clicks without visiting a detail page.

A shared scrubber is offered when the cells have overlapping frame ranges: one
slider moves all four to the same epoch. Comparing candidates at iteration 300
rather than at their respective finishes is often more informative.

## 8. New job flow

A single scrolling page in four sections, not a wizard with steps — every part
stays visible and adjustable until you submit.

### 8.1 Source

Drop or pick an image; it uploads content-addressed and returns a sha256.
Recently used sources are offered for reuse, since the same photo is often run
many times with different parameters.

### 8.2 Prepare — the canvas

All client-side, operating on an `ImageData` working copy. Source images are
downscaled to a working maximum of 2048px on the long edge for interaction
speed; the export is produced from the full-resolution original using the same
selection, upsampled.

**Tools:**

- **Select similar (magic wand).** Click a pixel; flood-fill contiguous
  neighbours within a tolerance. Tolerance is evaluated in CIE Lab, not RGB —
  RGB Euclidean distance produces visibly wrong selections on skies and skin,
  which is exactly the busy-background-behind-a-car case. Shift-click adds
  another cluster; alt-click subtracts. A `contiguous` toggle turns the same
  tool into a global chroma key: every pixel in the image within tolerance of
  the clicked colour, regardless of position.
- **Tolerance** and **Feather** are separate controls. Tolerance decides what is
  selected; feather decides how the edge falls off, applied as a blur on the
  selection's alpha after the binary selection is computed. Conflating them is
  the usual mistake and makes both unusable.
- **Brush add / remove** for touch-up, with adjustable radius and hardness.
- **Undo/redo** over the full selection history, keyboard bound.

Selection is displayed as a marching-ants outline plus an optional 50% wash, and
can be inverted. A checkerboard preview shows what will actually be removed.

**Mask mode — and an honest note about what the mask does.**

SLDgen always runs RMBG-1.4 on the target, and the RMBG mask also drives the
object bounding-box rescale. So a browser selection cannot simply "be the mask".
Three explicit modes, stated in the UI in plain terms:

| Mode | What is exported | Flags | Effect |
|---|---|---|---|
| **Clean up the image** | `target.png` with the selection knocked out to white | none | RMBG still decides density; you have only removed distractions |
| **Guide the ink** | also a grayscale `weight.png` | `--stipple-weight weight.png --stipple-weight-mode multiply` | Your painting modulates density within what RMBG considers subject |
| **Control the ink** | also `weight.png` | `--stipple-weight … --stipple-weight-mode replace` | Your painting *is* the density field; RMBG no longer affects density (it still affects the bounding-box rescale) |

The second and third modes turn the brush into a **density brush** — paint
towards 1 for full ink and towards 0 to hold it back, in `--stipple-weight`'s
own convention, so the exported PNG is white where the ink goes. On the canvas
the suppressed areas are the ones that shade over, since full density is the
untouched state. A soft field like this is a far better instrument than a binary
mask and is what the `--stipple-weight` flag was built for. The UI should offer
the density brush as its own tool once a non-default mask mode is chosen.

**Origin pin.** Click to place; shown as a crosshair with its normalized
coordinates in monospace. Draggable. A toggle enables or disables its use
without discarding the position (see §9).

**Reference overlays.** Any `avoid`, `attract` or `init_points` SVG chosen in
§8.3 is drawn over the canvas in a hairline so you can see the spatial
relationship between the new curve's constraints and the image before
submitting. This is the difference between the constraint flags being usable and
being guesswork.

### 8.3 Parameters

Grouped as: **Prompt** (caption, with a note that leaving it empty triggers
BLIP-2 auto-captioning and that the resolved caption will appear on the job
once it starts), **Curve** (control points, init method, width mode, seed),
**Guidance** (condition type, conditioning scale, LoRA weight), **Constraints**
(origin, avoid, attract, init points, stipple weight), and **Losses**
(the regularisation weights, collapsed by default).

Constraint inputs that reference other jobs use a picker showing job thumbnails,
restricted to `final_sld.svg` of completed jobs and to committed partitions,
per §6.2's warning.

Every optional parameter renders as a row with an enable toggle on the left and
its value controls on the right, greyed but **not cleared** when disabled.

### 8.4 Budget and submit

Horizon (`num_iter`, default 4000) and this run's budget (`target_epoch`,
default 400) as two clearly distinct fields, with a one-line explanation that
the horizon sets the schedule and the budget sets where this run stops — because
this distinction is the whole point of Spec 1 §3 and will otherwise be
misunderstood.

An estimated duration derived from the running average it/s across past
segments. `Queue job` submits; the rail selects the new job immediately.

## 9. Parameter persistence

**Requirement: the next job starts from the last job's settings.**

The model is that every parameter is stored as `{enabled, value}`, and **both
fields persist independently**. Disabling the origin does not discard its
coordinates; the next new-job form shows the pin exactly where it was, switched
off, one click from being used again. The same applies to every optional flag.

Persisted **server-side**, not in `localStorage`, so the state survives browser
changes and cache clears. This requires a small addition to Spec 2's API:

```
GET  /api/params/last            → the last-submitted parameter object
PUT  /api/params/last            → called on submit
GET  /api/params/presets         → [{id, name, params}]
POST /api/params/presets         {name, params}
DELETE /api/params/presets/{id}
```

Named presets are the natural extension and cost almost nothing once the
last-used record exists — "portrait, high detail" and "quick look" are the two
you will make within a week.

What does *not* persist: the source image, the selection, and the title. Those
are per-job by nature. The mask *mode*, however, does persist.

## 10. Status bar and telemetry

Always visible, one line, monospace throughout:

`● car_03 · 1512/4000 · 4.9 it/s · ~8m left │ queue 3 │ GPU 26.1 GB free │ disk 4.2 GB (+310 MB) │ worker ●`

- Running job with live epoch, rate and ETA, from the job's SSE stream.
- Queue depth; clicking filters the rail to `queued`.
- GPU free memory from `/api/health`, polled every 10 s. Diagnostic only — there
  is no VRAM gate (Spec 2 §9) — but it is what you look at when a job fails with
  OOM.
- Total disk under the work root, with the delta since the session started, so
  growth is visible without arithmetic.
- Worker liveness dot. If the worker unit is down, this is the *only* place that
  would show it, so it must be unmissable: the bar turns to `--st-failed` and
  reads "Worker not running", with a link to the worker journal
  (`/api/logs/worker`).

Clicking the disk figure expands a panel: total, breakdown by category
(checkpoints, frames, videos, logs, uploads, final artefacts), and the ten
largest jobs.

## 11. Cleanup

From the disk panel, and from rail multi-select:

- Delete selected jobs.
- Delete all failed jobs.
- Delete all completed jobs older than N days.
- Prune checkpoints from completed jobs, keeping the last.
- Prune intermediate frames from completed jobs that already have an mp4.
- Remove orphaned uploads no longer referenced by any job.

Every action states the exact number of jobs and bytes it will free **before**
confirmation. Deleting more than one job, or anything over 1 GB, requires typing
the job count to confirm — friction proportional to consequence. Deletion is
asynchronous (Spec 2 §10); affected rows show the `deleting` ring until gone.

Logs are never pruned (Spec 2 §13.3) and the UI should say so where a user might
expect otherwise.

## 12. Partitions

On a completed job: a partition panel with strategy, N, origins, connect-tails
and sample spacing. Because partitioning is CPU-only and synchronous
(Spec 2 §11), parameters update a **live preview** — `partition_preview.png`
overlaid on the master, with each partition in a distinct colour. This is the
one place colour appears outside job state, and it is functional: the colours
*are* the partition identity. Scrub the strategy, see the split, commit when
right.

Committing writes a `partitions` row; committed partitions then appear in the
constraint picker of §8.3 as `attract` and `avoid` sources, which is how the
sequential compositional workflow closes the loop.

For `labelmap`, the labels default to the job's own `condition_*.png` with an
option to upload a hand-painted map.

## 13. Realtime and resilience

The API will be restarted frequently during development, and jobs must survive
it. The UI must too:

- One SSE connection for the selected job, one for a global job-list stream.
- Exponential backoff reconnection with a visible but undramatic "reconnecting"
  state in the status bar — not a modal, not a toast storm.
- On reconnect, refetch rather than assuming continuity; log tails resume from
  the last received byte offset.
- Every SSE-fed view must also work under plain polling, because SSE through a
  misconfigured proxy fails in ways that are hard to diagnose. Poll at 2 s as
  the fallback.
- Optimistic UI only for actions that set `desired_state` (pause, cancel,
  delete). Never optimistically show a state the worker owns.

## 14. Keyboard

`j`/`k` move through the rail. `Enter` opens. `Space` plays/pauses the
filmstrip. `←`/`→` step frames. `p` promotes a waiting job. `l` opens the log.
`n` starts a new job. `/` focuses the filter. Shown in a `?` overlay.

## 15. Copy

Interface voice: plain, active, specific. Actions name their outcome and keep
that name through the flow — the button says `Promote`, the result says
`Promoted to 4000`. Never expose internal vocabulary: the user promotes a job,
they do not "set target_epoch". Errors state what happened and what to do, in
the interface's voice, without apology. Empty states are invitations — an empty
rail reads "No jobs yet. Prepare an image to get started." with the action
attached.

## 16. Non-goals

- Multi-user anything: no accounts, sessions, permissions, or presence.
- Mobile layouts. A phone cannot do the prep canvas usefully; the app declares a
  minimum width and says so rather than degrading badly.
- In-browser plotting, pen control, or device communication.
- Editing SVGs. This UI generates and inspects; it does not draw.
- Re-implementing partition strategies client-side. The script is the authority.

## 17. Open questions

1. **Density brush vs selection as separate tools, or one tool with a mode?**
   Specced as separate, appearing when the mask mode calls for it. Worth
   revisiting after first use — it may be that painting density is the *only*
   thing you want, and binary selection is vestigial.
2. **Should compare allow more than four?** Specced as a responsive grid with no
   hard cap, but the useful number is probably three or four; beyond that the
   cells are too small to judge line quality.
3. **Estimated pen travel** on the result tab assumes a plotting context and a
   speed constant. Either make the constant configurable or drop the time
   estimate and show path length only.
4. **Auto-promote rules.** A "promote automatically if it survives to N" policy
   would remove a decision point, but it also removes the review the whole
   workflow exists to enable. Not specced; noted as a temptation to resist.

---

## 18. As built

### Files

| Path | Contents |
|---|---|
| `sldgen_web/src/lib/` | the pure logic, all unit-tested: `lab` (Lab conversion, wand, feather, brush), `ring`, `params`, `paramdiff`, `variants`, `formstate`, `logcook`, `format`, `backoff` |
| `sldgen_web/src/api/` | `client` (typed fetch), `types` (the wire shapes), `stream` (SSE with the polling fallback) |
| `sldgen_web/src/components/` | `Ring`, `JobRail`, `StatusBar`, `DiskPanel`, `ArtworkPane`, `Filmstrip`, `LogViewer`, `JobData`, `ActionsPanel`, `ParamFields`, `ConstraintPicker`, `PrepCanvas`, `PartitionPanel`, `RunAgainDialog`, `HelpOverlay` |
| `sldgen_web/src/pages/` | `JobsPage`, `JobPage`, `ComparePage`, `NewJobPage` |
| `sldgen_web/src/styles/` | `tokens.css` (§3's palette and type), `app.css` |
| `start.sh`, `stop.sh` | the tmux deployment, and a graceful drain |
| `docs/RUNNING.md` | the operator's manual, written for someone who has not used tmux |
| `test_service_web.py` | 69 checks against a live API for everything §12 gained |

### Additions to the Spec 2 API

The UI needed seven things Spec 2 did not have. All are additive; no existing
endpoint changed shape.

```
GET/PUT  /api/params/last                the {enabled, value} form state (§9)
GET/POST /api/params/presets             named presets
DELETE   /api/params/presets/{id}
GET      /api/jobs/{id}/frames           the contact sheet, incl. the rescale flag
GET      /api/jobs/{id}/lineage          parent, variants, batch siblings
GET      /api/events                     the rail's global stream
POST     /api/maintenance/cleanup        every §11 action, with a truthful dry run
GET      /api/jobs?parent_job_id=&with_params=
```

Schema: two additive tables, `ui_state` and `presets`, and `SCHEMA_VERSION` 2.
`batch_id` was already there — Spec 2 built §6.6's one schema addition ahead of
time, so nothing was needed for batches.

### Departures from the design above

1. **`Cancel` pauses and withdraws the budget; it does not produce `waiting`.**
   §6.3 asks for cancel to leave the job at `waiting`, but Spec 2's state machine
   reaches `waiting` only by *reaching* a budget, and `paused` is precisely the
   state for "the user intervened". So cancel pauses, then pulls `target_epoch`
   back to the epoch actually reached — which makes `Resume` refuse with "promote
   it instead", exactly how a job that finished its budget behaves. Same meaning,
   no new state, no change to the worker.

2. **Log cooking happens in the browser, on the whole buffer.** §6.4 has the API
   cook (Spec 2 §13.2), and it still does for whole-file reads. But the viewer
   fetches incrementally by byte offset and a tqdm progress line routinely spans
   a chunk boundary, so cooking each chunk in isolation and concatenating
   produces a line no terminal would show. The viewer fetches `raw=true` and
   cooks locally; `lib/logcook.ts` mirrors `logs.py` and the two are tested
   against the same cases. The raw toggle became instant as a side effect.

3. **The parameter schema is duplicated in TypeScript.** `lib/params.ts` mirrors
   `PARAM_SPECS`. Fetching it instead would make the new-job form unusable until
   a round trip completed, and the server remains the authority either way — it
   canonicalises and validates every submission. `test_service_web.py` parses the
   TypeScript and asserts name-for-name, default-for-default agreement, so drift
   fails the suite rather than reaching a user.

4. **`stipple_weight` cannot be picked from an existing file.** §8.3 lists it
   among the constraint inputs, but a hand-supplied weight map has no way to
   register with the prepared target, whose framing the canvas has just changed.
   The picker says so and points at the density brush, which is where a weight
   map that *does* register comes from.

5. **No estimated pen travel time** (§17.3). Path length is measured with the
   browser's own `getTotalLength`, so curves are measured rather than
   approximated. The time estimate is dropped: it needs a speed constant nobody
   has established, and a confidently wrong number is worse than none.

6. **`with_params` is opt-in on the job list.** The variant table needs every
   job's parameters to flag a duplicate seed; the rail does not, and including
   them roughly triples the payload it polls.

7. **The default tmux session is `sldgen-service`, not `sldgen`.** Discovered by
   collision: `sldgen` was already in use on the host.

### Answers to §17's open questions

1. **Density brush vs selection — separate tools**, as specced, with the density
   brush appearing only once a non-default mask mode is chosen. Worth revisiting
   after real use, exactly as the spec says.
2. **Compare has no hard cap.** A responsive grid with a 300px minimum, so the
   cells shrink until they stop being judgeable and the layout tells you rather
   than a rule.
3. **Pen travel** — answered in departure 5: length only.
4. **Auto-promote** — not built. The temptation was resisted.

### What the first real GPU run found

A short run (20 iterations, then promoted to 40) through the shipping stack
produced `input.png`, `mask.png`, `condition_depth.png`, paired frames and SVGs,
a checkpoint, and a clean resume: segment 1 ran 0→20 fresh, segment 2 ran 20→40
from `latest.pt`, both exit 0.

It also settled something the spec treated as an edge case. **`rescaled` is true
for a default job**: `object_size_ratio` defaults to 0.75, which produced
`scale_w 0.8`, so `svg_logs/` is in a different coordinate space than
`final_sld.svg` on essentially *every* run. §6.2's warning is therefore not a
rare caveat but permanent furniture, which is why it is styled as a quiet note
rather than an alert — and why the constraint picker offers only `final_sld.svg`
rather than trying to detect the exception.

### Tests

| Suite | Checks | Covers |
|---|---|---|
| `sldgen_web` vitest | 137 | Lab conversion against CIE reference values, flood fill (contiguous and global), feather monotonicity, brush falloff and clipping, ring geometry incl. the partially-completed case, log cooking against `logs.py`'s cases, the `{enabled, value}` persistence contract, variant seeds and duplicate detection, parameter validation, ETA and rate maths, reconnection backoff |
| `test_service_web.py` | 69 | static serving, params/last surviving an API restart, presets, frames and the rescale flag, the API refusing a rescaled intermediate as an input, lineage and batches, the list filters, the global stream, and that cleanup's dry run reports exactly what the real run then does |

Run them:

```bash
cd sldgen_web && npm test
PYTHONPATH=. .venv-service/bin/python test_service_web.py
```

The Spec 2 suites (232 checks) still pass unchanged.

### Not done

- **No browser-driven end-to-end test.** The logic is unit-tested and the API is
  integration-tested against live daemons, but nothing drives a real Chromium.
  The prep canvas in particular is verified only through its pure functions.
- **The prep canvas has not been used in anger.** Its export path — selection at
  working resolution, upsampled onto the full-resolution original — is correct by
  construction and untested against a real photograph.
- **`GET /api/logs/worker`** still needs journalctl, so under `start.sh` (where
  the worker is a tmux pane, not a unit) it returns its explanatory message. The
  worker's output is in the pane; the UI links to the endpoint regardless.
- **Reduced motion** freezes the running indicator, but the filmstrip's play
  control is unaffected — it is user-initiated, so muting it would be wrong.