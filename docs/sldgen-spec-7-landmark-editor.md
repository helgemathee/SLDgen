# SLDgen Spec 7 — Landmark editor, auto-detect, turned and profile faces

Status: implementation spec, 2026-09-26. Builds on Spec 6 (image fidelity loss),
whose landmark term this makes usable on real portraits.

## 1. Problem

Spec 6's landmark term pulls the line toward a few weighted anchor points. The
term itself is anatomy-agnostic — `load_landmarks` reads `xy` and `weight`
and nothing else — but the only way to get anchors is `sld_landmarks.py`, a
one-shot MediaPipe Face Mesh extraction:

* **No editing.** A wrong point can only be fixed by hand-editing JSON.
* **Turned faces get invented points.** Face Mesh always fits a whole face.
  Past ~30° of head turn the far eye, far mouth corner and far jaw are placed
  on or behind the nose bridge — points that do not exist in the picture, at
  weight 3.0 for the eyes. Measured on test portraits: the far-side points
  collapse onto the nose line while the near-side points stay usable even at
  ~70° (§4.2).
* **Full profiles find nothing.** Face Mesh returns no face on a true profile
  (the `Gesa` job: full-range detector finds the box, mesh on the upscaled
  crop still fails). The service answers 422 and the user has no way forward.
* **Not only faces.** Anchors are useful on anything with a few must-hit
  points (knuckles, a building's corners) — there is no way to place them.

## 2. Outcome

1. A **landmark editor** in the Image fidelity panel, a second tab beside the
   edge map preview: zoom, pan, add, move, delete, rename and re-weight points
   on the canvas image, with a table on the right.
2. **Auto-detect** inside the editor that *fills in* — it never overwrites a
   point the user placed or moved — plus **Replace all** and **Detect in box**.
3. **View-aware detection**: head turn is estimated; hidden-side points are
   dropped past per-point limits; strongly turned and profile faces get
   profile points read off the subject's silhouette; if Face Mesh fails
   entirely, MediaPipe Pose supplies the visible eye and mouth corner.
4. A **profile template**: named, pre-weighted rows the user can place by hand
   when detection is not good enough.

Non-goals: preparing canvas space without a run (the editor, like every
canvas-space preview, needs a previous run of the image — §7); multi-face
images (the largest/most confident face is used); changing the loss.

## 3. Landmark file (extended, backward compatible)

```json
{
  "space": "canvas",
  "image_size": [512, 512],
  "preset": "portrait" | "all" | "edited",
  "view": {"kind": "frontal|turned|profile", "yaw": -47.0, "method": "mesh|pose", "facing": "left|right"},
  "dropped": ["right_eye_outer", ...],
  "landmarks": [
    {"name": "left_eye_outer", "xy": [212.4, 230.1], "weight": 3.0,
     "source": "mesh|pose|silhouette|manual", "edited": false}
  ],
  "all_landmarks": [[x, y], ...]
}
```

`source`, `edited`, `view` and `dropped` are new and optional; the loss
ignores them (it reads `xy` and `weight` only), so every Spec 6 file stays
valid and every Spec 7 file loads in Spec 6 code.

## 4. Detection (`sld_landmarks.py`)

### 4.1 Pipeline

```
image (canvas), optional --mask, optional --box x0 y0 x1 y1
  └─ region = box (grown 25 %) or whole canvas, upscaled to ≥ 512 px
      1. Face Mesh (as today, incl. the full-range-detector crop fallback)
         → 478 points with z  → yaw (§4.2) → portrait points, culled (§4.3)
         → |yaw| ≥ PROFILE_YAW and a mask: add silhouette points (§4.4)
      2. else Pose → nose, eyes, mouth corners, ears
         → facing from nose vs ears; near side from eye depth
         → near-side eye (inner, pupil, outer) + near mouth corner
         → with a mask: add silhouette points (§4.4)
      3. else exit 2 "no face found"
```

`--preset all` keeps today's behaviour (every mesh point, weight 1, no
culling); it needs step 1.

### 4.2 Head turn (yaw)

The legacy Face Mesh API gives no pose matrix and the canonical face model is
not bundled, so yaw is read from the detected mesh itself: triangles are
recovered from `FACEMESH_TESSELATION` (3-cliques of its edge graph, 854
triangles), per-vertex normals are the sum of outward-oriented face normals,
smoothed over a 3-ring neighbourhood, and yaw is the angle of the mean normal
of the forehead-to-nose-tip midline (`10 151 9 8 168 6 197 195 5 4`) in the
x–z plane. Negative yaw: the face turns toward image-left and the subject's
right side is the far side.

Measured: frontal portraits 0–9°, a three-quarter view −32°, a ~60° view −47°,
near-profiles −51°. The estimate **saturates at ~50°** because the mesh itself
cannot represent more turn — which is exactly why §4.4 exists.

Per-point normals were tried for culling and rejected: eye corners and mouth
corners are concave and face sideways even on a frontal face (inner eye
corners measure −0.1 to −0.2 facing at yaw 0).

### 4.3 Culling hidden-side points

Each portrait point has a side (`left`/`right`/`mid`, the subject's). Far-side
points are dropped when `|yaw|` exceeds their limit; near-side and midline
points are always kept:

| far-side point | limit |
|---|---|
| cheek, jaw | 20° |
| eye outer, brow outer | 25° |
| mouth corner | 35° |
| pupil, brow mid, nostril | 38° |
| eye inner | 40° |
| brow inner | 45° |

Dropped names go into `dropped` and are reported to the user.

### 4.4 Profile points from the silhouette

When the face is strongly turned (`|yaw| ≥ PROFILE_YAW = 42°`) or only Pose
found it, the identity of a face is in its outline, and the mesh (saturated)
or Pose (coarse) cannot place those points. With the run's `mask.png`:

* `facing` = sign of yaw (mesh) or nose vs ears (Pose).
* `eye_y`, `mouth_y` from the near eye / mouth; `s = mouth_y − eye_y`.
* For each row in `[eye_y − 1.4 s, mouth_y + 1.6 s]`, the front of the head is
  the most forward mask pixel within `3 s` of the eye horizontally; the
  profile `p(y)` is that forwardness.
* Named extrema of `p` in fixed bands: `brow_ridge` (max above the eye),
  `nasion` (min between brow and nose tip), `nose_tip` (max between eye and
  mouth), `subnasale` (min between nose tip and mouth), `upper_lip` (max),
  `stomion` (min), `lower_lip` (max), `chin_front` (max below the
  mentolabial fold, the min under the lower lip).
* **Consistency check (mesh path)**: the silhouette nose tip must lie within
  `0.5 s` of the mesh nose tip (index 4); otherwise the outline is not the
  face's front (hair, a hand, a bad mask) and no silhouette points are added.
* Missing extrema (flat bands) are skipped, never guessed.

A beard or fringe makes the outline follow the hair: the points are then on the
beard's outline — which is also what the line will draw.

Weights: `nose_tip` 3, `nasion` 2, `upper_lip` 2, `lower_lip` 2, `stomion` 1.5,
`subnasale` 1.5, `brow_ridge` 1.5, `chin_front` 1.

### 4.5 CLI

`--mask PNG` (canvas-space subject mask; enables §4.4), `--box X0 Y0 X1 Y1`
(canvas pixels; detect inside it). Output prints the view line
(`view  profile  yaw -51  method mesh`) and the dropped names.

## 5. API

`POST /api/image-loss/landmarks` gains `box: [x0, y0, x1, y1]` (optional) and
always passes the source run's `mask.png` when present. Response adds `view`
and `dropped`. Errors unchanged (422 no face / no MediaPipe, 400 bad input).

`GET /api/image-loss/canvas?target_sha256=…` (new): the latest run of this
image that has an `input.png` — `{source_job_id, image_url, image_size}` — so
the editor can show the canvas before any detection. 404 when there is none.

## 6. Web UI — the landmark editor

### 6.1 Placement

Inside Image fidelity, the preview area gets two tabs: **Edge map** (today's
preview, unchanged) and **Landmarks**. The Landmarks tab is available whenever
image fidelity is on; when points exist but the Landmark term is 0 the editor
shows "Landmark term is 0 — these points do nothing" with a button that sets it
to 1.

### 6.2 Canvas

One SVG in canvas coordinates: the canvas image, optionally the edge map at
low opacity ("Show edges"), and the points. Zoom by wheel (about the cursor,
1×–16×) and the +/−/Fit buttons; pan by dragging empty space. Points: radius
from weight (as today), colour by source (detected vs manual), selected point
ringed. Interactions:

* click a point: select; drag a point: move it (it becomes `edited`);
* double-click empty canvas: add a manual point there (`p1`, `p2`, …);
* with a row selected that has no position (a template row), click places it;
* `Delete`/`Backspace` removes the selected point; `Esc` deselects;
* **Detect in box** mode: drag a rectangle, release to detect inside it.

### 6.3 Table (right)

Columns: name (editable), weight (number, 0–5, step 0.5), x, y (editable,
canvas px), source, delete. Unplaced template rows show "click canvas to
place". Selecting a row selects the point and vice versa.

### 6.4 Actions

* **Auto-detect**: runs §5; merge rule (pure function `mergeDetected`):
  points the user placed or moved are kept untouched; detected points that were
  never edited are updated; new names are added; never-edited detected points
  the new detection no longer reports (now culled) are removed. The result line
  reports view, yaw, added/updated/kept/removed and dropped names.
* **Replace all**: detection result replaces the table.
* **Profile template**: adds the §4.4 names (and `ear`, `jaw_angle`) as
  unplaced rows with their default weights, skipping names already present.
* **Clear**.

### 6.5 Persistence

The editor's points are the value of the `image_loss_landmarks` input role.
After every change (debounced 400 ms), placed points are serialised to a
`preset: "edited"` file, uploaded (content-addressed) and attached with label
`N landmarks (edited)`. Submission waits while an upload is pending (same
`onBlock` mechanism as the prepared edge map). On load, an attached upload is
fetched from `/api/uploads/<sha>` and becomes the editor state, so a reload or
a restored form continues where it left off. In Run again the inherited file is
shown read-only (unchanged from Spec 6).

## 7. Constraints kept

* Canvas space only; the editor works on a previous run's `input.png` and the
  run refuses a file at another render size (Spec 6 contract).
* The API venv never imports MediaPipe or numpy; detection stays a subprocess
  under the sldgen interpreter. It is CPU-only and never touches the GPU.

## 8. Tests

* `test_landmarks_geom.py` (CPU, no MediaPipe): yaw from synthetic rotated
  meshes; culling table by yaw sign; silhouette extrema on a synthetic profile
  mask (known nose tip / lips / chin rows); consistency check rejecting a far
  nose tip; box mapping back to canvas coordinates.
* `test_service_image_loss.py`: `box` accepted and forwarded, `view`/`dropped`
  in the response, `/api/image-loss/canvas` 200/404.
* `sldgen_web/src/lib/landmarks.test.ts`: merge rule, serialise/parse
  round-trip (unplaced rows dropped), zoom-about-point and clamping, template
  insertion without duplicates, name generation.
* Manual (MediaPipe): frontal, three-quarter, ~60°, near-profile and the full
  profile of the `Gesa` job; results recorded in §9.

## 9. As built

(filled in on completion)
