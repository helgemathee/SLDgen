# Addendum to Spec 7: Configurable Landmark Set Generation

Status: implementation spec, 2026-09-26, **adjusted to the code**. The first
draft (commit `87ab565`) was written without the repository in view. §0 lists
what changed and why. §1–§9 are the spec that gets built.

Extends `sld_landmarks.py` (Spec 6 §8.2, Spec 7 §4), the landmark loader
`load_landmarks` in `SLDgen/image_loss.py`, the `POST /api/image-loss/landmarks`
endpoint, and the landmark editor (Spec 7 §6).

There is **no new loss mathematics**. `ImageLoss.landmark()` (weighted mean
distance from each anchor to its nearest curve point) stays unchanged.
Everything here decides which weighted points reach it. The one runtime change
is in the *loader*, which turns polylines into points (§6).

## 0. Changes from the first draft

| Draft | Code reality | Adjusted |
|---|---|---|
| `tools/extract_landmarks.py INPUT OUTPUT`, `--preset portrait` | The tool is `sld_landmarks.py --image … --out …`, with `--preset portrait\|all`, `--mask`, `--box`. Detection is view-aware (Spec 7 §4): turn estimate, far-side culling, silhouette profile points, Pose fallback. | New `--landmark-set` on `sld_landmarks.py`. `--preset` stays. Every set goes through the same view-aware pipeline. |
| Detector coordinates at photo size (`image_size: [1024, 1365]`) | Landmark files are canvas-space only: `space: "canvas"`, `image_size == [render, render]`, taken from a run's `input.png`. The run refuses anything else. | Unchanged contract. Polyline coordinates are canvas pixels too. |
| `sparse` ≈ 20 points | The portrait preset has 21 points and drops far-side points on turned faces. | `sparse` *is* the portrait preset, byte for byte (§2). |
| `pose-locked`: `cv2.solvePnP` against a canonical 3D model, then write the **projected canonical model** as the landmarks | (a) The canonical model is not bundled with MediaPipe 0.10.21. (b) Face Mesh already returns a 3D mesh (x, y, z in one weak-perspective frame), so an orthographic 3D fit needs no guessed camera intrinsics. (c) A projected *canonical* face is the average face: it would replace this person's eye spacing, nose length and mouth width with the mean's. That is the identity the landmark term exists to hold. | Vendor MediaPipe's `canonical_face_model.obj` (Apache-2.0). Fit it rigidly (similarity, trimmed) to the detected mesh. The fit decides visibility and replaces **inconsistent** detections: the far-side points Face Mesh collapses onto the nose, and outliers. It keeps detections that agree with it (§3.3). |
| `--pose-report` prints yaw/pitch/roll | The file already carries `view.yaw` from the mesh normals. | `--pose-report` adds a `pose` key from the rigid fit: yaw, pitch, roll, fit residual. The API always requests it and the editor shows it. |
| Weight renormalisation "so eyes vs jaw emphasis is preserved" (method unspecified) | The loss is `Σ wᵢdᵢ / Σ wᵢ`: total pull is already independent of the point count. What a denser set changes is *where* the pull goes. | Anchor-cell budgets (§4): each sparse anchor's weight is shared out over the points nearest to it. Per-region emphasis is then identical across sets, by construction. |
| `--include-hairline` from "the RMBG mask" | The run's `mask.png` separates subject from background. Hair belongs to the subject, so that mask has no hair/forehead boundary. | Skin-colour march up from the forehead, stopping where skin ends. An optional `--hair-mask PNG` replaces the colour test (§7.2). |
| `--include-glasses`: "click points, or a coordinate list" | A CLI cannot take clicks. The landmark editor is where points get clicked. | CLI: `--include-glasses FILE` with canvas-pixel polylines. Web: a **Glasses** template in the editor seeds two rims around the eyes, to be dragged onto the frames (§7.1, §8). |
| Loader densifies polylines, "linear or spline" | — | Linear, ≈ 8 canvas px spacing (§6). Rims are drawn with enough vertices. |
| — | The editor serialises its state as `preset: "edited"` and would silently drop unknown keys. | The editor round-trips `polylines` and `pose`, and edits polylines (§8). |

## 1. CLI

```
python sld_landmarks.py --image RUN/input.png --out landmarks.json \
    [--mask RUN/mask.png] [--box X0 Y0 X1 Y1] \
    [--landmark-set sparse|standard|dense|pose-locked] \
    [--include-glasses GLASSES.json] [--include-hairline [--hair-mask HAIR.png]] \
    [--pose-report]
```

* `--landmark-set` defaults to `sparse`.
* `--preset all` (every mesh point at weight 1, no culling) is kept unchanged.
  It is refused together with a non-`sparse` set. `--preset portrait` is the
  same as `--landmark-set sparse`.
* A set other than `sparse` needs Face Mesh. When only Pose finds the face (a
  true profile), every set falls back to the Spec 7 Pose + silhouette points.
  The file then says so in `landmark_set` and `view.method`.

## 2. The four sets

| Set | Points (frontal) | What it constrains |
|---|---|---|
| `sparse` | 21 | Current portrait preset, unchanged: eye corners, pupils, nostrils, mouth corners, brow arcs (3 each), chin, 2 jaw, 2 cheek. |
| `standard` | ≈ 50 | Sparse, plus: 2 more brow-arc points per side, upper/lower lid middle, nose tip / subnasale / nasion / alar wings, lip contour (top, peaks, bottom, stomion), 4 more face-outline samples per side. |
| `dense` | ≈ 120 | The standard set, extended by farthest-point sampling over the canonical face (deterministic) to 120 mesh points spread evenly over the face. Pose becomes over-determined. |
| `pose-locked` | ≈ 120 | As `dense`, but every point is checked against a rigid fit of the canonical face (§3.3). |

**Byte identity.** Without `--pose-report`, `--landmark-set sparse` output is
byte-identical to the pre-change `--preset portrait` output, including its
`"preset": "portrait"`. The new sets write `"preset": "<set>"` and add
`"landmark_set"`.

Names: standard points have anatomical names in the same `left_`/`right_`
convention as the portrait preset (the subject's side). Unnamed dense points
are `m<index>` (the Face Mesh index). The mesh midline points `nose_tip`,
`subnasale`, `nasion`, `upper_lip`, `stomion` and `lower_lip` share names with
the silhouette profile points (Spec 7 §4.4). When the silhouette supplies a
name, it replaces the mesh point of that name, because on a turned face the
outline is where that feature is actually drawn.

## 3. View-aware culling and the rigid fit

### 3.1 sparse, standard

These use the Spec 7 §4.3 per-name limits. The new names get limits in the
same spirit (outline samples 20°, outer brow arc 30°, lids 32°, lip peaks /
lower lip corners / alar wings 38°, inner brow arc 42°). Midline points are
never culled.

### 3.2 dense

Culling by name cannot cover 120 unnamed points, so the limit follows the
point's lateral position in the canonical face. A far-side point at
`|x| / max|x| = u` is dropped past `45° − 25°·u`: 45° at the midline, 20° at
the outline. This is the same range the name table spans, from inner brow to
cheek. Positions are the raw detections.

### 3.3 pose-locked

1. **Fit.** Map the canonical model into image axes (x, −y, −z). Fit a
   similarity transform (Umeyama) to the detected mesh (x, y, z in pixels) on
   the stable subset: eye corners, nostrils, nose base, mouth corners, chin.
   Near-side and midline points are always used. Far-side points are used only
   inside their §3.1 limits. After the first fit, drop the worst quarter by
   residual and refit.
2. **Visibility.** Rotate each canonical vertex normal by the fitted rotation.
   A point is visible when its normal faces the camera (`−n_z ≥ 0.2`).
   Invisible points are dropped.
3. **Position.** For a visible point, keep the detection if it is
   *consistent*: within its §3.2 lateral limit and within
   `max(3 px, 0.12 × inter-ocular distance)` of the fitted projection.
   Otherwise write the fitted projection with `source: "model"`. This is the
   case for far-side points the mesh has collapsed onto the nose, and for
   outliers.

On a three-quarter view, far-side points that are visible but collapsed in the
mesh come back at the rigid model's positions. Dense would drop them. Points
the detector placed consistently keep the subject's own geometry. Beyond ≈ 50°
the mesh saturates and so does the fit (Spec 7 §4.2), and the silhouette
points carry the face as before.

### 3.4 Pose report

With `--pose-report` (mesh path) the file gets:

```json
"pose": {"yaw": -18.4, "pitch": 6.1, "roll": -3.2, "method": "rigid-fit", "residual_px": 2.1}
```

Yaw uses the file's convention: negative faces image-left. Pitch is positive
when the head tilts up, roll positive when it tilts clockwise in the image. On
the Pose path, `pose` carries the coarse Pose yaw and `pitch`/`roll` null. The
loss never reads this key. It is a diagnostic, and it tells the author what to
write in the caption. The CLI prints it as a `pose` line.

## 4. Weight normalisation (anchor-cell budgets)

`sparse` keeps its weights. For `standard`, `dense` and `pose-locked`, every
point is assigned in the canonical model to its nearest sparse anchor (one of
the 21 portrait points). Points more than 2.5 canonical units (≈ cm) from
every anchor go to a separate *surface* cell with budget 1.0. Each anchor's
cell then shares the anchor's sparse weight equally. Weights are rounded to 4
decimals.

Consequences, each of them tested:
* Each anatomical region carries the same share of the total weight in every
  set: the sum of weights over the right-eye cells is the same fraction of the
  total in `sparse` and in `dense`.
* The same `--image-loss-landmark` alpha gives comparable pull. Shifting every
  landmark by d px gives a landmark loss of d for every set, because the loss
  is a weighted mean.
* Culling runs after normalisation, so a culled point takes its share with it,
  as a culled sparse point does today.

Silhouette points keep their Spec 7 weights. Polylines carry their own total
weight (§6).

## 5. Landmark file (extended, backward compatible)

```json
{
  "space": "canvas",
  "image_size": [512, 512],
  "preset": "portrait | all | standard | dense | pose-locked | edited",
  "landmark_set": "dense",
  "view": {"kind": "turned", "yaw": -31.9, "method": "mesh", "facing": "left"},
  "pose": {"yaw": -30.2, "pitch": 4.0, "roll": -2.1, "method": "rigid-fit", "residual_px": 1.8},
  "dropped": ["right_cheek", "m234"],
  "landmarks": [{"name": "left_eye_outer", "xy": [212.4, 230.1], "weight": 0.75,
                 "source": "mesh | model | pose | silhouette | manual"}],
  "polylines": [
    {"name": "glasses_left_rim", "closed": true, "weight": 3.0, "source": "manual",
     "xy": [[255, 260], [290, 230], [340, 240], [365, 285], [350, 340], [300, 355], [262, 320]]}
  ],
  "all_landmarks": [[x, y], ...]
}
```

`landmark_set`, `pose` and `polylines` are optional. A file without
`polylines` behaves exactly as before. A Spec 6/7 file loads unchanged.

## 6. Polylines in the loader

`load_landmarks` (`SLDgen/image_loss.py`) reads `polylines` when present:

* Each needs a name, at least 2 vertices (3 when `closed`), finite canvas
  coordinates, and `weight ≥ 0` (default 1).
* It is densified by linear interpolation. Each segment is split into
  `ceil(length / 8 px)` parts. A closed polyline adds its closing segment and
  does not repeat the first vertex. That gives ≈ 1 point per ≤ 8 canvas px.
* Every resulting point gets `weight / point_count`, so a polyline weighs as
  much in total as one landmark of that weight, however long it is.
* The points are appended to the landmark array. Everything downstream sees one
  flat weighted list, and `landmark()` is untouched.
* The "needs a positive weight" check counts polyline points too, so a file of
  polylines alone is valid.

## 7. Optional additions

### 7.1 Glasses

No detector provides eyewear landmarks: MediaPipe returns the eyes *behind*
the lenses. Hard frame geometry is a strong identity anchor, so it is
represented by hand-placed polylines:

* CLI: `--include-glasses FILE` reads canvas-pixel polylines, either
  `{"polylines": [...]}` or a bare list, in the §5 polyline format (`name`,
  `closed`, `weight`, `xy`). They are validated as in §6 and copied into the
  output with `source: "manual"`.
* Web: the editor's **Glasses** button (§8).

### 7.2 Hairline (`--include-hairline`)

Mesh path, frontal and turned faces (|yaw| < 42°):

1. Skin model: mean and spread of CIE-Lab colour in small discs around the
   forehead and cheek points (mesh 151, 108, 337, 50, 280), inside the subject
   mask when there is one.
2. For each upper face-outline point (mesh
   `162 21 54 103 67 109 10 338 297 332 284 251 389`), march outward from
   halfway between the brow and the outline point, along the direction away
   from the face centre, for up to one brow-to-outline distance beyond the
   outline.
3. The hairline is the first place where 3 consecutive pixels are non-skin
   (Lab distance > 3 spreads, at least 12). With `--hair-mask`, it is the
   first pixel inside the hair mask instead. A march that leaves the subject
   mask or runs out finds nothing for that sample (bald crown, hat).
4. At least 5 hits: emit an open polyline `hairline` (weight 1.5,
   `source: "hairline"`), each vertex replaced by the median of itself and its
   neighbours. Fewer hits: no polyline, and a `hairline: not found` line.

Noisy where the hairline is not distinctive, which is why it is optional.

## 8. API and editor

**API** — `POST /api/image-loss/landmarks` gains `landmark_set` (default
`sparse`), `include_hairline` (bool) and `pose_report` (bool). The response
adds `landmark_set`, `pose` and `polylines`. Bad values give 400. Errors are
otherwise unchanged.

**Editor** (Spec 7 §6):
* A set selector beside **Auto-detect** / **Replace all** (sparse, standard,
  dense, pose-locked) and a **hairline** checkbox. Detection always asks for
  the pose report, and the status line shows yaw/pitch/roll.
* Merge rule unchanged for points (`mergeDetected`). Detected polylines follow
  the same rule by name: edited or manual ones are kept, unedited detected
  ones are updated or removed, new ones added.
* **Glasses**: adds `glasses_right_rim` and `glasses_left_rim` (closed,
  12 vertices, weight 3) as ellipses around the eyes (the table's eye points,
  or the canvas centre), and `glasses_bridge` (open, 3 vertices, weight 1).
  Names already present are skipped.
* **Line**: adds an empty open polyline. Polylines are drawn as paths with
  their vertices as small squares. Drag a vertex to move it. With a polyline
  selected, double-click inserts a vertex on its nearest segment (appends,
  for fewer than 2 vertices). Delete removes the selected vertex, or the whole
  polyline when no vertex is selected. A second table under the points lists
  polylines: name, weight, closed, vertex count, delete.
* `serialize`/`parseFile` round-trip `polylines` (lines with too few vertices
  are left out of the file) and `pose`. The placed count and the saved label
  include polylines: `N landmarks + M lines (edited)`.
* Weights: the number input keeps 0–5, and its step becomes `any` so
  normalised weights (e.g. 0.1875) stay valid.

## 9. Tests

* `test_landmarks_geom.py` (CPU, no MediaPipe): set composition (sparse ≡
  portrait table; standard ⊃ sparse; dense = 120 unique indices ⊃ standard,
  deterministic); per-region weight share identical across sets; uniform shift
  → loss d for every set; name-table limits for new names; dense lateral
  limit; the rigid fit recovers a known yaw/pitch/roll and scale from a
  synthetically rotated canonical model; a collapsed far-side point is
  replaced by the projection, a consistent one kept; visibility culls
  back-facing points; silhouette names replace mesh names; hairline march on
  a synthetic skin/hair image, with and without a hair mask; glasses file
  validation.
* **Regression**: `--landmark-set sparse` (and the default) output is
  byte-identical to the pre-change `--preset portrait` output. Checked on 12
  real canvases (frontal, turned, profile via Pose) during the build. A
  permanent fixture (`test_support/landmarks_portrait_firefighter.json`, made
  by the pre-change script) is compared in `test_service_image_loss.py`
  whenever MediaPipe is present.
* `test_image_loss_geom.py`: polyline densification (spacing, closed without a
  duplicate vertex, weight split, validation errors, polyline-only file) and a
  weighted-mean check that a polyline weighs as one landmark.
* `test_service_image_loss.py`: `landmark_set`/`include_hairline`/`pose_report`
  forwarded and validated, and the response keys. With MediaPipe: every set on
  firefighter.
* `sldgen_web/src/lib/landmarks.test.ts`: polyline round-trip, polyline merge,
  the glasses template, vertex insertion on the nearest segment, labels.
* Manual (MediaPipe, CPU): all four sets on the frontal, three-quarter, `Gesa`
  and profile canvases. Point counts, dropped names, pose and overlays go in
  §10.

Build order (each step independently useful and tested): sparse/standard →
dense + normalisation → pose-locked + pose report → polylines (loader,
detector, API) → glasses + hairline → editor.

## 10. As built

(filled in after implementation)
