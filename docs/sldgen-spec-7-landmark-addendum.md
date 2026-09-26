# Addendum: Configurable Landmark Set Generation

Extends `tools/extract_landmarks.py` (spec §7.2) and the `landmark` loss type (§5.3).

Adds **no new loss mathematics**. Everything here is a choice about which points end up in the
JSON, made at preprocessing time. The loss module is unchanged and must stay unchanged.

## Motivation

The current 20-point portrait preset works but under-constrains head pose. Because each landmark
is satisfied independently by its nearest curve point, a drawing whose head is rotated or sheared
relative to the photograph can still score well. A denser, geometrically consistent set makes that
much harder to cheat.

It is not always wanted. For a loose, gestural drawing the sparse set is the better artistic
choice. So this must be selectable, and existing behaviour must remain the default.

## New flag

```
python tools/extract_landmarks.py INPUT.png OUTPUT.json \
    [--landmark-set sparse|standard|dense|pose-locked] \
    [--include-glasses] [--include-hairline] \
    [--pose-report]
```

`--landmark-set` defaults to `sparse`, which must produce byte-identical output to the current
`--preset portrait`. Existing landmark files and run commands keep working unchanged.

## The four sets

| Set | Points | What it constrains |
| --- | --- | --- |
| `sparse` | ~20 | Feature presence only. Current behaviour. Eyes, nose, mouth corners, a few jaw points. |
| `standard` | ~45 | Adds brow arcs, nostril wings, lip contour, more jaw samples. Better feature fidelity, pose still loosely held. |
| `dense` | ~120 | Subsampled from the full MediaPipe 468 mesh, distributed evenly over the face. Pose becomes strongly over-determined. |
| `pose-locked` | ~120 | As `dense`, but positions re-derived through PnP rather than taken raw from the detector. |

## `pose-locked` generation

The substantive addition. Rather than using detector output directly:

1. Run the detector to get 2D landmarks on the source image.
2. `cv2.solvePnP` against a canonical 3D face model, using a stable subset (eye corners, nose
   base, mouth corners, chin) to recover rotation and translation.
3. Project the **full** canonical model under that recovered pose back to 2D.
4. Write those projected positions as the landmark set.

The projected set is internally consistent by construction. Foreshortening, the asymmetry between
near and far eye, and the nose's offset from the face midline all encode head orientation, and
they agree with each other because they came from one rigid transform. A drawing with the wrong
head angle cannot satisfy them simultaneously. With raw detector output each point carries
independent noise and that mutual consistency is weaker.

It also fills in points the detector was uncertain about — useful on three-quarter views where
the far side of the face is partly occluded, which is exactly where pose drift has been worst.

With `--pose-report`, print recovered yaw/pitch/roll in degrees and write them into the JSON under
a `pose` key. Not consumed by the loss; it is a diagnostic, and it tells the author what to write
in the caption.

## Optional additions

- `--include-glasses` — no detector provides eyewear landmarks; MediaPipe returns eye positions
  *behind* the lenses. Add a manual path (click points, or read a coordinate list from a companion
  file) and emit them as a polyline. Hard geometric features like round frames are strong identity
  anchors and are currently unrepresented.
- `--include-hairline` — samples along the hair-to-forehead boundary from the RMBG mask or a
  supplied mask. Useful where the hairline is distinctive, noisy where it isn't. Hence optional.

## Polyline landmarks

Extend the JSON schema with a second array. Backward compatible: a file with no `polylines` key
behaves exactly as now.

```json
{
  "image_size": [1024, 1365],
  "pose": {"yaw": -18.4, "pitch": 6.1, "roll": -3.2},
  "landmarks": [ "... as before ..." ],
  "polylines": [
    {"name": "glasses_left_rim", "closed": true, "weight": 3.0,
     "xy": [[255,560],[290,530],[340,540],[365,585],[350,640],[300,655],[262,620]]}
  ]
}
```

At load time the loss module densifies each polyline by linear or spline interpolation to roughly
one point per 8–10 px at render resolution, assigns each resulting point the polyline's weight
divided by the point count, and appends them to the ordinary landmark array. Densification happens
in the loader; everything downstream sees a flat weighted point list.

## Cost and verification

- Point count drives the `cdist` in the landmark loss, shape (L, 2000). At L=120 this is trivial.
  No performance concern.
- **Weight normalization matters.** The loss divides by the weight sum, so a dense set with default
  weights dilutes each individual point. Renormalize so the *relative* emphasis on eyes vs jaw is
  preserved across sets — otherwise switching from `sparse` to `dense` silently changes the
  effective strength of the whole landmark term. Test explicitly: the same alpha should produce
  comparable pull with both sets.
- **Regression test:** `--landmark-set sparse` output hashes identical to the pre-change
  `--preset portrait` output.

## Build order

1. `sparse` and `standard` — trivial, just different point selections.
2. `pose-locked`.
3. Polylines.
4. `--include-glasses` and `--include-hairline`.

Each is independently useful and independently testable.
