"""Fast isolated tests for view-aware landmark detection (Spec 7 SS4, SS8).

No MediaPipe, no GPU: synthetic meshes, masks and keypoints exercise the yaw
estimate, the culling table, the silhouette profile points and their
consistency check, the box mapping and the Pose view; and the addendum's
landmark sets, weight budgets, rigid fit, visibility, hairline and glasses.

Run from the repo root with the sldgen interpreter (numpy, scipy, PIL):
    PYTHONPATH=. python test_landmarks_geom.py
"""
import sys

import numpy as np

import sld_landmarks as sl

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


# -- yaw from a synthetic head ------------------------------------------------


def sphere_mesh(n_lat=12, n_lon=24):
    """A closed sphere: vertices, triangles and neighbours, with a known 'front'."""
    vertices = [(0.0, -1.0, 0.0)]
    for i in range(1, n_lat):
        lat = np.pi * i / n_lat - np.pi / 2
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            vertices.append((np.cos(lat) * np.sin(lon), np.sin(lat), -np.cos(lat) * np.cos(lon)))
    vertices.append((0.0, 1.0, 0.0))
    vertices = np.array(vertices)
    edges = set()

    def ring(i):
        return 1 + (i - 1) * n_lon

    for j in range(n_lon):
        edges.add((0, ring(1) + j))
        edges.add((ring(n_lat - 1) + j, len(vertices) - 1))
    for i in range(1, n_lat):
        for j in range(n_lon):
            a = ring(i) + j
            b = ring(i) + (j + 1) % n_lon
            edges.add((a, b))
            if i < n_lat - 1:
                c = ring(i + 1) + j
                d = ring(i + 1) + (j + 1) % n_lon
                edges |= {(a, c), (a, d), (b, d)}  # the quad split along a-d
    triangles, neighbours = sl.mesh_triangles(edges)
    return vertices, triangles, neighbours


def test_yaw():
    print("\n--- yaw from mesh normals")
    vertices, triangles, neighbours = sphere_mesh()
    normals = sl.smoothed_normals(vertices, triangles, neighbours)
    outward = np.sum(normals * vertices, axis=1) > 0.9
    check("normals point outward", outward.mean() > 0.95, f"{outward.mean():.2f}")

    # The front of the sphere faces the camera (-z). Use the vertices nearest to
    # the front meridian as the 'midline' and rotate the whole shell.
    front = np.argsort(np.abs(vertices[:, 0]) + (vertices[:, 2] > 0) * 10)[:10]
    saved = sl.MIDLINE
    sl.MIDLINE = tuple(int(i) for i in front)
    try:
        for degrees in (0.0, -30.0, 45.0):
            # Turning by -theta about y points the front (0, 0, -1) toward
            # (sin, 0, -cos): negative faces image-left, the detector's convention.
            theta = -np.radians(degrees)
            rot = np.array(
                [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
            )
            turned = vertices @ rot.T
            yaw = sl.yaw_from_normals(sl.smoothed_normals(turned, triangles, neighbours))
            check(f"yaw of a shell turned {degrees:+.0f} deg", abs(yaw - degrees) < 8, f"{yaw:+.1f}")
    finally:
        sl.MIDLINE = saved


def test_triangles_from_edges():
    print("\n--- triangles from an edge set")
    edges = {(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)}
    triangles, _ = sl.mesh_triangles(edges)
    check("two triangles in a quad with a diagonal", triangles.tolist() == [[0, 1, 2], [1, 2, 3]])


# -- culling -----------------------------------------------------------------


def test_culling():
    print("\n--- culling hidden-side points")
    check("side of left_eye_outer", sl.split_side("left_eye_outer") == ("left", "eye_outer"))
    check("side of mouth_right", sl.split_side("mouth_right") == ("right", "mouth"))
    check("side of chin", sl.split_side("chin") == ("mid", "chin"))
    check("negative yaw hides the right", sl.far_side(-30) == "right")

    points = np.zeros((478, 3))
    chosen, dropped = sl.select_portrait(points, 0.0)
    check("frontal keeps all 21", len(chosen) == 21 and not dropped, f"{len(chosen)}")

    chosen, dropped = sl.select_portrait(points, -32.0)
    names = {p["name"] for p in chosen}
    check(
        "three-quarter drops far cheek, jaw, eye outer, brow outer",
        set(dropped) == {"cheek_right", "jaw_right", "right_eye_outer", "right_brow_outer"},
        ",".join(dropped),
    )
    check("three-quarter keeps the far pupil", "right_pupil" in names)
    check("three-quarter keeps every near point", all(n in names for n in ("left_eye_outer", "cheek_left")))

    chosen, dropped = sl.select_portrait(points, 50.0)
    names = {p["name"] for p in chosen}
    check("strong turn to the right hides the whole left eye", not any(n.startswith("left_eye") for n in names))
    check("strong turn keeps the midline chin", "chin" in names)
    check("every point is tagged with its source", all(p["source"] == "mesh" for p in chosen))

    check("view kinds", [sl.view_kind(y) for y in (5, -25, 45, None)] == ["frontal", "turned", "profile", "profile"])


# -- silhouette ----------------------------------------------------------------


#: A head facing left, as (y, front_x) knots of its outline; x grows backwards.
PROFILE = [
    (80, 190), (110, 182), (140, 180), (160, 184),  # forehead, brow ridge at 140
    (172, 190),  # nasion
    (200, 150),  # nose tip
    (212, 176),  # subnasale
    (222, 170),  # upper lip
    (230, 180),  # stomion
    (238, 172),  # lower lip
    (252, 184),  # fold
    (268, 176),  # chin front
    (285, 186),  # under the chin
    (290, 230), (330, 232),  # neck
]


def profile_mask(size=400, facing="left"):
    """A filled head whose front follows PROFILE (mirrored for facing right)."""
    ys = np.array([y for y, _ in PROFILE], dtype=float)
    xs = np.array([x for _, x in PROFILE], dtype=float)
    mask = np.zeros((size, size), dtype=bool)
    for y in range(int(ys[0]), int(ys[-1]) + 1):
        front = int(round(np.interp(y, ys, xs)))
        mask[y, front:330] = True
    if facing == "right":
        mask = mask[:, ::-1]
    return mask


EYE, MOUTH = (215.0, 176.0), (195.0, 228.0)


def mirrored(xy, size=400):
    return (size - 1 - xy[0], xy[1])


def test_silhouette():
    print("\n--- profile points from the silhouette")
    mask = profile_mask()
    found = sl.silhouette_points(mask, "left", EYE, MOUTH)
    expected = {
        "nose_tip": 200, "subnasale": 212, "upper_lip": 222, "stomion": 230,
        "lower_lip": 238, "chin_front": 268, "nasion": 172, "brow_ridge": 140,
    }
    for name, row in expected.items():
        got = found.get(name)
        # The brow is a wide rounded plateau; its middle is the answer.
        tolerance = 4 if name == "brow_ridge" else 2
        check(f"{name} at row {row}", got is not None and abs(got[1] - row) <= tolerance, str(got))
    check("the nose tip is on the outline", found["nose_tip"][0] == 150)

    right = sl.silhouette_points(profile_mask(facing="right"), "right", mirrored(EYE), mirrored(MOUTH))
    check(
        "facing right finds the same rows",
        {k: v[1] for k, v in right.items()} == {k: v[1] for k, v in found.items()},
    )

    flat = np.zeros((400, 400), dtype=bool)
    flat[60:340, 170:330] = True
    check("a flat front has no profile points", sl.silhouette_points(flat, "left", EYE, MOUTH) == {})

    specks = mask.copy()
    specks[190:196, 20:26] = True  # a speck far in front of the face
    check(
        "a speck in front of the face is ignored",
        sl.silhouette_points(specks, "left", EYE, MOUTH).get("nose_tip") == found["nose_tip"],
    )

    good = sl.profile_landmarks(mask, "left", EYE, MOUTH, nose_hint=(152.0, 198.0))
    check("consistent nose hint keeps the points", len(good) == len(found) and good[0]["source"] == "silhouette")
    check("weights come from the table", all(p["weight"] == sl.PROFILE_WEIGHTS[p["name"]] for p in good))
    far = sl.profile_landmarks(mask, "left", EYE, MOUTH, nose_hint=(215.0, 200.0))
    check("a nose hint far from the outline rejects them", far == [])


# -- box and pose -----------------------------------------------------------


def test_box_and_pose():
    print("\n--- box mapping and the Pose view")
    check("box grown and clipped", sl.grow_box((10, 20, 110, 220), 512, 512) == (0, 0, 135, 270))
    try:
        sl.grow_box((600, 600, 700, 700), 512, 512)
        check("box outside the canvas refused", False)
    except ValueError:
        check("box outside the canvas refused", True)

    back = sl.region_to_canvas([[100.0, 50.0, 8.0]], 40, 60, 2.0)
    check("region pixels map back to canvas", np.allclose(back, [[90.0, 85.0, 4.0]]), str(back))

    rgb = np.zeros((100, 60, 3), dtype=np.uint8)
    region, left, top, scale = sl._region(rgb, None)
    check("a small canvas is upscaled for detection", scale > 1 and region.shape[0] == 512, f"{scale:.2f}")

    profile = {
        "nose": (160, 212), "left_eye_inner": (181, 187), "left_eye": (188, 188),
        "left_eye_outer": (194, 188), "right_eye_inner": (174, 185), "right_eye": (175, 184),
        "right_eye_outer": (177, 182), "left_ear": (249, 205), "right_ear": (225, 197),
        "mouth_left": (181, 242), "mouth_right": (170, 241),
    }
    facing, near, yaw = sl.pose_view(profile)
    check("pose profile faces left with the left side near", (facing, near) == ("left", "left"))
    check("pose profile reads as strongly turned", yaw < -60, f"{yaw:.0f}")
    chosen, dropped = sl.pose_landmarks(profile, yaw)
    names = {p["name"] for p in chosen}
    check("pose keeps the near eye and mouth corner", {"left_pupil", "mouth_left"} <= names)
    check("pose drops the far eye", "right_pupil" in dropped and "right_pupil" not in names)

    frontal = dict(profile)
    frontal.update(left_eye=(230, 188), right_eye=(170, 188), nose=(200, 212), left_ear=(260, 200), right_ear=(140, 200))
    check("pose frontal reads as frontal", sl.view_kind(sl.pose_view(frontal)[2]) == "frontal")


# -- landmark sets (Spec 7 addendum) --------------------------------------------


def rotation(yaw=0.0, pitch=0.0, roll=0.0):
    return sl.euler_rotation(yaw, pitch, roll)


def posed_mesh(yaw=0.0, pitch=0.0, roll=0.0, scale=10.0, at=(256.0, 256.0, 0.0)):
    """The canonical face (478 points) as a detector would return it, in pixels."""
    vertices, _triangles = sl.canonical_model()
    return scale * vertices @ rotation(yaw, pitch, roll).T + np.array(at)


def test_sets():
    print("\n--- landmark sets")
    sparse = sl.set_entries("sparse")
    check("sparse is the portrait table", sparse == [(n, i) for n, i, _w in sl.PORTRAIT])
    standard = sl.set_entries("standard")
    names = [n for n, _i in standard]
    check("standard extends sparse", standard[: len(sparse)] == sparse and 45 <= len(standard) <= 55, str(len(standard)))
    check("standard names unique", len(set(names)) == len(names))
    check("standard indices unique", len({i for _n, i in standard}) == len(standard))
    sided = [sl.split_side(n) for n in names]
    check(
        "every sided standard part has a turn limit",
        all(part in sl.CULL_LIMIT for side, part in sided if side != "mid"),
        ",".join(sorted({p for s, p in sided if s != "mid" and p not in sl.CULL_LIMIT})),
    )
    dense = sl.set_entries("dense")
    check("dense has DENSE_COUNT points", len(dense) == sl.DENSE_COUNT, str(len(dense)))
    check("dense indices unique", len({i for _n, i in dense}) == len(dense))
    check("dense contains every standard point", set(standard) <= set(dense))
    check("dense is deterministic", dense == sl.set_entries("dense"))
    check("dense unnamed points are m<index>", all(n == f"m{i}" for n, i in dense if (n, i) not in standard))
    check("pose-locked uses the dense layout", sl.set_entries("pose-locked") == dense)
    vertices, _t = sl.canonical_model()
    chosen = vertices[[i for _n, i in dense if i < 468]]
    gaps = [np.sort(np.linalg.norm(chosen - p, axis=1))[1] for p in chosen]
    check("dense points are spread (no near-duplicates)", min(gaps) > 0.3, f"min gap {min(gaps):.2f} cm")


def test_weights():
    print("\n--- anchor-cell weight budgets")
    budgets = {i: w for _n, i, w in sl.PORTRAIT}
    anchor_total = sum(budgets.values())
    eyes = {33, 133, 362, 263, 468, 473}
    for name in ("standard", "dense"):
        entries = sl.set_entries(name)
        weights, cells = sl.anchor_weights(entries)
        weights, cells = np.array(weights), np.array(cells)
        per_anchor = [weights[cells == k].sum() for k in range(len(sl.PORTRAIT))]
        check(
            f"{name}: every anchor's cell sums to its portrait weight",
            np.allclose(per_anchor, [w for _n, _i, w in sl.PORTRAIT], atol=5e-3),
            f"max err {np.max(np.abs(np.array(per_anchor) - [w for _n, _i, w in sl.PORTRAIT])):.4f}",
        )
        surface = weights[cells == len(sl.PORTRAIT)].sum()
        check(f"{name}: surface cell within its budget", surface <= sl.SURFACE_BUDGET + 1e-3, f"{surface:.3f}")
        eye_share = sum(per_anchor[k] for k, (_n, i, _w) in enumerate(sl.PORTRAIT) if i in eyes) / sum(per_anchor)
        sparse_share = sum(budgets[i] for i in eyes) / anchor_total
        check(f"{name}: eyes keep the portrait share of the anchor weight", abs(eye_share - sparse_share) < 1e-3,
              f"{eye_share:.4f} vs {sparse_share:.4f}")
        # The loss is a weighted mean: every landmark d px off costs d, whatever the set.
        d = 7.0
        check(f"{name}: a uniform d px offset costs d", abs((weights * d).sum() / weights.sum() - d) < 1e-9)
        check(f"{name}: every weight positive", weights.min() > 0)


def test_set_culling():
    print("\n--- culling in the new sets")
    mesh = posed_mesh()
    for name in ("standard", "dense"):
        chosen, dropped = sl.select_set(mesh, 0.0, name)
        check(f"{name}: frontal keeps every point", not dropped and len(chosen) == len(sl.set_entries(name)))
    chosen, dropped = sl.select_set(mesh, -32.0, "standard")
    check("standard: the far outline goes at -32", {"right_temple", "right_jaw_low", "right_brow_outer_mid"} <= set(dropped))
    check("standard: the near outline stays", not any(n.startswith("left_") for n in dropped))
    check("standard: the midline stays", not ({"nose_tip", "stomion", "nasion"} & set(dropped)))
    chosen, dropped = sl.select_set(mesh, -32.0, "dense")
    vertices, _t = sl.canonical_model()
    far = [n for n in dropped if n.startswith("m")]
    check("dense: unnamed far points culled at -32", len(far) > 5, str(len(far)))
    check("dense: only image-left (the subject's right) points culled",
          all(vertices[int(n[1:]), 0] < 0 for n in far))
    outer = int(np.argmax(np.abs(vertices[:468, 0])))
    check("dense: lateral rule, outline at 21 degrees", sl.lateral_hidden(outer, 21.0 if vertices[outer, 0] > 0 else -21.0))
    check("dense: lateral rule, midline never", not sl.lateral_hidden(4, -60.0))
    check("dense: near side never", not sl.lateral_hidden(outer, -60.0 if vertices[outer, 0] > 0 else 60.0))


def test_rigid_fit():
    print("\n--- rigid fit, pose report, visibility")
    for angles in ((-25.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 8.0), (-20.0, 6.0, -4.0)):
        fit = sl.fit_rigid(posed_mesh(*angles, scale=11.0), angles[0])
        got = sl.pose_angles(fit["R"])
        check(f"fit recovers yaw/pitch/roll {angles}", np.allclose(got, angles, atol=0.2), str(np.round(got, 2)))
        check(f"fit recovers the scale {angles}", abs(fit["scale"] - 11.0) < 1e-6 and fit["residual_px"] < 1e-6)
    left = sl.pose_angles(rotation(yaw=-30.0))
    forward = rotation(yaw=-30.0) @ np.array([0.0, 0.0, -1.0])
    check("negative yaw faces image-left", forward[0] < 0 and left[0] < 0)
    report = sl.pose_report(sl.fit_rigid(posed_mesh(-20.0, 6.0, -4.0), -20.0))
    check("pose report keys", set(report) == {"yaw", "pitch", "roll", "method", "residual_px"}
          and report["method"] == "rigid-fit" and report["yaw"] == -20.0)

    # Visibility on the canonical surface.
    visible = sl.visible_points(sl.fit_rigid(posed_mesh(), 0.0))
    dense = [i for _n, i in sl.set_entries("dense")]
    check("frontal: every dense point visible", visible[dense].all())
    check("frontal: only the inner lip corners hide (behind the lips)",
          set(np.nonzero(~visible)[0]) <= {78, 80, 191, 308, 310, 415}, str(np.nonzero(~visible)[0]))
    visible = sl.visible_points(sl.fit_rigid(posed_mesh(-40.0), -40.0))
    check("-40: far cheek and far inner eye corner hidden", not visible[234] and not visible[133])
    check("-40: near cheek, midline and near pupil visible", visible[454] and visible[4] and visible[13] and visible[473])

    # A frontal detection that matches the model is kept as detected.
    chosen, dropped, _fit = sl.select_pose_locked(posed_mesh(), 0.0)
    check("pose-locked frontal: all points, all detected", len(chosen) == sl.DENSE_COUNT and not dropped
          and all(p["source"] == "mesh" for p in chosen), f"{len(chosen)} {dropped[:5]}")

    # Three-quarter: the far outer eye corner collapsed onto the nose bridge
    # comes back where the rigid model puts it; an outlier near point too.
    mesh = posed_mesh(-30.0)
    truth = mesh.copy()
    mesh[33] = mesh[168]
    mesh[263, 0] += 40.0
    chosen, dropped, fit = sl.select_pose_locked(mesh, -30.0)
    by = {p["name"]: p for p in chosen}
    check("pose-locked: collapsed far eye corner re-placed by the model",
          by["right_eye_outer"]["source"] == "model" and np.allclose(by["right_eye_outer"]["xy"], truth[33, :2], atol=0.5),
          str(by["right_eye_outer"]))
    check("pose-locked: near outlier re-placed by the model",
          by["left_eye_outer"]["source"] == "model" and np.allclose(by["left_eye_outer"]["xy"], truth[263, :2], atol=0.5))
    check("pose-locked: consistent near point kept as detected", by["left_eye_inner"]["source"] == "mesh")
    check("pose-locked: the fit ignores the corrupted points", fit["residual_px"] < 0.5, f"{fit['residual_px']:.2f}")
    check("pose-locked: far cheek dropped as hidden", "cheek_right" in dropped)

    profile = [sl._point("nose_tip", (1, 2), 3.0, "silhouette")]
    merged = sl.with_silhouette([sl._point("nose_tip", (5, 5), 0.5, "mesh"), sl._point("chin", (9, 9), 0.5, "mesh")], profile)
    check("silhouette names replace mesh names", [p["source"] for p in merged if p["name"] == "nose_tip"] == ["silhouette"]
          and len(merged) == 2)


def hairline_scene(hair_row, size=512):
    """A frontal canonical face on skin, hair above ``hair_row``: (rgb, points)."""
    points = posed_mesh(scale=14.0)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:] = (224, 172, 140)  # skin
    rgb[: int(hair_row)] = (60, 40, 30)  # dark hair
    return rgb, points


def test_hairline():
    print("\n--- hairline and glasses")
    check("Lab of white and black", np.allclose(sl.rgb_to_lab(np.array([255, 255, 255])), [100, 0, 0], atol=0.1)
          and np.allclose(sl.rgb_to_lab(np.array([0, 0, 0])), [0, 0, 0], atol=0.1))
    points = posed_mesh(scale=14.0)
    top = points[10, 1]
    brow = points[list(sl.BROWS), 1].min()
    for label, row in (("above the mesh", top - 20), ("a fringe below the mesh top", top + 0.3 * (brow - top))):
        rgb, points = hairline_scene(row)
        line = sl.hairline_polyline(rgb, points)
        ys = [] if line is None else [y for _x, y in line["xy"]]
        check(f"hairline {label}: found", line is not None and len(ys) >= 5, f"{len(ys)} vertices")
        # A march along a slanted ray lands within a couple of pixels of the edge.
        check(f"hairline {label}: on the edge", line is not None and max(abs(y - row) for y in ys) <= 3.0,
              f"row {row:.1f}, got {np.round(ys, 1).tolist()}")
    rgb, points = hairline_scene(top - 20)
    check("hairline: open polyline, weighted, tagged", (lambda l: l["closed"] is False and l["weight"] == sl.HAIRLINE_WEIGHT
          and l["source"] == "hairline")(sl.hairline_polyline(rgb, points)))
    bald = np.zeros_like(rgb)
    bald[:] = (224, 172, 140)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    mask[int(top - 30):, :] = True
    check("hairline: bald head (march leaves the mask) -> none", sl.hairline_polyline(bald, points, mask=mask) is None)
    hair = np.zeros(rgb.shape[:2], dtype=bool)
    hair[: int(top - 10)] = True
    line = sl.hairline_polyline(bald, points, hair_mask=hair)
    check("hairline: a supplied hair mask replaces the colour test",
          line is not None and max(abs(y - (top - 10)) for _x, y in line["xy"]) <= 3.0)

    import json
    import tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp())
    rim = {"name": "glasses_left_rim", "closed": True, "weight": 3.0, "xy": [[10, 10], [30, 10], [30, 25], [10, 25]]}
    (folder / "g.json").write_text(json.dumps({"polylines": [rim]}))
    (folder / "bare.json").write_text(json.dumps([rim]))
    (folder / "out.json").write_text(json.dumps([{**rim, "xy": [[10, 10], [600, 10], [30, 25]]}]))
    (folder / "bad.json").write_text("{nope")
    lines = sl.load_glasses(folder / "g.json", (512, 512))
    check("glasses file read", len(lines) == 1 and lines[0]["source"] == "manual" and lines[0]["closed"])
    check("glasses bare list read", sl.load_glasses(folder / "bare.json", (512, 512)) == lines)
    for name in ("out.json", "bad.json"):
        try:
            sl.load_glasses(folder / name, (512, 512))
            check(f"glasses {name} refused", False)
        except ValueError:
            check(f"glasses {name} refused", True)


def test_cli():
    print("\n--- CLI flags")
    base = ["--image", "x.png", "--out", "y.json"]
    check("landmark set defaults to sparse", sl.parse_arguments(base).landmark_set == "sparse")
    for bad in (["--preset", "all", "--landmark-set", "dense"], ["--hair-mask", "h.png"], ["--landmark-set", "huge"]):
        try:
            sl.parse_arguments(base + bad)
            check(f"refused: {' '.join(bad)}", False)
        except SystemExit:
            check(f"refused: {' '.join(bad)}", True)


def main():
    test_triangles_from_edges()
    test_yaw()
    test_culling()
    test_silhouette()
    test_box_and_pose()
    test_sets()
    test_weights()
    test_set_culling()
    test_rigid_fit()
    test_hairline()
    test_cli()
    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
