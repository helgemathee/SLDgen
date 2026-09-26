"""Fast isolated tests for view-aware landmark detection (Spec 7 SS4, SS8).

No MediaPipe, no GPU: synthetic meshes, masks and keypoints exercise the yaw
estimate, the culling table, the silhouette profile points and their
consistency check, the box mapping and the Pose view.

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


def main():
    test_triangles_from_edges()
    test_yaw()
    test_culling()
    test_silhouette()
    test_box_and_pose()
    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
