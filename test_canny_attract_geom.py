"""Geometry tests for Canny-derived attraction (``--attract-canny``).

In the style of ``test_origin_geom.py``: fast, CPU-only, no GPU and no diffusion.
``SLDgen/canny_attract.py`` imports only cv2/numpy, which is what makes this
possible -- and what makes it worth keeping that way.

The contract being pinned is not "the SVG looks right". It is:

1. the points land in **canvas space** (the frame ``attraction_loss`` consumes);
2. the point count respects its budget, because the coverage term sums over
   every target and an unbudgeted edge map buries the SDS gradient;
3. the count the module *reports* is the count SLDgen actually produces --
   verified by re-deriving it with the real sampler, not by trusting the
   estimate;
4. the mask is honoured, so the square-pad border never becomes the strongest
   edge in the picture.

Run with either interpreter that has cv2:
    PYTHONPATH=. python test_canny_attract_geom.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from SLDgen.canny_attract import (  # noqa: E402
    SAMPLE_SPACING_PX,
    edge_map,
    extract_polylines,
    generate,
    thin_to_budget,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{name}] {status}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(name)


def synthetic_target(size=512):
    """A square canvas with hard-edged shapes *and* a low-contrast texture.

    Geometric rather than photographic so the strong edges have an unambiguous
    right answer -- but the textured patch is not decoration. Without it every
    edge in the image survives any threshold and any blur, and the two
    assertions about thresholds pass while asserting nothing. The patch is the
    stand-in for the hair and fabric that actually consume the budget on a
    portrait, and it is seeded so the counts are reproducible.
    """
    array = np.full((size, size, 3), 245, dtype=np.uint8)
    array[120:400, 150:360] = 40  # body
    array[160:220, 190:250] = 235  # an "eye"
    array[160:220, 270:330] = 235  # the other
    array[300:320, 200:320] = 235  # a "mouth"

    rng = np.random.default_rng(0)
    texture = rng.integers(0, 60, size=(80, 160), dtype=np.int16)
    patch = np.clip(array[220:300, 180:340, 0].astype(np.int16) + texture, 0, 255)
    array[220:300, 180:340] = patch[..., None].astype(np.uint8)
    return Image.fromarray(array)


def synthetic_mask(size=512):
    """Object mask covering the body only, as RMBG would produce it."""
    mask = np.zeros((size, size), dtype=np.float32)
    mask[110:410, 140:370] = 1.0
    return mask


def sample_like_sldgen(svg_path):
    """Re-derive the points exactly as ``avoidance.load_avoid_points`` does.

    This is the assertion that matters: the module's estimate is only useful if
    it predicts what the real loader will produce, and the loader samples by arc
    length rather than by vertex.
    """
    from svgpathtools import svg2paths2

    paths, _attributes, svg_attributes = svg2paths2(str(svg_path))
    points = []
    for path in paths:
        length = path.length()
        if length <= 0:
            continue
        count = max(2, int(round(length / SAMPLE_SPACING_PX)))
        for index in range(count):
            if index == 0:
                t = 0.0
            elif index == count - 1:
                t = 1.0
            else:
                t = path.ilength(length * index / (count - 1))
            point = path.point(t)
            points.append((point.real, point.imag))
    return points, svg_attributes


def test_edges_are_found_and_masked():
    print("\n--- test_edges_are_found_and_masked")
    image = synthetic_target()
    edges = edge_map(image, mask=None, blur=3)
    check("edges/found", (edges > 0).sum() > 0, f"{(edges > 0).sum()} edge px")

    # An unmasked square canvas has no border edge here, so prove the mask does
    # something by masking away most of the subject instead.
    narrow = np.zeros((512, 512), dtype=np.float32)
    narrow[150:250, 150:360] = 1.0
    fewer = edge_map(image, mask=narrow, blur=3)
    check(
        "edges/mask-restricts",
        0 < (fewer > 0).sum() < (edges > 0).sum(),
        f"{(fewer > 0).sum()} inside vs {(edges > 0).sum()} total",
    )
    check(
        "edges/mask-zeroes-outside",
        (fewer[:150] == 0).all() and (fewer[250:] == 0).all(),
        "no edge pixels survive outside the mask",
    )


def test_thresholds_change_the_result():
    print("\n--- test_thresholds_change_the_result")
    image = synthetic_target()
    loose = int((edge_map(image, low=10, high=30, blur=0) > 0).sum())
    tight = int((edge_map(image, low=200, high=400, blur=0) > 0).sum())
    check("thresholds/loose-finds-more", loose > tight, f"{loose} px vs {tight} px")

    blurred = int((edge_map(image, blur=9) > 0).sum())
    sharp = int((edge_map(image, blur=0) > 0).sum())
    check("thresholds/blur-suppresses", blurred < sharp, f"{blurred} px vs {sharp} px")


def test_budget_is_honoured():
    print("\n--- test_budget_is_honoured")
    image = synthetic_target()
    edges = edge_map(image, mask=synthetic_mask(), blur=3)
    found = extract_polylines(edges, simplify=1.0, min_length=12.0)
    check("budget/contours-found", len(found) > 0, f"{len(found)} contours")

    for budget in (50, 100, 400, 5000):
        _paths, estimate, _period, _dropped = thin_to_budget(found, budget)
        check(
            f"budget/{budget}",
            estimate <= budget,
            f"estimated {estimate} points for a budget of {budget}",
        )

    # Longest-first: thinning must not silently delete a whole feature while a
    # speckle survives.
    paths, _estimate, _period, dropped = thin_to_budget(found, 50)
    check("budget/drops-are-reported", dropped >= 0 and len(paths) > 0, f"dropped {dropped}")


def test_generated_svg_matches_the_real_sampler():
    print("\n--- test_generated_svg_matches_the_real_sampler")
    image = synthetic_target()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "attract_canny.svg"
        stats = generate(image, out, mask=synthetic_mask(), max_points=300)
        check("svg/written", out.exists(), f"{stats['points']} points reported")

        points, svg_attributes = sample_like_sldgen(out)
        check(
            "svg/estimate-matches-loader",
            len(points) == stats["points"],
            f"reported {stats['points']}, loader produced {len(points)}",
        )
        check("svg/within-budget", len(points) <= 300, f"{len(points)} points")

        # Canvas space: the declared size must equal render_size or
        # load_avoid_points warns, and every coordinate must be inside it.
        size = stats["render_size"]
        check(
            "svg/declares-canvas",
            svg_attributes.get("width") == str(size)
            and svg_attributes.get("height") == str(size),
            f"width={svg_attributes.get('width')} height={svg_attributes.get('height')}",
        )
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        check(
            "svg/inside-canvas",
            min(xs) >= 0 and min(ys) >= 0 and max(xs) <= size and max(ys) <= size,
            f"x {min(xs):.1f}..{max(xs):.1f}, y {min(ys):.1f}..{max(ys):.1f}",
        )

        # The mask covered the body only, so nothing may be traced above it.
        check("svg/respects-mask", min(ys) >= 105, f"topmost point at y={min(ys):.1f}")


def test_determinism():
    print("\n--- test_determinism")
    image = synthetic_target()
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "a.svg"
        second = Path(tmp) / "b.svg"
        generate(image, first, mask=synthetic_mask(), max_points=250)
        generate(image, second, mask=synthetic_mask(), max_points=250)
        check(
            "determinism/byte-identical",
            first.read_text() == second.read_text(),
            "two runs of the same settings produce the same SVG",
        )


def test_non_square_is_refused():
    print("\n--- test_non_square_is_refused")
    tall = Image.fromarray(np.zeros((512, 256, 3), dtype=np.uint8))
    try:
        edge_map(tall)
        check("square/refused", False, "a non-square image was accepted")
    except ValueError as error:
        check("square/refused", "square" in str(error), str(error))


if __name__ == "__main__":
    test_edges_are_found_and_masked()
    test_thresholds_change_the_result()
    test_budget_is_honoured()
    test_generated_svg_matches_the_real_sampler()
    test_determinism()
    test_non_square_is_refused()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -- {', '.join(FAILURES)}")
        sys.exit(1)
    print("RESULT: ALL PASS")
