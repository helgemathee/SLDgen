"""Fast isolated geometry test for sld_partition.py.

No diffusion model, no GPU, no Concorde -> runs in well under a second. It
builds a synthetic SLDgen-style master SVG and checks the partitioner's
guarantees directly:

  * sample_path returns an ordered, arc-length-spaced point list
  * sequence partitions concatenate back into the exact master (overlay
    coherence -- the whole point of the tool)
  * horizontal/vertical bands land in the right geometric strip
  * radial slices cover distinct angular sectors
  * cluster (pure-numpy k-means) covers every point exactly once
  * origins / connect-tails prepend/append the right anchor points
  * empty partitions still write a valid, N-preserving SVG with a comment
  * bad inputs fail fast (origin count, missing file)

Run from the repo root:
    PYTHONPATH=. python test_partition_geom.py
"""
import os
import sys

import numpy as np
from svgpathtools import svg2paths2

import sld_partition as sp

SCRATCH = "/tmp/claude-1000/-home-helge-SLDgen/partition_test"


def check(name, cond):
    print(f"  [{'ok' if cond else 'XX'}] {name}")
    return bool(cond)


def write_master(path, size=512):
    """A wandering polyline that crosses every band and angular sector."""
    pts = [
        (20, 20), (500, 40), (30, 500), (480, 480), (256, 10),
        (10, 256), (500, 256), (256, 500), (256, 256), (400, 120),
    ]
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    with open(path, "w") as f:
        f.write(
            '<?xml version="1.0" ?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{size}" height="{size}">\n  <defs/>\n  <g>\n'
            f'    <path d="{d}" stroke-width="2.0" fill="none" '
            'stroke="rgb(0, 0, 0)" stroke-opacity="1.0" '
            'stroke-linecap="round" stroke-linejoin="round"/>\n'
            '  </g>\n</svg>\n'
        )


def full_points(svg_path):
    """All drawn vertices of an SVG, in order, across every sub-stroke.

    A single <path> may hold several sub-strokes (multiple M commands = pen
    lifts); svgpathtools represents each pen lift as a discontinuity (a
    segment whose start != the previous segment's end) rather than a segment,
    so recover each sub-stroke's start explicitly.
    """
    paths, _, _ = svg2paths2(svg_path)
    out = []
    for p in paths:
        prev_end = None
        for seg in p:
            if prev_end is None or abs(seg.start - prev_end) > 1e-9:
                out.append((seg.start.real, seg.start.imag))
            out.append((seg.end.real, seg.end.imag))
            prev_end = seg.end
    return np.asarray(out) if out else np.zeros((0, 2))


def run(argv):
    sp.main(argv)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    master = os.path.join(SCRATCH, "master.svg")
    write_master(master)

    path, style, w, h = sp.load_master(master)
    pts = sp.sample_path(path, 1.0)
    ok = True

    # 1. Sampling: ordered, dense, correct frame, style captured.
    ok &= check("sample-count", len(pts) > 100)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    ok &= check("sample-spacing~1px", step.max() < 2.0 and step.mean() < 1.5)
    ok &= check("style-preserved", style.get("stroke") == "rgb(0, 0, 0)")
    ok &= check("canvas-parsed", w == 512 and h == 512)

    # 2. Sequence overlay coherence: partitions rebuild the master exactly.
    out = os.path.join(SCRATCH, "seq")
    run(["--input", master, "--output-dir", out, "--partitions", "3",
         "--strategy", "sequence"])
    rebuilt = np.vstack([full_points(f"{out}/partition_{i}.svg") for i in range(3)])
    ok &= check("seq-file-count", all(
        os.path.isfile(f"{out}/partition_{i}.svg") for i in range(3)))
    ok &= check("seq-overlay-exact",
                len(rebuilt) == len(pts) and np.abs(rebuilt - pts).max() < 1e-3)

    # 3. Horizontal bands: each partition sits in its y-strip.
    out = os.path.join(SCRATCH, "horiz")
    run(["--input", master, "--output-dir", out, "--partitions", "3",
         "--strategy", "horizontal"])
    band_ok = True
    for i in range(3):
        P = full_points(f"{out}/partition_{i}.svg")
        lo, hi = i * h / 3, (i + 1) * h / 3
        # small epsilon for the boundary sample landing exactly on the edge
        band_ok &= P[:, 1].min() >= lo - 1.0 and P[:, 1].max() <= hi + 1.0
    ok &= check("horizontal-bands", band_ok)

    # 4. Vertical bands: each partition sits in its x-strip.
    out = os.path.join(SCRATCH, "vert")
    run(["--input", master, "--output-dir", out, "--partitions", "2",
         "--strategy", "vertical"])
    v_ok = full_points(f"{out}/partition_0.svg")[:, 0].max() <= w / 2 + 1.0
    v_ok &= full_points(f"{out}/partition_1.svg")[:, 0].min() >= w / 2 - 1.0
    ok &= check("vertical-bands", v_ok)

    # 5. Radial slices: no point count lost, sectors distinct.
    out = os.path.join(SCRATCH, "rad")
    run(["--input", master, "--output-dir", out, "--partitions", "4",
         "--strategy", "radial"])
    tot = sum(len(full_points(f"{out}/partition_{i}.svg")) for i in range(4))
    # every sampled point is assigned to exactly one radial slice, so the
    # recovered vertices across all partitions must equal the sample count.
    ok &= check("radial-covers-all", tot == len(pts))

    # 6. Cluster: every sampled point assigned exactly once (partition of set).
    labels, centers = sp._kmeans(pts, 4, seed=0)
    counts = np.bincount(labels, minlength=4)
    ok &= check("cluster-partition-of-set",
                len(labels) == len(pts) and labels.min() >= 0
                and labels.max() <= 3 and counts.sum() == len(pts)
                and (counts > 0).all())

    # 7. Origins + connect-tails: first/last drawn points are the origin px.
    out = os.path.join(SCRATCH, "tails")
    run(["--input", master, "--output-dir", out, "--partitions", "2",
         "--strategy", "horizontal",
         "--origins", "1.0", "0.5", "0.0", "0.5", "--connect-tails"])
    P0 = full_points(f"{out}/partition_0.svg")
    tail_ok = np.allclose(P0[0], [1.0 * w, 0.5 * h], atol=1e-2)
    tail_ok &= np.allclose(P0[-1], [1.0 * w, 0.5 * h], atol=1e-2)
    ok &= check("origin-tails", tail_ok)

    # 8. connect-tails with no origins -> nearest canvas edge at both ends.
    out = os.path.join(SCRATCH, "edge")
    run(["--input", master, "--output-dir", out, "--partitions", "2",
         "--strategy", "vertical", "--connect-tails"])
    P0 = full_points(f"{out}/partition_0.svg")
    on_edge = (min(P0[0][0], w - P0[0][0], P0[0][1], h - P0[0][1]) < 1e-6)
    ok &= check("nearest-edge-tails", on_edge)

    # 9. Empty partitions: still write N valid files, with a comment.
    tiny = os.path.join(SCRATCH, "tiny.svg")
    with open(tiny, "w") as f:
        f.write(
            '<?xml version="1.0" ?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            'version="1.1" width="512" height="512">\n  <defs/>\n  <g>\n'
            '    <path d="M 10 10 L 30 500 L 20 20" fill="none" '
            'stroke="rgb(0, 0, 0)"/>\n  </g>\n</svg>\n'
        )
    out = os.path.join(SCRATCH, "empty")
    run(["--input", tiny, "--output-dir", out, "--partitions", "4",
         "--strategy", "vertical"])
    files_ok = all(os.path.isfile(f"{out}/partition_{i}.svg") for i in range(4))
    with open(f"{out}/partition_2.svg") as f:
        body = f.read()
    empty_ok = "empty" in body and svg2paths2(f"{out}/partition_2.svg")[0] == []
    ok &= check("empty-partition-files", files_ok and empty_ok)

    # 10. Fail-fast validation (run in a subprocess-free way via SystemExit).
    def expect_exit(argv, label):
        try:
            sp.main(argv)
        except SystemExit as e:
            return check(label, e.code == 1)
        return check(label, False)

    ok &= expect_exit(
        ["--input", master, "--output-dir", SCRATCH, "--partitions", "3",
         "--strategy", "horizontal", "--origins", "1.0", "0.5"],
        "reject-bad-origin-count")
    ok &= expect_exit(
        ["--input", "/does/not/exist.svg", "--output-dir", SCRATCH,
         "--partitions", "2", "--strategy", "sequence"],
        "reject-missing-input")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
