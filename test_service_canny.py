"""Canny-derived attraction, end to end through the service.

Two halves, because the feature has two halves:

**The preview** (``/api/canny/preview``) shells out to ``sld_canny_svg.py`` under
the heavyweight interpreter, exactly as partitioning does, and is driven here
in-process with ``TestClient``.

**The run** generates its own ``attract_canny.svg`` in canvas space. That is
simulated with ``test_support/fake_sldgen.py``, which -- for this feature only --
calls the *real* ``SLDgen/canny_attract.py`` rather than faking its output. A
stub would prove the plumbing moves a file around; running the real generator
proves the file the UI reads is the file SLDgen writes.

Needs an interpreter with fastapi **and** cv2/numpy/PIL, since it plays both the
API and the conda env:

    PYTHONPATH=. SLDGEN_CANNY_PYTHON=<that interpreter> python test_service_canny.py
"""

import json
import os
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

from fastapi.testclient import TestClient

from sldgen_api.app import create_app
from sldgen_service import jobs as job_files
from sldgen_service.config import REPO_ROOT, ServiceConfig
from sldgen_service.params import argv_to_params, build_argv, canonical_params
from sldgen_service.store import Store

RESULTS = []
FAKE_SLDGEN = REPO_ROOT / "test_support" / "fake_sldgen.py"


def check(label, condition, detail=""):
    passed = bool(condition)
    RESULTS.append((label, passed))
    print(f"  [{label}] {'PASS' if passed else 'FAIL'}{(' -- ' + detail) if detail else ''}")
    return passed


def make_png(width, height):
    """A PNG with hard edges in it, so Canny has something to find."""

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            inside = width // 4 < x < 3 * width // 4 and height // 4 < y < 3 * height // 4
            eye = (
                height // 3 < y < height // 3 + height // 10
                and (
                    width // 3 < x < width // 3 + width // 12
                    or 2 * width // 3 < x < 2 * width // 3 + width // 12
                )
            )
            row.append(230 if eye else (40 if inside else 245))
        rows.append(b"\x00" + bytes(row))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def build_root(interpreter):
    root = Path(tempfile.mkdtemp(prefix="sldgen-canny-"))
    config = ServiceConfig(
        root=root,
        sldgen_python=Path(interpreter),
        sldgen_script=FAKE_SLDGEN,
        partition_script=REPO_ROOT / "sld_partition.py",
        default_checkpoint_interval=0,
    ).ensure_layout()
    return config


def run_segment(config, job_id, params, target_epoch):
    """Run one segment synchronously, the way the worker would spawn it."""
    import subprocess

    argv = build_argv(
        config.sldgen_python,
        config.sldgen_script,
        params,
        target=config.job_inputs_dir(job_id) / "target.png",
        output_dir=config.job_dir(job_id),
        stop_at=target_epoch,
        resume=None,
        root=config.root,
    )
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    return completed, argv


def test_params_round_trip():
    print("\n--- parameters survive the CLI round trip")
    params = canonical_params(
        {
            "attract_canny": True,
            "attract_canny_low": 60.0,
            "attract_canny_high": 180.0,
            "attract_canny_blur": 5,
            "attract_canny_simplify": 1.5,
            "attract_canny_min_length": 20.0,
            "attract_canny_max_points": 250,
        }
    )
    from sldgen_service.params import params_to_argv

    round_tripped = argv_to_params(params_to_argv(params))
    check(
        "params/round-trip",
        round_tripped == params,
        next(
            (
                f"{name}: {params[name]!r} -> {round_tripped[name]!r}"
                for name in params
                if params[name] != round_tripped[name]
            ),
            "identical",
        ),
    )
    flags = params_to_argv(params)
    check("params/flag-emitted", "--attract-canny" in flags)
    check(
        "params/values-emitted",
        "--attract-canny-max-points" in flags and "250" in flags,
        " ".join(flags[-6:]),
    )

    off = canonical_params({})
    check("params/off-by-default", off["attract_canny"] is False)
    check("params/flag-absent-when-off", "--attract-canny" not in params_to_argv(off))


def test_validation():
    print("\n--- validation refuses what SLDgen would refuse")
    from sldgen_service.params import ParamError, validate_params

    for label, overrides in (
        ("low-above-high", {"attract_canny_low": 300.0, "attract_canny_high": 200.0}),
        ("max-points-too-small", {"attract_canny_max_points": 1}),
        ("negative-blur", {"attract_canny_blur": -1}),
    ):
        try:
            validate_params({"attract_canny": True, **overrides})
            check(f"validate/{label}", False, "accepted")
        except ParamError as error:
            check(f"validate/{label}", True, str(error))

    # The knobs are inert without the flag, so a nonsense value alone is not an
    # error -- refusing it would block a form that merely has the box unticked.
    try:
        validate_params({"attract_canny": False, "attract_canny_max_points": 1})
        check("validate/inert-when-off", True)
    except ParamError as error:
        check("validate/inert-when-off", False, str(error))


def test_run_generates_the_svg(config):
    print("\n--- the run generates attract_canny.svg in canvas space")
    store = Store(config)
    digest, _ = job_files.store_upload(config, make_png(256, 256))
    params = canonical_params(
        {
            "num_iter": 20,
            "render_size": 512,
            "attract_canny": True,
            "attract_canny_max_points": 200,
            "checkpoint_interval": 0,
        }
    )
    job = job_files.create_job(
        store, config, target_sha256=digest, params=params, target_epoch=20, title="canny"
    )
    store.close()

    completed, argv = run_segment(config, job["id"], params, 20)
    check("run/exit-0", completed.returncode == 0, completed.stderr[-300:])
    check("run/flag-on-command-line", "--attract-canny" in argv)

    svg = config.run_dir(job["id"]) / "attract_canny.svg"
    check("run/svg-written", svg.exists(), str(svg))
    if svg.exists():
        text = svg.read_text()
        check("run/svg-declares-canvas", 'width="512"' in text and 'height="512"' in text)
        check("run/svg-has-paths", text.count("<path") > 0, f"{text.count('<path')} paths")
        check(
            "run/logged",
            "Canny attraction:" in completed.stdout,
            next(
                (line for line in completed.stdout.splitlines() if "Canny" in line),
                completed.stdout[-200:],
            ),
        )

    artifacts = job_files.artifacts(config, job["id"])
    check(
        "run/svg-is-an-artifact",
        any(entry["path"] == "target/run/attract_canny.svg" for entry in artifacts),
        ", ".join(entry["path"] for entry in artifacts[:6]),
    )
    return job["id"]


def test_determinism_across_segments(config):
    print("\n--- two segments of one job generate identical points")
    store = Store(config)
    digest, _ = job_files.store_upload(config, make_png(256, 256))
    params = canonical_params(
        {"num_iter": 40, "attract_canny": True, "attract_canny_max_points": 150}
    )
    job = job_files.create_job(
        store, config, target_sha256=digest, params=params, target_epoch=40, title="segmented"
    )
    store.close()

    run_segment(config, job["id"], params, 20)
    first = (config.run_dir(job["id"]) / "attract_canny.svg").read_text()
    run_segment(config, job["id"], params, 40)
    second = (config.run_dir(job["id"]) / "attract_canny.svg").read_text()
    check(
        "segments/identical",
        first == second,
        "a resumed segment regenerates the same attract points",
    )


def test_preview_endpoint(config, source_job_id):
    print("\n--- the preview endpoint traces a previous run's canvas")
    client = TestClient(create_app(config))

    response = client.post(
        "/api/canny/preview",
        json={"source_job_id": source_job_id, "params": {"max_points": 300}},
    )
    check("preview/succeeds", response.status_code == 200, response.text[:300])
    if response.status_code != 200:
        return
    body = response.json()
    check("preview/reports-points", isinstance(body["points"], int), str(body["points"]))
    check("preview/within-budget", body["points"] <= 300, str(body["points"]))
    check("preview/has-summary", "attract points" in body["summary"], body["summary"])

    svg = client.get(body["svg_url"])
    check("preview/svg-served", svg.status_code == 200 and b"<path" in svg.content)
    check(
        "preview/svg-content-type",
        svg.headers["content-type"].startswith("image/svg+xml"),
        svg.headers["content-type"],
    )
    image = client.get(body["image_url"])
    check("preview/canvas-image-served", image.status_code == 200, str(image.status_code))

    print("\n--- the knobs change the trace")
    tight = client.post(
        "/api/canny/preview",
        json={"source_job_id": source_job_id, "params": {"max_points": 60}},
    ).json()
    check("preview/budget-respected", tight["points"] <= 60, str(tight["points"]))
    check(
        "preview/budget-changes-output",
        tight["points"] < body["points"],
        f"{tight['points']} vs {body['points']}",
    )

    print("\n--- finding the source run by target instead of by job")
    job = client.get(f"/api/jobs/{source_job_id}").json()
    by_target = client.post(
        "/api/canny/preview",
        json={"target_sha256": job["target_sha256"], "params": {"max_points": 120}},
    )
    check("preview/by-target-sha", by_target.status_code == 200, by_target.text[:200])
    if by_target.status_code == 200:
        # Not necessarily *this* job: uploads are content-addressed, so several
        # jobs share one target, and the endpoint documents that it takes the
        # newest usable one. The contract is that whatever it picks is a run of
        # this image that actually reached target preprocessing -- which is what
        # makes its input.png the canvas the next run will reproduce.
        chosen = by_target.json()["source_job_id"]
        chosen_job = client.get(f"/api/jobs/{chosen}").json()
        check(
            "preview/by-target-picks-a-run-of-that-target",
            chosen_job["target_sha256"] == job["target_sha256"],
            f"{chosen} targets {chosen_job['target_sha256'][:12]}",
        )
        check(
            "preview/by-target-picks-a-preprocessed-run",
            (config.run_dir(chosen) / "input.png").exists(),
            f"{chosen} has an input.png",
        )

    missing = client.post("/api/canny/preview", json={"target_sha256": "0" * 64})
    check("preview/unknown-target-404s", missing.status_code == 404, str(missing.status_code))
    check(
        "preview/404-explains",
        "canvas" in missing.json().get("detail", ""),
        missing.json().get("detail", "")[:120],
    )

    empty = client.post("/api/canny/preview", json={})
    check("preview/needs-a-source", empty.status_code == 400, str(empty.status_code))


def main():
    interpreter = os.environ.get("SLDGEN_CANNY_PYTHON", sys.executable)
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as error:
        print(f"SKIP: this test needs cv2/numpy/PIL in the running interpreter ({error})")
        return 0

    config = build_root(interpreter)
    try:
        test_params_round_trip()
        test_validation()
        source_job_id = test_run_generates_the_svg(config)
        test_determinism_across_segments(config)
        test_preview_endpoint(config, source_job_id)
    finally:
        import shutil

        shutil.rmtree(config.root, ignore_errors=True)

    failed = [label for label, passed in RESULTS if not passed]
    print()
    if failed:
        print(f"RESULT: {len(failed)} FAILED -- {', '.join(failed)}")
        return 1
    print(f"RESULT: ALL PASS ({len(RESULTS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
