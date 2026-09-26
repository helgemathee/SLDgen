"""The image fidelity loss, end to end through the service (Spec 6 SS12.4).

* parameters: round trip and validation;
* the two input roles: uploads (PNG, JSON), a file from another job's run
  directory, refusal when set directly, inheritance by Run again;
* the run: ``test_support/fake_sldgen.py`` derives ``image_loss_target.png`` with
  the *real* ``SLDgen/image_loss.py`` code and writes ``image_loss_log.csv``;
  two segments leave one CSV with no duplicate epochs;
* the preview endpoint: 404 without a source run, 200 with one,
  ``derived_equivalent`` true/false, the ``sha256`` usable as an input.

The test process needs fastapi only; the fake and the preview script run under
the conda interpreter (cv2, numpy, torch):

    PYTHONPATH=. SLDGEN_CANNY_PYTHON=$CONDA_PREFIX/bin/python \\
        .venv-service/bin/python test_service_image_loss.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from sldgen_api.app import create_app
from sldgen_service import jobs as job_files
from sldgen_service.config import REPO_ROOT, ServiceConfig
from sldgen_service.params import (
    ParamError,
    argv_to_params,
    build_argv,
    canonical_params,
    params_to_argv,
    validate_params,
)
from sldgen_service.store import Store
from test_service_canny import make_png

RESULTS = []
FAKE_SLDGEN = REPO_ROOT / "test_support" / "fake_sldgen.py"


def check(label, condition, detail=""):
    passed = bool(condition)
    RESULTS.append((label, passed))
    print(f"  [{label}] {'PASS' if passed else 'FAIL'}{(' -- ' + detail) if detail else ''}")
    return passed


def build_root(interpreter):
    root = Path(tempfile.mkdtemp(prefix="sldgen-imageloss-"))
    return ServiceConfig(
        root=root,
        sldgen_python=Path(interpreter),
        sldgen_script=FAKE_SLDGEN,
        partition_script=REPO_ROOT / "sld_partition.py",
        default_checkpoint_interval=0,
    ).ensure_layout()


def run_segment(config, job_id, params, stop_at, resume=None):
    argv = build_argv(
        config.sldgen_python,
        config.sldgen_script,
        params,
        target=config.job_inputs_dir(job_id) / "target.png",
        output_dir=config.job_dir(job_id),
        stop_at=stop_at,
        resume=resume,
        root=config.root,
    )
    return subprocess.run(argv, capture_output=True, text=True, timeout=120), argv


def test_params():
    print("\n--- parameters survive the CLI round trip")
    params = canonical_params(
        {
            "image_loss": True,
            "image_loss_weight": 0.25,
            "image_loss_schedule": "decay",
            "image_loss_schedule_start": 0.6,
            "image_loss_chamfer": 1.0,
            "image_loss_pyramid": 0.5,
            "image_loss_landmark": 2.0,
            "image_loss_target": "jobs/x/inputs/image_loss_target.png",
            "image_loss_canny_low": 60.0,
            "image_loss_canny_high": 150.0,
            "image_loss_canny_blur": 5,
            "image_loss_curve_samples": 1500,
            "image_loss_landmarks": "jobs/x/inputs/image_loss_landmarks.json",
        }
    )
    back = argv_to_params(params_to_argv(params))
    check(
        "params/round-trip",
        back == params,
        next((f"{k}: {params[k]!r} -> {back[k]!r}" for k in params if params[k] != back[k]), ""),
    )
    off = params_to_argv(canonical_params({}))
    check("params/gate-absent-when-off", "--image-loss" not in off)
    check("params/start-absent-when-none", "--image-loss-schedule-start" not in off)
    check("params/paths-absent-when-none", "--image-loss-target" not in off)

    print("\n--- validation refuses what SLDgen would refuse")
    for label, overrides in (
        ("weight-zero", {"image_loss_weight": 0.0}),
        ("weight-above-one", {"image_loss_weight": 1.5}),
        ("bad-schedule", {"image_loss_schedule": "cosine"}),
        ("constant-with-start", {"image_loss_schedule_start": 0.4}),
        ("decay-start-below", {"image_loss_schedule": "decay", "image_loss_weight": 0.7}),
        (
            "ramp-start-above",
            {"image_loss_schedule": "ramp", "image_loss_weight": 0.3,
             "image_loss_schedule_start": 0.5},
        ),
        ("all-terms-zero", {"image_loss_chamfer": 0.0}),
        ("negative-term", {"image_loss_pyramid": -1.0}),
        ("canny-low-above-high", {"image_loss_canny_low": 250.0}),
        ("samples-above-rate", {"image_loss_curve_samples": 6000}),
    ):
        try:
            validate_params({"image_loss": True, **overrides})
            check(f"validate/{label}", False, "accepted")
        except ParamError as error:
            check(f"validate/{label}", True, str(error))
    try:
        validate_params({"image_loss": False, "image_loss_weight": 5.0})
        check("validate/inert-when-off", True)
    except ParamError as error:
        check("validate/inert-when-off", False, str(error))
    try:
        validate_params({"image_loss": True, "image_loss_schedule": "ramp",
                         "image_loss_weight": 0.3})
        check("validate/ramp-default-start-ok", True)
    except ParamError as error:
        check("validate/ramp-default-start-ok", False, str(error))


def read_csv_epochs(path):
    return [int(line.split(",")[0]) for line in path.read_text().splitlines()[1:] if line]


def test_run(config):
    print("\n--- the run derives the edge target and logs every epoch")
    store = Store(config)
    digest, _ = job_files.store_upload(config, make_png(256, 256))
    params = canonical_params(
        {"num_iter": 30, "render_size": 256, "image_loss": True, "checkpoint_interval": 0}
    )
    job = job_files.create_job(store, config, digest, params=params, target_epoch=30, title="il")
    store.close()

    completed, argv = run_segment(config, job["id"], params, 15)
    check("run/segment-1-exit-0", completed.returncode == 0, completed.stderr[-300:])
    check("run/gate-on-command-line", "--image-loss" in argv)
    run_dir = config.run_dir(job["id"])
    check("run/target-written", (run_dir / "image_loss_target.png").exists())
    log = run_dir / "image_loss_log.csv"
    check("run/log-written", log.exists())

    # A segment killed after its checkpoint leaves rows past it behind.
    with open(log, "a") as handle:
        handle.write("16,0.2,1,1,0,0,1,1,1,,\n")
    completed, _ = run_segment(
        config, job["id"], params, 30, resume=run_dir / "checkpoints" / "latest.pt"
    )
    check("run/segment-2-exit-0", completed.returncode == 0, completed.stderr[-300:])
    epochs = read_csv_epochs(log)
    check("run/one-row-per-epoch", epochs == list(range(31)), f"{len(epochs)} rows")

    names = {entry["path"]: entry["kind"] for entry in job_files.artifacts(config, job["id"])}
    check(
        "run/artifacts-listed",
        names.get("target/run/image_loss_target.png") == "image"
        and names.get("target/run/image_loss_log.csv") == "csv",
    )
    return job["id"], digest


def test_roles(config, source_job_id, target_digest):
    print("\n--- the two input roles")
    store = Store(config)
    base = {"num_iter": 10, "render_size": 256, "image_loss": True}

    try:
        job_files.create_job(
            store, config, target_digest,
            params={**base, "image_loss_target": "somewhere.png"},
        )
        check("roles/direct-path-refused", False, "accepted")
    except job_files.JobError as error:
        check("roles/direct-path-refused", True, str(error)[:80])

    try:
        job_files.create_job(store, config, target_digest, params={**base, "image_loss_landmark": 1.0})
        check("roles/landmark-weight-needs-file", False, "accepted")
    except job_files.JobError as error:
        check("roles/landmark-weight-needs-file", True, str(error)[:80])

    png_sha, _ = job_files.store_upload(config, make_png(256, 256))
    landmarks = json.dumps(
        {"space": "canvas", "image_size": [256, 256], "landmarks": []}
    ).encode()
    json_sha, _ = job_files.store_upload(config, landmarks, suffix=".json")
    job = job_files.create_job(
        store, config, target_digest,
        params={**base, "image_loss_landmark": 1.0},
        inputs=[
            {"role": "image_loss_target", "source_kind": "upload", "sha256": png_sha},
            {"role": "image_loss_landmarks", "source_kind": "upload", "sha256": json_sha},
        ],
    )
    params = job["params"]
    check(
        "roles/uploads-copied",
        params["image_loss_target"] == f"jobs/{job['id']}/inputs/image_loss_target.png"
        and params["image_loss_landmarks"] == f"jobs/{job['id']}/inputs/image_loss_landmarks.json"
        and (config.root / params["image_loss_target"]).exists()
        and (config.root / params["image_loss_landmarks"]).read_bytes() == landmarks,
        f"{params['image_loss_target']} / {params['image_loss_landmarks']}",
    )

    from_job = job_files.create_job(
        store, config, target_digest, params=base,
        inputs=[{"role": "image_loss_target", "source_kind": "job",
                 "source_job_id": source_job_id, "path": "image_loss_target.png"}],
    )
    copied = config.root / from_job["params"]["image_loss_target"]
    check(
        "roles/from-another-job",
        copied.read_bytes()
        == (config.run_dir(source_job_id) / "image_loss_target.png").read_bytes(),
    )

    children = job_files.run_again(store, config, job["id"], [{"params": {"image_loss_weight": 0.3}}])
    child = children[0]["params"]
    check(
        "roles/run-again-inherits-both",
        child["image_loss_target"] == f"jobs/{children[0]['id']}/inputs/image_loss_target.png"
        and child["image_loss_landmarks"]
        == f"jobs/{children[0]['id']}/inputs/image_loss_landmarks.json"
        and child["image_loss_weight"] == 0.3
        and (config.root / child["image_loss_landmarks"]).read_bytes() == landmarks,
        f"{child['image_loss_target']} / {child['image_loss_landmarks']}",
    )

    argv = build_argv("py", "s.py", child, "t.png", "out", 10, root=config.root)
    resumed = build_argv("py", "s.py", child, "t.png", "out", 10, resume="c.pt", root=config.root)
    check(
        "roles/resume-keeps-both-paths",
        "--image-loss-target" in resumed and "--image-loss-landmarks" in resumed
        and str(config.root / child["image_loss_target"]) in argv,
    )
    store.close()


def test_preview(config, source_job_id, target_digest):
    print("\n--- the edge-target preview")
    client = TestClient(create_app(config))

    missing = client.post("/api/image-loss/preview", json={"target_sha256": "0" * 64})
    check("preview/no-source-404", missing.status_code == 404, str(missing.status_code))
    check("preview/needs-a-source", client.post("/api/image-loss/preview", json={}).status_code == 400)

    plain = client.post(
        "/api/image-loss/preview",
        json={"target_sha256": target_digest, "params": {"low": 100, "high": 200, "blur": 3}},
    )
    check("preview/succeeds", plain.status_code == 200, plain.text[:300])
    if plain.status_code != 200:
        return
    body = plain.json()
    check("preview/derived-equivalent", body["derived_equivalent"] is True)
    check("preview/edge-pixels", isinstance(body["edge_pixels"], int) and body["edge_pixels"] > 0)
    run_target = config.run_dir(body["source_job_id"]) / "image_loss_target.png"
    served = client.get(body["edge_url"])
    check(
        "preview/equals-what-the-run-derived",
        served.status_code == 200 and served.content == run_target.read_bytes(),
    )
    upload = client.get(f"/api/uploads/{body['sha256']}")
    check("preview/sha256-is-an-upload", upload.status_code == 200 and upload.content == served.content)

    prepared = client.post(
        "/api/image-loss/preview",
        json={"source_job_id": source_job_id,
              "params": {"clahe_clip": 3, "roi": [0, 0, 128, 256], "preserve_silhouette": True}},
    ).json()
    check("preview/prepared-not-equivalent", prepared["derived_equivalent"] is False)
    check("preview/prepared-differs", prepared["sha256"] != body["sha256"])

    store = Store(config)
    job = job_files.create_job(
        store, config, target_digest, params={"image_loss": True, "render_size": 256},
        inputs=[{"role": "image_loss_target", "source_kind": "upload", "sha256": prepared["sha256"]}],
    )
    store.close()
    check("preview/attachable-as-input", job["params"]["image_loss_target"] is not None)

    bad = client.post(
        "/api/image-loss/preview",
        json={"source_job_id": source_job_id, "params": {"low": 300, "high": 100}},
    )
    check("preview/script-error-is-400", bad.status_code == 400, bad.json().get("detail", "")[:100])


def has_mediapipe(interpreter):
    probe = subprocess.run([interpreter, "-c", "import mediapipe"], capture_output=True)
    return probe.returncode == 0


def test_landmarks(config, source_job_id, target_digest):
    print("\n--- landmark extraction")
    client = TestClient(create_app(config))
    missing = client.post("/api/image-loss/landmarks", json={"target_sha256": "0" * 64})
    check("landmarks/no-source-404", missing.status_code == 404, str(missing.status_code))

    # The synthetic target has no face in it: unprocessable either way.
    no_face = client.post("/api/image-loss/landmarks", json={"source_job_id": source_job_id})
    check("landmarks/no-face-or-no-mediapipe-422", no_face.status_code == 422, no_face.text[:160])

    bad_box = client.post(
        "/api/image-loss/landmarks", json={"source_job_id": source_job_id, "box": [10, 10, 5]}
    )
    check("landmarks/bad-box-400", bad_box.status_code == 400, bad_box.text[:120])
    inverted = client.post(
        "/api/image-loss/landmarks", json={"source_job_id": source_job_id, "box": [50, 50, 10, 90]}
    )
    check("landmarks/inverted-box-400", inverted.status_code == 400, inverted.text[:120])
    for label, extra in (
        ("unknown-set", {"landmark_set": "huge"}),
        ("all-with-set", {"preset": "all", "landmark_set": "dense"}),
        ("hairline-not-bool", {"include_hairline": "yes"}),
        ("pose-report-not-bool", {"pose_report": 1}),
    ):
        refused = client.post(
            "/api/image-loss/landmarks", json={"source_job_id": source_job_id, **extra}
        )
        check(f"landmarks/{label}-400", refused.status_code == 400, refused.text[:120])

    canvas = client.get("/api/image-loss/canvas", params={"target_sha256": target_digest})
    body = canvas.json() if canvas.status_code == 200 else {}
    check(
        "canvas/found-by-target",
        canvas.status_code == 200
        and body.get("source_job_id") == source_job_id
        and len(body.get("image_size", [])) == 2
        and body.get("image_url", "").endswith("/target/run/input.png"),
        canvas.text[:160],
    )
    none = client.get("/api/image-loss/canvas", params={"target_sha256": "0" * 64})
    check("canvas/no-run-404", none.status_code == 404, str(none.status_code))

    if not has_mediapipe(str(config.sldgen_python)):
        print("  (MediaPipe missing in the conda interpreter: skipping the face case)")
        return

    store = Store(config)
    face_sha, _ = job_files.store_upload(config, (REPO_ROOT / "data" / "firefighter.png").read_bytes())
    params = canonical_params({"num_iter": 2, "render_size": 512})
    job = job_files.create_job(store, config, face_sha, params=params, target_epoch=2)
    store.close()
    completed, _ = run_segment(config, job["id"], params, 2)
    check("landmarks/face-run-exit-0", completed.returncode == 0, completed.stderr[-200:])

    found = client.post("/api/image-loss/landmarks", json={"target_sha256": face_sha})
    check("landmarks/face-200", found.status_code == 200, found.text[:200])
    if found.status_code != 200:
        return
    body = found.json()
    check(
        "landmarks/portrait-preset",
        body["count"] >= 19 and body["image_size"] == [512, 512]
        and {"left_eye_outer", "mouth_left", "chin"} <= {p["name"] for p in body["landmarks"]},
        f"{body['count']} landmarks",
    )
    check(
        "landmarks/view-reported",
        (body.get("view") or {}).get("method") == "mesh" and isinstance(body.get("dropped"), list),
        str(body.get("view")),
    )
    check("landmarks/mask-forwarded", "--mask" in body["argv"], " ".join(body["argv"][-4:]))
    boxed = client.post(
        "/api/image-loss/landmarks",
        json={"target_sha256": face_sha, "box": [0, 0, 512, 512]},
    )
    check(
        "landmarks/box-forwarded",
        boxed.status_code == 200 and "--box" in boxed.json()["argv"],
        boxed.text[:160],
    )
    check(
        "landmarks/sparse-default-keys",
        body.get("landmark_set") == "sparse" and body.get("pose") is None and body.get("polylines") == []
        and "--landmark-set" not in body["argv"],
    )
    for name in ("standard", "dense", "pose-locked"):
        built = client.post(
            "/api/image-loss/landmarks",
            json={"target_sha256": face_sha, "landmark_set": name, "pose_report": True,
                  "include_hairline": True},
        )
        data = built.json() if built.status_code == 200 else {}
        argv = data.get("argv", [])
        check(
            f"landmarks/{name}-set",
            built.status_code == 200 and data.get("landmark_set") == name
            and data["count"] > body["count"] and "--landmark-set" in argv
            and "--pose-report" in argv and "--include-hairline" in argv,
            built.text[:200] if built.status_code != 200 else f"{data.get('count')} points",
        )
        pose = data.get("pose") or {}
        check(
            f"landmarks/{name}-pose-report",
            pose.get("method") == "rigid-fit" and abs(pose.get("yaw", 99)) < 20
            and isinstance(data.get("polylines"), list),
            str(pose),
        )

    # Spec 7 addendum SS2: the sparse set is byte for byte the pre-addendum
    # --preset portrait file (fixture made by that script on the same image).
    script = REPO_ROOT / "sld_landmarks.py"
    fixture = REPO_ROOT / "test_support" / "landmarks_portrait_firefighter.json"
    out = config.tmp_dir / "sparse-regression.json"
    for flags in ([], ["--landmark-set", "sparse"]):
        subprocess.run(
            [str(config.sldgen_python), str(script), "--image", str(REPO_ROOT / "data" / "firefighter.png"),
             "--out", str(out), *flags],
            capture_output=True, check=False,
        )
        check(
            f"landmarks/sparse-byte-identical {' '.join(flags) or '(default)'}",
            out.exists() and out.read_bytes() == fixture.read_bytes(),
        )
        out.unlink(missing_ok=True)

    stored = client.get(f"/api/uploads/{body['sha256']}")
    payload = stored.json() if stored.status_code == 200 else {}
    check("landmarks/sha256-is-a-canvas-json-upload", payload.get("space") == "canvas")

    store = Store(config)
    job = job_files.create_job(
        store, config, face_sha,
        params={"image_loss": True, "image_loss_landmark": 1.0, "render_size": 512},
        inputs=[{"role": "image_loss_landmarks", "source_kind": "upload", "sha256": body["sha256"]}],
    )
    store.close()
    check("landmarks/attachable-as-input", job["params"]["image_loss_landmarks"].endswith(".json"))


def interpreter_ready(interpreter):
    probe = subprocess.run(
        [interpreter, "-c", "import cv2, numpy, torch, PIL"], capture_output=True, text=True
    )
    return probe.returncode == 0


def main():
    interpreter = os.environ.get("SLDGEN_CANNY_PYTHON", sys.executable)
    if not interpreter_ready(interpreter):
        print(f"SKIP: {interpreter} lacks cv2/numpy/torch/PIL; set SLDGEN_CANNY_PYTHON")
        return 0

    config = build_root(interpreter)
    try:
        test_params()
        source_job_id, digest = test_run(config)
        test_roles(config, source_job_id, digest)
        test_preview(config, source_job_id, digest)
        test_landmarks(config, source_job_id, digest)
    finally:
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
