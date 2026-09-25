"""The API surface Spec 3 added, against a live API and a live worker.

Everything Spec 2 already covered is in ``test_service_e2e.py``; this file tests
only what the web UI needed and what the UI depends on being true:

  * static serving of ``sldgen_web/dist`` at ``/``, and a useful answer when it
    has not been built
  * ``/api/params/last`` and the presets, including that they survive an API
    restart -- the whole point of holding them server-side (Spec 3 SS9)
  * ``/api/jobs/{id}/frames``, including the coordinate-space flag the contact
    sheet must warn about (SS6.2)
  * ``/api/jobs/{id}/lineage`` and the list filters the variant table needs
  * the marks a job carries between browsers: the frame it is parked on, the
    starred frames, and the archive of just those (SS6.2)
  * ``/api/events``, the rail's global stream
  * the constraint sources the picker offers: a frame of another job, and an
    uploaded SVG, both copied into the new job rather than referenced (SS8.3)
  * ``/api/maintenance/cleanup``, whose dry run must report exactly what the
    real run then does (SS11), including ``delete_jobs`` -- the rail's ticked
    selection
  * that ``sldgen_web/src/lib/params.ts`` still agrees with
    ``sldgen_service/params.py`` -- the one duplication in the whole design

It reuses the e2e harness, so it boots the same shipping daemons with the same
stub SLDgen and needs no GPU.

Run it with the service venv:
    PYTHONPATH=. .venv-service/bin/python test_service_web.py
"""

import ast
import io
import json
import re
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import httpx

from sldgen_service.params import PARAM_SPECS
from test_service_e2e import PNG, Harness, RESULTS, check, wait_for

REPO_ROOT = Path(__file__).resolve().parent
PARAMS_TS = REPO_ROOT / "sldgen_web" / "src" / "lib" / "params.ts"
WEB_DIST = REPO_ROOT / "sldgen_web" / "dist"


# -- the parameter schema mirror -------------------------------------------


def parse_params_ts(text):
    """Pull {name: default} out of the TypeScript PARAM_SPECS literal.

    A parser rather than a JSON load because the file is source, not data. It
    only has to understand the four value kinds a default can be.
    """
    body = text.split("export const PARAM_SPECS", 1)[1]
    body = body.split("\n]", 1)[0]
    found = {}
    for match in re.finditer(r"\{\s*name:\s*'([a-z_0-9]+)',(.*?)\}", body, re.S):
        name, rest = match.group(1), match.group(2)
        default = re.search(r"default:\s*(.+?),\s*label:", rest, re.S)
        if default:
            found[name] = _js_literal(default.group(1).strip())
    return found


def _js_literal(token):
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    return ast.literal_eval(token)


def test_param_schema_mirror():
    print("\n--- the client parameter schema mirrors the server's")
    if not PARAMS_TS.exists():
        check("schema/params.ts-present", False, str(PARAMS_TS))
        return

    client = parse_params_ts(PARAMS_TS.read_text())
    server = {spec.name: spec.default for spec in PARAM_SPECS}

    check("schema/same-names", set(client) == set(server),
          f"only in client: {sorted(set(client) - set(server))}; "
          f"only in server: {sorted(set(server) - set(client))}")

    mismatched = [
        f"{name}: client {client[name]!r} != server {server[name]!r}"
        for name in sorted(set(client) & set(server))
        if not _same_default(client[name], server[name])
    ]
    check("schema/same-defaults", not mismatched, "; ".join(mismatched))

    # The one place the service deliberately differs from SLDgen's own default.
    check("schema/checkpoint-interval-is-the-service-default",
          client.get("checkpoint_interval") == 200)


def _same_default(client, server):
    if isinstance(client, (int, float)) and isinstance(server, (int, float)):
        return float(client) == float(server)
    return client == server


# -- static serving ---------------------------------------------------------


def test_serves_the_web_ui(harness):
    print("\n--- static serving of the built UI")
    response = harness.client.get("/")
    check("web/root-answers", response.status_code == 200, str(response.status_code))

    if (WEB_DIST / "index.html").exists():
        check("web/serves-index-html", "<div id=\"root\">" in response.text)
        check("web/content-type-is-html", "text/html" in response.headers["content-type"])
        assets = list((WEB_DIST / "assets").glob("*.js"))
        if assets:
            asset = harness.client.get(f"/assets/{assets[0].name}")
            check("web/serves-hashed-assets", asset.status_code == 200)
    else:
        # A fresh checkout has no build; the answer must name the command rather
        # than being a bare 404 that sends you hunting for a server fault.
        check("web/explains-how-to-build", "npm run build" in response.text)

    # The mount owns everything that is not /api, but must not shadow it.
    check("web/api-still-wins", harness.client.get("/api/health").json()["ok"] is True)
    check("web/unknown-api-path-is-404",
          harness.client.get("/api/nope").status_code == 404)


# -- parameter persistence --------------------------------------------------


def test_params_last_and_presets(harness):
    print("\n--- parameter persistence and presets (SS9)")
    check("params/last-starts-null", harness.client.get("/api/params/last").json()["params"] is None)

    # The shape is the UI's, and the service must not interpret it: an origin
    # switched off still carries its coordinates.
    payload = {
        "params": {"seed": 7, "caption": "a racing car"},
        "optional": {"origin": {"enabled": False, "value": [0.31, 0.78]}},
        "maskMode": "guide",
        "numIter": 4000,
        "targetEpoch": 400,
    }
    put = harness.client.put("/api/params/last", json={"params": payload})
    check("params/put-accepted", put.status_code == 200)

    stored = harness.client.get("/api/params/last").json()["params"]
    check("params/round-trips-verbatim", stored == payload)
    check("params/keeps-a-disabled-origin-value",
          stored["optional"]["origin"] == {"enabled": False, "value": [0.31, 0.78]})

    # Server-side is the point: it must outlive the process, not just the tab.
    harness.restart_api()
    after = harness.client.get("/api/params/last").json()["params"]
    check("params/survives-an-api-restart", after == payload)

    check("presets/start-empty", harness.client.get("/api/params/presets").json()["presets"] == [])
    created = harness.client.post(
        "/api/params/presets", json={"name": "quick look", "params": payload}
    )
    check("presets/created", created.status_code == 201 and created.json()["name"] == "quick look")

    listed = harness.client.get("/api/params/presets").json()["presets"]
    check("presets/listed", len(listed) == 1 and listed[0]["params"] == payload)

    # Saving over a name replaces it: two presets you cannot tell apart in a
    # dropdown are worse than one that is out of date.
    harness.client.post("/api/params/presets", json={"name": "quick look", "params": {"seed": 9}})
    listed = harness.client.get("/api/params/presets").json()["presets"]
    check("presets/same-name-replaces", len(listed) == 1 and listed[0]["params"] == {"seed": 9})

    check("presets/rejects-an-empty-name",
          harness.client.post("/api/params/presets", json={"name": "  "}).status_code == 400)

    deleted = harness.client.delete(f"/api/params/presets/{listed[0]['id']}")
    check("presets/deleted", deleted.status_code == 204)
    check("presets/gone", harness.client.get("/api/params/presets").json()["presets"] == [])
    check("presets/delete-unknown-is-404",
          harness.client.delete("/api/params/presets/nope").status_code == 404)


# -- frames -----------------------------------------------------------------


def test_frames(harness, sha256):
    print("\n--- the contact sheet's frame list (SS6.2)")
    job = harness.create_job(sha256, target_epoch=200, num_iter=1000, title="frames")
    job_id = job["id"]
    harness.await_state(job_id, "waiting")

    frames = harness.client.get(f"/api/jobs/{job_id}/frames").json()
    epochs = [frame["epoch"] for frame in frames["frames"]]
    check("frames/lists-saved-frames", len(epochs) > 0, str(epochs))
    check("frames/epochs-ascend", epochs == sorted(epochs))
    check("frames/reports-save-interval", frames["save_interval"] == 50)
    check("frames/granularity-matches-save-interval",
          all(epoch % 50 == 0 for epoch in epochs), str(epochs))

    first = frames["frames"][0]
    png = harness.client.get(first["png_url"])
    check("frames/png-url-resolves", png.status_code == 200)
    check("frames/pairs-the-svg", first["svg_url"] is not None)
    if first["svg_url"]:
        check("frames/svg-url-resolves",
              harness.client.get(first["svg_url"]).status_code == 200)

    check("frames/not-rescaled-by-default", frames["rescaled"] is False)
    check("frames/no-final-svg-before-the-horizon", frames["final_svg_url"] is None)

    # `rescaled` means "svg_logs/ is in a different coordinate space than
    # final_sld.svg", and it is answered by comparing the canvases the two files
    # declare -- not by looking for scale_w in config.json, which is set on
    # nearly every real run and does not reach either file (see
    # jobs.coordinate_space_mismatch). A job with only the flag set is not
    # rescaled; a job whose files actually disagree is.
    run_dir = harness.root / "jobs" / job_id / "target" / "run"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config["scale_w"], config["scale_h"] = 0.864, 0.867
    config_path.write_text(json.dumps(config))
    check("frames/object-size-ratio-alone-is-not-a-rescale",
          harness.client.get(f"/api/jobs/{job_id}/frames").json()["rescaled"] is False)

    svg_header = '<?xml version="1.0" ?>\n<svg xmlns="http://www.w3.org/2000/svg" '
    (run_dir / "final_sld.svg").write_text(svg_header + 'width="512" height="512"><g/></svg>')
    newest = max(epochs)
    (run_dir / "svg_logs" / f"svg_iter{newest}.svg").write_text(
        svg_header + 'width="1024" height="1024"><g/></svg>'
    )
    rescaled = harness.client.get(f"/api/jobs/{job_id}/frames").json()
    check("frames/flags-a-genuine-canvas-mismatch", rescaled["rescaled"] is True)

    # And the API refuses such an input outright, so the UI's greying-out is a
    # convenience rather than the only guard.
    refused = harness.client.post(
        "/api/jobs",
        json={
            "target_sha256": sha256,
            "target_epoch": 100,
            "params": {"num_iter": 1000},
            "inputs": [
                {
                    "role": "avoid",
                    "source_kind": "job",
                    "source_job_id": job_id,
                    "path": f"svg_logs/svg_iter{newest}.svg",
                }
            ],
        },
    )
    check("frames/api-refuses-an-intermediate-on-a-different-canvas",
          refused.status_code == 400 and "coordinate space" in refused.text,
          refused.text[:160])

    check("frames/empty-for-a-job-that-has-not-run",
          harness.client.get(
              f"/api/jobs/{harness.create_job(sha256, title='unrun')['id']}/frames"
          ).json()["frames"] == [])
    return job_id


# -- marks: the viewed frame and the favourites -----------------------------


def test_marks(harness, sha256):
    """What the UI remembers about a job on your behalf (SS6.2).

    The whole point of keeping these on the server is that they survive the
    browser, so every check here is made through the API rather than against
    anything the client holds.
    """
    print("\n--- marks: the parked frame, the stars, and the archive of them")
    job = harness.create_job(sha256, target_epoch=200, num_iter=1000, title="marks")
    job_id = job["id"]
    check("marks/a-new-job-is-not-parked", job["viewed_epoch"] is None)
    check("marks/a-new-job-has-no-stars", job["favorite_count"] == 0)

    # Before anything has run there is still a picture: the submitted image.
    # A queue of white squares says nothing about what is in the queue.
    submitted = harness.client.get(f"/api/jobs/{job_id}/preview")
    check("marks/queued-job-previews-its-input", submitted.status_code == 200,
          str(submitted.status_code))
    check("marks/queued-preview-is-the-uploaded-target", submitted.content == PNG)

    harness.await_state(job_id, "waiting")
    frames = harness.client.get(f"/api/jobs/{job_id}/frames").json()["frames"]
    epochs = [frame["epoch"] for frame in frames]
    check("marks/have-frames-to-mark", len(epochs) >= 3, str(epochs))
    parked, other = epochs[1], epochs[2]

    newest = harness.client.get(f"/api/jobs/{job_id}/preview").content

    # -- the parked frame ---------------------------------------------------
    response = harness.client.put(f"/api/jobs/{job_id}/viewed-epoch", json={"epoch": parked})
    check("marks/park-accepted", response.status_code == 200, response.text[:120])
    check("marks/park-echoes-the-epoch", response.json()["viewed_epoch"] == parked)
    check("marks/park-shows-in-the-summary",
          harness.client.get("/api/jobs").json()["jobs"][0]["viewed_epoch"] == parked)

    frame_bytes = harness.client.get(
        f"/api/jobs/{job_id}/files/target/run/svg_to_png/iter_{parked:04d}.png"
    ).content
    parked_preview = harness.client.get(f"/api/jobs/{job_id}/preview")
    check("marks/thumbnail-follows-the-parked-frame",
          parked_preview.content == frame_bytes and frame_bytes != newest)

    harness.client.put(f"/api/jobs/{job_id}/viewed-epoch", json={"epoch": None})
    check("marks/unpark-clears-it",
          harness.job(job_id)["viewed_epoch"] is None)
    check("marks/unparked-thumbnail-is-newest-again",
          harness.client.get(f"/api/jobs/{job_id}/preview").content == newest)
    check("marks/park-rejects-nonsense",
          harness.client.put(
              f"/api/jobs/{job_id}/viewed-epoch", json={"epoch": "soon"}
          ).status_code == 400)

    # -- the stars ----------------------------------------------------------
    starred = harness.client.put(f"/api/jobs/{job_id}/favorites/{parked}")
    check("marks/star-accepted", starred.status_code == 200, starred.text[:120])
    harness.client.put(f"/api/jobs/{job_id}/favorites/{other}")
    again = harness.client.put(f"/api/jobs/{job_id}/favorites/{parked}")
    listed = [favorite["epoch"] for favorite in again.json()["favorites"]]
    check("marks/starring-twice-marks-once", listed == sorted({parked, other}), str(listed))
    check("marks/stars-carry-their-svg",
          all(favorite["svg_url"] for favorite in again.json()["favorites"]))
    check("marks/star-count-in-the-summary",
          harness.job(job_id)["favorite_count"] == 2)
    check("marks/star-epochs-in-the-detail",
          harness.job(job_id)["favorite_epochs"] == sorted({parked, other}))
    check("marks/cannot-star-a-frame-that-does-not-exist",
          harness.client.put(f"/api/jobs/{job_id}/favorites/999999").status_code == 404)

    archive = harness.client.get(f"/api/jobs/{job_id}/favorites.zip")
    check("marks/archive-served", archive.status_code == 200, str(archive.status_code))
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = sorted(bundle.namelist())
        check("marks/archive-holds-only-the-starred-svgs",
              names == sorted(f"{job_id}/epoch_{epoch:04d}.svg" for epoch in (parked, other)),
              str(names))
        source = (harness.root / "jobs" / job_id / "target" / "run"
                  / "svg_logs" / f"svg_iter{parked}.svg").read_bytes()
        check("marks/archived-svg-is-the-frame-itself",
              bundle.read(f"{job_id}/epoch_{parked:04d}.svg") == source)

    harness.client.delete(f"/api/jobs/{job_id}/favorites/{other}")
    check("marks/unstar-removes-one", harness.job(job_id)["favorite_epochs"] == [parked])
    check("marks/unstarring-twice-is-not-an-error",
          harness.client.delete(f"/api/jobs/{job_id}/favorites/{other}").status_code == 200)

    # -- marks belong to the job -------------------------------------------
    harness.client.put(f"/api/jobs/{job_id}/viewed-epoch", json={"epoch": parked})
    harness.client.delete(f"/api/jobs/{job_id}")
    wait_for(lambda: harness.client.get(f"/api/jobs/{job_id}").status_code == 404,
             timeout=30, what="the job to be deleted")
    connection = sqlite3.connect(harness.root / "sldgen.sqlite")
    try:
        left = connection.execute(
            "SELECT (SELECT COUNT(*) FROM job_views WHERE job_id = ?)"
            "     + (SELECT COUNT(*) FROM job_favorites WHERE job_id = ?)",
            (job_id, job_id),
        ).fetchone()[0]
    finally:
        connection.close()
    check("marks/deleting-a-job-takes-its-marks-with-it", left == 0, str(left))


# -- lineage and list filters ----------------------------------------------


def test_lineage_and_filters(harness, sha256, parent_id):
    print("\n--- lineage, batches and the list filters the variant table needs")
    harness.set_paused(True)
    try:
        created = harness.client.post(
            f"/api/jobs/{parent_id}/run-again",
            json={
                "variants": [
                    {"params": {"seed": 101}, "title": "v · s101"},
                    {"params": {"seed": 102}, "title": "v · s102"},
                    {"params": {"seed": 103}, "title": "v · s103"},
                ]
            },
        )
        check("lineage/run-again-created-three", created.status_code == 201)
        variants = created.json()["jobs"]
        batch_id = variants[0]["batch_id"]

        check("lineage/one-batch-id-for-the-whole-submission",
              batch_id and all(job["batch_id"] == batch_id for job in variants))
        check("lineage/every-variant-points-at-the-parent",
              all(job["parent_job_id"] == parent_id for job in variants))
        check("lineage/variants-are-queued-in-submission-order",
              [job["id"] for job in variants] == sorted(job["id"] for job in variants))

        lineage = harness.client.get(f"/api/jobs/{parent_id}/lineage").json()
        check("lineage/parent-sees-its-variants", len(lineage["variants"]) == 3)
        check("lineage/parent-has-no-parent", lineage["parent"] is None)

        child = harness.client.get(f"/api/jobs/{variants[0]['id']}/lineage").json()
        check("lineage/child-names-its-parent", child["parent"]["id"] == parent_id)
        check("lineage/child-sees-its-batch-siblings", len(child["batch_siblings"]) == 2)
        check("lineage/siblings-exclude-self",
              all(job["id"] != variants[0]["id"] for job in child["batch_siblings"]))

        by_parent = harness.client.get(f"/api/jobs?parent_job_id={parent_id}").json()["jobs"]
        check("filters/by-parent-job-id", len(by_parent) == 3)

        by_batch = harness.client.get(f"/api/jobs?batch_id={batch_id}").json()["jobs"]
        check("filters/by-batch-id", len(by_batch) == 3)

        # The variant table needs parameters to flag a duplicate seed, and the
        # rail does not -- so it is opt-in.
        plain = harness.client.get("/api/jobs?limit=5").json()["jobs"]
        check("filters/params-omitted-by-default", all("params" not in job for job in plain))
        with_params = harness.client.get("/api/jobs?limit=5&with_params=true").json()["jobs"]
        check("filters/with_params-includes-them",
              all("params" in job for job in with_params)
              and with_params[0]["params"]["num_iter"] > 0)

        seeds = {
            job["params"]["seed"]
            for job in harness.client.get(
                f"/api/jobs?batch_id={batch_id}&with_params=true"
            ).json()["jobs"]
        }
        check("filters/variant-seeds-are-visible-before-they-run", seeds == {101, 102, 103})

        for job in variants:
            harness.client.delete(f"/api/jobs/{job['id']}")
    finally:
        harness.set_paused(False)


# -- the global stream ------------------------------------------------------


def test_global_events(harness, sha256):
    print("\n--- the rail's global stream")
    harness.set_paused(True)
    try:
        job = harness.create_job(sha256, target_epoch=100, num_iter=1000, title="stream")
        payload = None
        with harness.client.stream("GET", "/api/events", timeout=20.0) as response:
            check("events/content-type-is-sse",
                  "text/event-stream" in response.headers["content-type"])
            event_name = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name == "jobs":
                    payload = json.loads(line.split(":", 1)[1])
                    break

        check("events/sends-the-whole-list", payload is not None and "jobs" in payload)
        check("events/includes-the-new-job",
              any(entry["id"] == job["id"] for entry in payload["jobs"]))
        check("events/reports-queue-depth", payload["queue_depth"] >= 1, str(payload["queue_depth"]))
        check("events/reports-worker-liveness", payload["worker_alive"] is True)

        harness.client.delete(f"/api/jobs/{job['id']}")
    finally:
        harness.set_paused(False)


# -- cleanup ----------------------------------------------------------------


def test_cleanup(harness, sha256):
    print("\n--- cleanup: the dry run must be exactly what the real run does (SS11)")
    check("cleanup/rejects-an-unknown-action",
          harness.client.post(
              "/api/maintenance/cleanup", json={"action": "rm -rf", "dry_run": True}
          ).status_code == 400)

    # A job that fails, so delete_failed has something to find.
    failing = harness.create_job(sha256, target_epoch=100, num_iter=1000, title="doomed")
    harness.write_fault(failing["id"], {"exit_code": 4})
    harness.await_state(failing["id"], "failed")

    dry = harness.client.post(
        "/api/maintenance/cleanup", json={"action": "delete_failed", "dry_run": True}
    ).json()
    check("cleanup/dry-run-finds-the-failed-job",
          any(item["id"] == failing["id"] for item in dry["items"]))
    check("cleanup/dry-run-reports-bytes", dry["bytes"] > 0, str(dry["bytes"]))
    check("cleanup/dry-run-is-marked-as-one", dry["dry_run"] is True)
    check("cleanup/dry-run-did-not-act",
          harness.client.get(f"/api/jobs/{failing['id']}").status_code == 200)

    real = harness.client.post(
        "/api/maintenance/cleanup", json={"action": "delete_failed", "dry_run": False}
    ).json()
    check("cleanup/real-run-matches-the-dry-run-count", real["job_count"] == dry["job_count"])
    check("cleanup/real-run-matches-the-dry-run-bytes", real["bytes"] == dry["bytes"])
    check("cleanup/the-job-is-gone",
          wait_for(
              lambda: harness.client.get(f"/api/jobs/{failing['id']}").status_code == 404,
              timeout=20, what="the failed job to be reclaimed",
          ))

    # Orphaned uploads: one nobody references.
    orphan = harness.client.post(
        "/api/uploads", files={"file": ("orphan.png", PNG[:-1] + b"\x01", "image/png")}
    ).json()
    dry = harness.client.post(
        "/api/maintenance/cleanup", json={"action": "delete_orphan_uploads", "dry_run": True}
    ).json()
    check("cleanup/finds-the-orphan-upload",
          any(item["id"].startswith(orphan["sha256"]) for item in dry["items"]))
    check("cleanup/spares-the-upload-a-job-uses",
          not any(item["id"].startswith(sha256) for item in dry["items"]))

    harness.client.post(
        "/api/maintenance/cleanup", json={"action": "delete_orphan_uploads", "dry_run": False}
    )
    check("cleanup/orphan-removed",
          harness.client.get(f"/api/uploads/{orphan['sha256']}").status_code == 404)
    check("cleanup/referenced-upload-survives",
          harness.client.get(f"/api/uploads/{sha256}").status_code == 200)

    # Frames are only pruned where the video exists, so they are never the last
    # copy of what the run drew.
    complete = harness.create_job(sha256, target_epoch=300, num_iter=300, title="finished")
    harness.await_state(complete["id"], "complete", timeout=90)
    run_dir = harness.root / "jobs" / complete["id"] / "target" / "run"
    for path in (run_dir / "svg_to_png").glob("iter_*.png"):
        path.unlink()
        break  # keep the rest; we only need the directory to be non-empty
    if not (run_dir / "sketch.mp4").exists():
        dry = harness.client.post(
            "/api/maintenance/cleanup", json={"action": "prune_frames", "dry_run": True}
        ).json()
        check("cleanup/will-not-prune-frames-without-an-mp4",
              not any(item["id"] == complete["id"] for item in dry["items"]))

    dry = harness.client.post(
        "/api/maintenance/cleanup", json={"action": "prune_checkpoints", "dry_run": True}
    ).json()
    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if len(checkpoints) > 1:
        check("cleanup/prunes-all-but-the-last-checkpoint",
              any(item["id"] == complete["id"] for item in dry["items"]))
        harness.client.post(
            "/api/maintenance/cleanup", json={"action": "prune_checkpoints", "dry_run": False}
        )
        remaining = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
        check("cleanup/one-checkpoint-remains", len(remaining) == 1, str(len(remaining)))
        check("cleanup/the-reproducible-record-is-untouched",
              (run_dir / "final_sld.svg").exists() and (run_dir / "config.json").exists())
        check("cleanup/logs-are-never-pruned",
              any((harness.root / "jobs" / complete["id"] / "logs").glob("segment_*.log")))


# -- constraint sources: which SVGs may feed --avoid -------------------------


def test_constraint_sources(harness, sha256):
    """The three things the picker can hand to ``--avoid`` (and its siblings).

    The UI offers a job's final SVG, *any* of its frames, and an uploaded SVG.
    Only the first of those was exercised before, and the other two are the
    interesting ones: a frame is what you want when the good curve was at 1700
    rather than at 4000, and an upload is geometry from outside the service
    entirely. All three must end up **copied** into the new job (Spec 2 SS4.3),
    because a constraint that reads from another job's directory breaks when
    that job is deleted.
    """
    print("\n--- constraint sources: a frame, and an uploaded SVG")
    source = harness.create_job(sha256, target_epoch=200, num_iter=1000, title="obstacle")
    source_id = source["id"]
    harness.await_state(source_id, "waiting")

    frames = harness.client.get(f"/api/jobs/{source_id}/frames").json()
    epochs = [frame["epoch"] for frame in frames["frames"] if frame["svg"]]
    check("sources/the-source-has-frames-with-svgs", len(epochs) >= 2, str(epochs))
    check("sources/an-unrescaled-run-offers-its-frames", frames["rescaled"] is False)

    # "That other job, at iteration N" -- the path the UI builds from the epoch.
    chosen = epochs[-1]
    created = harness.client.post(
        "/api/jobs",
        json={
            "title": "avoids a frame",
            "target_sha256": sha256,
            "target_epoch": 100,
            "params": {"num_iter": 1000},
            "inputs": [
                {
                    "role": "avoid",
                    "source_kind": "job",
                    "source_job_id": source_id,
                    "path": f"svg_logs/svg_iter{chosen}.svg",
                }
            ],
        },
    )
    check("sources/a-frame-is-accepted-as-an-avoid-input",
          created.status_code == 201, created.text[:200])
    if created.status_code == 201:
        detail = created.json()
        recorded = [entry for entry in detail["inputs"] if entry["role"] == "avoid"]
        check("sources/the-frame-input-is-recorded", len(recorded) == 1, str(recorded))
        check("sources/it-remembers-which-job-it-came-from",
              recorded and recorded[0]["source_job_id"] == source_id)
        copied = harness.root / recorded[0]["stored_path"]
        original = (harness.root / "jobs" / source_id / "target" / "run"
                    / "svg_logs" / f"svg_iter{chosen}.svg")
        check("sources/the-bytes-are-copied-not-referenced",
              copied.exists() and copied.read_bytes() == original.read_bytes(),
              str(copied))
        check("sources/the-copy-keeps-the-svg-suffix", copied.suffix == ".svg")
        check("sources/the-param-points-at-the-copy",
              detail["params"]["avoid"] == [recorded[0]["stored_path"]],
              str(detail["params"]["avoid"]))

    # An SVG uploaded from outside: content-addressed like any other upload, and
    # resolved by digest rather than by a path in someone's run directory.
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
    svg += b'<path d="M 10 10 L 400 400" stroke="black" fill="none"/></svg>'
    uploaded = harness.client.post(
        "/api/uploads", files={"file": ("ring.svg", svg, "image/svg+xml")}
    )
    check("sources/an-svg-can-be-uploaded", uploaded.status_code == 200, uploaded.text[:120])
    digest = uploaded.json()["sha256"]
    check("sources/the-uploaded-svg-is-served-back",
          harness.client.get(f"/api/uploads/{digest}").content == svg)

    from_upload = harness.client.post(
        "/api/jobs",
        json={
            "title": "avoids an upload",
            "target_sha256": sha256,
            "target_epoch": 100,
            "params": {"num_iter": 1000},
            "inputs": [{"role": "avoid", "source_kind": "upload", "sha256": digest}],
        },
    )
    check("sources/an-uploaded-svg-is-accepted-as-an-avoid-input",
          from_upload.status_code == 201, from_upload.text[:200])
    if from_upload.status_code == 201:
        recorded = [entry for entry in from_upload.json()["inputs"] if entry["role"] == "avoid"]
        stored = harness.root / recorded[0]["stored_path"]
        check("sources/the-uploaded-svg-is-copied-in",
              stored.exists() and stored.read_bytes() == svg, str(stored))
        check("sources/the-uploaded-copy-keeps-the-svg-suffix", stored.suffix == ".svg")

    check("sources/a-frame-that-does-not-exist-is-refused",
          harness.client.post(
              "/api/jobs",
              json={
                  "target_sha256": sha256,
                  "target_epoch": 100,
                  "params": {"num_iter": 1000},
                  "inputs": [{"role": "avoid", "source_kind": "job",
                              "source_job_id": source_id,
                              "path": "svg_logs/svg_iter999999.svg"}],
              },
          ).status_code == 400)

    check("sources/a-path-that-escapes-the-run-directory-is-refused",
          harness.client.post(
              "/api/jobs",
              json={
                  "target_sha256": sha256,
                  "target_epoch": 100,
                  "params": {"num_iter": 1000},
                  "inputs": [{"role": "avoid", "source_kind": "job",
                              "source_job_id": source_id,
                              "path": "../../../../etc/passwd"}],
              },
          ).status_code == 400)


# -- deleting a ticked selection --------------------------------------------


def test_bulk_delete(harness, sha256):
    """``delete_jobs``: what the rail's "Delete 7" button sends.

    The dialog states a count and a byte figure before it asks, and both come
    from this endpoint's dry run of the same selection it then performs -- so
    what is checked here is that the two agree, and that the dry run really
    does nothing.
    """
    print("\n--- deleting a ticked selection (delete_jobs)")
    doomed = [
        harness.create_job(sha256, target_epoch=100, num_iter=1000, title=f"bulk {index}")["id"]
        for index in range(3)
    ]
    spared = harness.create_job(sha256, target_epoch=100, num_iter=1000, title="spared")["id"]
    for job_id in doomed:
        harness.await_state(job_id, "waiting", timeout=90)

    dry = harness.client.post(
        "/api/maintenance/cleanup",
        json={"action": "delete_jobs", "job_ids": doomed, "dry_run": True},
    ).json()
    check("bulk-delete/dry-run-counts-exactly-what-was-ticked",
          dry["job_count"] == 3 and sorted(item["id"] for item in dry["items"]) == sorted(doomed),
          str(dry["job_count"]))
    check("bulk-delete/dry-run-reports-bytes", dry["bytes"] > 0, str(dry["bytes"]))
    check("bulk-delete/dry-run-did-not-act",
          all(harness.client.get(f"/api/jobs/{job_id}").status_code == 200 for job_id in doomed))

    check("bulk-delete/an-unknown-id-fails-the-whole-call",
          harness.client.post(
              "/api/maintenance/cleanup",
              json={"action": "delete_jobs", "job_ids": doomed + ["01NOSUCHJOB"],
                    "dry_run": False},
          ).status_code == 404)
    check("bulk-delete/nothing-was-deleted-by-the-refused-call",
          all(harness.client.get(f"/api/jobs/{job_id}").status_code == 200 for job_id in doomed))

    real = harness.client.post(
        "/api/maintenance/cleanup",
        json={"action": "delete_jobs", "job_ids": doomed, "dry_run": False},
    ).json()
    check("bulk-delete/real-run-matches-the-dry-run-count", real["job_count"] == dry["job_count"])
    check("bulk-delete/real-run-matches-the-dry-run-bytes", real["bytes"] == dry["bytes"])
    check("bulk-delete/all-of-them-are-gone",
          wait_for(
              lambda: all(
                  harness.client.get(f"/api/jobs/{job_id}").status_code == 404
                  for job_id in doomed
              ),
              timeout=30, what="the ticked jobs to be reclaimed",
          ))
    check("bulk-delete/their-directories-are-gone",
          not any((harness.root / "jobs" / job_id).exists() for job_id in doomed))
    check("bulk-delete/the-unticked-job-survives",
          harness.client.get(f"/api/jobs/{spared}").status_code == 200)

    check("bulk-delete/an-empty-selection-is-a-no-op",
          harness.client.post(
              "/api/maintenance/cleanup",
              json={"action": "delete_jobs", "job_ids": [], "dry_run": False},
          ).json()["job_count"] == 0)


def test_rename(harness, sha256):
    """One job by PATCH, a ticked selection by ``/api/jobs/rename``.

    The bulk name comes from each job's stored seed, never its title, and is
    unique across every job -- so the checks plant a stale seed in a title and
    an outsider already holding one of the names.
    """
    print("\n--- renaming")
    single = harness.create_job(sha256, target_epoch=50, num_iter=1000, title="before", seed=5)
    renamed = harness.client.patch(f"/api/jobs/{single['id']}", json={"title": "  after  "}).json()
    check("rename/single-sets-trimmed-title", renamed["title"] == "after", renamed["title"])
    cleared = harness.client.patch(f"/api/jobs/{single['id']}", json={"title": "   "}).json()
    check("rename/single-blank-clears", cleared["title"] is None, repr(cleared["title"]))

    outsider = harness.create_job(sha256, target_epoch=50, num_iter=1000,
                                  title="batch · s2", seed=99)
    picked = [
        harness.create_job(sha256, target_epoch=50, num_iter=1000, title=title, seed=seed)["id"]
        for title, seed in (("x · s1060", 1), ("y", 2), ("z", 3), ("w", 3))
    ]
    request = {"job_ids": picked, "base": "batch", "dry_run": True}
    dry = harness.client.post("/api/jobs/rename", json=request).json()
    proposed = [item["title"] for item in dry["items"]]
    check("rename/bulk-names-from-seeds-with-collisions-numbered",
          proposed == ["batch · s1", "batch · s2 · v01", "batch · s3 · v01", "batch · s3 · v02"],
          str(proposed))
    check("rename/bulk-dry-run-did-not-act",
          harness.job(picked[0])["title"] == "x · s1060")

    real = harness.client.post("/api/jobs/rename", json={**request, "dry_run": False}).json()
    check("rename/bulk-real-run-matches-dry-run",
          [item["title"] for item in real["items"]] == proposed)
    check("rename/bulk-titles-landed",
          [harness.job(job_id)["title"] for job_id in picked] == proposed)
    check("rename/bulk-left-the-outsider-alone",
          harness.job(outsider["id"])["title"] == "batch · s2")

    check("rename/bulk-blank-base-is-400",
          harness.client.post("/api/jobs/rename",
                              json={"job_ids": picked, "base": " "}).status_code == 400)
    check("rename/bulk-unknown-id-is-404",
          harness.client.post("/api/jobs/rename",
                              json={"job_ids": picked + ["01NOSUCHJOB"], "base": "q"}
                              ).status_code == 404)
    check("rename/bulk-refused-call-changed-nothing",
          [harness.job(job_id)["title"] for job_id in picked] == proposed)


# -- runner -----------------------------------------------------------------


def main():
    harness = Harness()
    failures = []
    try:
        test_param_schema_mirror()
        harness.start_api()
        harness.start_worker()
        test_serves_the_web_ui(harness)
        test_params_last_and_presets(harness)
        sha256 = harness.upload()["sha256"]
        parent_id = test_frames(harness, sha256)
        test_marks(harness, sha256)
        test_lineage_and_filters(harness, sha256, parent_id)
        test_global_events(harness, sha256)
        test_constraint_sources(harness, sha256)
        test_bulk_delete(harness, sha256)
        test_rename(harness, sha256)
        test_cleanup(harness, sha256)
    finally:
        failures = [label for label, passed in RESULTS if not passed]
        harness.close()

    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    print("\nRESULT:", "ALL PASS" if not failures else "FAILURE")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
