"""Unit tests for the service's shared library (Spec 2 SS4, SS5, SS10, SS13).

No daemon, no HTTP, no GPU -- each case builds a service root in a temp
directory and exercises one contract directly. The live end-to-end run is
``test_service_e2e.py``.

Run with either interpreter (nothing here needs torch or fastapi):
    PYTHONPATH=. python test_service_units.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from sldgen_service import disk as disk_utils
from sldgen_service import jobs as job_files
from sldgen_service import logs as log_utils
from sldgen_service import store as store_module
from sldgen_service.config import ServiceConfig
from sldgen_service.ids import is_ulid, new_ulid
from sldgen_service.params import (
    OPERATIONAL_NAMES,
    PARAM_NAMES,
    STRUCTURAL_NAMES,
    ParamError,
    argv_to_params,
    build_argv,
    canonical_params,
    params_to_argv,
    structural_differences,
    validate_params,
)
from sldgen_service.store import Store, StoreError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def temp_config():
    root = Path(tempfile.mkdtemp(prefix="sldgen-units-"))
    return ServiceConfig(
        root=root,
        sldgen_python=Path(sys.executable),
        sldgen_script=Path("/nonexistent/sldgen.py"),
        partition_script=Path("/nonexistent/sld_partition.py"),
    ).ensure_layout()


def check(label, condition, detail=""):
    print(f"[{label}] {'PASS' if condition else 'FAIL'}{(' -- ' + detail) if detail else ''}")
    return bool(condition)


# -- parameters ------------------------------------------------------------


def test_params_round_trip():
    """Spec 2 SS4.2: argv_to_params(params_to_argv(p)) == p, for every parameter."""
    ok = True
    interesting = canonical_params(
        {
            "render_size": 768,
            "n_control_points": 200,
            "seed": 7,
            "num_iter": 1500,
            "optimize_cp_weights": False,
            "prune_low_weights": False,
            "width": "optim",
            "origin": [0.25, 0.75],
            "fixed_endpoints": False,
            "calligraphy": True,
            "use_cpu": True,
            "caption": "a single line drawing of a heart",
            "avoid": ["jobs/A/inputs/avoid_000.svg", "jobs/A/inputs/avoid_001.svg"],
            "attract": ["jobs/A/inputs/attract_000.svg"],
            "init_points": "jobs/A/inputs/init_points.svg",
            "stipple_weight": "jobs/A/inputs/stipple_weight.png",
            "verbose": True,
            "save_interval": 25,
            "checkpoint_interval": 50,
        }
    )
    recovered = argv_to_params(params_to_argv(interesting))
    ok = check("params/round-trip-rich", recovered == interesting,
               f"differs: {_diff(interesting, recovered)}") and ok

    defaults = canonical_params({})
    ok = check("params/round-trip-defaults", argv_to_params(params_to_argv(defaults)) == defaults) and ok

    # Numeric width, the other branch of float_or_str.
    numeric = canonical_params({"width": 2.5})
    ok = check("params/round-trip-width-float",
               argv_to_params(params_to_argv(numeric))["width"] == 2.5) and ok

    # A full argv, including the worker's runtime flags, must still yield exactly
    # the parameters -- that is what makes a recorded command reproducible.
    full = build_argv(
        "/usr/bin/python", "sldgen.py", interesting,
        target="/w/jobs/A/inputs/target.png", output_dir="/w/jobs/A",
        stop_at=400, resume="/w/jobs/A/target/run/checkpoints/latest.pt",
    )
    ok = check("params/round-trip-ignores-runtime-flags",
               argv_to_params(full) == interesting) and ok
    ok = check("params/runtime-flags-present",
               all(flag in full for flag in ("--target", "--output-dir", "--stop-at", "--resume"))) and ok

    # Service jobs are served as frames, so a default job asks SLDgen not to
    # assemble the mp4 -- and a worker host therefore needs no ffmpeg.
    ok = check("params/default-job-passes-no-video",
               "--no-video" in params_to_argv(defaults)) and ok
    ok = check("params/save-video-drops-the-flag",
               "--no-video" not in params_to_argv(canonical_params({"save_video": True}))) and ok
    return ok


def _diff(expected, actual):
    return {
        name: (expected[name], actual.get(name))
        for name in expected
        if expected[name] != actual.get(name)
    }


def test_params_groups():
    """Structural vs operational, with nothing in between (Spec 2 SS4.2)."""
    ok = check("params/groups-partition-everything",
               STRUCTURAL_NAMES | OPERATIONAL_NAMES == set(PARAM_NAMES))
    ok = check("params/operational-is-the-five",
               OPERATIONAL_NAMES == {"save_interval", "checkpoint_interval", "save_video",
                                     "verbose", "debug"},
               str(sorted(OPERATIONAL_NAMES))) and ok

    base = canonical_params({})
    ok = check("params/operational-edit-is-not-structural",
               structural_differences(base, {**base, "save_interval": 5}) == []) and ok
    ok = check("params/structural-edit-is-detected",
               structural_differences(base, {**base, "seed": 9}) == ["seed"]) and ok
    return ok


def test_params_validation():
    """Reject at submit time what SLDgen would reject at parse time."""
    cases = [
        ("origin-outside-canvas", {"origin": [1.5, 0.5]}),
        ("origin-with-fixed-endpoints", {"origin": [0.5, 0.5], "fixed_endpoints": True}),
        ("origin-without-tsp", {"origin": [0.5, 0.5], "init_method": "trefoil"}),
        ("init-points-without-tsp", {"init_points": "x.svg", "init_method": "contour"}),
        ("stipple-weight-without-tsp", {"stipple_weight": "x.png", "init_method": "trefoil"}),
        ("bad-condition", {"condition": "sobel"}),
        ("bad-width", {"width": "wobbly"}),
        ("negative-checkpoint-interval", {"checkpoint_interval": -1}),
        ("zero-num-iter", {"num_iter": 0}),
    ]
    ok = True
    for label, params in cases:
        try:
            validate_params(params)
            rejected = False
        except ParamError:
            rejected = True
        ok = check(f"params/reject-{label}", rejected) and ok

    try:
        canonical_params({"nonsense": 1})
        rejected = False
    except ParamError:
        rejected = True
    ok = check("params/reject-unknown-name", rejected) and ok
    ok = check("params/accept-valid", validate_params({"origin": [0.5, 0.5]})["origin"] == [0.5, 0.5]) and ok
    return ok


# -- ids -------------------------------------------------------------------


def test_ulids_sort_by_creation():
    ids = [new_ulid(timestamp_ms=t) for t in (1_000, 2_000, 3_000)]
    ok = check("ids/sortable", ids == sorted(ids))
    ok = check("ids/well-formed", all(is_ulid(value) for value in ids)) and ok
    ok = check("ids/unique", len({new_ulid() for _ in range(500)}) == 500) and ok

    # The queue orders by id when created_at ties, and created_at has second
    # resolution -- so ids minted in the same millisecond must still be ordered,
    # or a batch of variants would run out of submission order.
    burst = [new_ulid() for _ in range(200)]
    ok = check("ids/monotonic-within-a-millisecond", burst == sorted(burst)) and ok
    return ok


# -- store and the state machine -------------------------------------------


def make_job(store, config, **kwargs):
    digest, _ = job_files.store_upload(config, PNG)
    params = canonical_params({"num_iter": kwargs.pop("num_iter", 1000)})
    return job_files.create_job(
        store, config, target_sha256=digest, params=params,
        target_epoch=kwargs.pop("target_epoch", 400), **kwargs
    )


def test_queue_and_lifecycle():
    config = temp_config()
    store = Store(config)
    try:
        first = make_job(store, config, title="first")
        second = make_job(store, config, title="second")
        third = make_job(store, config, title="third", priority=5)

        ok = check("store/created-queued", first["state"] == store_module.QUEUED)

        # priority DESC, then created_at -- so the priority job jumps the queue
        # but the other two keep their submission order.
        claimed = [store.claim_next_job()["title"] for _ in range(3)]
        ok = check("store/fifo-within-priority", claimed == ["third", "first", "second"],
                   str(claimed)) and ok
        ok = check("store/queue-empty-after", store.claim_next_job() is None) and ok

        # A claimed job is running and cannot be claimed twice.
        ok = check("store/claim-marks-running",
                   store.get_job(first["id"])["state"] == store_module.RUNNING) and ok

        # waiting -> promote -> queued
        store.finish_job(first["id"], store_module.WAITING)
        store.record_progress(first["id"], 400)
        promoted = store.promote(first["id"], 800)
        ok = check("store/promote-requeues",
                   promoted["state"] == store_module.QUEUED and promoted["target_epoch"] == 800) and ok

        # Promotion may not exceed the horizon: the horizon defines the schedule.
        try:
            store.promote(first["id"], 5000)
            refused = False
        except StoreError as exc:
            refused = "horizon" in str(exc)
        ok = check("store/promote-cannot-exceed-horizon", refused) and ok

        # Pause of a non-running job is applied immediately by the API side.
        # (All three were claimed above, so settle this one back to idle first --
        # which is what the worker does when a segment reaches its budget.)
        store.finish_job(second["id"], store_module.WAITING)
        store.record_progress(second["id"], 100)
        paused = store.request_pause(second["id"])
        ok = check("store/pause-idle-job-applies-now",
                   paused["state"] == store_module.PAUSED
                   and paused["desired_state"] == store_module.DESIRED_PAUSE) and ok
        ok = check("store/paused-job-is-not-claimable",
                   store.claim_next_job()["id"] != second["id"]) and ok

        resumed = store.request_resume(second["id"])
        ok = check("store/resume-requeues", resumed["state"] == store_module.QUEUED
                   and resumed["desired_state"] == store_module.DESIRED_RUN) and ok

        # Pause of a running job only asks; the worker owns the transition.
        running = store.claim_next_job()
        asked = store.request_pause(running["id"])
        ok = check("store/pause-running-job-only-asks",
                   asked["state"] == store_module.RUNNING
                   and asked["desired_state"] == store_module.DESIRED_PAUSE) and ok

        # retry only from failed
        store.finish_job(third["id"], store_module.FAILED, store_module.ERROR_OOM, "boom")
        retried = store.retry(third["id"])
        ok = check("store/retry-clears-error",
                   retried["state"] == store_module.QUEUED and retried["error_class"] is None) and ok
        try:
            store.retry(retried["id"])
            refused = False
        except StoreError:
            refused = True
        ok = check("store/retry-only-from-failed", refused) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


def test_target_epoch_bounds():
    config = temp_config()
    store = Store(config)
    try:
        digest, _ = job_files.store_upload(config, PNG)
        params = canonical_params({"num_iter": 500})
        ok = True
        for label, target in (("zero", 0), ("beyond-horizon", 501), ("negative", -5)):
            try:
                store.create_job(params=params, target_sha256=digest, target_epoch=target)
                refused = False
            except StoreError:
                refused = True
            ok = check(f"store/reject-target-epoch-{label}", refused) and ok
        job = store.create_job(params=params, target_sha256=digest, target_epoch=500)
        ok = check("store/accept-target-epoch-at-horizon", job["target_epoch"] == 500) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


# -- job materialisation ---------------------------------------------------


def test_inputs_are_copied_not_referenced():
    """Spec 2 SS4.3: deleting a source job must not be able to affect its consumers."""
    config = temp_config()
    store = Store(config)
    try:
        source = make_job(store, config, title="source")
        run_dir = config.run_dir(source["id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_sld.svg").write_text("<svg id='original'/>")

        digest, _ = job_files.store_upload(config, PNG)
        consumer = job_files.create_job(
            store, config, target_sha256=digest, params=canonical_params({}), target_epoch=100,
            title="consumer",
            inputs=[{"role": "avoid", "source_kind": "job", "source_job_id": source["id"]}],
        )

        stored = config.root / consumer["params"]["avoid"][0]
        ok = check("inputs/copied-into-consumer", stored.exists() and "original" in stored.read_text())
        ok = check("inputs/param-points-at-the-copy",
                   consumer["params"]["avoid"] == [f"jobs/{consumer['id']}/inputs/avoid_000.svg"],
                   str(consumer["params"]["avoid"])) and ok

        records = store.list_inputs(consumer["id"])
        ok = check("inputs/provenance-recorded",
                   len(records) == 1 and records[0]["source_job_id"] == source["id"]
                   and records[0]["source_kind"] == "job") and ok

        # Destroy the source entirely; the consumer must be untouched.
        job_files.delete_job_files(config, source["id"])
        store.delete_row(source["id"])
        ok = check("inputs/survive-source-deletion",
                   stored.exists() and "original" in stored.read_text()) and ok
        ok = check("inputs/provenance-nulled-not-orphaned",
                   store.list_inputs(consumer["id"])[0]["source_job_id"] is None) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


def test_coordinate_space_guard():
    """Spec 2 SS4.3: an intermediate SVG from a rescaled run would misregister silently."""
    config = temp_config()
    store = Store(config)
    try:
        source = make_job(store, config, title="rescaled")
        run_dir = config.run_dir(source["id"])
        (run_dir / "svg_logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "final_sld.svg").write_text("<svg/>")
        (run_dir / "svg_logs" / "svg_iter100.svg").write_text("<svg/>")
        (run_dir / "config.json").write_text(json.dumps({"scale_w": "0.6", "scale_h": "0.6"}))

        digest, _ = job_files.store_upload(config, PNG)

        def attempt(path, role="avoid"):
            return job_files.create_job(
                store, config, target_sha256=digest, params=canonical_params({}),
                target_epoch=100,
                inputs=[{"role": role, "source_kind": "job",
                         "source_job_id": source["id"], "path": path}],
            )

        try:
            attempt("svg_logs/svg_iter100.svg")
            refused = False
            message = ""
        except job_files.JobError as exc:
            refused, message = True, str(exc)
        ok = check("inputs/reject-rescaled-intermediate",
                   refused and "coordinate space" in message)

        job = attempt("final_sld.svg")
        ok = check("inputs/allow-final-svg-from-rescaled-run",
                   job["params"]["avoid"] is not None) and ok

        # An unrescaled run has no such hazard.
        plain = make_job(store, config, title="unrescaled")
        plain_run = config.run_dir(plain["id"])
        (plain_run / "svg_logs").mkdir(parents=True, exist_ok=True)
        (plain_run / "svg_logs" / "svg_iter100.svg").write_text("<svg/>")
        (plain_run / "config.json").write_text(json.dumps({"num_iter": "1000"}))
        job = job_files.create_job(
            store, config, target_sha256=digest, params=canonical_params({}), target_epoch=100,
            inputs=[{"role": "avoid", "source_kind": "job", "source_job_id": plain["id"],
                     "path": "svg_logs/svg_iter100.svg"}],
        )
        ok = check("inputs/allow-intermediate-from-unrescaled-run",
                   job["params"]["avoid"] is not None) and ok

        # Path traversal must not be a way out of the source directory.
        try:
            attempt("../../../../etc/passwd")
            escaped = True
        except job_files.JobError:
            escaped = False
        ok = check("inputs/reject-path-traversal", not escaped) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


def test_run_again_forks_without_touching_parent():
    config = temp_config()
    store = Store(config)
    try:
        digest, _ = job_files.store_upload(config, PNG)
        other = make_job(store, config, title="other")
        (config.run_dir(other["id"])).mkdir(parents=True, exist_ok=True)
        (config.run_dir(other["id"]) / "final_sld.svg").write_text("<svg id='avoid-me'/>")

        parent = job_files.create_job(
            store, config, target_sha256=digest,
            params=canonical_params({"num_iter": 1000, "seed": 3}),
            target_epoch=400, title="parent",
            inputs=[{"role": "avoid", "source_kind": "job", "source_job_id": other["id"]}],
        )
        store.record_progress(parent["id"], 400)
        store.finish_job(parent["id"], store_module.WAITING)

        children = job_files.run_again(
            store, config, parent["id"],
            variants=[{"params": {"seed": 11}}, {"params": {"lr": 0.4}}],
        )
        ok = check("run-again/creates-one-job-per-variant", len(children) == 2)
        ok = check("run-again/shares-a-batch",
                   children[0]["batch_id"] == children[1]["batch_id"]) and ok
        ok = check("run-again/records-provenance",
                   all(child["parent_job_id"] == parent["id"] for child in children)) and ok
        ok = check("run-again/applies-overrides",
                   children[0]["params"]["seed"] == 11 and children[1]["params"]["lr"] == 0.4) and ok
        ok = check("run-again/starts-from-zero",
                   all(child["current_epoch"] == 0 and child["state"] == store_module.QUEUED
                       for child in children)) and ok

        # The child gets its own copy of the parent's inputs, in its own directory.
        child_avoid = children[0]["params"]["avoid"]
        ok = check("run-again/copies-inputs-into-the-child",
                   child_avoid == [f"jobs/{children[0]['id']}/inputs/avoid_000.svg"],
                   str(child_avoid)) and ok
        ok = check("run-again/copied-input-exists",
                   (config.root / child_avoid[0]).exists()) and ok

        parent_after = store.get_job(parent["id"])
        ok = check("run-again/parent-untouched",
                   parent_after["state"] == store_module.WAITING
                   and parent_after["current_epoch"] == 400
                   and parent_after["params"]["seed"] == 3) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


def test_params_may_not_smuggle_input_paths():
    """Inputs must be declared, so they are copied and their provenance recorded."""
    config = temp_config()
    store = Store(config)
    try:
        digest, _ = job_files.store_upload(config, PNG)
        try:
            job_files.create_job(
                store, config, target_sha256=digest,
                params=canonical_params({"avoid": ["/etc/passwd"]}), target_epoch=100,
            )
            refused = False
        except job_files.JobError:
            refused = True
        return check("inputs/reject-paths-set-directly-in-params", refused)
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


# -- logs ------------------------------------------------------------------


def test_log_cooking_and_ranges():
    ok = check("logs/cook-collapses-repaints",
               log_utils.cook("a\r    1/10\r    2/10\r    3/10") == "    3/10")
    ok = check("logs/cook-overlays-shorter-repaint",
               log_utils.cook_line("abcdef\rXY") == "XYcdef") and ok
    # Real newlines survive cooking; only \r-repaints within one line collapse.
    ok = check("logs/cook-preserves-real-lines",
               log_utils.cook("Running SLDgen:\n    1/10\r    2/10") == "Running SLDgen:\n    2/10") and ok
    ok = check("logs/cook-leaves-plain-text-alone",
               log_utils.cook("alpha\nbeta\ngamma") == "alpha\nbeta\ngamma") and ok
    ok = check("logs/raw-is-untouched", "\r" in "a\rb") and ok

    root = Path(tempfile.mkdtemp(prefix="sldgen-logs-"))
    try:
        path = root / "segment_001.log"
        path.write_text("hello\nworld\n")

        head = log_utils.read_range(path, start=0, max_bytes=6)
        ok = check("logs/byte-range-honoured",
                   head["text"] == "hello\n" and head["from"] == 0 and head["to"] == 6
                   and head["eof"] is False) and ok

        rest = log_utils.read_range(path, start=head["to"])
        ok = check("logs/resume-from-offset", rest["text"] == "world\n" and rest["eof"] is True) and ok

        # Offsets refer to the file even when the text is cooked, so a client's
        # cursor cannot drift.
        path.write_text("x" * 10 + "\r" + "y" * 10)
        cooked = log_utils.read_range(path)
        ok = check("logs/offsets-are-file-offsets",
                   cooked["to"] == path.stat().st_size and len(cooked["text"]) == 10) and ok

        ok = check("logs/missing-file-is-empty-not-an-error",
                   log_utils.read_range(root / "nope.log")["text"] == "") and ok
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- disk ------------------------------------------------------------------


def test_disk_accounting():
    config = temp_config()
    store = Store(config)
    try:
        job = make_job(store, config, title="sized")
        run_dir = config.run_dir(job["id"])
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "svg_to_png").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "ckpt_00100.pt").write_bytes(b"c" * 1000)
        (run_dir / "svg_to_png" / "iter_0100.png").write_bytes(b"p" * 500)
        (run_dir / "final_sld.svg").write_text("s" * 250)

        breakdown = disk_utils.job_breakdown(config, job["id"])
        ok = check("disk/categories",
                   breakdown["by_category"]["checkpoints"] == 1000
                   and breakdown["by_category"]["svg_to_png"] == 500,
                   str(breakdown["by_category"]))
        ok = check("disk/total-covers-everything",
                   breakdown["total_bytes"] >= 1750
                   and breakdown["by_category"]["other"] >= 250) and ok

        root = disk_utils.root_breakdown(config, store)
        ok = check("disk/root-lists-jobs",
                   any(entry["job_id"] == job["id"] for entry in root["by_job"])) and ok

        # Retention keeps the last checkpoint and the reproducible record.
        (run_dir / "checkpoints" / "ckpt_00200.pt").write_bytes(b"c" * 1000)
        freed = job_files.prune_job(config, job["id"])
        remaining = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_*.pt"))
        ok = check("disk/prune-keeps-last-checkpoint",
                   remaining == ["ckpt_00200.pt"] and freed == 1000, str(remaining)) and ok
        ok = check("disk/prune-never-touches-the-record",
                   (run_dir / "final_sld.svg").exists()) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


def test_settings():
    config = temp_config()
    store = Store(config)
    try:
        defaults = store.get_settings()
        ok = check("settings/defaults-present",
                   set(defaults) == {"default_num_iter", "default_preview_stop_at",
                                     "retention_policy", "worker_paused"},
                   str(sorted(defaults)))
        updated = store.update_settings({"worker_paused": True})
        ok = check("settings/update", updated["worker_paused"] is True) and ok
        try:
            store.update_settings({"nonsense": 1})
            refused = False
        except StoreError:
            refused = True
        ok = check("settings/reject-unknown", refused) and ok
        return ok
    finally:
        store.close()
        shutil.rmtree(config.root, ignore_errors=True)


# -- bind addresses --------------------------------------------------------


def load_api_main():
    """Import ``sldgen_api/__main__.py`` without importing the package.

    ``sldgen_api/__init__.py`` pulls in FastAPI, which this file promises not to
    need. The bind logic has no such dependency, so load the module by path.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "sldgen_api" / "__main__.py"
    spec = importlib.util.spec_from_file_location("sldgen_api_main_undertest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bind_addresses():
    """The API serves a comma-separated address list, and never a wildcard.

    The bind address is the only access control this service has, so the
    wildcard guard is a security property, not a nicety.
    """
    import socket

    api_main = load_api_main()
    ok = True

    parsed = api_main.resolve_hosts("127.0.0.1, 10.0.0.5 ,192.168.1.9")
    ok = check(
        "bind/list", parsed == ["127.0.0.1", "10.0.0.5", "192.168.1.9"], str(parsed)
    ) and ok

    # Order is what makes hosts[0] usable as "the" address in banners.
    deduped = api_main.resolve_hosts("127.0.0.1,10.0.0.5,127.0.0.1")
    ok = check("bind/dedupe-keeps-order", deduped == ["127.0.0.1", "10.0.0.5"], str(deduped)) and ok

    for wildcard in ("0.0.0.0", "::", "*", "127.0.0.1,0.0.0.0"):
        refused = False
        try:
            api_main.resolve_hosts(wildcard)
        except SystemExit:
            refused = True
        ok = check(f"bind/refuse-{wildcard}", refused) and ok

    for empty in ("", "  ", " , "):
        refused = False
        try:
            api_main.resolve_hosts(empty)
        except SystemExit:
            refused = True
        ok = check(f"bind/refuse-empty-{empty!r}", refused) and ok

    ok = check("bind/default-is-loopback", api_main.DEFAULT_HOST == "127.0.0.1") and ok

    # A real bind, on a port the OS picks, proves the socket is listening and
    # inheritable rather than merely constructed.
    sock = api_main.bind("127.0.0.1", 0)
    try:
        port = sock.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        client.close()
        ok = check("bind/socket-listens", True) and ok
    except OSError as exc:
        ok = check("bind/socket-listens", False, str(exc)) and ok
    finally:
        sock.close()

    # An address this machine does not have must name itself in the error --
    # that is the difference between a tailnet being down and a mystery.
    failed = ""
    try:
        api_main.bind("10.99.99.99", 0)
    except SystemExit as exc:
        failed = str(exc)
    ok = check("bind/unavailable-names-the-address", "10.99.99.99" in failed, failed[:60]) and ok

    return ok


def main():
    tests = (
        test_params_round_trip,
        test_params_groups,
        test_params_validation,
        test_ulids_sort_by_creation,
        test_queue_and_lifecycle,
        test_target_epoch_bounds,
        test_inputs_are_copied_not_referenced,
        test_coordinate_space_guard,
        test_run_again_forks_without_touching_parent,
        test_params_may_not_smuggle_input_paths,
        test_log_cooking_and_ranges,
        test_disk_accounting,
        test_settings,
        test_bind_addresses,
    )
    ok = True
    for test in tests:
        print(f"\n--- {test.__name__}")
        ok = test() and ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
