"""Partition endpoints against the real ``sld_partition.py`` (Spec 2 SS11).

Partitioning is the one part of the API that shells out to the heavyweight conda
env, so this test uses that interpreter for real rather than stubbing it: the
point of the exercise is that a lightweight API process can drive a script whose
dependencies (numpy, scipy, svgpathtools, matplotlib) it does not itself have.

The API is driven in-process with FastAPI's TestClient -- no daemons needed,
because partitioning is synchronous and never touches the GPU queue.

    PYTHONPATH=. .venv-service/bin/python test_service_partitions.py
"""

import math
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from fastapi.testclient import TestClient

from sldgen_api.app import create_app
from sldgen_service import jobs as job_files
from sldgen_service.config import DEFAULT_SLDGEN_PYTHON, REPO_ROOT, ServiceConfig
from sldgen_service.params import canonical_params
from sldgen_service.store import Store

RESULTS = []


def check(label, condition, detail=""):
    passed = bool(condition)
    RESULTS.append((label, passed))
    print(f"  [{label}] {'PASS' if passed else 'FAIL'}{(' -- ' + detail) if detail else ''}")
    return passed


def make_png(width, height, value=128):
    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(
        b"\x00" + bytes([min(255, value + (y * 255) // max(height - 1, 1))]) * width
        for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def spiral_svg(size=512, turns=6, points=1200):
    """A single open polyline that looks enough like an SLDgen result to partition.

    Every strategy needs a long, spatially spread traversal: horizontal splits on
    y, radial on angle, sequence on traversal order.
    """
    centre = size / 2
    coordinates = []
    for index in range(points):
        t = index / (points - 1)
        angle = turns * 2 * math.pi * t
        radius = 0.45 * size * t
        coordinates.append((centre + radius * math.cos(angle), centre + radius * math.sin(angle)))
    commands = " ".join(
        f"{'M' if i == 0 else 'L'} {x:.3f},{y:.3f}" for i, (x, y) in enumerate(coordinates)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<path d="{commands}" fill="none" stroke="black" stroke-width="1"/></svg>'
    )


def build_root():
    root = Path(tempfile.mkdtemp(prefix="sldgen-partitions-"))
    config = ServiceConfig(
        root=root,
        sldgen_python=DEFAULT_SLDGEN_PYTHON,
        sldgen_script=REPO_ROOT / "sldgen.py",
        partition_script=REPO_ROOT / "sld_partition.py",
    ).ensure_layout()

    store = Store(config)
    digest, _ = job_files.store_upload(config, make_png(64, 64))
    job = job_files.create_job(
        store, config, target_sha256=digest, params=canonical_params({"num_iter": 100}),
        target_epoch=100, title="master",
    )
    run_dir = config.run_dir(job["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_sld.svg").write_text(spiral_svg())
    (run_dir / "condition_depth.png").write_bytes(make_png(512, 512))
    store.close()
    return config, job["id"], digest


def main():
    if not DEFAULT_SLDGEN_PYTHON.exists():
        print(f"SKIP: the sldgen conda interpreter is not at {DEFAULT_SLDGEN_PYTHON}")
        return 0

    config, job_id, digest = build_root()
    client = TestClient(create_app(config))
    try:
        print("\n--- preview: re-runnable, overwrites in place")
        response = client.post(
            "/api/partitions/preview",
            json={"source_job_id": job_id, "strategy": "sequence", "n": 3},
        )
        check("preview/succeeds", response.status_code == 200, response.text[:200])
        body = response.json()
        check("preview/returns-n-svgs", len(body["svgs"]) == 3, str(body["svgs"]))
        check("preview/returns-a-preview-image", body["preview_url"] is not None)
        check("preview/preview-image-served",
              client.get(body["preview_url"]).status_code == 200)
        check("preview/svgs-served", client.get(body["svg_urls"][0]).status_code == 200)

        preview_dir = config.partitions_dir / f"preview-{job_id}"
        response = client.post(
            "/api/partitions/preview",
            json={"source_job_id": job_id, "strategy": "radial", "n": 2},
        )
        check("preview/rerun-succeeds", response.status_code == 200, response.text[:200])
        # Scrubbing N downwards must not leave the previous run's extra SVGs behind.
        check("preview/overwrites-in-place",
              sorted(p.name for p in preview_dir.glob("partition_*.svg"))
              == ["partition_0.svg", "partition_1.svg"],
              str(sorted(p.name for p in preview_dir.glob("partition_*.svg"))))

        print("\n--- strategies")
        for strategy in ("horizontal", "vertical", "radial", "sequence", "cluster"):
            response = client.post(
                "/api/partitions/preview",
                json={"source_job_id": job_id, "strategy": strategy, "n": 3},
            )
            check(f"strategy/{strategy}",
                  response.status_code == 200 and len(response.json()["svgs"]) == 3,
                  response.text[:160])

        response = client.post(
            "/api/partitions/preview",
            json={"source_job_id": job_id, "strategy": "labelmap", "n": 3},
        )
        check("strategy/labelmap-defaults-to-the-condition-image",
              response.status_code == 200, response.text[:200])

        response = client.post(
            "/api/partitions/preview",
            json={"source_job_id": job_id, "strategy": "nonsense", "n": 3},
        )
        check("strategy/unknown-is-400", response.status_code == 400)

        print("\n--- commit")
        response = client.post(
            "/api/partitions",
            json={"source_job_id": job_id, "strategy": "sequence", "n": 3,
                  "params": {"connect_tails": True}},
        )
        check("commit/created", response.status_code == 201, response.text[:200])
        partition = response.json()
        check("commit/row-recorded",
              partition["strategy"] == "sequence" and partition["n"] == 3)
        check("commit/output-dir-recorded", partition["output_dir"].startswith("partitions/"))
        check("commit/files-on-disk",
              len(list((config.root / partition["output_dir"]).glob("partition_*.svg"))) == 3)
        check("commit/listed",
              any(row["id"] == partition["id"]
                  for row in client.get("/api/partitions").json()["partitions"]))

        archive = client.get(f"/api/partitions/{partition['id']}/download.zip")
        check("commit/zip-downloads",
              archive.status_code == 200 and archive.content[:2] == b"PK")

        print("\n--- a partition can seed the next job (the DAG closes)")
        response = client.post(
            "/api/jobs",
            json={
                "title": "from-partition",
                "target_sha256": digest,
                "target_epoch": 100,
                "params": {"num_iter": 100},
                "inputs": [
                    {"role": "attract", "source_kind": "partition",
                     "source_partition_id": partition["id"], "path": "partition_0.svg"},
                    {"role": "avoid", "source_kind": "partition",
                     "source_partition_id": partition["id"], "path": "partition_1.svg"},
                ],
            },
        )
        check("dag/job-accepts-partition-inputs", response.status_code == 201, response.text[:300])
        child = response.json()
        check("dag/attract-points-at-a-copy",
              child["params"]["attract"] == [f"jobs/{child['id']}/inputs/attract_000.svg"],
              str(child["params"]["attract"]))
        check("dag/copy-exists", (config.root / child["params"]["attract"][0]).exists())
        check("dag/provenance-recorded",
              any(record["source_partition_id"] == partition["id"]
                  for record in child["inputs"]))
        check("dag/avoid-and-attract-compose",
              child["params"]["avoid"] is not None and child["params"]["attract"] is not None)
    finally:
        client.close()
        shutil.rmtree(config.root, ignore_errors=True)

    failures = [label for label, passed in RESULTS if not passed]
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    print("\nRESULT:", "ALL PASS" if not failures else "FAILURE")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
