"""Where everything lives, and how to invoke SLDgen.

Every path the service touches is derived from a single root (Spec 2 SS3),
defaulting to ``work/`` inside the repository checkout. Both units must agree on
that root, so it is read from one environment variable and never defaulted
independently.

The interpreter and script used to run a segment are configurable for the same
reason the root is: the worker's supervision logic (spawn, poll, signal,
classify) is worth testing without a GPU, and a stub script driven through the
real code path tests far more than a mocked subprocess would.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Interpreter of the conda env that can actually import torch/pydiffvg. The
#: service itself never imports them; it only spawns this.
DEFAULT_SLDGEN_PYTHON = Path("/home/helge/miniforge3/envs/sldgen/bin/python")


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class ServiceConfig:
    root: Path
    sldgen_python: Path
    sldgen_script: Path
    partition_script: Path
    #: Defaulted, unlike its siblings, because it is only ever the repo's own
    #: script: nothing about a deployment changes it, and a test building a
    #: config by hand should not have to know it exists.
    canny_script: Path = REPO_ROOT / "sld_canny_svg.py"
    #: How often the worker re-reads a running segment's state.json (Spec 2 SS7:
    #: poll, don't watch -- no inotify watch limits, and it survives the atomic
    #: replace that writing state.json uses).
    poll_interval: float = 1.0
    #: How long the worker sleeps when the queue is empty.
    claim_interval: float = 2.0
    #: SIGTERM -> SIGKILL grace period. Must exceed one iteration plus a
    #: checkpoint write, and ideally a whole finalisation (metrics + ffmpeg),
    #: since interrupting that mid-write corrupts outputs.
    grace_seconds: float = 120.0
    #: Non-zero by default so a crash never costs more than this many iterations
    #: (Spec 2 SS8: without a checkpoint the only honest recovery is epoch 0).
    default_checkpoint_interval: int = 200

    @classmethod
    def from_env(cls):
        return cls(
            root=_env_path("SLDGEN_WORK_ROOT", REPO_ROOT / "work"),
            sldgen_python=_env_path("SLDGEN_PYTHON", DEFAULT_SLDGEN_PYTHON),
            sldgen_script=_env_path("SLDGEN_SCRIPT", REPO_ROOT / "sldgen.py"),
            partition_script=_env_path("SLDGEN_PARTITION_SCRIPT", REPO_ROOT / "sld_partition.py"),
            canny_script=_env_path("SLDGEN_CANNY_SCRIPT", REPO_ROOT / "sld_canny_svg.py"),
            poll_interval=_env_float("SLDGEN_POLL_INTERVAL", 1.0),
            claim_interval=_env_float("SLDGEN_CLAIM_INTERVAL", 2.0),
            grace_seconds=_env_float("SLDGEN_GRACE_SECONDS", 120.0),
            default_checkpoint_interval=_env_int("SLDGEN_CHECKPOINT_INTERVAL", 200),
        )

    # -- filesystem layout (Spec 2 SS3) -----------------------------------

    @property
    def db_path(self):
        return self.root / "sldgen.sqlite"

    @property
    def lock_path(self):
        return self.root / "worker.lock"

    @property
    def uploads_dir(self):
        return self.root / "uploads"

    @property
    def jobs_dir(self):
        return self.root / "jobs"

    @property
    def partitions_dir(self):
        return self.root / "partitions"

    @property
    def tmp_dir(self):
        return self.root / "tmp"

    def job_dir(self, job_id):
        return self.jobs_dir / job_id

    def job_inputs_dir(self, job_id):
        return self.job_dir(job_id) / "inputs"

    def job_logs_dir(self, job_id):
        return self.job_dir(job_id) / "logs"

    def run_dir(self, job_id):
        """SLDgen's own output directory for a job.

        ``set_output_directories`` inserts ``Path(args.target).stem`` between
        --output-dir and --experiment-name. The worker always names the input
        ``target.png`` and passes ``--experiment-name run``, so this path is
        deterministic rather than something to discover after the fact.
        """
        return self.job_dir(job_id) / "target" / "run"

    def state_path(self, job_id):
        return self.run_dir(job_id) / "state.json"

    def latest_checkpoint(self, job_id):
        return self.run_dir(job_id) / "checkpoints" / "latest.pt"

    def segment_log(self, job_id, seq):
        return self.job_logs_dir(job_id) / f"segment_{seq:03d}.log"

    def partition_dir(self, partition_id):
        return self.partitions_dir / partition_id

    def upload_path(self, sha256):
        return self.uploads_dir / f"{sha256}.png"

    def ensure_layout(self):
        """Create the root and its fixed subdirectories.

        Safe to call repeatedly and safe after the whole root has been deleted:
        the service recreates it empty and nothing in the repo depends on it.
        """
        for path in (self.root, self.uploads_dir, self.jobs_dir, self.partitions_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self
