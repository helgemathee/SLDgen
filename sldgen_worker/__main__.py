"""``python -m sldgen_worker`` -- the systemd entry point (Spec 2 SS14).

Must be run by the conda env's interpreter directly, not through
``conda activate``: the unit declares the environment variables activation would
otherwise provide, because they are not applied when activation is bypassed.
"""

import logging
import sys

from sldgen_service.config import ServiceConfig

from .worker import Worker


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    config = ServiceConfig.from_env()
    worker = Worker(config)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
