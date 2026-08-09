"""``python -m sldgen_api`` -- the API systemd entry point (Spec 2 SS14).

**Binds to the tailnet, never 0.0.0.0.** The service has no authentication by
design, so the bind address *is* the access control. The default is loopback;
``SLDGEN_API_HOST`` selects the tailnet address on the real host.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    import uvicorn

    host = os.environ.get("SLDGEN_API_HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        raise SystemExit(
            "refusing to bind to a wildcard address: this API has no authentication. "
            "Set SLDGEN_API_HOST to the tailnet address."
        )
    port = int(os.environ.get("SLDGEN_API_PORT", "8765"))

    uvicorn.run(
        "sldgen_api.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=os.environ.get("SLDGEN_API_LOG_LEVEL", "info"),
        access_log=False,
    )


if __name__ == "__main__":
    main()
