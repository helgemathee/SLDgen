"""``python -m sldgen_api`` -- the API systemd entry point (Spec 2 SS14).

**Binds to named addresses, never a wildcard.** The service has no
authentication by design, so the bind address *is* the access control. The
default is loopback; ``SLDGEN_API_HOST`` selects where else it answers.

``SLDGEN_API_HOST`` accepts a comma-separated list, because "the tailnet *and*
the LAN" is a normal thing to want::

    SLDGEN_API_HOST=127.0.0.1,100.93.68.60,192.168.178.101

Each address gets its own listening socket, all served by one process on one
port. ``start.sh`` can expand the tokens ``loopback``, ``tailscale`` and ``lan``
into the machine's current addresses; this module only takes literal ones.
"""

import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HOST = "127.0.0.1"

# "" is how uvicorn spells 0.0.0.0, and "*" is how people expect to.
WILDCARDS = {"0.0.0.0", "::", "*"}


def resolve_hosts(raw):
    """Parse ``SLDGEN_API_HOST`` into a list of literal addresses.

    Order is preserved and duplicates are dropped, so the first entry stays
    usable as "the" address for banners and health checks.
    """
    hosts = []
    for entry in raw.split(","):
        host = entry.strip()
        if not host:
            continue
        if host in WILDCARDS:
            raise SystemExit(
                f"refusing to bind to the wildcard address '{host}': this API has no "
                "authentication, so the bind address is the access control. List the "
                "addresses you actually want instead, e.g. "
                "SLDGEN_API_HOST=127.0.0.1,$(tailscale ip -4)"
            )
        if host not in hosts:
            hosts.append(host)
    if not hosts:
        raise SystemExit(
            "SLDGEN_API_HOST is empty. Unset it for loopback, or give it one or more "
            "addresses separated by commas."
        )
    return hosts


def bind(host, port):
    """Create one listening socket, or explain which address failed and why.

    Binding several addresses makes failure more likely, not less: a tailnet
    address disappears when Tailscale is down, and a DHCP lease can hand the
    LAN interface a different address overnight. Naming the address that failed
    is the difference between a five-second fix and a puzzle.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        raise SystemExit(
            f"cannot bind {host}:{port} ({exc.strerror}).\n"
            "  Address not available means this machine no longer has that address -- "
            "check `ip -4 -o addr show scope global` and `tailscale ip -4`.\n"
            "  Address already in use means the service is already running: ./stop.sh first."
        ) from exc
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def main():
    import uvicorn

    hosts = resolve_hosts(os.environ.get("SLDGEN_API_HOST", DEFAULT_HOST))
    port = int(os.environ.get("SLDGEN_API_PORT", "8765"))

    config = uvicorn.Config(
        "sldgen_api.app:create_app",
        factory=True,
        host=hosts[0],
        port=port,
        log_level=os.environ.get("SLDGEN_API_LOG_LEVEL", "info"),
        access_log=False,
    )
    server = uvicorn.Server(config)

    sockets = []
    try:
        for host in hosts:
            sockets.append(bind(host, port))
    except SystemExit:
        for sock in sockets:
            sock.close()
        raise

    for host in hosts:
        display = f"[{host}]" if ":" in host else host
        print(f"SLDgen API listening on http://{display}:{port}", flush=True)

    # serve(sockets=...) skips uvicorn's own binding and uses ours, which is the
    # only way to get more than one address out of a single process.
    server.run(sockets=sockets)


if __name__ == "__main__":
    main()
