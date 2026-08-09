"""Streaming helpers: zip archives and server-sent events.

Both exist to avoid staging files. A job archive is generated as it is sent, and
progress is pushed rather than polled -- though every SSE endpoint has a plain
JSON counterpart, because SSE through a badly configured reverse proxy is
exactly the kind of thing that fails in the field.
"""

import json
import zipfile

#: Compressing PNG, MP4 or checkpoint bytes again wastes CPU for ~0% gain.
COMPRESSED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".mp4", ".pt", ".zip"})


class _Sink:
    """A write-only, unseekable sink; zipfile falls back to data descriptors."""

    def __init__(self):
        self.buffer = bytearray()

    def write(self, payload):
        self.buffer += payload
        return len(payload)

    def flush(self):
        pass

    def seekable(self):
        return False

    def drain(self):
        chunk = bytes(self.buffer)
        self.buffer.clear()
        return chunk


def stream_zip(entries, chunk_size=1 << 16):
    """Yield a zip archive of ``(arcname, path)`` pairs without staging it.

    zipfile supports unseekable output, so the archive can be produced and sent
    incrementally; memory stays at roughly one chunk regardless of archive size.
    """
    sink = _Sink()
    archive = zipfile.ZipFile(sink, "w")
    try:
        for arcname, path in entries:
            suffix = str(path).lower().rsplit(".", 1)
            compress = (
                zipfile.ZIP_STORED
                if len(suffix) == 2 and f".{suffix[1]}" in COMPRESSED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            info = zipfile.ZipInfo(arcname)
            info.compress_type = compress
            with archive.open(info, "w") as destination, open(path, "rb") as source:
                while True:
                    block = source.read(chunk_size)
                    if not block:
                        break
                    destination.write(block)
                    if sink.buffer:
                        yield sink.drain()
            if sink.buffer:
                yield sink.drain()
    finally:
        archive.close()
    if sink.buffer:
        yield sink.drain()


def sse(event, data):
    """One server-sent event. Named events let the client attach one handler each."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"
