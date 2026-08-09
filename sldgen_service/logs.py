"""Reading segment logs (Spec 2 SS13).

Two properties the UI depends on:

**Byte offsets, not line numbers.** A client fetches a range, remembers ``to``,
and asks again from there. Reconnecting after a network blip costs one request,
not a re-download of megabytes.

**Carriage-return cooking at serve time.** SLDgen's logs are dominated by tqdm
repainting one progress line thousands of times with ``\\r``. Cooked output
collapses each such run to its final state -- what a terminal actually shows --
while the file on disk keeps every byte, so ``raw=true`` is always the truth.

Offsets always refer to the *file*, never to the cooked text, so cooking can
never desynchronise a client's cursor.
"""

from pathlib import Path

DEFAULT_MAX_BYTES = 1 << 20  # 1 MiB per request; the UI pages through more


def cook_line(line):
    """Apply one line's carriage returns, overlaying rather than discarding.

    ``"abcdef\\rXY"`` becomes ``"XYcdef"``, exactly as a terminal would render it.
    tqdm always repaints the full line so the overlay is usually a plain
    replacement, but a shorter repaint must not silently truncate what it did not
    cover.
    """
    if "\r" not in line:
        return line
    out = ""
    for fragment in line.split("\r"):
        out = fragment + out[len(fragment):]
    return out


def cook(text):
    return "\n".join(cook_line(line) for line in text.split("\n"))


def read_range(path, start=0, max_bytes=DEFAULT_MAX_BYTES, raw=False):
    """Read ``[start, start + max_bytes)`` of a log file.

    Returns offsets into the file plus the (optionally cooked) text. A missing
    file is not an error: a segment that has been queued but not yet spawned has
    no log, and the UI should show an empty one rather than a 404.
    """
    path = Path(path)
    if not path.exists():
        return {"from": 0, "to": 0, "text": "", "eof": True, "size": 0}

    size = path.stat().st_size
    start = max(0, min(int(start), size))
    with open(path, "rb") as handle:
        handle.seek(start)
        payload = handle.read(max_bytes) if max_bytes else handle.read()
    end = start + len(payload)
    text = payload.decode("utf-8", errors="replace")
    return {
        "from": start,
        "to": end,
        "text": text if raw else cook(text),
        "eof": end >= size,
        "size": size,
    }


def tail(path, max_bytes=DEFAULT_MAX_BYTES, raw=False):
    """The last ``max_bytes`` of a log -- what "show me the end" needs."""
    path = Path(path)
    if not path.exists():
        return read_range(path)
    size = path.stat().st_size
    return read_range(path, start=max(0, size - max_bytes), max_bytes=max_bytes, raw=raw)


def write_header(handle, segment_seq, start_epoch, stop_at, resume_from, argv, timestamp):
    """Make the log self-contained: it can be pasted into an issue and reproduced."""
    handle.write(f"{'=' * 78}\n")
    handle.write(f"SLDgen segment {segment_seq}  started {timestamp}\n")
    handle.write(f"epochs {start_epoch} -> {stop_at}\n")
    handle.write(f"resume from: {resume_from or '(fresh start)'}\n")
    handle.write("argv:\n")
    handle.write("  " + " ".join(_quote(part) for part in argv) + "\n")
    handle.write(f"{'=' * 78}\n")
    handle.flush()


def write_footer(handle, exit_code, error_class, end_epoch, wall_seconds, iters_per_sec):
    handle.write(f"\n{'=' * 78}\n")
    handle.write(f"exit code   : {exit_code}\n")
    handle.write(f"error class : {error_class or '(none)'}\n")
    handle.write(f"end epoch   : {end_epoch}\n")
    handle.write(f"wall time   : {wall_seconds:.1f}s\n")
    if iters_per_sec:
        handle.write(f"mean it/s   : {iters_per_sec:.2f}\n")
    handle.write(f"{'=' * 78}\n")
    handle.flush()


def _quote(part):
    part = str(part)
    return f"'{part}'" if (not part or " " in part) else part
