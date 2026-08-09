"""ULIDs for job and partition identifiers.

Spec 2 SS3 asks for identifiers that are sortable and opaque. A ULID is both:
48 bits of millisecond timestamp followed by 80 random bits, Crockford base32.
Lexicographic order is creation order, which is what the queue index and the
job grid both want, and no dependency is needed to produce one.
"""

import os
import threading
import time

# Crockford base32: no I, L, O or U, so an id read aloud or typed by hand cannot
# be confused with a similar-looking one.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ULID_LENGTH = 26
RANDOM_BITS = 80

_lock = threading.Lock()
_last = (0, 0)  # (timestamp_ms, randomness) of the previous id


def _encode(value, length):
    chars = []
    for _ in range(length):
        value, remainder = divmod(value, 32)
        chars.append(ALPHABET[remainder])
    return "".join(reversed(chars))


def new_ulid(timestamp_ms=None):
    """A fresh, monotonically increasing ULID.

    Sort order must be creation order, and a plain random suffix does not give
    that: several ids minted inside the same millisecond -- which is exactly what
    submitting a batch of variants does -- would sort arbitrarily, and the queue
    would run them out of submission order.

    Within a millisecond the randomness is incremented rather than redrawn, which
    is the standard ULID monotonic rule.

    Pass ``timestamp_ms`` only to make a test deterministic.
    """
    global _last
    with _lock:
        now = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
        last_timestamp, last_randomness = _last
        if now == last_timestamp:
            randomness = (last_randomness + 1) % (1 << RANDOM_BITS)
        elif now < last_timestamp:
            # Clock went backwards (NTP step). Keep advancing rather than minting
            # an id that sorts before one already handed out.
            now, randomness = last_timestamp, (last_randomness + 1) % (1 << RANDOM_BITS)
        else:
            randomness = int.from_bytes(os.urandom(RANDOM_BITS // 8), "big")
        _last = (now, randomness)
    return _encode(now, 10) + _encode(randomness, 16)


def is_ulid(value):
    return (
        isinstance(value, str)
        and len(value) == ULID_LENGTH
        and all(character in ALPHABET for character in value)
    )
