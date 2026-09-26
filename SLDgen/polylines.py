"""Polyline landmarks (Spec 7 addendum SS6): validation and densification.

A polyline in a landmark file is a hard edge to hit -- a glasses rim, a
hairline -- given as a few vertices. The loader turns it into ordinary weighted
landmarks, about one per ``SPACING`` canvas pixels, sharing the polyline's
weight, so everything downstream sees a flat list of points.

Pure Python on purpose: ``sld_landmarks.py`` (no torch) and the run's loader
both use it.
"""

import math

#: Largest gap, in canvas pixels, between two points of a densified polyline.
SPACING = 8.0


def parse_polylines(entries, where="polylines"):
    """Validated polylines from a file's ``polylines`` value (None -> []).

    Each comes back as ``{"name", "closed", "weight", "xy", "source"}`` with
    float coordinates. Raises ValueError naming the bad entry.
    """
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{where} must be a list of polylines")
    lines = []
    for index, entry in enumerate(entries):
        label = f"{where}[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} needs a name")
        closed = bool(entry.get("closed", False))
        try:
            weight = float(entry.get("weight", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"{label} ({name}) has a non-numeric weight") from None
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{label} ({name}) needs a weight >= 0")
        raw = entry.get("xy")
        try:
            xy = [(float(point[0]), float(point[1])) for point in raw]
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"{label} ({name}) needs xy as a list of [x, y] pairs") from None
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in xy):
            raise ValueError(f"{label} ({name}) has a non-finite coordinate")
        needed = 3 if closed else 2
        if len(xy) < needed:
            kind = "a closed" if closed else "an open"
            raise ValueError(f"{label} ({name}) has {len(xy)} vertices; {kind} polyline needs {needed}")
        lines.append(
            {
                "name": name,
                "closed": closed,
                "weight": weight,
                "xy": xy,
                "source": str(entry.get("source", "manual")),
            }
        )
    return lines


def densify(xy, closed, spacing=SPACING):
    """Points along the polyline, no gap wider than ``spacing``.

    Every vertex is kept. Each segment is split into ``ceil(length / spacing)``
    equal parts. A closed polyline gets its closing segment and does not repeat
    its first vertex.
    """
    vertices = [(float(x), float(y)) for x, y in xy]
    segments = list(zip(vertices, vertices[1:]))
    if closed and len(vertices) > 2:
        segments.append((vertices[-1], vertices[0]))
    points = [vertices[0]]
    for (x0, y0), (x1, y1) in segments:
        parts = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing))
        for step in range(1, parts + 1):
            t = step / parts
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    if closed and len(vertices) > 2:
        points.pop()  # the closing segment ends on the first vertex
    return points


def polyline_points(lines, spacing=SPACING):
    """``(xy, weight)`` lists: every polyline densified, its weight shared equally."""
    xy, weight = [], []
    for line in lines:
        points = densify(line["xy"], line["closed"], spacing)
        xy.extend([x, y] for x, y in points)
        weight.extend([line["weight"] / len(points)] * len(points))
    return xy, weight
