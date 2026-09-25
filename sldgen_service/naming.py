"""Titles for a bulk rename: one base name, told apart by what actually differs.

Pure: takes plain dicts, touches nothing, so the API's dry run and its real run
are the same computation.
"""

SEPARATOR = " · "


def clean_title(title):
    """A title as stored: trimmed, and ``None`` rather than an empty string.

    ``None`` is what makes the UI fall back to the id, so a cleared title reads
    as "untitled" instead of as a blank row.
    """
    if title is None:
        return None
    title = " ".join(str(title).split())
    return title or None


def bulk_titles(base, jobs, taken=()):
    """``{job_id: title}`` for ``jobs`` renamed to ``base``, all distinct.

    Each job is ``base · s<seed>``. The seed comes from the job's stored
    parameters, never from its old title: titles accumulate stale ``· sNNNN``
    parts through run-again, while the parameters are what the run really used.

    Where that is not enough -- several selected jobs share a seed, or a job
    outside the selection already holds the name -- a ``· vNN`` suffix counts
    them up in creation order, skipping any number already ``taken``. Selected
    jobs' current titles are not in ``taken``: they are being replaced.
    """
    base = clean_title(base)
    if base is None:
        raise ValueError("a bulk rename needs a base name")
    taken = {clean_title(title) for title in taken} - {None}

    groups = {}
    for job in sorted(jobs, key=lambda job: (job["created_at"], job["id"])):
        stem = f"{base}{SEPARATOR}s{job['params']['seed']}"
        groups.setdefault(stem, []).append(job)

    titles = {}
    for stem, members in groups.items():
        if len(members) == 1 and stem not in taken:
            titles[members[0]["id"]] = stem
            continue
        number = 0
        for job in members:
            while True:
                number += 1
                candidate = f"{stem}{SEPARATOR}v{number:02d}"
                if candidate not in taken:
                    break
            titles[job["id"]] = candidate
    return titles
