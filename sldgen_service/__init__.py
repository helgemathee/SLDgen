"""Shared library for the SLDgen job service (Spec 2).

The spec's repository layout lists two packages, ``sldgen_worker`` and
``sldgen_api``. They necessarily share the database schema, the filesystem
layout, the CLI translator and the log reader, and duplicating any of those
would let the two units disagree about the contract they communicate through.
This package holds that common ground.

It deliberately imports **neither fastapi nor torch**, so it can be imported by
the lightweight API venv and the heavyweight conda env alike -- and by tests in
either.
"""

from .config import ServiceConfig

__all__ = ["ServiceConfig"]
