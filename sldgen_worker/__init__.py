"""The GPU queue: one job at a time, one segment per claim (Spec 2 SS6-SS9)."""

from .worker import Worker

__all__ = ["Worker"]
