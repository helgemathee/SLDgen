"""The HTTP surface (Spec 2 SS12).

Runs in a lightweight venv with no torch: importing FastAPI next to
torch/diffusers/pydiffvg costs tens of seconds of start-up and holds CUDA context
memory the API never uses. Restarting the API must take under a second and must
never disturb a running job -- which it cannot, because the only channels between
the two units are SQLite and the filesystem.
"""

from .app import create_app

__all__ = ["create_app"]
