"""SLDgen entry point.

Optional ``--origin X Y`` pins the start of the generated single-line drawing to a
normalized (X, Y) location in [0, 1] (X from the left, Y from the top). The
mechanism mirrors ``--fixed-endpoints``:

  1. ``tsp_art.py`` injects a node at the origin before Concorde runs, then rotates
     the resulting tour so the origin becomes ``control_points[0]``.
  2. ``painter.py`` stores the origin as a no-grad tensor with 3 coincident copies
     (clamping the uniform cubic B-spline so the ink starts exactly at the origin)
     and concatenates it in the forward pass. It is never added to the optimizer's
     param groups, so gradient descent cannot move it.

``--origin`` is strictly opt-in: when unset, behavior is identical to upstream.
"""

from SLDgen import config
from SLDgen.run import run, save_config, set_error_logging

if __name__ == "__main__":
    args = config.parse_arguments()

    if not args.debug:
        set_error_logging()

    run(args)

    save_config(args)
