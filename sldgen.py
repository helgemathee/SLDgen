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

Optional ``--stop-at N`` / ``--resume CKPT`` / ``--checkpoint-interval N`` split a
run into segments that can be paused and picked back up:

  1. ``--num-iter`` remains the *horizon* every schedule normalizes against (the
     sparse-loss ramp is ``weight * epoch / num_iter``) and the epoch at which the
     run is complete; ``--stop-at`` only says where *this invocation* stops. That
     separation is what makes a 400-iteration preview a genuine prefix of the
     4000-iteration run it previews instead of a differently-scheduled drawing.
  2. ``checkpoint.py`` stores the optimized tensors, the monotone prune mask, the
     pinned TSP-derived geometry, Adam's state and all four RNG states, so a
     resumed segment continues the same trajectory rather than a similar one.
  3. Resume refuses to start if any run-shaping argument differs from the one the
     checkpoint recorded: it continues a trajectory, it never redirects one.

All three are strictly opt-in: with none of them set there is no ``checkpoints/``
directory, no ``state.json``, no signal handler, and behavior is unchanged.

Exit codes, so a supervising process can classify a finished segment without
parsing logs:

  0  reached the stop point, or stopped gracefully on SIGTERM (see state.json)
  2  validation error (bad flags, missing file, unusable checkpoint)
  3  environment error (HuggingFace auth, gated model access, CUDA unavailable)
  4  out of memory
  143 aborted by a second SIGTERM without writing a checkpoint
  1  anything else
"""

import sys
import traceback

from SLDgen import config
from SLDgen.checkpoint import EXIT_OK, classify_exception
from SLDgen.run import run, save_config, set_error_logging

if __name__ == "__main__":
    # argparse already exits 2 on a bad invocation, which is the validation code.
    args = config.parse_arguments()

    if not args.debug:
        set_error_logging()

    try:
        run(args)
    except BaseException as exc:  # noqa: BLE001 - classify, then re-report faithfully
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code = classify_exception(exc)
        traceback.print_exc()
        print(f"SLDgen failed ({type(exc).__name__}); exit code {code}.", file=sys.stderr)
        sys.exit(code)

    save_config(args)
    sys.exit(EXIT_OK)
