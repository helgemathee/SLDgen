"""Attraction constraint for SLDgen (opt-in via ``--attract``).

This module is the structural mirror of :mod:`SLDgen.avoidance`. Where
avoidance *repels* the generated curve from a fixed set of obstacle points,
attraction *pulls* the curve toward a fixed set of target points loaded from
one or more SVG files.

The motivating workflow: partition a high-detail "master" curve into N pieces
(see ``sld_partition.py``), then run N fresh SLDgen optimizations, each
attracted to its own partition SVG. Every line is a genuine SDS optimization
(its own origin, its own organic tails), but leashed to its slice of the shared
structure so the N results compose.

Two pieces live here:

* :func:`load_attract_points` -- an alias for :func:`avoidance.load_avoid_points`;
  the *exact same* SVG loader/sampler (points in canvas pixel coordinates,
  uniform by arc length) is reused so both features share one coordinate
  convention.
* :func:`attraction_loss` -- a two-sided (Chamfer) pull with a **dead zone**.

Dead zone -- the key difference from avoidance
----------------------------------------------
Avoidance is active *within* its threshold (push away when close, silent when
far). Attraction is the inverse: **inactive within** ``deadzone`` of the target
points, pulling only *beyond* it. The dead zone is essential -- without it the
curve gets glued to the master and the SDS objective becomes decorative. Inside
the dead zone the curve is free to explore.

The whole feature is strictly opt-in: with no ``--attract`` argument,
``load_attract_points`` is never called and the loss is never evaluated, so
behavior is byte-for-byte identical to upstream.
"""

import torch

# Reuse the exact avoidance loader (same arc-length sampling, same canvas-pixel
# coordinate convention, same viewBox/size-mismatch warning). Aliased so the
# painter can import it symmetrically with load_avoid_points.
from .avoidance import load_avoid_points as load_attract_points

__all__ = ["load_attract_points", "attraction_loss"]


def attraction_loss(active_control_points, attract_points, deadzone):
    """Two-sided (Chamfer) pull toward fixed target points, with a dead zone.

    For a dead-zone radius ``r`` the per-pair penalty is the squared *hinge*
    ``max(0, dist - r)^2`` -- zero while within ``r`` of the counterpart point,
    growing quadratically beyond it. This is the sign-flipped mirror of the
    avoidance hinge ``max(0, r - dist)^2`` (active inside ``r`` there, inactive
    inside ``r`` here). Both terms below are required:

    1. **curve -> attract** (structure): each active curve point is pulled
       toward its nearest attract point::

           d_i      = min_j || p_i - q_j ||          # nearest target per curve pt
           term1    = sum_i  max(0, d_i - r)^2

       This keeps the curve sitting near the target structure.

    2. **attract -> curve** (coverage): each attract point pulls the nearest
       curve point toward it::

           e_j      = min_i || p_i - q_j ||          # nearest curve pt per target
           term2    = sum_j  max(0, e_j - r)^2

       Without this coverage term the curve could satisfy term1 by collapsing
       onto a small portion of the partition; term2 penalizes any target region
       left uncovered.

    The two share one pairwise distance matrix (``min`` over each axis). Point
    counts are small (hundreds to low thousands per side), so the brute-force
    ``(M, N)`` computation matches how :func:`avoidance.avoidance_loss` works.

    Consistency note: like the avoidance loss, this operates on the curve's
    *active control points* (pinned origin / fixed-endpoint / deactivated points
    already excluded by the caller), not on the sampled polyline -- consistency
    with the existing feature over theory.

    Args:
        active_control_points: ``(M, 2)`` tensor of the currently optimized
            control points, canvas pixel coordinates. Carries gradient.
        attract_points: ``(N, 2)`` fixed tensor of target points, same frame.
            No gradient.
        deadzone: dead-zone radius in canvas pixel units. The loss is zero for
            any pair closer than this; the pull only acts beyond it.

    Returns:
        Scalar tensor loss (0-d), on the same device/dtype as the inputs.
    """
    # Pairwise distances (M, N) via broadcasting.
    diff = active_control_points.unsqueeze(1) - attract_points.unsqueeze(0)  # (M, N, 2)
    dists = torch.linalg.vector_norm(diff, dim=-1)  # (M, N)

    # term1: nearest attract point per curve point (structure).
    curve_to_attract, _ = dists.min(dim=1)  # (M,)
    # term2: nearest curve point per attract point (coverage).
    attract_to_curve, _ = dists.min(dim=0)  # (N,)

    pull_curve = torch.clamp(curve_to_attract - deadzone, min=0.0) ** 2
    pull_attract = torch.clamp(attract_to_curve - deadzone, min=0.0) ** 2
    return pull_curve.sum() + pull_attract.sum()
