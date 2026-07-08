from pathlib import Path

import cv2
import numpy as np

from .tsp_art import init_tsp_art


def initialize_control_points(args, mask=None):
    """Initialize control points based on the specified initialization method.

    Parameters
    ----------
    args : object
        Configuration object containing:
        - init_method : str
            Initialization method: 'trefoil', 'contour', or 'tsp'.
        - n_control_points : int
            Number of control points to generate.
        - output_dir : str
            Directory for output files (used by 'tsp' method).
    mask : torch.Tensor, optional
        Binary mask tensor for contour or TSP-based initialization, by default None.

    Returns
    -------
    np.ndarray
        Control points of shape (n_control_points, 2), normalized to [0, 1].

    Raises
    ------
    NotImplementedError
        If the init_method is not one of 'trefoil', 'contour', or 'tsp'.
    """
    print(f"\tInitializing control points from {args.init_method}.", flush=True)
    if args.init_method == "trefoil":
        return initialize_from_trefoil(n_control_points=args.n_control_points, args=args)
    elif args.init_method == "contour":
        return initialize_from_contour(n_control_points=args.n_control_points, mask=mask)
    elif args.init_method == "tsp":
        return initialize_from_tsp(
            n_control_points=args.n_control_points,
            mask=mask,
            output_dir=args.output_dir,
            debug=args.debug,
            fixed_endpoints=args.fixed_endpoints,
            origin=getattr(args, "origin", None),
            init_points=getattr(args, "init_points", None),
            verbose=getattr(args, "verbose", False),
            stipple_weight=getattr(args, "stipple_weight", None),
            stipple_weight_mode=getattr(args, "stipple_weight_mode", "multiply"),
        )
    else:
        raise NotImplementedError(f"Initialization method {args.init_method} not implemented.")


def initialize_from_trefoil(n_control_points, args=None):
    # Create regular control points on trefoil shape
    control_points = np.zeros((n_control_points, 2), dtype=np.float32)
    ts = np.linspace(0, 2 * np.pi, n_control_points, endpoint=False)
    control_points[:, 0] = np.sin(ts) + 2 * np.sin(2 * ts)
    control_points[:, 1] = np.cos(ts) - 2 * np.cos(2 * ts)

    if hasattr(args, "scale_w") and hasattr(args, "scale_h"):
        print(f"Scaling control points by width: {args.true_scale_w}, height: {args.true_scale_h}")
        control_points[:, 0] *= max(args.true_scale_w, args.true_scale_h)
        control_points[:, 1] *= max(args.true_scale_w, args.true_scale_h)

    # Normalize and shift to roughly fit into [0, 1] canvas coordinates
    control_points /= 6
    control_points += 0.5

    return control_points


def initialize_from_contour(n_control_points, mask):
    binary_image = (mask.detach().numpy() > 0.5).astype(np.uint8)

    # Find the longest contour on the mask (most points) to use as the main outline
    contours, hierarchy = cv2.findContours(
        binary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )
    contours_len = [len(contour) for contour in contours]
    longest_contour = contours[np.argmax(contours_len)]

    cv2.drawContours(binary_image, [longest_contour], -1, 2, 1)

    # Sample `n_control_points` evenly along the chosen contour and normalize
    control_points = longest_contour.squeeze()
    start = np.random.randint(len(longest_contour))  # Defines where the line starts and ends
    control_points = np.concatenate([control_points[start:], control_points[:start]])
    control_points_sample = np.linspace(
        0, len(control_points) - 1, n_control_points, endpoint=False, dtype=int
    )
    control_points = control_points[control_points_sample].astype(float)

    control_points[:, 0] = control_points[:, 0] / binary_image.shape[1]
    control_points[:, 1] = control_points[:, 1] / binary_image.shape[0]

    return control_points


def get_longest_polyline_segment(polyline):
    # Find the index of the longest segment between consecutive polyline points
    max_length = 0
    longest_segment = None
    for i in range(len(polyline)):
        start = polyline[i]
        end = polyline[(i + 1) % len(polyline)]
        length = np.linalg.norm(end - start)
        if length > max_length:
            max_length = length
            longest_segment = i
    return longest_segment


def reorder_polyline(polyline):
    # Rotate the polyline so that the longest segment becomes the starting edge
    start_longest_segment_index = get_longest_polyline_segment(polyline)
    ordered_polyline = np.roll(polyline, -start_longest_segment_index - 1, axis=0)
    return ordered_polyline


def apply_stipple_weight(density, weight_path, mode, verbose=False):
    """Modulate the stipple density array with an external grayscale weight map.

    Opt-in (--stipple-weight). ``density`` is the RMBG-1.4 mask as a numpy array
    at render-size canvas resolution. ``rescale_obj`` bakes the
    --object-size-ratio scaling into the *content* of that array but keeps its
    shape at (render_size, render_size), so the array is a 1:1 overlay of canvas
    space. The weight map is authored in that same canvas space at --render-size;
    we therefore resample it to ``density.shape`` and combine directly, with no
    further coordinate transform. (Getting this space right is the primary risk
    called out in the brief -- the alignment test guards it.)

    Conventions (documented per the brief):
      * normalize: byte PNG -> [0, 1] by dividing by the dtype max -- the
        codebase's image->float convention (see ``create_masked_image`` in
        targets.py). Painted values are preserved (no min-max stretch), so soft
        weight fields stay soft.
      * resample: bilinear. The weight map is a continuous density field, so
        bilinear preserves its gradients without block artifacts; a hard region
        edge shifts by at most one density pixel, far below stipple()'s
        ~500 px/region zoom.
    """
    import cv2  # already a module-level dependency of this file

    wmap = cv2.imread(str(weight_path), cv2.IMREAD_GRAYSCALE)
    if wmap is None:
        raise ValueError(f"--stipple-weight: could not read image '{weight_path}'.")
    wmap = wmap.astype(np.float64) / np.iinfo(np.uint8).max  # [0, 1], values preserved

    # Resample to the density array's resolution (canvas space at render_size).
    if wmap.shape != density.shape:
        wmap = cv2.resize(
            wmap, (density.shape[1], density.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    if verbose:
        # Cheap diagnostic (same rationale as the init-coords line): confirm the
        # weight map landed in [0, 1] at the density resolution.
        print(
            f"\tstipple weight ({mode}): "
            f"min={wmap.min():.3f} max={wmap.max():.3f} mean={wmap.mean():.3f} "
            f"(resampled to {density.shape[0]}x{density.shape[1]})",
            flush=True,
        )

    if mode == "replace":
        # Weight map becomes the density field directly (RMBG mask bypassed).
        return wmap.astype(density.dtype)
    # multiply (default): keep subject-awareness -- a bright weight value over
    # background still can't stipple, because the RMBG density is ~0 there.
    return (density * wmap).astype(density.dtype)


def initialize_from_tsp(
    n_control_points, mask, output_dir, debug, fixed_endpoints, origin=None, init_points=None,
    verbose=False, stipple_weight=None, stipple_weight_mode="multiply",
):
    # The stipple density is the RMBG mask (canvas space, render_size). Opt-in:
    # --stipple-weight modulates it before it seeds the TSP curve; without the
    # flag this is exactly mask.numpy() and the path is byte-identical to upstream.
    # A copy via astype() avoids mutating the shared mask tensor's buffer.
    density = mask.numpy()
    if stipple_weight is not None:
        density = apply_stipple_weight(
            density.astype(np.float64), stipple_weight, stipple_weight_mode, verbose
        )

    # Create initial ordered points using the TSP-based initializer
    control_points = init_tsp_art(
        density,
        n_point=n_control_points,
        n_iter=25,
        reverse=True,
        output_dir=str(Path(output_dir) / "tsp_init"),
        debug=debug,
        fixed_endpoints=fixed_endpoints,
        origin=origin,
        init_points=init_points,
    )
    # Reorder so the polyline starts at the longest segment and convert to array.
    # When an origin is pinned, init_tsp_art has already rotated the tour so the
    # origin is index 0, so the longest-segment heuristic must NOT run here.
    if not fixed_endpoints and origin is None:
        control_points = reorder_polyline(control_points)
    control_points = np.array(control_points)

    control_points[:, 0] = control_points[:, 0] / mask.shape[1]
    control_points[:, 1] = control_points[:, 1] / mask.shape[0]

    if verbose:
        # Coordinate-stat instrumentation: init points should span the subject
        # region within the [0, 1] normalized canvas. Values well outside [0, 1]
        # (or a mean far from the subject centroid) signal an init offset/scale bug.
        print(
            f"\tinit points (normalized): "
            f"x[{control_points[:, 0].min():.3f}, {control_points[:, 0].max():.3f}] "
            f"y[{control_points[:, 1].min():.3f}, {control_points[:, 1].max():.3f}] "
            f"mean=({control_points[:, 0].mean():.3f}, {control_points[:, 1].mean():.3f})",
            flush=True,
        )

    return control_points
