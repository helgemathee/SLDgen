import subprocess

import torch


def get_sparse_loss_weight(args, epoch):
    """Return sparse loss weight, supporting progressive schedules.

    A pure function of the absolute epoch and the horizon (``--num-iter``), never
    of "iterations executed so far in this process". That is what makes a
    segmented run identical to an uninterrupted one, and what makes a short
    preview a genuine prefix of the long run it previews. Any schedule added
    later must obey the same rule.

    Lives in utils (not run) so it can be imported without pulling in diffusers.
    """
    target_weight = args.sparse_loss_weight
    if args.sparse_loss_progressive == "linear":
        return target_weight * (epoch / args.num_iter)
    else:
        return target_weight


def increase_object_size(renderer, args):
    """Restore the object's original size and position on the renderer canvas."""
    with torch.no_grad():
        # scaling factors for x and y
        w, h = args.scale_w, args.scale_h
        canvas_width, canvas_height = args.render_size, args.render_size
        for path in renderer.shapes:
            # Normalize points to [-1, 1] relative coordinates
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            # Apply inverse scale
            path.points[:, 0] /= w
            path.points[:, 1] /= h
            # Convert back to pixel coordinates
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            # Recenter paths to original object center
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:, 0] += args.original_center_x * canvas_width - center_x
            path.points[:, 1] += args.original_center_y * canvas_height - center_y


def make_video(args):
    """Render PNG frames in `svg_to_png` to a video file using ffmpeg."""
    # Output resolution for the generated video
    output_width = args.render_size
    output_height = args.render_size

    # Use ffmpeg to assemble frames into an mp4 video
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-pattern_type",
            "glob",
            "-i",
            f"{args.output_dir}/svg_to_png/iter_*.png",
            "-vb",
            "20M",
            "-vf",
            f"scale={output_width}:{output_height}",
            f"{args.output_dir}/sketch.mp4",
        ],
        stdout=subprocess.DEVNULL if not args.debug else None,
        stderr=subprocess.DEVNULL if not args.debug else None,
    )
