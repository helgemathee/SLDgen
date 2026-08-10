import os
import warnings

# Disable tokenizers parallelism warning cleanly before importing HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import json
import time
from pathlib import Path

import numpy as np
import torch
import wiregrad as wg
from PIL import Image
from tqdm.auto import tqdm

from .avoidance import avoidance_loss
from .attraction import attraction_loss
from .checkpoint import (
    GracefulStop,
    assert_fingerprint_matches,
    checkpointing_enabled,
    load_checkpoint,
    relative_to_output,
    restore_rng,
    save_checkpoint,
    structural_fingerprint,
    target_sha256,
    write_state,
)
from .guidance.sd3_sds_guidance_control import SD3GuidanceControl
from .metrics import get_all_metrics
from .painter.painter import SLDBSplinePainter
from .painter.painter_optimizer import PainterOptimizer
from .targets import get_target
from .utils import get_sparse_loss_weight, increase_object_size, make_video


def save_current_step(renderer, args, epoch, img):
    """Save the current sketch as SVG and PNG, and optionally the basis spline visualization."""
    renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
    init_img = img.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    init_img = Image.fromarray((init_img * 255).astype(np.uint8))
    init_img.save(f"{args.output_dir}/svg_to_png/iter_{epoch:04d}.png")
    if hasattr(renderer, "save_basis_spline"):
        renderer.save_basis_spline(f"{args.output_dir}/weights_logs/basis_spline_iter{epoch}.svg")


def finalize(renderer, args):
    """Export the finished drawing, leaving the renderer exactly as it was found.

    Finalisation used to mutate the renderer in place -- doubling the canvas, the
    control points and the widths -- so a "finished" run could not be continued
    from its own end state and any checkpoint written afterwards would have been
    wrong by a factor of two. This does the same export on the same values, then
    puts every mutated attribute back, so the last checkpoint of a run always
    describes the trajectory rather than the export.

    Restoring the original tensor *objects* matters as much as restoring their
    values: ``control_points * 2`` produces a new non-leaf tensor, which would
    otherwise leave the optimizer holding a parameter the renderer no longer uses.
    """
    pinned = (
        ("first_origin_points", "first_origin_widths")
        if getattr(args, "origin", None) is not None
        else ()
    )
    saved = {
        "canvas_width": renderer.canvas_width,
        "canvas_height": renderer.canvas_height,
        "control_points": renderer.control_points,
        "width": renderer.width,
    }
    saved.update({name: getattr(renderer, name) for name in pinned})

    try:
        if hasattr(args, "scale_w") or hasattr(args, "scale_h"):
            # Increases the size of the object on the canvas to its original size
            # if it has been reduced. NOTE: save_svg below rebuilds renderer.shapes
            # from the control points via set_shapes(), so this rescale does not
            # currently reach final_sld.svg. That is long-standing behaviour and
            # is preserved deliberately -- changing it would silently alter the
            # output of every run whose object was downscaled.
            increase_object_size(renderer, args)
        renderer.save_svg(args.output_dir, "final_sld")

        # Rasterize final SLD at double resolution for better quality, and save it
        renderer.canvas_width *= 2
        renderer.canvas_height *= 2
        renderer.control_points = renderer.control_points * 2
        renderer.width = renderer.width * 2
        if getattr(args, "origin", None) is not None:
            # Keep the pinned origin (and its widths) consistent with the doubled
            # control points for the final double-resolution export.
            renderer.first_origin_points = renderer.first_origin_points * 2
            renderer.first_origin_widths = renderer.first_origin_widths * 2
        raster_sld = renderer.get_image().permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        raster_sld = Image.fromarray((raster_sld * 255).astype(np.uint8))
        raster_sld.save(f"{args.output_dir}/final_sld.png")
    finally:
        for name, value in saved.items():
            setattr(renderer, name, value)


def run(args):
    print("Running SLDgen:", flush=True)

    # Segmented execution (opt-in). --num-iter is the horizon every schedule
    # normalises against and the epoch at which the run is complete; stop_at is
    # only where THIS invocation stops. Both default to the same value, which is
    # why an invocation with no new flags behaves exactly as it always did.
    ckpt_enabled = checkpointing_enabled(args)
    stop_at = args.stop_at if args.stop_at is not None else args.num_iter
    target_hash = target_sha256(args.target) if ckpt_enabled else None

    checkpoint = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        assert_fingerprint_matches(
            checkpoint["structural_fingerprint"],
            structural_fingerprint(args, target_hash=target_hash),
        )
        # Take the caption from the checkpoint rather than re-deriving it: with
        # --caption "" it was produced by BLIP-2, which is both a nondeterminism
        # risk and a model load we can skip entirely.
        args.caption = checkpoint["resolved_caption"]
        print(
            f"\tResuming {args.resume} at epoch {checkpoint['epoch']} "
            f"(horizon {args.num_iter}, this segment stops at {stop_at}).",
            flush=True,
        )

    # The loop counter is 0-based and epoch 0 performs a step, so "nothing done
    # yet" is epoch -1. A checkpoint stores the last COMPLETED epoch.
    start_epoch = -1 if checkpoint is None else int(checkpoint["epoch"])
    if ckpt_enabled:
        write_state(args, epoch=max(start_epoch, 0), phase="init")

    # Set up input, renderer and optimizer
    inputs, mask = get_target(args)
    renderer = SLDBSplinePainter(args=args, device=args.device, mask=mask)
    renderer = renderer.to(args.device)
    optimizer = PainterOptimizer(args, renderer)

    # Initialize renderer and optimizer
    if checkpoint is None:
        init_img = renderer.init_image()
    else:
        init_img = renderer.init_from_checkpoint(checkpoint)
    optimizer.init_optimizers()
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"], checkpoint["param_group_names"])

    # Setting up the SDS loss
    print(f"\tUsing {args.diffusion_model} as the diffusion model.", flush=True)
    sds_loss = SD3GuidanceControl(args=args, device=args.device)
    resolved_caption = args.caption

    print("\tStarting the optimization process...", flush=True)

    # Save the initial drawing before optimization. On resume that frame already
    # exists, and re-rendering it would insert a duplicate into the video.
    latest_preview = None
    latest_checkpoint = None
    if checkpoint is None:
        save_current_step(renderer, args, epoch=0, img=init_img)
        latest_preview = f"svg_to_png/iter_{0:04d}.png"
    else:
        # Point the heartbeat at the newest frame an earlier segment left behind,
        # so a resumed job has a preview to show before its first save interval.
        previews = list(Path(args.output_dir).glob("svg_to_png/iter_*.png"))
        if previews:
            newest = max(previews, key=lambda p: int(p.stem.split("_")[-1]))
            latest_preview = relative_to_output(args, newest)

    # Restoring the RNG must be the last thing before the loop: everything above
    # (target masking, initialisation, building the guidance pipeline) draws from
    # these generators, and a resumed segment must enter its first iteration in
    # the state the uninterrupted run would have been in.
    if checkpoint is not None:
        restore_rng(checkpoint["rng"])

    graceful = GracefulStop()
    if ckpt_enabled:
        graceful.install()

    # Optimization loop
    inputs = inputs.detach()
    epoch_range = tqdm(range(start_epoch + 1, stop_at + 1), bar_format="    {l_bar}{bar}{r_bar}")
    last_epoch = start_epoch
    segment_start_time = time.time()
    for epoch in epoch_range:
        optimizer.zero_grad_()

        # Semantic loss
        raster_sld = renderer.get_image().to(args.device)
        loss = sds_loss(raster_sld)
        loss.backward(retain_graph=True)

        # Regularization losses
        tqdm_update = dict()
        loss = None
        if args.repulsion_loss_weight > 0:
            loss = 0.0
            repulsion_loss = wg.repulsion_loss(renderer.sampled_curve3d, d0=25, cyclic=False)
            loss += args.repulsion_loss_weight * repulsion_loss

            tqdm_update["Repulsion Loss"] = (args.repulsion_loss_weight * repulsion_loss).item()

        # Avoidance constraint (opt-in). Only runs when --avoid loaded obstacle
        # points; otherwise this block is skipped entirely and behavior matches
        # upstream. Repels the actively-optimized control points away from the
        # fixed obstacle points, in canvas pixel coordinates (same frame as both).
        if getattr(renderer, "avoid_points", None) is not None:
            if loss is None:
                loss = 0.0
            avoid_loss = avoidance_loss(
                renderer.active_control_points,
                renderer.avoid_points,
                d0=args.avoidance_distance,
            )
            loss += args.avoidance_weight * avoid_loss

            tqdm_update["Avoidance Loss"] = (args.avoidance_weight * avoid_loss).item()

        # Attraction constraint (opt-in). Mirror of the avoidance block above and
        # composes with it: only runs when --attract loaded target points, else
        # skipped entirely. Pulls the actively-optimized control points TOWARD the
        # fixed target points, inactive within the dead-zone radius so the curve
        # stays free near the target structure.
        if getattr(renderer, "attract_points", None) is not None:
            if loss is None:
                loss = 0.0
            attract_loss = attraction_loss(
                renderer.active_control_points,
                renderer.attract_points,
                deadzone=args.attraction_distance,
            )
            loss += args.attraction_weight * attract_loss

            tqdm_update["Attraction Loss"] = (args.attraction_weight * attract_loss).item()

        if args.sparse_loss_weight > 0.0 and args.optimize_cp_weights:
            if loss is None:
                loss = 0.0
            if args.sparse_loss_type == 0.0:
                sparse_loss = get_sparse_loss_weight(args, epoch) / (
                    torch.var(renderer.weights) + 1e-5
                )
            else:
                sparse_loss = (
                    get_sparse_loss_weight(args, epoch)
                    * torch.pow(torch.abs(renderer.weights) + 1e-8, args.sparse_loss_type).mean()
                )
            loss += sparse_loss

            tqdm_update["Sparse Loss Weight"] = get_sparse_loss_weight(args, epoch)
            tqdm_update["Sparse Loss"] = sparse_loss.item()

        if args.length_shortening_loss_weight > 0.0:
            if loss is None:
                loss = 0.0
            sampled_curve = renderer.sampled_curve2d
            segments = sampled_curve[1:] - sampled_curve[:-1]
            lengths = torch.norm(segments, dim=-1)
            length_shortening_loss = torch.sum(lengths) * args.length_shortening_loss_weight
            loss += length_shortening_loss

            tqdm_update["Length Shortening Loss"] = length_shortening_loss.item()

        if loss is not None:
            loss.backward()

        if args.verbose:
            epoch_range.set_description(
                " - ".join(
                    [
                        f"{k}: {v:.2e}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in tqdm_update.items()
                    ]
                )
            )

        # Update parameters
        optimizer.step_()
        with torch.no_grad():
            renderer.post_process_params()

        # Save intermediate steps
        if epoch % args.save_interval == 0 and epoch > 0:
            save_current_step(renderer, args, epoch, raster_sld)
            latest_preview = f"svg_to_png/iter_{epoch:04d}.png"

        last_epoch = epoch

        # Heartbeat and periodic checkpoints (opt-in; nothing below runs unless
        # one of the checkpointing flags was passed).
        if ckpt_enabled:
            rate = (epoch - start_epoch) / max(time.time() - segment_start_time, 1e-9)
            # `epoch > 0` mirrors the intermediate-frame guard: epoch 0 is a
            # completed iteration, but checkpointing it right after start is noise.
            due = (
                args.checkpoint_interval > 0
                and epoch > 0
                and epoch % args.checkpoint_interval == 0
            )
            if due and epoch < stop_at:
                path = save_checkpoint(
                    renderer, optimizer, args, epoch, resolved_caption, target_hash=target_hash
                )
                save_config(args)
                write_state(
                    args,
                    epoch,
                    phase="optimizing",
                    iters_per_sec=rate,
                    latest_checkpoint=relative_to_output(args, path),
                    latest_preview=latest_preview,
                    resolved_caption=resolved_caption,
                )
            elif epoch % args.save_interval == 0:
                write_state(
                    args,
                    epoch,
                    phase="optimizing",
                    iters_per_sec=rate,
                    latest_preview=latest_preview,
                    resolved_caption=resolved_caption,
                )

            if graceful.requested:
                break

    if ckpt_enabled:
        graceful.uninstall()
        rate = (last_epoch - start_epoch) / max(time.time() - segment_start_time, 1e-9)
        path = save_checkpoint(
            renderer, optimizer, args, last_epoch, resolved_caption, target_hash=target_hash
        )
        save_config(args)
        latest_checkpoint = relative_to_output(args, path)
        write_state(
            args,
            last_epoch,
            phase="optimizing",
            iters_per_sec=rate,
            latest_checkpoint=latest_checkpoint,
            latest_preview=latest_preview,
            resolved_caption=resolved_caption,
        )

    # A run is complete only when it reaches its horizon. A segment that stops
    # short leaves a checkpoint and nothing else: no final SVG, no metrics, no
    # video, and svg_logs/ + svg_to_png/ keep accumulating across segments so the
    # assembled video still spans the whole trajectory.
    if last_epoch < args.num_iter:
        reason = "SIGTERM" if graceful.requested else f"--stop-at {stop_at}"
        resume_hint = (
            f" Resume with --resume {Path(args.output_dir) / latest_checkpoint}"
            if latest_checkpoint
            else ""
        )
        print(
            f"\tStopped at epoch {last_epoch} of {args.num_iter} ({reason}).{resume_hint}",
            flush=True,
        )
        return

    if ckpt_enabled:
        write_state(
            args,
            last_epoch,
            phase="finalizing",
            latest_checkpoint=latest_checkpoint,
            latest_preview=latest_preview,
            resolved_caption=resolved_caption,
        )

    # Save final SLD (pure export: the renderer is unchanged afterwards)
    finalize(renderer, args)

    # Compute all metrics
    print("\tComputing metrics...")
    metrics = get_all_metrics(
        f"{args.output_dir}/final_sld.png",
        args.original_target_path,
        args.caption,
        args.device,
        args.aesthetic_predictor_model_path,
    )
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save video (opt-out: --no-video skips the ffmpeg call, and with it the only
    # hard dependency on ffmpeg being installed. The frames stay on disk either way.)
    if args.save_video:
        print("\tSaving video...")
        make_video(args)

    if ckpt_enabled:
        write_state(
            args,
            last_epoch,
            phase="done",
            latest_checkpoint=latest_checkpoint,
            latest_preview=latest_preview,
            resolved_caption=resolved_caption,
        )
    print("Done!")


def set_error_logging():
    """
    Set logging to only show errors for diffusers and transformers libraries to reduce clutter in
    the output.
    """
    from diffusers.utils.logging import disable_progress_bar as diffusers_disable_pb
    from diffusers.utils.logging import set_verbosity_error as diffusers_set_verbosity_error
    from transformers.utils.logging import disable_progress_bar as transformers_disable_pb

    diffusers_set_verbosity_error()
    diffusers_disable_pb()
    transformers_disable_pb()


def save_config(args):
    """Save the configuration parameters to a JSON file in the output directory."""

    final_config = dict()
    for k, v in vars(args).items():
        if k == "mask":
            continue
        else:
            final_config[k] = str(v)
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(final_config, f, indent=4)
